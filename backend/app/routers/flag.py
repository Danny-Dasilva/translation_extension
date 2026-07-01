"""Flag-for-finetune router.

Exposes ``POST /flag``: the extension flags a poor translation, and the backend
persists the ORIGINAL source image + metadata locally so the model can be
fine-tuned on those cases later. The image write and JSONL append run
ASYNCHRONOUSLY (FastAPI ``BackgroundTasks``) so the endpoint returns 200
immediately without blocking on disk I/O.

Storage layout (under ``settings.flagged_dir``, default ``data/flagged``):
    <flagged_dir>/<timestamp>_<shortid>.png   # the source image
    <flagged_dir>/flagged.jsonl               # one JSON record per flag

Each flagged.jsonl record is the fine-tune dataset seed:
    { id, timestamp, image_path (relative), page_url, target_language,
      boxes: [{ ocr_text, translated_text, minX, minY, maxX, maxY }],
      note, reason }

This is purely additive — it does not touch translate.py or any model service.
CORS is inherited from the global CORSMiddleware (the router is mounted on the
same app), so the extension reaches /flag under the same policy as /translate.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["flag"])


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class FlagBox(BaseModel):
    """A single flagged text box: source OCR, its translation, and bbox."""

    ocr_text: str = ""
    translated_text: str = ""
    minX: int = 0
    minY: int = 0
    maxX: int = 0
    maxY: int = 0


class FlagRequest(BaseModel):
    """Body for POST /flag."""

    # data URL ("data:image/png;base64,...") or raw base64 of the ORIGINAL image
    image_base64: str = Field(..., description="Source image as data URL or raw base64")
    page_url: Optional[str] = None
    target_language: Optional[str] = None
    boxes: List[FlagBox] = Field(default_factory=list)
    note: Optional[str] = None
    reason: Optional[str] = None


class FlagResponse(BaseModel):
    ok: bool
    id: str
    image_path: str


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _decode_image_b64(image_base64: str) -> bytes:
    """Decode a data-URL-or-raw base64 image into raw bytes.

    Raises ValueError on malformed input so the caller can return HTTP 400.
    """
    data = image_base64.strip()
    if data.startswith("data:"):
        # data:[<mediatype>][;base64],<payload>
        comma = data.find(",")
        if comma == -1:
            raise ValueError("malformed data URL: no comma separator")
        data = data[comma + 1 :]
    # Tolerate URL-safe variants and missing padding.
    data = data.replace("-", "+").replace("_", "/")
    padding = len(data) % 4
    if padding:
        data += "=" * (4 - padding)
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 image payload: {exc}") from exc
    if not raw:
        raise ValueError("empty image payload")
    return raw


def _persist_flag(
    flagged_dir: str,
    image_filename: str,
    image_bytes: bytes,
    record: dict,
) -> None:
    """Write the image and append the metadata record (runs in background).

    Synchronous disk I/O — invoked via BackgroundTasks so it executes AFTER the
    response is sent, keeping the request path non-blocking.
    """
    base = Path(flagged_dir)
    base.mkdir(parents=True, exist_ok=True)

    image_path = base / image_filename
    # Atomic-ish write: temp then rename so a partial file is never seen.
    tmp_path = image_path.with_suffix(image_path.suffix + ".tmp")
    tmp_path.write_bytes(image_bytes)
    tmp_path.replace(image_path)

    jsonl_path = base / "flagged.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Flagged translation persisted: id=%s image=%s boxes=%d",
        record.get("id"),
        image_path.name,
        len(record.get("boxes", [])),
    )


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #
@router.post("/flag", response_model=FlagResponse)
async def flag_translation(
    request: FlagRequest,
    background_tasks: BackgroundTasks,
) -> FlagResponse:
    """Flag a poor translation; persist source image + metadata for fine-tuning.

    Returns immediately ({ok, id, image_path}); the actual image write + JSONL
    append run in a BackgroundTask so the response isn't blocked on disk I/O.
    """
    # Decode synchronously so a malformed image fails fast with HTTP 400 (a
    # background task can't influence the status code). Decode is cheap; only
    # the disk writes are deferred.
    try:
        image_bytes = _decode_image_b64(request.image_base64)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image_base64: {exc}",
        )

    flag_id = uuid.uuid4().hex
    short_id = flag_id[:8]
    ts = time.time()
    timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
    image_filename = f"{timestamp_str}_{short_id}.png"

    flagged_dir = settings.flagged_dir
    # The response carries the full storage path (the caller may want it), but
    # the persisted RECORD must store a RELATIVE, portable image path so the
    # flagged.jsonl is a self-contained dataset seed regardless of where
    # flagged_dir points (it may be an absolute path in tests / deployments).
    # The image always lives directly under flagged_dir, so the bare filename
    # is the portable relative reference.
    response_image_path = str(Path(flagged_dir) / image_filename)
    record_image_path = image_filename

    record = {
        "id": flag_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
        + f".{int((ts - int(ts)) * 1000):03d}",
        "image_path": record_image_path,
        "page_url": request.page_url,
        "target_language": request.target_language,
        "boxes": [box.model_dump() for box in request.boxes],
        "note": request.note,
        "reason": request.reason,
    }

    # Defer disk I/O to after the response is returned.
    background_tasks.add_task(
        _persist_flag, flagged_dir, image_filename, image_bytes, record
    )

    return FlagResponse(ok=True, id=flag_id, image_path=response_image_path)

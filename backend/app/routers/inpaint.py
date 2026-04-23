"""Inpainting router.

Exposes `POST /inpaint` that takes a page image and a binary mask, both
base64-encoded PNGs, and returns an inpainted page (base64 PNG).

This is purely additive — it does not touch `translate.py`, nor any
existing service. It wires a lazily-instantiated `LamaInpaintService`
singleton behind an async dispatch so the forward pass (potentially
long-running) runs in a worker thread and does not block the event loop.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.lama_inpaint_service import LamaInpaintService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["inpaint"])


# ------------------------------------------------------------------------
# Lazy singleton — loading LaMa eagerly adds ~1 s of startup even on CPU.
# ------------------------------------------------------------------------

_service_lock = asyncio.Lock()
_service: Optional[LamaInpaintService] = None


async def _get_service() -> LamaInpaintService:
    global _service
    if _service is not None:
        return _service
    async with _service_lock:
        if _service is None:
            logger.info("Lazy-loading LamaInpaintService on first /inpaint request")
            _service = await asyncio.to_thread(LamaInpaintService)
    return _service


# ------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------


class InpaintRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded PNG (RGB or RGBA).")
    mask_base64: str = Field(..., description="Base64-encoded PNG, single channel. Any non-zero pixel is masked.")
    max_side: int = Field(
        1024,
        ge=64,
        le=8192,
        description="Upper bound on per-crop longer side before resize to the 512×512 model input.",
    )


class InpaintResponse(BaseModel):
    inpainted_image_base64: str
    width: int
    height: int
    components: int
    fastpath_hits: int
    forward_calls: int
    forward_ms: float
    total_ms: float


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------


def _strip_data_uri(s: str) -> str:
    if "," in s and s.startswith("data:"):
        return s.split(",", 1)[1]
    return s


def _decode_png_rgb(b64: str) -> np.ndarray:
    raw = base64.b64decode(_strip_data_uri(b64))
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("image_base64 is not a decodable PNG/JPEG.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _decode_png_gray(b64: str) -> np.ndarray:
    raw = base64.b64decode(_strip_data_uri(b64))
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("mask_base64 is not a decodable PNG/JPEG.")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _encode_png_rgb_to_b64(image_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ------------------------------------------------------------------------
# Route
# ------------------------------------------------------------------------


@router.post("/inpaint", response_model=InpaintResponse)
async def inpaint_endpoint(req: InpaintRequest) -> InpaintResponse:
    t0 = time.perf_counter()

    try:
        image_rgb = _decode_png_rgb(req.image_base64)
        mask_gray = _decode_png_gray(req.mask_base64)
    except Exception as e:  # noqa: BLE001 — surfaces to client
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode inputs: {e}",
        )

    if image_rgb.shape[:2] != mask_gray.shape:
        # Be forgiving: if shapes differ, resize mask (nearest) to match image.
        mh, mw = image_rgb.shape[:2]
        mask_gray = cv2.resize(mask_gray, (mw, mh), interpolation=cv2.INTER_NEAREST)

    svc = await _get_service()

    inpainted = await asyncio.to_thread(
        svc.inpaint, image_rgb, mask_gray, max_side=req.max_side
    )
    stats = svc.last_stats

    out_b64 = _encode_png_rgb_to_b64(inpainted)
    total_ms = (time.perf_counter() - t0) * 1000.0

    return InpaintResponse(
        inpainted_image_base64=out_b64,
        width=int(inpainted.shape[1]),
        height=int(inpainted.shape[0]),
        components=int(stats.get("components", 0)),
        fastpath_hits=int(stats.get("fastpath_hits", 0)),
        forward_calls=int(stats.get("forward_calls", 0)),
        forward_ms=float(stats.get("forward_ms", 0.0)),
        total_ms=total_ms,
    )

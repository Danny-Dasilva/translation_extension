"""Detect + OCR adapters that turn a PAGE IMAGE into a list of bubbles.

A bubble is::

    {"idx": int, "text": str, "bbox": {"minX","minY","maxX","maxY"}, "conf": float}

Two sides, two engines (see README for the rationale + GPU cost):

  JP side -- CTD (comictextdetector v26) detection + PARSeq recognition. Both
             run CPU-only via onnxruntime (auto CPU fallback) for a smoke test;
             on GPU they are ~100-130 FPS detect + ~7 ms/crop recognize. The
             PARSeq recognition confidence is the SAME ``ocr_conf`` the garble
             gate consumes.

  EN side -- a vision-language model (Qwen2.5/Qwen3-VL) reads the *typeset*
             English directly per bubble (text + bbox). Typeset English OCR is
             far more robust with a VLM than with a JP recognizer, and it doubles
             as detection. This needs the VLM server (GPU) -> the EN path is
             READY-BUT-DEFERRED until the GPU frees up; the JP path can smoke on
             CPU today.

These wrappers are intentionally thin -- all heavy logic stays in the existing
production services so the mined data matches what serving produces.
"""
from __future__ import annotations

import base64
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parents[3]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# --------------------------------------------------------------------------- #
# JP side: CTD detect + PARSeq recognize (CPU-capable)
# --------------------------------------------------------------------------- #
def build_jp_engines(cpu_only: bool = True):
    """Construct (detector, ocr) for the JP side.

    With ``cpu_only`` the PARSeq AR/hybrid path (which refuses CPU) is disabled
    so the whole JP side runs on the CPU onnxruntime provider. Both services pick
    up their default model paths from ``app.config.settings`` -- CTD defaults to
    ``comictextdetector_v26_round9_onofix_20260622.onnx`` (the v26 model)."""
    from app.services.ctd_service import ComicTextDetectorService
    from app.services.parseq_ocr_service import ParseqOCRService

    detector = ComicTextDetectorService()
    ocr = ParseqOCRService(
        hybrid_enabled=not cpu_only,
        vertical_ar_default=not cpu_only,
    )
    return detector, ocr


async def ocr_jp_page(image_path: str | Path, detector, ocr) -> list[dict]:
    """Detect + OCR every JP text region on a page image.

    Returns bubbles with PARSeq recognition ``conf`` (the garble-gate ``ocr_conf``).
    """
    from PIL import Image

    img = np.array(Image.open(image_path).convert("RGB"))
    det = await detector.detect(img)
    blocks = det.get("blocks") or []
    text_lines = det.get("text_lines") or []
    if not blocks:
        return []

    if text_lines:
        texts, confs = await ocr.recognize_blocks_with_lines(
            img, blocks, text_lines, return_confidence=True
        )
    else:
        crops = detector.crop_regions(img, blocks)
        pairs = await ocr.recognize_text_batch_with_conf(crops)
        texts = [t for t, _ in pairs]
        confs = [c for _, c in pairs]

    bubbles: list[dict] = []
    for i, blk in enumerate(blocks):
        txt = (texts[i] if i < len(texts) else "") or ""
        conf = float(confs[i]) if i < len(confs) and confs[i] is not None else 0.0
        bubbles.append({
            "idx": i,
            "text": txt.strip(),
            "bbox": {
                "minX": int(blk["minX"]), "minY": int(blk["minY"]),
                "maxX": int(blk["maxX"]), "maxY": int(blk["maxY"]),
            },
            "conf": conf,
        })
    return bubbles


# --------------------------------------------------------------------------- #
# EN side: VLM per-bubble typeset transcription (GPU / remote -- DEFERRED)
# --------------------------------------------------------------------------- #
VLM_ENDPOINT = "http://100.64.235.63:8001/v1/chat/completions"
VLM_MODEL = "qwenvl"


def _vlm_prompt(coord_norm: int) -> str:
    from scripts.eval.transcribe_gt_vision import VLM_PROMPT, VLM_PROMPT_NORM1000

    return VLM_PROMPT_NORM1000 if coord_norm == 1000 else VLM_PROMPT


def transcribe_en_page(
    image_path: str | Path,
    endpoint: str = VLM_ENDPOINT,
    model: str = VLM_MODEL,
    coord_norm: int = 0,
    timeout: float = 120.0,
) -> list[dict]:
    """Read typeset English per bubble off an EN page image via the VLM.

    READY-BUT-DEFERRED: requires the VLM server (GPU). Reuses the EXACT prompt +
    response parser from ``scripts/eval/transcribe_gt_vision.py`` so the EN bubble
    schema matches the eval gold. Returns bubbles WITHOUT a ``conf`` (the VLM does
    not emit a recognition probability)."""
    from PIL import Image

    from scripts.eval.transcribe_gt_vision import _parse_vision_response

    img = Image.open(image_path)
    w, h = img.size
    raw_bytes = Path(image_path).read_bytes()
    b64 = base64.b64encode(raw_bytes).decode()
    ext = Path(image_path).suffix.lstrip(".").lower() or "webp"
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _vlm_prompt(coord_norm)},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    req = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    content = body["choices"][0]["message"]["content"]
    vbubbles = _parse_vision_response(content, img_w=w, img_h=h, coord_norm=coord_norm)
    out: list[dict] = []
    for i, vb in enumerate(vbubbles):
        if not vb.bbox:
            continue
        out.append({"idx": i, "text": (vb.text or "").strip(), "bbox": vb.bbox, "conf": None})
    return out

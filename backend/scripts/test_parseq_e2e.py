"""End-to-end OCR sanity test for the new PARSeq service.

Loads a real manga image from backend/debug_output, runs the configured text
detector on it, crops every detected bubble, runs PARSeq on all crops, and
prints the recognized text. Also times the OCR pass.
"""
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import torch  # noqa: F401  (load CUDA libs before onnxruntime)

from app.services.detector_factory import create_detector
from app.services.parseq_ocr_service import ParseqOCRService


def pad_crop(img: np.ndarray, box: dict, pad: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, box["minX"] - pad)
    y0 = max(0, box["minY"] - pad)
    x1 = min(w, box["maxX"] + pad)
    y1 = min(h, box["maxY"] + pad)
    return img[y0:y1, x0:x1]


async def run(image_path: Path) -> None:
    img = np.array(Image.open(image_path).convert("RGB"))
    print(f"Image: {image_path.name} ({img.shape[1]}x{img.shape[0]})")

    # Prefer CTD so we get text_lines; fall back to whatever is configured.
    from app.services.ctd_service import ComicTextDetectorService
    try:
        detector = ComicTextDetectorService()
        print("Using CTD detector for line-level crops")
    except Exception:
        detector = create_detector()

    t0 = time.perf_counter()
    result = await detector.detect(img)
    det_ms = (time.perf_counter() - t0) * 1000
    blocks = result["blocks"]
    text_lines = result.get("text_lines", []) or []
    print(f"Detector: {len(blocks)} blocks, {len(text_lines)} text_lines in {det_ms:.1f} ms")
    if not blocks:
        print("No blocks — exiting.")
        return

    ocr = ParseqOCRService()
    await ocr.recognize_text_batch([pad_crop(img, blocks[0])])  # warmup

    t0 = time.perf_counter()
    texts = await ocr.recognize_blocks_with_lines(img, blocks, text_lines)
    ocr_ms = (time.perf_counter() - t0) * 1000
    print(f"\nPARSeq OCR: {len(blocks)} blocks in {ocr_ms:.1f} ms\n")

    for i, (b, t) in enumerate(zip(blocks, texts)):
        print(f"  [{i:02d}] ({b['minX']},{b['minY']})->({b['maxX']},{b['maxY']})  {t}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        # Pick the newest debug image as default.
        debug = BACKEND / "debug_output"
        cands = sorted(debug.glob("input_image_*.jpg"))
        if not cands:
            print("No test image available.")
            sys.exit(1)
        path = cands[-1]
    asyncio.run(run(path))

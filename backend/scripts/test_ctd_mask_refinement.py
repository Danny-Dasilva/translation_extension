"""Visualize CTD mask refinement: before (legacy) vs after (koharu-style).

Outputs (in thoughts/koharu-improvements/ctd/):
  before.png   - legacy threshold-only mask
  after.png    - koharu block-aware refined mask
  compare.png  - side-by-side (original | before | after | overlay)

Run:
  cd backend && uv run python scripts/test_ctd_mask_refinement.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import cv2
import numpy as np

# Make ``app`` imports resolve when invoked from backend/.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.ctd_service import ComicTextDetectorService  # noqa: E402


REPO_ROOT = BACKEND_DIR.parent
OUT_DIR = REPO_ROOT / "thoughts" / "koharu-improvements" / "ctd"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _find_sample_image() -> Path:
    candidates = [
        REPO_ROOT / "de.png",
        REPO_ROOT / "runs" / "detect" / "runs" / "manga_detect" / "de.jpg",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Fallback: search under runs/detect for any png/jpg.
    for path in (REPO_ROOT / "runs").rglob("*.png"):
        return path
    raise FileNotFoundError("No sample image found under repo root or runs/")


def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def _overlay(image_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.45) -> np.ndarray:
    out = image_bgr.copy()
    if mask is None:
        return out
    tint = np.zeros_like(out)
    tint[mask > 0] = color
    return cv2.addWeighted(out, 1.0, tint, alpha, 0)


async def main() -> None:
    img_path = _find_sample_image()
    print(f"[ctd-test] sample image: {img_path}")
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2.imread failed for {img_path}")

    service = ComicTextDetectorService()

    # Run detection once; reuse outputs so before/after operate on the same
    # raw mask (we just toggle the legacy flag on _process_mask).
    h, w = bgr.shape[:2]
    img_in, scale, padded_size = service._preprocess(bgr, input_is_bgr=True)
    input_name = service.session.get_inputs()[0].name
    outputs = await asyncio.to_thread(service.session.run, None, {input_name: img_in})
    blks, raw_mask, lines_map = service._assign_outputs(outputs)

    blocks = service._parse_blocks(blks, scale, (w, h))
    text_lines = service._extract_text_lines(lines_map, scale, padded_size, (w, h))
    if not blocks and text_lines:
        blocks = service._derive_blocks_from_text_lines(text_lines)

    if raw_mask is None:
        raise RuntimeError("CTD model did not produce a segmentation mask")

    legacy_mask = service._process_mask(raw_mask, padded_size, (w, h), blocks=None, legacy=True)
    refined_mask = service._process_mask(raw_mask, padded_size, (w, h), blocks=blocks, legacy=False)

    before_path = OUT_DIR / "before.png"
    after_path = OUT_DIR / "after.png"
    cv2.imwrite(str(before_path), legacy_mask)
    cv2.imwrite(str(after_path), refined_mask)

    legacy_nz = int((legacy_mask > 0).sum())
    refined_nz = int((refined_mask > 0).sum())
    total = legacy_mask.size
    print(f"[ctd-test] legacy  nonzero: {legacy_nz:>10d} ({legacy_nz / total:6.2%})")
    print(f"[ctd-test] refined nonzero: {refined_nz:>10d} ({refined_nz / total:6.2%})")
    print(f"[ctd-test] blocks detected : {len(blocks)}")
    print(f"[ctd-test] text lines      : {len(text_lines)}")

    # Side-by-side comparison panel (2x2 grid).
    panels = [
        ("input", bgr),
        ("before (legacy)", _overlay(bgr, legacy_mask, color=(0, 0, 255))),
        ("after (koharu)", _overlay(bgr, refined_mask, color=(0, 200, 0))),
        ("after mask", _mask_to_bgr(refined_mask)),
    ]

    # Normalize sizes and stack as 2x2.
    th = bgr.shape[0]
    tw = bgr.shape[1]
    labeled: list[np.ndarray] = []
    for name, panel in panels:
        if panel.shape[:2] != (th, tw):
            panel = cv2.resize(panel, (tw, th))
        p = panel.copy()
        cv2.rectangle(p, (0, 0), (tw, 36), (0, 0, 0), -1)
        cv2.putText(p, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        labeled.append(p)
    top = np.hstack([labeled[0], labeled[1]])
    bottom = np.hstack([labeled[2], labeled[3]])
    compare = np.vstack([top, bottom])
    compare_path = OUT_DIR / "compare.png"
    cv2.imwrite(str(compare_path), compare)

    print(f"[ctd-test] wrote {before_path}")
    print(f"[ctd-test] wrote {after_path}")
    print(f"[ctd-test] wrote {compare_path}")


if __name__ == "__main__":
    asyncio.run(main())

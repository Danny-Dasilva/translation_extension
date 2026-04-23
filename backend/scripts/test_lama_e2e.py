#!/usr/bin/env python3
"""End-to-end LaMa inpainting sanity test.

Pipeline:
    de.png  ─►  CTD  ─►  binary text mask
       │                       │
       └──────► LaMa inpaint ◄─┘
                     │
                     ▼
          inpainted page + compare strip

Saves the following under `thoughts/koharu-improvements/inpainting/`:
    original.png   — the source page
    mask.png       — the CTD-derived binary text mask
    inpainted.png  — the LaMa output
    compare.png    — horizontal concat [original | mask | inpainted]

Usage:
    uv run python scripts/test_lama_e2e.py [path_to_image]
    # (defaults to repo-root /de.png)
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import torch  # noqa: F401  — load CUDA libs before onnxruntime

from app.services.ctd_service import ComicTextDetectorService  # noqa: E402
from app.services.lama_inpaint_service import LamaInpaintService  # noqa: E402


OUT_DIR = REPO / "thoughts" / "koharu-improvements" / "inpainting"


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        bgr = image
    cv2.imwrite(str(path), bgr)


def _compare_strip(original: np.ndarray, mask: np.ndarray, inpainted: np.ndarray) -> np.ndarray:
    """Build a horizontal concat [original | mask(RGB) | inpainted].

    Pads shorter images with white so heights match.
    """
    h = max(original.shape[0], mask.shape[0], inpainted.shape[0])
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) if mask.ndim == 2 else mask

    def pad(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img
        pad_h = h - img.shape[0]
        return cv2.copyMakeBorder(
            img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    return np.concatenate(
        [pad(original), pad(mask_rgb), pad(inpainted)], axis=1
    )


async def run(image_path: Path) -> None:
    print(f"Image: {image_path}")
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"  size : {rgb.shape[1]}x{rgb.shape[0]}")

    # --- CTD for mask -------------------------------------------------
    ctd = ComicTextDetectorService()
    t0 = time.perf_counter()
    det = await ctd.detect(rgb)
    ctd_ms = (time.perf_counter() - t0) * 1000.0
    mask = det.get("mask")
    blocks = det.get("blocks") or []
    text_lines = det.get("text_lines") or []
    print(
        f"  CTD  : {len(blocks)} blocks, {len(text_lines)} lines, "
        f"mask={'Y' if mask is not None else 'N'} in {ctd_ms:.0f} ms"
    )

    if mask is None:
        # Fallback — rasterize the detected text-line boxes as a mask so we
        # still have something to inpaint even if CTD didn't emit a mask.
        print("  CTD produced no mask — rasterizing text_line boxes as mask.")
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        for line in text_lines or blocks:
            x1 = int(line.get("minX", line.get("x", 0)))
            y1 = int(line.get("minY", line.get("y", 0)))
            x2 = int(line.get("maxX", x1 + int(line.get("width", 0))))
            y2 = int(line.get("maxY", y1 + int(line.get("height", 0))))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    # Make sure mask is uint8 HxW same shape as image.
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    if mask.shape[:2] != rgb.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_pixel_frac = float((mask > 0).mean())
    print(f"  mask : {mask_pixel_frac*100:.2f}% of pixels masked")

    # --- LaMa ---------------------------------------------------------
    svc = LamaInpaintService()

    # Warm up on 512×512 dummy so the first real forward is hot.
    warm_img = np.zeros((512, 512, 3), dtype=np.uint8)
    warm_msk = np.zeros((512, 512), dtype=np.uint8)
    warm_msk[128:384, 128:384] = 255
    _ = svc.inpaint(warm_img, warm_msk)
    print(f"  LaMa : warmed up on device={svc.device}")

    t0 = time.perf_counter()
    inpainted = svc.inpaint(rgb, mask)
    lama_ms = (time.perf_counter() - t0) * 1000.0
    stats = svc.last_stats
    print(
        f"  LaMa : {lama_ms:.0f} ms ("
        f"components={stats['components']}, "
        f"fastpath_hits={stats['fastpath_hits']}, "
        f"forward_calls={stats['forward_calls']}, "
        f"forward_ms={stats['forward_ms']:.0f})"
    )
    fastpath_rate = (
        stats["fastpath_hits"] / stats["components"]
        if stats["components"]
        else 0.0
    )
    print(f"  fast-path hit rate: {fastpath_rate*100:.1f}%")

    # --- write outputs ------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_png(OUT_DIR / "original.png", rgb)
    _save_png(OUT_DIR / "mask.png", mask)
    _save_png(OUT_DIR / "inpainted.png", inpainted)
    _save_png(OUT_DIR / "compare.png", _compare_strip(rgb, mask, inpainted))

    print(f"\nWrote outputs to {OUT_DIR}")

    # Dump stats into a side-car so SUMMARY.md can reference real numbers.
    stats_txt = OUT_DIR / "stats.txt"
    stats_txt.write_text(
        "\n".join(
            [
                f"image: {image_path.name} ({rgb.shape[1]}x{rgb.shape[0]})",
                f"ctd_ms: {ctd_ms:.1f}",
                f"lama_ms: {lama_ms:.1f}",
                f"device: {svc.device}",
                f"components: {stats['components']}",
                f"fastpath_hits: {stats['fastpath_hits']}",
                f"fastpath_rate: {fastpath_rate:.3f}",
                f"forward_calls: {stats['forward_calls']}",
                f"forward_ms: {stats['forward_ms']:.1f}",
                f"mask_pixel_frac: {mask_pixel_frac:.4f}",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    default = REPO / "de.png"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    asyncio.run(run(path))

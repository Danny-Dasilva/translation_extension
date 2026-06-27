"""A/B re-render harness for the font readability + page-consistency change.

Loads real Ikenie 4 inspect pages (clean LaMa plate + per-bubble bbox +
translation_en) and renders each page through ``compose_final`` BEFORE and
AFTER the new floor/consistency policy, side by side, and reports the
min/median/max rendered font size per page.

BEFORE  = consistency OFF + the OLD fixed 14px soft floor (simulated by
          monkeypatching the resolution floor back to a flat 14).
AFTER   = the new resolution-aware floor + page consistency ON (defaults).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/ab_font_rerender.py 058 129 048 093 020
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import refit_final_composites as R  # noqa: E402

INSP_ROOT = Path(
    "/home/danny/Documents/personal/extension/backend/.bench/ikenie4_final_insp"
)
OUT_DIR = Path("/tmp/font_ab")


def load_page(page: str):
    d = INSP_ROOT / page
    plate_p = d / "07_inpaint.webp"
    bj = d / "bubbles.json"
    if not (plate_p.exists() and bj.exists()):
        return None
    plate = np.array(Image.open(plate_p).convert("RGB"))
    data = json.load(open(bj))
    blocks, fits, texts = [], [], []
    for b in data:
        t = b.get("translation_en")
        if not t:
            continue
        bb = b["bbox"]
        blk = {"minX": bb["minX"], "minY": bb["minY"],
               "maxX": bb["maxX"], "maxY": bb["maxY"],
               "orphan": bool(b.get("is_orphan"))}
        blocks.append(blk)
        # In the inspect data the bbox IS the bubble/text region we typeset to.
        # Non-orphan, non-filtered entries are dialogue (give a fit_rect);
        # orphans are clamped SFX/caption (fit_rect = None).
        fits.append(None if b.get("is_orphan") else dict(blk))
        texts.append(t)
    return plate, blocks, fits, texts


def size_stats(sizes):
    s = [x for x in sizes if x is not None]
    if not s:
        return (0, 0, 0, 0)
    return (min(s), round(statistics.median(s)), max(s), len(s))


def render_before(plate, blocks, fits, texts):
    """Old behaviour: flat 14px floor, no page consistency."""
    orig_floor = R.resolution_font_floor
    orig_clamped = R.CLAMPED_HARD_FLOOR
    try:
        R.resolution_font_floor = lambda _h: 14  # the OLD fixed soft floor
        R.CLAMPED_HARD_FLOOR = 9                  # the OLD hard floor
        img = R.compose_final(plate, blocks, texts, fit_rects=fits,
                              consistent_font=False)
        sizes = R.compose_final(plate, blocks, texts, fit_rects=fits,
                                consistent_font=False, _debug_sizes=True)
    finally:
        R.resolution_font_floor = orig_floor
        R.CLAMPED_HARD_FLOOR = orig_clamped
    return img, sizes


def render_after(plate, blocks, fits, texts):
    """New behaviour: resolution-aware floor + page consistency (defaults)."""
    img = R.compose_final(plate, blocks, texts, fit_rects=fits)
    sizes = R.compose_final(plate, blocks, texts, fit_rects=fits,
                            _debug_sizes=True)
    return img, sizes


def main(pages):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'page':>5} {'floor':>6} | {'BEFORE min/med/max(n)':>26} | "
          f"{'AFTER min/med/max(n)':>26}")
    print("-" * 74)
    for page in pages:
        loaded = load_page(page)
        if loaded is None:
            print(f"{page:>5}  SKIP (missing plate/bubbles)")
            continue
        plate, blocks, fits, texts = loaded
        if not blocks:
            print(f"{page:>5}  SKIP (no translated bubbles)")
            continue
        H = plate.shape[0]
        floor = R.resolution_font_floor(H)
        before_img, before_sizes = render_before(plate, blocks, fits, texts)
        after_img, after_sizes = render_after(plate, blocks, fits, texts)
        bmin, bmed, bmax, bn = size_stats(before_sizes)
        amin, amed, amax, an = size_stats(after_sizes)
        print(f"{page:>5} {floor:>6} | "
              f"{f'{bmin}/{bmed}/{bmax} (n={bn})':>26} | "
              f"{f'{amin}/{amed}/{amax} (n={an})':>26}")

        # side-by-side: BEFORE | AFTER
        h, w = after_img.shape[:2]
        gap = 16
        canvas = Image.new("RGB", (w * 2 + gap, h), (32, 32, 32))
        canvas.paste(Image.fromarray(before_img), (0, 0))
        canvas.paste(Image.fromarray(after_img), (w + gap, 0))
        canvas.save(OUT_DIR / f"{page}_ab.png")
        Image.fromarray(before_img).save(OUT_DIR / f"{page}_before.png")
        Image.fromarray(after_img).save(OUT_DIR / f"{page}_after.png")
    print(f"\nwrote before/after/side-by-side PNGs to {OUT_DIR}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["058", "129", "048", "093", "020", "122", "079", "069"]
    main(args)

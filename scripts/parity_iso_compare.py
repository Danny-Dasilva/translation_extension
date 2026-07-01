#!/usr/bin/env python3
"""parity_iso_compare.py — ISOLATED renderer-parity diff: path A (extension
canvas) vs path B (backend PIL), both rendered from the SAME captured /translate
response. Because the text content + plate are byte-identical inputs, the ONLY
sources of pixel diff are renderer differences: font rasterizer (browser Skia vs
FreeType/PIL, +/-1-2px per glyph), stroke anti-aliasing, sub-pixel layout, and
JPEG (extension data-url) vs PNG encoding.

Reuses the tolerant metric shape of scripts/parity_compare.py:
  frac/pct pixels differing (per-channel max delta > thresh), mean/p95/p99 abs
  delta, and an 8x12 structural-tile fraction (catches "text in wrong place").

Usage: python scripts/parity_iso_compare.py <page> [--thresh 40]
Outputs under backend/.bench/_parity/iso_<page>/:
  iso_diff.png   (white = differing pixel above thresh)
Prints a JSON summary incl. PASS/FAIL on renderer-matches-backend.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page")
    ap.add_argument("--thresh", type=int, default=40,
                    help="per-channel max delta to count a pixel as differing")
    args = ap.parse_args()
    page = args.page
    iso = os.path.join(ROOT, "backend/.bench/_parity", f"iso_{page}")

    a_path = os.path.join(iso, "ext_render.jpg")      # path A (extension, renderer-exact)
    if not os.path.exists(a_path):
        a_path = os.path.join(iso, "ext_render.png")
    b_path = os.path.join(iso, "backend_render.png")  # path B (backend PIL)

    out = {"page": page, "A_extension": a_path, "B_backend": b_path}
    for p in (a_path, b_path):
        if not os.path.exists(p):
            out["error"] = f"missing render: {p}"
            print(json.dumps(out, indent=2)); sys.exit(2)

    b = load_rgb(b_path)                 # backend is the reference dimension
    a = load_rgb(a_path, size=b.size)    # extension normalized to backend dims
    out["size"] = list(b.size)
    out["A_native_size"] = list(Image.open(a_path).size)
    out["resampled_A"] = Image.open(a_path).size != b.size

    aa = np.asarray(a, dtype=np.int16)
    bb = np.asarray(b, dtype=np.int16)
    delta = np.abs(aa - bb).max(axis=2)

    total = delta.size
    differing = int((delta > args.thresh).sum())
    frac = differing / total
    out["threshold"] = args.thresh
    out["frac_pixels_differing"] = round(frac, 5)
    out["pct_pixels_differing"] = round(frac * 100, 3)
    out["mean_abs_delta"] = round(float(delta.mean()), 3)
    out["p95_abs_delta"] = int(np.percentile(delta, 95))
    out["p99_abs_delta"] = int(np.percentile(delta, 99))

    gh, gw = 12, 8
    H, W = delta.shape
    th, tw = H // gh, W // gw
    big_tiles = 0
    for r in range(gh):
        for c in range(gw):
            tile = delta[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            if tile.mean() > 18:
                big_tiles += 1
    out["structural_tiles_total"] = gh * gw
    out["structural_tiles_differing"] = big_tiles
    out["structural_frac"] = round(big_tiles / (gh * gw), 4)

    diff_img = (delta > args.thresh).astype(np.uint8) * 255
    Image.fromarray(diff_img, mode="L").save(os.path.join(iso, "iso_diff.png"))
    out["diff_artifact"] = os.path.join(iso, "iso_diff.png")

    # PASS criteria for a renderer-only comparison. Since inputs are identical,
    # we hold to a TIGHTER pixel band than parity_compare.py's 0.12 (which also
    # absorbed translation-inference variance): renderer-only diffs from font
    # rasterization concentrate on glyph edges and should be a few percent.
    structural_match = out["structural_frac"] < 0.18
    pixel_band_ok = frac < 0.08
    out["PASS"] = bool(structural_match and pixel_band_ok)
    out["verdict"] = (
        "PASS — extension renderer matches backend within font-raster tolerance "
        "(text/placement/size identical; residual diff = glyph-edge AA + JPEG)"
        if out["PASS"] else
        "FAIL — diff exceeds font-raster band; inspect iso_diff.png "
        "(text/placement may differ, or rasterization gap is large)"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

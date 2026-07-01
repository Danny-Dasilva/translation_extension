#!/usr/bin/env python3
"""
parity_compare.py — tolerant pixel comparison of the extension's rendered output
vs the backend's reference render (11_final.webp) for the SAME manga page.

Tolerant on purpose. EXPECTED, non-failing diffs:
  - browser font rasterization vs PIL/FreeType (sub-pixel glyph edges, +/-1-2px)
  - JPEG (extension) vs WEBP (backend) encoding noise
  - minor stroke anti-aliasing
We therefore report the FRACTION of pixels whose per-channel max delta exceeds a
threshold (default 40/255), and a coarse structural check (region-level).

Usage:
  python scripts/parity_compare.py <page> [--ext <ext_render.jpg>] [--thresh 40]
  page = 044 / 030
Outputs (under backend/.bench/_parity/):
  cmp_<page>_ext.png        normalized extension render
  cmp_<page>_ref.png        normalized backend reference
  cmp_<page>_diff.png       heat diff (white = differing pixel above thresh)
Prints a JSON summary.
"""
import sys, os, json, argparse
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARITY = os.path.join(ROOT, "backend/.bench/_parity")


def load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page")
    ap.add_argument("--ext", default=None, help="extension render image (jpg/png)")
    ap.add_argument("--thresh", type=int, default=40, help="per-channel max delta to count a pixel as 'differing'")
    args = ap.parse_args()

    page = args.page
    ref_path = os.path.join(ROOT, f"backend/.bench/full_pipeline_v2/588828_mesu2_insp/{page}/11_final.webp")
    ext_path = args.ext or os.path.join(PARITY, f"ext_render_{page}.jpg")
    if not os.path.exists(ext_path):
        ext_png = os.path.join(PARITY, f"ext_render_{page}.png")
        if os.path.exists(ext_png):
            ext_path = ext_png

    out = {"page": page, "ext": ext_path, "ref": ref_path}

    if not os.path.exists(ext_path):
        out["error"] = f"extension render not found: {ext_path}"
        print(json.dumps(out, indent=2)); sys.exit(2)
    if not os.path.exists(ref_path):
        out["error"] = f"reference not found: {ref_path}"
        print(json.dumps(out, indent=2)); sys.exit(2)

    ref = load_rgb(ref_path)
    ext = load_rgb(ext_path, size=ref.size)  # normalize ext to ref dims
    out["ref_size"] = list(ref.size)
    out["ext_native_size"] = list(Image.open(ext_path).size)

    a = np.asarray(ext, dtype=np.int16)
    b = np.asarray(ref, dtype=np.int16)
    delta = np.abs(a - b).max(axis=2)  # per-pixel max channel delta

    total = delta.size
    differing = int((delta > args.thresh).sum())
    frac = differing / total
    out["threshold"] = args.thresh
    out["frac_pixels_differing"] = round(frac, 5)
    out["pct_pixels_differing"] = round(frac * 100, 3)
    out["mean_abs_delta"] = round(float(delta.mean()), 3)
    out["p95_abs_delta"] = int(np.percentile(delta, 95))
    out["p99_abs_delta"] = int(np.percentile(delta, 99))

    # Coarse structural check: tile the image into an 8x12 grid and report the
    # fraction of TILES whose mean delta is large (catches "text in wrong place"
    # / "bubble not translated" rather than uniform encoding noise).
    gh, gw = 12, 8
    H, W = delta.shape
    th, tw = H // gh, W // gw
    big_tiles = 0
    for r in range(gh):
        for c in range(gw):
            tile = delta[r*th:(r+1)*th, c*tw:(c+1)*tw]
            if tile.mean() > 18:  # whole tile structurally different
                big_tiles += 1
    out["structural_tiles_total"] = gh * gw
    out["structural_tiles_differing"] = big_tiles
    out["structural_frac"] = round(big_tiles / (gh * gw), 4)

    # Save normalized images + diff heatmap
    os.makedirs(PARITY, exist_ok=True)
    ext.save(os.path.join(PARITY, f"cmp_{page}_ext.png"))
    ref.save(os.path.join(PARITY, f"cmp_{page}_ref.png"))
    diff_img = (delta > args.thresh).astype(np.uint8) * 255
    Image.fromarray(diff_img, mode="L").save(os.path.join(PARITY, f"cmp_{page}_diff.png"))
    out["artifacts"] = {
        "ext_norm": os.path.join(PARITY, f"cmp_{page}_ext.png"),
        "ref_norm": os.path.join(PARITY, f"cmp_{page}_ref.png"),
        "diff": os.path.join(PARITY, f"cmp_{page}_diff.png"),
    }

    # Interpretation heuristic (tolerant): structural match if the bulk of tiles
    # agree and the differing-pixel fraction is in the "text rasterization" band.
    structural_match = out["structural_frac"] < 0.18
    pixel_band_ok = frac < 0.12
    out["VISUALLY_MATCHES"] = bool(structural_match and pixel_band_ok)
    out["interpretation"] = (
        "structural MATCH — bubbles translated in same places, art preserved; "
        "residual diff consistent with font-raster + JPEG/WEBP encoding"
        if out["VISUALLY_MATCHES"] else
        "STRUCTURAL MISMATCH — differing tiles exceed font-raster band; inspect cmp_*_diff.png"
    )

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

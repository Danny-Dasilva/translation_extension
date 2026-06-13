"""Render final composites for v26-detected pages.

Uses:
  - 01_original.png / 07_inpainted.png from /home/danny/manga-output/644289/<slug>/
  - blocks from <variant_dir>/<slug>/stats.json (v26 bboxes + translations)

Outputs:
  - <variant_dir>/<slug>/11_final_composite.png
  - <variant_dir>/<slug>/12_final_side_by_side.png
  - <mirror>/<slug>.png

Bypasses the existing render_manga_finals.py's CTD re-detection step since
we already have v26 bboxes embedded in stats.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from scripts.refit_final_composites import compose_final, load_font  # noqa: E402
from PIL import ImageDraw  # noqa: E402

ORIG_DIR_DEFAULT = Path("/home/danny/manga-output/644289")


def render_one(variant_dir: Path, slug: str, orig_root: Path, mirror_dir: Path | None) -> bool:
    page_dir = variant_dir / slug
    stats_p = page_dir / "stats.json"
    if not stats_p.exists():
        return False
    with open(stats_p) as f:
        s = json.load(f)
    blocks_in = s.get("blocks") or []
    translations = s.get("translations") or []
    if not blocks_in:
        return False

    # Source plate (inpainted) and original
    orig_page = orig_root / slug
    orig_path = orig_page / "01_original.png"
    inpaint_path = orig_page / "07_inpainted.png"
    if not orig_path.exists() or not inpaint_path.exists():
        return False
    orig = np.array(Image.open(orig_path).convert("RGB"))
    inpainted = np.array(Image.open(inpaint_path).convert("RGB"))
    if inpainted.shape[0] == orig.shape[0] + 32 and inpainted.shape[1] == orig.shape[1]:
        inpainted = inpainted[: orig.shape[0], : orig.shape[1]]

    # Map our v26 bbox format -> compose_final's expected dict format
    blocks = []
    for b in blocks_in:
        x0, y0, x1, y1 = b["bbox"]
        blocks.append({
            "minX": x0, "minY": y0, "maxX": x1, "maxY": y1,
        })

    final = compose_final(inpainted, blocks, translations)
    Image.fromarray(final).save(page_dir / "11_final_composite.png")

    # Side-by-side: inpainted plate vs final
    h, w = final.shape[:2]
    side = Image.new("RGB", (w * 2 + 20, h + 40), (24, 24, 24))
    side.paste(Image.fromarray(inpainted), (0, 30))
    side.paste(Image.fromarray(final), (w + 20, 30))
    sd = ImageDraw.Draw(side)
    sd.text((10, 6), "LaMa plate", fill=(200, 200, 200), font=load_font(16))
    sd.text((w + 30, 6), f"+ v7 translations ({len(blocks)} blocks)",
            fill=(0, 255, 180), font=load_font(16))
    side.save(page_dir / "12_final_side_by_side.png")

    if mirror_dir:
        mirror_dir.mkdir(parents=True, exist_ok=True)
        target = mirror_dir / f"{slug}.png"
        if target.exists():
            target.unlink()
        target.write_bytes((page_dir / "11_final_composite.png").read_bytes())
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-dir", type=Path, required=True,
                    help="Output dir from ocr_and_translate_v26 (with stats.json per page)")
    ap.add_argument("--orig-root", type=Path, default=ORIG_DIR_DEFAULT,
                    help="Original gallery with 01_original.png + 07_inpainted.png per page")
    ap.add_argument("--mirror", type=Path,
                    help="Optional flat output dir for finals as <slug>.png")
    args = ap.parse_args()

    slugs = sorted(p.name for p in args.variant_dir.iterdir() if p.is_dir() and p.name.isdigit())
    rendered = 0
    for slug in slugs:
        ok = render_one(args.variant_dir, slug, args.orig_root, args.mirror)
        marker = "✓" if ok else "✗"
        print(f"  [{slug}] {marker}")
        if ok:
            rendered += 1
    print(f"DONE: {rendered}/{len(slugs)} rendered")
    return 0


if __name__ == "__main__":
    sys.exit(main())

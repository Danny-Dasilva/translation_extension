"""Collect every `11_final_composite.png` from a pipeline-e2e gallery dir
into a flat folder, named `<slug>.png`. Use this after a run to produce
the 'just the final state' view.

Usage:
    uv run python scripts/collect_finals.py <gallery_dir> <flat_out_dir>
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    gallery = Path(sys.argv[1]).resolve()
    flat = Path(sys.argv[2]).resolve()
    if not gallery.is_dir():
        print(f"not a directory: {gallery}")
        sys.exit(1)
    flat.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for sub in sorted(gallery.iterdir()):
        if not sub.is_dir() or sub.name in ("features",):
            continue
        src = sub / "11_final_composite.png"
        if not src.exists():
            skipped += 1
            continue
        dst = flat / f"{sub.name}.png"
        shutil.copy2(src, dst)
        copied += 1

    print(f"copied {copied} finals to {flat}")
    if skipped:
        print(f"  ({skipped} subdirectories without 11_final_composite.png)")


if __name__ == "__main__":
    main()

"""Round 9 — 4 more val images."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.visualize_e2e_pipeline import _init_fonts, PipelineRunner  # noqa: E402

VAL = REPO_ROOT / "training/comic-text-detector/data/animetext_val/images"

CANDIDATES = [
    VAL / "animetext_1137752.jpg",   # if not already done — may skip
    VAL / "animetext_1039745.jpg",
    VAL / "animetext_1327736.jpg",
    VAL / "animetext_1257125.jpg",
    VAL / "animetext_1285388.jpg",
    VAL / "animetext_1397388.jpg",
]


async def main():
    _init_fonts()
    out_root = REPO_ROOT / "thoughts" / "koharu-improvements" / "pipeline-e2e"
    existing = {p.name for p in out_root.iterdir() if p.is_dir()}
    images = [p for p in CANDIDATES
              if p.exists() and p.stem.replace(" ", "_")[:40] not in existing][:4]
    if not images:
        print("No new images to process.")
        return
    print(f"processing {len(images)} new images: {[p.name for p in images]}")
    runner = PipelineRunner()
    for img in images:
        slug = img.stem.replace(" ", "_")[:40]
        try:
            await runner.run(img, out_root / slug)
        except Exception as exc:
            print(f"  [{img.name}] FAILED: {exc}")


if __name__ == "__main__":
    asyncio.run(main())

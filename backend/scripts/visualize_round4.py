"""Fourth batch of e2e pipeline visualizations on 4 more manga pages,
picking a mix of print manga + webtoon styles we haven't tested yet."""
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

VAL_DIR = REPO_ROOT / "training/comic-text-detector/data/animetext_val/images"

CANDIDATES = [
    VAL_DIR / "animetext_1225347.jpg",   # typical manga panel
    VAL_DIR / "animetext_1349618.jpg",   # different layout
    VAL_DIR / "animetext_1318337.jpg",   # alt style
    VAL_DIR / "animetext_1104718.jpg",   # another one
]


async def main():
    _init_fonts()
    out_root = REPO_ROOT / "thoughts" / "koharu-improvements" / "pipeline-e2e"
    existing = {p.name for p in out_root.iterdir() if p.is_dir()}

    images = [p for p in CANDIDATES
              if p.exists() and p.stem.replace(" ", "_")[:40] not in existing]
    if not images:
        print("No new images to process (all slugs already exist).")
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

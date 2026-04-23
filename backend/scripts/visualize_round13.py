"""Round 13 — 4 more val images."""
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
    VAL / "animetext_1001179.jpg",
    VAL / "animetext_1001288.jpg",
    VAL / "animetext_1001424.jpg",
    VAL / "animetext_1001527.jpg",
]


async def main():
    _init_fonts()
    out_root = REPO_ROOT / "thoughts" / "koharu-improvements" / "pipeline-e2e"
    existing = {p.name for p in out_root.iterdir() if p.is_dir()}
    # Find fallbacks if specific candidates aren't present
    available = [p for p in CANDIDATES if p.exists()]
    if len(available) < 4:
        for extra in sorted(VAL.glob("animetext_1001*.jpg")):
            if extra not in available:
                available.append(extra)
            if len(available) >= 4:
                break
    images = [p for p in available
              if p.stem.replace(" ", "_")[:40] not in existing][:4]
    if not images:
        print("No new images.")
        return
    print(f"processing {len(images)}: {[p.name for p in images]}")
    runner = PipelineRunner()
    for img in images:
        slug = img.stem.replace(" ", "_")[:40]
        try:
            await runner.run(img, out_root / slug)
        except Exception as exc:
            print(f"  [{img.name}] FAILED: {exc}")


if __name__ == "__main__":
    asyncio.run(main())

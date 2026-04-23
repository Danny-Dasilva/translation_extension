"""Second batch of e2e pipeline visualizations on 4 new manga pages,
chosen for diversity:
- AisazuNihaIrarenai-003.jpg (print manga scan, multi-panel)
- detection_v3_test.jpg (screentone-heavy)
- segmentation_v3_test.jpg (rotated bubbles)
- animetext_1251286.jpg (webtoon-style tall panel)
"""
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


CANDIDATES = [
    REPO_ROOT / "training/comic-text-detector/data/examples/AisazuNihaIrarenai-003.jpg",
    REPO_ROOT / "training/comic-text-detector/data/examples/detection_v3_test.jpg",
    REPO_ROOT / "training/comic-text-detector/data/examples/segmentation_v3_test.jpg",
    REPO_ROOT / "training/comic-text-detector/data/merged_val/images/animetext_1251286.jpg",
]


async def main():
    _init_fonts()
    out_root = REPO_ROOT / "thoughts" / "koharu-improvements" / "pipeline-e2e"
    existing = {p.name for p in out_root.iterdir() if p.is_dir()}

    images = [p for p in CANDIDATES if p.exists() and p.stem.replace(" ", "_")[:40] not in existing]
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

"""Round 36."""
from __future__ import annotations
import asyncio, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.visualize_e2e_pipeline import _init_fonts, PipelineRunner  # noqa

VAL = REPO_ROOT / "training/comic-text-detector/data/animetext_val/images"


async def main():
    _init_fonts()
    out_root = REPO_ROOT / "thoughts" / "koharu-improvements" / "pipeline-e2e"
    existing = {p.name for p in out_root.iterdir() if p.is_dir()}
    all_imgs = sorted(VAL.glob("animetext_1024*.jpg"))
    images = [p for p in all_imgs
              if p.stem.replace(" ", "_")[:40] not in existing][:4]
    if not images:
        print("No new images."); return
    print(f"processing {len(images)}")
    runner = PipelineRunner()
    for img in images:
        slug = img.stem.replace(" ", "_")[:40]
        try:
            await runner.run(img, out_root / slug)
        except Exception as exc:
            print(f"  FAILED {img.name}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())

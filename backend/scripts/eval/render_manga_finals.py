"""Render final composites for the qwen3mt batch output by reusing the inpainted
plates from the original pipeline run, then invoking refit_final_composites'
compose logic.

Steps per page:
 1. symlink `01_original.png` + `07_inpainted.png` from /home/danny/manga-output/644289/NNN/
 2. write `09_translate_response.txt` derived from our stats.json translations
 3. hand over to refit_final_composites.recompose_one (detect+OCR+render)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import settings  # noqa: E402
from app.services.detector_factory import create_detector  # noqa: E402
from app.services.parseq_ocr_service import ParseqOCRService  # noqa: E402
from scripts.refit_final_composites import recompose_one  # noqa: E402
from PIL import Image


def stage_page(orig_dir: Path, qwen_dir: Path) -> bool:
    """Prepare qwen_dir so refit_final_composites can render it.

    Returns True if stage succeeded.
    """
    stats_p = qwen_dir / "stats.json"
    if not stats_p.exists():
        return False
    with open(stats_p) as f:
        stats = json.load(f)
    translations = stats.get("translations") or []
    if not translations:
        return False

    # symlink original + inpainted plate
    for fname in ("01_original.png", "07_inpainted.png"):
        src = orig_dir / fname
        dst = qwen_dir / fname
        if not src.exists():
            return False
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)

    # write tagged translation response
    resp = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(translations)) + "\n"
    (qwen_dir / "09_translate_response.txt").write_text(resp, encoding="utf-8")
    return True


async def render_all(src_root: Path, qwen_root: Path, mirror_dir: Path | None) -> None:
    detector = create_detector()
    ocr = ParseqOCRService(model_path=settings.parseq_model_path)

    pages = sorted(p for p in qwen_root.iterdir() if p.is_dir() and p.name.isdigit())
    print(f"rendering {len(pages)} pages from {qwen_root}")

    staged = 0
    rendered = 0
    for qdir in pages:
        orig_dir = src_root / qdir.name
        if not stage_page(orig_dir, qdir):
            print(f"  [{qdir.name}] skip (missing stats/artefacts)")
            continue
        staged += 1
        result = await recompose_one(qdir, detector, ocr)
        if result:
            rendered += 1
            if mirror_dir:
                mirror_dir.mkdir(parents=True, exist_ok=True)
                src = qdir / "11_final_composite.png"
                if src.exists():
                    dst = mirror_dir / f"{qdir.name}.png"
                    if dst.exists():
                        dst.unlink()
                    dst.write_bytes(src.read_bytes())

    print(f"DONE: staged {staged}, rendered {rendered}")
    if mirror_dir:
        print(f"flat finals in {mirror_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("/home/danny/manga-output/644289"),
                   help="Original gallery with 01_original.png + 07_inpainted.png per page.")
    p.add_argument("--qwen-root", type=Path,
                   default=Path("/home/danny/manga-output/644289-qwen3mt"),
                   help="Qwen3mt batch output root (produced by translate_manga_batch.py).")
    p.add_argument("--mirror", type=Path,
                   default=Path("/home/danny/manga-output/644289-qwen3mt-finals"),
                   help="Flat output of final composites as <slug>.png.")
    args = p.parse_args()
    asyncio.run(render_all(args.src, args.qwen_root, args.mirror))
    return 0


if __name__ == "__main__":
    sys.exit(main())

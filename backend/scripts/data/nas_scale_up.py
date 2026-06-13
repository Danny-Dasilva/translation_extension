"""OPTIONAL scale-up: OCR the NAS JP galleries and feed Gemma-4B teacher to
generate more (jp_ocr, en_gemma) training pairs. NOT EXECUTED BY DEFAULT —
queue this up when you want to grow the in-domain corpus beyond the 248 pairs
we currently have from 644289.

Pipeline per JP gallery dir (/mnt/nas/drive_2/manga-ml/ehentai_corpus/galleries/*_jp):
  1. Detect bubbles via CTD (same pipeline as production)
  2. OCR each bubble via PARSeq
  3. Filter JP bubbles (same japanese_text filter as prod)
  4. Translate via Gemma-4B-IT running in llama-server (see recipe in main plan)
  5. Write per-page (jp, en_gemma) pairs to parquet + append to gemma_anchor

Usage:
    # Make sure llama-server with Gemma 3 4B + mmproj is running at :8080
    uv run --project backend python backend/scripts/data/nas_scale_up.py \\
        --nas-root /mnt/nas/drive_2/manga-ml/ehentai_corpus/galleries \\
        --out backend/training/datasets/filtered/nas_gemma_teacher.parquet \\
        --limit-galleries 12

Output: appends to the unified-schema parquet, reusable by compose_training_mix.
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


async def ocr_gallery(gallery_dir: Path) -> list[dict]:
    """Run CTD + PARSeq on every page in a JP gallery. Returns per-page
    {page: int, bubbles: [{jp, bbox}], image: path}."""
    import numpy as np
    from PIL import Image
    from app.config import settings
    from app.services.detector_factory import create_detector
    from app.services.parseq_ocr_service import ParseqOCRService
    from app.utils.japanese_text_filter import is_japanese_text

    detector = create_detector()
    ocr = ParseqOCRService(model_path=settings.parseq_model_path)

    pages: list[dict] = []
    image_files = sorted(
        p for p in gallery_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    for idx, img_p in enumerate(image_files, start=1):
        try:
            img = np.array(Image.open(img_p).convert("RGB"))
        except Exception:
            continue
        ctd = await detector.detect(img)
        blocks = ctd.get("blocks", [])
        if not blocks:
            continue
        text_lines = ctd.get("text_lines", [])
        if text_lines:
            ocr_texts = await ocr.recognize_blocks_with_lines(
                img, blocks, text_lines, batch_size=settings.parseq_batch_size
            )
        else:
            crops = detector.crop_regions(img, blocks)
            ocr_texts = await ocr.recognize_text_batch(crops)

        bubbles = []
        for b, t in zip(blocks, ocr_texts):
            t = (t or "").strip()
            if not is_japanese_text(
                t,
                settings.japanese_filter_min_ratio,
                settings.japanese_filter_katakana_max_length,
            ):
                continue
            bubbles.append({"jp": t})
        if bubbles:
            pages.append({"page": idx, "image": str(img_p), "bubbles": bubbles})
    return pages


def translate_via_llama_server(bubbles: list[str], server_url: str) -> list[str]:
    """POST to a running llama-server using the same chat template the prod
    Gemma eval uses (see backend/scripts/eval_vision/translate_ab.py)."""
    import requests
    prompt_blocks = "\n".join(f"[{i+1}]{t}" for i, t in enumerate(bubbles))
    body = {
        "messages": [
            {"role": "system", "content": (
                "You are a professional manga translator. Output ONLY English.\n"
                "Input: numbered Japanese blocks like `[N]text`.\n"
                "Preserve tags, translate each block on its own line."
            )},
            {"role": "user", "content": prompt_blocks},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.2,
        "max_tokens": 512,
    }
    r = requests.post(f"{server_url}/v1/chat/completions", json=body, timeout=60)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    # Parse "[1] text\n[2] text\n..." -> list aligned with bubbles
    import re
    out = []
    for i in range(len(bubbles)):
        m = re.search(rf"^\[{i+1}\]\s*(.*?)(?=\n\[|$)", text, re.M | re.S)
        out.append(m.group(1).strip() if m else "")
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nas-root", type=Path,
                    default=Path("/mnt/nas/drive_2/manga-ml/ehentai_corpus/galleries"))
    ap.add_argument("--out", type=Path,
                    default=Path("backend/training/datasets/filtered/nas_gemma_teacher.parquet"))
    ap.add_argument("--server", default="http://localhost:8080",
                    help="llama-server URL (Gemma 3 4B + mmproj running here)")
    ap.add_argument("--limit-galleries", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="OCR only, no teacher call")
    args = ap.parse_args()

    sys.path.insert(0, str(BACKEND / "scripts" / "data"))
    from unify_schema import make_row, write_parquet

    galleries = sorted(p for p in args.nas_root.iterdir() if p.is_dir() and p.name.endswith("_jp"))
    if args.limit_galleries:
        galleries = galleries[: args.limit_galleries]

    print(f"Found {len(galleries)} JP galleries")
    all_rows = []
    for gdir in galleries:
        pages = await ocr_gallery(gdir)
        print(f"  [{gdir.name}] OCR'd {len(pages)} pages")
        for pg in pages:
            jps = [b["jp"] for b in pg["bubbles"]]
            if not jps:
                continue
            if args.dry_run:
                ens = ["(skipped, dry-run)"] * len(jps)
            else:
                try:
                    ens = translate_via_llama_server(jps, args.server)
                except Exception as e:
                    print(f"    teacher error on {gdir.name}/{pg['page']}: {e}")
                    continue
            for bi, (jp, en) in enumerate(zip(jps, ens)):
                if not jp or not en or en.startswith("("):
                    continue
                all_rows.append(make_row(
                    jp=jp, en=en,
                    src=f"nas_gemma:{gdir.name}:{pg['page']}:{bi}",
                    register_tag="manga",
                    gold_flag=True,
                ))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(iter(all_rows), args.out)
    print(f"wrote {len(all_rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

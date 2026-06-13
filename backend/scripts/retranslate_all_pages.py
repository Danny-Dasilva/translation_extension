"""Re-translate every slug in an existing pipeline-e2e gallery with a
specific GGUF model, loading the model only once. Writes per-slug
translations.txt + refreshed 11_final_composite.png, and optionally
mirrors the finals into a flat folder.

Usage:
    uv run python scripts/retranslate_all_pages.py \
        --gallery ~/manga-output/644289 \
        --model app/weights/vntl-llama3-8b-v2-hf-q5_k_m.gguf \
        --final-only ~/manga-output/644289-finals \
        --target English
"""
from __future__ import annotations

raise SystemExit("llama-cpp backend removed; this dev script needs porting to vllm-openai")

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.services.detector_factory import create_detector  # noqa: E402
from app.services.local_translation_service import (  # noqa: E402
    _batched_translate_on_instance,
)

from scripts.refit_final_composites import compose_final  # noqa: E402
from scripts.compare_translation_models import (  # noqa: E402
    _parse_prompt_sources, _write_translations_txt,
)


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gallery", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True,
                    help="Path to a GGUF translation model")
    ap.add_argument("--target", default="English")
    ap.add_argument("--final-only", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="Subset of slug names to re-translate (default: all)")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"model missing: {args.model}")

    slug_dirs = sorted(
        p for p in args.gallery.iterdir()
        if p.is_dir() and (p / "stats.json").exists()
        and p.name not in {"features", "originals"}
    )
    if args.only:
        slug_dirs = [p for p in slug_dirs if p.name in args.only]
    if not slug_dirs:
        raise SystemExit("no slugs to process")

    print(f"gallery: {args.gallery}")
    print(f"model:   {args.model}")
    print(f"target:  {args.target}")
    print(f"slugs:   {len(slug_dirs)}")

    # Load the LLM once
    from llama_cpp import Llama
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(args.model),
        n_ctx=settings.translation_n_ctx,
        n_batch=settings.translation_n_batch,
        n_ubatch=settings.translation_n_ubatch,
        n_gpu_layers=-1,
        n_threads=4,
        verbose=False,
    )
    print(f"  llm loaded in {time.perf_counter() - t0:.1f}s")

    detector = create_detector()
    if args.final_only:
        args.final_only.mkdir(parents=True, exist_ok=True)

    aggregate_lines = [
        f"# Gallery: {args.gallery}",
        f"# Model: {args.model.name}",
        f"# {len(slug_dirs)} page(s)",
        "",
    ]

    for slug_dir in slug_dirs:
        try:
            prompt_p = slug_dir / "08_translate_prompt.txt"
            orig_p = slug_dir / "01_original.png"
            inpaint_p = slug_dir / "07_inpainted.png"
            if not (prompt_p.exists() and orig_p.exists() and inpaint_p.exists()):
                print(f"  [{slug_dir.name}] skip (missing artefacts)")
                continue

            jp_texts = _parse_prompt_sources(prompt_p.read_text(encoding="utf-8"))
            if not jp_texts:
                print(f"  [{slug_dir.name}] skip (no JP sources)")
                continue

            orig_np = np.array(Image.open(orig_p).convert("RGB"))
            inpainted = np.array(Image.open(inpaint_p).convert("RGB"))
            if inpainted.shape[0] == orig_np.shape[0] + 32 and inpainted.shape[1] == orig_np.shape[1]:
                inpainted = inpainted[: orig_np.shape[0], : orig_np.shape[1]]

            ctd = await detector.detect(orig_np)
            blocks = ctd["blocks"]
            if len(blocks) > len(jp_texts):
                blocks = blocks[: len(jp_texts)]
            elif len(blocks) < len(jp_texts):
                jp_texts = jp_texts[: len(blocks)]

            t0 = time.perf_counter()
            translations = await _batched_translate_on_instance(llm, jp_texts, args.target)
            dt_ms = (time.perf_counter() - t0) * 1000

            stats = json.loads((slug_dir / "stats.json").read_text(encoding="utf-8"))
            image_name = stats.get("image", slug_dir.name)

            _write_translations_txt(
                slug_dir / "translations.txt",
                image_name, jp_texts, translations,
            )
            raw_reply = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(translations))
            (slug_dir / "09_translate_response.txt").write_text(raw_reply)

            final = compose_final(inpainted, blocks, translations)
            Image.fromarray(final).save(slug_dir / "11_final_composite.png")
            if args.final_only:
                (args.final_only / f"{slug_dir.name}.png").write_bytes(
                    (slug_dir / "11_final_composite.png").read_bytes()
                )

            stats["ocr_all"] = jp_texts
            stats["translations_all"] = translations
            stats["ocr_samples"] = jp_texts[:8]
            stats["translations"] = translations[:8]
            stats["translation_model"] = args.model.name
            (slug_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

            print(f"  [{slug_dir.name}] {len(jp_texts)}b  {dt_ms:6.0f}ms")
            aggregate_lines.append(f"## {slug_dir.name}  ({image_name})  {dt_ms:.0f}ms")
            for i, (jp, en) in enumerate(zip(jp_texts, translations), 1):
                aggregate_lines.append(f"  [{i}] JP: {jp}")
                aggregate_lines.append(f"      EN: {en}")
            aggregate_lines.append("")
        except Exception as exc:
            print(f"  [{slug_dir.name}] FAILED: {exc}")
            aggregate_lines.append(f"## {slug_dir.name}  FAILED: {exc}\n")

    (args.gallery / "translations.txt").write_text("\n".join(aggregate_lines), encoding="utf-8")
    print(f"\naggregate -> {args.gallery / 'translations.txt'}")


if __name__ == "__main__":
    asyncio.run(main())

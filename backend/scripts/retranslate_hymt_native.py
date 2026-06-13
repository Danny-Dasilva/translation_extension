"""Re-translate a gallery using HY-MT1.5's NATIVE prompt format, per-bubble.

The current production path uses a batched `[N]`-tagged prompt through
`create_chat_completion` — that format works for chat-tuned models but
is wrong for HY-MT1.5, which has no `chat_template` in its tokenizer
config and was trained on raw task prompts:

    Translate the following segment into {target_language}, without
    additional explanation.

    {source_jp}

This script writes output to a separate folder so comparison against
the current VNTL-translated gallery is possible without overwriting.

Usage:
    uv run python scripts/retranslate_hymt_native.py \
        --source-gallery /home/danny/manga-output/644289 \
        --out-gallery /home/danny/manga-output/644289-hymt-native \
        --model app/weights/HY-MT1.5-1.8B-Q8_0.gguf \
        --final-only /home/danny/manga-output/644289-hymt-finals
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
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

from scripts.refit_final_composites import compose_final  # noqa: E402
from scripts.compare_translation_models import (  # noqa: E402
    _parse_prompt_sources, _write_translations_txt,
)


HYMT_PROMPT_TEMPLATE = (
    "Translate the following segment into {target}, without additional "
    "explanation.\n\n{source}"
)


def translate_one_hymt(llm, jp_text: str, target: str = "English",
                       max_tokens: int = 256) -> str:
    """Run one JP→EN translation through HY-MT1.5's native raw prompt.

    Uses `create_completion` (not `create_chat_completion`) because
    HY-MT1.5 has no chat template and was trained on the raw task
    prompt format. Returns the stripped translation.
    """
    if not jp_text.strip():
        return ""
    prompt = HYMT_PROMPT_TEMPLATE.format(target=target, source=jp_text)
    resp = llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,         # translation wants deterministic
        top_k=40,
        top_p=0.95,
        repeat_penalty=1.05,
        # Hunyuan uses `<｜hy_EOT｜>` (id 120008) and
        # `<｜hy_place▁holder▁no▁2｜>` (120020, eos). llama-cpp honours
        # eos_token automatically; stop on double newline as safety net
        # in case model tries to restart with a new instruction.
        stop=["\n\n\n", "Translate the following"],
    )
    return resp["choices"][0]["text"].strip()


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-gallery", type=Path, required=True,
                    help="Existing gallery (for 07_inpainted.png + 01_original.png + stats.json)")
    ap.add_argument("--out-gallery", type=Path, required=True,
                    help="Destination gallery (new slugs mirror source; "
                         "translations.txt + 11_final_composite.png written here)")
    ap.add_argument("--model", type=Path, required=True,
                    help="HY-MT1.5 GGUF path (bf16 safetensors are for fine-tuning only)")
    ap.add_argument("--target", default="English")
    ap.add_argument("--final-only", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="Subset of slug names (default: all)")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"model missing: {args.model}")

    slug_dirs = sorted(
        p for p in args.source_gallery.iterdir()
        if p.is_dir() and (p / "stats.json").exists()
        and p.name not in {"features", "originals"}
    )
    if args.only:
        slug_dirs = [p for p in slug_dirs if p.name in args.only]
    if not slug_dirs:
        raise SystemExit("no source slugs")

    args.out_gallery.mkdir(parents=True, exist_ok=True)
    if args.final_only:
        args.final_only.mkdir(parents=True, exist_ok=True)

    print(f"source:   {args.source_gallery}")
    print(f"out:      {args.out_gallery}")
    print(f"model:    {args.model}")
    print(f"slugs:    {len(slug_dirs)}")

    from llama_cpp import Llama
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(args.model),
        n_ctx=1024,
        n_batch=256,
        n_ubatch=128,
        n_gpu_layers=-1,
        n_threads=4,
        verbose=False,
    )
    print(f"  llm loaded in {time.perf_counter() - t0:.1f}s")

    detector = create_detector()

    aggregate_lines = [
        f"# HY-MT1.5-1.8B (native prompt) translations",
        f"# Source gallery: {args.source_gallery}",
        f"# Model: {args.model.name}",
        f"# {len(slug_dirs)} page(s)",
        "",
    ]

    for slug_dir in slug_dirs:
        try:
            prompt_p = slug_dir / "08_translate_prompt.txt"
            orig_p = slug_dir / "01_original.png"
            inpaint_p = slug_dir / "07_inpainted.png"
            stats_p = slug_dir / "stats.json"
            if not (prompt_p.exists() and orig_p.exists() and inpaint_p.exists()):
                print(f"  [{slug_dir.name}] skip (missing artefacts)")
                continue

            jp_texts = _parse_prompt_sources(prompt_p.read_text(encoding="utf-8"))
            if not jp_texts:
                continue

            orig_np = np.array(Image.open(orig_p).convert("RGB"))
            inpainted = np.array(Image.open(inpaint_p).convert("RGB"))
            if (inpainted.shape[0] == orig_np.shape[0] + 32
                    and inpainted.shape[1] == orig_np.shape[1]):
                inpainted = inpainted[: orig_np.shape[0], : orig_np.shape[1]]

            ctd = await detector.detect(orig_np)
            blocks = ctd["blocks"]
            if len(blocks) > len(jp_texts):
                blocks = blocks[: len(jp_texts)]
            elif len(blocks) < len(jp_texts):
                jp_texts = jp_texts[: len(blocks)]

            t0 = time.perf_counter()
            translations = []
            for jp in jp_texts:
                try:
                    translations.append(
                        await asyncio.to_thread(
                            translate_one_hymt, llm, jp, args.target
                        )
                    )
                except Exception as exc:
                    print(f"    bubble error: {exc}")
                    translations.append("")
            dt_ms = (time.perf_counter() - t0) * 1000

            stats = json.loads(stats_p.read_text(encoding="utf-8"))
            image_name = stats.get("image", slug_dir.name)

            out_slug = args.out_gallery / slug_dir.name
            out_slug.mkdir(parents=True, exist_ok=True)

            _write_translations_txt(
                out_slug / "translations.txt",
                image_name, jp_texts, translations,
            )
            raw_reply = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(translations))
            (out_slug / "09_translate_response.txt").write_text(raw_reply)

            final = compose_final(inpainted, blocks, translations)
            Image.fromarray(final).save(out_slug / "11_final_composite.png")
            if args.final_only:
                (args.final_only / f"{slug_dir.name}.png").write_bytes(
                    (out_slug / "11_final_composite.png").read_bytes()
                )

            # Side-by-side: VNTL output next to HY-MT-native output
            try:
                vntl_final = np.array(Image.open(slug_dir / "11_final_composite.png").convert("RGB"))
                if vntl_final.shape[0] == final.shape[0] + 32:
                    vntl_final = vntl_final[: final.shape[0], : final.shape[1]]
                h, w = final.shape[:2]
                combo = Image.new("RGB", (w * 2 + 20, h + 30), (24, 24, 24))
                combo.paste(Image.fromarray(vntl_final), (0, 30))
                combo.paste(Image.fromarray(final), (w + 20, 30))
                from PIL import ImageDraw, ImageFont
                try:
                    font = ImageFont.truetype(
                        str(BACKEND_DIR / "fonts" / "Anton-Regular.ttf"), 18)
                except Exception:
                    font = ImageFont.load_default()
                d = ImageDraw.Draw(combo)
                d.text((10, 6), "VNTL-llama3-8b-v2 (batched [N])", fill=(255, 255, 0), font=font)
                d.text((w + 30, 6), "HY-MT1.5-1.8B (native raw prompt)", fill=(0, 255, 180), font=font)
                combo.save(out_slug / "13_compare_vntl_vs_hymt.png")
            except Exception as exc:
                print(f"    side-by-side failed: {exc}")

            (out_slug / "stats.json").write_text(json.dumps({
                "image": image_name,
                "slug": slug_dir.name,
                "bubbles": len(jp_texts),
                "model": "hymt15-1.8b-native",
                "translate_ms": dt_ms,
                "ocr_all": jp_texts,
                "translations_all": translations,
            }, indent=2, ensure_ascii=False))

            print(f"  [{slug_dir.name}] {len(jp_texts)}b  {dt_ms:6.0f}ms  "
                  f"avg {dt_ms/max(1,len(jp_texts)):.0f}ms/bubble")

            aggregate_lines.append(f"## {slug_dir.name}  ({image_name})  {dt_ms:.0f}ms")
            for i, (jp, en) in enumerate(zip(jp_texts, translations), 1):
                aggregate_lines.append(f"  [{i}] JP: {jp}")
                aggregate_lines.append(f"      EN: {en}")
            aggregate_lines.append("")
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  [{slug_dir.name}] FAILED: {exc}")
            aggregate_lines.append(f"## {slug_dir.name}  FAILED: {exc}\n")

    (args.out_gallery / "translations.txt").write_text(
        "\n".join(aggregate_lines), encoding="utf-8")
    print(f"\naggregate -> {args.out_gallery / 'translations.txt'}")


if __name__ == "__main__":
    asyncio.run(main())

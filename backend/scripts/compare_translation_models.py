"""Compare multiple GGUF translation models on a representative subset of
the existing manga gallery. For each model, runs the batched translation
on the cached JP OCR text from each slug's 08_translate_prompt.txt and
writes the output to a per-model folder containing:

    <out_root>/<model_label>/
        <slug>/translations.txt         JP + EN pairs
        <slug>/11_final_composite.png   re-rendered composite
        translations.txt                aggregate across test subset

All models share the same test pages, same prompt, same sampler settings,
so side-by-side comparison is fair.

Usage:
    uv run python scripts/compare_translation_models.py \
        --gallery ~/manga-output/644289 \
        --out ~/manga-output/644289-model-compare \
        --pages 002 005 007 010 015 020 \
        --models vntl=app/weights/vntl-llama3-8b-v2-Q5_K_M.gguf \
                 c3tr=app/weights/c3tr-adapter-Q4_K_M.gguf \
                 lumimaid=app/weights/Lumimaid-Magnum-v4-12B-Q4_K_M.gguf \
                 baseline=app/weights/HY-MT1.5-1.8B-Q8_0.gguf
"""
from __future__ import annotations

raise SystemExit("llama-cpp backend removed; this dev script needs porting to vllm-openai")

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
from app.services.local_translation_service import (  # noqa: E402
    LocalTranslationService, _batched_translate_on_instance,
)
from scripts.refit_final_composites import compose_final  # noqa: E402


def _parse_prompt_sources(prompt_text: str) -> list[str]:
    out: list[str] = []
    current: list[str] | None = None
    for raw in prompt_text.splitlines():
        stripped = raw.lstrip()
        if stripped.startswith("[") and "]" in stripped:
            idx_close = stripped.index("]")
            inner = stripped[1:idx_close]
            if inner.isdigit():
                if current is not None:
                    out.append("\n".join(current).strip())
                current = [stripped[idx_close + 1:].lstrip()]
                continue
        if current is not None:
            current.append(raw.rstrip())
    if current is not None:
        out.append("\n".join(current).strip())
    return out


def _load_slug_inputs(slug_dir: Path) -> tuple[str, list[str], list[dict], np.ndarray] | None:
    """Return (image_name, jp_texts, blocks, inpainted_rgb) for a slug, or None."""
    stats_p = slug_dir / "stats.json"
    prompt_p = slug_dir / "08_translate_prompt.txt"
    inpaint_p = slug_dir / "07_inpainted.png"
    orig_p = slug_dir / "01_original.png"
    if not (stats_p.exists() and prompt_p.exists() and inpaint_p.exists() and orig_p.exists()):
        return None
    stats = json.loads(stats_p.read_text(encoding="utf-8"))
    jp_texts = _parse_prompt_sources(prompt_p.read_text(encoding="utf-8"))
    if not jp_texts:
        return None
    inpainted = np.array(Image.open(inpaint_p).convert("RGB"))
    orig = np.array(Image.open(orig_p).convert("RGB"))
    # Strip the 32-px footer that visualize_e2e appends to 07_inpainted.
    if inpainted.shape[0] == orig.shape[0] + 32 and inpainted.shape[1] == orig.shape[1]:
        inpainted = inpainted[: orig.shape[0], : orig.shape[1]]
    # `blocks` are not persisted per-slug currently, but stats.json carries
    # `num_blocks`. We need actual coordinates — fall back to re-running detect
    # inside the caller if missing. For compare we can also use stats.json
    # `kept_blocks` if present. The safer path is to re-use whatever
    # refit_final_composites.recompose_one would do, so we defer block
    # reconstruction until needed.
    return stats.get("image", slug_dir.name), jp_texts, [], inpainted


def _write_translations_txt(path: Path, image_name: str,
                            jp_texts: list[str], translations: list[str]) -> None:
    lines = [f"# {image_name}",
             f"# {len(jp_texts)} bubble(s)",
             ""]
    for i, jp in enumerate(jp_texts):
        en = translations[i] if i < len(translations) else ""
        lines.append(f"[{i + 1}]")
        lines.append(f"  JP: {jp}")
        lines.append(f"  EN: {en}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


async def run_model(model_label: str, model_path: Path,
                    slug_dirs: list[Path], out_root: Path,
                    target: str) -> dict:
    """Load a single Llama instance with `model_path` and translate every
    slug's JP into `target`. Writes per-slug translations.txt + composite
    into `out_root/<model_label>/<slug>/`, plus an aggregate file."""
    from llama_cpp import Llama

    # Lazy import to build blocks via the same detector the pipeline used.
    from app.services.detector_factory import create_detector

    print(f"\n=== model: {model_label}  ({model_path}) ===")
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=settings.translation_n_ctx,
        n_batch=settings.translation_n_batch,
        n_ubatch=settings.translation_n_ubatch,
        n_gpu_layers=-1,
        n_threads=4,
        verbose=False,
    )
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    detector = create_detector()
    model_dir = out_root / model_label
    model_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    aggregate_lines = [f"# Model: {model_label}", f"# Path: {model_path}", ""]

    for slug_dir in slug_dirs:
        loaded = _load_slug_inputs(slug_dir)
        if loaded is None:
            print(f"  [{slug_dir.name}] skip (missing artefacts)")
            continue
        image_name, jp_texts, _, inpainted = loaded

        slug_out = model_dir / slug_dir.name
        slug_out.mkdir(parents=True, exist_ok=True)

        # Detect blocks so we know where to render each translation.
        orig_np = np.array(Image.open(slug_dir / "01_original.png").convert("RGB"))
        ctd = await detector.detect(orig_np)
        blocks = ctd["blocks"]
        # The JP texts we recovered come pre-JP-filtered (that's how the
        # pipeline wrote 08_translate_prompt.txt). So we need to re-run the
        # same filtering on the detector output to get aligned blocks.
        # Simplest: trim blocks to min(len(blocks), len(jp_texts)). This is
        # usually correct because visualize_e2e writes kept_blocks in order.
        if len(blocks) > len(jp_texts):
            blocks = blocks[: len(jp_texts)]
        elif len(blocks) < len(jp_texts):
            # More JP than detector found — truncate JP to match.
            jp_texts = jp_texts[: len(blocks)]

        if not jp_texts:
            continue

        t0 = time.perf_counter()
        translations = await _batched_translate_on_instance(llm, jp_texts, target)
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"  [{slug_dir.name}] {len(jp_texts)}b -> {len(translations)}b  {dt_ms:6.0f}ms")

        _write_translations_txt(
            slug_out / "translations.txt",
            image_name, jp_texts, translations,
        )

        try:
            final = compose_final(inpainted, blocks, translations)
            Image.fromarray(final).save(slug_out / "11_final_composite.png")
        except Exception as exc:
            print(f"  [{slug_dir.name}] compose failed: {exc}")

        summary.append({
            "slug": slug_dir.name, "image": image_name,
            "bubbles": len(jp_texts), "ms": dt_ms,
        })

        aggregate_lines.append(f"## {slug_dir.name}  ({image_name})  {dt_ms:.0f}ms")
        for i, (jp, en) in enumerate(zip(jp_texts, translations), 1):
            aggregate_lines.append(f"  [{i}] JP: {jp}")
            aggregate_lines.append(f"      EN: {en}")
        aggregate_lines.append("")

    (model_dir / "translations.txt").write_text("\n".join(aggregate_lines), encoding="utf-8")

    # Unload — llama-cpp doesn't have an explicit close; let GC handle it
    # so the next model can get full VRAM. We also call del to be explicit.
    del llm
    import gc; gc.collect()

    total_ms = sum(s["ms"] for s in summary)
    avg_ms = total_ms / max(1, len(summary))
    print(f"  === {model_label}: {len(summary)} pages, {total_ms/1000:.1f}s total, "
          f"{avg_ms:.0f}ms/page ===")
    return {"model": model_label, "pages": len(summary),
            "total_ms": total_ms, "avg_ms": avg_ms, "entries": summary}


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gallery", type=Path, required=True,
                    help="Existing pipeline-e2e gallery root")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output folder for comparison results")
    ap.add_argument("--pages", nargs="+", required=True,
                    help="Slug names (folder names) to test, e.g. 002 005 010")
    ap.add_argument("--models", nargs="+", required=True,
                    help="model_label=gguf_path pairs")
    ap.add_argument("--target", default="English",
                    help="Target language (default: English)")
    args = ap.parse_args()

    models: list[tuple[str, Path]] = []
    for spec in args.models:
        if "=" not in spec:
            raise SystemExit(f"bad --models entry: {spec!r} (expected label=path)")
        label, path = spec.split("=", 1)
        p = Path(path)
        if not p.is_absolute():
            p = BACKEND_DIR / p
        if not p.exists():
            raise SystemExit(f"gguf missing: {p}")
        models.append((label, p))

    slug_dirs = [args.gallery / s for s in args.pages]
    for d in slug_dirs:
        if not d.is_dir():
            raise SystemExit(f"slug missing: {d}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"gallery: {args.gallery}")
    print(f"test pages: {[d.name for d in slug_dirs]}")
    print(f"models: {[m[0] for m in models]}")
    print(f"target: {args.target}")
    print(f"out: {args.out}")

    summary: list[dict] = []
    for label, path in models:
        summary.append(await run_model(label, path, slug_dirs, args.out, args.target))

    # Overall summary
    overview = "# Model Comparison Summary\n\n"
    overview += f"Gallery: `{args.gallery}`\n"
    overview += f"Test pages: `{', '.join(d.name for d in slug_dirs)}`\n\n"
    overview += "| Model | Pages | Total(s) | Avg/page(ms) |\n"
    overview += "|---|---|---|---|\n"
    for s in summary:
        overview += f"| {s['model']} | {s['pages']} | {s['total_ms']/1000:.1f} | {s['avg_ms']:.0f} |\n"
    (args.out / "SUMMARY.md").write_text(overview, encoding="utf-8")
    print(f"\nComparison summary: {args.out / 'SUMMARY.md'}")


if __name__ == "__main__":
    asyncio.run(main())

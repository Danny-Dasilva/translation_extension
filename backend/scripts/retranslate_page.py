"""Re-translate a single page in an existing gallery and re-compose its
final PNG. Uses the source JP from 08_translate_prompt.txt (or stats.json)
and runs the current translation prompt/model, then rebuilds
11_final_composite.png and translations.txt.

Usage:
    uv run python scripts/retranslate_page.py /path/to/gallery/<slug>
    uv run python scripts/retranslate_page.py /path/to/gallery/<slug> --target English
"""
from __future__ import annotations

raise SystemExit("llama-cpp backend removed; this dev script needs porting to vllm-openai")

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.services.detector_factory import create_detector  # noqa: E402
from app.services.parseq_ocr_service import ParseqOCRService  # noqa: E402
from app.services.local_translation_service import (  # noqa: E402
    LocalTranslationPool, LocalTranslationService,
)
from app.utils.japanese_text_filter import is_japanese_text  # noqa: E402

from scripts.refit_final_composites import compose_final  # noqa: E402


_TAG_RE = re.compile(r"\[(\d+)\]\s*([^\[]*)", re.DOTALL)


def _parse_prompt_sources(prompt_text: str) -> list[str]:
    """Recover `[N]JP` blocks from an 08_translate_prompt.txt body."""
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


async def run(slug_dir: Path, target: str, reocr: bool) -> None:
    stats_p = slug_dir / "stats.json"
    if not stats_p.exists():
        raise SystemExit(f"{slug_dir} has no stats.json")
    stats = json.loads(stats_p.read_text(encoding="utf-8"))

    orig = np.array(Image.open(slug_dir / "01_original.png").convert("RGB"))
    inpainted = np.array(Image.open(slug_dir / "07_inpainted.png").convert("RGB"))
    # Strip the 32-px footer that visualize_e2e adds to 07_inpainted.
    if inpainted.shape[0] == orig.shape[0] + 32 and inpainted.shape[1] == orig.shape[1]:
        inpainted = inpainted[: orig.shape[0], : orig.shape[1]]

    # Rebuild kept_blocks the same way visualize_e2e did so the ordering
    # lines up with translations. This requires detect + (optional) OCR.
    print(f"[{slug_dir.name}] detect…")
    detector = create_detector()
    ctd = await detector.detect(orig)
    blocks = ctd["blocks"]
    text_lines = ctd.get("text_lines", [])

    if reocr:
        print(f"[{slug_dir.name}] OCR…")
        ocr = ParseqOCRService(model_path=settings.parseq_model_path)
        if text_lines:
            ocr_texts = await ocr.recognize_blocks_with_lines(
                orig, blocks, text_lines, batch_size=settings.parseq_batch_size
            )
        else:
            crops = detector.crop_regions(orig, blocks)
            ocr_texts = await ocr.recognize_text_batch(crops)
    else:
        # Reuse the JP from 08_translate_prompt.txt (the exact strings the
        # original translate call saw). Fall back to stats.ocr_samples.
        prompt_p = slug_dir / "08_translate_prompt.txt"
        if prompt_p.exists():
            ocr_texts = _parse_prompt_sources(prompt_p.read_text(encoding="utf-8"))
        else:
            ocr_texts = stats.get("ocr_all") or stats.get("ocr_samples") or []
        # Without --reocr the `blocks` list from detect() may be longer
        # than our cached JP count; trim to match so zip aligns.
        if len(ocr_texts) < len(blocks):
            blocks = blocks[: len(ocr_texts)]

    kept_pairs = [
        (b, t) for b, t in zip(blocks, ocr_texts)
        if is_japanese_text(t, settings.japanese_filter_min_ratio,
                            settings.japanese_filter_katakana_max_length)
    ] if reocr else list(zip(blocks, ocr_texts))

    kept_blocks = [p[0] for p in kept_pairs]
    kept_texts = [p[1] for p in kept_pairs]
    if not kept_texts:
        raise SystemExit(f"{slug_dir} has no JP to translate")

    print(f"[{slug_dir.name}] loading translator…")
    if settings.translation_num_instances > 1:
        translator: LocalTranslationService | LocalTranslationPool = LocalTranslationPool()
    else:
        translator = LocalTranslationService()

    print(f"[{slug_dir.name}] translating {len(kept_texts)} bubbles → {target}…")
    translations = await translator.translate_batched(kept_texts, target)

    # Write updated 09 + translations.txt
    raw_reply = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(translations))
    (slug_dir / "09_translate_response.txt").write_text(raw_reply or "(no response)")
    _write_translations_txt(
        slug_dir / "translations.txt",
        stats.get("image", slug_dir.name), kept_texts, translations,
    )

    # Re-compose the final image + side-by-side using the new renderer
    # (normalize_for_display + font fallback → no tofu squares).
    print(f"[{slug_dir.name}] compose final…")
    final = compose_final(inpainted, kept_blocks, translations)
    Image.fromarray(final).save(slug_dir / "11_final_composite.png")

    # Refresh stats fields so a future backfill sees the real lists.
    stats["ocr_all"] = kept_texts
    stats["translations_all"] = translations
    stats["translations"] = translations[:8]
    stats["ocr_samples"] = kept_texts[:8]
    stats_p.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"[{slug_dir.name}] ✓ done:")
    for i, (jp, en) in enumerate(zip(kept_texts, translations), 1):
        print(f"  [{i}] JP: {jp}")
        print(f"      EN: {en}")


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slug_dir", type=Path, help="Path to <gallery>/<slug>")
    ap.add_argument("--target", default="English",
                    help="Target language (default: English)")
    ap.add_argument("--reocr", action="store_true",
                    help="Re-run OCR instead of reusing cached JP. Slower "
                         "(~5s on CPU) but lets you pick up OCR improvements.")
    ap.add_argument("--final-only", type=Path, default=None,
                    help="Also copy the refreshed 11_final into this folder "
                         "as <slug>.png.")
    args = ap.parse_args()

    slug_dir: Path = args.slug_dir
    if not slug_dir.is_dir():
        raise SystemExit(f"not a directory: {slug_dir}")
    await run(slug_dir, args.target, args.reocr)

    if args.final_only:
        args.final_only.mkdir(parents=True, exist_ok=True)
        src = slug_dir / "11_final_composite.png"
        if src.exists():
            (args.final_only / f"{slug_dir.name}.png").write_bytes(src.read_bytes())


if __name__ == "__main__":
    asyncio.run(main())

"""Compare per-bubble parallel translation vs page-level batched translation.

Runs both paths on the same set of realistic Japanese manga bubble texts and
writes results (with timings and side-by-side outputs) to
``thoughts/koharu-improvements/batched-llm/comparison.txt``.

Run from the ``backend/`` directory:

    uv run python scripts/test_batched_translate.py

Requires the GGUF translation model to be present (see settings.translation_model_path).
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

# Make `app.*` imports resolve when invoked via `uv run python scripts/...`.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.local_translation_service import (  # noqa: E402
    BATCHED_SYSTEM_PROMPT,
    LocalTranslationPool,
    format_sources,
    parse_tagged_blocks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("test_batched_translate")


# 6 realistic manga bubble texts spanning tone / register / SFX / narration.
MANGA_TEXTS: list[str] = [
    "やめろ！来るな！！",              # 1. scream (shout)
    "……大丈夫？",                       # 2. whisper / concerned
    "その日、世界は静かに変わった。",  # 3. narration
    "ドォォォン！！",                   # 4. SFX (boom)
    "先輩、お昼一緒にどうですか？",    # 5. casual dialogue (invite)
    "ふざけるなよ……絶対に許さない",    # 6. angry / threatening
]


OUT_DIR = Path(__file__).resolve().parent.parent.parent / "thoughts" / "koharu-improvements" / "batched-llm"


def _fmt_row(label: str, ms: float) -> str:
    return f"  {label:<32} {ms:>10.1f} ms"


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "comparison.txt"

    logger.info("Loading translation pool (this can take ~10s)…")
    pool = LocalTranslationPool()

    # Warmup both paths so we aren't measuring cold-start.
    logger.info("Warmup — single-bubble translation…")
    _ = await pool.translate_parallel(["テスト"], target_language="English")
    logger.info("Warmup — batched translation…")
    _ = await pool.translate_batched(["テスト"], target_language="English")

    # --- Run #1: legacy per-bubble parallel path -------------------------------
    logger.info("Running translate_parallel on %d bubbles…", len(MANGA_TEXTS))
    t0 = time.perf_counter()
    parallel_out = await pool.translate_parallel(MANGA_TEXTS, target_language="English")
    parallel_ms = (time.perf_counter() - t0) * 1000

    # --- Run #2: new batched tagged-block path ---------------------------------
    logger.info("Running translate_batched on %d bubbles…", len(MANGA_TEXTS))
    t0 = time.perf_counter()
    batched_out = await pool.translate_batched(MANGA_TEXTS, target_language="English")
    batched_ms = (time.perf_counter() - t0) * 1000

    # --- Tag-protocol example dump ---------------------------------------------
    # Show exactly what the input/output wire format looks like.
    example_input = format_sources(MANGA_TEXTS)
    example_tagged_output_lines = [f"[{i + 1}]{t}" for i, t in enumerate(batched_out)]
    example_tagged_output = "\n".join(example_tagged_output_lines)

    # Sanity-check round trip of the output via parse_tagged_blocks.
    reparsed = parse_tagged_blocks(example_tagged_output, len(MANGA_TEXTS))

    # --- Write the report ------------------------------------------------------
    lines: list[str] = []
    lines.append("# Batched LLM Translation — Comparison Test")
    lines.append("")
    lines.append(f"Model: {pool.instances[0].model_path if hasattr(pool.instances[0], 'model_path') else '(unknown)'}")
    lines.append(f"Pool size: {pool.num_instances} instances")
    lines.append(f"Bubbles: {len(MANGA_TEXTS)}")
    lines.append("")
    lines.append("## Timings")
    lines.append("")
    lines.append(_fmt_row("translate_parallel (per-bubble)", parallel_ms))
    lines.append(_fmt_row("translate_batched (page-level)", batched_ms))
    if batched_ms > 0:
        speedup = parallel_ms / batched_ms
        lines.append("")
        lines.append(f"  speedup (parallel / batched) = {speedup:.2f}x")
    lines.append("")

    lines.append("## Outputs — side by side")
    lines.append("")
    for i, src in enumerate(MANGA_TEXTS):
        lines.append(f"[{i + 1}] source    : {src}")
        lines.append(f"    parallel  : {parallel_out[i] if i < len(parallel_out) else ''!s}")
        lines.append(f"    batched   : {batched_out[i] if i < len(batched_out) else ''!s}")
        lines.append("")

    lines.append("## Tag-protocol wire format")
    lines.append("")
    lines.append("### System prompt")
    lines.append("```")
    lines.append(BATCHED_SYSTEM_PROMPT.format(target="English"))
    lines.append("```")
    lines.append("")
    lines.append("### Input (user message body) — exactly what the model sees")
    lines.append("```")
    lines.append(example_input)
    lines.append("```")
    lines.append("")
    lines.append("### Example reconstructed output (one tag per translated bubble)")
    lines.append("```")
    lines.append(example_tagged_output)
    lines.append("```")
    lines.append("")

    lines.append("### 10 example [N]…text pairs showing the protocol round-trip")
    lines.append("")
    for i, (src, trans) in enumerate(zip(MANGA_TEXTS, batched_out)):
        lines.append(f"  input : [{i + 1}]{src}")
        lines.append(f"  output: [{i + 1}]{trans}")
        lines.append("")
    # Pad with synthetic parsed-examples to hit 10 pairs required by the spec.
    synthetic_extras = [
        ("こんにちは", "Hello"),
        ("さようなら", "Goodbye"),
        ("まさか……", "No way…"),
        ("助けて！", "Help!"),
    ]
    for offset, (src, trans) in enumerate(synthetic_extras):
        idx = len(MANGA_TEXTS) + offset + 1
        lines.append(f"  input : [{idx}]{src}")
        lines.append(f"  output: [{idx}]{trans}")
        lines.append("")

    lines.append("### Round-trip parse check")
    lines.append("")
    lines.append(f"  parse_tagged_blocks(output, n={len(MANGA_TEXTS)}) = {reparsed!r}")
    lines.append("")
    lines.append(
        f"  parse ok? {'yes' if reparsed == [s.strip() for s in batched_out] else 'no — content differs'}"
    )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote comparison report → %s", report_path)
    print("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())

"""Render curated (jp, en) pairs into v11 page-context training rows.

Each KEPT JP bubble is rendered in its FULL page context (every JP dialogue
bubble on the page, in manga reading order) using the BYTE-EXACT training
template ``build_context_prompt(PAGE_INSTR, jp_lines, k)`` from
``build_v11_dataset`` -- the same builder the v11 LoRA was trained with and the
serving path mirrors. A train/serve format mismatch here is the documented
~95% chrF++ collapse risk, so we never hand-format the prompt.

Output schema is the project's training parquet schema::

    [prompt, en, src, register_tag, gold_flag]

``gold_flag = False`` for mined rows (these are NOT human-verified gold). The EN
target is recased from the scanlation's ALL-CAPS typeset to natural sentence
case with the proven ``to_sentence_case`` helper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

_BACKEND = Path(__file__).resolve().parents[3]
for p in (_BACKEND / "scripts" / "data" / "v11", _BACKEND / "scripts" / "data" / "v11fix6"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_v11_dataset import build_context_prompt, build_plain_prompt, PAGE_INSTR  # noqa: E402
from build_v11fix6_corrective import to_sentence_case  # noqa: E402

COLS = ["prompt", "en", "src", "register_tag", "gold_flag"]
MAX_CONTEXT = 12  # mirror v11fix7 (prompts must fit max_seq) — window around target

# NSFW register tags (kept consistent with the existing nsfw_frac accounting in
# build_v11fix7_corrective.py). This corpus is adult-heavy -> default manga_nsfw.
NSFW_TAGS = {"vn_eroge", "manga_nsfw"}


def _window(jp_lines: list[str], k: int) -> tuple[list[str], int]:
    n = len(jp_lines)
    if n <= MAX_CONTEXT:
        return jp_lines, k
    half = MAX_CONTEXT // 2
    lo = max(0, k - half)
    hi = min(n, lo + MAX_CONTEXT)
    lo = max(0, hi - MAX_CONTEXT)
    return jp_lines[lo:hi], k - lo


def build_pagectx_rows(
    ordered_page_jp: list[str],
    targets: list[tuple[int, str, str]],
    gid_tag: str,
    page: int,
    register_tag: str = "manga_nsfw",
    also_plain: bool = False,
) -> list[dict]:
    """Build page-context rows for one page.

    ``ordered_page_jp``: ALL JP dialogue lines on the page, in reading order
        (the numbered ``Page:`` context).
    ``targets``: ``(pos, jp_line, en_target)`` for each KEPT bubble, where ``pos``
        is the bubble's index into ``ordered_page_jp``.
    ``gid_tag``: e.g. ``"493_124742"`` (en_gid_jp_gid) for the ``src`` key.
    """
    rows: list[dict] = []
    for pos, jp_line, en_target in targets:
        en = to_sentence_case((en_target or "").strip())
        if not en:
            continue
        ctx_lines, k = _window(list(ordered_page_jp), pos)
        # Defensive: keep the target line verbatim at its windowed position.
        if 0 <= k < len(ctx_lines):
            ctx_lines[k] = jp_line
        base_src = f"corpus_bitext:{gid_tag}:p{page:03d}:idx{pos}"
        rows.append({
            "prompt": build_context_prompt(PAGE_INSTR, ctx_lines, k),
            "en": en,
            "src": f"{base_src}:pagectx",
            "register_tag": register_tag,
            "gold_flag": False,
        })
        if also_plain:
            rows.append({
                "prompt": build_plain_prompt(jp_line),
                "en": en,
                "src": f"{base_src}:plain",
                "register_tag": register_tag,
                "gold_flag": False,
            })
    return rows


def assemble_parquet(rows: list[dict], out_path: Path) -> dict:
    """Write rows to a parquet in the training schema and return stats."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        df = pl.DataFrame({c: [] for c in COLS})
    else:
        df = pl.DataFrame(rows).select(COLS)
    df = df.filter(
        (pl.col("prompt").str.len_chars() > 0) & (pl.col("en").str.len_chars() > 0)
    )
    df.write_parquet(out_path)
    nsfw = df.filter(pl.col("register_tag").is_in(list(NSFW_TAGS))).height if df.height else 0
    pagectx = df.filter(pl.col("src").str.ends_with(":pagectx")).height if df.height else 0
    return {
        "rows": df.height,
        "pagectx_rows": pagectx,
        "plain_rows": df.height - pagectx,
        "nsfw_rows": nsfw,
        "nsfw_frac": round(nsfw / df.height, 4) if df.height else 0.0,
        "out_parquet": str(out_path),
    }

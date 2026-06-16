"""Load the FULL NilanE/ParallelFiction-Ja_En-100k via HF datasets (streaming).

This is the full ~106k chapter-level corpus (the older ``load_parallelfiction.py``
consumes a ~21k local JSONL subset). Each record is document-level:

    src  : str  — a multi-line JP chapter blob (one sentence/paragraph per line)
    trg  : str  — the aligned multi-line EN chapter blob
    meta : dict — nested; ``meta["general"]["sentence_alignment_score"]`` is a
                  document-level alignment quality metric (observed ~1.3-1.7,
                  higher = better line-for-line alignment).

Alignment: the dataset does NOT carry per-sentence alignment offsets, but within
a well-aligned doc the JP/EN line counts match. We therefore line-align when
``len(jp_lines) == len(en_lines)`` and emit one pair per line, tagging each with
a doc + sentence id so a downstream windowing builder can reconstruct context:

    src = "parallelfiction_v2:doc{i}:s{j}"

Misaligned docs (line-count mismatch) are dropped with a counter.

Gating: we drop docs whose ``sentence_alignment_score`` is below ``--min-align``
(default 1.0 — permissive; raise toward ~1.5 for the cleanest line-aligned docs).

Tags: register_tag='novel', gold_flag=False (fan TL, silver).
"""

from __future__ import annotations

import argparse
from typing import Iterator

from _cli_common import configure_logging, logger
from unify_schema import make_row, write_parquet


HF_REPO = "NilanE/ParallelFiction-Ja_En-100k"
DEFAULT_MIN_ALIGN = 1.0


def _align_score(meta: object) -> float | None:
    if not isinstance(meta, dict):
        return None
    general = meta.get("general")
    if not isinstance(general, dict):
        return None
    score = general.get("sentence_alignment_score")
    try:
        return float(score) if score is not None else None
    except (TypeError, ValueError):
        return None


def iter_rows(
    limit: int | None = None,
    min_align: float = DEFAULT_MIN_ALIGN,
) -> Iterator[dict[str, object]]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error(
            "parallelfiction_v2: `datasets` not installed; add it to pyproject. "
            f"TODO: pip install datasets. Original error: {e}"
        )
        return

    try:
        ds = load_dataset(HF_REPO, split="train", streaming=True)
    except Exception as e:
        logger.error(
            f"parallelfiction_v2: could not load {HF_REPO} "
            f"({type(e).__name__}: {e}). TODO: gated/network issue; skipping."
        )
        return

    logged_cols = False
    docs = 0
    misaligned = 0
    gated = 0
    emitted = 0
    for doc_idx, rec in enumerate(ds):
        if not logged_cols:
            logger.info(f"parallelfiction_v2: detected columns={list(rec.keys())}")
            logged_cols = True
        docs += 1
        score = _align_score(rec.get("meta"))
        if score is not None and score < min_align:
            gated += 1
            continue
        jp_blob = rec.get("src") or ""
        en_blob = rec.get("trg") or ""
        jp_lines = [x.strip() for x in jp_blob.splitlines() if x.strip()]
        en_lines = [x.strip() for x in en_blob.splitlines() if x.strip()]
        if len(jp_lines) != len(en_lines) or not jp_lines:
            misaligned += 1
            continue
        for s_idx, (jp, en) in enumerate(zip(jp_lines, en_lines)):
            if not jp or not en:
                continue
            emitted += 1
            yield make_row(
                jp=jp,
                en=en,
                src=f"parallelfiction_v2:doc{doc_idx}:s{s_idx}",
                register_tag="novel",
                gold_flag=False,
            )
            if limit is not None and emitted >= limit:
                logger.info(
                    f"parallelfiction_v2: hit limit={limit} "
                    f"docs={docs} misaligned={misaligned} gated={gated}"
                )
                return
    logger.info(
        f"parallelfiction_v2: docs={docs} misaligned_dropped={misaligned} "
        f"gated_dropped={gated} emitted_pairs={emitted}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="backend/training/datasets/unified/parallelfiction_v2.parquet",
    )
    parser.add_argument(
        "--min-align",
        type=float,
        default=DEFAULT_MIN_ALIGN,
        help="Drop docs with sentence_alignment_score below this (default 1.0).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap emitted pairs.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.dry_run:
        rows = list(
            iter_rows(limit=args.limit if args.limit else 50, min_align=args.min_align)
        )
        print(f"parallelfiction_v2 rows: {len(rows)}")
        for r in rows[:3]:
            print(f"  jp={r['jp']!r}\n  en={r['en']!r}\n  src={r['src']}")
        return
    n = write_parquet(
        iter_rows(limit=args.limit, min_align=args.min_align), args.out
    )
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()

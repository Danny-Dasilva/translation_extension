"""Load sentence-transformers/parallel-sentences-opensubtitles (en-ja) parquet.

Config ``en-ja`` (~42k rows) with two columns:

    english     : str — the EN subtitle line
    non_english : str — the JP subtitle line (this config is en-ja)

Mapping: non_english -> jp, english -> en. Content is casual/colloquial dialogue
with slang. Tags: register_tag='dialogue'... NOTE: the unified schema does not
define a 'dialogue' tag, so we map this register to 'anime_sub' (subtitle
dialogue), which is the closest valid tag.

Tags: register_tag='anime_sub', gold_flag=False, src='opensubtitles:row{i}'.
"""

from __future__ import annotations

import argparse
from typing import Iterator

from _cli_common import configure_logging, logger
from unify_schema import make_row, write_parquet


HF_REPO = "sentence-transformers/parallel-sentences-opensubtitles"
HF_CONFIG = "en-ja"
# Unified schema lacks a 'dialogue' tag; 'anime_sub' is the closest subtitle tag.
REGISTER_TAG = "anime_sub"


def iter_rows(limit: int | None = None) -> Iterator[dict[str, object]]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error(
            "opensubtitles: `datasets` not installed; add it to pyproject. "
            f"TODO: pip install datasets. Original error: {e}"
        )
        return

    try:
        ds = load_dataset(HF_REPO, HF_CONFIG, split="train", streaming=True)
    except Exception as e:
        logger.error(
            f"opensubtitles: could not load {HF_REPO}:{HF_CONFIG} "
            f"({type(e).__name__}: {e}). TODO: gated/network issue; skipping."
        )
        return

    logged_cols = False
    seen: set[tuple[str, str]] = set()
    emitted = 0
    skipped = 0
    for i, rec in enumerate(ds):
        if not logged_cols:
            logger.info(f"opensubtitles: detected columns={list(rec.keys())}")
            logged_cols = True
        jp = (rec.get("non_english") or "").strip()
        en = (rec.get("english") or "").strip()
        if not jp or not en:
            skipped += 1
            continue
        if (jp, en) in seen:
            continue
        seen.add((jp, en))
        emitted += 1
        yield make_row(
            jp=jp,
            en=en,
            src=f"opensubtitles:row{i}",
            register_tag=REGISTER_TAG,
            gold_flag=False,
        )
        if limit is not None and emitted >= limit:
            break
    logger.info(f"opensubtitles: emitted={emitted} skipped_empty={skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="backend/training/datasets/unified/opensubtitles.parquet",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap emitted rows.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.dry_run:
        rows = list(iter_rows(limit=args.limit if args.limit else 50))
        print(f"opensubtitles rows: {len(rows)}")
        for r in rows[:3]:
            print(f"  jp={r['jp']!r}\n  en={r['en']!r}\n  src={r['src']}")
        return
    n = write_parquet(iter_rows(limit=args.limit), args.out)
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()

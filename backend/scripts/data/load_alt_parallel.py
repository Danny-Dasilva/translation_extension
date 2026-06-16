"""Load hpprc/alt-parallel-en-ja (human gold, CC-BY-4.0) into unified parquet.

~20k human-translated pairs across pre-split train/dev/test. Columns:

    en : str — English sentence
    ja : str — Japanese sentence

Mapping: ja -> jp, en -> en. Splits exposed by HF are ``train``,
``validation``, ``test``; we ingest all three by default (configurable via
``--splits``). Each row's source id records its split:

    src = "alt:{split}:row{i}"

Content is general-domain human translation. Tags: register_tag='general'...
NOTE: the unified schema does not define a 'general' tag, so we map to 'anchor'
(the schema's general-purpose clean-anchor tag), which is the closest valid tag.

Tags: register_tag='anchor', gold_flag=True (human gold).
"""

from __future__ import annotations

import argparse
from typing import Iterator

from _cli_common import configure_logging, logger
from unify_schema import make_row, write_parquet


HF_REPO = "hpprc/alt-parallel-en-ja"
# HF split names; "validation" is the dev split.
DEFAULT_SPLITS = ("train", "validation", "test")
# Unified schema lacks a 'general' tag; 'anchor' is the closest general/clean tag.
REGISTER_TAG = "anchor"


def iter_rows(
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    limit: int | None = None,
) -> Iterator[dict[str, object]]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error(
            "alt: `datasets` not installed; add it to pyproject. "
            f"TODO: pip install datasets. Original error: {e}"
        )
        return

    seen: set[tuple[str, str]] = set()
    emitted = 0
    skipped = 0
    logged_cols = False
    for split in splits:
        try:
            ds = load_dataset(HF_REPO, split=split, streaming=True)
        except Exception as e:
            logger.error(
                f"alt: could not load {HF_REPO}:{split} "
                f"({type(e).__name__}: {e}). TODO: gated/network issue; skipping split."
            )
            continue
        for i, rec in enumerate(ds):
            if not logged_cols:
                logger.info(f"alt: detected columns={list(rec.keys())}")
                logged_cols = True
            jp = (rec.get("ja") or "").strip()
            en = (rec.get("en") or "").strip()
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
                src=f"alt:{split}:row{i}",
                register_tag=REGISTER_TAG,
                gold_flag=True,
            )
            if limit is not None and emitted >= limit:
                logger.info(f"alt: hit limit={limit} emitted={emitted}")
                return
    logger.info(f"alt: emitted={emitted} skipped_empty={skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="backend/training/datasets/unified/alt_parallel.parquet",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="HF splits to ingest (train validation test).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap emitted rows.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    splits = tuple(args.splits)
    if args.dry_run:
        rows = list(
            iter_rows(splits=splits, limit=args.limit if args.limit else 50)
        )
        print(f"alt_parallel rows: {len(rows)}")
        for r in rows[:3]:
            print(f"  jp={r['jp']!r}\n  en={r['en']!r}\n  src={r['src']}")
        return
    n = write_parquet(iter_rows(splits=splits, limit=args.limit), args.out)
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()

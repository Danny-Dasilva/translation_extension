"""Load JESC (Japanese-English Subtitle Corpus) TSV splits into unified parquet.

Input:  tab-separated files under slang-colloquial/JESC-data/split/{train,dev,test},
        each line is ``<english>\\t<japanese>``.
Output: one parquet per split (default to ``--out`` single-file if splits merged,
        or ``<out-dir>/{train,dev,test}.parquet`` if ``--per-split``).

Tags:   register_tag='anime_sub', gold_flag=False, src='jesc'.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

from _cli_common import configure_logging, logger
from unify_schema import make_row, write_parquet


DEFAULT_INPUT_DIR = (
    "backend/training/datasets/translation/slang-colloquial/JESC-data/split"
)


def _iter_tsv(path: Path) -> Iterator[dict[str, object]]:
    """Yield schema rows from a single tab-separated JESC file.

    JESC lines look like: ``english_text\\tjapanese_text`` (1 tab separator).
    We emit each as (jp, en).
    """
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            en = parts[0].strip()
            jp = parts[1].strip()
            if not jp or not en:
                continue
            yield make_row(
                jp=jp,
                en=en,
                src=f"jesc:{path.name}:{lineno}",
                register_tag="anime_sub",
                gold_flag=False,
            )


def load(input_dir: Path, splits: list[str]) -> Iterator[dict[str, object]]:
    for split in splits:
        p = input_dir / split
        if not p.exists():
            logger.warning(f"JESC split missing: {p}")
            continue
        logger.info(f"streaming {p}")
        yield from _iter_tsv(p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing train/dev/test files.",
    )
    parser.add_argument(
        "--out",
        default="backend/training/datasets/unified/jesc.parquet",
        help="Output parquet path (all splits merged).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        help="Which splits to load.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count rows without writing parquet.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"input-dir does not exist: {input_dir}")
        raise SystemExit(2)

    rows = load(input_dir, args.splits)
    if args.dry_run:
        n = sum(1 for _ in rows)
        logger.info(f"[dry-run] JESC rows: {n:,}")
        print(f"JESC rows: {n}")
        return

    n = write_parquet(rows, args.out)
    logger.info(f"wrote {n:,} rows to {args.out}")
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()

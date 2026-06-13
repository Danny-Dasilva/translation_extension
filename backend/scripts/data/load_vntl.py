"""Load lmg-anon VNTL datasets (v3.1-1k + raw pool) into unified parquet.

Both datasets store multi-turn visual-novel scenes in a single ``text`` column,
using ``<<JAPANESE>>``/``<<ENGLISH>>`` markers. ``parse_vntl_packed_text`` in
``_cli_common`` does the split.

Outputs:
  - ``vntl_v31_1k_train.parquet``: register_tag='vn_eroge', gold_flag=True
  - ``vntl_raw.parquet``:          register_tag='vn',       gold_flag=False
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import polars as pl

import re

from _cli_common import configure_logging, logger, parse_vntl_packed_text
from unify_schema import make_row, write_parquet


# Raw VNTL pool uses "Japanese: ...\nEnglish: ...\n---" blocks (NOT the <<...>>
# markers that VNTL-v3.1-1k uses). Same format as NilanE SmallParallelDocs.
_RAW_BLOCK_RE = re.compile(
    r"Japanese:\s*(?P<jp>.*?)\s*\n+English:\s*(?P<en>.*?)(?=\n---|\Z)",
    re.DOTALL,
)


V31_TRAIN = (
    "backend/training/datasets/translation/vn-ln-manga/"
    "lmg-anon__VNTL-v3.1-1k/data/train-00000-of-00001-eb879b20cbd4854b.parquet"
)
V31_VAL = (
    "backend/training/datasets/translation/vn-ln-manga/"
    "lmg-anon__VNTL-v3.1-1k/data/val-00000-of-00001-51ab569c62b2bbc8.parquet"
)
RAW_TRAIN = (
    "backend/training/datasets/translation/vn-ln-manga/"
    "lmg-anon__VNTL/data/train-00000-of-00001-5646cfb373a5b142.parquet"
)


def _iter_raw_blocks(blob: str) -> Iterator[tuple[str, str]]:
    """Yield (jp, en) from the raw VNTL pool's Japanese:/English:/--- blocks."""
    for m in _RAW_BLOCK_RE.finditer(blob):
        jp = m.group("jp").strip()
        en = m.group("en").strip()
        if jp and en:
            yield jp, en


def iter_pairs_from_parquet(
    path: Path,
    *,
    src_label: str,
    register_tag: str,
    gold_flag: bool,
) -> Iterator[dict[str, object]]:
    df = pl.read_parquet(path)
    if "text" not in df.columns:
        logger.error(f"{path}: missing 'text' column; columns={df.columns}")
        return
    for row_idx, record in enumerate(df.iter_rows(named=True)):
        blob = record.get("text") or ""
        # Auto-detect format. v3.1-1k uses <<JAPANESE>>/<<ENGLISH>> markers;
        # the raw pool uses "Japanese:\n...\nEnglish:\n...\n---" blocks.
        if "<<JAPANESE>>" in blob:
            pairs = parse_vntl_packed_text(blob)
        else:
            pairs = _iter_raw_blocks(blob)
        for turn_idx, (jp, en) in enumerate(pairs):
            yield make_row(
                jp=jp,
                en=en,
                src=f"{src_label}:row{row_idx}:turn{turn_idx}",
                register_tag=register_tag,
                gold_flag=gold_flag,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v31-train", default=V31_TRAIN)
    parser.add_argument("--raw-train", default=RAW_TRAIN)
    parser.add_argument(
        "--out-v31",
        default="backend/training/datasets/unified/vntl_v31_1k_train.parquet",
    )
    parser.add_argument(
        "--out-raw",
        default="backend/training/datasets/unified/vntl_raw.parquet",
    )
    parser.add_argument(
        "--skip-raw", action="store_true", help="Skip the raw-pool parquet."
    )
    parser.add_argument(
        "--skip-v31", action="store_true", help="Skip the v3.1-1k train parquet."
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    if not args.skip_v31:
        p = Path(args.v31_train)
        if not p.exists():
            logger.error(f"VNTL v3.1 train missing: {p}")
        else:
            rows = iter_pairs_from_parquet(
                p, src_label="vntl_v31_1k", register_tag="vn_eroge", gold_flag=True
            )
            if args.dry_run:
                print(f"vntl_v31_1k pairs: {sum(1 for _ in rows)}")
            else:
                n = write_parquet(rows, args.out_v31)
                print(f"wrote {n} rows to {args.out_v31}")

    if not args.skip_raw:
        p = Path(args.raw_train)
        if not p.exists():
            logger.error(f"VNTL raw missing: {p}")
            return
        rows = iter_pairs_from_parquet(
            p, src_label="vntl_raw", register_tag="vn", gold_flag=False
        )
        if args.dry_run:
            print(f"vntl_raw pairs: {sum(1 for _ in rows)}")
        else:
            n = write_parquet(rows, args.out_raw)
            print(f"wrote {n} rows to {args.out_raw}")


if __name__ == "__main__":
    main()

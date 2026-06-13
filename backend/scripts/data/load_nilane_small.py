"""Load NilanE/SmallParallelDocs-Ja_En-6k into unified parquet.

Input JSONL format: one record per doc chunk with ``{"src": <japanese_blob>,
"trg": <english_blob>, "meta": ...}``. Each blob is a multi-line document.
We line-align: if ``len(src.splitlines()) == len(trg.splitlines())`` we zip,
otherwise drop the record with a warning counter.

Tags: register_tag='novel', gold_flag=False, src='nilane_small:<rec>:<pair_idx>'.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from _cli_common import configure_logging, logger
from unify_schema import make_row, write_parquet


DEFAULT_INPUT = (
    "backend/training/datasets/translation/vn-ln-manga/"
    "NilanE__SmallParallelDocs-Ja_En-6k/dataset-parallel-complete.jsonl"
)


def iter_rows(path: Path) -> Iterator[dict[str, object]]:
    emitted = 0
    records = 0
    misaligned = 0
    with path.open("r", encoding="utf-8") as fh:
        for rec_idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            records += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"nilane_small: bad json line {rec_idx}")
                continue
            jp_blob = d.get("src") or ""
            en_blob = d.get("trg") or ""
            jp_lines = [x.strip() for x in jp_blob.splitlines() if x.strip()]
            en_lines = [x.strip() for x in en_blob.splitlines() if x.strip()]
            if len(jp_lines) != len(en_lines):
                misaligned += 1
                continue
            for pair_idx, (jp, en) in enumerate(zip(jp_lines, en_lines)):
                emitted += 1
                yield make_row(
                    jp=jp,
                    en=en,
                    src=f"nilane_small:rec{rec_idx}:pair{pair_idx}",
                    register_tag="novel",
                    gold_flag=False,
                )
    logger.info(
        f"nilane_small: records={records} misaligned_dropped={misaligned} "
        f"pairs_emitted={emitted}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--out",
        default="backend/training/datasets/unified/nilane_small.parquet",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    p = Path(args.input)
    if not p.exists():
        logger.error(f"input missing: {p}")
        raise SystemExit(2)
    rows = iter_rows(p)
    if args.dry_run:
        print(f"nilane_small rows: {sum(1 for _ in rows)}")
        return
    n = write_parquet(rows, args.out)
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()

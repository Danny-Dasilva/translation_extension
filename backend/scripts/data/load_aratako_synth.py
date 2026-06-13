"""Load Aratako Synthetic-JP-EN-Translation-Dataset into unified parquet.

Input JSONL format: chat-style records
    {"id": ..., "messages": [
        {"role": "system", ...},
        {"role": "user",      "content": "<english source>"},
        {"role": "assistant", "content": "<japanese target>"},
    ], ...}

The task is EN->JA synthesis, so user=EN, assistant=JP. We emit (jp, en) flipped.
Tags: register_tag='synthetic', gold_flag=False, src='aratako_synth:<id>'.
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
    "Aratako__Synthetic-JP-EN-Translation-Dataset-Magpie-Nemotron-4-20k/"
    "Synthetic-JP-EN-Translation-Dataset-Magpie-Nemotron-4-20k.jsonl"
)


def _extract_pair(rec: dict) -> tuple[str, str] | None:
    msgs = rec.get("messages") or []
    user_content: str | None = None
    asst_content: str | None = None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if role == "user" and user_content is None:
            user_content = content.strip()
        elif role == "assistant" and asst_content is None:
            asst_content = content.strip()
    if not user_content or not asst_content:
        return None
    # user=EN, assistant=JP
    return asst_content, user_content


def iter_rows(path: Path) -> Iterator[dict[str, object]]:
    records = 0
    emitted = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("aratako: bad json")
                continue
            pair = _extract_pair(rec)
            if pair is None:
                continue
            jp, en = pair
            rid = rec.get("id")
            emitted += 1
            yield make_row(
                jp=jp,
                en=en,
                src=f"aratako_synth:{rid}",
                register_tag="synthetic",
                gold_flag=False,
            )
    logger.info(f"aratako: records={records} emitted={emitted}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--out",
        default="backend/training/datasets/unified/aratako_synth.parquet",
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
        print(f"aratako rows: {sum(1 for _ in rows)}")
        return
    n = write_parquet(rows, args.out)
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()

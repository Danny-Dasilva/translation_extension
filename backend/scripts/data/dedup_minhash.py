"""MinHash-LSH near-duplicate detection.

Two modes:
  --within           dedup a single parquet in place (drop near-dupes)
  --cross --held-out <path>
                     remove rows in --input that near-match any row in --held-out
                     (held-out may be JSONL or parquet; auto-detected by suffix)

Config (plan §4):
  - shingle: char 5-gram on concat(jp + "||" + en)
  - num_perm: 14 * 8 = 112 (matches "14 bands x 8 rows" LSH config)
  - Jaccard threshold: 0.75

Requires ``datasketch``. Install via ``uv add --project backend datasketch``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import polars as pl

from _cli_common import configure_logging, logger

NUM_PERM = 112  # 14 bands x 8 rows/band in LSH
BANDS = 14
ROWS_PER_BAND = 8
JACCARD_THRESHOLD = 0.75
SHINGLE_N = 5


def _shingles(text: str, n: int = SHINGLE_N) -> set[bytes]:
    if len(text) < n:
        return {text.encode("utf-8")}
    return {text[i : i + n].encode("utf-8") for i in range(len(text) - n + 1)}


def _mk_minhash(text: str, MinHash):
    mh = MinHash(num_perm=NUM_PERM)
    for sh in _shingles(text):
        mh.update(sh)
    return mh


def _concat(jp: str, en: str) -> str:
    return f"{jp}||{en}"


def _read_any(path: Path) -> pl.DataFrame:
    if path.suffix == ".jsonl":
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rows.append({"jp": d.get("jp", ""), "en": d.get("en", "")})
        return pl.DataFrame(rows) if rows else pl.DataFrame({"jp": [], "en": []})
    return pl.read_parquet(path)


def _iter_texts(df: pl.DataFrame) -> Iterable[str]:
    for row in df.iter_rows(named=True):
        yield _concat(row.get("jp") or "", row.get("en") or "")


def dedup_within(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error("datasketch not installed. `uv add --project backend datasketch`")
        raise SystemExit(2) from e

    lsh = MinHashLSH(
        threshold=JACCARD_THRESHOLD,
        num_perm=NUM_PERM,
        params=(BANDS, ROWS_PER_BAND),
    )
    keep_mask: list[bool] = []
    removed = 0
    for i, text in enumerate(_iter_texts(df)):
        mh = _mk_minhash(text, MinHash)
        if lsh.query(mh):
            removed += 1
            keep_mask.append(False)
            continue
        lsh.insert(f"r{i}", mh)
        keep_mask.append(True)
    return df.filter(pl.Series(keep_mask)), removed


def dedup_cross(df: pl.DataFrame, held_out: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error("datasketch not installed. `uv add --project backend datasketch`")
        raise SystemExit(2) from e

    lsh = MinHashLSH(
        threshold=JACCARD_THRESHOLD,
        num_perm=NUM_PERM,
        params=(BANDS, ROWS_PER_BAND),
    )
    logger.info(f"indexing {len(held_out):,} held-out rows")
    for i, text in enumerate(_iter_texts(held_out)):
        mh = _mk_minhash(text, MinHash)
        lsh.insert(f"ho{i}", mh)

    keep_mask: list[bool] = []
    removed = 0
    logger.info(f"probing {len(df):,} train rows for overlap with held-out")
    for text in _iter_texts(df):
        mh = _mk_minhash(text, MinHash)
        if lsh.query(mh):
            removed += 1
            keep_mask.append(False)
        else:
            keep_mask.append(True)
    return df.filter(pl.Series(keep_mask)), removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats-out", default=None)
    parser.add_argument("--within", action="store_true", help="Dedup within --input.")
    parser.add_argument(
        "--cross",
        action="store_true",
        help="Remove --input rows that near-match any row in --held-out.",
    )
    parser.add_argument(
        "--held-out",
        default=None,
        help="Held-out parquet or jsonl (used with --cross).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    if not (args.within or args.cross):
        print("Must pass either --within or --cross", file=sys.stderr)
        raise SystemExit(2)
    if args.cross and not args.held_out:
        print("--cross requires --held-out PATH", file=sys.stderr)
        raise SystemExit(2)

    df = pl.read_parquet(args.input)
    logger.info(f"loaded {len(df):,} rows from {args.input}")

    if args.dry_run:
        print(f"would dedup {len(df)} rows; mode={'within' if args.within else 'cross'}")
        return

    if args.within:
        out_df, removed = dedup_within(df)
        mode = "within"
    else:
        held = _read_any(Path(args.held_out))
        out_df, removed = dedup_cross(df, held)
        mode = f"cross:{args.held_out}"

    stats = {
        "mode": mode,
        "input_rows": len(df),
        "output_rows": len(out_df),
        "removed": removed,
        "jaccard_threshold": JACCARD_THRESHOLD,
        "num_perm": NUM_PERM,
        "bands": BANDS,
        "rows_per_band": ROWS_PER_BAND,
        "shingle_n": SHINGLE_N,
    }
    logger.info(f"dedup_minhash stats: {stats}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path)
    stats_path = Path(args.stats_out) if args.stats_out else out_path.with_suffix(
        ".stats.json"
    )
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"wrote {len(out_df)} rows to {out_path} (removed {removed})")


if __name__ == "__main__":
    main()

"""Compose v10 SFT training parquet from v7.1 mix + Manga109 dedup'd corpus.

v10 plan: oversample new register data (manga dialog from Manga109) at 1.5x.

Schema MUST match v7.1: [jp, en, src, register_tag, gold_flag]
- v7.1 mix is read AS-IS from data_v71.parquet (already in canonical schema).
- Manga109 (bubbles_translated_qe_deduped.parquet) is mapped:
    jp_text       -> jp
    en_text       -> en
    book/page/id  -> src = "manga109:{book}:p{page}:{text_id}"
    register_tag  -> "manga_dialog"
    gold_flag     -> True   (Gemma 4 31B-it teacher + COMET/LaBSE QE filtered)

Writes:
    backend/scripts/data/manga109/data_v10.parquet
    backend/scripts/data/manga109/data_v10.mix-summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli_common import configure_logging, logger  # noqa: E402


DEFAULT_V71 = Path(
    "/home/danny/Documents/personal/extension/backend/training/runs/manga-bubbles/data_v71.parquet"
)
DEFAULT_MANGA109 = Path(
    "/home/danny/Documents/personal/extension/backend/scripts/data/manga109/bubbles_translated_qe_deduped.parquet"
)
DEFAULT_OUTPUT = Path(
    "/home/danny/Documents/personal/extension/backend/scripts/data/manga109/data_v10.parquet"
)


def map_manga109_to_v71_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Project Manga109 (jp_text, en_text, ...) -> (jp, en, src, register_tag, gold_flag)."""
    return df.with_columns([
        pl.col("jp_text").alias("jp"),
        pl.col("en_text").alias("en"),
        (
            pl.lit("manga109:")
            + pl.col("book").cast(pl.Utf8)
            + pl.lit(":p")
            + pl.col("page").cast(pl.Utf8)
            + pl.lit(":")
            + pl.col("text_id").cast(pl.Utf8)
        ).alias("src"),
        pl.lit("manga_dialog").alias("register_tag"),
        pl.lit(True).alias("gold_flag"),
    ]).select(["jp", "en", "src", "register_tag", "gold_flag"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v71", type=Path, default=DEFAULT_V71)
    p.add_argument("--manga109", type=Path, default=DEFAULT_MANGA109)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--manga109-weight", type=float, default=1.5,
        help="oversample multiplier for Manga109 (default 1.5x per v10 plan)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    if not args.v71.exists():
        logger.error(f"v7.1 mix not found: {args.v71}")
        return 2
    if not args.manga109.exists():
        logger.error(f"Manga109 dedup'd parquet not found: {args.manga109}")
        return 2

    v71 = pl.read_parquet(args.v71)
    logger.info(f"v7.1 mix: {len(v71):,} rows; cols={v71.columns}")

    m109_raw = pl.read_parquet(args.manga109)
    logger.info(f"manga109 dedup: {len(m109_raw):,} rows")

    # Drop rows where either jp_text or en_text is empty / null after strip.
    pre = len(m109_raw)
    m109_raw = m109_raw.filter(
        (pl.col("jp_text").is_not_null())
        & (pl.col("en_text").is_not_null())
        & (pl.col("jp_text").str.strip_chars().str.len_chars() > 0)
        & (pl.col("en_text").str.strip_chars().str.len_chars() > 0)
    )
    logger.info(f"manga109 after non-empty filter: {len(m109_raw):,} (dropped {pre - len(m109_raw):,})")

    m109 = map_manga109_to_v71_schema(m109_raw)
    n_unique = len(m109)

    # Apply oversampling weight: e.g. 1.5x means take ceil(1.5 * n_unique) rows
    # via sample-with-replacement.
    weight = args.manga109_weight
    target_m109 = int(round(weight * n_unique))
    if weight <= 1.0:
        m109_sampled = m109.sample(n=target_m109, with_replacement=False, seed=args.seed)
    else:
        m109_sampled = m109.sample(n=target_m109, with_replacement=True, seed=args.seed)
    logger.info(f"manga109 oversampled {n_unique:,} -> {len(m109_sampled):,} (x{weight})")

    # Concat (column order is identical) + shuffle.
    combined = pl.concat([v71, m109_sampled], how="vertical_relaxed")
    combined = combined.sample(fraction=1.0, shuffle=True, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(args.output)
    logger.info(f"wrote {len(combined):,} rows to {args.output}")

    # Summary.
    summary = {
        "v71_rows": len(v71),
        "manga109_unique_rows": n_unique,
        "manga109_weight": weight,
        "manga109_sampled_rows": len(m109_sampled),
        "final_rows": len(combined),
        "effective_epoch_length": len(combined),
        "seed": args.seed,
        "schema": combined.columns,
        "v71_path": str(args.v71),
        "manga109_path": str(args.manga109),
    }
    summary_path = args.output.with_suffix(".mix-summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"summary -> {summary_path}")

    # Print 5 random rows.
    sample = combined.sample(n=min(5, len(combined)), seed=0)
    print("\n=== 5 random samples ===", file=sys.stderr)
    for r in sample.iter_rows(named=True):
        print(f"  src={r['src']}", file=sys.stderr)
        print(f"  register={r['register_tag']}  gold={r['gold_flag']}", file=sys.stderr)
        print(f"  JP: {r['jp']}", file=sys.stderr)
        print(f"  EN: {r['en']}", file=sys.stderr)
        print("", file=sys.stderr)

    print(f"\nv10 row count: {len(combined):,}")
    print(f"  v7.1 contribution:        {len(v71):,}")
    print(f"  manga109 unique rows:     {n_unique:,}")
    print(f"  manga109 with x{weight} oversample: {len(m109_sampled):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

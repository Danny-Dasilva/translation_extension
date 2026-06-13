"""Score JP/EN pairs with LaBSE cosine similarity and optionally drop.

Uses ``sentence-transformers/LaBSE``. Adds a ``labse_cos`` float column.
If ``--drop-below`` is set, drops rows with ``labse_cos < --threshold``
(plan default 0.70).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from _cli_common import configure_logging, logger


MODEL_ID = "sentence-transformers/LaBSE"


def _load_labse() -> Any:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error(
            "sentence-transformers not installed. Install with: "
            "`uv add --project backend sentence-transformers`"
        )
        raise SystemExit(2) from e
    logger.info(f"loading {MODEL_ID} (downloads on first run)")
    return SentenceTransformer(MODEL_ID)


def score(df: pl.DataFrame, *, batch_size: int) -> pl.DataFrame:
    import numpy as np

    model = _load_labse()
    jp_list = df["jp"].to_list()
    en_list = df["en"].to_list()

    logger.info(f"encoding {len(jp_list):,} JP sentences")
    jp_emb = model.encode(jp_list, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    logger.info(f"encoding {len(en_list):,} EN sentences")
    en_emb = model.encode(en_list, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)

    # Both normalized, so cosine = dot product row-wise.
    cos = (jp_emb * en_emb).sum(axis=1).astype(np.float32)
    return df.with_columns(pl.Series("labse_cos", cos.tolist(), dtype=pl.Float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats-out", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--drop-below", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    df = pl.read_parquet(args.input)
    logger.info(f"loaded {len(df):,} rows from {args.input}")
    if args.dry_run:
        print(f"would score {len(df)} with LaBSE; threshold={args.threshold} drop={args.drop_below}")
        return

    scored = score(df, batch_size=args.batch_size)
    n_before = len(scored)
    if args.drop_below:
        scored = scored.filter(pl.col("labse_cos") >= args.threshold)
    n_after = len(scored)

    def _num(x) -> float | None:  # noqa: ANN001
        if x is None:
            return None
        try:
            return float(x)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    stats = {
        "model": MODEL_ID,
        "input_rows": n_before,
        "output_rows": n_after,
        "threshold": args.threshold,
        "drop_below": args.drop_below,
        "mean_labse_cos": _num(scored["labse_cos"].mean()) if n_after else None,
        "p25": _num(scored["labse_cos"].quantile(0.25)) if n_after else None,
        "p50": _num(scored["labse_cos"].quantile(0.50)) if n_after else None,
        "p75": _num(scored["labse_cos"].quantile(0.75)) if n_after else None,
    }
    logger.info(f"labse stats: {stats}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(out_path)
    stats_path = Path(args.stats_out) if args.stats_out else out_path.with_suffix(
        ".stats.json"
    )
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"wrote {len(scored)} rows to {out_path} (stats: {stats_path})")


if __name__ == "__main__":
    main()

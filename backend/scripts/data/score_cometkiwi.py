"""Score JP→EN pairs with CometKiwi (reference-free QE) and emit parquet.

Tries ``Unbabel/wmt23-cometkiwi-da`` first, falls back to
``Unbabel/wmt22-cometkiwi-da`` with a warning (plan spec).

Adds a ``cometkiwi`` float column. If ``--drop-below`` is passed, rows with
``cometkiwi < --threshold`` are removed (plan default 0.78 for non-gold).

Models are snapshot-downloaded lazily on first run. Requires ``unbabel-comet``
(``pip install unbabel-comet``) at runtime — if not installed, the script
errors out with install instructions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from _cli_common import configure_logging, logger


PRIMARY_MODEL = "Unbabel/wmt23-cometkiwi-da-xl"
# wmt23 has xl / xxl variants; the plan says "try -23 fall back to -22".
FALLBACK_MODEL = "Unbabel/wmt22-cometkiwi-da"


def _load_comet_model() -> tuple[Any, str]:
    try:
        from comet import download_model, load_from_checkpoint  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error(
            "unbabel-comet not installed. Install with: "
            "`uv add --project backend unbabel-comet`"
        )
        raise SystemExit(2) from e

    for model_id in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            logger.info(f"downloading CometKiwi model: {model_id}")
            ckpt_path = download_model(model_id)
            model = load_from_checkpoint(ckpt_path)
            logger.info(f"loaded {model_id}")
            return model, model_id
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{model_id} unavailable: {e}")
    raise SystemExit("No CometKiwi model could be loaded.")


def score(df: pl.DataFrame, *, batch_size: int, gpus: int) -> tuple[pl.DataFrame, str]:
    model, model_id = _load_comet_model()
    data = [
        {"src": row["jp"], "mt": row["en"]}
        for row in df.iter_rows(named=True)
    ]
    logger.info(f"scoring {len(data):,} pairs with {model_id}")
    preds = model.predict(
        data,
        batch_size=batch_size,
        gpus=gpus,
        progress_bar=True,
    )
    scores = list(preds.scores)
    if len(scores) != len(df):
        raise RuntimeError(
            f"CometKiwi returned {len(scores)} scores for {len(df)} rows"
        )
    out = df.with_columns(pl.Series("cometkiwi", scores, dtype=pl.Float32))
    return out, model_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats-out", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.78)
    parser.add_argument(
        "--drop-below",
        action="store_true",
        help="If set, drop rows with cometkiwi < --threshold; else keep all and just annotate.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    df = pl.read_parquet(args.input)
    logger.info(f"loaded {len(df):,} rows from {args.input}")

    if args.dry_run:
        print(f"would score {len(df)} rows with CometKiwi; threshold={args.threshold} drop={args.drop_below}")
        return

    scored, model_id = score(df, batch_size=args.batch_size, gpus=args.gpus)

    n_before = len(scored)
    if args.drop_below:
        scored = scored.filter(pl.col("cometkiwi") >= args.threshold)
    n_after = len(scored)

    def _num(x) -> float | None:  # noqa: ANN001
        if x is None:
            return None
        try:
            return float(x)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    stats = {
        "model": model_id,
        "input_rows": n_before,
        "output_rows": n_after,
        "threshold": args.threshold,
        "drop_below": args.drop_below,
        "mean_cometkiwi": _num(scored["cometkiwi"].mean()) if n_after else None,
        "p25": _num(scored["cometkiwi"].quantile(0.25)) if n_after else None,
        "p50": _num(scored["cometkiwi"].quantile(0.50)) if n_after else None,
        "p75": _num(scored["cometkiwi"].quantile(0.75)) if n_after else None,
    }
    logger.info(f"cometkiwi stats: {stats}")

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

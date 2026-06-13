"""Filter Manga109 (jp, en) translations by reference-free QE.

Two scorers are supported:

  --scorer cometkiwi  (default, REQUIRED for the spec)
      Uses ``Unbabel/wmt23-cometkiwi-da-xl`` (~14GB bf16).  This is a
      gated HuggingFace repo: you must (a) accept the Unbabel terms on
      the model card and (b) ``huggingface-cli login`` before running.
      Falls back to ``-23-da`` then ``-22-da`` if the XL is unavailable.

  --scorer labse
      Pure cosine similarity between LaBSE JP and EN embeddings.  Public
      model, no HF auth needed.  Useful when COMET-Kiwi is gated and the
      user hasn't authenticated yet.  Lower-quality QE -- use only as a
      fallback for the single-book proof.

Adds a ``kiwi_score`` column to the output (renamed from ``labse_cos``
when --scorer labse so downstream code is uniform).

Run via the dedicated COMET sidecar venv:
    /home/danny/.venvs/comet/bin/python filter_manga109_qe.py ...
The main backend env can NOT have unbabel-comet installed because the
package has a hard ``numpy<2`` pin (see pyproject.toml comment).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli_common import configure_logging, logger  # noqa: E402


# Model preference order: -23-XL (best, ~14GB) -> -23 -> -22 fallback.
MODEL_PREFERENCE = (
    "Unbabel/wmt23-cometkiwi-da-xl",
    "Unbabel/wmt23-cometkiwi-da",
    "Unbabel/wmt22-cometkiwi-da",
)

DEFAULT_INPUT = Path(
    "/home/danny/Documents/personal/extension/backend/scripts/data/manga109/bubbles_translated.parquet"
)
DEFAULT_OUTPUT = Path(
    "/home/danny/Documents/personal/extension/backend/scripts/data/manga109/bubbles_translated_qe.parquet"
)


def _load_comet_model() -> tuple[Any, str]:
    try:
        from comet import download_model, load_from_checkpoint  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error(
            "unbabel-comet not installed. Install with:\n"
            "    uv add --project backend unbabel-comet\n"
            "or in a separate venv (the package has a numpy<2 pin)."
        )
        raise SystemExit(2) from e

    last_err: Exception | None = None
    for model_id in MODEL_PREFERENCE:
        try:
            logger.info(f"downloading CometKiwi: {model_id}")
            ckpt = download_model(model_id)
            model = load_from_checkpoint(ckpt)
            logger.info(f"loaded {model_id}")
            return model, model_id
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{model_id} unavailable: {e}")
            last_err = e
    raise SystemExit(f"No CometKiwi model could be loaded: {last_err}")


def _score_comet(model: Any, df: pl.DataFrame, *, batch_size: int, gpus: int) -> list[float]:
    data = [
        {"src": (row["jp_text"] or "").strip(), "mt": (row["en_text"] or "").strip()}
        for row in df.iter_rows(named=True)
    ]
    logger.info(f"scoring {len(data):,} pairs with CometKiwi")
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
    return [float(s) for s in scores]


def _score_labse(df: pl.DataFrame, *, batch_size: int) -> tuple[list[float], str]:
    """Cosine similarity in LaBSE space. Public, no auth required."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error("sentence-transformers required; "
                     "`uv pip install sentence-transformers` in the active venv")
        raise SystemExit(2) from e
    import numpy as np

    model_id = "sentence-transformers/LaBSE"
    logger.info(f"loading {model_id}")
    model = SentenceTransformer(model_id)
    jp = [(r["jp_text"] or "").strip() for r in df.iter_rows(named=True)]
    en = [(r["en_text"] or "").strip() for r in df.iter_rows(named=True)]
    logger.info(f"encoding {len(jp):,} JP + {len(en):,} EN")
    e_jp = model.encode(jp, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=True)
    e_en = model.encode(en, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=True)
    cos = (e_jp * e_en).sum(axis=1)
    return [float(s) for s in np.clip(cos, 0.0, 1.0)], model_id


def _histogram(scores: list[float]) -> dict[str, int]:
    edges = [0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
    out: dict[str, int] = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        label = f"[{lo:.2f},{hi:.2f})"
        out[label] = sum(1 for s in scores if lo <= s < hi)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--scorer", choices=["cometkiwi", "labse"], default="cometkiwi",
                   help="cometkiwi (gated, requires HF auth) or labse (public fallback)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument(
        "--drop-pct", type=float, default=30.0,
        help="drop the bottom N pct of rows by score (0=annotate only)"
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="if set, drop rows with kiwi_score < threshold (overrides --drop-pct)"
    )
    p.add_argument("--n-rejected-samples", type=int, default=10)
    return p.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    if not args.input.exists():
        logger.error(f"input parquet not found: {args.input}")
        return 2
    df = pl.read_parquet(args.input)
    logger.info(f"loaded {len(df):,} rows from {args.input}")

    if args.scorer == "cometkiwi":
        model, model_id = _load_comet_model()
        scores = _score_comet(model, df, batch_size=args.batch_size, gpus=args.gpus)
    else:
        scores, model_id = _score_labse(df, batch_size=args.batch_size)
    df = df.with_columns(pl.Series("kiwi_score", scores, dtype=pl.Float32))
    df = df.with_columns(pl.lit(model_id).alias("kiwi_model"))

    # Stats.
    s_sorted = sorted(scores)
    n = len(s_sorted)
    p_lo = s_sorted[max(0, int(n * 0.05))]
    p_med = s_sorted[n // 2]
    p_hi = s_sorted[min(n - 1, int(n * 0.95))]
    logger.info(
        f"kiwi_score stats: min={s_sorted[0]:.3f} p05={p_lo:.3f} "
        f"median={p_med:.3f} p95={p_hi:.3f} max={s_sorted[-1]:.3f} mean={sum(scores)/n:.3f}"
    )
    logger.info(f"histogram: {_histogram(scores)}")

    # Filter.
    if args.threshold is not None:
        threshold = args.threshold
        keep_mask = pl.col("kiwi_score") >= threshold
    else:
        # Drop the bottom drop_pct% by score.
        cutoff_idx = max(0, int(n * args.drop_pct / 100.0))
        threshold = s_sorted[cutoff_idx] if cutoff_idx < n else s_sorted[-1]
        keep_mask = pl.col("kiwi_score") >= threshold
    logger.info(
        f"filter threshold: {threshold:.4f} "
        f"({'fixed' if args.threshold is not None else f'{args.drop_pct}th percentile'})"
    )

    rejected = df.filter(~keep_mask)
    kept = df.filter(keep_mask)
    logger.info(f"kept: {len(kept):,} / {n:,} ({len(kept)/n:.1%})")
    logger.info(f"rejected: {len(rejected):,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept.write_parquet(args.output)
    logger.info(f"wrote {len(kept):,} rows to {args.output}")
    rejected_path = args.output.with_name(args.output.stem + "_rejected.parquet")
    rejected.write_parquet(rejected_path)
    logger.info(f"wrote rejected: {len(rejected):,} rows to {rejected_path}")

    # Sample low-scoring rejected.
    if len(rejected) > 0:
        sample_n = min(args.n_rejected_samples, len(rejected))
        sample = rejected.sort("kiwi_score").head(sample_n)
        print("\n=== lowest-scoring rejected pairs (manual sanity) ===", file=sys.stderr)
        for r in sample.iter_rows(named=True):
            print(
                f"  [{r['book']} p{r['page']}] kiwi={r['kiwi_score']:.3f}",
                file=sys.stderr,
            )
            print(f"    JP: {r['jp_text']}", file=sys.stderr)
            print(f"    EN: {r['en_text']}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""VNTL leaderboard cosine-accuracy metric.

Uses ``sentence-transformers/all-mpnet-base-v2`` embeddings, computes a
per-row cosine between ``en_pred`` and ``en_ref``, and reports the mean,
standard deviation, and a bootstrap 95% CI.

CLI:
    python -m backend.scripts.eval.vntl_cosine \
        --predictions pred.jsonl \
        --out cosine.json

Output JSON: ``{n, mean, std, ci95: [lo, hi]}``.

Note: the model is loaded *lazily* on first use - importing this module does
not download anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
_BOOTSTRAP_ITERS = 1000
_RNG_SEED = 0xC05E


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_encoder(model_name: str = _MODEL_NAME):
    """Lazy import so test collection doesn't download the model."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformers model {}", model_name)
    return SentenceTransformer(model_name)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (N, D) matrices."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(an * bn, axis=1)


def _bootstrap_ci(
    values: np.ndarray, *, iters: int = _BOOTSTRAP_ITERS, seed: int = _RNG_SEED
) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(iters, dtype=np.float64)
    n = values.shape[0]
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        means[i] = float(values[idx].mean())
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return lo, hi


def compute_cosine(
    predictions: list[dict[str, Any]],
    *,
    encoder: Any | None = None,
    model_name: str = _MODEL_NAME,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Compute mean cosine + bootstrap CI over (en_pred, en_ref) pairs.

    ``encoder`` is optional; primarily used by tests to inject a mock
    SentenceTransformer.  If omitted, the model is loaded via
    ``_load_encoder``.
    """
    pairs: list[tuple[str, str]] = []
    for row in predictions:
        en_pred = row.get("en_pred") or row.get("en") or ""
        en_ref = row.get("en_ref") or row.get("ref") or ""
        if not en_pred or not en_ref:
            continue
        pairs.append((en_pred, en_ref))

    if not pairs:
        logger.warning("No (en_pred, en_ref) pairs with both sides present.")
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95": [float("nan"), float("nan")]}

    if encoder is None:
        encoder = _load_encoder(model_name)

    preds_txt = [p[0] for p in pairs]
    refs_txt = [p[1] for p in pairs]

    pred_emb = np.asarray(
        encoder.encode(preds_txt, batch_size=batch_size, show_progress_bar=False)
    )
    ref_emb = np.asarray(
        encoder.encode(refs_txt, batch_size=batch_size, show_progress_bar=False)
    )

    sims = _cosine(pred_emb, ref_emb)
    mean = float(sims.mean())
    std = float(sims.std(ddof=0))
    ci_lo, ci_hi = _bootstrap_ci(sims)

    return {
        "n": int(sims.shape[0]),
        "mean": mean,
        "std": std,
        "ci95": [ci_lo, ci_hi],
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VNTL cosine-accuracy metric (all-mpnet-base-v2).")
    p.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="JSONL with fields {en_pred, en_ref}.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    p.add_argument(
        "--model",
        type=str,
        default=_MODEL_NAME,
        help=f"sentence-transformers model name (default: {_MODEL_NAME}).",
    )
    p.add_argument("--batch-size", type=int, default=32)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rows = _read_jsonl(args.predictions)
    result = compute_cosine(rows, model_name=args.model, batch_size=args.batch_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    logger.info("n={} mean={:.4f} std={:.4f} ci95=[{:.4f}, {:.4f}]",
                result["n"], result["mean"], result["std"],
                result["ci95"][0], result["ci95"][1])
    return 0


if __name__ == "__main__":
    sys.exit(main())

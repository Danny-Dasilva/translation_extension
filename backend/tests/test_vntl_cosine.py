"""Unit tests for backend.scripts.eval.vntl_cosine.

Uses a 3-row fixture with fixed hand-crafted vectors so we verify the math
without loading sentence-transformers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR.parent))

from backend.scripts.eval.vntl_cosine import (  # noqa: E402
    _bootstrap_ci,
    _cosine,
    compute_cosine,
)


class _FakeEncoder:
    """Mock SentenceTransformer. Returns pre-canned vectors keyed by text.

    encode() is called TWICE per compute_cosine() call - first for preds,
    then for refs - so we maintain a strict mapping.
    """

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        return np.asarray([self._table[t] for t in texts], dtype=np.float64)


def test_cosine_helper_simple_identity() -> None:
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = _cosine(a, b)
    assert np.allclose(out, [1.0, 1.0])


def test_cosine_helper_orthogonal() -> None:
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    out = _cosine(a, b)
    assert np.allclose(out, [0.0])


def test_compute_cosine_with_mock_encoder_three_rows() -> None:
    # 3 rows; we set up vectors so cosines are (1.0, 0.5, 0.0) => mean = 0.5.
    # Use angles: 0 deg, 60 deg, 90 deg between pred & ref.
    # Vectors lie on the unit circle.
    table = {
        "p1": [1.0, 0.0],
        "r1": [1.0, 0.0],
        "p2": [1.0, 0.0],
        "r2": [0.5, np.sqrt(3) / 2.0],  # 60 deg -> cos = 0.5
        "p3": [1.0, 0.0],
        "r3": [0.0, 1.0],  # 90 deg -> cos = 0.0
    }
    encoder = _FakeEncoder(table)

    rows = [
        {"en_pred": "p1", "en_ref": "r1"},
        {"en_pred": "p2", "en_ref": "r2"},
        {"en_pred": "p3", "en_ref": "r3"},
    ]

    result = compute_cosine(rows, encoder=encoder)
    assert result["n"] == 3
    assert abs(result["mean"] - 0.5) < 1e-9
    # Std with 3 samples (ddof=0) where vals = [1, 0.5, 0]: mean=0.5, var = (0.25+0+0.25)/3 = 1/6
    expected_std = float(np.sqrt(1.0 / 6.0))
    assert abs(result["std"] - expected_std) < 1e-9
    lo, hi = result["ci95"]
    assert lo <= result["mean"] <= hi


def test_compute_cosine_skips_rows_missing_ref() -> None:
    table = {"p1": [1.0, 0.0], "r1": [1.0, 0.0]}
    encoder = _FakeEncoder(table)
    rows = [
        {"en_pred": "p1", "en_ref": "r1"},
        {"en_pred": "something", "en_ref": ""},  # skipped
        {"en_pred": "", "en_ref": "also"},  # skipped
    ]
    result = compute_cosine(rows, encoder=encoder)
    assert result["n"] == 1
    assert abs(result["mean"] - 1.0) < 1e-9


def test_compute_cosine_empty_rows() -> None:
    result = compute_cosine([], encoder=_FakeEncoder({}))
    assert result["n"] == 0


def test_bootstrap_ci_deterministic_with_seed() -> None:
    values = np.linspace(0.0, 1.0, 100)
    lo1, hi1 = _bootstrap_ci(values, iters=100, seed=42)
    lo2, hi2 = _bootstrap_ci(values, iters=100, seed=42)
    assert lo1 == lo2
    assert hi1 == hi2
    assert lo1 < hi1

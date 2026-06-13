"""Unit tests for ``scripts/data/compose_training_mix.py``.

We construct tiny per-source parquets (3 rows each) and verify:
- target_n respected.
- gold oversampling (with replacement) kicks in for gold sources.
- non-gold sources with shortfall are used as-is (no replacement).
- shuffled output preserves unified schema columns.
"""

from __future__ import annotations

from pathlib import Path

from compose_training_mix import compose
from unify_schema import make_row, write_parquet


def _write_toy(path: Path, n: int, register_tag: str, gold: bool) -> None:
    rows = [
        make_row(
            jp=f"日本語{i}",
            en=f"English{i}",
            src=f"toy:{path.stem}:{i}",
            register_tag=register_tag,
            gold_flag=gold,
        )
        for i in range(n)
    ]
    write_parquet(rows, path)


def test_compose_respects_weights_and_oversamples_gold(tmp_path) -> None:
    gold_path = tmp_path / "gold.parquet"
    mid_path = tmp_path / "mid.parquet"
    fill_path = tmp_path / "fill.parquet"

    # gold source: only 3 rows but weight 0.5 of 100 == 50; must oversample.
    _write_toy(gold_path, n=3, register_tag="manga", gold=True)
    # non-gold short source: 5 rows, weight 0.2 of 100 == 20; SHORT-fall, no oversample.
    _write_toy(mid_path, n=5, register_tag="novel", gold=False)
    # non-gold abundant: 100 rows, weight 0.3 of 100 == 30; sample down.
    _write_toy(fill_path, n=100, register_tag="anime_sub", gold=False)

    spec = {
        "target_n": 100,
        "sources": [
            {"name": "gold", "path": str(gold_path), "weight": 0.5, "gold": True},
            {"name": "mid", "path": str(mid_path), "weight": 0.2, "gold": False},
            {"name": "fill", "path": str(fill_path), "weight": 0.3, "gold": False},
        ],
    }
    merged, results = compose(spec, seed=123)

    by_name = {r.name: r for r in results}
    assert by_name["gold"].target == 50
    assert by_name["gold"].sampled == 50  # oversampled
    assert by_name["gold"].oversampled is True

    assert by_name["mid"].target == 20
    assert by_name["mid"].sampled == 5  # shortfall, no oversample
    assert by_name["mid"].oversampled is False

    assert by_name["fill"].target == 30
    assert by_name["fill"].sampled == 30
    assert by_name["fill"].oversampled is False

    # Merged rows = 50 + 5 + 30
    assert len(merged) == 85
    # Unified schema preserved
    assert set(["jp", "en", "src", "register_tag", "gold_flag"]).issubset(set(merged.columns))


def test_compose_skips_missing_path(tmp_path) -> None:
    # Empty case: missing path → sampled=0.
    spec = {
        "target_n": 10,
        "sources": [
            {"name": "absent", "path": str(tmp_path / "nope.parquet"), "weight": 1.0, "gold": True},
        ],
    }
    merged, results = compose(spec, seed=1)
    assert len(merged) == 0
    assert results[0].available == 0
    assert results[0].sampled == 0

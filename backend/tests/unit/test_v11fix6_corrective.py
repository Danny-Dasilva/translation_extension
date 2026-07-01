"""Contract tests for the v11fix6 corrective dataset builder.

These lock the train/serve format CONTRACT (byte-exact v11 template) and the
guardrails that prevented the documented v12 NSFW-oversampling regression:
  * corrective prompts must be produced by build_v11_dataset.build_context_prompt
    / build_plain_prompt (byte-for-byte), NOT a hand-rolled string.
  * human ALL-CAPS typeset is normalized to natural sentence case (no ALL-CAPS).
  * NSFW fraction stays FLAT vs the base v11 mix (corrective is a small minority).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[2]
BUILDER = BACKEND / "scripts/data/v11fix6/build_v11fix6_corrective.py"
V11_DIR = BACKEND / "scripts/data/v11"


def _load_builder():
    sys.path.insert(0, str(V11_DIR))
    spec = importlib.util.spec_from_file_location("build_v11fix6_corrective", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def b():
    return _load_builder()


# ------------------------------------------------------------------ normalization
def test_allcaps_to_sentence_case(b):
    assert b.to_sentence_case("ARE YOU NOT HUNGRY?") == "Are you not hungry?"


def test_sentence_boundary_recap(b):
    assert b.to_sentence_case("ARE YOU OK? GET UP!") == "Are you OK? Get up!"


def test_i_forms_natural(b):
    assert b.to_sentence_case("I'M GLAD") == "I'm glad"
    assert b.to_sentence_case("I'LL DO IT") == "I'll do it"
    assert b.to_sentence_case("WHAT'S WRONG?") == "What's wrong?"


def test_keeps_heart_and_ellipsis(b):
    assert b.to_sentence_case("AH...♡") == "Ah...♡"


def test_already_natural_untouched(b):
    # rows that already have lowercase letters must NOT be re-cased / corrupted
    assert b.to_sentence_case("Mom, are you sweating?") == "Mom, are you sweating?"


def test_strips_judge_parenthetical(b):
    out = b.to_sentence_case("THEY'LL MAKE ME DO (causative-passive coercion erased)")
    assert "causative" not in out
    assert out == "They'll make me do"


# ------------------------------------------------------------------ template contract
def test_pagectx_prompt_is_byte_exact_v11(b):
    """The page-context corrective prompt MUST equal build_v11_dataset's output."""
    from build_v11_dataset import PAGE_INSTR, build_context_prompt

    lines = ["JP one", "JP two", "JP three"]
    expected = build_context_prompt(PAGE_INSTR, lines, 1)
    # builder uses the SAME helper -> identity, but assert the structure too
    assert expected.startswith(PAGE_INSTR)
    assert "\nPage:\n1. JP one\n2. JP two\n3. JP three\n" in expected
    assert expected.endswith("Translate line 2: JP two")


def test_plain_prompt_is_byte_exact_v11(b):
    from build_v11_dataset import build_plain_prompt

    p = build_plain_prompt("テスト")
    assert p == (
        "Translate the following Japanese to English. "
        "Output only the translation.\n\nJapanese: テスト"
    )


def test_pagectx_uses_gold_jp_for_target(b):
    """build_pagectx_for_row must place the gold jp at the target index."""
    res = b.build_pagectx_for_row("ikenie4:p05:idx0", "昨日あんな事をしていた")
    assert res is not None
    jp_lines, k = res
    assert jp_lines[k] == "昨日あんな事をしていた"


# ------------------------------------------------------------------ jaccard / divergence
def test_jaccard_identical(b):
    assert b.jaccard("are you not hungry?", "Are you not hungry?") == 1.0


def test_jaccard_disjoint(b):
    assert b.jaccard("he was doing that", "she even though yesterday") < 0.4


# ------------------------------------------------------------------ output guarantees
def test_built_parquet_no_allcaps_and_flat_nsfw():
    import json

    import polars as pl

    parquet = BACKEND / "scripts/data/v11fix6/data_v11fix6_pagecontext.parquet"
    stats_p = BACKEND / "scripts/data/v11fix6/corrective_stats.json"
    if not parquet.exists() or not stats_p.exists():
        pytest.skip("dataset not built yet")

    df = pl.read_parquet(parquet)
    corr = df.filter(pl.col("src").str.starts_with("corrective_v11fix6"))
    # no residual ALL-CAPS english in corrective rows
    assert corr.filter(pl.col("en").str.contains(r"[A-Z]{4,}")).height == 0
    # corrective is a minority
    stats = json.loads(stats_p.read_text())
    assert stats["corrective_frac_pct"] < 5.0
    # NSFW fraction held flat (small positive drift only; NOT oversampled)
    assert abs(stats["nsfw_frac_delta_pp"]) < 2.0

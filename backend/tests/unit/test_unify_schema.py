"""Unit tests for ``scripts/data/unify_schema.py``."""

from __future__ import annotations

import pytest
import polars as pl

from unify_schema import (
    SCHEMA,
    VALID_REGISTER_TAGS,
    make_row,
    read_all_sources,
    read_parquet,
    rows_to_df,
    write_parquet,
)


def _toy_rows() -> list[dict]:
    return [
        make_row("おはよう", "Good morning.", "test:0", "manga", True),
        make_row(
            "昨日は晴れだった",
            "Yesterday was sunny.",
            "test:1",
            "novel",
            False,
        ),
        make_row("ドキドキ", "*thump-thump*", "test:2", "sfx", True),
    ]


def test_make_row_rejects_unknown_register() -> None:
    with pytest.raises(ValueError):
        make_row("x", "y", "z", "not_a_tag", False)


def test_make_row_rejects_non_string_jp_en() -> None:
    with pytest.raises(TypeError):
        make_row(None, "y", "z", "manga", False)  # type: ignore[arg-type]


def test_rows_to_df_schema_matches() -> None:
    df = rows_to_df(_toy_rows())
    assert df.columns == list(SCHEMA.keys())
    assert df.schema["jp"] == pl.Utf8
    assert df.schema["gold_flag"] == pl.Boolean
    assert len(df) == 3


def test_rows_to_df_empty() -> None:
    df = rows_to_df([])
    assert df.columns == list(SCHEMA.keys())
    assert len(df) == 0


def test_all_register_tags_declared() -> None:
    expected = {"manga", "vn_eroge", "vn", "anime_sub", "novel", "sfx", "anchor", "synthetic"}
    assert VALID_REGISTER_TAGS == expected


def test_write_and_read_round_trip(tmp_path) -> None:
    out = tmp_path / "toy.parquet"
    n = write_parquet(_toy_rows(), out)
    assert n == 3
    df = read_parquet(out)
    assert len(df) == 3
    assert df.columns == list(SCHEMA.keys())
    assert set(df["register_tag"].to_list()) == {"manga", "novel", "sfx"}
    assert set(df["gold_flag"].to_list()) == {True, False}


def test_read_all_sources_concatenates(tmp_path) -> None:
    # Write two schema-conformant parquets + one unrelated parquet.
    write_parquet(_toy_rows()[:1], tmp_path / "a.parquet")
    write_parquet(_toy_rows()[1:], tmp_path / "b.parquet")
    pl.DataFrame({"unrelated": [1, 2]}).write_parquet(tmp_path / "bad.parquet")

    df = read_all_sources(tmp_path)
    assert len(df) == 3
    assert df.columns == list(SCHEMA.keys())


def test_read_all_sources_missing_dir(tmp_path) -> None:
    df = read_all_sources(tmp_path / "does_not_exist")
    assert len(df) == 0
    assert df.columns == list(SCHEMA.keys())

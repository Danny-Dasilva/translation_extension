"""Unified schema + parquet I/O helpers for the JP->EN manga translator pipeline.

Schema (fixed):
    jp           : str  — Japanese source line
    en           : str  — English target line
    src          : str  — free-form source identifier (e.g. "vntl_v31_1k_train")
    register_tag : str  — one of {manga, vn_eroge, vn, anime_sub, novel, sfx, anchor, synthetic}
    gold_flag    : bool — True if curated/high-trust source

This module is intentionally dependency-light (polars only) so loaders can be
chained in CI without a full NLP env.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import polars as pl

VALID_REGISTER_TAGS: frozenset[str] = frozenset(
    {
        "manga",
        "vn_eroge",
        "vn",
        "anime_sub",
        "dialogue",      # subtitle / spoken casual register (v12)
        "novel",
        "sfx",
        "anchor",
        "general",       # clean general-purpose anchor (v12)
        "nsfw_doujin",   # mined NSFW doujin bubble pairs (v12)
        "synthetic",
        "garbage",  # OCR-noise -> "..." refusal examples
    }
)

# We annotate loosely (Any) because polars expresses dtypes both as classes
# (``pl.Boolean``) and as instances (``pl.Utf8``) depending on construction,
# and the common base type varies by polars version.
SCHEMA: dict[str, Any] = {
    "jp": pl.Utf8,
    "en": pl.Utf8,
    "src": pl.Utf8,
    "register_tag": pl.Utf8,
    "gold_flag": pl.Boolean,
}


def make_row(
    jp: str,
    en: str,
    src: str,
    register_tag: str,
    gold_flag: bool,
) -> dict[str, object]:
    """Build a single schema-conformant row.

    Raises ValueError on invalid register_tag or non-string/empty jp or en.
    """
    if register_tag not in VALID_REGISTER_TAGS:
        raise ValueError(
            f"register_tag={register_tag!r} not in {sorted(VALID_REGISTER_TAGS)}"
        )
    if not isinstance(jp, str) or not isinstance(en, str):
        raise TypeError(f"jp/en must be str, got {type(jp)}, {type(en)}")
    return {
        "jp": jp,
        "en": en,
        "src": src,
        "register_tag": register_tag,
        "gold_flag": bool(gold_flag),
    }


def rows_to_df(rows: Iterable[dict[str, object]]) -> pl.DataFrame:
    """Materialize an iterable of row-dicts into a DataFrame with the fixed schema."""
    rows_list = list(rows)
    if not rows_list:
        # Empty-frame with correct dtypes so downstream code doesn't choke.
        return pl.DataFrame(schema=SCHEMA)
    df = pl.DataFrame(rows_list)
    # Coerce to exact dtypes; fail loudly if a column is missing.
    for col, dtype in SCHEMA.items():
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r} in rows")
        df = df.with_columns(pl.col(col).cast(dtype))
    # Reorder.
    return df.select(list(SCHEMA.keys()))


def write_parquet(rows: Iterable[dict[str, object]], path: str | os.PathLike) -> int:
    """Write rows to parquet at ``path``. Returns number of rows written."""
    df = rows_to_df(rows)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return len(df)


def read_parquet(path: str | os.PathLike) -> pl.DataFrame:
    """Read a parquet, validating the schema conforms."""
    df = pl.read_parquet(str(path))
    missing = set(SCHEMA.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"Parquet at {path} missing columns: {missing}")
    return df.select(list(SCHEMA.keys()))


def read_all_sources(directory: str | os.PathLike) -> pl.DataFrame:
    """Read every ``*.parquet`` under ``directory`` and concatenate.

    Parquets that don't match the unified schema are skipped with a warning.
    Returns an empty DataFrame (with schema) if nothing matches.
    """
    from loguru import logger

    base = Path(directory)
    if not base.exists():
        logger.warning(f"read_all_sources: directory does not exist: {base}")
        return pl.DataFrame(schema=SCHEMA)
    frames: list[pl.DataFrame] = []
    for p in sorted(base.rglob("*.parquet")):
        try:
            frames.append(read_parquet(p))
        except ValueError as e:
            logger.warning(f"Skipping non-conformant parquet {p}: {e}")
    if not frames:
        return pl.DataFrame(schema=SCHEMA)
    return pl.concat(frames, how="vertical_relaxed")


def iter_rows(df: pl.DataFrame) -> Iterator[dict[str, object]]:
    """Yield row-dicts from a DataFrame in insertion order."""
    yield from df.iter_rows(named=True)


__all__ = [
    "SCHEMA",
    "VALID_REGISTER_TAGS",
    "iter_rows",
    "make_row",
    "read_all_sources",
    "read_parquet",
    "rows_to_df",
    "write_parquet",
]

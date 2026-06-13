"""Filter English-side quality on a unified-schema parquet.

Filters applied (all must pass):
  - KenLM perplexity < 1000 (requires a KenLM binary model; skipped with
    warning if not available — pass --kenlm PATH to enable).
  - No refusal regex match: ``I can't|I cannot|inappropriate|as an AI``.
  - No 4-gram token loop with ≥3 repetitions.
  - JP/EN char-length ratio ∈ [0.3, 4.0].

Writes:
  - <out>.parquet (rows that pass)
  - <out>.stats.json (per-filter drop counters)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from _cli_common import configure_logging, logger

if TYPE_CHECKING:
    from typing import Any


REFUSAL_RE = re.compile(
    r"\b(I can't|I cannot|inappropriate|as an AI)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class EnFilterStats:
    input_rows: int = 0
    output_rows: int = 0
    dropped_ppl: int = 0
    dropped_refusal: int = 0
    dropped_ngram_loop: int = 0
    dropped_length_ratio: int = 0
    skipped_filters: list[str] = field(default_factory=list)


def _load_kenlm(path: str | None) -> object | None:
    if not path:
        return None
    try:
        import kenlm  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("kenlm not installed; skipping PPL filter")
        return None
    try:
        return kenlm.Model(path)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"kenlm load failed: {e}")
        return None


def _ppl(model: "Any", s: str) -> float:
    # kenlm.Model.perplexity tokenizes on whitespace internally.
    return float(model.perplexity(s))


def has_ngram_loop(tokens: list[str], n: int = 4, threshold: int = 3) -> bool:
    """True if any n-gram appears ≥ threshold times consecutively."""
    if len(tokens) < n * threshold:
        return False
    for i in range(len(tokens) - n * threshold + 1):
        window = tokens[i : i + n]
        # Check if this n-gram repeats `threshold` times starting at i.
        repeats = 1
        for k in range(1, threshold):
            nxt = tokens[i + k * n : i + (k + 1) * n]
            if nxt == window:
                repeats += 1
            else:
                break
        if repeats >= threshold:
            return True
    return False


def _len_ratio(jp: str, en: str) -> float:
    if not en:
        return float("inf")
    return len(jp) / max(1, len(en))


def filter_df(
    df: pl.DataFrame,
    *,
    kenlm_path: str | None,
    ppl_threshold: float = 1000.0,
    len_ratio_lo: float = 0.3,
    len_ratio_hi: float = 4.0,
) -> tuple[pl.DataFrame, EnFilterStats]:
    stats = EnFilterStats(input_rows=len(df))
    kenlm_model = _load_kenlm(kenlm_path)
    if kenlm_model is None:
        stats.skipped_filters.append("kenlm_ppl")

    keep_mask: list[bool] = []
    for row in df.iter_rows(named=True):
        jp = row.get("jp") or ""
        en = row.get("en") or ""
        # length ratio
        lr = _len_ratio(jp, en)
        if not (len_ratio_lo <= lr <= len_ratio_hi):
            stats.dropped_length_ratio += 1
            keep_mask.append(False)
            continue
        # refusal
        if REFUSAL_RE.search(en):
            stats.dropped_refusal += 1
            keep_mask.append(False)
            continue
        # ngram loop
        tokens = _WORD_RE.findall(en.lower())
        if has_ngram_loop(tokens, n=4, threshold=3):
            stats.dropped_ngram_loop += 1
            keep_mask.append(False)
            continue
        # ppl
        if kenlm_model is not None:
            if _ppl(kenlm_model, en) >= ppl_threshold:
                stats.dropped_ppl += 1
                keep_mask.append(False)
                continue
        keep_mask.append(True)

    out_df = df.filter(pl.Series(keep_mask))
    stats.output_rows = len(out_df)
    return out_df, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats-out", default=None)
    parser.add_argument("--kenlm", default=None, help="Path to KenLM .binary model")
    parser.add_argument("--ppl-threshold", type=float, default=1000.0)
    parser.add_argument("--len-ratio-lo", type=float, default=0.3)
    parser.add_argument("--len-ratio-hi", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    df = pl.read_parquet(args.input)
    logger.info(f"loaded {len(df):,} rows from {args.input}")
    out_df, stats = filter_df(
        df,
        kenlm_path=args.kenlm,
        ppl_threshold=args.ppl_threshold,
        len_ratio_lo=args.len_ratio_lo,
        len_ratio_hi=args.len_ratio_hi,
    )
    stats_dict = {
        "input_rows": stats.input_rows,
        "output_rows": stats.output_rows,
        "dropped_ppl": stats.dropped_ppl,
        "dropped_refusal": stats.dropped_refusal,
        "dropped_ngram_loop": stats.dropped_ngram_loop,
        "dropped_length_ratio": stats.dropped_length_ratio,
        "skipped_filters": stats.skipped_filters,
        "pass_rate": (
            stats.output_rows / stats.input_rows if stats.input_rows else 0.0
        ),
    }
    logger.info(f"filter_en stats: {stats_dict}")
    if args.dry_run:
        print(json.dumps(stats_dict, indent=2))
        return
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path)
    stats_path = Path(args.stats_out) if args.stats_out else out_path.with_suffix(
        ".stats.json"
    )
    stats_path.write_text(json.dumps(stats_dict, indent=2))
    print(f"wrote {len(out_df)} rows to {out_path} (stats: {stats_path})")


if __name__ == "__main__":
    main()

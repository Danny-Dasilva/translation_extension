"""Filter Japanese-side quality on a unified-schema parquet.

Filters applied (all must pass):
  - fasttext langid probability for Japanese ≥ 0.70
  - Ratio of (hiragana + katakana + kanji) / len(jp) ≥ 0.60
  - Length: 3 ≤ chars ≤ 400
  - MeCab unk-token ratio < 0.40

Writes:
  - <out>.parquet — rows that pass
  - <out>.stats.json — counts per-filter, pass rate, p50/p95 char len

Models are loaded lazily on first row. The fasttext model is downloaded via
``huggingface_hub.snapshot_download("facebook/fasttext-language-identification")``
on first run if not already cached; see ``FASTTEXT_HF_REPO`` constant.

MeCab requires the ``fugashi`` + ``unidic-lite`` (or ``unidic``) packages.
If either fasttext or MeCab is unavailable, the corresponding filter is
SKIPPED (and the stats JSON records a `"skipped": [...]` list). This keeps
pipelines runnable in environments without the heavy NLP stack.
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


FASTTEXT_HF_REPO = "facebook/fasttext-language-identification"
FASTTEXT_FILENAME = "model.bin"

# Unicode ranges for Japanese scripts.
_HIRA_KATA_KANJI_RE = re.compile(
    r"[぀-ゟ"  # Hiragana
    r"゠-ヿ"  # Katakana
    r"ㇰ-ㇿ"  # Katakana Phonetic Extensions
    r"一-鿿"  # CJK Unified Ideographs
    r"㐀-䶿"  # CJK Ext A
    r"]"
)


@dataclass
class FilterStats:
    input_rows: int = 0
    output_rows: int = 0
    dropped_langid: int = 0
    dropped_char_ratio: int = 0
    dropped_length: int = 0
    dropped_mecab_unk: int = 0
    skipped_filters: list[str] = field(default_factory=list)


def _load_fasttext() -> object | None:
    try:
        import fasttext  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("fasttext not installed; skipping langid filter")
        return None
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("huggingface_hub missing; skipping langid")
        return None
    try:
        path = hf_hub_download(
            repo_id=FASTTEXT_HF_REPO, filename=FASTTEXT_FILENAME
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"fasttext model download failed: {e}")
        return None
    return fasttext.load_model(path)


def _load_mecab() -> object | None:
    try:
        import fugashi  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("fugashi not installed; skipping MeCab unk filter")
        return None
    try:
        return fugashi.Tagger()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"fugashi init failed: {e}; skipping MeCab unk filter")
        return None


def _jp_char_ratio(s: str) -> float:
    if not s:
        return 0.0
    total = len(s)
    jp = len(_HIRA_KATA_KANJI_RE.findall(s))
    return jp / total


def _fasttext_ja_prob(model: "Any", s: str) -> float:
    labels, probs = model.predict(s.replace("\n", " "), k=5)
    for lab, pr in zip(labels, probs):
        if "__label__jpn" in lab or "__label__ja" in lab:
            return float(pr)
    return 0.0


def _mecab_unk_ratio(tagger: "Any", s: str) -> float:
    tokens = list(tagger(s))
    if not tokens:
        return 1.0
    unk = 0
    for tok in tokens:
        # fugashi Token: .feature.pos1 may be "補助記号" for punct; we count
        # "名詞/固有名詞" unk by inspecting `tok.feature` which for unk becomes
        # a pos1 of "記号" + rare; the robust signal is `tok.is_unk`.
        is_unk = getattr(tok, "is_unk", None)
        if is_unk:
            unk += 1
    return unk / len(tokens)


def filter_df(
    df: pl.DataFrame,
    *,
    langid_threshold: float = 0.70,
    char_ratio_threshold: float = 0.60,
    min_chars: int = 3,
    max_chars: int = 400,
    unk_threshold: float = 0.40,
) -> tuple[pl.DataFrame, FilterStats]:
    stats = FilterStats(input_rows=len(df))

    ft_model = _load_fasttext()
    if ft_model is None:
        stats.skipped_filters.append("langid")
    mecab = _load_mecab()
    if mecab is None:
        stats.skipped_filters.append("mecab_unk")

    keep_mask: list[bool] = []
    for row in df.iter_rows(named=True):
        jp = row.get("jp") or ""
        # length
        if not (min_chars <= len(jp) <= max_chars):
            stats.dropped_length += 1
            keep_mask.append(False)
            continue
        # jp char ratio
        if _jp_char_ratio(jp) < char_ratio_threshold:
            stats.dropped_char_ratio += 1
            keep_mask.append(False)
            continue
        # fasttext
        if ft_model is not None:
            if _fasttext_ja_prob(ft_model, jp) < langid_threshold:
                stats.dropped_langid += 1
                keep_mask.append(False)
                continue
        # mecab unk
        if mecab is not None:
            if _mecab_unk_ratio(mecab, jp) >= unk_threshold:
                stats.dropped_mecab_unk += 1
                keep_mask.append(False)
                continue
        keep_mask.append(True)

    out_df = df.filter(pl.Series(keep_mask))
    stats.output_rows = len(out_df)
    return out_df, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input parquet (unified schema).")
    parser.add_argument("--out", required=True, help="Output parquet.")
    parser.add_argument("--stats-out", default=None, help="Stats JSON (default: <out>.stats.json).")
    parser.add_argument("--langid-threshold", type=float, default=0.70)
    parser.add_argument("--char-ratio-threshold", type=float, default=0.60)
    parser.add_argument("--min-chars", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=400)
    parser.add_argument("--unk-threshold", type=float, default=0.40)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    df = pl.read_parquet(args.input)
    logger.info(f"loaded {len(df):,} rows from {args.input}")

    out_df, stats = filter_df(
        df,
        langid_threshold=args.langid_threshold,
        char_ratio_threshold=args.char_ratio_threshold,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        unk_threshold=args.unk_threshold,
    )

    stats_dict = {
        "input_rows": stats.input_rows,
        "output_rows": stats.output_rows,
        "dropped_langid": stats.dropped_langid,
        "dropped_char_ratio": stats.dropped_char_ratio,
        "dropped_length": stats.dropped_length,
        "dropped_mecab_unk": stats.dropped_mecab_unk,
        "skipped_filters": stats.skipped_filters,
        "pass_rate": (
            stats.output_rows / stats.input_rows if stats.input_rows else 0.0
        ),
    }
    logger.info(f"filter_jp stats: {stats_dict}")

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

"""Load merged SFX/onomatopoeia sources → unified parquet.

Merges:
  1. ``sfx-onomatopoeia/yuiseki__onomatopoeia-ja-flat`` (parquet, 10,346 rows).
     Rows look like ``{text, text_lang, onomatopoeia_ja}`` — one English gloss
     per row. We emit (jp=onomatopoeia_ja, en=text) when text_lang == 'en'.
  2. ``sfx-onomatopoeia/github-composite/jp-onomatopoeia/onomatopoeia.json``
     — dict: ``{ja_word: [{english, details}, ...]}``. We expand multi-gloss
     to one row per gloss.
  3. ``sfx-onomatopoeia/github-composite/nihongoresources/giongo.txt``
     — TSV: ``hiragana\\tkatakana\\ttranslation\\tapplies to\\tmore specifically``
     (header row present). One row per hiragana and per katakana form.

Dedup by exact (jp, en). Tags: register_tag='sfx', gold_flag=True.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterator

import polars as pl

from _cli_common import configure_logging, logger
from unify_schema import make_row, write_parquet


YUISEKI_FLAT = (
    "backend/training/datasets/translation/sfx-onomatopoeia/"
    "yuiseki__onomatopoeia-ja-flat/data/train-00000-of-00001-6d5880b44e742ff9.parquet"
)
JP_ONO_JSON = (
    "backend/training/datasets/translation/sfx-onomatopoeia/"
    "github-composite/jp-onomatopoeia/onomatopoeia.json"
)
NIHONGO_TSV = (
    "backend/training/datasets/translation/sfx-onomatopoeia/"
    "github-composite/nihongoresources/giongo.txt"
)


# English gloss cleanup: strip wrapping triple-quotes and collapse whitespace.
_WRAP_QUOTES_RE = re.compile(r'^"""(.*)"""$', re.DOTALL)


def _clean_gloss(s: str) -> str:
    s = (s or "").strip()
    m = _WRAP_QUOTES_RE.match(s)
    if m:
        s = m.group(1).strip()
    # collapse internal whitespace (keep punctuation)
    s = re.sub(r"\s+", " ", s)
    return s


def load_yuiseki_flat(path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield (jp, en, src) triples."""
    if not path.exists():
        logger.warning(f"yuiseki-flat missing: {path}")
        return
    df = pl.read_parquet(path)
    for i, r in enumerate(df.iter_rows(named=True)):
        if (r.get("text_lang") or "").lower() != "en":
            continue
        jp = (r.get("onomatopoeia_ja") or "").strip()
        en = _clean_gloss(r.get("text") or "")
        if not jp or not en:
            continue
        yield jp, en, f"sfx_yuiseki_flat:{i}"


def load_jp_onomatopoeia(path: Path) -> Iterator[tuple[str, str, str]]:
    if not path.exists():
        logger.warning(f"jp-onomatopoeia json missing: {path}")
        return
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    for jp_word, glosses in data.items():
        jp = (jp_word or "").strip()
        if not jp:
            continue
        if not isinstance(glosses, list):
            continue
        for i, gloss in enumerate(glosses):
            if not isinstance(gloss, dict):
                continue
            en = _clean_gloss(gloss.get("english") or "")
            if not en:
                continue
            yield jp, en, f"sfx_jp_ono:{jp}:{i}"


def load_nihongo_giongo(path: Path) -> Iterator[tuple[str, str, str]]:
    if not path.exists():
        logger.warning(f"nihongoresources giongo missing: {path}")
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()  # skip header
        _ = header
        for lineno, line in enumerate(fh, start=2):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            hira = (cols[0] or "").strip()
            kata = (cols[1] or "").strip()
            translation = _clean_gloss(cols[2] or "")
            if not translation:
                continue
            if hira:
                yield hira, translation, f"sfx_giongo:{lineno}:hira"
            if kata and kata != hira:
                yield kata, translation, f"sfx_giongo:{lineno}:kata"


def iter_merged(
    yuiseki_flat: Path,
    jp_ono_json: Path,
    nihongo_tsv: Path,
) -> Iterator[dict[str, object]]:
    seen: set[tuple[str, str]] = set()
    dups = 0
    for jp, en, src in load_yuiseki_flat(yuiseki_flat):
        key = (jp, en)
        if key in seen:
            dups += 1
            continue
        seen.add(key)
        yield make_row(jp=jp, en=en, src=src, register_tag="sfx", gold_flag=True)
    for jp, en, src in load_jp_onomatopoeia(jp_ono_json):
        key = (jp, en)
        if key in seen:
            dups += 1
            continue
        seen.add(key)
        yield make_row(jp=jp, en=en, src=src, register_tag="sfx", gold_flag=True)
    for jp, en, src in load_nihongo_giongo(nihongo_tsv):
        key = (jp, en)
        if key in seen:
            dups += 1
            continue
        seen.add(key)
        yield make_row(jp=jp, en=en, src=src, register_tag="sfx", gold_flag=True)
    logger.info(f"sfx: emitted={len(seen)} exact_dups_dropped={dups}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yuiseki-flat", default=YUISEKI_FLAT)
    parser.add_argument("--jp-ono-json", default=JP_ONO_JSON)
    parser.add_argument("--nihongo-tsv", default=NIHONGO_TSV)
    parser.add_argument(
        "--out",
        default="backend/training/datasets/unified/sfx_merged.parquet",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    rows = iter_merged(
        Path(args.yuiseki_flat),
        Path(args.jp_ono_json),
        Path(args.nihongo_tsv),
    )
    if args.dry_run:
        print(f"sfx rows: {sum(1 for _ in rows)}")
        return
    n = write_parquet(rows, args.out)
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()

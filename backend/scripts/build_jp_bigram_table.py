"""Build the JP char-bigram frequency table used by the OCR substitution-garble
guard (FIX P3 in ocr_confidence_gate).

The guard flags OCR strings whose character-bigram perplexity is implausibly
high relative to real Japanese. We build the reference bigram distribution
OFFLINE from real manga GT text so the guard is corpus-grounded (not heuristic),
then ship a compact committed JSON table.

Corpus (MAIN tree, absolute paths):
  * manga109 per-line GT      (scripts/data/manga109/perline_gt.parquet, jp_text)
  * v11 translation dataset   (training/runs/manga-bubbles/data.parquet,  jp col)

Output (committed): app/data/jp_char_bigram.json
  {
    "version": 1,
    "total_unigrams": <int>,
    "total_bigrams": <int>,
    "unigram": {"<ch>": <count>, ...},   # only JP glyphs
    "bigram":  {"<ch><ch>": <count>, ...} # pruned to count >= MIN_COUNT
  }

Run from the MAIN backend dir (data lives there):
  cd backend && PYTHONPATH=. .venv/bin/python scripts/build_jp_bigram_table.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

# JP glyph ranges: hiragana, katakana, CJK kanji (+ ext-A), iteration mark, chōonpu.
_HIRA = (0x3040, 0x309F)
_KATA = (0x30A0, 0x30FF)
_KANJI = (0x4E00, 0x9FFF)
_KANJI_EXT_A = (0x3400, 0x4DBF)


def _is_jp(ch: str) -> bool:
    o = ord(ch)
    return (
        _HIRA[0] <= o <= _HIRA[1]
        or _KATA[0] <= o <= _KATA[1]
        or _KANJI[0] <= o <= _KANJI[1]
        or _KANJI_EXT_A[0] <= o <= _KANJI_EXT_A[1]
    )


def _jp_only(s: str) -> str:
    return "".join(c for c in s if _is_jp(c))


# Bigrams seen fewer than this many times across the whole corpus are pruned
# (they're rare/noise and would inflate the file). The guard treats any pruned
# (unseen) bigram as low-probability via add-k smoothing, which is exactly the
# signal we want for OCR substitution garbles.
MIN_COUNT = 3


def build(corpus_texts) -> dict:
    uni: Counter = Counter()
    bi: Counter = Counter()
    for text in corpus_texts:
        if not isinstance(text, str):
            continue
        jp = _jp_only(text)
        if len(jp) < 2:
            uni.update(jp)
            continue
        uni.update(jp)
        for i in range(len(jp) - 1):
            bi[jp[i] + jp[i + 1]] += 1

    total_uni = sum(uni.values())
    total_bi = sum(bi.values())
    pruned_bi = {k: c for k, c in bi.items() if c >= MIN_COUNT}
    return {
        "version": 1,
        "min_count": MIN_COUNT,
        "total_unigrams": total_uni,
        "total_bigrams": total_bi,
        "n_unigram_types": len(uni),
        "n_bigram_types_kept": len(pruned_bi),
        "unigram": dict(uni),
        "bigram": pruned_bi,
    }


def main() -> None:
    # Output goes next to THIS script's package (worktree). Corpus is resolved
    # from the CWD (run from the MAIN backend dir, where the data lives).
    script_backend = Path(__file__).resolve().parents[1]
    backend = Path.cwd()
    texts: list[str] = []

    perline = backend / "scripts" / "data" / "manga109" / "perline_gt.parquet"
    if perline.exists():
        df = pd.read_parquet(perline, columns=["jp_text"])
        texts.extend(df["jp_text"].dropna().astype(str).tolist())
        print(f"perline_gt: {len(df)} rows")

    v11 = backend / "training" / "runs" / "manga-bubbles" / "data.parquet"
    if v11.exists():
        df = pd.read_parquet(v11, columns=["jp"])
        texts.extend(df["jp"].dropna().astype(str).tolist())
        print(f"v11 data:  {len(df)} rows")

    if not texts:
        raise SystemExit("no corpus found — run from the MAIN backend dir")

    table = build(texts)
    out_dir = script_backend / "app" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "jp_char_bigram.json"
    out.write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    print(
        f"wrote {out} :: unigram_types={table['n_unigram_types']} "
        f"bigram_types_kept={table['n_bigram_types_kept']} "
        f"total_bigrams={table['total_bigrams']}"
    )


if __name__ == "__main__":
    main()

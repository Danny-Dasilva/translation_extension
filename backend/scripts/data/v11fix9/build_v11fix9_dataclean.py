#!/usr/bin/env python3
"""Build the v11fix9 SFT parquet = v11fix8 mix with LOCAL data-quality fixes applied.

This is a *pure data-cleaning* pass over the v11fix8 page-context parquet. It does NOT
re-mine, re-OCR, or change row text formatting. The schema/format is preserved exactly
(same columns, same dtypes, same prompt/en strings) so it feeds the identical training
pipeline with no format drift.

Fixes (each independently toggleable, thresholds are named constants):

  1. fix_cap_ellipsis_targets   -- DEFECT 1: '...' target saturation.
       ~7.5k rows whose EN target is just '...' (16x the next-most-frequent target,
       almost all from the deliberate gemma_anchor / ocr_garbage "garbage->ellipsis"
       anchor slices). Cap to ELLIPSIS_TARGET_CAP via a seeded random sample that keeps
       JP-source diversity. NOTE: these anchors are an anti-hallucination defence; capping
       trades some of that signal for a less '...'-saturated target distribution.

  2. fix_dedup_sources          -- DEFECT 2: open_mantra / manga109 duplication.
       open_mantra_train = 12,000 rows from only 897 unique (~13.4x); manga109 = 132,122
       rows from 75,332 unique (~1.75x, long tail up to 39x). Collapse each listed source
       to at most MAX_COPIES_PER_DUP copies of every unique (prompt, en).

  3. fix_drop_fragment_to_full  -- DEFECT 3: fragment->full-sentence mined rows.
       In the mined corpus_bitext slice, a short JP fragment (no sentence-final punct) is
       paired with a long, complete EN sentence -> the model learned to over-complete a
       partial line into a whole sentence (a root cause of repeated/over-generated text).
       Drop rows matching the fragment(JP) + full-sentence(EN) + length-blowup heuristic.

  4. fix_drop_garble (proxy)    -- DEFECT 4 (CONTEXT ONLY, real fix is the GPU re-mine).
       ~49.6% of mined rows carry non-AR JP garble from CPU OCR per the audit. That figure
       is only recoverable by comparing against an AR-GPU PARSeq re-OCR (DEFERRED, GPU box
       offline). Here we only drop the *obviously* garbled mined rows that simple character
       heuristics can catch with high precision (phrase-doubling, scattered interior
       ellipsis, Latin-in-CJK, repeated-kana kana-only). This is a conservative lower
       bound, NOT the real fix -- most garbled-but-plausible kana sequences are NOT caught
       and still require the re-mine.

Output: scripts/data/v11fix9/data_v11fix9_pagecontext.parquet  (+ v11fix9_stats.json,
        + dropped_rows_audit.json sidecar listing what each fix removed).

Run (CPU only):
  .venv/bin/python scripts/data/v11fix9/build_v11fix9_dataclean.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[2]

# Exact training schema -- MUST be preserved on output (no format drift).
COLS = ["prompt", "en", "src", "register_tag", "gold_flag"]
NSFW_TAGS = ["vn_eroge", "manga_nsfw"]

DEFAULT_BASE = BACKEND / "scripts/data/v11fix8/data_v11fix8_pagecontext.parquet"
DEFAULT_OUT_DIR = BACKEND / "scripts/data/v11fix9"

SEED = 20260629

# ---- Fix 1: ellipsis target cap ----
ELLIPSIS_TARGET_CAP = 2000          # keep ~this many ellipsis-only target rows
# stripped EN consisting solely of dots / ellipsis / katakana-middle-dot / whitespace
ELLIPSIS_RE = r"^[.…・．。\s]*$"

# ---- Fix 2: source dedup ----
# source_prefix -> max copies to keep per unique (prompt, en). 1 == collapse to unique set.
DEDUP_SOURCES = {"open_mantra_train": 1, "manga109": 1}

# ---- Fix 3 + 4 scope: only the CPU-OCR mined slice ----
MINED_PREFIX = "corpus_bitext"

# ---- Fix 3: fragment(JP) -> full-sentence(EN) ----
FRAG_JP_MAXLEN = 12        # JP fragment if <= this many chars and ...
FRAG_EN_MINWORDS = 8       # EN counts as a "full sentence" if >= this many words and ...
FRAG_EN_JP_CHAR_RATIO = 5.0  # ... EN is >= this many x longer (chars) than the JP fragment
# JP terminal-ish chars: if the JP ends with one of these it is NOT treated as a fragment.
_JP_TERMINAL = set("。．.！？!?…、，,・「」『』（）()♪")
_EN_TERMINAL = {".", "!", "?"}

# ---- Fix 4: obvious-garble proxy (high precision, conservative) ----
_KANJI = lambda c: "一" <= c <= "鿿"
_KANA = lambda c: ("぀" <= c <= "ゟ") or ("゠" <= c <= "ヿ")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _add_src_prefix(df: pl.DataFrame) -> pl.DataFrame:
    """Add a transient `_src_prefix` column (dataset id before the first ':')."""
    return df.with_columns(pl.col("src").str.split(":").list.get(0).alias("_src_prefix"))


def _extract_jp(prompt: str) -> str:
    """Recover the JP source line from either prompt format.

    pagectx:  '... Translate line N: <JP>'
    plain:    'Translate the following Japanese to English. ... Japanese: <JP>'
    """
    m = re.search(r"Translate line \d+:\s*(.+)\Z", prompt, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"Japanese:\s*(.+)\Z", prompt, re.S)
    return m.group(1).strip() if m else ""


# --------------------------------------------------------------------------------------
# Fix 1
# --------------------------------------------------------------------------------------
def fix_cap_ellipsis_targets(df: pl.DataFrame, cap: int = ELLIPSIS_TARGET_CAP,
                             seed: int = SEED) -> tuple[pl.DataFrame, dict]:
    """Cap ellipsis-only EN-target rows to `cap`, keeping a seeded diverse sample."""
    is_ell = pl.col("en").str.strip_chars().str.contains(ELLIPSIS_RE)
    ell = df.filter(is_ell)
    rest = df.filter(~is_ell)
    n_before = ell.height
    if n_before <= cap:
        return df, {"ellipsis_before": n_before, "ellipsis_after": n_before, "dropped": 0,
                    "note": "already under cap"}
    # seeded random sample preserves proportional source diversity in expectation
    kept = ell.sample(n=cap, shuffle=True, seed=seed)
    out = pl.concat([rest, kept], how="vertical")
    return out, {
        "ellipsis_before": n_before,
        "ellipsis_after": kept.height,
        "dropped": n_before - kept.height,
        "kept_distinct_jp_prompts": kept.select("prompt").n_unique(),
        "kept_by_src_prefix": dict(
            _add_src_prefix(kept)["_src_prefix"].value_counts(sort=True).iter_rows()
        ),
    }


# --------------------------------------------------------------------------------------
# Fix 2
# --------------------------------------------------------------------------------------
def fix_dedup_sources(df: pl.DataFrame, spec: dict[str, int] = DEDUP_SOURCES,
                      seed: int = SEED) -> tuple[pl.DataFrame, dict]:
    """For each source prefix in `spec`, keep <= N copies of every unique (prompt, en)."""
    df = _add_src_prefix(df)
    info: dict[str, dict] = {}
    keep_frames = [df.filter(~pl.col("_src_prefix").is_in(list(spec)))]
    for prefix, max_copies in spec.items():
        sub = df.filter(pl.col("_src_prefix") == prefix)
        if sub.height == 0:
            continue
        # deterministic shuffle, then keep first `max_copies` per (prompt, en)
        sub_s = sub.sample(fraction=1.0, shuffle=True, seed=seed)
        kept = (
            sub_s.with_columns(pl.cum_count("prompt").over(["prompt", "en"]).alias("_rk"))
            .filter(pl.col("_rk") <= max_copies)
            .drop("_rk")
        )
        info[prefix] = {
            "rows_before": sub.height,
            "unique_prompt_en": sub.select(["prompt", "en"]).n_unique(),
            "rows_after": kept.height,
            "dropped": sub.height - kept.height,
            "max_copies_per_dup": max_copies,
        }
        keep_frames.append(kept)
    out = pl.concat(keep_frames, how="vertical").drop("_src_prefix")
    return out, info


# --------------------------------------------------------------------------------------
# Fix 3
# --------------------------------------------------------------------------------------
def _is_fragment_jp(jp: str) -> bool:
    jp = jp.strip()
    return 0 < len(jp) <= FRAG_JP_MAXLEN and jp[-1] not in _JP_TERMINAL


def _is_full_sentence_en(en: str) -> bool:
    e = en.strip()
    if not e:
        return False
    core = e.rstrip("\"'”’) 。")
    if not core:
        return False
    return (
        len(e.split()) >= FRAG_EN_MINWORDS
        and core[-1] in _EN_TERMINAL
        and e[0].isupper()
    )


def _is_fragment_to_full(jp: str, en: str) -> bool:
    if not (_is_fragment_jp(jp) and _is_full_sentence_en(en)):
        return False
    jl = len(jp.strip())
    el = len(en.strip())
    return jl > 0 and (el / jl) >= FRAG_EN_JP_CHAR_RATIO


def fix_drop_fragment_to_full(df: pl.DataFrame,
                              scope_prefix: str = MINED_PREFIX) -> tuple[pl.DataFrame, dict]:
    """Drop mined rows pairing a short JP fragment with a long full EN sentence."""
    df = _add_src_prefix(df)
    scope = df.filter(pl.col("_src_prefix") == scope_prefix)
    rest = df.filter(pl.col("_src_prefix") != scope_prefix)
    if scope.height == 0:
        return df.drop("_src_prefix"), {"scope_rows": 0, "dropped": 0}
    jp = [_extract_jp(p) for p in scope["prompt"].to_list()]
    en = scope["en"].to_list()
    drop_mask = [_is_fragment_to_full(j, e) for j, e in zip(jp, en)]
    keep = pl.Series("_keep", [not d for d in drop_mask])
    kept = scope.filter(keep)
    n_drop = scope.height - kept.height
    uniq_dropped = (
        scope.filter(pl.Series("_d", drop_mask)).select(["prompt", "en"]).n_unique()
    )
    out = pl.concat([rest, kept], how="vertical").drop("_src_prefix")
    return out, {
        "scope_rows": scope.height,
        "dropped": n_drop,
        "dropped_unique_prompt_en": uniq_dropped,
        "pct_of_scope": round(n_drop / scope.height * 100, 1),
        "thresholds": {
            "FRAG_JP_MAXLEN": FRAG_JP_MAXLEN,
            "FRAG_EN_MINWORDS": FRAG_EN_MINWORDS,
            "FRAG_EN_JP_CHAR_RATIO": FRAG_EN_JP_CHAR_RATIO,
        },
    }


# --------------------------------------------------------------------------------------
# Fix 4 (proxy)
# --------------------------------------------------------------------------------------
def _garble_phrase_double(s: str) -> bool:
    t = re.sub(r"[。、．，！？!?…ー〜～・「」『』（）()♪♡\s.]+$", "", s.strip())
    n = len(t)
    return n >= 6 and n % 2 == 0 and t[: n // 2] == t[n // 2:] and len(set(t[: n // 2])) >= 2


def _garble_scattered_dots(s: str) -> bool:
    body = re.sub(r"[.．。]+$", "", s.strip())
    return len(re.findall(r"[.．]{2,}", body)) >= 2


def _garble_latin_in_cjk(s: str) -> bool:
    has_latin = bool(re.search(r"[A-Za-z]", s))
    has_cjk = any(_KANA(c) or _KANJI(c) for c in s)
    return has_latin and has_cjk


def _garble_repeat_kana_nokanji(s: str) -> bool:
    t = s.strip()
    repeat3 = bool(re.search(r"([぀-ゟ゠-ヿ])\1{2,}", t))
    no_kanji_long = len(t) >= 8 and not any(_KANJI(c) for c in t)
    return repeat3 and no_kanji_long


def _is_obvious_garble(jp: str) -> bool:
    return (
        _garble_phrase_double(jp)
        or _garble_scattered_dots(jp)
        or _garble_latin_in_cjk(jp)
        or _garble_repeat_kana_nokanji(jp)
    )


def fix_drop_garble(df: pl.DataFrame,
                    scope_prefix: str = MINED_PREFIX) -> tuple[pl.DataFrame, dict]:
    """Drop the *obviously* garbled mined rows (high-precision proxy only)."""
    df = _add_src_prefix(df)
    scope = df.filter(pl.col("_src_prefix") == scope_prefix)
    rest = df.filter(pl.col("_src_prefix") != scope_prefix)
    if scope.height == 0:
        return df.drop("_src_prefix"), {"scope_rows": 0, "dropped": 0}
    jp = [_extract_jp(p) for p in scope["prompt"].to_list()]
    drop_mask = [_is_obvious_garble(j) for j in jp]
    keep = pl.Series("_keep", [not d for d in drop_mask])
    kept = scope.filter(keep)
    n_drop = scope.height - kept.height
    uniq_dropped = (
        scope.filter(pl.Series("_d", drop_mask)).select(["prompt", "en"]).n_unique()
    )
    out = pl.concat([rest, kept], how="vertical").drop("_src_prefix")
    return out, {
        "scope_rows": scope.height,
        "dropped": n_drop,
        "dropped_unique_prompt_en": uniq_dropped,
        "pct_of_scope_proxy": round(n_drop / scope.height * 100, 1),
        "note": "high-precision proxy only; audit's 49.6% needs the AR-GPU PARSeq re-mine",
    }


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def _nsfw_frac(df: pl.DataFrame) -> float:
    n = df.filter(pl.col("register_tag").is_in(NSFW_TAGS)).height
    return round(n / df.height, 4) if df.height else 0.0


def _source_counts(df: pl.DataFrame) -> dict:
    d = _add_src_prefix(df)
    return dict(d["_src_prefix"].value_counts(sort=True).iter_rows())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--ellipsis-cap", type=int, default=ELLIPSIS_TARGET_CAP)
    ap.add_argument("--no-cap-ellipsis", action="store_true")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--no-drop-fragment", action="store_true")
    ap.add_argument("--no-drop-garble", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pl.read_parquet(args.base).select(COLS)
    in_schema = {c: str(t) for c, t in zip(base.columns, base.dtypes)}
    df = base

    before = {
        "total_rows": base.height,
        "ellipsis_rows": base.filter(
            pl.col("en").str.strip_chars().str.contains(ELLIPSIS_RE)
        ).height,
        "exact_dots_rows": base.filter(pl.col("en").str.strip_chars() == "...").height,
        "nsfw_frac": _nsfw_frac(base),
        "source_counts": _source_counts(base),
    }

    audit: dict[str, dict] = {}

    if not args.no_dedup:
        df, audit["fix2_dedup"] = fix_dedup_sources(df)
    if not args.no_cap_ellipsis:
        df, audit["fix1_ellipsis"] = fix_cap_ellipsis_targets(df, cap=args.ellipsis_cap)
    if not args.no_drop_fragment:
        df, audit["fix3_fragment_to_full"] = fix_drop_fragment_to_full(df)
    if not args.no_drop_garble:
        df, audit["fix4_garble_proxy"] = fix_drop_garble(df)

    # enforce exact schema / column order before writing
    df = df.select(COLS)
    out_schema = {c: str(t) for c, t in zip(df.columns, df.dtypes)}
    assert out_schema == in_schema, f"SCHEMA DRIFT! in={in_schema} out={out_schema}"

    out_parquet = out_dir / "data_v11fix9_pagecontext.parquet"
    df.write_parquet(out_parquet)

    after = {
        "total_rows": df.height,
        "ellipsis_rows": df.filter(
            pl.col("en").str.strip_chars().str.contains(ELLIPSIS_RE)
        ).height,
        "exact_dots_rows": df.filter(pl.col("en").str.strip_chars() == "...").height,
        "nsfw_frac": _nsfw_frac(df),
        "source_counts": _source_counts(df),
    }

    stats = {
        "base_parquet": str(args.base),
        "out_parquet": str(out_parquet),
        "schema_in": in_schema,
        "schema_out": out_schema,
        "schema_parity": out_schema == in_schema,
        "before": before,
        "after": after,
        "fixes": audit,
    }
    (out_dir / "v11fix9_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    (out_dir / "dropped_rows_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    # ---- console before -> after table ----
    def g(d, k, sub=None):
        return d[k] if sub is None else d[k].get(sub, 0)

    print("\n================  v11fix9 data-clean: before -> after  ================")
    print(f"{'metric':32s} {'before':>10s} {'after':>10s}")
    print(f"{'total rows':32s} {before['total_rows']:>10d} {after['total_rows']:>10d}")
    print(f"{'ellipsis-only EN targets':32s} {before['ellipsis_rows']:>10d} {after['ellipsis_rows']:>10d}")
    exact_label = '  (exact "...")'
    print(f"{exact_label:32s} {before['exact_dots_rows']:>10d} {after['exact_dots_rows']:>10d}")
    print(f"{'open_mantra_train rows':32s} {g(before['source_counts'],'open_mantra_train'):>10d} {g(after['source_counts'],'open_mantra_train'):>10d}")
    print(f"{'manga109 rows':32s} {g(before['source_counts'],'manga109'):>10d} {g(after['source_counts'],'manga109'):>10d}")
    print(f"{'corpus_bitext rows':32s} {g(before['source_counts'],'corpus_bitext'):>10d} {g(after['source_counts'],'corpus_bitext'):>10d}")
    print(f"{'nsfw_frac':32s} {before['nsfw_frac']:>10.4f} {after['nsfw_frac']:>10.4f}")
    print("-" * 70)
    if "fix3_fragment_to_full" in audit:
        f3 = audit["fix3_fragment_to_full"]
        print(f"fragment->full dropped : {f3['dropped']} rows ({f3['dropped_unique_prompt_en']} unique, {f3['pct_of_scope']}% of mined)")
    if "fix4_garble_proxy" in audit:
        f4 = audit["fix4_garble_proxy"]
        print(f"garble proxy dropped   : {f4['dropped']} rows ({f4['dropped_unique_prompt_en']} unique, {f4['pct_of_scope_proxy']}% of mined)  [proxy only]")
    print(f"schema parity          : {stats['schema_parity']}")
    print(f"\nwrote {out_parquet} ({df.height} rows)")
    print(f"wrote {out_dir / 'v11fix9_stats.json'}")
    print(f"wrote {out_dir / 'dropped_rows_audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

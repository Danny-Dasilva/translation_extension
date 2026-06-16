"""Build the v12 page-context SFT dataset (heavier NSFW + slang).

v12 = the FULL v11 page-context mix (manga109 page-context + plain, VN/novel
windowed + plain, sfx/garbage/anchor passthrough, corrective seed) PLUS a set of
NEW sources merged in, with a config knob to OVERSAMPLE the NSFW/slang slices.

It deliberately *reuses v11's build logic by importing it* (see ``v11`` below) so
the manga reading-order, bubble-collapse, windowing, plain-emission fractions and
prompt builders are byte-for-byte identical to the proven v11 dataset. v12 only
adds new sources on top and reweights the NSFW/slang registers.

Schema (unchanged from v11)
---------------------------
    [prompt, en, src, register_tag, gold_flag]

`prompt` is the full user message; the training template is passthrough "{prompt}".

NEW sources over v11
--------------------
1. New unified parquets (INGEST IF PRESENT, gracefully skip+log if missing).
   These are in the unified [jp, en, src, register_tag, gold_flag] schema:

     unified/vntl_chat.parquet         register_tag="vn_eroge"  (NSFW slang)  -> PLAIN
     unified/vntl_dpo.parquet          register_tag="vn_eroge"  (NSFW slang)  -> PLAIN
     unified/parallelfiction_v2.parquet register_tag="novel"                  -> WINDOWED (if groupable) else PLAIN
     unified/opensubtitles.parquet     register_tag="dialogue"  (NSFW/slang)  -> PLAIN
     unified/alt_parallel.parquet      register_tag="general"   (gold)        -> PLAIN

   We do NOT trust the file's own register_tag blindly — each source is INGESTED
   with the register_tag the v12 plan mandates (so oversampling buckets are
   stable even if upstream tagging drifts).

2. NSFW doujin page-context pairs (INGEST IF PRESENT, else skip+log):

     scripts/data/doujin/doujin_pairs.parquet   register_tag="nsfw_doujin"

   Schema [jp, en, src, register_tag, gold_flag] with per-page grouping
   recoverable from ``src``. We build manga-style page-context prompts from it.
   If the grouping is not recoverable we fall back to plain single-line.

NSFW / slang oversampling
-------------------------
The registers considered "NSFW/slang" are in ``NSFW_REGISTERS``. Every row whose
``register_tag`` is in that set is repeated ``NSFW_OVERSAMPLE`` times (config
knob, default 2x) before the final shuffle. The composition report prints
per-register-tag counts and the NSFW% of the final mix.

Outputs
-------
  data_v12_pagecontext.parquet      final training parquet
  data_v12_pagecontext.sample.jsonl small readable sample
  eval_pagecontext_heldout.jsonl    disjoint-page page-context eval (chrF++)

Run:
  backend/.venv/bin/python backend/scripts/data/v12/build_v12_dataset.py
  backend/.venv/bin/python backend/scripts/data/v12/build_v12_dataset.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import polars as pl

# Reuse v11's proven builders verbatim (manga reading order, windowing,
# prompt builders, plain-emission fractions). Importing the module is
# side-effect-free (build only runs under __main__).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v11"))
import build_v11_dataset as v11  # noqa: E402

# ---------------------------------------------------------------- paths
BACKEND = Path("/home/danny/Documents/personal/extension/backend")
V10 = BACKEND / "scripts/data/manga109/data_v10.parquet"
BUBBLES = BACKEND / "scripts/data/manga109/bubbles.parquet"
CORRECTIVE = BACKEND / "scripts/data/corrective/v11_corrective_seed.parquet"
UNIFIED = BACKEND / "training/datasets/unified"

# NEW unified parquets (may not exist yet -> skip+log).
VNTL_CHAT = UNIFIED / "vntl_chat.parquet"
VNTL_DPO = UNIFIED / "vntl_dpo.parquet"
PARALLELFICTION_V2 = UNIFIED / "parallelfiction_v2.parquet"
OPENSUBTITLES = UNIFIED / "opensubtitles.parquet"
ALT_PARALLEL = UNIFIED / "alt_parallel.parquet"

# NEW NSFW doujin page-context pairs (may not exist yet -> skip+log).
DOUJIN_PAIRS = BACKEND / "scripts/data/doujin/doujin_pairs.parquet"

OUT_DIR = BACKEND / "scripts/data/v12"
OUT_PARQUET = OUT_DIR / "data_v12_pagecontext.parquet"
OUT_SAMPLE = OUT_DIR / "data_v12_pagecontext.sample.jsonl"
OUT_EVAL = OUT_DIR / "eval_pagecontext_heldout.jsonl"

SEED = 42
COLS = ["prompt", "en", "src", "register_tag", "gold_flag"]

# ---------------------------------------------------------------- config knobs
# Manga page-context upweight (kept identical to v11).
MANGA_CTX_REPEAT = 2
# Corrective seed repeat (kept identical to v11).
CORRECTIVE_REPEAT = 3
# NSFW doujin page-context upweight (flagship NEW NSFW skill).
DOUJIN_CTX_REPEAT = 2

# parallelfiction_v2 is ~10M line-pairs — pure novel register. It is fluency
# BALLAST, not the manga/NSFW focus, so we subsample it to keep novel from
# flooding the mix. Sampled at the PAIR level before windowing (seeded).
PARALLELFICTION_CAP = 40_000

# Registers that count as NSFW / slang for the oversampling knob + report.
NSFW_REGISTERS: frozenset[str] = frozenset({"vn_eroge", "nsfw_doujin", "dialogue"})
# How many times to repeat every NSFW/slang row before the final shuffle.
# The user wants MUCH heavier NSFW/slang representation than v11 -> default 2x.
NSFW_OVERSAMPLE = 2


# ---------------------------------------------------------------- new-source ingest
def _coerce_unified(df: pl.DataFrame, register_tag: str) -> pl.DataFrame:
    """Force a unified [jp,en,src,register_tag,gold_flag] frame onto the v12
    register bucket. Upstream register_tag is overridden so oversampling buckets
    stay stable. Drops empty jp/en."""
    missing = {"jp", "en", "src", "register_tag", "gold_flag"} - set(df.columns)
    if missing:
        raise ValueError(f"unified parquet missing columns: {missing}")
    return (
        df.with_columns(pl.lit(register_tag).alias("register_tag"))
        .filter(
            pl.col("jp").is_not_null()
            & pl.col("en").is_not_null()
            & (pl.col("jp").str.strip_chars().str.len_chars() > 0)
            & (pl.col("en").str.strip_chars().str.len_chars() > 0)
        )
    )


def ingest_plain_unified(path: Path, register_tag: str, gold_flag: bool | None = None):
    """Load a unified parquet (if present) and emit PLAIN single-line rows.

    Returns (rows, status_str). Gracefully returns ([], 'missing') if absent.
    """
    if not path.exists():
        return [], "missing"
    df = _coerce_unified(pl.read_parquet(path), register_tag)
    out = []
    for r in df.iter_rows(named=True):
        out.append({
            "prompt": v11.build_plain_prompt(r["jp"]),
            "en": r["en"],
            "src": r["src"] + ":plain",
            "register_tag": register_tag,
            "gold_flag": bool(r["gold_flag"]) if gold_flag is None else gold_flag,
        })
    return out, f"loaded {len(out)}"


def _recover_group_turn(src: str) -> tuple[str, int | None]:
    """Best-effort (group, turn) recovery from a src like 'prefix:GROUP:idxN'.

    Returns (group_key, turn_index). turn is None if not parseable. Used by the
    windowed ingest for parallelfiction_v2 / doujin when grouping is present.
    """
    parts = src.split(":")
    if len(parts) < 3:
        return src, None
    grp = parts[1]
    tail = parts[-1]
    digits = "".join(ch for ch in tail if ch.isdigit())
    turn = int(digits) if digits else None
    return f"{parts[0]}:{grp}", turn


def ingest_windowed_unified(path: Path, register_tag: str, instr: str, cap: int | None = None):
    """Load a unified parquet (if present) and emit sliding-window CONTEXT rows
    (+ a plain fraction), grouped via src 'prefix:GROUP:idxN'.

    Falls back to PLAIN for any row whose group has <2 lines or unparseable
    grouping. If ``cap`` is set and the source exceeds it, whole GROUPS are
    sampled (seeded) until the row budget is met — this keeps windows intact
    rather than slicing mid-conversation. Returns (ctx_rows, plain_rows, status).
    """
    if not path.exists():
        return [], [], "missing"
    df = _coerce_unified(pl.read_parquet(path), register_tag)
    capped_note = ""
    if cap is not None and df.height > cap:
        # Sample whole GROUPS so conversation windows stay contiguous. Group key
        # = first two ':'-tokens (e.g. 'parallelfiction_v2:doc123'), computed with
        # vectorized polars (NOT a per-row UDF — the raw frame can be ~10M rows).
        total = df.height
        df = df.with_columns(
            pl.col("src").str.splitn(":", 3).struct.field("field_0")
            .add(pl.lit(":"))
            .add(pl.col("src").str.splitn(":", 3).struct.field("field_1").fill_null(""))
            .alias("_grp")
        )
        per_group = df.group_by("_grp").len().sort("_grp")  # deterministic order
        groups_shuffled = per_group.sample(fraction=1.0, shuffle=True, seed=SEED)
        cum = groups_shuffled.with_columns(pl.col("len").cum_sum().alias("_cum"))
        keep_groups = cum.filter(
            (pl.col("_cum") - pl.col("len")) < cap  # include the group that crosses the cap
        )["_grp"]
        df = df.filter(pl.col("_grp").is_in(keep_groups)).drop("_grp")
        capped_note = f" [capped {df.height:,}/{total:,} pairs from {len(keep_groups):,} groups]"
    groups: dict[str, list[dict]] = {}
    ungrouped: list[dict] = []
    for r in df.iter_rows(named=True):
        grp, turn = _recover_group_turn(r["src"])
        if turn is None:
            ungrouped.append(r)
        else:
            groups.setdefault(grp, []).append({**r, "_turn": turn})

    ctx_rows: list[dict] = []
    plain_rows: list[dict] = []
    for grp, rows in groups.items():
        rows.sort(key=lambda r: r["_turn"])
        jp_lines = [r["jp"] for r in rows]
        en_lines = [r["en"] for r in rows]
        for k in range(len(rows)):
            lo = max(0, k - v11.WINDOW)
            ctx = jp_lines[lo:k + 1]
            k2 = k - lo
            if len(ctx) >= 2:
                ctx_rows.append({
                    "prompt": v11.build_context_prompt(instr, ctx, k2),
                    "en": en_lines[k],
                    "src": rows[k]["src"] + ":convctx",
                    "register_tag": register_tag,
                    "gold_flag": bool(rows[k]["gold_flag"]),
                })
            if random.random() < v11.PLAIN_VN_NOVEL_FRAC:
                plain_rows.append({
                    "prompt": v11.build_plain_prompt(jp_lines[k]),
                    "en": en_lines[k],
                    "src": rows[k]["src"] + ":plain",
                    "register_tag": register_tag,
                    "gold_flag": bool(rows[k]["gold_flag"]),
                })
    # ungrouped rows -> plain
    for r in ungrouped:
        plain_rows.append({
            "prompt": v11.build_plain_prompt(r["jp"]),
            "en": r["en"],
            "src": r["src"] + ":plain",
            "register_tag": register_tag,
            "gold_flag": bool(r["gold_flag"]),
        })
    status = f"ctx={len(ctx_rows)} plain={len(plain_rows)} (groups={len(groups)} ungrouped={len(ungrouped)}){capped_note}"
    return ctx_rows, plain_rows, status


def ingest_doujin_pagectx(path: Path):
    """Load NSFW doujin page-context pairs (if present) and build manga-style
    page-context prompts grouped per page recovered from src.

    Expected src like 'doujin:WORK:pN:idx' (mirrors manga109's
    'manga109:BOOK:pN:hash'); we group on (work, page). If grouping is not
    recoverable for a row it falls back to plain. Returns (ctx, plain, status).
    """
    if not path.exists():
        return [], [], "missing"
    df = _coerce_unified(pl.read_parquet(path), "nsfw_doujin")
    pages: dict[tuple, list[dict]] = {}
    ungrouped: list[dict] = []
    for r in df.iter_rows(named=True):
        parts = r["src"].split(":")
        # need at least prefix:work:pPAGE:... to group a page
        page_tok = next((p for p in parts if p.startswith("p") and p[1:].isdigit()), None)
        if len(parts) >= 3 and page_tok is not None:
            work = parts[1]
            pages.setdefault((work, page_tok), []).append(r)
        else:
            ungrouped.append(r)

    ctx_rows: list[dict] = []
    plain_rows: list[dict] = []
    for key, rows in pages.items():
        # doujin pairs are already in reading order from the pipeline; we keep
        # the file order (no bbox available). Collapse consecutive identical
        # (jp,en) the same way manga109 does.
        collapsed: list[dict] = []
        for r in rows:
            if collapsed and collapsed[-1]["jp"] == r["jp"] and collapsed[-1]["en"] == r["en"]:
                continue
            collapsed.append(r)
        jp_lines = [r["jp"] for r in collapsed]
        en_lines = [r["en"] for r in collapsed]
        for k in range(len(collapsed)):
            ctx_lines, k2 = v11.window_slice(jp_lines, k)
            ctx_rows.append({
                "prompt": v11.build_context_prompt(v11.PAGE_INSTR, ctx_lines, k2),
                "en": en_lines[k],
                "src": collapsed[k]["src"] + ":pagectx",
                "register_tag": "nsfw_doujin",
                "gold_flag": bool(collapsed[k]["gold_flag"]),
            })
            if random.random() < v11.PLAIN_MANGA_FRAC:
                plain_rows.append({
                    "prompt": v11.build_plain_prompt(jp_lines[k]),
                    "en": en_lines[k],
                    "src": collapsed[k]["src"] + ":plain",
                    "register_tag": "nsfw_doujin",
                    "gold_flag": bool(collapsed[k]["gold_flag"]),
                })
    for r in ungrouped:
        plain_rows.append({
            "prompt": v11.build_plain_prompt(r["jp"]),
            "en": r["en"],
            "src": r["src"] + ":plain",
            "register_tag": "nsfw_doujin",
            "gold_flag": bool(r["gold_flag"]),
        })
    status = f"ctx={len(ctx_rows)} plain={len(plain_rows)} (pages={len(pages)} ungrouped={len(ungrouped)})"
    return ctx_rows, plain_rows, status


# ---------------------------------------------------------------- report
def kind_of(s: str) -> str:
    if s.endswith(":pagectx"):
        if s.startswith("doujin"):
            return "doujin_pagectx"
        return "manga_pagectx"
    if s.endswith(":convctx"):
        return "conv_ctx"
    if s.startswith("corrective_v11"):
        return "corrective"
    if s.endswith(":plain"):
        return f"plain:{s.split(':', 1)[0]}"
    return f"other:{s.split(':', 1)[0]}"


def composition_report(out: pl.DataFrame) -> None:
    rep = (
        out.with_columns(
            pl.col("src").map_elements(kind_of, return_dtype=pl.String).alias("kind")
        )
        .group_by("kind").len().sort("len", descending=True)
    )
    print("\n=== COMPOSITION (by src kind) ===")
    print(f"TOTAL rows: {out.height:,}")
    for r in rep.iter_rows(named=True):
        print(f"  {r['kind']:28s} {r['len']:>9,}  ({r['len']/out.height*100:4.1f}%)")

    # per-register-tag
    reg = out.group_by("register_tag").len().sort("len", descending=True)
    print("\n=== COMPOSITION (by register_tag) ===")
    for r in reg.iter_rows(named=True):
        flag = "  <-- NSFW/slang" if r["register_tag"] in NSFW_REGISTERS else ""
        print(f"  {r['register_tag']:16s} {r['len']:>9,}  ({r['len']/out.height*100:4.1f}%){flag}")

    nsfw = out.filter(pl.col("register_tag").is_in(list(NSFW_REGISTERS)))
    print(f"\n  NSFW/slang total: {nsfw.height:,} ({nsfw.height/out.height*100:.1f}%)"
          f"  [registers={sorted(NSFW_REGISTERS)}]")

    is_ctx = out.filter(pl.col("src").str.ends_with("ctx") | pl.col("src").str.ends_with("pagectx"))
    print(f"  context-augmented: {is_ctx.height:,} ({is_ctx.height/out.height*100:.1f}%)")
    print(f"  plain/single-line: {out.height-is_ctx.height:,} ({(out.height-is_ctx.height)/out.height*100:.1f}%)")


# ---------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Build a 1%% subsample, print the report, do NOT write outputs.")
    p.add_argument("--nsfw-oversample", type=int, default=NSFW_OVERSAMPLE,
                   help=f"repeat factor for NSFW/slang registers (default {NSFW_OVERSAMPLE})")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    v11.random.seed(args.seed)  # keep the reused builders deterministic too

    df = pl.read_parquet(V10)
    bubbles = pl.read_parquet(BUBBLES)
    corr = pl.read_parquet(CORRECTIVE)
    print(f"v10 rows={df.height:,}  bubbles={bubbles.height:,}  corrective={corr.height:,}")

    # ============================ v11 base mix (reused verbatim) ============================
    m_ctx, m_plain, eval_rows, n_heldout_pages = v11.build_manga(df, bubbles)
    print(f"[v11] manga: ctx={len(m_ctx)} plain={len(m_plain)} eval={len(eval_rows)} heldout_pages={n_heldout_pages}")

    vn_ctx, vn_plain = v11.build_windowed(df, "vntl_v31_1k", 1, "turn")
    print(f"[v11] vntl_v31_1k: ctx={len(vn_ctx)} plain={len(vn_plain)}")

    nv_ctx, nv_plain = v11.build_windowed(df, "nilane_small", 1, "pair")
    print(f"[v11] nilane_small: ctx={len(nv_ctx)} plain={len(nv_plain)}")

    plain_other = v11.passthrough_plain(
        df,
        (
            "vntl_raw", "parallelfiction", "open_mantra_train", "gemma_anchor",
            "sfx_jp_ono", "aratako_synth", "uuf_sfx", "sfx_yuiseki_flat",
            "sfx_giongo", "ocr_garbage",
        ),
    )
    print(f"[v11] plain_other: {len(plain_other)}")

    corr_rows = v11.corrective_rows(corr)
    print(f"[v11] corrective: {len(corr_rows)}")

    # ============================ v12 NEW sources ============================
    print("\n--- v12 NEW sources (ingest if present) ---")
    new_ctx: list[dict] = []
    new_plain: list[dict] = []

    # 1. Plain unified NSFW/slang + gold sources.
    rows, st = ingest_plain_unified(VNTL_CHAT, "vn_eroge")
    print(f"  vntl_chat            [vn_eroge]: {st}"); new_plain += rows
    rows, st = ingest_plain_unified(VNTL_DPO, "vn_eroge")
    print(f"  vntl_dpo             [vn_eroge]: {st}"); new_plain += rows
    rows, st = ingest_plain_unified(OPENSUBTITLES, "dialogue")
    print(f"  opensubtitles        [dialogue]: {st}"); new_plain += rows
    rows, st = ingest_plain_unified(ALT_PARALLEL, "general", gold_flag=True)
    print(f"  alt_parallel         [general] : {st}"); new_plain += rows

    # 2. parallelfiction_v2 -> windowed if groupable, else plain.
    pf_ctx, pf_plain, st = ingest_windowed_unified(PARALLELFICTION_V2, "novel", v11.CONV_INSTR, cap=PARALLELFICTION_CAP)
    print(f"  parallelfiction_v2   [novel]   : {st}")
    new_ctx += pf_ctx; new_plain += pf_plain

    # 3. NSFW doujin page-context pairs.
    dj_ctx, dj_plain, st = ingest_doujin_pagectx(DOUJIN_PAIRS)
    print(f"  doujin_pairs         [nsfw_doujin]: {st}")
    new_plain += dj_plain  # doujin ctx is upweighted separately below

    # ============================ compose ============================
    # Context rows are deduped (accidental dupes are real noise) — same as v11.
    m_ctx_df = pl.DataFrame(m_ctx, schema_overrides=None).select(COLS).unique(
        subset=["prompt", "en"], keep="first", maintain_order=True)
    conv_ctx_rows = vn_ctx + nv_ctx + new_ctx  # vntl/nilane + parallelfiction_v2 windows
    conv_ctx_df = (
        pl.DataFrame(conv_ctx_rows).select(COLS).unique(
            subset=["prompt", "en"], keep="first", maintain_order=True)
        if conv_ctx_rows else pl.DataFrame(schema={c: pl.Utf8 if c != "gold_flag" else pl.Boolean for c in COLS})
    )
    doujin_ctx_df = (
        pl.DataFrame(dj_ctx).select(COLS).unique(
            subset=["prompt", "en"], keep="first", maintain_order=True)
        if dj_ctx else pl.DataFrame(schema={c: pl.Utf8 if c != "gold_flag" else pl.Boolean for c in COLS})
    )

    ctx_parts = [m_ctx_df] * MANGA_CTX_REPEAT + [conv_ctx_df] + [doujin_ctx_df] * DOUJIN_CTX_REPEAT
    ctx_df = pl.concat([p for p in ctx_parts if p.height > 0], how="vertical") \
        if any(p.height > 0 for p in ctx_parts) else m_ctx_df

    # Plain rows keep multiplicity (v10/v11 robustness intent).
    plain_rows_all = m_plain + vn_plain + nv_plain + plain_other + new_plain
    plain_df = pl.DataFrame(plain_rows_all).select(COLS)

    # Corrective seed Nx, no dedup.
    corr_df = pl.DataFrame(corr_rows * CORRECTIVE_REPEAT).select(COLS)

    out = pl.concat([ctx_df, plain_df, corr_df], how="vertical")
    out = out.filter(
        (pl.col("prompt").str.len_chars() > 0) & (pl.col("en").str.len_chars() > 0)
    )

    # ---- NSFW/slang oversample knob: repeat NSFW rows before shuffle.
    if args.nsfw_oversample > 1:
        nsfw_mask = pl.col("register_tag").is_in(list(NSFW_REGISTERS))
        nsfw_df = out.filter(nsfw_mask)
        if nsfw_df.height > 0:
            extra = pl.concat([nsfw_df] * (args.nsfw_oversample - 1), how="vertical")
            out = pl.concat([out, extra], how="vertical")
            print(f"\nNSFW oversample x{args.nsfw_oversample}: +{extra.height:,} rows "
                  f"({nsfw_df.height:,} unique NSFW rows)")
        else:
            print(f"\nNSFW oversample x{args.nsfw_oversample}: no NSFW rows present yet (new sources missing)")

    out = out.sample(fraction=1.0, shuffle=True, seed=args.seed)

    if args.dry_run:
        out = out.sample(fraction=0.01, shuffle=True, seed=args.seed)
        print("\n[DRY-RUN] 1% subsample; NOT writing parquet/sample/eval.")

    composition_report(out)

    if args.dry_run:
        print("\n[DRY-RUN] done (no files written).")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_PARQUET)
    with OUT_SAMPLE.open("w") as f:
        for r in out.head(40).iter_rows(named=True):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT_EVAL.open("w") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {OUT_PARQUET} ({out.height:,} rows)")
    print(f"wrote {OUT_SAMPLE} (40 rows)")
    print(f"wrote {OUT_EVAL} ({len(eval_rows):,} held-out page-context eval rows, {n_heldout_pages} disjoint pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

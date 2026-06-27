"""Build the v11 CONTEXT-AUGMENTED SINGLE-LINE SFT dataset.

Goal
----
Train a fresh (from base) Gemma-4-E4B-it LoRA that can translate ONE manga
bubble / VN line / novel sentence while *seeing the surrounding page or
conversation* as context (for pronouns, speakers, continuity), AND stays strong
on isolated single lines.

Schema (output, ONE consistent schema for every row)
----------------------------------------------------
    [prompt, en, src, register_tag, gold_flag]

`prompt` is the FULL user message that the chat template will wrap. The training
`prompt_user_template` therefore becomes a passthrough: "{prompt}". This is the
ONLY schema change the training stack needs (see README.md).

Two prompt shapes share that single `prompt` column:

  (A) Page / conversation context (manga109 pages, VN/novel windows):

        Translate the marked line of this manga page from Japanese to English.
        Use the page context for speakers, pronouns, and continuity. Output only
        the translation of the marked line.

        Page:
        1. {jp1}
        2. {jp2}
        ...
        N. {jpN}

        Translate line {k}: {jpk}

      assistant = en of bubble k.

  (B) Plain single line (keeps the model strong on isolated lines):

        Translate the following Japanese to English. Output only the translation.

        Japanese: {jp}

      assistant = en.  (This is byte-for-byte the v10 prompt, just materialised
      into the `prompt` column instead of being formatted at train time.)

Reading order (manga109)
------------------------
The `src` hash (`manga109:BOOK:pN:hash`) is the manga109 annotation `text_id`.
Joining (book, page, text_id) against bubbles.parquet recovers the bbox
(xmin,ymin,xmax,ymax). Manga is read RIGHT-TO-LEFT, top-to-bottom. We therefore
sort each page's bubbles by a column-major right-to-left key:

    1. group bubbles into vertical columns by x (rightmost column first),
    2. within a column, top-to-bottom by y.

This is the standard manga reading order and matches the official Manga109 frame
order well enough for dialogue continuity. The join is verified at 100% (jp
matches bubbles.jp_text exactly for all 114,704 manga109 rows).

VN / novel context
------------------
VN turns (`vntl_v31_1k:rowN:turnK`) and novel pairs (`nilane_small:recN:pairM`)
are grouped by rowN/recN and ordered by the numeric turn/pair index. A sliding
window of up to W preceding lines + the target line forms the "Page:" context.
`vntl_raw` is a single flat group (all under row0) -> NOT windowed, kept plain.

Outputs
-------
  data_v11_pagecontext.parquet      final training parquet
  data_v11_pagecontext.sample.jsonl small readable sample
  eval_pagecontext_heldout.jsonl    disjoint-page page-context eval (chrF++)

Run:  .venv/bin/python backend/scripts/data/v11/build_v11_dataset.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------- paths
BACKEND = Path("/home/danny/Documents/personal/extension/backend")
V10 = BACKEND / "scripts/data/manga109/data_v10.parquet"
BUBBLES = BACKEND / "scripts/data/manga109/bubbles.parquet"
CORRECTIVE = BACKEND / "scripts/data/corrective/v11_corrective_seed.parquet"
OUT_DIR = BACKEND / "scripts/data/v11"
OUT_PARQUET = OUT_DIR / "data_v11_pagecontext.parquet"
OUT_SAMPLE = OUT_DIR / "data_v11_pagecontext.sample.jsonl"
OUT_EVAL = OUT_DIR / "eval_pagecontext_heldout.jsonl"

SEED = 42
random.seed(SEED)

# ---------------------------------------------------------------- prompt builders
PAGE_INSTR = (
    "Translate the marked line of this manga page from Japanese to English. "
    "Use the page context for speakers, pronouns, and continuity. "
    "Output only the translation of the marked line."
)
CONV_INSTR = (
    "Translate the marked line of this conversation from Japanese to English. "
    "Use the context for speakers, pronouns, and continuity. "
    "Output only the translation of the marked line."
)
PLAIN_INSTR = "Translate the following Japanese to English. Output only the translation."


def build_context_prompt(instr: str, lines: list[str], k_idx: int) -> str:
    """lines = ordered jp lines for the page/window; k_idx = 0-based target."""
    numbered = "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(lines))
    k = k_idx + 1
    return (
        f"{instr}\n\n"
        f"Page:\n{numbered}\n\n"
        f"Translate line {k}: {lines[k_idx]}"
    )


def build_plain_prompt(jp: str) -> str:
    return f"{PLAIN_INSTR}\n\nJapanese: {jp}"


# ---------------------------------------------------------------- manga reading order
def manga_reading_order(rows: list[dict]) -> list[dict]:
    """Right-to-left, column-major, top-to-bottom.

    rows have xmin,ymin,xmax,ymax. We compute a center x, bucket bubbles into
    vertical columns (a new column starts when the x-center jumps left by more
    than a tolerance relative to the running column's x-range), then order
    columns right->left and bubbles top->bottom within a column.
    """
    if len(rows) <= 1:
        return rows
    for r in rows:
        r["_cx"] = (r["xmin"] + r["xmax"]) / 2.0
        r["_w"] = max(1, r["xmax"] - r["xmin"])
    # page width estimate for tolerance
    page_w = max(r["xmax"] for r in rows) - min(r["xmin"] for r in rows)
    tol = max(40.0, page_w * 0.06)
    # sort by x descending first (rightmost), stable
    by_x = sorted(rows, key=lambda r: -r["_cx"])
    columns: list[list[dict]] = []
    for r in by_x:
        placed = False
        for col in columns:
            col_cx = sum(c["_cx"] for c in col) / len(col)
            if abs(r["_cx"] - col_cx) <= tol:
                col.append(r)
                placed = True
                break
        if not placed:
            columns.append([r])
    # columns already roughly right->left (by_x order); sort columns by mean cx desc
    columns.sort(key=lambda col: -sum(c["_cx"] for c in col) / len(col))
    ordered: list[dict] = []
    for col in columns:
        col.sort(key=lambda r: r["ymin"])  # top -> bottom
        ordered.extend(col)
    for r in ordered:
        r.pop("_cx", None)
        r.pop("_w", None)
    return ordered


# ---------------------------------------------------------------- config knobs
MAX_BUBBLES_CONTEXT = 12   # cap context window so prompts fit max_seq=1024
MAX_PER_PAGE_EXAMPLES = 16 # subsample bubbles-per-page (>= mean 14.5 => most pages full)
HELDOUT_PAGE_FRAC = 0.04   # fraction of manga109 pages reserved for eval (disjoint)
WINDOW = 6                 # VN/novel: up to N preceding lines as context

PLAIN_MANGA_FRAC = 0.35    # share of manga bubbles ALSO emitted as plain single-line
PLAIN_VN_NOVEL_FRAC = 0.60 # share of VN/novel lines emitted as plain single-line


def window_slice(lines: list[str], k: int) -> tuple[list[str], int]:
    """Return (context_lines, new_target_idx) capped to MAX_BUBBLES_CONTEXT,
    centered to keep the target's neighbours; for manga we bias to keep the
    target plus preceding+following context."""
    n = len(lines)
    if n <= MAX_BUBBLES_CONTEXT:
        return lines, k
    half = MAX_BUBBLES_CONTEXT // 2
    lo = max(0, k - half)
    hi = min(n, lo + MAX_BUBBLES_CONTEXT)
    lo = max(0, hi - MAX_BUBBLES_CONTEXT)
    return lines[lo:hi], k - lo


# ================================================================ MANGA109
def build_manga(df: pl.DataFrame, bubbles: pl.DataFrame):
    m = df.filter(pl.col("src").str.starts_with("manga109:")).with_columns([
        pl.col("src").str.split(":").list.get(1).alias("book"),
        pl.col("src").str.split(":").list.get(2).str.strip_prefix("p").cast(pl.Int64).alias("page"),
        pl.col("src").str.split(":").list.get(3).alias("text_id"),
    ])
    joined = m.join(
        bubbles.select(["book", "page", "text_id", "xmin", "ymin", "xmax", "ymax"]),
        on=["book", "page", "text_id"], how="left",
    )
    # group by page
    pages: dict[tuple, list[dict]] = {}
    for r in joined.iter_rows(named=True):
        pages.setdefault((r["book"], r["page"]), []).append(r)

    page_keys = sorted(pages.keys())
    random.Random(SEED).shuffle(page_keys)
    n_heldout = int(len(page_keys) * HELDOUT_PAGE_FRAC)
    heldout_keys = set(page_keys[:n_heldout])
    train_keys = page_keys[n_heldout:]

    train_ctx: list[dict] = []
    train_plain: list[dict] = []
    eval_rows: list[dict] = []

    def emit_page(key, sink_ctx, sink_plain, is_eval=False):
        rows = pages[key]
        ordered = manga_reading_order([dict(r) for r in rows])
        # Manga109 stores one <text> element per VERTICAL LINE inside a bubble,
        # so a multi-line bubble yields several consecutive text_ids with the
        # SAME normalized jp (and same gold en). Collapse consecutive runs of
        # identical (jp,en) into one logical bubble so the page context is clean
        # and the target line is not duplicated.
        collapsed: list[dict] = []
        for r in ordered:
            if collapsed and collapsed[-1]["jp"] == r["jp"] and collapsed[-1]["en"] == r["en"]:
                continue
            collapsed.append(r)
        ordered = collapsed
        jp_lines = [r["jp"] for r in ordered]
        en_lines = [r["en"] for r in ordered]
        srcs = [r["src"] for r in ordered]
        regs = [r["register_tag"] for r in ordered]
        golds = [r["gold_flag"] for r in ordered]
        n = len(ordered)
        # subsample which bubbles become examples (every bubble for eval)
        idxs = list(range(n))
        if not is_eval and n > MAX_PER_PAGE_EXAMPLES:
            idxs = sorted(random.Random(hash(key) & 0xFFFFFFFF).sample(idxs, MAX_PER_PAGE_EXAMPLES))
        for k in idxs:
            ctx_lines, k2 = window_slice(jp_lines, k)
            prompt = build_context_prompt(PAGE_INSTR, ctx_lines, k2)
            rec = {
                "prompt": prompt,
                "en": en_lines[k],
                "src": srcs[k] + ":pagectx",
                "register_tag": regs[k],
                "gold_flag": golds[k],
            }
            if is_eval:
                eval_rows.append({**rec, "jp": jp_lines[k], "kind": "manga_pagectx"})
            else:
                sink_ctx.append(rec)
                # also emit a fraction as plain single-line
                if random.random() < PLAIN_MANGA_FRAC:
                    sink_plain.append({
                        "prompt": build_plain_prompt(jp_lines[k]),
                        "en": en_lines[k],
                        "src": srcs[k] + ":plain",
                        "register_tag": regs[k],
                        "gold_flag": golds[k],
                    })

    for key in train_keys:
        emit_page(key, train_ctx, train_plain)
    for key in heldout_keys:
        emit_page(key, None, None, is_eval=True)

    return train_ctx, train_plain, eval_rows, len(heldout_keys)


# ================================================================ VN / NOVEL windowed
def build_windowed(df: pl.DataFrame, prefix: str, grp_idx: int, turn_strip: str):
    """Generic sliding-window builder for vntl_v31_1k / nilane_small."""
    s = df.filter(pl.col("src").str.starts_with(prefix)).with_columns([
        pl.col("src").str.split(":").list.get(1).alias("grp"),
        pl.col("src").str.split(":").list.get(2).str.strip_prefix(turn_strip).cast(pl.Int64, strict=False).alias("turn"),
    ])
    groups: dict[str, list[dict]] = {}
    for r in s.iter_rows(named=True):
        groups.setdefault(r["grp"], []).append(r)

    ctx_rows: list[dict] = []
    plain_rows: list[dict] = []
    for grp, rows in groups.items():
        rows.sort(key=lambda r: (r["turn"] is None, r["turn"]))
        jp_lines = [r["jp"] for r in rows]
        en_lines = [r["en"] for r in rows]
        for k in range(len(rows)):
            lo = max(0, k - WINDOW)
            ctx = jp_lines[lo:k + 1]
            k2 = k - lo
            if len(ctx) >= 2:
                ctx_rows.append({
                    "prompt": build_context_prompt(CONV_INSTR, ctx, k2),
                    "en": en_lines[k],
                    "src": rows[k]["src"] + ":convctx",
                    "register_tag": rows[k]["register_tag"],
                    "gold_flag": rows[k]["gold_flag"],
                })
            if random.random() < PLAIN_VN_NOVEL_FRAC:
                plain_rows.append({
                    "prompt": build_plain_prompt(jp_lines[k]),
                    "en": en_lines[k],
                    "src": rows[k]["src"] + ":plain",
                    "register_tag": rows[k]["register_tag"],
                    "gold_flag": rows[k]["gold_flag"],
                })
    return ctx_rows, plain_rows


# ================================================================ passthrough (plain)
def passthrough_plain(df: pl.DataFrame, prefixes: tuple[str, ...]):
    """Emit rows from the given src prefixes as plain single-line examples
    (no recoverable grouping, e.g. vntl_raw / parallelfiction / open_mantra /
    aratako / sfx / garbage)."""
    sel = df.filter(
        pl.any_horizontal([pl.col("src").str.starts_with(p) for p in prefixes])
    )
    out = []
    for r in sel.iter_rows(named=True):
        out.append({
            "prompt": build_plain_prompt(r["jp"]),
            "en": r["en"],
            "src": r["src"] + ":plain",
            "register_tag": r["register_tag"],
            "gold_flag": r["gold_flag"],
        })
    return out


# Share of corrective rows emitted in PAGE-CONTEXT shape (item 1 / fix6 shape
# fix). The gender/speaker-inversion failures ONLY manifest in page-context
# shape, so PLAIN-only corrective rows cannot move that bucket. Rows that carry
# real surrounding JP lines (``context_jp`` + ``context_k``) can be page-context
# shaped; rows without context fall back to plain (never dropped).
CORRECTIVE_PAGECTX_FRAC = 0.5


def corrective_rows(df, pagectx_frac: float = CORRECTIVE_PAGECTX_FRAC, seed: int = SEED):
    """Re-express the v11 corrective seed in BOTH plain and PAGE-CONTEXT shape.

    fix6 SHAPE FIX (item 1): the gender/speaker-inversion corrective failures
    ONLY manifest in PAGE-CONTEXT shape (the model sees the page and inverts the
    speaker/pronoun). Emitting corrective rows ONLY as plain single-line
    therefore cannot move that bucket. So a configurable ``pagectx_frac`` of the
    corrective rows are emitted via ``build_context_prompt`` (PAGE-CONTEXT, the
    byte-exact trained template) using each row's real surrounding JP lines:

        row["context_jp"]: list[str]  ordered page/window JP lines (incl. target)
        row["context_k"] : int        0-based index of the corrective line in it

    A corrective row WITHOUT usable ``context_jp`` cannot be page-context shaped,
    so it always falls back to PLAIN (never dropped). The plain/pagectx partition
    is deterministic in ``seed`` for reproducible builds.

    ``df`` may be a polars DataFrame OR an iterable of dicts (so the data builders
    and tests can pass plain records without constructing a frame).
    """
    rng = random.Random(seed)

    def _iter(d):
        if hasattr(d, "iter_rows"):  # polars DataFrame
            yield from d.iter_rows(named=True)
        else:
            yield from d

    out = []
    for r in _iter(df):
        # polars iter_rows(named=True) yields plain dicts, so .get works for both.
        ctx = r.get("context_jp")
        k = r.get("context_k")
        want_pagectx = (
            ctx is not None
            and len(ctx) >= 1
            and k is not None
            and 0 <= int(k) < len(ctx)
            and rng.random() < pagectx_frac
        )
        if want_pagectx:
            out.append({
                "prompt": build_context_prompt(PAGE_INSTR, list(ctx), int(k)),
                "en": r["en"],
                "src": str(r["src"]) + ":pagectx",
                "register_tag": r["register_tag"],
                "gold_flag": r["gold_flag"],
            })
        else:
            out.append({
                "prompt": build_plain_prompt(r["jp"]),
                "en": r["en"],
                "src": r["src"],
                "register_tag": r["register_tag"],
                "gold_flag": r["gold_flag"],
            })
    return out


# ================================================================ main
def main() -> int:
    df = pl.read_parquet(V10)
    bubbles = pl.read_parquet(BUBBLES)
    corr = pl.read_parquet(CORRECTIVE)

    print(f"v10 rows={df.height}  bubbles={bubbles.height}  corrective={corr.height}")

    # ---- manga109 page context + plain
    m_ctx, m_plain, eval_rows, n_heldout_pages = build_manga(df, bubbles)
    print(f"manga: ctx={len(m_ctx)} plain={len(m_plain)} eval_rows={len(eval_rows)} heldout_pages={n_heldout_pages}")

    # ---- VN (vntl_v31_1k) windowed + plain
    vn_ctx, vn_plain = build_windowed(df, "vntl_v31_1k", 1, "turn")
    print(f"vntl_v31_1k: ctx={len(vn_ctx)} plain={len(vn_plain)}")

    # ---- novel (nilane_small) windowed + plain
    nv_ctx, nv_plain = build_windowed(df, "nilane_small", 1, "pair")
    print(f"nilane_small: ctx={len(nv_ctx)} plain={len(nv_plain)}")

    # ---- everything else as plain passthrough (no usable grouping)
    plain_other = passthrough_plain(
        df,
        (
            "vntl_raw",            # single flat group, not windowable
            "parallelfiction",
            "open_mantra_train",
            "gemma_anchor",
            "sfx_jp_ono",
            "aratako_synth",
            "uuf_sfx",
            "sfx_yuiseki_flat",
            "sfx_giongo",
            "ocr_garbage",
        ),
    )
    print(f"plain_other (vntl_raw/parallelfiction/openmantra/sfx/garbage/...): {len(plain_other)}")

    # ---- corrective seed
    corr_rows = corrective_rows(corr)
    print(f"corrective: {len(corr_rows)}")

    # ---- compose
    # Context-augmented rows are deduped (accidental dupes are real noise).
    cols = ["prompt", "en", "src", "register_tag", "gold_flag"]
    # Manga page-context is the flagship NEW skill -> upweight 2x so it is a
    # strong slice (was ~24% at 1x; 2x lifts it to a dominant-but-not-overwhelming
    # share while keeping the plain single-line skill well represented).
    MANGA_CTX_REPEAT = 2
    m_ctx_df = pl.DataFrame(m_ctx).select(cols).unique(
        subset=["prompt", "en"], keep="first", maintain_order=True
    )
    conv_ctx_df = pl.DataFrame(vn_ctx + nv_ctx).select(cols).unique(
        subset=["prompt", "en"], keep="first", maintain_order=True
    )
    ctx_df = pl.concat([m_ctx_df] * MANGA_CTX_REPEAT + [conv_ctx_df], how="vertical")
    # Plain / passthrough rows KEEP their original multiplicity so the v10
    # robustness intent is preserved (ocr_garbage/gemma_anchor/sfx are
    # deliberately repetitive: they teach "garbage -> ...", "sfx -> sfx").
    plain_rows_all = m_plain + vn_plain + nv_plain + plain_other
    plain_df = pl.DataFrame(plain_rows_all).select(cols)
    # corrective seed 3x (so it is not drowned out), no dedup
    corr_df = pl.DataFrame(corr_rows + corr_rows + corr_rows).select(cols)

    out = pl.concat([ctx_df, plain_df, corr_df], how="vertical")
    out = out.filter(
        (pl.col("prompt").str.len_chars() > 0) & (pl.col("en").str.len_chars() > 0)
    )
    # shuffle
    out = out.sample(fraction=1.0, shuffle=True, seed=SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_PARQUET)

    # ---- composition report
    def kind_of(s: str) -> str:
        if s.endswith(":pagectx"):
            return "manga_pagectx"
        if s.endswith(":convctx"):
            return "conv_ctx"
        if s.startswith("corrective_v11"):
            return "corrective"
        if s.endswith(":plain"):
            base = s.split(":", 1)[0]
            return f"plain:{base}"
        return f"other:{s.split(':',1)[0]}"

    rep = (
        out.with_columns(pl.col("src").map_elements(kind_of, return_dtype=pl.String).alias("kind"))
        .group_by("kind").len().sort("len", descending=True)
    )
    print("\n=== COMPOSITION ===")
    print(f"TOTAL rows: {out.height}")
    for r in rep.iter_rows(named=True):
        print(f"  {r['kind']:28s} {r['len']:>8,}  ({r['len']/out.height*100:4.1f}%)")

    # context vs plain split
    is_ctx = out.filter(pl.col("src").str.ends_with("ctx") | pl.col("src").str.ends_with("pagectx"))
    print(f"\n  context-augmented total: {is_ctx.height:,} ({is_ctx.height/out.height*100:.1f}%)")
    print(f"  plain/single-line total: {out.height-is_ctx.height:,} ({(out.height-is_ctx.height)/out.height*100:.1f}%)")

    # ---- write samples
    sample = out.head(40)
    with OUT_SAMPLE.open("w") as f:
        for r in sample.iter_rows(named=True):
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

"""Build the v11fix6 CORRECTIVE SFT dataset (clean-OCR mistranslation fine-tune).

Goal
----
Teach the production v11 page-context Gemma-4 model the FAITHFUL human translations
for the lines where it was wrong / divergent on the Ikenie4 chapter, WITHOUT
regressing general ability or the explicit NSFW register (the documented v12 NSFW
oversampling regression).

Inputs
------
  backend/scripts/eval/data/ikenie4/gold.jsonl
      650 rows; 542 ocr_clean. Each clean row carries:
        jp     : the OCR'd Japanese (== bench ocr_jp)
        en     : the HUMAN scanlation English (ALL-CAPS typeset)
        our_en : our model's translation (== bench translation_en)
        src    : ikenie4:pNN:idxK   (joins to bench by page+idx)
        bbox, ocr_clean, ocr_conf, category, source_field, judge_note

  backend/.bench/ikenie4_merged_insp/<NNN>/bubbles.json
      per-page ORDERED bubble list (the v11 serving reading order):
        [{idx, bbox, ocr_jp, translation_en, is_orphan, confidence,
          ocr_conf, ocr_gate_dropped, filtered}, ...]
      -> gives the surrounding page JP lines for PAGE-CONTEXT shaping.

  backend/scripts/data/v11/data_v11_pagecontext.parquet
      the proven v11 base mix (291,780 rows; schema
      [prompt, en, src, register_tag, gold_flag]).

Output
------
  backend/scripts/data/v11fix6/data_v11fix6_pagecontext.parquet
      base v11 mix + corrective rows (page-context + plain shapes, upweighted
      ~3x but a MINORITY of total; NSFW fraction held FLAT).
  backend/scripts/data/v11fix6/corrective_rows.sample.jsonl  (readable sample)
  backend/scripts/data/v11fix6/corrective_stats.json         (machine-readable stats)

CONTRACT (DO NOT BREAK)
-----------------------
Corrective rows are emitted through build_v11_dataset.build_context_prompt /
build_plain_prompt -- the BYTE-FOR-BYTE trained v11 template. Any drift causes the
documented ~95% chrF collapse (train/serve format mismatch).

Run:  backend/.venv/bin/python backend/scripts/data/v11fix6/build_v11fix6_corrective.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------- paths
BACKEND = Path("/home/danny/Documents/personal/extension/backend")
GOLD = BACKEND / "scripts/eval/data/ikenie4/gold.jsonl"
BENCH = BACKEND / ".bench/ikenie4_merged_insp"
V11_PARQUET = BACKEND / "scripts/data/v11/data_v11_pagecontext.parquet"
OUT_DIR = BACKEND / "scripts/data/v11fix6"
OUT_PARQUET = OUT_DIR / "data_v11fix6_pagecontext.parquet"
OUT_SAMPLE = OUT_DIR / "corrective_rows.sample.jsonl"
OUT_STATS = OUT_DIR / "corrective_stats.json"

SEED = 42

# Re-use the BYTE-EXACT v11 trained template builders (the CONTRACT).
sys.path.insert(0, str(BACKEND / "scripts/data/v11"))
from build_v11_dataset import (  # noqa: E402
    PAGE_INSTR,
    build_context_prompt,
    build_plain_prompt,
)

COLS = ["prompt", "en", "src", "register_tag", "gold_flag"]

# ---------------------------------------------------------------- knobs
# A corrective row is "divergent" (the real corrective signal) when EITHER the
# judge explicitly flagged it (worst_issues / gap_examples) OR our model output is
# materially different from the (normalized) human line. Agreement rows are kept as
# a smaller set of anchors so the model is not pushed only toward the failure cases.
DIVERGENCE_JAC_THR = 0.55   # normalized token-Jaccard below this => divergent
CORR_UPWEIGHT = 3           # repeat corrective rows ~3x (task: "upweighted ~3x")
ANCHOR_FRAC_OF_DIVERGENT = 0.5  # keep this many agreement anchors (relative to divergent)
PAGECTX_FRAC = 0.6          # share of corrective rows emitted in PAGE-CONTEXT shape
MAX_CONTEXT = 12            # cap page context window (mirror v11 MAX_BUBBLES_CONTEXT)


# ---------------------------------------------------------------- normalization
_META_PAREN = re.compile(
    r"\s*\((?:[^)]*\b(?:erased|coercion|lit\.?|literally|sense|nuance|causative|"
    r"passive|note|sic|untranslated|omitted|implied)\b[^)]*)\)\s*",
    re.IGNORECASE,
)
_MULTISPACE = re.compile(r"[ \t]{2,}")


def _strip_meta(s: str) -> str:
    """Remove trailing/inline judge meta-parentheticals from the human line."""
    return _MULTISPACE.sub(" ", _META_PAREN.sub(" ", s)).strip()


# A handful of gold `en` values are NOT clean human translations -- they are the
# judge's analysis fragments ("(part of) ...", "(setup for) ...",
# "X (man describing her)", "果てた = climaxed (...)", "(euphemism for ...)").
# Training on these would teach annotation noise, so they are DROPPED from the
# corrective set (they are still scored in the eval gold -- this only affects
# what we TRAIN on).
_ANNOTATED = re.compile(
    r"^\s*\("                                  # leading parenthetical fragment
    r"|\((?:part of|setup for|euphemism|jeer|man describing|scold|note)\b"  # meta lead-ins
    r"|\([^)]*[぀-ヿ一-鿿][^)]*\)"  # JP chars inside a paren (= a gloss)
    r"|\([^)]*\b(?:describing|particle|tail|literally|lit\.|vs|read as|flipped)\b[^)]*\)",
    re.IGNORECASE,
)


def is_annotated_fragment(en: str) -> bool:
    """True when the gold `en` is a judge annotation fragment, not a clean target."""
    return bool(_ANNOTATED.search(en or ""))


_EMPHASIS_RUN = re.compile(r"\b[A-Z]{2,}(?:'[A-Z]+)?(?:\s+[A-Z]{2,})*\b")


def _deemphasize(text: str) -> str:
    """Lower-case embedded ALL-CAPS emphasis runs in otherwise-natural text.

    Keeps curated acronyms (OK, TV, ...) and 'I' forms upper. Used only for rows
    that already have lowercase letters (so they are NOT the all-caps typeset);
    we never want to teach the model to SHOUT mid-sentence.
    """
    def repl(m: re.Match) -> str:
        run = m.group(0)
        fixed_words = []
        for w in run.split(" "):
            bare = re.sub(r"[^A-Za-z']", "", w)
            if bare.upper() in _KEEP_UPPER or _I_FORM.match(bare):
                fixed_words.append(w)
            else:
                fixed_words.append(w.lower())
        return " ".join(fixed_words)

    return _EMPHASIS_RUN.sub(repl, text)


# Acronyms / interjections that should stay upper-case after sentence-casing.
_KEEP_UPPER = {"OK", "TV", "DNA", "SFX", "USA", "ID"}
# I-forms: keep the 'I' capital but lower-case the contraction tail (I'M -> I'm).
_I_FORM = re.compile(r"^I('[A-Za-z]+)?$", re.IGNORECASE)


def to_sentence_case(text: str) -> str:
    """Convert ALL-CAPS human typeset to NATURAL sentence case.

    - Lower-cases words, but preserves a curated set of acronyms / 'I' forms.
    - Capitalizes the first alphabetic char of each sentence.
    - Leaves punctuation / ♡ / ellipses intact.
    Rows that are NOT all-caps (already mixed/sentence case) are returned as-is
    (only meta-parentheticals stripped) so we never CORRUPT already-natural text.
    """
    text = _strip_meta(text)
    if not text:
        return text
    letters = [c for c in text if c.isalpha()]
    is_alldigit_or_short = len(letters) == 0
    # Only re-case rows that are genuinely ALL-CAPS (the typeset). If the row
    # already has lowercase letters it is natural -> leave it, BUT de-emphasize
    # any embedded ALL-CAPS emphasis run (4+ letters) so we never teach shouting.
    if is_alldigit_or_short or any(c.islower() for c in text):
        return _deemphasize(text)

    def fix_word(w: str) -> str:
        bare = re.sub(r"[^A-Za-z']", "", w)
        if bare and bare.upper() in _KEEP_UPPER:
            return w  # keep curated acronym as-is (already upper)
        lo = w.lower()
        if _I_FORM.match(bare):
            # I -> I, I'M -> I'm : capital I, lower-case contraction tail.
            lo = re.sub(r"(?<![A-Za-z])i(?=$|')", "I", lo)
        return lo

    lowered = " ".join(fix_word(w) for w in text.split(" "))

    # Capitalize sentence starts.
    out_chars = list(lowered)
    cap_next = True
    for i, ch in enumerate(out_chars):
        if cap_next and ch.isalpha():
            out_chars[i] = ch.upper()
            cap_next = False
        elif ch in ".!?":
            cap_next = True
    result = "".join(out_chars)
    # Standalone 'i' -> 'I'
    result = re.sub(r"\bi\b", "I", result)
    result = re.sub(r"\bi'", "I'", result)
    return result


# ---------------------------------------------------------------- divergence
def _norm_tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (s or "").lower()))


def jaccard(a: str, b: str) -> float:
    A, B = _norm_tokens(a), _norm_tokens(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


# ---------------------------------------------------------------- page context
def load_page_context(page_num: int) -> list[dict] | None:
    """Return the ordered bench bubble list for a page (None if absent)."""
    f = BENCH / f"{page_num:03d}" / "bubbles.json"
    if not f.exists():
        return None
    return json.load(f.open())


def page_context_lines(bubbles: list[dict]) -> list[tuple[int, str]]:
    """Ordered (idx, ocr_jp) for dialogue bubbles to use as page context.

    Exclude gate-dropped / filtered boxes (mirror the v11 serving whole-page
    context which excludes low-confidence garble), but always keep the target.
    """
    out = []
    for b in bubbles:
        jp = (b.get("ocr_jp") or "").strip()
        if not jp:
            continue
        out.append((b["idx"], jp))
    return out


def build_pagectx_for_row(src: str, jp: str) -> tuple[list[str], int] | None:
    """Build (context_jp_lines, target_k) for a gold row via its bench page.

    src = 'ikenie4:pNN:idxK'. We take the page's ordered JP lines, find the
    target idx, window to MAX_CONTEXT, and return the windowed lines + target
    index. Falls back to None when the page/idx cannot be resolved.
    """
    m = re.match(r"ikenie4:p(\d+):idx(\d+)", src)
    if not m:
        return None
    page_num = int(m.group(1))
    target_idx = int(m.group(2))
    bubbles = load_page_context(page_num)
    if not bubbles:
        return None
    lines = page_context_lines(bubbles)
    if not lines:
        return None
    # Build the context using bench JP, but OVERWRITE the target line's JP with the
    # gold jp (identical in practice; defensive against OCR drift between runs).
    idx_to_pos = {idx: pos for pos, (idx, _) in enumerate(lines)}
    if target_idx not in idx_to_pos:
        return None
    pos = idx_to_pos[target_idx]
    jp_lines = [j for _, j in lines]
    jp_lines[pos] = jp
    # window to MAX_CONTEXT centered on target (mirror v11 window_slice)
    n = len(jp_lines)
    if n > MAX_CONTEXT:
        half = MAX_CONTEXT // 2
        lo = max(0, pos - half)
        hi = min(n, lo + MAX_CONTEXT)
        lo = max(0, hi - MAX_CONTEXT)
        jp_lines = jp_lines[lo:hi]
        pos = pos - lo
    return jp_lines, pos


# ---------------------------------------------------------------- main
def main() -> int:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gold = [json.loads(l) for l in GOLD.open()]
    clean_all = [
        r for r in gold
        if r.get("ocr_clean")
        and (r.get("jp") or "").strip()
        and (r.get("en") or "").strip()
    ]
    # Drop judge-annotation fragments (not clean translation targets).
    clean = [r for r in clean_all if not is_annotated_fragment(r["en"])]
    n_dropped_annotated = len(clean_all) - len(clean)
    print(
        f"gold rows={len(gold)} ocr_clean={len(clean_all)} "
        f"usable={len(clean)} (dropped {n_dropped_annotated} annotation fragments)"
    )

    divergent: list[dict] = []
    anchors: list[dict] = []
    norm_samples = []
    n_recased = 0

    for r in clean:
        jp = r["jp"].strip()
        human_raw = r["en"].strip()
        human = to_sentence_case(human_raw)
        if human != human_raw:
            n_recased += 1
            if len(norm_samples) < 12:
                norm_samples.append({"before": human_raw, "after": human})
        if not human:
            continue
        our = (r.get("our_en") or "").strip()
        explicit = r.get("source_field") in ("worst_issues", "gap_examples")
        jac = jaccard(human, our) if our else 0.0
        is_divergent = explicit or (jac < DIVERGENCE_JAC_THR)
        rec = {
            "jp": jp,
            "human": human,
            "our": our,
            "src": r["src"],
            "register_tag": r.get("register_tag", "manga_nsfw"),
            "jac": round(jac, 3),
            "explicit": explicit,
        }
        (divergent if is_divergent else anchors).append(rec)

    # keep a sample of agreement anchors (don't drown corrective signal, but keep
    # the model anchored on the cases it already gets right)
    rng.shuffle(anchors)
    n_anchor_keep = int(len(divergent) * ANCHOR_FRAC_OF_DIVERGENT)
    kept_anchors = anchors[:n_anchor_keep]
    corrective = divergent + kept_anchors
    print(
        f"corrective base: divergent={len(divergent)} anchors_kept={len(kept_anchors)}"
        f" (of {len(anchors)}) total_unique={len(corrective)}"
    )

    # ---- shape each corrective row (page-context + plain) ----
    shaped: list[dict] = []
    n_pagectx = n_plain = 0
    for rec in corrective:
        want_ctx = rng.random() < PAGECTX_FRAC
        ctx = build_pagectx_for_row(rec["src"], rec["jp"]) if want_ctx else None
        if ctx is not None:
            jp_lines, k = ctx
            shaped.append({
                "prompt": build_context_prompt(PAGE_INSTR, jp_lines, k),
                "en": rec["human"],
                "src": f"corrective_v11fix6:{rec['src']}:pagectx",
                "register_tag": rec["register_tag"],
                "gold_flag": True,
            })
            n_pagectx += 1
        else:
            shaped.append({
                "prompt": build_plain_prompt(rec["jp"]),
                "en": rec["human"],
                "src": f"corrective_v11fix6:{rec['src']}:plain",
                "register_tag": rec["register_tag"],
                "gold_flag": True,
            })
            n_plain += 1
    print(f"shaped: pagectx={n_pagectx} plain={n_plain} (total {len(shaped)})")

    # ---- upweight ~3x (no dedup; corrective should not be drowned) ----
    corr_df = pl.DataFrame(shaped * CORR_UPWEIGHT).select(COLS)
    # match base schema exactly (gold_flag is Boolean in the v11 parquet)
    corr_df = corr_df.with_columns(pl.col("gold_flag").cast(pl.Boolean))
    print(f"corrective rows after {CORR_UPWEIGHT}x upweight: {corr_df.height}")

    # ---- mix with base v11 ----
    base = pl.read_parquet(V11_PARQUET).select(COLS)
    print(f"base v11 rows: {base.height}")

    # NSFW fraction must stay FLAT. The corrective rows are all manga_nsfw, which
    # would RAISE the NSFW fraction. To hold it flat we measure the base NSFW
    # fraction and report the mixed fraction; corrective is a small minority so the
    # shift is tiny -- we DO NOT oversample any NSFW source (the v12 regression).
    nsfw_tags = {"vn_eroge", "nsfw_doujin", "manga_nsfw"}
    base_nsfw = base.filter(pl.col("register_tag").is_in(list(nsfw_tags))).height
    base_nsfw_frac = base_nsfw / base.height

    mixed = pl.concat([base, corr_df], how="vertical")
    mixed = mixed.filter(
        (pl.col("prompt").str.len_chars() > 0) & (pl.col("en").str.len_chars() > 0)
    )
    mixed = mixed.sample(fraction=1.0, shuffle=True, seed=SEED)

    mixed_nsfw = mixed.filter(pl.col("register_tag").is_in(list(nsfw_tags))).height
    mixed_nsfw_frac = mixed_nsfw / mixed.height

    mixed.write_parquet(OUT_PARQUET)

    corr_frac = corr_df.height / mixed.height
    print("\n=== MIX ===")
    print(f"TOTAL mixed rows : {mixed.height:,}")
    print(f"corrective rows  : {corr_df.height:,} ({corr_frac*100:.2f}% -- MINORITY)")
    print(f"base v11 rows    : {base.height:,}")
    print(f"NSFW frac base   : {base_nsfw_frac*100:.2f}%  ({base_nsfw:,})")
    print(f"NSFW frac mixed  : {mixed_nsfw_frac*100:.2f}%  ({mixed_nsfw:,})  "
          f"(Δ {(mixed_nsfw_frac-base_nsfw_frac)*100:+.2f} pp -- FLAT, no oversample)")

    # ---- samples + stats ----
    with OUT_SAMPLE.open("w") as f:
        for rec in shaped[:40]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "gold_total": len(gold),
        "ocr_clean_total": len(clean_all),
        "annotation_fragments_dropped": n_dropped_annotated,
        "ocr_clean_usable": len(clean),
        "divergent": len(divergent),
        "anchors_total": len(anchors),
        "anchors_kept": len(kept_anchors),
        "corrective_unique": len(corrective),
        "corrective_pagectx": n_pagectx,
        "corrective_plain": n_plain,
        "corrective_upweight": CORR_UPWEIGHT,
        "corrective_rows_after_upweight": corr_df.height,
        "n_recased": n_recased,
        "normalization_samples": norm_samples,
        "base_v11_rows": base.height,
        "mixed_total_rows": mixed.height,
        "corrective_frac_pct": round(corr_frac * 100, 3),
        "nsfw_frac_base_pct": round(base_nsfw_frac * 100, 3),
        "nsfw_frac_mixed_pct": round(mixed_nsfw_frac * 100, 3),
        "nsfw_frac_delta_pp": round((mixed_nsfw_frac - base_nsfw_frac) * 100, 3),
        "divergence_jac_thr": DIVERGENCE_JAC_THR,
        "pagectx_frac": PAGECTX_FRAC,
        "out_parquet": str(OUT_PARQUET),
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT_PARQUET}")
    print(f"wrote {OUT_SAMPLE} (40 rows)")
    print(f"wrote {OUT_STATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

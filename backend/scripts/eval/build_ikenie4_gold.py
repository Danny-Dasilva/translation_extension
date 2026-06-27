#!/usr/bin/env python
"""(Re)generate the frozen Ikenie4 MT regression gold set.

Seeds a deterministic gold JSONL from the 24-agent LLM-judge comparison output
(``worst_issues[]`` + ``synthesis.gap_categories[].examples``) and joins each
``jp`` line back to the on-disk bubbles.json to recover ``ocr_conf`` / ``bbox``
and a stable ``idx``-based ``src`` id.

Why this exists
---------------
The 24-agent comparison is *stochastic* (LLM judge, ~±20 chrF-equivalent noise).
We use it ONCE, here, to mine a curated set of (jp, human_en) pairs plus failure
labels, then FREEZE the result into ``data/ikenie4/gold.jsonl``.  All future
regression eval scores against this frozen file with deterministic metrics
(chrF++/COMET via score_jsonl_metrics + seeded paired bootstrap), so a code or
model change gets a trustworthy Δ instead of re-rolling the judge.

ocr_clean flag (CRITICAL)
-------------------------
chrF++/COMET must only be computed on the ``ocr_clean`` subset.  The OCR'd ``jp``
on a garbled bubble is corrupted input, so scoring our EN against the human EN
there measures OCR noise, not the translation model.  We classify each row from
the judge ``note``:

  * note mentions "OCR-error" / "garbage-in" / "garbled" / "corrupted" /
    "mis-OCR" / "noise" / "low-conf"   -> ocr_clean = False
  * note mentions "clean OCR" / "translation error" / explicit model fault
                                       -> ocr_clean = True
  * category in {ocr_garbled, name_romaji}  -> ocr_clean = False (OCR-rooted)
  * otherwise default to ocr_clean = True (pure-model bucket) UNLESS the note
    flags partial garble.

Output schema (matches score_jsonl_metrics.py gold side: needs jp + en)
-----------------------------------------------------------------------
    {
      "jp": "<OCR'd source line>",
      "en": "<human_en, the reference>",
      "src": "ikenie4:p05:idx0",
      "register_tag": "manga_nsfw",
      "category": "pronoun_gender",
      "severity": 2,
      "ocr_clean": true,
      "ocr_conf": 0.9265,          # recovered from bubbles.json when matched
      "bbox": {...},               # recovered from bubbles.json when matched
      "our_en": "He was doing that thing yesterday.",  # for reference/debug
      "source_field": "worst_issues" | "gap_examples",
      "judge_note": "<...>"
    }

Usage
-----
    PYTHONPATH=. python backend/scripts/eval/build_ikenie4_gold.py \
        --comparison /tmp/.../wdcumj4jp.output \
        --bubbles-root /home/danny/Documents/personal/extension/backend/.bench/ikenie4_final_insp \
        --out backend/scripts/eval/data/ikenie4/gold.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# -- The p41 offset (BAKED IN, documented) ----------------------------------
#
# The bench output (our pipeline) has 134 pages (001..134).  The human GT
# scanlation directory has 133 webp.  The comparison reported
# ``missing_gt_page: 41`` -- GT page 41 does not exist, so from bench page 41
# onward the GT image index is shifted by one.  Concretely, to map a bench
# (our) page number to the GT webp page number:
#
#     gt_page(our_page) = our_page          if our_page < 41
#                       = our_page - 1       if our_page >= 41
#
# This module only deals with the *bench / our* page numbers (the judge's
# ``page`` field and bubbles.json paths are all our-page indexed), so the
# offset is not applied here; it is exported as ``our_page_to_gt_page`` for the
# one-time vision transcription pass (transcribe_gt_vision.py) which must read
# the actual GT image.
MISSING_GT_PAGE = 41


def our_page_to_gt_page(our_page: int, *, missing_gt_page: int = MISSING_GT_PAGE) -> int:
    """Map a bench/our page number to the human GT webp page number.

    The GT is missing one interior page (``missing_gt_page``), so every bench
    page at or after it is shifted by +1 relative to the GT file index.
    """
    if our_page >= missing_gt_page:
        return our_page - 1
    return our_page


# ---------------------------------------------------------------------------
# ocr_clean classification
# ---------------------------------------------------------------------------

# Substrings in the judge note that signal the OCR input itself is corrupted
# (so the row must be EXCLUDED from chrF++/COMET scoring).
_OCR_BAD_MARKERS = (
    "ocr-error",
    "ocr error",
    "garble",
    "garbled",
    "garbage-in",
    "garbage in",
    "corrupt",
    "mis-ocr",
    "misocr",
    "mis-read",
    "misread",
    "noise",
    "low-conf",
    "low conf",
    "partial ocr",
    "ocr partial",
    "ocr head",
    "ocr badly",
    "ocr inserted",
    "ocr is corrupted",
)

# Substrings that affirmatively mark a CLEAN-OCR, pure-model fault.
_OCR_CLEAN_MARKERS = (
    "clean ocr",
    "clean-ocr",
    "clean-ish ocr",
    "translation error",
    "translation-error",
    "pure model",
    "model error",
)

# Categories whose ROOT cause is OCR; default these to ocr_clean=False even if
# the note wording is ambiguous.
_OCR_ROOTED_CATEGORIES = {"ocr_garbled", "name_romaji"}

# Map the long gap_category titles (synthesis.gap_categories[].category) to the
# short, stable category slugs used by worst_issues, so the gold set has one
# consistent taxonomy.  Keyed by a lowercase substring of the title.
_GAP_TITLE_TO_SLUG: list[tuple[str, str]] = [
    ("hallucinated proper names", "name_romaji"),
    ("garbled ocr", "ocr_garbled"),
    ("subject/pronoun and gender", "pronoun_gender"),
    ("lexical/idiom mistranslation", "mistranslation"),
    ("sentence split across bubbles", "punctuation_split"),
    ("wrongly gate-dropped", "omission"),
    ("sfx onomatopoeia", "sfx_untranslated"),
    ("meta-description", "meta_leak"),
    ("register softening", "register_tone"),
    ("literal/unnatural", "literal_unnatural"),
    ("number/unit", "number_unit"),
]


def normalize_category(category: str) -> str:
    """Collapse long gap-category titles to short worst_issues-style slugs."""
    c = (category or "").strip()
    low = c.lower()
    for needle, slug in _GAP_TITLE_TO_SLUG:
        if needle in low:
            return slug
    return c


def classify_ocr_clean(category: str, note: str) -> bool:
    """Return True iff the OCR input for this row is clean enough to score.

    Note markers win over category defaults: an explicit "clean OCR" note on a
    name_romaji row marks it clean; an "OCR-error" note on a mistranslation row
    marks it dirty.
    """
    n = (note or "").lower()
    has_bad = any(m in n for m in _OCR_BAD_MARKERS)
    has_clean = any(m in n for m in _OCR_CLEAN_MARKERS)

    # Explicit "clean OCR" with no contradicting garble marker -> clean.
    if has_clean and not has_bad:
        return True
    # Any garble/OCR-error marker -> dirty (input corrupted).
    if has_bad:
        return False
    # No decisive marker: fall back to category root-cause.
    if (category or "").lower() in _OCR_ROOTED_CATEGORIES:
        return False
    # Pure-model buckets (mistranslation/pronoun_gender/...) default to clean.
    return True


# ---------------------------------------------------------------------------
# Parsing the comparison output
# ---------------------------------------------------------------------------

# A gap-category example line looks like:
#   "p64 新しいベニュニの子 => 'the new Beignet child' | I tried asking for ..."
# i.e.  "p<NN> <jp> => '<our_en>' | <human_en>"
_GAP_RE = re.compile(
    r"^p(?P<page>\d+)\s+(?P<jp>.+?)\s*=>\s*(?P<our>.+?)\s*\|\s*(?P<human>.+)$"
)


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "'\"" and s[-1] in "'\"":
        return s[1:-1].strip()
    return s


# Trailing ASCII-parenthetical annotation the judge sometimes appends to the JP
# in a gap example, e.g. "濃く (thick) dropped via duplicate-OCR gate".  We keep
# only the leading JP token run and drop the English annotation tail.
_JP_ANNOT_RE = re.compile(r"\s*[\(（].*$")
_TRAILING_ASCII_RE = re.compile(r"\s+[A-Za-z].*$")


def clean_gap_jp(jp: str) -> str:
    """Strip trailing English/parenthetical annotation from a gap-example JP."""
    s = jp.strip()
    s = _JP_ANNOT_RE.sub("", s).strip()
    # Drop a trailing ASCII-word annotation like "... dropped" / "... split".
    s = _TRAILING_ASCII_RE.sub("", s).strip()
    return s


def parse_worst_issues(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract structured rows from ``result.worst_issues[]``."""
    rows: list[dict[str, Any]] = []
    issues = comparison.get("result", {}).get("worst_issues", []) or []
    for it in issues:
        jp = (it.get("jp") or "").strip()
        human = (it.get("human_en") or "").strip()
        if not jp or not human:
            continue
        rows.append(
            {
                "page": int(it["page"]),
                "jp": jp,
                "our_en": (it.get("our_en") or "").strip(),
                "human_en": human,
                "category": (it.get("category") or "unknown").strip(),
                "severity": int(it.get("severity") or 0),
                "judge_note": (it.get("note") or "").strip(),
                "judge_ocr_conf": it.get("ocr_conf"),
                "source_field": "worst_issues",
            }
        )
    return rows


def parse_gap_examples(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract rows from ``synthesis.gap_categories[].examples[]`` strings."""
    rows: list[dict[str, Any]] = []
    syn = comparison.get("result", {}).get("synthesis", {}) or {}
    for gc in syn.get("gap_categories", []) or []:
        cat = (gc.get("category") or "unknown").strip()
        root = (gc.get("root_cause") or "").strip()
        sev = gc.get("severity_avg")
        for ex in gc.get("examples", []) or []:
            m = _GAP_RE.match(ex.strip())
            if not m:
                continue
            jp = clean_gap_jp(m.group("jp"))
            human = m.group("human").strip()
            our = _strip_quotes(m.group("our"))
            if not jp or not human:
                continue
            # The gap-category note IS the category description; carry the
            # root_cause so ocr_clean classification can use it.
            note = f"root_cause={root}; {gc.get('description','')[:240]}"
            rows.append(
                {
                    "page": int(m.group("page")),
                    "jp": jp,
                    "our_en": our,
                    "human_en": human,
                    "category": normalize_category(cat),
                    "severity": int(round(float(sev))) if sev is not None else 0,
                    "judge_note": note,
                    "judge_ocr_conf": None,
                    "source_field": "gap_examples",
                    "root_cause": root,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Join back to bubbles.json
# ---------------------------------------------------------------------------


def _load_page_bubbles(bubbles_root: Path, page: int) -> list[dict[str, Any]]:
    p = bubbles_root / f"{page:03d}" / "bubbles.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except Exception:
        return []
    if isinstance(data, list):
        return data
    # tolerate {"bubbles": [...]} shape
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def _match_bubble(
    bubbles: list[dict[str, Any]], jp: str
) -> dict[str, Any] | None:
    """Find the bubble whose ocr_jp best matches ``jp``.

    Exact match first; then substring containment either way (the judge
    occasionally trims trailing particles); finally None.
    """
    for b in bubbles:
        if (b.get("ocr_jp") or "").strip() == jp:
            return b
    for b in bubbles:
        oj = (b.get("ocr_jp") or "").strip()
        if oj and (jp in oj or oj in jp):
            return b
    return None


def build_rows(
    comparison: dict[str, Any], bubbles_root: Path
) -> list[dict[str, Any]]:
    # worst_issues first so they win the dedup over gap_examples (richer note +
    # real ocr_conf).  Dedup key is (page, jp): the same OCR line on a page is a
    # single gold row, and score_jsonl_metrics also collapses gold by jp.
    raw = parse_worst_issues(comparison) + parse_gap_examples(comparison)

    out: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()  # dedup on (page, jp)
    for r in raw:
        key = (r["page"], r["jp"])
        if key in seen:
            continue
        seen.add(key)

        page = r["page"]
        bubbles = _load_page_bubbles(bubbles_root, page)
        bub = _match_bubble(bubbles, r["jp"])
        idx = bub.get("idx") if bub else None
        ocr_conf = (
            bub.get("ocr_conf")
            if bub and bub.get("ocr_conf") is not None
            else r.get("judge_ocr_conf")
        )
        bbox = bub.get("bbox") if bub else None

        src_idx = f"idx{idx}" if idx is not None else "idxNA"
        ocr_clean = classify_ocr_clean(r["category"], r["judge_note"])

        row: dict[str, Any] = {
            "jp": r["jp"],
            "en": r["human_en"],
            "src": f"ikenie4:p{page:02d}:{src_idx}",
            "register_tag": "manga_nsfw",
            "category": r["category"],
            "severity": r["severity"],
            "ocr_clean": ocr_clean,
            "ocr_conf": ocr_conf,
            "bbox": bbox,
            "our_en": r["our_en"],
            "source_field": r["source_field"],
            "judge_note": r["judge_note"],
            "matched_bubble": bub is not None,
        }
        out.append(row)

    # Stable, deterministic ordering: by page then idx then jp.
    def _sort_key(row: dict[str, Any]) -> tuple:
        m = re.match(r"ikenie4:p(\d+):idx(\w+)", row["src"])
        pg = int(m.group(1)) if m else 999
        ix = m.group(2) if m else "zz"
        ixn = int(ix) if ix.isdigit() else 999
        return (pg, ixn, row["jp"])

    out.sort(key=_sort_key)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_COMPARISON = (
    "/tmp/claude-1000/-home-danny-Documents-personal-extension/"
    "bed1300f-1944-4e45-8974-0e884ae39ae0/tasks/wdcumj4jp.output"
)
DEFAULT_BUBBLES_ROOT = (
    "/home/danny/Documents/personal/extension/backend/.bench/ikenie4_final_insp"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", default=DEFAULT_COMPARISON,
                    help="24-agent comparison .output JSON")
    ap.add_argument("--bubbles-root", default=DEFAULT_BUBBLES_ROOT,
                    help="Root holding <NNN>/bubbles.json for each page")
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parent / "data" / "ikenie4" / "gold.jsonl"
        ),
    )
    args = ap.parse_args(argv)

    comp_path = Path(args.comparison)
    if not comp_path.exists():
        sys.stderr.write(f"ERROR: comparison not found: {comp_path}\n")
        return 2
    comparison = json.loads(comp_path.read_text())

    rows = build_rows(comparison, Path(args.bubbles_root))
    if not rows:
        sys.stderr.write("ERROR: no gold rows built\n")
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    n_clean = sum(1 for r in rows if r["ocr_clean"])
    n_matched = sum(1 for r in rows if r["matched_bubble"])
    cats: dict[str, int] = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(f"wrote {out_path}")
    print(f"  total rows      : {n}")
    print(f"  ocr_clean=true  : {n_clean}  (scored by chrF++/COMET)")
    print(f"  ocr_clean=false : {n - n_clean}  (excluded from neural metrics)")
    print(f"  matched bubble  : {n_matched}/{n}")
    print(f"  categories      : "
          + ", ".join(f"{k}={v}" for k, v in sorted(cats.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())

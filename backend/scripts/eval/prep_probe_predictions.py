#!/usr/bin/env python
"""Join a predictions source to a probe-case fixture and emit rows that
``probes.py`` can consume directly (each row carries ``probe_type`` + the
per-row config + ``en_pred``).

Two input modes:

``--inspect-dir`` (PREFERRED, page-scoped)
    A run's inspect dir with per-page ``<NNN>/bubbles.json``.  Each probe row
    carries a ``src`` like ``"ikenie4:p22:idx_suck"`` (page parsed from it) and
    a ``jp`` (the intended clean source).  We restrict the candidate bubbles to
    the probe's PAGE, then pick the bubble whose ``ocr_jp`` best matches the
    probe ``jp`` (exact, then containment either direction, then char-overlap).
    Page-scoping keeps the probe pinned to its panel even when OCR garbles the
    text or reorders bubbles -- the same stability the bbox join gives the gold
    set.  (Probe rows have no bbox, so jp-on-page is the best available key.)

``--predictions`` (LEGACY, global jp-join)
    A flat predictions.jsonl ({jp, en}); matched by global jp containment.
    UNSTABLE across OCR text changes -- kept for back-compat.

probe-case rows: {probe_type, src, jp, [banned_en_substrings,
required_en_substrings, referent], ...}.

Output rows: probe-case row + ``en_pred`` from the matched prediction.  A probe
case with no matching prediction on its page is DROPPED (can't score it).

Usage
-----
    python scripts/eval/prep_probe_predictions.py \
        --inspect-dir .bench/ikenie4_merged_insp \
        --probe-cases scripts/eval/data/ikenie4/probes.jsonl \
        --out /tmp/probe_preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_SRC_RE = re.compile(r"^ikenie4:p(\d+):")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _parse_page(src: str) -> int | None:
    m = _SRC_RE.match(src or "")
    return int(m.group(1)) if m else None


def _char_overlap(a: str, b: str) -> float:
    """Jaccard over character sets -- a cheap fuzzy score for garbled OCR."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _best_pred(pred_by_jp: dict[str, str], jp: str) -> str | None:
    """Exact jp match first, then containment either direction (global)."""
    if jp in pred_by_jp:
        return pred_by_jp[jp]
    for pjp, pen in pred_by_jp.items():
        if pjp and (jp in pjp or pjp in jp):
            return pen
    return None


def _best_pred_on_page(
    bubbles: list[dict[str, Any]], jp: str, pred_key: str
) -> str | None:
    """Pick the on-page bubble whose ocr_jp best matches the probe jp.

    Precedence: exact -> containment either way -> max char-overlap (>= 0.34).
    """
    if not bubbles:
        return None
    # 1. exact
    for b in bubbles:
        if (b.get("ocr_jp") or "").strip() == jp:
            return (b.get(pred_key) or "").strip()
    # 2. containment either direction
    for b in bubbles:
        bjp = (b.get("ocr_jp") or "").strip()
        if bjp and (jp in bjp or bjp in jp):
            return (b.get(pred_key) or "").strip()
    # 3. fuzzy char-overlap
    best_b: dict[str, Any] | None = None
    best_s = 0.0
    for b in bubbles:
        bjp = (b.get("ocr_jp") or "").strip()
        s = _char_overlap(jp, bjp)
        if s > best_s:
            best_s, best_b = s, b
    if best_b is not None and best_s >= 0.34:
        return (best_b.get(pred_key) or "").strip()
    return None


def _bubbles_for_page(inspect_dir: Path, page: int) -> list[dict[str, Any]]:
    fp = inspect_dir / f"{page:03d}" / "bubbles.json"
    if not fp.exists():
        return []
    try:
        data = json.loads(fp.read_text())
    except Exception:
        return []
    return data if isinstance(data, list) else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src_grp = ap.add_mutually_exclusive_group(required=True)
    src_grp.add_argument(
        "--inspect-dir",
        help="run inspect dir with per-page <NNN>/bubbles.json (page-scoped join)",
    )
    src_grp.add_argument(
        "--predictions", help="flat predictions.jsonl (legacy global jp-join)"
    )
    ap.add_argument("--probe-cases", required=True)
    ap.add_argument(
        "--pred-key",
        default="translation_en",
        help="bubble field with the translation (inspect-dir mode); "
        "for --predictions legacy mode pass 'en'",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cases = _read_jsonl(Path(args.probe_cases))
    joined: list[dict[str, Any]] = []
    n_unmatched = 0

    if args.inspect_dir:
        # STABLE probe set: emit ALL probe cases every run, so before/after is
        # scored on the SAME rows.  An unmatched probe -> en_pred="" (a genuine
        # "we produced nothing for this panel" fail), NOT a dropped row -- that
        # keeps the per-category n identical across runs (no apples-to-oranges).
        inspect_dir = Path(args.inspect_dir)
        page_cache: dict[int, list[dict[str, Any]]] = {}
        for c in cases:
            jp = (c.get("jp") or "").strip()
            page = _parse_page(c.get("src", ""))
            en_pred: str | None = None
            if page is not None:
                if page not in page_cache:
                    page_cache[page] = _bubbles_for_page(inspect_dir, page)
                en_pred = _best_pred_on_page(page_cache[page], jp, args.pred_key)
            if en_pred is None:
                n_unmatched += 1
                en_pred = ""
            row = dict(c)
            row["en_pred"] = en_pred
            joined.append(row)
    else:
        preds = _read_jsonl(Path(args.predictions))
        # in legacy mode pred-key defaults to translation_en but flat preds use
        # 'en'; fall back to 'en' if translation_en isn't present.
        pk = args.pred_key
        if preds and pk not in preds[0] and "en" in preds[0]:
            pk = "en"
        pred_by_jp: dict[str, str] = {}
        for p in preds:
            jp = (p.get("jp") or "").strip()
            if jp:
                pred_by_jp[jp] = (p.get(pk) or "").strip()
        for c in cases:
            jp = (c.get("jp") or "").strip()
            en_pred = _best_pred(pred_by_jp, jp)
            if en_pred is None:
                n_unmatched += 1
                continue
            row = dict(c)
            row["en_pred"] = en_pred
            joined.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in joined:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"prep_probe_predictions: {len(joined)} probe rows joined, "
        f"{n_unmatched} unmatched (dropped) -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

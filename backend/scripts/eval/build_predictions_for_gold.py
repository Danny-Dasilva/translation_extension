#!/usr/bin/env python
"""Build a predictions JSONL that aligns 1:1 with the Ikenie4 gold set BY THE
STABLE SPATIAL KEY (bbox IoU on the same page) instead of by the Japanese OCR
text.

WHY
---
The old harness joined predictions to gold by the ``jp`` field.  But our OCR
*changes* the ``jp`` text between runs, so a jp-join scores a DIFFERENT subset
of gold rows for every run (e.g. a baseline matched 40/77 gold rows; an
OCR-improved run matched 61/77 -- but a *different* 61).  The chrF "Δ" and
the probes were then computed on disjoint row sets -> apples-to-oranges, an
INVALID before/after gauge.

This script joins on the bbox instead, which is stable across OCR text changes
*and* reading-order changes: the run's per-page ``bubbles.json`` may emit
bubbles in a different ``idx`` order, but each gold row is matched to the
bubble with the MAX bbox IoU on the SAME page (require IoU >= threshold).

INPUT
-----
* ``--inspect-dir``  a run's inspect dir, with per-page ``<NNN>/bubbles.json``
  (zero-padded 3-digit page folder; each bubble has ``bbox`` + ``ocr_jp`` +
  ``translation_en``).  e.g. ``.bench/ikenie4_merged_insp``.
* ``--gold``         gold.jsonl; each row has a STABLE ``src`` like
  ``"ikenie4:p05:idx0"`` (page parsed from here) AND a ``bbox`` + ``jp`` + ``en``.

PAGE NUMBERS
------------
Our-run page numbers == gold ``src`` page numbers (both derive from the 583875
raws).  The p41 GT-image offset is ONLY for ground-truth IMAGE alignment, NOT
for our-run bubbles -- do NOT apply it here.

OUTPUT
------
One row per gold row, KEYED BY THE GOLD ``src`` so every run scores the SAME
77 gold rows::

    {src, jp (=our ocr_jp), en (=our translation_en), matched (bool), iou,
     gold_jp (=gold.jp), gold_en (=gold.en), ocr_clean (passthrough),
     <other gold passthrough fields>}

Unmatched gold rows (no overlapping bubble on the page, or page missing) ->
``matched=false``, ``en=""``, ``jp=""``, ``iou=0.0``.

Usage
-----
    python scripts/eval/build_predictions_for_gold.py \
        --inspect-dir .bench/ikenie4_merged_insp \
        --gold scripts/eval/data/ikenie4/gold.jsonl \
        --out  scorecards/ikenie4/preds_for_gold_merged.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_SRC_RE = re.compile(r"^[^:]+:p(\d+):idx(\d+)")  # any chapter slug (ikenie4, ikenie5, …)

# Gold fields to pass straight through onto the prediction row so downstream
# scoring/probes keep their per-row config (probe_type, ocr_clean, category…).
_PASSTHROUGH = (
    "ocr_clean",
    "ocr_conf",
    "register_tag",
    "category",
    "severity",
    "probe_type",
    "banned_en_substrings",
    "required_en_substrings",
    "referent",
)


def iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Intersection-over-union of two ``{minX,minY,maxX,maxY}`` boxes."""
    ax0, ay0, ax1, ay1 = a["minX"], a["minY"], a["maxX"], a["maxY"]
    bx0, by0, bx1, by1 = b["minX"], b["minY"], b["maxX"], b["maxY"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def parse_page(src: str) -> int | None:
    """``"ikenie4:p05:idx0"`` -> ``5``; non-matching src -> ``None``."""
    m = _SRC_RE.match(src or "")
    return int(m.group(1)) if m else None


def best_match_bubble(
    gold_box: dict[str, Any], bubbles: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, float]:
    """Return ``(bubble, iou)`` for the bubble with the max IoU vs ``gold_box``.

    Returns ``(None, 0.0)`` when ``bubbles`` is empty.  The caller applies the
    IoU threshold: a returned bubble with iou below threshold means *no match*.
    """
    best_bub: dict[str, Any] | None = None
    best_iou = 0.0
    for b in bubbles:
        box = b.get("bbox")
        if not box:
            continue
        v = iou(gold_box, box)
        if v > best_iou:
            best_iou = v
            best_bub = b
    return best_bub, best_iou


def _load_bubbles(inspect_dir: Path, page: int) -> list[dict[str, Any]]:
    """Load ``<inspect_dir>/<page:03d>/bubbles.json`` (missing -> [])."""
    fp = inspect_dir / f"{page:03d}" / "bubbles.json"
    if not fp.exists():
        return []
    try:
        data = json.loads(fp.read_text())
    except Exception:
        return []
    return data if isinstance(data, list) else []


def build_rows(
    gold: list[dict[str, Any]],
    inspect_dir: Path,
    iou_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """Join each gold row to its best-IoU bubble; emit one row per gold row."""
    inspect_dir = Path(inspect_dir)
    # cache bubbles per page so we read each page file once
    cache: dict[int, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for g in gold:
        src = g.get("src", "")
        page = parse_page(src)
        gold_box = g.get("bbox")
        row: dict[str, Any] = {
            "src": src,
            "gold_jp": (g.get("jp") or "").strip(),
            "gold_en": (g.get("en") or "").strip(),
        }
        for k in _PASSTHROUGH:
            if k in g:
                row[k] = g[k]

        if page is None or not gold_box:
            row.update(matched=False, iou=0.0, jp="", en="")
            rows.append(row)
            continue

        if page not in cache:
            cache[page] = _load_bubbles(inspect_dir, page)
        bub, score = best_match_bubble(gold_box, cache[page])

        if bub is not None and score >= iou_threshold:
            row.update(
                matched=True,
                iou=round(score, 4),
                jp=(bub.get("ocr_jp") or "").strip(),
                en=(bub.get("translation_en") or "").strip(),
            )
        else:
            row.update(matched=False, iou=round(score, 4), jp="", en="")
        rows.append(row)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inspect-dir",
        required=True,
        help="run inspect dir with per-page <NNN>/bubbles.json",
    )
    ap.add_argument("--gold", required=True, help="gold.jsonl (rows have src+bbox+jp+en)")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="min bbox IoU to count a gold row as matched (default 0.3)",
    )
    args = ap.parse_args(argv)

    gold = _read_jsonl(Path(args.gold))
    rows = build_rows(gold, Path(args.inspect_dir), iou_threshold=args.iou_threshold)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_matched = sum(1 for r in rows if r["matched"])
    print(
        f"build_predictions_for_gold: {n_matched}/{len(rows)} gold rows matched "
        f"(IoU>={args.iou_threshold}) from {args.inspect_dir} -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

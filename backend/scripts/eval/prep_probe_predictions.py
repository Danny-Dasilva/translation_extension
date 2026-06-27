#!/usr/bin/env python
"""Join a predictions.jsonl to a probe-case fixture by ``jp`` and emit rows that
``probes.py`` can consume directly (each row carries ``probe_type`` + the
per-row config + ``en_pred``).

predictions.jsonl rows: {jp, en} (the bubbles.json -> {jp,en} shape).
probe-case rows: {probe_type, jp, [banned_en_substrings, required_en_substrings,
referent], ...}.

Output rows: probe-case row + ``en_pred`` from the matched prediction.  A probe
case with no matching prediction is DROPPED (can't score it).

Usage
-----
    PYTHONPATH=. python backend/scripts/eval/prep_probe_predictions.py \
        --predictions preds.jsonl \
        --probe-cases backend/scripts/eval/data/ikenie4/probes.jsonl \
        --out /tmp/probe_preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _best_pred(pred_by_jp: dict[str, str], jp: str) -> str | None:
    """Exact jp match first, then containment either direction."""
    if jp in pred_by_jp:
        return pred_by_jp[jp]
    for pjp, pen in pred_by_jp.items():
        if pjp and (jp in pjp or pjp in jp):
            return pen
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--probe-cases", required=True)
    ap.add_argument("--pred-key", default="en")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    preds = _read_jsonl(Path(args.predictions))
    cases = _read_jsonl(Path(args.probe_cases))

    pred_by_jp: dict[str, str] = {}
    for p in preds:
        jp = (p.get("jp") or "").strip()
        if jp:
            pred_by_jp[jp] = (p.get(args.pred_key) or "").strip()

    joined: list[dict[str, Any]] = []
    n_unmatched = 0
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

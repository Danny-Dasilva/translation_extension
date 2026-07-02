#!/usr/bin/env python
"""One-off adapter: ikenie4 preds_for_gold_*.jsonl -> gold-shaped flat JSONL
for the page-adequacy judge (gate signal #4).

The ikenie4 fair preds files (scripts/eval/scorecards/ikenie4/
preds_for_gold_{v1,v11fix8}_fair.jsonl) are flat rows keyed by ``src``
("ikenie4:pNN:idxK") carrying:

    gold_jp  -- clean human JP transcription
    gold_en  -- gold EN reference
    jp       -- OCR JP (often garbled; NOT used here)
    en       -- the model's candidate EN (the judge reads this via
                --candidates, whose loader accepts ``en`` as an alias)

``page_adequacy_judge.build_pages_from_gold`` wants a gold file whose rows are
``{"src": ..., "jp": <clean JP>, "en": <gold EN>}``. This script emits exactly
that, restricted to a deterministic page sample:

  * rows with empty gold_jp or gold_en are skipped (none exist today);
  * duplicate ``src`` keys (3 in the file: p07:idx7, p19:idx1, p23:idx5 --
    two bubbles collapsed onto one key) keep the FIRST occurrence, which is
    identical treatment for both models since gold fields match across files;
  * pages are sorted NUMERICALLY by page number and ``n_sample`` pages are
    picked at evenly spaced indices (round(i*(P-1)/(n-1)) for i in 0..n-1),
    so the sample spans the whole chapter and is identical for both models.

Usage (from backend/):
    .venv/bin/python scripts/eval/build_ikenie4_adequacy_pages.py \
        --preds scripts/eval/scorecards/ikenie4/preds_for_gold_v1_fair.jsonl \
        --out scripts/eval/scorecards/ikenie4/adequacy_gold_sample35.jsonl \
        --n-sample 35
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

_PAGE_NUM_RE = re.compile(r":p(\d+):")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", type=Path, required=True,
                    help="A preds_for_gold_*_fair.jsonl (gold fields are "
                         "identical across the two model files).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-sample", type=int, default=35)
    args = ap.parse_args()

    rows_by_src: OrderedDict[str, dict] = OrderedDict()
    n_skipped_empty = n_dup = 0
    with args.preds.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            src = str(row["src"])
            if not (row.get("gold_jp") or "").strip() or not (
                row.get("gold_en") or ""
            ).strip():
                n_skipped_empty += 1
                continue
            if src in rows_by_src:
                n_dup += 1
                continue  # keep first occurrence
            rows_by_src[src] = row

    # Group srcs by page id, sort pages numerically.
    pages: dict[str, list[str]] = {}
    for src in rows_by_src:
        page_id = ":".join(src.split(":")[:2])
        pages.setdefault(page_id, []).append(src)

    def page_num(pid: str) -> int:
        m = re.search(r"p(\d+)$", pid)
        assert m, pid
        return int(m.group(1))

    sorted_pages = sorted(pages, key=page_num)
    p = len(sorted_pages)
    n = min(args.n_sample, p)
    picked_idx = sorted({round(i * (p - 1) / (n - 1)) for i in range(n)}) if n > 1 else [0]
    picked = [sorted_pages[i] for i in picked_idx]

    n_rows = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for pid in picked:
            for src in pages[pid]:
                row = rows_by_src[src]
                fh.write(json.dumps(
                    {"src": src, "jp": row["gold_jp"], "en": row["gold_en"]},
                    ensure_ascii=False) + "\n")
                n_rows += 1

    print(json.dumps({
        "total_pages": p,
        "sampled_pages": len(picked),
        "sampled_page_ids": picked,
        "rows_written": n_rows,
        "skipped_empty_gold": n_skipped_empty,
        "dup_src_dropped": n_dup,
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

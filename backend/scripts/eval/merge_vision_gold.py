#!/usr/bin/env python
"""Merge the one-time vision-gold rows into the frozen Ikenie4 gold set.

POLICY
------
* KEEP every judge-seeded row (``source_field in {worst_issues, gap_examples}``)
  verbatim -- they carry verified human_en + rich category/severity/judge_note.
* ADD a vision-gold row (``source_field == vision_gt``) ONLY for a ``src`` that
  is NOT already covered by a judge row.  The judge rows win on overlap.
* The seed file is backed up to ``gold.seed77.jsonl`` before the merged set is
  written back to ``gold.jsonl`` (idempotent: a re-run re-derives from the seed
  backup if present, so re-running never double-merges).

Usage
-----
    python scripts/eval/merge_vision_gold.py \
        --seed   scripts/eval/data/ikenie4/gold.jsonl \
        --vision scripts/eval/data/ikenie4/gold_full.jsonl \
        --backup scripts/eval/data/ikenie4/gold.seed77.jsonl \
        --out    scripts/eval/data/ikenie4/gold.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR / "data" / "ikenie4"


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", default=str(DATA / "gold.jsonl"))
    ap.add_argument("--vision", default=str(DATA / "gold_full.jsonl"))
    ap.add_argument("--backup", default=str(DATA / "gold.seed77.jsonl"))
    ap.add_argument("--out", default=str(DATA / "gold.jsonl"))
    args = ap.parse_args(argv)

    seed_path = Path(args.seed)
    backup_path = Path(args.backup)

    # Idempotency: if a seed backup already exists, the seed file may already be
    # a merged set -- re-derive the judge rows from the backup instead.
    if backup_path.exists():
        judge_rows = _read_jsonl(backup_path)
        print(f"[merge] reusing existing seed backup ({len(judge_rows)} rows)")
    else:
        judge_rows = _read_jsonl(seed_path)
        backup_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in judge_rows) + "\n",
            encoding="utf-8",
        )
        print(f"[merge] backed up {len(judge_rows)} seed rows -> {backup_path.name}")

    vision_rows = _read_jsonl(Path(args.vision))

    judge_srcs = {r.get("src", "") for r in judge_rows}
    merged: list[dict[str, Any]] = list(judge_rows)
    added = 0
    skipped_overlap = 0
    for r in vision_rows:
        src = r.get("src", "")
        if src in judge_srcs:
            skipped_overlap += 1
            continue
        merged.append(r)
        added += 1

    out_path = Path(args.out)
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n",
        encoding="utf-8",
    )

    clean = sum(1 for r in merged if r.get("ocr_clean") is True)
    print(
        f"[merge] judge={len(judge_rows)} + vision_added={added} "
        f"(skipped {skipped_overlap} overlapping src) = {len(merged)} total"
    )
    print(f"[merge] ocr_clean rows: {clean}/{len(merged)}")
    print(f"[merge] wrote -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Quality spot-check: llama.cpp Q4_K_M v9c output vs Unsloth (unmerged) v9c.

Reads a llama.cpp bench `outputs.jsonl` (from bench_llamacpp_v9c.py) and the
Unsloth baseline preds_open_mantra_test_clean.jsonl, aligns by `jp` string,
and prints a side-by-side table for the first N items.

Usage:
    uv run python backend/scripts/eval/compare_llamacpp_vs_unsloth.py \
        --llamacpp backend/scripts/eval/bench_out/q4km-fa-vanilla/outputs.jsonl \
        --unsloth backend/training/runs/manga-bubbles/preds_open_mantra_test_clean.jsonl \
        --n 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_by_jp(path: Path, key_pred: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            jp = row.get("jp")
            if jp:
                row["__pred__"] = row.get(key_pred, "")
                out[jp] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llamacpp", required=True, help="bench_llamacpp_v9c outputs.jsonl")
    ap.add_argument("--unsloth", required=True, help="preds_open_mantra_test_clean.jsonl from Unsloth run")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--md-out", default=None, help="optional markdown output path")
    args = ap.parse_args()

    llama_rows = _load_by_jp(Path(args.llamacpp), key_pred="en_pred")
    unsl_rows = _load_by_jp(Path(args.unsloth), key_pred="en_pred")

    common = [jp for jp in llama_rows if jp in unsl_rows]
    if not common:
        print("ERROR: no overlapping `jp` strings between the two files", file=sys.stderr)
        return 1

    print(f"compared {len(common)} prompts (showing first {min(args.n, len(common))})")
    md_lines: list[str] = ["| # | JP | EN ref | llama.cpp Q4_K_M | Unsloth bf16 |", "|---|----|--------|------------------|--------------|"]
    print()
    for i, jp in enumerate(common[: args.n], 1):
        l = llama_rows[jp]
        u = unsl_rows[jp]
        en_ref = l.get("en_ref") or u.get("en_ref") or ""
        ll = l["__pred__"].replace("\n", " ").strip()
        un = u["__pred__"].replace("\n", " ").strip()
        ref = (en_ref or "").replace("\n", " ").strip()
        print(f"[{i}] JP : {jp}")
        print(f"    REF: {ref}")
        print(f"    LCP: {ll}")
        print(f"    UNS: {un}")
        print()
        md_lines.append(f"| {i} | {jp} | {ref} | {ll} | {un} |")

    if args.md_out:
        Path(args.md_out).write_text("\n".join(md_lines), encoding="utf-8")
        print(f"wrote markdown table to {args.md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

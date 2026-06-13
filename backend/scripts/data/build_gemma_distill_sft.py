"""Convert the Gemma teacher cache into an SFT parquet for v8 distillation.

Reads the (jp, gemma) pairs from the CPO pilot phase1 jsonl, filters for
quality (length, garbage), emits a unified-schema parquet usable as a new
gold source in compose_training_mix.py.

This is the *robust* alternative to CPO when the chosen-vs-rejected length
imbalance is too large (CPO sequence-logp is dominated by length).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "data"))
from unify_schema import make_row, write_parquet  # noqa: E402


def is_quality(jp: str, en: str) -> bool:
    if not jp or not en:
        return False
    en = en.strip()
    # garbage refusal — keep these as garbage examples, not manga gold
    if en in ("...", "…"):
        return False
    if len(en) < 2 or len(en) > 400:
        return False
    # too short JP → noise
    ja_chars = re.findall(r"[぀-ヿ一-鿿]", jp)
    if len(ja_chars) < 1:
        return False
    if len(ja_chars) / max(1, len(jp)) < 0.3:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="backend/training/datasets/filtered/cpo_phase1_pilot.jsonl")
    ap.add_argument("--out", default="backend/training/datasets/filtered/gemma_distill_v8.parquet")
    args = ap.parse_args()

    rows = []
    n_total = 0
    n_kept = 0
    n_garbage = 0
    with open(args.cache) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_total += 1
            jp = r.get("jp", "").strip()
            en = r.get("gemma", "").strip()
            if en in ("...", "…"):
                # Garbage refusal — emit as garbage register so model learns refusal
                if jp:
                    n_garbage += 1
                    rows.append(make_row(
                        jp=jp, en="...", src=f"gemma31_distill:{r.get('src','?')}:garbage",
                        register_tag="garbage", gold_flag=True,
                    ))
                continue
            if not is_quality(jp, en):
                continue
            n_kept += 1
            rows.append(make_row(
                jp=jp, en=en,
                src=f"gemma31_distill:{r.get('src','?')}",
                register_tag="manga",
                gold_flag=True,
            ))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_parquet(iter(rows), Path(args.out))
    print(f"total {n_total} → kept {n_kept} manga + {n_garbage} garbage = {len(rows)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

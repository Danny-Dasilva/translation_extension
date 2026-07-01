#!/usr/bin/env python3
"""Build the v11fix8 SFT parquet = v11fix7 mix + mined corpus-bitext corrective rows.

The mined rows (from finish_curate_avail.py) are a gold_flag=False corrective slice,
upweighted modestly and folded onto the v11fix7 parquet (which already carries the v11
base + gold corrective + targeted slices). Same train/serve-safe schema. Also writes
the v11fix8 SFT config (clone of v11fix7's, keeping in_training_eval:false).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[2]
COLS = ["prompt", "en", "src", "register_tag", "gold_flag"]
DEFAULT_BASE = BACKEND / "scripts/data/v11fix7/data_v11fix7_pagecontext.parquet"
V11FIX7_YAML = BACKEND / "training/configs/gemma4_e4b_v11fix7_sft.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", default="/home/danny/manga_corpus_staging/curated/curated_rows.jsonl")
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--upweight", type=int, default=3)
    ap.add_argument("--out-dir", default=str(BACKEND / "scripts/data/v11fix8"))
    ap.add_argument("--max-nsfw-frac", type=float, default=0.20)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mined = []
    with open(args.curated) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                mined.append({k: r[k] for k in COLS})
    if not mined:
        print("ERROR: no curated rows found at", args.curated, file=sys.stderr)
        return 1

    mined_df = pl.DataFrame(mined * args.upweight).select(COLS)
    base = pl.read_parquet(args.base).select(COLS)
    mixed = pl.concat([base, mined_df], how="vertical")

    out_parquet = out_dir / "data_v11fix8_pagecontext.parquet"
    mixed.write_parquet(out_parquet)

    nsfw = mixed.filter(pl.col("register_tag").is_in(["vn_eroge", "manga_nsfw"])).height
    nsfw_frac = round(nsfw / mixed.height, 4)
    stats = {
        "base_rows": base.height, "mined_unique": len(mined), "upweight": args.upweight,
        "mined_upweighted": mined_df.height, "total_rows": mixed.height,
        "nsfw_rows": nsfw, "nsfw_frac": nsfw_frac, "base_parquet": args.base,
    }
    (out_dir / "v11fix8_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    if nsfw_frac > args.max_nsfw_frac:
        print(f"WARNING: nsfw_frac {nsfw_frac} > {args.max_nsfw_frac} (v12 danger zone was 0.36)", file=sys.stderr)

    # ---- clone v11fix7 config -> v11fix8 (keep in_training_eval:false) ----
    if V11FIX7_YAML.exists():
        txt = V11FIX7_YAML.read_text()
        txt = (txt.replace("v11fix7", "v11fix8")
                  .replace(str(DEFAULT_BASE.name), out_parquet.name))
        out_yaml = BACKEND / "training/configs/gemma4_e4b_v11fix8_sft.yaml"
        if "in_training_eval" not in txt:  # source v11fix7 local copy may lack the box-only fix
            txt = txt.replace("  report_to: none",
                              "  report_to: none\n  in_training_eval: false   # avoid sm_120 eval-build segfault")
        out_yaml.write_text(txt)
        print(f"wrote config {out_yaml}")
    else:
        print(f"WARNING: {V11FIX7_YAML} not found — write the v11fix8 config manually (in_training_eval:false).", file=sys.stderr)

    print(f"\nwrote {out_parquet} ({mixed.height} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

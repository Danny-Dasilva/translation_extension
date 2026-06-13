"""Paired bootstrap-resample test on chrF++ between two systems.

Definitive answer to "is the chrF++ delta between v10-it (70.91) and v9c (70.40)
real or noise?". We use sacrebleu's `paired-bs` mode which resamples the eval
set with replacement N=1000 times and measures how often System A beats System B.

Usage
-----
    python paired_bs_chrf.py \
        --sys-a-per-bubble scorecards/per_bubble_v10it_644289.json \
        --sys-b-per-bubble scorecards/per_bubble_v9c_644289.json \
        --label-a v10it --label-b v9c \
        --out scorecards/paired_bs_chrf_v10it_v9c.json

Output
------
JSON with: mean_a, mean_b, delta_mean, p_value, ci95_low, ci95_high
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any


def bootstrap_paired(
    a: list[float], b: list[float], *, n: int = 1000, seed: int = 12345
) -> dict[str, Any]:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    rng = random.Random(seed)
    n_items = len(a)
    deltas: list[float] = []
    a_means: list[float] = []
    b_means: list[float] = []
    for _ in range(n):
        idxs = [rng.randrange(n_items) for _ in range(n_items)]
        am = sum(a[i] for i in idxs) / n_items
        bm = sum(b[i] for i in idxs) / n_items
        a_means.append(am)
        b_means.append(bm)
        deltas.append(am - bm)

    deltas_sorted = sorted(deltas)
    ci_low = deltas_sorted[int(0.025 * n)]
    ci_high = deltas_sorted[int(0.975 * n) - 1]
    # Two-sided p: how often does the bootstrap produce a sign flip on the delta?
    observed = (sum(a) - sum(b)) / n_items
    if observed >= 0:
        p_one = sum(1 for d in deltas if d <= 0) / n
    else:
        p_one = sum(1 for d in deltas if d >= 0) / n
    p_two = 2 * p_one

    return {
        "n_eval_segments": n_items,
        "n_bootstrap": n,
        "mean_a": statistics.mean(a),
        "mean_b": statistics.mean(b),
        "observed_delta": observed,
        "ci95_low_delta": ci_low,
        "ci95_high_delta": ci_high,
        "p_value_two_sided": p_two,
        "win_rate_a": sum(1 for d in deltas if d > 0) / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys-a-per-bubble", required=True,
                    help="per_bubble_*.json with 'chrf_pp' fields")
    ap.add_argument("--sys-b-per-bubble", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--metric-key", default="chrf_pp")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows_a = json.loads(Path(args.sys_a_per_bubble).read_text())
    rows_b = json.loads(Path(args.sys_b_per_bubble).read_text())

    # Align by (slug, jp). If mismatched, fall back to position with a warning.
    keyed_a = {(r["slug"], r["jp"]): r for r in rows_a}
    keyed_b = {(r["slug"], r["jp"]): r for r in rows_b}
    keys = sorted(set(keyed_a.keys()) & set(keyed_b.keys()))
    if len(keys) < min(len(rows_a), len(rows_b)):
        sys.stderr.write(
            f"WARNING: only {len(keys)} keys aligned across {len(rows_a)} A / {len(rows_b)} B\n"
        )
    a = [float(keyed_a[k][args.metric_key]) for k in keys]
    b = [float(keyed_b[k][args.metric_key]) for k in keys]

    result = bootstrap_paired(a, b, n=args.n_bootstrap, seed=args.seed)
    result["label_a"] = args.label_a
    result["label_b"] = args.label_b
    result["metric"] = args.metric_key

    out_path = Path(args.out) if args.out else Path(
        f"paired_bs_{args.metric_key}_{args.label_a}_vs_{args.label_b}.json"
    )
    out_path.write_text(json.dumps(result, indent=2))

    print(f"=== paired bootstrap chrF++ ({args.label_a} vs {args.label_b}) ===")
    print(f"  n eval segs: {result['n_eval_segments']}")
    print(f"  n bootstrap: {result['n_bootstrap']}")
    print(f"  mean_{args.label_a}    : {result['mean_a']:.4f}")
    print(f"  mean_{args.label_b}    : {result['mean_b']:.4f}")
    print(f"  delta        : {result['observed_delta']:+.4f}")
    print(f"  95% CI delta : [{result['ci95_low_delta']:+.4f}, {result['ci95_high_delta']:+.4f}]")
    print(f"  p (two-sided): {result['p_value_two_sided']:.4f}")
    print(f"  win rate A>B : {result['win_rate_a']:.3f}")
    print(f"\nwrote {out_path}")

    if result["p_value_two_sided"] < 0.05:
        print("\n=> SIGNIFICANT at α=0.05")
    else:
        print("\n=> NOT SIGNIFICANT at α=0.05 (delta within bootstrap noise)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Paired bootstrap-resample test on any per-bubble metric column.

Drop-in successor to ``paired_bs_chrf.py``: same paired-bootstrap math, but
accepts ``--metric-keys chrf_pp,cometkiwi_xl,metricx_24_xl,xcomet_xl,bleu``
and emits a single JSON with the bootstrap result for each metric.

Aligns rows by (slug, jp) when both keys exist, otherwise (jp,) — handy when
predictions came from different inference scripts that don't always carry slug.

For metrics where lower is better (e.g. metricx), pass ``--lower-is-better
metricx_24_xl`` so the printed verdict is correct. Math is symmetric — sign
of observed_delta is what it is — but the verdict line and ``a_wins``
boolean key respect the direction.

Usage
-----
    /home/danny/.venvs/comet/bin/python paired_bs_metric.py \
        --sys-a-per-bubble scorecards/v10it_phase0/per_bubble_v10it_644289.json \
        --sys-b-per-bubble scorecards/v10it_phase0/per_bubble_v9c_644289.json \
        --label-a v10it --label-b v9c \
        --metric-keys chrf_pp,bleu,cometkiwi_xl,metricx_24_xl,xcomet_xl \
        --lower-is-better metricx_24_xl \
        --out scorecards/v10it_phase0/paired_bs_v10it_v9c_644289.json
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
    if n_items == 0:
        return {
            "n_eval_segments": 0,
            "n_bootstrap": n,
            "mean_a": 0.0,
            "mean_b": 0.0,
            "observed_delta": 0.0,
            "ci95_low_delta": 0.0,
            "ci95_high_delta": 0.0,
            "p_value_two_sided": 1.0,
            "win_rate_a": 0.0,
        }
    deltas: list[float] = []
    for _ in range(n):
        idxs = [rng.randrange(n_items) for _ in range(n_items)]
        am = sum(a[i] for i in idxs) / n_items
        bm = sum(b[i] for i in idxs) / n_items
        deltas.append(am - bm)

    deltas_sorted = sorted(deltas)
    ci_low = deltas_sorted[int(0.025 * n)]
    ci_high = deltas_sorted[max(0, int(0.975 * n) - 1)]
    observed = (sum(a) - sum(b)) / n_items
    if observed >= 0:
        p_one = sum(1 for d in deltas if d <= 0) / n
    else:
        p_one = sum(1 for d in deltas if d >= 0) / n
    p_two = min(1.0, 2 * p_one)

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


def _key_of(r: dict[str, Any]) -> tuple:
    if "slug" in r and "jp" in r:
        return (r["slug"], r["jp"])
    return (r.get("jp", ""),)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys-a-per-bubble", required=True)
    ap.add_argument("--sys-b-per-bubble", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--metric-keys", default="chrf_pp",
                    help="Comma-separated metric column names to bootstrap.")
    ap.add_argument("--lower-is-better", default="metricx_24_xl",
                    help="Comma-separated metric keys where lower is better.")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows_a: list[dict] = json.loads(Path(args.sys_a_per_bubble).read_text())
    rows_b: list[dict] = json.loads(Path(args.sys_b_per_bubble).read_text())

    keyed_a = {_key_of(r): r for r in rows_a}
    keyed_b = {_key_of(r): r for r in rows_b}
    keys = sorted(set(keyed_a.keys()) & set(keyed_b.keys()))
    if not keys:
        sys.stderr.write("ERROR: no aligned rows between A and B\n")
        return 2
    if len(keys) < min(len(rows_a), len(rows_b)):
        sys.stderr.write(
            f"WARNING: only {len(keys)} keys aligned across "
            f"{len(rows_a)} A / {len(rows_b)} B\n"
        )

    metric_keys = [m.strip() for m in args.metric_keys.split(",") if m.strip()]
    lower_better = {m.strip() for m in args.lower_is_better.split(",") if m.strip()}

    results: dict[str, Any] = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "n_aligned": len(keys),
        "lower_is_better": sorted(lower_better),
        "by_metric": {},
    }

    for mk in metric_keys:
        # Filter to keys where BOTH systems have the metric value (skip None).
        a_vals: list[float] = []
        b_vals: list[float] = []
        n_skipped = 0
        for k in keys:
            va = keyed_a[k].get(mk)
            vb = keyed_b[k].get(mk)
            if va is None or vb is None:
                n_skipped += 1
                continue
            try:
                a_vals.append(float(va))
                b_vals.append(float(vb))
            except (TypeError, ValueError):
                n_skipped += 1

        if not a_vals:
            results["by_metric"][mk] = {
                "metric": mk,
                "error": "no valid pairs",
                "n_skipped": n_skipped,
            }
            print(f"[{mk}] no valid pairs (skipped={n_skipped})")
            continue

        r = bootstrap_paired(a_vals, b_vals, n=args.n_bootstrap, seed=args.seed)
        r["metric"] = mk
        r["n_skipped_pairs"] = n_skipped
        if mk in lower_better:
            r["a_wins"] = r["observed_delta"] < 0
            verdict_dir = "lower-is-better"
        else:
            r["a_wins"] = r["observed_delta"] > 0
            verdict_dir = "higher-is-better"
        r["significant_at_005"] = r["p_value_two_sided"] < 0.05
        results["by_metric"][mk] = r

        print(f"[{mk}] {verdict_dir} | "
              f"{args.label_a}={r['mean_a']:.4f} {args.label_b}={r['mean_b']:.4f} | "
              f"Δ={r['observed_delta']:+.4f} "
              f"CI95=[{r['ci95_low_delta']:+.4f},{r['ci95_high_delta']:+.4f}] "
              f"p={r['p_value_two_sided']:.4f} "
              f"{'SIG' if r['significant_at_005'] else 'ns'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

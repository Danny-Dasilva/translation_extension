"""Consolidate all per-system scorecards into one ``score_summary_metrics_v2.json``.

Reads `<scorecards_dir>/score_summary_metrics_v2_<label>.json` for each label
and emits a single combined file with the structure the spec requested:

    {
      "metrics": ["chrF++", "BLEU", "CometKiwi-23-XL", "MetricX-24-Hybrid-XL", "teacher_fidelity"],
      "metric_notes": {...},
      "systems": {
        "v7":   {"chrF++": 71.10, ..., "n": 257, "eval_set": "644289"},
        "v9c":  {...},
        "v10it":{...},
        "v10it_openmantra_greedy": {...},
        "v10it_openmantra_bon_chrf": {...},
        ...
      }
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorecards-dir", default="backend/scripts/eval/scorecards")
    ap.add_argument("--out", default="backend/scripts/eval/score_summary_metrics_v2.json")
    ap.add_argument(
        "--labels",
        default="v7_644289,v9c_644289,v10it_644289,"
                "v10it_om_greedy,v10it_om_constrained,v10it_om_bon_chrf,v10it_om_bon_chrf_rag",
    )
    args = ap.parse_args()
    sd = Path(args.scorecards_dir)

    metric_notes = {
        "chrF++": "sacrebleu corpus_chrf, word_order=2. Reference-based; "
                  "for 644289 the reference is the Gemma-3-4B teacher modeA.jsonl, "
                  "for openmantra the reference is the human gold annotation.",
        "BLEU": "sacrebleu corpus_bleu (reference-based, same caveat as chrF).",
        "CometKiwi-23-XL": "Unbabel/wmt23-cometkiwi-da-xl. Reference-FREE QE — "
                           "this is the metric that actually correlates with human "
                           "judgment on out-of-domain text.",
        "MetricX-24-Hybrid-XL": "google/metricx-24-hybrid-xl-v2p6. Reference-based, "
                                "MQM-trained. Range 0..25, **lower is better**.",
        "teacher_fidelity": "% of preds that exact-match the Gemma-3-4B teacher "
                            "reference (RENAMED from gemma_em). NOT a translation "
                            "quality metric — measures distillation copy fidelity. "
                            "Only meaningful for the 644289 eval where the ref is "
                            "the Gemma teacher itself.",
        "_xcomet_xl": "REQUESTED: Unbabel/XCOMET-XL — gated for this HF account, "
                      "skipped. Both XCOMET-XL and XCOMET-XXL on Unbabel are gated "
                      "and access requests are pending.",
    }

    systems: dict[str, dict] = {}
    for label in args.labels.split(","):
        label = label.strip()
        if not label:
            continue
        path = sd / f"score_summary_metrics_v2_{label}.json"
        if not path.exists():
            print(f"WARN: {path} missing — skipping")
            continue
        d = json.loads(path.read_text())
        eval_set = "644289" if "644289" in label else (
            "openmantra-heldout" if "om_" in label or "openmantra" in label else "?"
        )
        systems[label] = {
            "chrF++": d.get("chrf_pp"),
            "BLEU": d.get("bleu"),
            "CometKiwi-23-XL": d.get("cometkiwi_xl_mean"),
            "MetricX-24-Hybrid-XL": d.get("metricx_24_xl_mean"),
            "teacher_fidelity_pct": d.get("teacher_fidelity", {}).get("exact_match_pct"),
            "n_aligned": d.get("n_aligned"),
            "empty_pct": d.get("empty_pct"),
            "jp_passthrough_pct": d.get("jp_passthrough_pct"),
            "eval_set": eval_set,
        }

    out = {
        "metrics": list(metric_notes.keys()),
        "metric_notes": metric_notes,
        "systems": systems,
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {op}")

    # Print a nice table
    print("\n" + "=" * 100)
    print(f"{'system':<32} {'eval_set':<22} {'chrF++':>7} {'BLEU':>6} {'Kiwi':>7} {'MetricX↓':>8} {'em%':>5}")
    print("-" * 100)
    for label, row in systems.items():
        chrf = f"{row['chrF++']:.2f}" if row.get('chrF++') is not None else "—"
        bleu = f"{row['BLEU']:.2f}" if row.get('BLEU') is not None else "—"
        kiwi = f"{row['CometKiwi-23-XL']:.4f}" if row.get('CometKiwi-23-XL') is not None else "—"
        mx = f"{row['MetricX-24-Hybrid-XL']:.3f}" if row.get('MetricX-24-Hybrid-XL') is not None else "—"
        em = f"{row['teacher_fidelity_pct']:.1f}" if row.get('teacher_fidelity_pct') is not None else "—"
        print(f"{label:<32} {row['eval_set']:<22} {chrf:>7} {bleu:>6} {kiwi:>7} {mx:>8} {em:>5}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

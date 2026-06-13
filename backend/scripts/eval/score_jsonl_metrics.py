"""Score a JSONL of {jp, en (=ref), pred (or candidates+best from inference_v10it_quality)}.

Companion to score_summary_metrics_v2.py, but for OpenMantra and similar
already-flat JSONL eval sets where there's no per-page stats.json layout.

Usage
-----
    /home/danny/.venvs/comet/bin/python score_jsonl_metrics.py \
        --in-jsonl backend/scripts/eval/openmantra_v10it/greedy/translations.jsonl \
        --gold-key en \
        --pred-key en  \
        # but inference_v10it_quality writes pred to "en" too — use --pred-from-candidates if needed
        --label v10it_openmantra_greedy \
        --metrics chrf,bleu,kiwi,metricx \
        --out-dir backend/scripts/eval/scorecards
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Use relative imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-jsonl", required=True,
                    help="JSONL with reference translations (must have 'jp' + ref key)")
    ap.add_argument("--pred-jsonl", required=True,
                    help="JSONL of predictions (from inference_v10it_quality.py); must have 'jp' + pred key")
    ap.add_argument("--gold-ref-key", default="en")
    ap.add_argument("--pred-key", default="en")
    ap.add_argument("--label", required=True)
    ap.add_argument("--metrics", default="chrf,bleu,kiwi,metricx",
                    help="Comma-separated. Supported: chrf, bleu, kiwi, metricx, xcomet.")
    ap.add_argument("--xcomet-model", default="Unbabel/XCOMET-XL",
                    help="HF id for xCOMET. Default XL (~3.5B, ~7GB VRAM bf16).")
    ap.add_argument("--xcomet-batch-size", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    # Load gold
    gold: dict[str, str] = {}
    with open(args.gold_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            jp = (r.get("jp") or "").strip()
            ref = (r.get(args.gold_ref_key) or "").strip()
            if jp and ref:
                gold[jp] = ref

    # Load pred and align by jp
    aligned: list[dict[str, str]] = []
    with open(args.pred_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            jp = (r.get("jp") or "").strip()
            pred = (r.get(args.pred_key) or "").strip()
            ref = gold.get(jp)
            if not ref or not jp:
                continue
            aligned.append({"slug": r.get("src", "?"), "jp": jp, "pred": pred, "ref": ref})

    if not aligned:
        print("ERROR: no aligned rows", file=sys.stderr)
        return 2
    print(f"[{args.label}] aligned {len(aligned)} rows")

    metrics = {m.strip() for m in args.metrics.split(",")}

    summary: dict = {
        "label": args.label,
        "n_aligned": len(aligned),
        "gold_jsonl": args.gold_jsonl,
        "pred_jsonl": args.pred_jsonl,
    }
    per_bubble = [dict(r) for r in aligned]

    if "chrf" in metrics or "bleu" in metrics:
        import sacrebleu
        preds = [r["pred"] for r in aligned]
        refs = [r["ref"] for r in aligned]
        chrf = float(sacrebleu.corpus_chrf(preds, [refs], word_order=2).score)
        bleu = float(sacrebleu.corpus_bleu(preds, [refs]).score)
        summary["chrf_pp"] = chrf
        summary["bleu"] = bleu
        # Per-bubble chrF
        for r, p in zip(per_bubble, aligned):
            s = sacrebleu.sentence_chrf(p["pred"], [p["ref"]], word_order=2)
            r["chrf_pp"] = float(s.score)
        print(f"  chrF++={chrf:.2f}  BLEU={bleu:.2f}")

    if "kiwi" in metrics:
        from comet import download_model, load_from_checkpoint
        ck = download_model("Unbabel/wmt23-cometkiwi-da-xl")
        model = load_from_checkpoint(ck)
        data = [{"src": r["jp"], "mt": r["pred"]} for r in aligned]
        t0 = time.time()
        out = model.predict(data, batch_size=32, gpus=1, progress_bar=True)
        scores = [float(s) for s in out.scores]
        mean = sum(scores) / len(scores)
        summary["cometkiwi_xl_mean"] = mean
        for r, s in zip(per_bubble, scores):
            r["cometkiwi_xl"] = s
        print(f"  Kiwi={mean:.4f} ({time.time()-t0:.1f}s)")
        del model
        import torch; torch.cuda.empty_cache()

    if "metricx" in metrics:
        from _metricx_inference import score_metricx
        triples = [(r["jp"], r["pred"], r["ref"]) for r in aligned]
        t0 = time.time()
        scores = score_metricx(triples, batch_size=8)
        mean = sum(scores) / len(scores)
        summary["metricx_24_xl_mean"] = mean
        summary["metricx_lower_is_better"] = True
        for r, s in zip(per_bubble, scores):
            r["metricx_24_xl"] = s
        print(f"  MetricX={mean:.4f} (lower=better; {time.time()-t0:.1f}s)")

    if "xcomet" in metrics:
        # xCOMET-XL: reference-based MQM-grade neural metric with per-segment
        # error spans (severity = critical | major | minor; range [0, 1]).
        # Output mirrors COMET API: out.scores and out.metadata.error_spans
        # per segment (list of {start,end,severity,confidence,text}).
        from comet import download_model, load_from_checkpoint
        ck = download_model(args.xcomet_model)
        model = load_from_checkpoint(ck)
        data = [{"src": r["jp"], "mt": r["pred"], "ref": r["ref"]} for r in aligned]
        t0 = time.time()
        out = model.predict(
            data, batch_size=args.xcomet_batch_size, gpus=1, progress_bar=True
        )
        scores = [float(s) for s in out.scores]
        mean = sum(scores) / len(scores)
        # Newer comet versions expose error spans on out.metadata.error_spans
        # (list-of-list-of-dict) when running xCOMET. Older releases attach
        # them per-system on out.metadata. Be defensive.
        spans_per_seg: list[list[dict]] = []
        meta = getattr(out, "metadata", None) or {}
        if isinstance(meta, dict):
            spans_per_seg = meta.get("error_spans") or []
        else:  # pydantic-style
            spans_per_seg = getattr(meta, "error_spans", []) or []
        if not spans_per_seg or len(spans_per_seg) != len(aligned):
            spans_per_seg = [[] for _ in aligned]
        # Aggregate counts for the summary scorecard.
        sev_counts = {"critical": 0, "major": 0, "minor": 0}
        for spans in spans_per_seg:
            for sp in spans:
                sev = (sp.get("severity") if isinstance(sp, dict) else None) or "minor"
                sev = sev.lower()
                if sev not in sev_counts:
                    sev_counts[sev] = 0
                sev_counts[sev] += 1
        summary["xcomet_xl_mean"] = mean
        summary["xcomet_xl_model"] = args.xcomet_model
        summary["xcomet_error_span_counts"] = sev_counts
        summary["xcomet_total_error_spans"] = sum(sev_counts.values())
        for r, s, spans in zip(per_bubble, scores, spans_per_seg):
            r["xcomet_xl"] = s
            # Normalize spans to plain dicts for JSON serialization.
            norm_spans: list[dict] = []
            for sp in spans:
                if isinstance(sp, dict):
                    norm_spans.append({
                        "start": sp.get("start"),
                        "end": sp.get("end"),
                        "severity": sp.get("severity"),
                        "confidence": sp.get("confidence"),
                        "text": sp.get("text"),
                    })
                else:
                    norm_spans.append({
                        "start": getattr(sp, "start", None),
                        "end": getattr(sp, "end", None),
                        "severity": getattr(sp, "severity", None),
                        "confidence": getattr(sp, "confidence", None),
                        "text": getattr(sp, "text", None),
                    })
            r["xcomet_error_spans"] = norm_spans
        print(f"  xCOMET={mean:.4f} (spans={sum(sev_counts.values())}; "
              f"crit={sev_counts.get('critical',0)} maj={sev_counts.get('major',0)} "
              f"min={sev_counts.get('minor',0)}; {time.time()-t0:.1f}s)")
        del model
        import torch; torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"score_summary_metrics_v2_{args.label}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    (out_dir / f"per_bubble_{args.label}.json").write_text(
        json.dumps(per_bubble, indent=2, ensure_ascii=False)
    )
    print(f"\nwrote {out_dir}/score_summary_metrics_v2_{args.label}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

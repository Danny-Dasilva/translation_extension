"""Multi-metric scorecard for v7 / v9c / v10-it predictions.

Wraps the per-bubble alignment logic from `score_v10it_vs_gemma.py` and adds
neural metrics so we can stop relying on chrF++ alone.

Metrics produced
----------------
* chrF++          — sacrebleu corpus_chrf (word_order=2). Reference-based.
                    Reference here = the Gemma-3-4B base teacher modeA.jsonl.
* BLEU            — sacrebleu corpus_bleu. Reference-based.
* gemma_em_pct    — exact-match vs the Gemma teacher reference. **RENAMED** to
                    "teacher fidelity" — measures distillation copy, NOT
                    translation quality.
* cometkiwi_xl    — Unbabel/wmt23-cometkiwi-da-xl reference-free QE. Range ~0..1.
* xcomet_xl       — Unbabel/XCOMET-XL reference-based + error spans.
                    Skipped if model is gated for this account (logged).
* metricx_24_xl   — google/metricx-24-hybrid-xl-v2p6 reference-based.
                    Returned 0..25 error score (LOWER is better).

Per-system outputs
------------------
* score_summary_metrics_v2.json (default `<pred-dir>/score_summary_metrics_v2.json`)
  Top-level: {label, n_aligned, chrf, bleu, teacher_fidelity_pct, kiwi_mean,
              xcomet_mean, metricx_mean, per_bubble: [{slug, jp, pred, ref,
                                                       kiwi, xcomet, metricx, ...}]}

Usage
-----
    source /home/danny/.venvs/comet/bin/activate
    python score_summary_metrics_v2.py \
       --pred-dir /home/danny/manga-output/644289-gemma4-v10it-unsloth-fixed \
       --label v10it \
       --out-dir backend/scripts/eval

Pass --metrics to opt out of slow ones, e.g. --metrics chrf,bleu,kiwi for a fast
sanity pass before the full neural sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# We may run this from any cwd; resolve relative to repo root.
THIS_FILE = Path(__file__).resolve()


def is_jp_passthrough(en: str, jp: str) -> bool:
    if not en:
        return False
    if en.strip() == jp.strip():
        return True
    head = en.strip()[:20]
    return any(
        0x3040 <= ord(c) <= 0x309F or
        0x30A0 <= ord(c) <= 0x30FF or
        0x4E00 <= ord(c) <= 0x9FFF
        for c in head
    )


def load_pred_aligned(pred_dir: Path, ref_jsonl: Path) -> list[dict[str, str]]:
    """Return list of {slug, jp, pred, ref} aligned with the Gemma teacher ref."""
    gemma: dict[str, list[tuple[str, str]]] = {}
    with open(ref_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gemma[r["slug"]] = list(zip(r.get("jp_texts", []), r.get("en_texts", [])))

    rows: list[dict[str, str]] = []
    for d in sorted(pred_dir.iterdir()):
        if not d.is_dir() or not d.name.isdigit():
            continue
        sj = d / "stats.json"
        if not sj.exists():
            continue
        s = json.loads(sj.read_text(encoding="utf-8"))
        ocr = s.get("ocr_samples") or []
        preds = s.get("translations") or []
        gref_pairs = gemma.get(d.name, [])
        gref_map = {jp.strip(): en for jp, en in gref_pairs}
        for jp, pred in zip(ocr, preds):
            gref = gref_map.get(jp.strip(), "")
            if not gref:
                continue
            rows.append({"slug": d.name, "jp": jp, "pred": pred, "ref": gref})
    return rows


def compute_chrf_bleu(rows: list[dict[str, str]]) -> tuple[float, float]:
    import sacrebleu
    preds = [r["pred"] for r in rows]
    refs = [r["ref"] for r in rows]
    chrf = float(sacrebleu.corpus_chrf(preds, [refs], word_order=2).score)
    bleu = float(sacrebleu.corpus_bleu(preds, [refs]).score)
    return chrf, bleu


def compute_per_bubble_chrf(rows: list[dict[str, str]]) -> list[float]:
    """Per-segment chrF++ scores; needed for downstream bootstrap analysis."""
    import sacrebleu
    out: list[float] = []
    for r in rows:
        s = sacrebleu.sentence_chrf(r["pred"], [r["ref"]], word_order=2)
        out.append(float(s.score))
    return out


def compute_kiwi(rows: list[dict[str, str]], *, batch_size: int = 32) -> list[float]:
    from comet import download_model, load_from_checkpoint
    ckpt = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    model = load_from_checkpoint(ckpt)
    data = [{"src": r["jp"], "mt": r["pred"]} for r in rows]
    out = model.predict(data, batch_size=batch_size, gpus=1, progress_bar=True)
    return [float(s) for s in out.scores]


def compute_xcomet(rows: list[dict[str, str]], *, batch_size: int = 16) -> list[float]:
    """XCOMET-XL is gated for some accounts; if download fails, raise to caller."""
    from comet import download_model, load_from_checkpoint
    # Direct snapshot path since `download_model` rejects XCOMET-XL by name on
    # this comet version.
    from huggingface_hub import snapshot_download
    snap = snapshot_download("Unbabel/XCOMET-XL")
    ckpt = os.path.join(snap, "checkpoints", "model.ckpt")
    model = load_from_checkpoint(ckpt)
    data = [{"src": r["jp"], "mt": r["pred"], "ref": r["ref"]} for r in rows]
    out = model.predict(data, batch_size=batch_size, gpus=1, progress_bar=True)
    return [float(s) for s in out.scores]


def compute_metricx(
    rows: list[dict[str, str]], *, batch_size: int = 8
) -> list[float]:
    """MetricX-24-Hybrid-XL — reference-based MQM-trained metric.

    Score is an *error* score 0..25; LOWER is better.

    Uses the in-tree _metricx_inference helper (reproduces the official
    google-research/metricx MT5ForRegression class + prompt format).
    """
    from _metricx_inference import score_metricx
    triples = [(r["jp"], r["pred"], r["ref"]) for r in rows]
    return score_metricx(triples, batch_size=batch_size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument(
        "--ref-jsonl",
        default="/home/danny/manga-output/644289-abc-gemma4-base-45/modeA.jsonl",
    )
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="Optional output dir. Defaults to <pred-dir>.")
    ap.add_argument(
        "--metrics",
        default="chrf,bleu,kiwi,metricx,xcomet",
        help="Comma-separated subset of {chrf,bleu,kiwi,xcomet,metricx}.",
    )
    ap.add_argument("--save-per-bubble", action="store_true",
                    help="Persist per-bubble scores to JSON (for bootstrap analysis).")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_dir():
        print(f"ERROR: pred_dir not found: {pred_dir}", file=sys.stderr)
        return 2
    ref_jsonl = Path(args.ref_jsonl)
    out_dir = Path(args.out_dir) if args.out_dir else pred_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_run = {m.strip() for m in args.metrics.split(",") if m.strip()}
    print(f"[score_summary_metrics_v2] label={args.label} pred_dir={pred_dir}")
    print(f"  metrics requested: {sorted(metrics_to_run)}")

    rows = load_pred_aligned(pred_dir, ref_jsonl)
    print(f"  aligned bubbles: {len(rows)}")
    if not rows:
        print("ERROR: no aligned rows; check pred-dir + ref-jsonl", file=sys.stderr)
        return 3

    # Empty-rate / JP-passthrough are essentially free — always compute.
    empty_n = sum(1 for r in rows if not r["pred"].strip())
    jp_pass_n = sum(1 for r in rows if is_jp_passthrough(r["pred"], r["jp"]))
    em_hits = sum(
        1 for r in rows
        if r["pred"].strip() and r["pred"].strip().lower() == r["ref"].strip().lower()
    )

    summary: dict[str, Any] = {
        "label": args.label,
        "pred_dir": str(pred_dir),
        "ref_jsonl": str(ref_jsonl),
        "n_aligned": len(rows),
        "empty_n": empty_n,
        "empty_pct": round(empty_n / len(rows) * 100, 2),
        "jp_passthrough_n": jp_pass_n,
        "jp_passthrough_pct": round(jp_pass_n / len(rows) * 100, 2),
        "teacher_fidelity": {
            "exact_match_n": em_hits,
            "exact_match_pct": round(em_hits / len(rows) * 100, 2),
            "_note": "Renamed from gemma_em — measures distillation fidelity vs the Gemma-3-4B base teacher, NOT translation quality. The teacher itself is just one source of truth; high fidelity means we copied the teacher, not that we translated well.",
        },
    }
    per_bubble: list[dict[str, Any]] = [dict(r) for r in rows]

    if "chrf" in metrics_to_run or "bleu" in metrics_to_run:
        t0 = time.time()
        chrf, bleu = compute_chrf_bleu(rows)
        print(f"  chrF++={chrf:.2f}, BLEU={bleu:.2f}  ({time.time()-t0:.1f}s)")
        summary["chrf_pp"] = chrf
        summary["bleu"] = bleu
        # Per-bubble chrF for downstream paired-bs analysis.
        per_chrf = compute_per_bubble_chrf(rows)
        for r, s in zip(per_bubble, per_chrf):
            r["chrf_pp"] = s

    if "kiwi" in metrics_to_run:
        t0 = time.time()
        try:
            kiwi_scores = compute_kiwi(rows)
            mean = sum(kiwi_scores) / len(kiwi_scores)
            print(f"  CometKiwi-23-XL mean={mean:.4f}  ({time.time()-t0:.1f}s)")
            summary["cometkiwi_xl_mean"] = float(mean)
            summary["cometkiwi_xl_p50"] = float(sorted(kiwi_scores)[len(kiwi_scores)//2])
            for r, s in zip(per_bubble, kiwi_scores):
                r["cometkiwi_xl"] = s
        except Exception as e:
            summary["cometkiwi_xl_error"] = str(e)
            print(f"  CometKiwi failed: {e}")

    if "xcomet" in metrics_to_run:
        t0 = time.time()
        try:
            xc_scores = compute_xcomet(rows)
            mean = sum(xc_scores) / len(xc_scores)
            print(f"  XCOMET-XL mean={mean:.4f}  ({time.time()-t0:.1f}s)")
            summary["xcomet_xl_mean"] = float(mean)
            for r, s in zip(per_bubble, xc_scores):
                r["xcomet_xl"] = s
        except Exception as e:
            summary["xcomet_xl_error"] = str(e)
            print(f"  XCOMET-XL failed (likely gated): {e}")

    if "metricx" in metrics_to_run:
        t0 = time.time()
        try:
            mx_scores = compute_metricx(rows)
            mean = sum(mx_scores) / len(mx_scores)
            print(f"  MetricX-24-Hybrid-XL mean(error)={mean:.4f} (lower=better)  ({time.time()-t0:.1f}s)")
            summary["metricx_24_xl_mean"] = float(mean)
            summary["metricx_lower_is_better"] = True
            for r, s in zip(per_bubble, mx_scores):
                r["metricx_24_xl"] = s
        except Exception as e:
            summary["metricx_24_xl_error"] = str(e)
            print(f"  MetricX failed: {e}")

    if args.save_per_bubble:
        summary["per_bubble"] = per_bubble

    out_path = out_dir / f"score_summary_metrics_v2_{args.label}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path}")

    # Also write per-bubble in a separate file (always — needed for bootstrap)
    pb_path = out_dir / f"per_bubble_{args.label}.json"
    pb_path.write_text(json.dumps(per_bubble, indent=2, ensure_ascii=False))
    print(f"wrote {pb_path}  (per-bubble scores for bootstrap)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

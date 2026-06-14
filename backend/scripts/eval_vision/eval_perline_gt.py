"""Per-LINE OCR accuracy A/B on the labeled ground-truth set.

Confirms the production model swap holds accuracy:
  A = parseq_manga_best_ep60_AR_single.onnx      (OLD production, batch=1, charset 4407)
  B = parseq_manga_ep60_nonAR_dynbatch.fp16.onnx (NEW production, batched, charset 4407, SAME weights as A)
  C = parseq_manga_large_5p16.fp16.onnx          (stopgap, charset 4400)  -- optional

Ground truth: backend/scripts/data/manga109/perline_gt.parquet
  Columns: book, page, xmin, ymin, xmax, ymax, w, h, aspect, orientation, n_chars, jp_text, image_path
  image_path already points at the local NAS mount (.../manga109s/.../images/<book>/<page>.jpg).

Each sampled line is cropped (xmin:xmax, ymin:ymax + pad) from its page image and run through
the REAL ParseqOCRService (recognize_text_batch), so preprocessing (_maybe_rotate_vertical, resize,
normalize) and decode mirror production exactly.

Two scoring passes:
  - POSTPROCESSED: ParseqOCRService output as-is. The service's _decode() applies _finalize_ocr
    (ocr_postprocess.apply_all + normalize_japanese_text + repetition guard) -> this IS what ships.
  - RAW: same models with _finalize_ocr monkeypatched to identity, so we also see bare model output.

Metrics: exact-match rate + mean/median CER (Levenshtein/len(ref)), overall and split by orientation.

KEY QUESTION: is B within +0.5pp mean CER and within 3pp exact-match of A? (they share weights)

Does NOT modify config.py / parseq_ocr_service.py / translate.py. Adds this script + writes results.

Usage:
    backend/.venv/bin/python backend/scripts/eval_vision/eval_perline_gt.py \
        --n 1000 --seed 0 --batch-size 24 [--with-c]
"""
from __future__ import annotations

# torch first so its bundled CUDA libs are on the loader path before
# onnxruntime-gpu probes for libcublas/libcudnn (mirrors ParseqOCRService).
import torch  # noqa: F401

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import polars as pl

BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))  # backend/ on path for `app.*`

from app.services import parseq_ocr_service as pos  # noqa: E402
from app.services.parseq_ocr_service import ParseqOCRService  # noqa: E402

MODELS_DIR = BACKEND_ROOT / "models"
GT_PARQUET = BACKEND_ROOT / "scripts" / "data" / "manga109" / "perline_gt.parquet"

MODEL_A = "parseq_manga_best_ep60_AR_single.onnx"      # OLD production (batch=1)
MODEL_B = "parseq_manga_ep60_nonAR_dynbatch.fp16.onnx"  # NEW production (batched)
MODEL_C = "parseq_manga_large_5p16.fp16.onnx"           # stopgap (charset 4400)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, ref) / len(ref)


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _median(xs: List[float]) -> float:
    return float(np.median(xs)) if xs else 0.0


# --------------------------------------------------------------------------
# Sampling: stratify so we get a healthy mix of BOTH orientations.
# Vertical dominates ~94%; we cap horizontal share so it is well represented.
# --------------------------------------------------------------------------
def sample_rows(n: int, seed: int, min_chars: int, horiz_target_frac: float) -> List[dict]:
    df = pl.read_parquet(GT_PARQUET)
    df = df.filter(pl.col("jp_text").str.len_chars() >= min_chars)

    vert = df.filter(pl.col("orientation") == "vertical")
    horiz = df.filter(pl.col("orientation") == "horizontal")

    n_horiz = min(len(horiz), int(round(n * horiz_target_frac)))
    n_vert = min(len(vert), n - n_horiz)
    # if horizontal pool too small, backfill with vertical
    n_vert = min(len(vert), n - n_horiz)

    sv = vert.sample(n=n_vert, seed=seed)
    sh = horiz.sample(n=n_horiz, seed=seed) if n_horiz > 0 else horiz.head(0)
    out = pl.concat([sv, sh]).sample(fraction=1.0, shuffle=True, seed=seed)
    return out.to_dicts()


def load_crops(rows: List[dict], pad: int) -> List[Optional[np.ndarray]]:
    """Crop each line from its page; page-image cache keyed by path."""
    cache: Dict[str, Optional[np.ndarray]] = {}
    crops: List[Optional[np.ndarray]] = []
    for r in rows:
        path = r["image_path"]
        if path not in cache:
            cache[path] = cv2.imread(path)
        img = cache[path]
        if img is None:
            crops.append(None)
            continue
        h, w = img.shape[:2]
        x0 = max(0, int(r["xmin"]) - pad)
        y0 = max(0, int(r["ymin"]) - pad)
        x1 = min(w, int(r["xmax"]) + pad)
        y1 = min(h, int(r["ymax"]) + pad)
        if x1 <= x0 or y1 <= y0:
            crops.append(None)
            continue
        crops.append(img[y0:y1, x0:x1].copy())
    return crops


# --------------------------------------------------------------------------
# Model running through the REAL service.
# --------------------------------------------------------------------------
def make_service(model_name: str) -> ParseqOCRService:
    model_path = MODELS_DIR / model_name
    meta_name = model_path.with_suffix(".json").name
    meta_path = MODELS_DIR / meta_name
    if not meta_path.exists():
        meta_path = None  # let service fall back
    # fallback_fp32_path: only the dynbatch export ships an fp32 sibling; for
    # the others point fallback at the model itself so the candidate list still
    # resolves (the primary load will succeed on CUDA anyway).
    return ParseqOCRService(
        model_path=str(model_path),
        fallback_fp32_path=str(model_path),
        meta_path=str(meta_path) if meta_path else None,
    )


def run_service(svc: ParseqOCRService, crops: List[np.ndarray], batch_size: int) -> List[str]:
    return asyncio.run(svc.recognize_text_batch(crops, batch_size=batch_size))


# --------------------------------------------------------------------------
def score(preds: List[str], gts: List[str], orients: List[str]) -> dict:
    cers = [cer(p, g) for p, g in zip(preds, gts)]
    exact = [p == g for p, g in zip(preds, gts)]

    def subset(orient: str) -> dict:
        idx = [i for i, o in enumerate(orients) if o == orient]
        if not idx:
            return {"n": 0, "exact_match_acc": 0.0, "mean_cer": 0.0, "median_cer": 0.0}
        sc = [cers[i] for i in idx]
        se = [exact[i] for i in idx]
        return {
            "n": len(idx),
            "exact_match_acc": _mean(se),
            "mean_cer": _mean(sc),
            "median_cer": _median(sc),
        }

    return {
        "overall": {
            "n": len(preds),
            "exact_match_acc": _mean(exact),
            "mean_cer": _mean(cers),
            "median_cer": _median(cers),
        },
        "vertical": subset("vertical"),
        "horizontal": subset("horizontal"),
        "_cers": cers,
        "_exact": exact,
    }


def fmt_block(name: str, s: dict) -> str:
    o, v, h = s["overall"], s["vertical"], s["horizontal"]
    return (
        f"  {name:<28} overall  n={o['n']:<5} exact={o['exact_match_acc']*100:6.2f}%  "
        f"meanCER={o['mean_cer']*100:6.2f}%  medCER={o['median_cer']*100:6.2f}%\n"
        f"  {'':<28} vert     n={v['n']:<5} exact={v['exact_match_acc']*100:6.2f}%  "
        f"meanCER={v['mean_cer']*100:6.2f}%  medCER={v['median_cer']*100:6.2f}%\n"
        f"  {'':<28} horiz    n={h['n']:<5} exact={h['exact_match_acc']*100:6.2f}%  "
        f"meanCER={h['mean_cer']*100:6.2f}%  medCER={h['median_cer']*100:6.2f}%"
    )


def strip_internal(s: dict) -> dict:
    return {k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v)
            for k, v in s.items() if not k.startswith("_")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--min-chars", type=int, default=1)
    ap.add_argument("--pad", type=int, default=2)
    ap.add_argument("--horiz-frac", type=float, default=0.25,
                    help="target fraction of sample that is horizontal (oversampled vs 6%% natural)")
    ap.add_argument("--with-c", action="store_true", help="also evaluate model C (parseq_large_5p16)")
    ap.add_argument("--max-disagreements", type=int, default=15)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "eval_perline_gt_results.json"))
    args = ap.parse_args()

    if not GT_PARQUET.exists():
        print(f"ERROR: GT parquet not found: {GT_PARQUET}", file=sys.stderr)
        return 1

    print(f"Sampling n={args.n} (seed={args.seed}, horiz_frac={args.horiz_frac}, min_chars={args.min_chars})...")
    rows = sample_rows(args.n, args.seed, args.min_chars, args.horiz_frac)
    print(f"  sampled {len(rows)} rows; cropping lines...")
    crops_raw = load_crops(rows, args.pad)

    # drop rows whose image/crop failed
    keep = [i for i, c in enumerate(crops_raw) if c is not None]
    dropped = len(rows) - len(keep)
    rows = [rows[i] for i in keep]
    crops = [crops_raw[i] for i in keep]
    gts = [r["jp_text"] for r in rows]
    orients = [r["orientation"] for r in rows]
    n_vert = orients.count("vertical")
    n_horiz = orients.count("horizontal")
    print(f"  usable crops: {len(crops)} (dropped {dropped}); vertical={n_vert} horizontal={n_horiz}")
    if not crops:
        print("ERROR: no usable crops", file=sys.stderr)
        return 1

    results: dict = {
        "n_sampled": len(rows),
        "n_dropped_unresolved": dropped,
        "n_vertical": n_vert,
        "n_horizontal": n_horiz,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gt_parquet": str(GT_PARQUET),
        "models": {"A": MODEL_A, "B": MODEL_B, "C": MODEL_C if args.with_c else None},
        "models_note": "A=OLD prod (batch=1), B=NEW prod (dynbatch, same weights as A), C=stopgap",
    }

    model_specs = [("A", MODEL_A, 1), ("B", MODEL_B, args.batch_size)]
    if args.with_c:
        model_specs.append(("C", MODEL_C, args.batch_size))

    preds_pp: Dict[str, List[str]] = {}
    preds_raw: Dict[str, List[str]] = {}
    timing: Dict[str, float] = {}

    orig_finalize = pos._finalize_ocr

    for key, name, bs in model_specs:
        print(f"\nLoading model {key}: {name} (batch_size={bs})")
        try:
            svc = make_service(name)
        except Exception as e:
            print(f"ERROR: model {key} failed to load: {e}", file=sys.stderr)
            results.setdefault("load_errors", {})[key] = str(e)
            continue
        print(f"  -> device={svc.device}")

        # POSTPROCESSED pass (production behavior): service applies _finalize_ocr.
        pos._finalize_ocr = orig_finalize
        t0 = time.perf_counter()
        try:
            pp = run_service(svc, crops, bs)
        except Exception as e:
            print(f"ERROR: model {key} inference failed (postprocessed pass): {e}", file=sys.stderr)
            results.setdefault("infer_errors", {})[key] = str(e)
            del svc
            continue
        timing[key] = (time.perf_counter() - t0) * 1000 / len(crops)
        preds_pp[key] = pp

        # RAW pass: monkeypatch _finalize_ocr to identity so we see bare decode.
        pos._finalize_ocr = lambda s: s
        try:
            raw = run_service(svc, crops, bs)
        finally:
            pos._finalize_ocr = orig_finalize
        preds_raw[key] = raw
        del svc
        print(f"  {key} done: {timing[key]:.1f} ms/crop")

    # restore
    pos._finalize_ocr = orig_finalize

    # ---- score every loaded model ----
    scored_pp: Dict[str, dict] = {}
    scored_raw: Dict[str, dict] = {}
    for key in preds_pp:
        scored_pp[key] = score(preds_pp[key], gts, orients)
        scored_raw[key] = score(preds_raw[key], gts, orients)

    print("\n" + "=" * 78)
    print("POSTPROCESSED (production-shipped output)")
    print("=" * 78)
    for key in scored_pp:
        print(fmt_block(f"{key} ({results['models'][key]})", scored_pp[key]))
    print("\n" + "-" * 78)
    print("RAW (bare model decode, no postprocess)")
    print("-" * 78)
    for key in scored_raw:
        print(fmt_block(f"{key}", scored_raw[key]))

    print("\nTiming (ms/crop):")
    for key, ms in timing.items():
        print(f"  {key}: {ms:.2f}")

    # ---- VERDICT: B vs A ----
    verdict = None
    if "A" in scored_pp and "B" in scored_pp:
        a, b = scored_pp["A"]["overall"], scored_pp["B"]["overall"]
        d_cer = (b["mean_cer"] - a["mean_cer"]) * 100  # percentage points
        d_exact = (b["exact_match_acc"] - a["exact_match_acc"]) * 100
        passes = (d_cer <= 0.5) and (d_exact >= -3.0)
        verdict = {
            "pass": bool(passes),
            "delta_mean_cer_pp": d_cer,
            "delta_exact_match_pp": d_exact,
            "threshold_cer_pp": 0.5,
            "threshold_exact_pp": -3.0,
            "scoring": "postprocessed",
        }
        print("\n" + "=" * 78)
        print("VERDICT: B (new prod) vs A (old prod), postprocessed")
        print("=" * 78)
        print(f"  delta mean CER:    {d_cer:+.3f} pp  (must be <= +0.50)")
        print(f"  delta exact-match: {d_exact:+.3f} pp  (must be >= -3.00)")
        print(f"  --> {'PASS  (B holds accuracy)' if passes else 'FAIL  (regression)'}")

    # ---- disagreements A vs B (raw, so we see true model divergence) ----
    sample_diffs: List[dict] = []
    if "A" in preds_raw and "B" in preds_raw:
        diffs = []
        for r, pa, pb in zip(rows, preds_raw["A"], preds_raw["B"]):
            if pa != pb:
                diffs.append({
                    "src": f"{r['book']}:p{r['page']}",
                    "orientation": r["orientation"],
                    "ground_truth": r["jp_text"],
                    "model_a": pa,
                    "model_b": pb,
                    "a_correct": pa == r["jp_text"],
                    "b_correct": pb == r["jp_text"],
                })
        diffs.sort(key=lambda d: cer(d["model_b"], d["model_a"]), reverse=True)
        sample_diffs = diffs[: args.max_disagreements]
        a_only = sum(1 for d in diffs if d["a_correct"] and not d["b_correct"])
        b_only = sum(1 for d in diffs if d["b_correct"] and not d["a_correct"])
        both_wrong = sum(1 for d in diffs if not d["a_correct"] and not d["b_correct"])
        results["ab_disagreement_breakdown"] = {
            "total_disagreements": len(diffs),
            "a_right_b_wrong": a_only,
            "b_right_a_wrong": b_only,
            "both_wrong": both_wrong,
            "agreement_rate": 1 - len(diffs) / len(crops),
        }
        print("\n" + "=" * 78)
        print(f"A vs B disagreements (raw decode): {len(diffs)}/{len(crops)} "
              f"(agree {(1-len(diffs)/len(crops))*100:.2f}%)")
        print(f"  A right / B wrong: {a_only}   B right / A wrong: {b_only}   both wrong: {both_wrong}")
        print("=" * 78)
        for d in sample_diffs:
            tag = "A" if d["a_correct"] else ("B" if d["b_correct"] else "-")
            print(f"  [{tag}|{d['orientation'][:1]}] GT={d['ground_truth']!r}")
            print(f"            A ={d['model_a']!r}")
            print(f"            B ={d['model_b']!r}")

    results["postprocessed"] = {k: strip_internal(v) for k, v in scored_pp.items()}
    results["raw"] = {k: strip_internal(v) for k, v in scored_raw.items()}
    results["timing_ms_per_crop"] = timing
    results["verdict_b_vs_a"] = verdict
    results["sample_ab_disagreements"] = sample_diffs

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nSaved results: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

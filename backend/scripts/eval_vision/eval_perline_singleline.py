"""Single-line / PRODUCTION-regime A/B re-test for the batched PARSeq export.

Context
-------
A prior bubble-level A/B FAILED (B +1.65pp CER vs A). But that used bubble-level
GT (multi-line; many jp_text rows contain newlines) which is OUT OF DISTRIBUTION
for B -- a single-line recognizer whose non-AR decode hallucinates repeat runs on
too-long inputs. PRODUCTION feeds SINGLE-LINE CTD crops (recognize_blocks_with_lines),
so this re-tests on that regime.

  A = parseq_manga_best_ep60_AR_single.onnx        (OLD prod, batch=1, charset 4407) -- REFERENCE
  B = parseq_manga_ep60_nonAR_dynbatch.fp16.onnx   (NEW prod candidate, batched, charset 4407, SAME weights as A)

Both are run through the REAL ParseqOCRService (recognize_text_batch), so preprocess
(_maybe_rotate_vertical/resize/normalize) and decode mirror production exactly. The
service's _decode applies _finalize_ocr (production postprocess) -> that IS what ships.

TEST 1 -- GT eval, SINGLE-LINE rows only
  Filter perline_gt.parquet to truly single-line rows: '\n' not in jp_text AND
  n_chars <= --max-chars (default 20). Crop each line, run A and B, compute
  exact-match + mean/median CER vs jp_text (postprocessed). Split by orientation.
  Verdict: B passes if mean CER <= A + 0.5pp AND exact-match within 3pp.

TEST 2 -- real CTD per-line crops (true production distribution, NO GT)
  Run the production CTD detector on real manga pages (nhentai webp), take the
  per-line crops, run A and B on the SAME crops. Report A-vs-B exact agreement %.
  Artifact scan: count B outputs containing a REPEAT ARTIFACT (run of >=4 identical
  chars, OR trailing punctuation run !!!!/...../etc of len >=4) that A's matching
  output does NOT contain. Report artifact rate B vs A + examples.

Does NOT modify config.py / parseq_ocr_service.py / translate.py.

Usage:
    backend/.venv/bin/python backend/scripts/eval_vision/eval_perline_singleline.py \
        --n 2000 --max-chars 20 --pages 18 --batch-size 24 --seed 0
"""
from __future__ import annotations

# torch first so its bundled CUDA libs are on the loader path before
# onnxruntime-gpu probes for libcublas/libcudnn (mirrors ParseqOCRService).
import torch  # noqa: F401

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import polars as pl

BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))  # backend/ on path for `app.*`

from app.services.parseq_ocr_service import ParseqOCRService  # noqa: E402

MODELS_DIR = BACKEND_ROOT / "models"
GT_PARQUET = BACKEND_ROOT / "scripts" / "data" / "manga109" / "perline_gt.parquet"

MODEL_A = "parseq_manga_best_ep60_AR_single.onnx"        # OLD prod, REFERENCE
MODEL_B = "parseq_manga_ep60_nonAR_dynbatch.fp16.onnx"   # NEW prod candidate

# Real production-distribution pages (single-line CTD feed).
NHENTAI_DIRS = [
    "/mnt/nas/drive_2/onlyfans/external_content/nhentai/637653_Haha to Ochite Iku Part 12",
    "/mnt/nas/drive_2/onlyfans/external_content/nhentai/653631_Haha to Ochite Iku Part 13",
]


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
# Repeat-artifact detection (Test 2)
# --------------------------------------------------------------------------
# A run of >= 4 identical characters anywhere.
_RUN_RE = re.compile(r"(.)\1{3,}")
# A trailing punctuation run of length >= 4 (covers !!!!, ...., ・・・・, ーーーー, 。。。。 etc).
_PUNCT = "!?！？.。．・…ー―—-~〜、，,"
_TRAIL_RE = re.compile(rf"[{re.escape(_PUNCT)}]{{4,}}$")


def has_repeat_artifact(s: str) -> bool:
    if not s:
        return False
    if _RUN_RE.search(s):
        return True
    if _TRAIL_RE.search(s):
        return True
    return False


# --------------------------------------------------------------------------
# Sampling (Test 1): single-line rows only, stratified across orientations.
# --------------------------------------------------------------------------
def sample_single_line_rows(n: int, seed: int, max_chars: int,
                            min_chars: int, horiz_frac: float) -> tuple[List[dict], int]:
    df = pl.read_parquet(GT_PARQUET)
    total = len(df)
    # TRULY single-line: no embedded newline AND a sane length to exclude run-on
    # multi-line annotations collapsed into one row.
    sl = df.filter(~pl.col("jp_text").str.contains("\n"))
    sl = sl.filter(pl.col("n_chars") >= min_chars)
    sl = sl.filter(pl.col("n_chars") <= max_chars)
    n_single = len(sl)

    vert = sl.filter(pl.col("orientation") == "vertical")
    horiz = sl.filter(pl.col("orientation") == "horizontal")
    n_horiz = min(len(horiz), int(round(n * horiz_frac)))
    n_vert = min(len(vert), n - n_horiz)
    sv = vert.sample(n=n_vert, seed=seed)
    sh = horiz.sample(n=n_horiz, seed=seed) if n_horiz > 0 else horiz.head(0)
    out = pl.concat([sv, sh]).sample(fraction=1.0, shuffle=True, seed=seed)
    print(f"  parquet total={total}, single-line pool={n_single} "
          f"(vert={len(vert)} horiz={len(horiz)})")
    return out.to_dicts(), n_single


def load_gt_crops(rows: List[dict], pad: int) -> List[Optional[np.ndarray]]:
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
# CTD line crops (Test 2): real production feed on real pages.
# --------------------------------------------------------------------------
def sample_nhentai_pages(n_pages: int) -> List[str]:
    pages: List[str] = []
    for d in NHENTAI_DIRS:
        dd = Path(d)
        if not dd.exists():
            print(f"  WARNING: nhentai dir missing: {d}", file=sys.stderr)
            continue
        webps = sorted(dd.glob("*.webp"))
        # take an evenly-spaced slice so we don't bias to first pages (covers)
        if not webps:
            continue
        take = min(n_pages, len(webps))
        step = max(1, len(webps) // take)
        pages.extend(str(p) for p in webps[::step][:take])
    return pages


def ctd_line_crops(page_paths: List[str], pad: int) -> tuple[List[np.ndarray], List[str]]:
    """Run the production CTD detector on each page, return per-line crops + ids."""
    from app.services.ctd_service import ComicTextDetectorService

    ctd = ComicTextDetectorService()
    crops: List[np.ndarray] = []
    ids: List[str] = []

    async def _run():
        for pp in page_paths:
            img = cv2.imread(pp)
            if img is None:
                print(f"  WARNING: could not read {pp}", file=sys.stderr)
                continue
            det = await ctd.detect(img, input_is_bgr=True)
            h, w = img.shape[:2]
            tag = Path(pp).parent.name[:8] + "/" + Path(pp).stem
            for j, ln in enumerate(det.get("text_lines", [])):
                x0 = max(0, int(ln["minX"]) - pad); y0 = max(0, int(ln["minY"]) - pad)
                x1 = min(w, int(ln["maxX"]) + pad); y1 = min(h, int(ln["maxY"]) + pad)
                if x1 > x0 and y1 > y0:
                    crops.append(img[y0:y1, x0:x1].copy())
                    ids.append(f"{tag}#L{j}")

    asyncio.run(_run())
    return crops, ids


# --------------------------------------------------------------------------
def make_service(model_name: str) -> ParseqOCRService:
    model_path = MODELS_DIR / model_name
    meta_path = model_path.with_suffix(".json")
    return ParseqOCRService(
        model_path=str(model_path),
        fallback_fp32_path=str(model_path),
        meta_path=str(meta_path) if meta_path.exists() else None,
    )


def run_service(svc: ParseqOCRService, crops: List[np.ndarray], batch_size: int) -> List[str]:
    return asyncio.run(svc.recognize_text_batch(crops, batch_size=batch_size))


def score(preds: List[str], gts: List[str], orients: List[str]) -> dict:
    cers = [cer(p, g) for p, g in zip(preds, gts)]
    exact = [p == g for p, g in zip(preds, gts)]

    def subset(orient: str) -> dict:
        idx = [i for i, o in enumerate(orients) if o == orient]
        if not idx:
            return {"n": 0, "exact_match_acc": 0.0, "mean_cer": 0.0, "median_cer": 0.0}
        sc = [cers[i] for i in idx]
        se = [exact[i] for i in idx]
        return {"n": len(idx), "exact_match_acc": _mean(se),
                "mean_cer": _mean(sc), "median_cer": _median(sc)}

    return {
        "overall": {"n": len(preds), "exact_match_acc": _mean(exact),
                    "mean_cer": _mean(cers), "median_cer": _median(cers)},
        "vertical": subset("vertical"),
        "horizontal": subset("horizontal"),
    }


def fmt_block(name: str, s: dict) -> str:
    o, v, h = s["overall"], s["vertical"], s["horizontal"]
    return (
        f"  {name:<42} overall n={o['n']:<5} exact={o['exact_match_acc']*100:6.2f}%  "
        f"meanCER={o['mean_cer']*100:6.2f}%  medCER={o['median_cer']*100:6.2f}%\n"
        f"  {'':<42} vert    n={v['n']:<5} exact={v['exact_match_acc']*100:6.2f}%  "
        f"meanCER={v['mean_cer']*100:6.2f}%  medCER={v['median_cer']*100:6.2f}%\n"
        f"  {'':<42} horiz   n={h['n']:<5} exact={h['exact_match_acc']*100:6.2f}%  "
        f"meanCER={h['mean_cer']*100:6.2f}%  medCER={h['median_cer']*100:6.2f}%"
    )


def main() -> int:
    global MODEL_A, MODEL_B
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="single-line GT rows to sample (Test 1)")
    ap.add_argument("--max-chars", type=int, default=20,
                    help="exclude rows with n_chars > this (run-on multi-line annotations)")
    ap.add_argument("--min-chars", type=int, default=1)
    ap.add_argument("--horiz-frac", type=float, default=0.25)
    ap.add_argument("--pages", type=int, default=18, help="real manga pages to CTD (Test 2)")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pad", type=int, default=2)
    ap.add_argument("--max-disagreements", type=int, default=15)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "eval_perline_singleline_results.json"))
    ap.add_argument("--model-a", type=str, default=MODEL_A, help="reference model filename")
    ap.add_argument("--model-b", type=str, default=MODEL_B, help="candidate model filename")
    args = ap.parse_args()
    MODEL_A = args.model_a
    MODEL_B = args.model_b

    if not GT_PARQUET.exists():
        print(f"ERROR: GT parquet not found: {GT_PARQUET}", file=sys.stderr)
        return 1

    results: dict = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": {"A_reference": MODEL_A, "B_candidate": MODEL_B},
        "regime": "single-line (production: recognize_blocks_with_lines feeds single-line CTD crops)",
        "config": {"n": args.n, "max_chars": args.max_chars, "min_chars": args.min_chars,
                   "pages": args.pages, "batch_size": args.batch_size, "seed": args.seed,
                   "artifact_def": "run of >=4 identical chars OR trailing punct run >=4, in B-only"},
    }

    # ===================================================================
    # Load both services once (reused across tests).
    # ===================================================================
    print(f"Loading A (reference): {MODEL_A}")
    svc_a = make_service(MODEL_A)
    print(f"  -> device={svc_a.device}")
    print(f"Loading B (candidate): {MODEL_B}")
    svc_b = make_service(MODEL_B)
    print(f"  -> device={svc_b.device}")

    # ===================================================================
    # TEST 1 -- single-line GT
    # ===================================================================
    print("\n" + "#" * 78)
    print("TEST 1: single-line GT eval (postprocessed = production output)")
    print("#" * 78)
    rows, n_single_pool = sample_single_line_rows(
        args.n, args.seed, args.max_chars, args.min_chars, args.horiz_frac)
    print(f"  sampled {len(rows)} single-line rows; cropping...")
    crops_raw = load_gt_crops(rows, args.pad)
    keep = [i for i, c in enumerate(crops_raw) if c is not None]
    dropped = len(rows) - len(keep)
    rows = [rows[i] for i in keep]
    crops = [crops_raw[i] for i in keep]
    gts = [r["jp_text"] for r in rows]
    orients = [r["orientation"] for r in rows]
    n_vert = orients.count("vertical")
    n_horiz = orients.count("horizontal")
    print(f"  usable crops: {len(crops)} (dropped {dropped}); vert={n_vert} horiz={n_horiz}")

    t0 = time.perf_counter()
    preds_a = run_service(svc_a, crops, 1)  # A is batch=1-only export
    ta = (time.perf_counter() - t0) * 1000 / max(1, len(crops))
    t0 = time.perf_counter()
    preds_b = run_service(svc_b, crops, args.batch_size)
    tb = (time.perf_counter() - t0) * 1000 / max(1, len(crops))

    sa = score(preds_a, gts, orients)
    sb = score(preds_b, gts, orients)
    print(fmt_block(f"A ({MODEL_A})", sa))
    print(fmt_block(f"B ({MODEL_B})", sb))
    print(f"  speed: A={ta:.1f} ms/crop  B={tb:.1f} ms/crop  (B speedup {ta/tb:.2f}x)")

    ao, bo = sa["overall"], sb["overall"]
    d_cer = (bo["mean_cer"] - ao["mean_cer"]) * 100
    d_exact = (bo["exact_match_acc"] - ao["exact_match_acc"]) * 100
    test1_pass = (d_cer <= 0.5) and (d_exact >= -3.0)
    print("\n  VERDICT (Test 1, B vs A):")
    print(f"    delta mean CER:    {d_cer:+.3f} pp  (must be <= +0.50)")
    print(f"    delta exact-match: {d_exact:+.3f} pp  (must be >= -3.00)")
    print(f"    --> {'PASS' if test1_pass else 'FAIL'}")

    results["test1_single_line_gt"] = {
        "n_single_line_pool": n_single_pool,
        "n_used": len(crops), "n_dropped": dropped,
        "n_vertical": n_vert, "n_horizontal": n_horiz,
        "A": sa, "B": sb,
        "delta_mean_cer_pp": d_cer, "delta_exact_match_pp": d_exact,
        "pass": bool(test1_pass),
        "speed_ms_per_crop": {"A": ta, "B": tb, "B_speedup": ta / tb if tb else None},
    }

    # ===================================================================
    # TEST 2 -- real CTD per-line crops + artifact scan
    # ===================================================================
    print("\n" + "#" * 78)
    print("TEST 2: real CTD per-line crops (production distribution, no GT)")
    print("#" * 78)
    pages = sample_nhentai_pages(args.pages)
    print(f"  pages selected: {len(pages)}")
    test2: dict = {"n_pages": len(pages), "pages": pages}
    if not pages:
        print("  ERROR: no real pages resolved; skipping Test 2", file=sys.stderr)
        test2["error"] = "no pages resolved"
        results["test2_ctd_lines"] = test2
    else:
        try:
            ctd_crops, ctd_ids = ctd_line_crops(pages, args.pad)
        except Exception as e:
            print(f"  ERROR: CTD failed: {e}", file=sys.stderr)
            test2["error"] = f"CTD failed: {e}"
            results["test2_ctd_lines"] = test2
            ctd_crops = []
        if ctd_crops:
            print(f"  CTD produced {len(ctd_crops)} line crops")
            cpa = run_service(svc_a, ctd_crops, 1)
            cpb = run_service(svc_b, ctd_crops, args.batch_size)

            agree = [a == b for a, b in zip(cpa, cpb)]
            agreement = _mean(agree)

            art_a = [has_repeat_artifact(a) for a in cpa]
            art_b = [has_repeat_artifact(b) for b in cpb]
            # B-only artifacts: artifact in B for a crop where A has none.
            b_only_art = [i for i in range(len(cpb)) if art_b[i] and not art_a[i]]
            a_only_art = [i for i in range(len(cpa)) if art_a[i] and not art_b[i]]
            n = len(ctd_crops)

            print(f"  A-vs-B exact agreement:   {agreement*100:.2f}%  "
                  f"({sum(agree)}/{n})")
            print(f"  A artifact rate:          {_mean(art_a)*100:.2f}%  ({sum(art_a)}/{n})")
            print(f"  B artifact rate:          {_mean(art_b)*100:.2f}%  ({sum(art_b)}/{n})")
            print(f"  B-only artifacts (not A): {len(b_only_art)}  ({len(b_only_art)/n*100:.2f}%)")
            print(f"  A-only artifacts (not B): {len(a_only_art)}  ({len(a_only_art)/n*100:.2f}%)")

            # Build example list: prioritize B-only artifacts, then plain disagreements.
            examples: List[dict] = []
            seen = set()
            for i in b_only_art:
                examples.append({"crop": ctd_ids[i], "A": cpa[i], "B": cpb[i],
                                 "b_artifact": True})
                seen.add(i)
                if len(examples) >= 10:
                    break
            if len(examples) < 10:
                for i in range(n):
                    if i in seen or cpa[i] == cpb[i]:
                        continue
                    examples.append({"crop": ctd_ids[i], "A": cpa[i], "B": cpb[i],
                                     "b_artifact": False})
                    if len(examples) >= 10:
                        break

            print("\n  Examples (crop | A | B)  [* = B-only repeat artifact]:")
            for ex in examples:
                star = " *" if ex["b_artifact"] else "  "
                print(f"  {star}[{ex['crop']}]")
                print(f"       A = {ex['A']!r}")
                print(f"       B = {ex['B']!r}")

            test2.update({
                "n_crops": n,
                "ab_exact_agreement": agreement,
                "a_artifact_rate": _mean(art_a),
                "b_artifact_rate": _mean(art_b),
                "b_only_artifacts": len(b_only_art),
                "b_only_artifact_rate": len(b_only_art) / n,
                "a_only_artifacts": len(a_only_art),
                "a_only_artifact_rate": len(a_only_art) / n,
                "examples": examples,
            })
            results["test2_ctd_lines"] = test2

    # ===================================================================
    # DECISION
    # ===================================================================
    t2 = results.get("test2_ctd_lines", {})
    b_only_rate = t2.get("b_only_artifact_rate")
    a_art = t2.get("a_artifact_rate")
    b_art = t2.get("b_artifact_rate")
    # Artifacts low/comparable: B-only artifact rate <= 1% AND B not materially
    # worse than A (<= A + 1pp).
    artifacts_ok = (
        b_only_rate is not None
        and b_only_rate <= 0.01
        and (b_art is None or a_art is None or (b_art - a_art) <= 0.01)
    )
    ship = bool(test1_pass and artifacts_ok)
    decision = {
        "test1_pass": bool(test1_pass),
        "artifacts_ok": bool(artifacts_ok),
        "verdict": "SHIP B" if ship else "DO NOT SHIP",
    }
    if not ship:
        if not test1_pass:
            decision["reason"] = "B regresses CER/exact-match on single-line GT"
        elif b_only_rate is None:
            decision["reason"] = "Test 2 could not run (no CTD crops); artifacts unverified"
        else:
            decision["reason"] = "B introduces repeat artifacts on real single-line crops"
    results["decision"] = decision

    print("\n" + "=" * 78)
    print(f"DECISION: {decision['verdict']}")
    print("=" * 78)
    print(f"  Test 1 (single-line CER bar):  {'PASS' if test1_pass else 'FAIL'}")
    print(f"  Test 2 (artifacts comparable): {'OK' if artifacts_ok else 'NOT OK'}")
    if "reason" in decision:
        print(f"  Reason: {decision['reason']}")

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nSaved results: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

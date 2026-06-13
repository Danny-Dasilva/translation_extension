"""Compare PARSeq ONNX exports for manga OCR accuracy.

Measurement-only A/B between:
  A = parseq_manga_best_ep60_AR_single.onnx  (production, batch=1, head_dim 4407)
  B = parseq_manga_large_5p16.fp16.onnx       (batch-capable, head_dim 4401)

Ground truth comes from the Manga109-s annotations (one `<text>` bbox per
bubble, with the transcribed JP). Each sampled bubble is cropped from the
page JPEG, run through BOTH models with the SHARED preprocessing pipeline
(copied from ParseqOCRService), and scored against the annotated text.

Reports per model:
  - exact-match accuracy vs ground truth
  - mean / median character error rate (CER) vs ground truth
Plus inter-model:
  - exact-match agreement between A and B
  - mean CER between A and B (treating A as reference)
  - a sample of disagreements for human inspection

The two models share charset indices 1..4400 (B's charset is a strict prefix
of A's; A appends 6 extra glyphs). So argmax indices are directly
label-comparable for all but those 6 rare symbols.

Usage (run with the backend venv that has onnxruntime-gpu):
    backend/.venv/bin/python backend/scripts/eval_vision/compare_parseq_exports.py \
        --n 300 --batch-size 24 --seed 0

Does NOT touch config.py / parseq_ocr_service.py / production model.
"""
from __future__ import annotations

# torch first so its bundled CUDA libs are on the loader path before
# onnxruntime-gpu probes for libcublas/libcudnn (mirrors ParseqOCRService).
import torch  # noqa: F401

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort
import polars as pl

# Production OCR postprocessing (NFC, fullwidth<->halfwidth, punct map,
# middle-dot collapse, ...). Applied as a SECOND scoring pass so we also see
# accuracy under the real deployed pipeline, not just raw model logits.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/ on path
try:
    from app.utils.ocr_postprocess import apply_all as postprocess_ocr
except Exception as _pp_err:  # pragma: no cover - postprocess optional
    print(f"WARNING: ocr_postprocess import failed: {_pp_err}", file=sys.stderr)
    postprocess_ocr = None

ort.set_default_logger_severity(3)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = BACKEND_ROOT / "models"
BUBBLES_PARQUET = BACKEND_ROOT / "scripts" / "data" / "manga109" / "bubbles.parquet"
MANGA109_IMAGES = Path(
    "/mnt/nas/drive_2/manga-ml/datasets/manga109s/"
    "Manga109s_released_2023_12_07/images"
)

MODEL_A = "parseq_manga_best_ep60_AR_single.onnx"  # production
MODEL_B = "parseq_manga_large_5p16.fp16.onnx"      # batch-capable candidate


# --------------------------------------------------------------------------
# Shared preprocessing / decode (lifted from ParseqOCRService, no behavior
# change to the service itself).
# --------------------------------------------------------------------------
def _maybe_rotate_vertical(crop: np.ndarray, thresh_aspect: float = 1.5) -> np.ndarray:
    h, w = crop.shape[:2]
    if h > thresh_aspect * w:
        return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop


def preprocess(crops: List[np.ndarray], img_h: int, img_w: int,
               mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    batch = np.empty((len(crops), 3, img_h, img_w), dtype=np.float32)
    for i, crop in enumerate(crops):
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
        crop = _maybe_rotate_vertical(crop)
        resized = cv2.resize(crop, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        batch[i] = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    batch -= mean
    batch /= std
    return batch


class OnnxParseq:
    """Minimal ONNX PARSeq runner mirroring ParseqOCRService decode."""

    def __init__(self, model_path: Path, meta_path: Path):
        meta = json.loads(meta_path.read_text())
        self.charset: str = meta["charset"]
        self.img_h, self.img_w = meta["img_size"]
        self.eos_id: int = meta["eos_id"]
        self.mean = np.array(meta["normalize_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(meta["normalize_std"], dtype=np.float32).reshape(1, 3, 1, 1)
        self._itos: List[str] = ["[E]"] + list(self.charset)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
            "CPUExecutionProvider",
        ]
        self.session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
        # warm + surface CUDA failures up front
        dummy = np.zeros((1, 3, self.img_h, self.img_w), dtype=np.float32)
        self.session.run(None, {self.session.get_inputs()[0].name: dummy})
        self.providers = self.session.get_providers()
        self.device = "cuda" if "CUDAExecutionProvider" in self.providers else "cpu"
        self._input_name = self.session.get_inputs()[0].name

    def _decode(self, logits: np.ndarray) -> List[str]:
        ids = logits.argmax(-1)
        texts: List[str] = []
        for row in ids:
            chars: List[str] = []
            for tok in row:
                if tok == self.eos_id:
                    break
                if 0 < tok < len(self._itos):
                    chars.append(self._itos[int(tok)])
            texts.append("".join(chars))
        return texts

    def run(self, crops: List[np.ndarray], batch_size: int) -> List[str]:
        """Raw decode (NO postprocessing) so we measure model output, not
        the postprocess pipeline. batch_size is clamped to 1 for batch=1
        exports automatically by falling back on RuntimeException."""
        out: List[str] = []
        i = 0
        bs = batch_size
        while i < len(crops):
            chunk = crops[i:i + bs]
            batch = preprocess(chunk, self.img_h, self.img_w, self.mean, self.std)
            try:
                logits = self.session.run(None, {self._input_name: batch})[0]
            except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
                if bs > 1:
                    # batch=1-only export (AR_single Reshape node) -> fall back
                    bs = 1
                    continue
                raise RuntimeError(f"ONNX inference failed at batch=1: {e}") from e
            out.extend(self._decode(logits))
            i += len(chunk)
        return out


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


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def sample_bubbles(n: int, seed: int, min_chars: int, pad: int) -> List[dict]:
    df = pl.read_parquet(BUBBLES_PARQUET)
    df = df.filter(pl.col("jp_text").str.len_chars() >= min_chars)
    df = df.sample(n=min(n * 3, len(df)), seed=seed)  # oversample; some images may be missing
    rows = df.to_dicts()
    out: List[dict] = []
    for r in rows:
        page_path = MANGA109_IMAGES / r["book"] / f"{int(r['page']):03d}.jpg"
        if not page_path.exists():
            continue
        out.append({**r, "page_path": str(page_path), "pad": pad})
        if len(out) >= n:
            break
    return out


def load_crops(samples: List[dict]) -> List[np.ndarray]:
    crops: List[np.ndarray] = []
    cache: dict[str, np.ndarray] = {}
    for s in samples:
        img = cache.get(s["page_path"])
        if img is None:
            img = cv2.imread(s["page_path"])
            cache[s["page_path"]] = img
        h, w = img.shape[:2]
        p = s["pad"]
        x0 = max(0, s["xmin"] - p); y0 = max(0, s["ymin"] - p)
        x1 = min(w, s["xmax"] + p); y1 = min(h, s["ymax"] + p)
        crops.append(img[y0:y1, x0:x1].copy())
    return crops


def sample_pages(n_pages: int, seed: int) -> List[str]:
    """Sample distinct page-image paths from the bubble parquet."""
    df = pl.read_parquet(BUBBLES_PARQUET).select(["book", "page"]).unique()
    df = df.sample(n=min(n_pages * 3, len(df)), seed=seed)
    out: List[str] = []
    for r in df.to_dicts():
        p = MANGA109_IMAGES / r["book"] / f"{int(r['page']):03d}.jpg"
        if p.exists():
            out.append(str(p))
        if len(out) >= n_pages:
            break
    return out


def ctd_line_crops(page_paths: List[str], pad: int) -> List[np.ndarray]:
    """Run the production CTD detector on each page, return per-line crops.

    This reproduces the production OCR feed: PARSeq is a single-line STR model
    and is fed CTD `text_lines` crops, not whole bubbles.
    """
    import asyncio
    from app.services.ctd_service import ComicTextDetectorService

    ctd = ComicTextDetectorService()
    crops: List[np.ndarray] = []

    async def _run():
        for pp in page_paths:
            img = cv2.imread(pp)
            if img is None:
                continue
            det = await ctd.detect(img, input_is_bgr=True)
            h, w = img.shape[:2]
            for ln in det.get("text_lines", []):
                x0 = max(0, int(ln["minX"]) - pad); y0 = max(0, int(ln["minY"]) - pad)
                x1 = min(w, int(ln["maxX"]) + pad); y1 = min(h, int(ln["maxY"]) + pad)
                if x1 > x0 and y1 > y0:
                    crops.append(img[y0:y1, x0:x1].copy())

    asyncio.run(_run())
    return crops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bubble-gt", "ctd-lines"], default="bubble-gt",
                    help="bubble-gt: whole-bubble crops vs manga109 ground truth. "
                         "ctd-lines: production CTD per-line crops, inter-model only.")
    ap.add_argument("--n", type=int, default=300, help="number of bubbles to sample (bubble-gt)")
    ap.add_argument("--pages", type=int, default=40, help="number of pages to detect (ctd-lines)")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=2,
                    help="skip single-char bubbles (noisy ground truth)")
    ap.add_argument("--pad", type=int, default=2)
    ap.add_argument("--max-disagreements", type=int, default=25)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = str(Path(__file__).resolve().parent /
                       f"compare_parseq_exports_{args.mode}.json")

    if not BUBBLES_PARQUET.exists():
        print(f"ERROR: bubbles parquet not found: {BUBBLES_PARQUET}", file=sys.stderr)
        return 1
    if not MANGA109_IMAGES.exists():
        print(f"ERROR: manga109 images dir not found: {MANGA109_IMAGES}", file=sys.stderr)
        return 1

    if args.mode == "ctd-lines":
        print(f"Sampling {args.pages} pages, running CTD for per-line crops...")
        pages = sample_pages(args.pages, args.seed)
        crops = ctd_line_crops(pages, args.pad)
        samples = [{"book": "?", "page": -1, "text_id": str(i), "jp_text": ""}
                   for i in range(len(crops))]
        gts = ["" for _ in crops]  # no per-line ground truth available
        print(f"Loaded {len(crops)} CTD line crops from {len(pages)} pages")
    else:
        print(f"Sampling {args.n} bubbles (seed={args.seed}, min_chars={args.min_chars})...")
        samples = sample_bubbles(args.n, args.seed, args.min_chars, args.pad)
        if len(samples) < args.n:
            print(f"WARNING: only resolved {len(samples)} bubbles to existing images")
        crops = load_crops(samples)
        gts = [s["jp_text"] for s in samples]
        print(f"Loaded {len(crops)} crops")

    if not crops:
        print("ERROR: no crops produced", file=sys.stderr)
        return 1
    has_gt = args.mode != "ctd-lines"

    meta_a = MODELS_DIR / "parseq_manga_best_ep60_AR_single.json"
    meta_b = MODELS_DIR / "parseq_manga_large_5p16.json"

    print(f"Loading A (production): {MODEL_A}")
    model_a = OnnxParseq(MODELS_DIR / MODEL_A, meta_a)
    print(f"  -> device={model_a.device} providers={model_a.providers}")
    print(f"Loading B (candidate):  {MODEL_B}")
    model_b = OnnxParseq(MODELS_DIR / MODEL_B, meta_b)
    print(f"  -> device={model_b.device} providers={model_b.providers}")

    t0 = time.perf_counter()
    preds_a = model_a.run(crops, batch_size=args.batch_size)  # auto-falls to bs=1
    ta = time.perf_counter() - t0
    t0 = time.perf_counter()
    preds_b = model_b.run(crops, batch_size=args.batch_size)
    tb = time.perf_counter() - t0
    print(f"A: {len(preds_a)} crops in {ta*1000:.0f}ms ({ta*1000/len(crops):.1f}ms/crop)")
    print(f"B: {len(preds_b)} crops in {tb*1000:.0f}ms ({tb*1000/len(crops):.1f}ms/crop)")

    cer_a = [cer(p, g) for p, g in zip(preds_a, gts)]
    cer_b = [cer(p, g) for p, g in zip(preds_b, gts)]
    cer_ab = [cer(pb, pa) for pa, pb in zip(preds_a, preds_b)]  # B vs A as ref
    exact_a = [p == g for p, g in zip(preds_a, gts)]
    exact_b = [p == g for p, g in zip(preds_b, gts)]
    agree = [pa == pb for pa, pb in zip(preds_a, preds_b)]

    # Second pass under production postprocessing (normalizes punctuation /
    # fullwidth-halfwidth so we compare what users actually see).
    if postprocess_ocr is not None:
        pp = postprocess_ocr
        pa_pp = [pp(p) for p in preds_a]
        pb_pp = [pp(p) for p in preds_b]
        gt_pp = [pp(g) for g in gts]
        cer_a_pp = [cer(p, g) for p, g in zip(pa_pp, gt_pp)]
        cer_b_pp = [cer(p, g) for p, g in zip(pb_pp, gt_pp)]
        exact_a_pp = [p == g for p, g in zip(pa_pp, gt_pp)]
        exact_b_pp = [p == g for p, g in zip(pb_pp, gt_pp)]
        agree_pp = [a == b for a, b in zip(pa_pp, pb_pp)]
        cer_ab_pp = [cer(b, a) for a, b in zip(pa_pp, pb_pp)]
    else:
        cer_a_pp = cer_b_pp = exact_a_pp = exact_b_pp = agree_pp = cer_ab_pp = []

    def m(x): return float(np.mean(x)) if x else 0.0
    def med(x): return float(np.median(x)) if x else 0.0

    summary = {
        "n_crops": len(crops),
        "batch_size_requested": args.batch_size,
        "model_a": MODEL_A,
        "model_b": MODEL_B,
        "ground_truth": "manga109-s annotations (jp_text per <text> bbox)",
        "model_a_vs_gt": {
            "exact_match_acc": m(exact_a),
            "mean_cer": m(cer_a),
            "median_cer": med(cer_a),
        },
        "model_b_vs_gt": {
            "exact_match_acc": m(exact_b),
            "mean_cer": m(cer_b),
            "median_cer": med(cer_b),
        },
        "inter_model": {
            "exact_agreement": m(agree),
            "mean_cer_b_vs_a": m(cer_ab),
        },
        "postprocessed": {
            "model_a_vs_gt": {"exact_match_acc": m(exact_a_pp), "mean_cer": m(cer_a_pp), "median_cer": med(cer_a_pp)},
            "model_b_vs_gt": {"exact_match_acc": m(exact_b_pp), "mean_cer": m(cer_b_pp), "median_cer": med(cer_b_pp)},
            "inter_model": {"exact_agreement": m(agree_pp), "mean_cer_b_vs_a": m(cer_ab_pp)},
        },
        "timing": {
            "a_ms_per_crop": ta * 1000 / len(crops),
            "b_ms_per_crop": tb * 1000 / len(crops),
            "speedup_b_over_a": ta / tb if tb else None,
        },
    }

    # Disagreements where models differ, sorted by inter-model CER desc.
    diffs = []
    for s, pa, pb, g, c in zip(samples, preds_a, preds_b, gts, cer_ab):
        if pa != pb:
            diffs.append({
                "src": f"{s['book']}:p{s['page']}:{s['text_id']}",
                "ground_truth": g,
                "model_a": pa,
                "model_b": pb,
                "a_correct": pa == g,
                "b_correct": pb == g,
            })
    diffs.sort(key=lambda d: cer(d["model_b"], d["model_a"]), reverse=True)
    sample_diffs = diffs[:args.max_disagreements]

    summary["mode"] = args.mode
    print("\n" + "=" * 70)
    if has_gt:
        print("RESULTS (ground truth = Manga109-s annotations, whole-bubble crops)")
    else:
        print("RESULTS (production CTD per-line crops, INTER-MODEL ONLY, no GT)")
    print("=" * 70)
    print(f"  N crops:               {summary['n_crops']}")
    if has_gt:
        print(f"  A exact-match acc:     {summary['model_a_vs_gt']['exact_match_acc']*100:.2f}%")
        print(f"  B exact-match acc:     {summary['model_b_vs_gt']['exact_match_acc']*100:.2f}%")
        print(f"  A mean CER:            {summary['model_a_vs_gt']['mean_cer']*100:.2f}%   (median {summary['model_a_vs_gt']['median_cer']*100:.2f}%)")
        print(f"  B mean CER:            {summary['model_b_vs_gt']['mean_cer']*100:.2f}%   (median {summary['model_b_vs_gt']['median_cer']*100:.2f}%)")
    print(f"  Inter-model agreement: {summary['inter_model']['exact_agreement']*100:.2f}%")
    print(f"  Inter-model CER (B|A): {summary['inter_model']['mean_cer_b_vs_a']*100:.2f}%")
    print(f"  A speed:               {summary['timing']['a_ms_per_crop']:.1f} ms/crop")
    print(f"  B speed:               {summary['timing']['b_ms_per_crop']:.1f} ms/crop")
    sp = summary['timing']['speedup_b_over_a']
    print(f"  B speedup over A:      {sp:.2f}x" if sp else "  B speedup: n/a")
    if postprocess_ocr is not None:
        ppp = summary["postprocessed"]
        print("  --- under production postprocessing ---")
        if has_gt:
            print(f"  A exact-match acc:     {ppp['model_a_vs_gt']['exact_match_acc']*100:.2f}%")
            print(f"  B exact-match acc:     {ppp['model_b_vs_gt']['exact_match_acc']*100:.2f}%")
            print(f"  A mean CER:            {ppp['model_a_vs_gt']['mean_cer']*100:.2f}%   (median {ppp['model_a_vs_gt']['median_cer']*100:.2f}%)")
            print(f"  B mean CER:            {ppp['model_b_vs_gt']['mean_cer']*100:.2f}%   (median {ppp['model_b_vs_gt']['median_cer']*100:.2f}%)")
        print(f"  Inter-model agreement: {ppp['inter_model']['exact_agreement']*100:.2f}%")
        print(f"  Inter-model CER (B|A): {ppp['inter_model']['mean_cer_b_vs_a']*100:.2f}%")
    print(f"  Total disagreements:   {len(diffs)} / {len(crops)}  (raw, pre-postprocess)")

    # of disagreements, who is right
    a_right_only = sum(1 for d in diffs if d["a_correct"] and not d["b_correct"])
    b_right_only = sum(1 for d in diffs if d["b_correct"] and not d["a_correct"])
    both_wrong = sum(1 for d in diffs if not d["a_correct"] and not d["b_correct"])
    if has_gt:
        print(f"    A right / B wrong:   {a_right_only}")
        print(f"    B right / A wrong:   {b_right_only}")
        print(f"    both wrong:          {both_wrong}")

    print("\nSample disagreements (highest inter-model CER first):")
    for d in sample_diffs[:15]:
        tag = "A✓" if d["a_correct"] else ("B✓" if d["b_correct"] else "--")
        print(f"  [{tag}] GT={d['ground_truth']!r}")
        print(f"        A ={d['model_a']!r}")
        print(f"        B ={d['model_b']!r}")

    summary["disagreement_breakdown"] = {
        "total": len(diffs),
        "a_right_b_wrong": a_right_only,
        "b_right_a_wrong": b_right_only,
        "both_wrong": both_wrong,
    }
    summary["sample_disagreements"] = sample_diffs
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CPU-only JP-side OCR over the STAGED ``available_pairs.jsonl`` corpus.

``run_gallery.py`` drives off ``manifest_pages_375k.jsonl`` + a ``galleries/``
layout; the real staged corpus instead lives under
``/home/danny/manga_corpus_staging/avail/<gid>_<lang>/NNN.jpg`` and is indexed by
``available_pairs.jsonl``. This driver consumes THAT layout directly, runs the
JP detect+OCR half (CTD v26 + PARSeq) on the CPU onnxruntime provider, and caches
per-gallery JP detections (text + bbox + ocr_conf, reading-order-sortable) so the
EN VLM pass + alignment can consume them once the GPU frees.

Faithful-to-production choices:
  * CUDA is hard-disabled in-process (never contends with / touches the GPU).
  * Detector = ComicTextDetectorService() default = v26 round9 onofix.
  * Recognizer = the PRODUCTION ``settings.parseq_model_path`` (ep60 nonAR fp16),
    NOT the ``build_jp_engines`` constructor default (large_5p16). This matches
    what serving emits and what the garble gate (0.65 floor) was calibrated on.

Resumable: a gallery whose ``<en_gid>_<jp_gid>.jp_ocr.json`` already exists is
skipped. Output is LOCAL ONLY (never /mnt/nas).

    cd backend && CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        scripts/data/corpus_bitext/run_jp_avail.py \
        --out-dir /home/danny/manga_corpus_staging/jp_ocr_cache \
        --time-budget-sec 5400 --resume
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

# GPU AR re-mine (v11fix9): allow CUDA so the AR PARSeq path runs on the idle
# local GPU. (Was force-CPU to avoid training-run contention; now defaults to
# GPU 0 but stays overridable via the env for a CPU run.)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Surface OCR-service warnings (AR OOM / AR-decode-fallback) — parseq_ocr_service
# uses stdlib logging; without a configured handler these were silently dropped,
# which is why the broken AR re-mine looked clean while emptying every crop.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parents[2]
_REPO = _BACKEND.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BACKEND))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def page_sort_key(p: Path):
    digits = "".join(ch for ch in p.stem if ch.isdigit())
    return (int(digits) if digits else 0, p.name)


def list_pages(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    imgs = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS]
    return sorted(imgs, key=page_sort_key)


def load_pairs(avail_path: Path):
    return [json.loads(l) for l in avail_path.open() if l.strip()]


def load_manifest_index(manifest: Path) -> dict:
    idx = {}
    if not manifest.exists():
        return idx
    for l in manifest.open():
        r = json.loads(l)
        idx[(r.get("en_gid"), r.get("jp_gid"))] = r
    return idx


def rank_pairs(pairs: list[dict], manifest_idx: dict) -> list[dict]:
    """Best-first: good before partial, then tightest redraw alignment
    (low aligned_avg hamming), then highest page coverage."""
    bucket_rank = {"good": 0, "partial": 1}

    def key(r):
        m = manifest_idx.get((r["en_gid"], r["jp_gid"]), {})
        aa = m.get("aligned_avg")
        aa = aa if aa is not None else 1e9
        cov_e = m.get("coverage_en") or 0.0
        cov_j = m.get("coverage_jp") or 0.0
        cov = (cov_e + cov_j) / 2.0
        return (bucket_rank.get(r.get("source_bucket"), 9), aa, -cov)

    ranked = sorted(pairs, key=key)
    # attach the join metrics for the cache/report
    for r in ranked:
        m = manifest_idx.get((r["en_gid"], r["jp_gid"]), {})
        r["_aligned_avg"] = m.get("aligned_avg")
        r["_coverage_en"] = m.get("coverage_en")
        r["_coverage_jp"] = m.get("coverage_jp")
        r["_aligned_count"] = m.get("aligned_count")
    return ranked


def conf_bucket(c: float) -> str:
    if c >= 0.95:
        return "0.95-1.0"
    if c >= 0.85:
        return "0.85-0.95"
    if c >= 0.75:
        return "0.75-0.85"
    if c >= 0.65:
        return "0.65-0.75"
    return "<0.65"


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    tmp.replace(path)


async def main_async(args) -> int:
    from app.config import settings
    from app.services.ctd_service import ComicTextDetectorService
    from app.services.parseq_ocr_service import ParseqOCRService
    from app.utils.ocr_confidence_gate import is_garbled_low_conf
    from ocr_adapters import ocr_jp_page  # type: ignore

    parseq_model = args.parseq_model or settings.parseq_model_path
    ctd_model = settings.ctd_model_path

    detector = ComicTextDetectorService()
    ocr = ParseqOCRService(
        model_path=parseq_model,
        hybrid_enabled=True,       # GPU AR re-mine: route vertical kana to AR
        ar_model_path=settings.parseq_ar_model_path,
        vertical_ar_default=True,
        vertical_ar_aspect=settings.ocr_vertical_ar_aspect,
    )
    prov = detector.session.get_providers() if hasattr(detector, "session") else "n/a"
    print(f"CTD providers: {prov}")
    print(f"CTD model:    {ctd_model}")
    print(f"PARSeq model: {parseq_model}")
    assert "CUDAExecutionProvider" in (prov or []), f"CUDA EP not active (providers={prov}) -- AR re-mine requires GPU; check LD_LIBRARY_PATH/cuda13-libs"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args.available_pairs)
    manifest_idx = load_manifest_index(args.manifest)
    ranked = rank_pairs(pairs, manifest_idx)
    if args.limit_pairs:
        ranked = ranked[: args.limit_pairs]
    print(f"pairs={len(pairs)} ranked(processing)={len(ranked)} out={args.out_dir}")

    t_start = time.time()
    grand = {
        "pairs_done": 0, "pages_done": 0, "bubbles": 0,
        "garble_drops": 0, "empty_short_drops": 0,
        "conf_hist": Counter(), "page_secs": [],
        "pairs_with_no_jp_dir": 0, "pages_failed": 0,
    }

    for n, r in enumerate(ranked, 1):
        en_gid, jp_gid = r["en_gid"], r["jp_gid"]
        cache_path = args.out_dir / f"{en_gid}_{jp_gid}.jp_ocr.json"
        if args.resume and cache_path.exists():
            continue
        if time.time() - t_start >= args.time_budget_sec:
            print(f"[time-budget {args.time_budget_sec}s reached] stopping before pair {n}")
            break

        jp_dir = Path(r["jp_dir"])
        pages = list_pages(jp_dir)
        if not pages:
            grand["pairs_with_no_jp_dir"] += 1
            print(f"[{n}/{len(ranked)}] {en_gid}_{jp_gid}: NO jp pages at {jp_dir}")
            continue

        page_records = []
        g_bub = g_garble = g_emptyshort = g_fail = 0
        for img_path in pages:
            if time.time() - t_start >= args.time_budget_sec:
                break
            t0 = time.time()
            try:
                bubbles = await ocr_jp_page(img_path, detector, ocr)  # opens image internally
            except Exception as e:  # noqa: BLE001
                g_fail += 1
                page_records.append({"jp_page": img_path.name, "error": str(e)[:200], "bubbles": []})
                continue
            dt = time.time() - t0
            grand["page_secs"].append(dt)
            # per-bubble garble-gate audit (does not mutate the cached OCR)
            for b in bubbles:
                c = float(b.get("conf") or 0.0)
                grand["conf_hist"][conf_bucket(c)] += 1
                g_bub += 1
                txt = (b.get("text") or "").strip()
                if not txt:
                    g_emptyshort += 1
                    b["gate"] = "empty"
                elif is_garbled_low_conf(txt, c, conf_threshold=0.65):
                    g_garble += 1
                    b["gate"] = "garble"
                else:
                    b["gate"] = "keep"
            page_records.append({
                "jp_page": img_path.name,
                "n_bubbles": len(bubbles),
                "sec": round(dt, 3),
                "bubbles": bubbles,
            })

        kept = g_bub - g_garble - g_emptyshort
        rec = {
            "en_gid": en_gid, "jp_gid": jp_gid,
            "source_bucket": r.get("source_bucket"),
            "title": r.get("title"),
            "jp_dir": str(jp_dir),
            "en_dir": r.get("en_dir"),
            "ctd_model": ctd_model,
            "parseq_model": parseq_model,
            "aligned_avg": r.get("_aligned_avg"),
            "coverage_en": r.get("_coverage_en"),
            "coverage_jp": r.get("_coverage_jp"),
            "page_alignment": r.get("page_alignment"),  # en_page <-> jp_page join for align step
            "summary": {
                "n_pages": len(page_records),
                "n_bubbles": g_bub,
                "garble_drops": g_garble,
                "empty_short_drops": g_emptyshort,
                "kept_bubbles": kept,
                "pages_failed": g_fail,
            },
            "pages": page_records,
        }
        atomic_write_json(cache_path, rec)

        grand["pairs_done"] += 1
        grand["pages_done"] += len(page_records)
        grand["bubbles"] += g_bub
        grand["garble_drops"] += g_garble
        grand["empty_short_drops"] += g_emptyshort
        grand["pages_failed"] += g_fail
        spp = (sum(grand["page_secs"][-len(page_records):]) / max(1, len(page_records)))
        elapsed = time.time() - t_start
        print(f"[{n}/{len(ranked)}] {en_gid}_{jp_gid} ({r.get('source_bucket')}, "
              f"aa={r.get('_aligned_avg')}): {len(page_records)}pg {g_bub}bub "
              f"garble={g_garble} kept={kept} {spp:.2f}s/pg | "
              f"tot {grand['pairs_done']}pr/{grand['pages_done']}pg {elapsed:.0f}s")

    secs = grand.pop("page_secs")
    grand["sec_per_page"] = round(sum(secs) / len(secs), 3) if secs else None
    grand["median_sec_per_page"] = round(sorted(secs)[len(secs) // 2], 3) if secs else None
    grand["conf_hist"] = dict(grand["conf_hist"])
    grand["wall_sec"] = round(time.time() - t_start, 1)
    grand["garble_rate"] = round(grand["garble_drops"] / grand["bubbles"], 4) if grand["bubbles"] else None
    atomic_write_json(args.out_dir / "_grand_stats.json", grand)
    print("\n==== GRAND STATS ====")
    print(json.dumps(grand, ensure_ascii=False, indent=2))
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--available-pairs", type=Path,
                    default=Path("/home/danny/manga_corpus_staging/available_pairs.jsonl"))
    ap.add_argument("--manifest", type=Path,
                    default=_REPO / "data/manga_datasets/merged/export/manifest_pairs_375k.jsonl")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/home/danny/manga_corpus_staging/jp_ocr_cache"))
    ap.add_argument("--parseq-model", default="", help="override; default = production settings.parseq_model_path")
    ap.add_argument("--time-budget-sec", type=float, default=5400.0)
    ap.add_argument("--limit-pairs", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    return ap


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(build_argparser().parse_args())))

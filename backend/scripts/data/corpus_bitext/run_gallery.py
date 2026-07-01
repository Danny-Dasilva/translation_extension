"""Mine curated bitext over the page-aligned corpus, per gallery, to LOCAL shards.

Reads ``manifest_pages_375k.jsonl`` (one JP-page <-> EN-page row), remaps the NAS
image paths to the local staging root, and for each gallery:

    for each (jp_page, en_page):
        jp_bubbles = CTD detect + PARSeq OCR (JP)        [CPU-capable]
        en_bubbles = VLM per-bubble transcription (EN)   [GPU / remote, deferred]
        rows      += align -> curate -> v11 page-context format

and writes ``<out_dir>/<en_gid>_<jp_gid>.parquet`` + a per-gallery stats json.
Resumable (skips galleries whose shard already exists). Output is LOCAL ONLY
(never /mnt/nas -- the NAS share silently reaps output dirs).

Run (once images land in --staging-root):

    cd backend && .venv/bin/python scripts/data/corpus_bitext/run_gallery.py \
        --staging-root /home/danny/manga_corpus_staging \
        --status good,partial --out-dir scripts/data/corpus_bitext/shards \
        --vlm-coord-norm 1000              # if EN VLM is Qwen3-VL (0-1000 coords)

Until the GPU is free, ``--jp-only`` runs/validates the JP OCR half and writes the
detected JP bubbles per page (no EN, no rows) so the detect+OCR half can be
smoke-tested at scale on CPU.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parents[2]
_REPO = _BACKEND.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BACKEND))

from curate import CurationStats  # type: ignore  # noqa: E402
from format_rows import COLS, assemble_parquet  # type: ignore  # noqa: E402
from pipeline import PipelineConfig, align_and_curate  # type: ignore  # noqa: E402

DEFAULT_MANIFEST = _REPO / "data/manga_datasets/merged/export/manifest_pages_375k.jsonl"
DEFAULT_NAS_PREFIX = "/mnt/nas/drive_2/manga-ml/ehentai_corpus"


def remap_path(nas_path: str, nas_prefix: str, staging_root: Path) -> Path:
    """Map a NAS image path to the local staging root, preserving the
    ``galleries/<gid>_<lang>/<file>`` tail."""
    p = nas_path
    if p.startswith(nas_prefix):
        rel = p[len(nas_prefix):].lstrip("/")
    elif "galleries/" in p:
        rel = p[p.index("galleries/"):]
    else:
        rel = Path(p).name
    return staging_root / rel


def page_num_from_path(path: str) -> int:
    stem = Path(path).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def load_galleries(manifest: Path, statuses: set[str]):
    galleries: dict[tuple, list[dict]] = defaultdict(list)
    for line in manifest.open():
        r = json.loads(line)
        if statuses and r.get("source_bucket") not in statuses and r.get("status") not in statuses:
            continue
        galleries[(r["en_gid"], r["jp_gid"])].append(r)
    return galleries


async def process_gallery(rows, cfg, detector, ocr, args, stats: CurationStats):
    from ocr_adapters import ocr_jp_page, transcribe_en_page  # type: ignore

    seen: set = set()
    out_rows: list[dict] = []
    jp_only_dump: list[dict] = []
    for r in sorted(rows, key=lambda x: page_num_from_path(x["en_path"])):
        jp_path = remap_path(r["jp_path"], args.nas_prefix, args.staging_root)
        en_path = remap_path(r["en_path"], args.nas_prefix, args.staging_root)
        page = page_num_from_path(r["en_path"])
        if not jp_path.exists():
            continue
        try:
            jp_bubbles = await ocr_jp_page(jp_path, detector, ocr)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] JP OCR failed {jp_path}: {e}")
            continue
        if args.jp_only:
            jp_only_dump.append({"page": page, "n_bubbles": len(jp_bubbles),
                                 "bubbles": jp_bubbles})
            continue
        if not en_path.exists():
            continue
        try:
            en_bubbles = transcribe_en_page(
                en_path, endpoint=args.vlm_endpoint, coord_norm=args.vlm_coord_norm)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] EN VLM failed {en_path}: {e}")
            continue
        gid_tag = f"{r['en_gid']}_{r['jp_gid']}"
        rows_pg, _ = align_and_curate(jp_bubbles, en_bubbles, gid_tag, page, cfg, stats, seen)
        out_rows.extend(rows_pg)
    return out_rows, jp_only_dump


async def main_async(args) -> int:
    from ocr_adapters import build_jp_engines  # type: ignore

    statuses = set(s.strip() for s in args.status.split(",") if s.strip())
    galleries = load_galleries(args.manifest, statuses)
    keys = list(galleries.keys())
    if args.limit_galleries:
        keys = keys[: args.limit_galleries]
    print(f"galleries={len(galleries)}  processing={len(keys)}  statuses={statuses}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detector, ocr = build_jp_engines(cpu_only=args.cpu_only)
    cfg = PipelineConfig.default()

    grand = CurationStats()
    total_rows = 0
    for n, key in enumerate(keys, 1):
        en_gid, jp_gid = key
        shard = args.out_dir / f"{en_gid}_{jp_gid}.parquet"
        if args.resume and shard.exists():
            continue
        gstats = CurationStats()
        rows, jp_dump = await process_gallery(galleries[key], cfg, detector, ocr, args, gstats)
        if args.jp_only:
            (args.out_dir / f"{en_gid}_{jp_gid}.jp_ocr.json").write_text(
                json.dumps(jp_dump, ensure_ascii=False))
            print(f"[{n}/{len(keys)}] {en_gid}_{jp_gid}: JP-only {sum(d['n_bubbles'] for d in jp_dump)} bubbles / {len(jp_dump)} pages")
            continue
        pq = assemble_parquet(rows, shard)
        (args.out_dir / f"{en_gid}_{jp_gid}.stats.json").write_text(
            json.dumps({"parquet": pq, "curation": gstats.as_dict()}, ensure_ascii=False, indent=2))
        total_rows += pq["rows"]
        for fld in gstats.__dict__:
            if fld == "quality_hist":
                grand.quality_hist += gstats.quality_hist
            else:
                setattr(grand, fld, getattr(grand, fld) + getattr(gstats, fld))
        print(f"[{n}/{len(keys)}] {en_gid}_{jp_gid}: {pq['rows']} rows "
              f"(nsfw {pq['nsfw_frac']})  running_total={total_rows}")

    if not args.jp_only:
        # merge shards into one parquet for convenience
        shards = sorted(args.out_dir.glob("*.parquet"))
        if shards:
            merged = pl.concat([pl.read_parquet(s).select(COLS) for s in shards], how="vertical")
            merged_path = args.out_dir / "data_corpus_bitext_pagecontext.parquet"
            merged.write_parquet(merged_path)
            print(f"\nMERGED {len(shards)} shards -> {merged_path} ({merged.height} rows)")
        (args.out_dir / "grand_stats.json").write_text(
            json.dumps(grand.as_dict(), ensure_ascii=False, indent=2))
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--staging-root", type=Path, default=Path("/home/danny/manga_corpus_staging"))
    ap.add_argument("--nas-prefix", default=DEFAULT_NAS_PREFIX)
    ap.add_argument("--out-dir", type=Path, default=_HERE / "shards")
    ap.add_argument("--status", default="good,partial")
    ap.add_argument("--limit-galleries", type=int, default=0)
    ap.add_argument("--vlm-endpoint", default="http://100.64.235.63:8001/v1/chat/completions")
    ap.add_argument("--vlm-coord-norm", type=int, default=0, help="1000 for Qwen3-VL 0-1000 grid coords")
    ap.add_argument("--cpu-only", action="store_true", help="force JP OCR onto the CPU onnxruntime provider")
    ap.add_argument("--jp-only", action="store_true", help="only run+dump JP OCR (EN VLM deferred)")
    ap.add_argument("--resume", action="store_true")
    return ap


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(build_argparser().parse_args())))

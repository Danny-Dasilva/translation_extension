"""CPU-only smoke test of the REAL JP detect+OCR half (no GPU needed).

Runs CTD (v26) detection + PARSeq recognition on a few Ikenie JP source pages
(``.bench/<chapter>_v11fix6_box_insp/<NNN>/01_source.webp``) on the CPU
onnxruntime provider, then feeds those REAL JP bubbles + the gold EN bubbles for
the same page through the full align->curate->format pipeline. This proves the
JP OCR adapter, the aligner, and the formatter all work end-to-end on real OCR
output -- the only deferred piece is the EN VLM (which needs the GPU box).

CUDA is disabled in-process (``CUDA_VISIBLE_DEVICES=""``) so it never contends
with the training run on the GPU.

    cd backend && .venv/bin/python scripts/data/corpus_bitext/smoke_cpu.py --pages 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Force CPU BEFORE onnxruntime/services import (so CUDA EP is unavailable).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BACKEND))

from curate import CurationStats  # type: ignore  # noqa: E402
from pipeline import PipelineConfig, align_and_curate  # type: ignore  # noqa: E402

CHAPTERS = {
    "ikenie4": (_BACKEND / "scripts/eval/data/ikenie4/gold_q3.jsonl",
                _BACKEND / ".bench/ikenie4_v11fix6_box_insp"),
    "ikenie5": (_BACKEND / "scripts/eval/data/ikenie5/gold_q3.jsonl",
                _BACKEND / ".bench/ikenie5_v11fix6_box_insp"),
}


def gold_en_bubbles(gold_path: Path, page: int):
    out = []
    for line in gold_path.open():
        r = json.loads(line)
        m = re.match(r"[^:]+:p(\d+):idx(\d+)", r.get("src", ""))
        if not m or int(m.group(1)) != page or not r.get("bbox"):
            continue
        out.append({"idx": int(m.group(2)), "text": (r.get("en") or "").strip(),
                    "bbox": dict(r["bbox"]), "conf": None})
    return out


async def main_async(args) -> int:
    from ocr_adapters import build_jp_engines, ocr_jp_page  # type: ignore

    detector, ocr = build_jp_engines(cpu_only=True)
    print(f"CTD providers: {detector.session.get_providers() if hasattr(detector,'session') else 'n/a'}")
    pcfg = PipelineConfig.default()

    gold_path, insp = CHAPTERS[args.chapter]
    if args.page_nums:
        page_dirs = [insp / f"{int(p):03d}" for p in args.page_nums.split(",")]
    else:
        page_dirs = sorted(insp.glob("[0-9][0-9][0-9]"))[: args.pages]
    for pd in page_dirs:
        page = int(pd.name)
        src_img = pd / "01_source.webp"
        if not src_img.exists():
            continue
        jp_bubbles = await ocr_jp_page(src_img, detector, ocr)
        en_bubbles = gold_en_bubbles(gold_path, page)
        print(f"\n=== {args.chapter} page {page}: JP-OCR {len(jp_bubbles)} bubbles, "
              f"gold EN {len(en_bubbles)} bubbles ===")
        for b in jp_bubbles[:6]:
            print(f"  JP[{b['idx']}] conf={b['conf']:.3f} {b['text']!r}")
        if not en_bubbles:
            continue
        stats = CurationStats()
        rows, kept = align_and_curate(jp_bubbles, en_bubbles, args.chapter, page, pcfg, stats, set())
        print(f"  -> aligned+curated: {len(rows)} rows kept (curation={stats.as_dict()})")
        for r in rows[:2]:
            print(f"     row src={r['src']}  en={r['en']!r}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chapter", default="ikenie4", choices=list(CHAPTERS))
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--page-nums", default="", help="explicit comma-separated page numbers, e.g. 5,6,7")
    raise SystemExit(asyncio.run(main_async(ap.parse_args())))

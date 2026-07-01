#!/usr/bin/env python3
"""EN-side VLM OCR for the staged good+partial galleries (RUNS ON THE BOX / GPU).

For each gallery with a JP OCR cache, transcribe the EN scanlation pages referenced
in page_alignment via the box Qwen-VL endpoint (reusing ocr_adapters.transcribe_en_page,
which wraps scripts/eval/transcribe_gt_vision.py). Output one cache per gallery:

  /home/danny/manga_corpus_staging/en_ocr_cache/<en_gid>_<jp_gid>.en_ocr.json
  { en_gid, jp_gid, en_dir, coord_norm, endpoint, model,
    pages: [ { en_page:int, en_file, n_bubbles, bubbles:[{idx,text,bbox,conf}] } ] }

Resumable (skips galleries already cached). coord_norm: use 1000 for Qwen3-VL
(established by the eval-gold rebuild); 0 for Qwen2.5-VL pixel coords. CONFIRM the
served model's convention on a couple of pages before a full run (wrong coords
destroy alignment).
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ocr_adapters import transcribe_en_page, VLM_ENDPOINT, VLM_MODEL  # type: ignore

STAGE = "/home/danny/manga_corpus_staging"


def page_int(s: str) -> int | None:
    d = re.sub(r"\D", "", os.path.splitext(str(s))[0])
    return int(d) if d else None


def dir_index(d: str) -> dict[int, str]:
    out: dict[int, str] = {}
    try:
        for fn in os.listdir(d):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                pi = page_int(fn)
                if pi is not None and pi not in out:
                    out[pi] = os.path.join(d, fn)
    except FileNotFoundError:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jp-cache", default=f"{STAGE}/jp_ocr_cache")
    ap.add_argument("--out", default=f"{STAGE}/en_ocr_cache")
    ap.add_argument("--endpoint", default=VLM_ENDPOINT)
    ap.add_argument("--model", default=VLM_MODEL)
    ap.add_argument("--coord-norm", type=int, default=1000, help="1000 for Qwen3-VL, 0 for pixel")
    ap.add_argument("--limit", type=int, default=0, help="max galleries (0=all)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    caches = sorted(glob.glob(f"{args.jp_cache}/*.jp_ocr.json"))
    done = pages = 0
    for jp_path in caches:
        jp = json.load(open(jp_path))
        key = f"{jp['en_gid']}_{jp['jp_gid']}"
        out_path = f"{args.out}/{key}.en_ocr.json"
        if os.path.exists(out_path):
            continue
        en_dir = jp.get("en_dir")
        idx = dir_index(en_dir) if en_dir else {}
        if not idx:
            print(f"  [skip] {key}: no EN images at {en_dir}", flush=True)
            continue
        want = sorted({page_int(pa.get("en_page")) for pa in jp.get("page_alignment", [])} - {None})
        pages_out = []
        t0 = time.time()
        for eni in want:
            f = idx.get(eni)
            if not f:
                continue
            try:
                bubbles = transcribe_en_page(f, endpoint=args.endpoint, model=args.model,
                                             coord_norm=args.coord_norm)
            except Exception as e:
                sys.stderr.write(f"  EN-OCR fail {key} p{eni}: {e}\n")
                continue
            pages_out.append({"en_page": eni, "en_file": os.path.basename(f),
                              "n_bubbles": len(bubbles), "bubbles": bubbles})
            pages += 1
        tmp = out_path + ".tmp"
        with open(tmp, "w") as g:
            json.dump({"en_gid": jp["en_gid"], "jp_gid": jp["jp_gid"], "en_dir": en_dir,
                       "coord_norm": args.coord_norm, "endpoint": args.endpoint,
                       "model": args.model, "pages": pages_out}, g, ensure_ascii=False)
        os.replace(tmp, out_path)
        done += 1
        print(f"  [{done}] {key}: {len(pages_out)} EN pages in {time.time()-t0:.0f}s", flush=True)
        if args.limit and done >= args.limit:
            break
    print(f"DONE: {done} galleries, {pages} EN pages -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

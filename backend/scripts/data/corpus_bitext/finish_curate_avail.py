#!/usr/bin/env python3
"""Join JP + EN OCR caches -> align -> curate -> v11 page-context training rows.

CPU-only. Consumes:
  /home/danny/manga_corpus_staging/jp_ocr_cache/<en_gid>_<jp_gid>.jp_ocr.json
  /home/danny/manga_corpus_staging/en_ocr_cache/<en_gid>_<jp_gid>.en_ocr.json
and emits curated rows (schema [prompt, en, src, register_tag, gold_flag], gold_flag=False)
to  /home/danny/manga_corpus_staging/curated/curated_rows.jsonl  + stats.json.

Page join is by INTEGER page number: page_alignment stores filename strings
('0001.jpg') while the caches key pages by the staged filename ('001.jpg'), so we
normalize both to int(digits-of-stem).
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # corpus_bitext modules use bare imports

from pipeline import PipelineConfig, align_and_curate  # type: ignore
from curate import CurationStats  # type: ignore

STAGE = "/home/danny/manga_corpus_staging"


def page_int(s: str) -> int | None:
    digits = re.sub(r"\D", "", os.path.splitext(str(s))[0])
    return int(digits) if digits else None


def load(path):
    with open(path) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jp-cache", default=f"{STAGE}/jp_ocr_cache")
    ap.add_argument("--en-cache", default=f"{STAGE}/en_ocr_cache")
    ap.add_argument("--out", default=f"{STAGE}/curated")
    ap.add_argument("--require-en", action="store_true",
                    help="skip galleries lacking an EN cache (default). If unset, also skip them.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = PipelineConfig.default()
    stats = CurationStats()
    seen: set = set()
    all_rows: list[dict] = []
    gal_done = gal_skipped = pages_joined = 0
    per_gal = []

    for jp_path in sorted(glob.glob(f"{args.jp_cache}/*.jp_ocr.json")):
        jp = load(jp_path)
        key = f"{jp['en_gid']}_{jp['jp_gid']}"
        en_path = f"{args.en_cache}/{key}.en_ocr.json"
        if not os.path.exists(en_path):
            gal_skipped += 1
            continue
        en = load(en_path)
        jp_by = {}
        for p in jp.get("pages", []):
            pi = page_int(p.get("jp_page"))
            if pi is not None:
                jp_by[pi] = p.get("bubbles", [])
        en_by = {}
        for p in en.get("pages", []):
            pi = p.get("en_page") if isinstance(p.get("en_page"), int) else page_int(p.get("en_page"))
            if pi is not None:
                en_by[pi] = p.get("bubbles", [])

        before = len(all_rows)
        for pa in jp.get("page_alignment", []):
            jpi, eni = page_int(pa.get("jp_page")), page_int(pa.get("en_page"))
            if jpi is None or eni is None:
                continue
            jb, eb = jp_by.get(jpi), en_by.get(eni)
            if not jb or not eb:
                continue
            pages_joined += 1
            rows, _ = align_and_curate(jb, eb, key, eni, cfg, stats, seen)
            all_rows.extend(rows)
        gal_done += 1
        per_gal.append({"gallery": key, "rows": len(all_rows) - before,
                        "bucket": jp.get("source_bucket")})

    out_rows = Path(args.out) / "curated_rows.jsonl"
    with open(out_rows, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    nsfw = sum(1 for r in all_rows if r["register_tag"] in ("vn_eroge", "manga_nsfw"))
    st = {
        "galleries_curated": gal_done, "galleries_skipped_no_en": gal_skipped,
        "pages_joined": pages_joined, "curated_rows": len(all_rows),
        "nsfw_rows": nsfw, "nsfw_frac": round(nsfw / len(all_rows), 4) if all_rows else 0.0,
        "curation_stats": getattr(stats, "__dict__", str(stats)),
        "per_gallery": per_gal[:50],
    }
    (Path(args.out) / "stats.json").write_text(json.dumps(st, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: st[k] for k in ("galleries_curated", "galleries_skipped_no_en",
                                          "pages_joined", "curated_rows", "nsfw_frac")}, indent=2))
    print(f"wrote {out_rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

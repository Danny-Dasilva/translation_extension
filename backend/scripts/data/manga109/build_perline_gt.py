#!/usr/bin/env python3
"""Build a per-LINE OCR ground-truth set from Manga109-s annotations.

The PARSeq model is a single-LINE recognizer. The existing bubbles.parquet is
whole-bubble (multi-line), which confounds OCR A/B accuracy (a single forward
pass cannot reproduce a multi-line bubble). This script extracts the subset of
Manga109-s ``<text>`` annotations whose bounding box is wider-than-tall — a
reliable single-line heuristic — so each crop maps to one line of ground truth.

Output schema is a drop-in for ``compare_parseq_exports.py --parquet``:
    book (str), page (int), xmin/ymin/xmax/ymax (int), jp_text (str)

Run on the machine that has the Manga109-s dataset (the training box), then point
the A/B harness at it:

    python build_perline_gt.py --out scripts/data/manga109/perline_gt.parquet
    .venv/bin/python scripts/eval_vision/compare_parseq_exports.py \
        --mode bubble-gt --parquet scripts/data/manga109/perline_gt.parquet \
        --n 500 --batch-size 24 --seed 42

The gate: adopt the dynamic-batch large_5p16 export only if its mean CER vs this
per-line GT is within +0.5% (absolute) of AR_single and exact-match within 3pp.
See thoughts/shared/research/translation-perf-display/2026-06-13_parseq-dynamic-batch-proposal.md
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import polars as pl

DEFAULT_ROOT = Path(
    "/mnt/nas/drive_2/manga-ml/datasets/manga109s/Manga109s_released_2023_12_07"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "perline_gt.parquet"


def build(root: Path, max_aspect: float, min_chars: int) -> pl.DataFrame:
    ann_dir = root / "annotations"
    if not ann_dir.is_dir():
        raise FileNotFoundError(
            f"Manga109 annotations dir not found: {ann_dir}\n"
            f"Pass --manga109-root pointing at the Manga109s release dir."
        )

    rows: list[dict] = []
    n_text = 0
    xml_paths = sorted(ann_dir.glob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No .xml annotation files in {ann_dir}")

    for xml_path in xml_paths:
        book = xml_path.stem  # matches the image folder name (<root>/images/<book>/<page>.jpg)
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as e:  # noqa: PERF203 - skip malformed, keep going
            print(f"  WARN: skipping malformed {xml_path.name}: {e}", file=sys.stderr)
            continue
        for page in tree.findall(".//page"):
            pg = int(page.get("index"))
            for text in page.findall(".//text"):
                n_text += 1
                x0, y0 = int(text.get("xmin")), int(text.get("ymin"))
                x1, y1 = int(text.get("xmax")), int(text.get("ymax"))
                gt = (text.text or "").strip()
                if len(gt) < min_chars:
                    continue
                w, h = x1 - x0, y1 - y0
                if w <= 0 or h <= 0:
                    continue
                # wider-than-tall (h/w small) => a single horizontal line of text.
                if h / w < max_aspect:
                    rows.append(
                        {
                            "book": book,
                            "page": pg,
                            "xmin": x0,
                            "ymin": y0,
                            "xmax": x1,
                            "ymax": y1,
                            "jp_text": gt,
                        }
                    )

    df = pl.DataFrame(
        rows,
        schema={
            "book": pl.Utf8,
            "page": pl.Int64,
            "xmin": pl.Int64,
            "ymin": pl.Int64,
            "xmax": pl.Int64,
            "ymax": pl.Int64,
            "jp_text": pl.Utf8,
        },
    )
    print(
        f"Scanned {len(xml_paths)} books, {n_text} <text> annotations -> "
        f"{len(df)} single-line GT rows (h/w < {max_aspect}, >= {min_chars} chars)"
    )
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manga109-root", type=Path, default=DEFAULT_ROOT,
                    help="Manga109s release dir (contains annotations/ and images/)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output parquet path")
    ap.add_argument("--max-aspect", type=float, default=0.7,
                    help="keep boxes with height/width below this (single-line heuristic)")
    ap.add_argument("--min-chars", type=int, default=2,
                    help="drop ground-truth strings shorter than this")
    args = ap.parse_args()

    df = build(args.manga109_root, args.max_aspect, args.min_chars)
    if df.is_empty():
        print("ERROR: no single-line rows extracted — check --manga109-root / --max-aspect",
              file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"Wrote {len(df)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

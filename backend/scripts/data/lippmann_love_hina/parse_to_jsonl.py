"""Parse Lippmann Love Hina JA-PL annotations into a flat JSONL.

IMPORTANT — Language pair is JA→PL (Japanese to Polish), NOT JA→EN. The
upstream README at multimodal-manga-translation/README.md states:

    "We provide professional JA-PL translations of the slice-of-life manga
     *Love Hina* to create a data set for research purposes."

The 3705 contributed annotations across vol1 + vol14 are Polish, not English.
To use Love Hina images for JA→EN benchmarking you would need separate
English references (the Lippmann paper itself only contributes Polish).

This script still exports a JSONL for completeness — JA + PL — so a future
JA→PL eval (or a JA→EN re-translation pass that uses Polish as a
sanity-check pivot) is one command away.

License: MIT (Copyright (c) 2024 Philip Lippmann), see
multimodal-manga-translation/LICENSE.

Usage
-----
    /home/danny/.venvs/comet/bin/python parse_to_jsonl.py \
        --in-dir multimodal-manga-translation/LoveHina \
        --out heldout_ja_pl.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_lines(json_path: Path, vol_tag: str):
    """Yield (jp, pl, src) for each line entry in a vol JSON."""
    pages = json.loads(json_path.read_text())
    if not isinstance(pages, list):
        raise SystemExit(f"unexpected format in {json_path}: not a list")
    for p_idx, page in enumerate(pages):
        for ln_idx, line in enumerate(page.get("lines") or []):
            jp = (line.get("text_jp") or "").strip()
            pl = (line.get("text_pl") or "").strip()
            if not jp or not pl:
                continue
            yield {
                "jp": jp,
                "pl": pl,
                "src": f"lippmann_love_hina:{vol_tag}:p{p_idx}:t{ln_idx}",
            }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir",
        default=str(Path(__file__).resolve().parent
                    / "multimodal-manga-translation" / "LoveHina"),
    )
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "heldout_ja_pl.jsonl"),
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    files = [
        ("vol01", in_dir / "LoveHina_vol01_fixed_order.json"),
        ("vol14", in_dir / "LoveHina_vol14_fixed_order.json"),
    ]
    for tag, p in files:
        if not p.exists():
            raise SystemExit(f"missing {p}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for tag, p in files:
            for row in _iter_lines(p, tag):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} JA-PL pairs to {out_path}")
    print("NOTE: language pair is JA->PL, not JA->EN. Lippmann Love Hina "
          "does NOT contribute English references; this dataset cannot be "
          "used as a JA-EN holdout out-of-the-box.")
    return 0


if __name__ == "__main__":
    main()

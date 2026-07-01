#!/usr/bin/env python3
"""Build a SMALL proof-of-concept MULTIMODAL dataset for a future vision-LoRA
fine-tune of the manga translator (Phase 2 / "v12 vision").

WHAT THIS IS
------------
The production translator (Gemma-4 E4B, v11 page-context) is served TEXT-ONLY:
    OCR'd JP + numbered page context  ->  EN line.
The MT roadmap verdict is that adding the page IMAGE as context is the #1 lever
to fix the pronoun / speaker-collapse ceiling. The Ikenie gold pages are the one
place we have BOTH the page IMAGE and high-quality human JP->EN pairs, so they
are the natural seed corpus for a vision LoRA.

This script does NOT train anything. It groups the per-bubble gold rows by page
and emits ONE row per page in the schema the downstream trainer agent consumes.

OUTPUT SCHEMA (one JSON object per line, one line per page)
----------------------------------------------------------
    {
      "image_path":   "<absolute local path to the page image (01_source.webp)>",
      "jp_ocr":       "<full-page OCR'd JP lines, reading order, newline-joined>",
      "page_context": "<v11 serve-format page block (instruction + numbered page)>",
      "en_target":    "<human EN for the page's gold bubbles, reading order>",
      "meta": {"chapter": "ikenie4|ikenie5", "page": <int>, "n_bubbles": <int>}
    }

DESIGN DECISIONS (see README.md for the long form)
--------------------------------------------------
* jp_ocr / page_context use the FULL page OCR from the pipeline's per-page
  ``bubbles.json`` (reading order = ``idx`` order). These are the exact OCR lines
  the v11 text model was served, so the JP side is byte-faithful to serve format.
* page_context replicates ``build_v11_context_prompt`` byte-for-byte EXCEPT the
  trailing per-line ``Translate line k: ...`` suffix is omitted, because a
  page-level row has no single marked target. The instruction string and the
  ``Page:\n1. ...\nN. ...`` block are verbatim (they are a train/serve contract).
* en_target is the HUMAN EN from gold_q3.jsonl, which is a curated "worst-issues"
  SUBSET of each page's bubbles (the bubbles a judge flagged + corrected). We only
  have trustworthy human EN for that subset, so en_target covers those bubbles in
  reading order. jp_ocr/page_context cover the WHOLE page (the model still sees
  the full context + image); en_target is the high-quality supervised signal.
* idx in gold ``src`` ("ikenieN:pPP:idxII") == the idx in that page's bubbles.json
  (verified: page/idx/bbox/jp all match), and the box-inspection page dir name is
  the zero-padded page number (p5 -> "005", p132 -> "132").

USAGE
-----
    python build_v12vision_poc.py            # both chapters, default paths
    python build_v12vision_poc.py --chapters ikenie4
    python build_v12vision_poc.py --out /some/dir   # write elsewhere (local only)

All writes are LOCAL. Never write under /mnt/nas (the CIFS share silently reaps
files ~9 min after write).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# v11 serve-format contract (copied VERBATIM from
# backend/app/services/vllm_openai_translation_service.py so this builder has no
# heavy app/model import side effects). Keep these byte-identical to that file:
# a whitespace/marker drift silently degrades translation quality.
# --------------------------------------------------------------------------- #
V11_PAGE_INSTR = (
    "Translate the marked line of this manga page from Japanese to English. "
    "Use the page context for speakers, pronouns, and continuity. "
    "Output only the translation of the marked line."
)

# this file lives at <repo>/backend/scripts/data/v12vision/ -> 4 levels up.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

# chapter -> (gold_q3.jsonl, box-inspection root holding NNN/{01_source.webp,bubbles.json})
DEFAULT_CHAPTERS: Dict[str, Tuple[str, str]] = {
    "ikenie4": (
        os.path.join(REPO_ROOT, "backend/scripts/eval/data/ikenie4/gold_q3.jsonl"),
        os.path.join(REPO_ROOT, "backend/.bench/ikenie4_v11fix6_box_insp"),
    ),
    "ikenie5": (
        os.path.join(REPO_ROOT, "backend/scripts/eval/data/ikenie5/gold_q3.jsonl"),
        os.path.join(REPO_ROOT, "backend/.bench/ikenie5_v11fix6_box_insp"),
    ),
}

_SRC_RE = re.compile(r"^(?P<chapter>\w+):p(?P<page>\d+):idx(?P<idx>\d+)\s*$")


def parse_src(src: str) -> Optional[Tuple[str, int, int]]:
    """('ikenie4:p05:idx0') -> ('ikenie4', 5, 0); None if it doesn't parse."""
    m = _SRC_RE.match(src or "")
    if not m:
        return None
    return m.group("chapter"), int(m.group("page")), int(m.group("idx"))


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def page_dir(bench_root: str, page: int) -> str:
    return os.path.join(bench_root, f"{page:03d}")


def load_page_ocr_lines(bench_root: str, page: int) -> Optional[List[str]]:
    """Full-page OCR lines in reading order from the pipeline's bubbles.json.

    Returns the ``ocr_jp`` of every block sorted by ``idx`` (= reading order), or
    None when bubbles.json is absent (caller falls back to the gold JP lines).
    """
    bj = os.path.join(page_dir(bench_root, page), "bubbles.json")
    if not os.path.exists(bj):
        return None
    blocks = json.load(open(bj, encoding="utf-8"))
    blocks = sorted(blocks, key=lambda b: b.get("idx", 0))
    return [(b.get("ocr_jp") or "").strip() for b in blocks]


def find_image(bench_root: str, page: int) -> Optional[str]:
    """Absolute path to the page's source image, or None. Prefers 01_source.webp."""
    pdir = page_dir(bench_root, page)
    for name in ("01_source.webp", "01_source.png", "01_source.jpg"):
        p = os.path.join(pdir, name)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def build_page_context(lines: List[str]) -> str:
    """v11 serve-format page block (instruction + numbered page), page-level.

    Byte-identical to build_v11_context_prompt's prefix; the per-line
    ``Translate line k: ...`` suffix is intentionally dropped (page-level row).
    """
    numbered = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))
    return f"{V11_PAGE_INSTR}\n\nPage:\n{numbered}"


def assemble_chapter(
    chapter: str, gold_path: str, bench_root: str
) -> Tuple[List[dict], dict]:
    """Group gold rows by page -> one dataset row per page. Returns (rows, stats)."""
    gold = load_jsonl(gold_path)

    # page -> ordered list of (idx, gold_row); dedupe repeated idx (same bubble
    # flagged for >1 issue) keeping the first occurrence.
    by_page: Dict[int, "dict[int, dict]"] = defaultdict(dict)
    skipped_src = 0
    for r in gold:
        parsed = parse_src(r.get("src", ""))
        if parsed is None or parsed[0] != chapter:
            skipped_src += 1
            continue
        _, page, idx = parsed
        if idx not in by_page[page]:
            by_page[page][idx] = r

    rows: List[dict] = []
    images_found = 0
    images_missing = 0
    pages_ocr_full = 0       # jp_ocr came from full-page bubbles.json
    pages_ocr_approx = 0     # fell back to gold JP lines only
    missing_image_pages: List[int] = []

    for page in sorted(by_page):
        bubbles = by_page[page]
        target_idxs = sorted(bubbles)
        target_jp = [(bubbles[i].get("jp") or "").strip() for i in target_idxs]
        target_en = [(bubbles[i].get("en") or "").strip() for i in target_idxs]

        # Full-page OCR context (serve-faithful) with gold-subset fallback.
        page_lines = load_page_ocr_lines(bench_root, page)
        if page_lines is None:
            page_lines = target_jp  # approximation: gold subset only
            pages_ocr_approx += 1
        else:
            pages_ocr_full += 1
        # Drop blank OCR lines so the numbered context has no empty entries.
        page_lines = [ln for ln in page_lines if ln]

        img = find_image(bench_root, page)
        if img:
            images_found += 1
        else:
            images_missing += 1
            missing_image_pages.append(page)

        rows.append(
            {
                "image_path": img or "",
                "jp_ocr": "\n".join(page_lines),
                "page_context": build_page_context(page_lines),
                "en_target": "\n".join(target_en),
                "meta": {
                    "chapter": chapter,
                    "page": page,
                    "n_bubbles": len(target_idxs),
                },
            }
        )

    stats = {
        "chapter": chapter,
        "rows": len(rows),
        "images_found": images_found,
        "images_missing": images_missing,
        "missing_image_pages": missing_image_pages,
        "pages_ocr_full_page_context": pages_ocr_full,
        "pages_ocr_gold_subset_fallback": pages_ocr_approx,
        "gold_rows_skipped_unparseable_src": skipped_src,
        "total_target_bubbles": sum(r["meta"]["n_bubbles"] for r in rows),
    }
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--chapters",
        nargs="*",
        default=list(DEFAULT_CHAPTERS),
        choices=list(DEFAULT_CHAPTERS),
        help="Which chapters to build (default: all).",
    )
    ap.add_argument(
        "--out",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Output directory (LOCAL ONLY — never /mnt/nas).",
    )
    ap.add_argument(
        "--gold",
        nargs="*",
        default=None,
        help="Override gold jsonl paths, aligned 1:1 with --chapters.",
    )
    ap.add_argument(
        "--image-dir",
        nargs="*",
        default=None,
        help="Override box-inspection image roots, aligned 1:1 with --chapters.",
    )
    args = ap.parse_args()

    if "/mnt/nas" in os.path.abspath(args.out):
        raise SystemExit("Refusing to write under /mnt/nas (CIFS reaps files).")
    os.makedirs(args.out, exist_ok=True)

    overrides_gold = dict(zip(args.chapters, args.gold)) if args.gold else {}
    overrides_img = dict(zip(args.chapters, args.image_dir)) if args.image_dir else {}

    all_rows: List[dict] = []
    per_chapter_stats: List[dict] = []
    for chapter in args.chapters:
        gold_path = overrides_gold.get(chapter, DEFAULT_CHAPTERS[chapter][0])
        bench_root = overrides_img.get(chapter, DEFAULT_CHAPTERS[chapter][1])
        if not os.path.exists(gold_path):
            raise SystemExit(f"gold jsonl not found: {gold_path}")
        rows, stats = assemble_chapter(chapter, gold_path, bench_root)
        all_rows.extend(rows)
        per_chapter_stats.append(stats)
        print(
            f"[{chapter}] pages={stats['rows']} "
            f"img_found={stats['images_found']} img_missing={stats['images_missing']} "
            f"target_bubbles={stats['total_target_bubbles']}"
        )

    data_path = os.path.join(args.out, "data_v12vision_poc.jsonl")
    with open(data_path, "w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "total_rows": len(all_rows),
        "images_found_total": sum(s["images_found"] for s in per_chapter_stats),
        "images_missing_total": sum(s["images_missing"] for s in per_chapter_stats),
        "image_source": "backend/.bench/<chapter>_v11fix6_box_insp/<NNN>/01_source.webp (local)",
        "jp_ocr_source": "full-page bubbles.json (serve-faithful); gold-subset fallback if absent",
        "page_context_format": "v11 build_v11_context_prompt prefix (instruction + numbered Page block), page-level (no 'Translate line k' suffix)",
        "en_target_source": "gold_q3.jsonl human EN (curated worst-issues bubble subset), reading order",
        "per_chapter": per_chapter_stats,
    }
    stats_path = os.path.join(args.out, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    print(f"\nwrote {len(all_rows)} rows -> {data_path}")
    print(f"wrote stats -> {stats_path}")


if __name__ == "__main__":
    main()

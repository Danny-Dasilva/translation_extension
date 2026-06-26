#!/usr/bin/env python3
"""Extract (jp, our_en, human_en) seed rows for the fix6 clean-OCR corrective set.

SKELETON — runnable signature + structure + TODOs. Not fully wired (the human-GT
side is in page IMAGES, not structured data — see big NOTE below).

Plan: thoughts/shared/plans/fix6-clean-ocr-mistranslation-finetune.md

What this does
--------------
The Ikenie 4 audit found the most frequent model fault is "mistranslation on
CLEAN OCR": the Japanese was read correctly but the English is wrong (158
bubbles, avg severity 2.25). This script walks our pipeline outputs
(`bubbles.json`), keeps only the CLEAN-OCR bubbles (high `ocr_conf`, not garbled,
not filtered/gated), pairs each kept bubble's wrong English against the HUMAN
ground-truth English for that page, and emits one JSONL seed row per pair:

    {"page": "050", "idx": 0, "jp": <ocr_jp>, "our_en": <translation_en>,
     "human_en": <typeset GT or "" placeholder>, "ocr_conf": <float>,
     "submode": "", "register_tag": "", "keep": null}

Those triples are then human-reviewed and promoted (separately) into the v11
corrective seed schema [jp, en=human_en, src, register_tag, gold_flag] consumed
by build_v11_corrective_seed.py / build_v11_dataset.py. We TRAIN on (jp ->
human_en) SFT only; `our_en` is kept as a curation trigger + held-out contrastive
probe, NOT as a DPO rejected signal (see plan §3 — DPO toward NSFW caused the v12
euphemism regression).

bubbles.json schema (per-page list; one dict per bubble)
--------------------------------------------------------
    idx              int     bubble index on the page
    bbox             [x,y,x,y]
    ocr_jp           str     the (correct) Japanese — our INPUT jp
    translation_en   str|None our (wrong) English — our_en  (None if not translated)
    ocr_conf         float   OCR confidence (clean-OCR gate)
    confidence       float   detector confidence
    is_orphan        bool
    ocr_gate_dropped bool    dropped by the OCR confidence gate
    filtered         bool    filtered out of the final render

  >>> THE HUMAN GT IS NOT IN THIS FILE. <<<
  The human scanlation English is TYPESET INTO THE PAGE IMAGE at
      /mnt/nas/drive_2/onlyfans/external_content/nhentai/616137_Ikenie no Haha 4/<NNN>.webp
  Recovering `human_en` therefore needs OCR/vision over the GT image, or a manual
  transcription pass. This script emits the `human_en` field EMPTY (or a draft
  from --gt-mode vision, if implemented) and flags it for the human pass. A
  vision-OCR draft must NEVER be auto-promoted to a gold target.

Usage
-----
    python backend/scripts/data/v11/extract_mistranslation_pairs.py \
        --bench-dir /home/danny/Documents/personal/extension/backend/.bench/ikenie4_inspect \
        --gt-dir "/mnt/nas/drive_2/onlyfans/external_content/nhentai/616137_Ikenie no Haha 4" \
        --out backend/scripts/data/corrective/fix6_cleanocr_seed.jsonl \
        --ocr-conf-min 0.85 \
        --gt-mode none           # none | vision | manual-merge

Output: JSONL of seed triples (one per kept bubble) for human review.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional

# --------------------------------------------------------------------------- #
# Defaults / constants
# --------------------------------------------------------------------------- #
DEFAULT_BENCH_DIR = Path(
    "/home/danny/Documents/personal/extension/backend/.bench/ikenie4_inspect"
)
DEFAULT_GT_DIR = Path(
    "/mnt/nas/drive_2/onlyfans/external_content/nhentai/616137_Ikenie no Haha 4"
)
DEFAULT_OCR_CONF_MIN = 0.85

# A bubble whose ocr_jp carries >= this many latin letters is a romaji/garbage
# leak, not clean Japanese (e.g. "yo」oubnnnhhoo", "kenie nohhaha"). Tune in the
# human pass.
_LATIN_LEAK_MIN = 3
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass
class SeedRow:
    """One (jp, our_en, human_en) triple awaiting human review/promotion."""

    page: str
    idx: int
    jp: str
    our_en: str
    human_en: str            # "" until the human/vision GT pass fills it
    ocr_conf: float
    # to be filled in the human pass:
    submode: str = ""        # "sense" | "polarity" | "vocab"
    register_tag: str = ""   # "manga_dialog" | "vn_eroge" (must exist in data_v10)
    keep: Optional[bool] = None  # null = undecided; human sets True/False


# --------------------------------------------------------------------------- #
# Clean-OCR filter
# --------------------------------------------------------------------------- #
def is_garbled(jp: Optional[str]) -> bool:
    """Heuristic: empty, or carries a romaji/latin leak => garbled OCR.

    TODO: extend — detect runaway repeated kana, all-symbol bubbles, length-1
    noise. The serve-side `normalize_short_utterance` in
    vllm_openai_translation_service.py is a reference for kana cleanup, but here
    we only REJECT garbage, we do not normalize.
    """
    if not jp or not jp.strip():
        return True
    if len(_LATIN_RE.findall(jp)) >= _LATIN_LEAK_MIN:
        return True
    return False


def is_clean_ocr_bubble(b: dict, ocr_conf_min: float) -> bool:
    """Keep only bubbles where OCR is CORRECT and the line was actually output.

    By construction this corrective set targets clean-OCR mistranslations, so we
    exclude anything filtered/gated/untranslated/garbled.
    """
    if b.get("filtered"):
        return False
    if b.get("ocr_gate_dropped"):
        return False
    if b.get("translation_en") is None:
        return False
    if float(b.get("ocr_conf", 0.0)) < ocr_conf_min:
        return False
    if is_garbled(b.get("ocr_jp")):
        return False
    return True


# --------------------------------------------------------------------------- #
# Human-GT (typeset English) recovery — the hard part
# --------------------------------------------------------------------------- #
def load_human_gt_for_page(gt_dir: Path, page: str, mode: str) -> dict[int, str]:
    """Return {bubble_idx -> human_en} for a page. SKELETON.

    The GT is TYPESET INTO THE IMAGE at gt_dir/<page>.webp — there is no
    structured English to read. Strategies:

      mode == "none":   return {} (emit empty human_en; fill in the manual pass).
      mode == "vision": TODO run a vision/OCR model over the GT webp, detect the
                        English text regions, and (hard) align each English
                        region to our bubble idx by bbox overlap. Output is a
                        DRAFT for human verification — never auto-gold.
      mode == "manual-merge": TODO read a side human-transcribed file
                        (e.g. gt_dir/<page>.gt.json: {idx: "english"}) and merge.

    Alignment note: our bbox order/idx will NOT match the GT image's reading
    order automatically; matching needs bbox IoU between our detected bubbles and
    the GT English regions, or a manual idx mapping.
    """
    if mode == "none":
        return {}
    if mode == "manual-merge":
        # TODO: load gt_dir/<page>.gt.json if present
        gt_file = gt_dir / f"{page}.gt.json"
        if gt_file.exists():
            data = json.loads(gt_file.read_text())
            return {int(k): str(v) for k, v in data.items()}
        return {}
    if mode == "vision":
        # TODO: vision-OCR the webp, detect EN regions, align to bubble idx by
        # bbox IoU. Out of scope for the skeleton.
        raise NotImplementedError("vision GT extraction not implemented yet")
    raise ValueError(f"unknown gt-mode: {mode!r}")


# --------------------------------------------------------------------------- #
# Walk
# --------------------------------------------------------------------------- #
def iter_bubbles_files(bench_dir: Path) -> Iterator[tuple[str, Path]]:
    """Yield (page_id, bubbles.json path) sorted by page."""
    for d in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
        bj = d / "bubbles.json"
        if bj.exists():
            yield d.name, bj


def extract(
    bench_dir: Path,
    gt_dir: Path,
    ocr_conf_min: float,
    gt_mode: str,
) -> list[SeedRow]:
    rows: list[SeedRow] = []
    for page, bj in iter_bubbles_files(bench_dir):
        bubbles = json.loads(bj.read_text())
        human = load_human_gt_for_page(gt_dir, page, gt_mode)
        for b in bubbles:
            if not is_clean_ocr_bubble(b, ocr_conf_min):
                continue
            idx = int(b.get("idx", -1))
            rows.append(
                SeedRow(
                    page=page,
                    idx=idx,
                    jp=b["ocr_jp"],
                    our_en=b.get("translation_en") or "",
                    human_en=human.get(idx, ""),
                    ocr_conf=float(b.get("ocr_conf", 0.0)),
                )
            )
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH_DIR)
    ap.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("backend/scripts/data/corrective/fix6_cleanocr_seed.jsonl"),
    )
    ap.add_argument("--ocr-conf-min", type=float, default=DEFAULT_OCR_CONF_MIN)
    ap.add_argument(
        "--gt-mode",
        choices=("none", "vision", "manual-merge"),
        default="none",
        help="how to recover the typeset human English (default: none = empty placeholder)",
    )
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.bench_dir.exists():
        print(f"bench dir not found: {args.bench_dir}")
        return 1

    rows = extract(args.bench_dir, args.gt_dir, args.ocr_conf_min, args.gt_mode)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    have_gt = sum(1 for r in rows if r.human_en)
    print(f"bench-dir       : {args.bench_dir}")
    print(f"ocr-conf-min    : {args.ocr_conf_min}")
    print(f"gt-mode         : {args.gt_mode}")
    print(f"clean-OCR rows  : {len(rows)}")
    print(f"  with human_en : {have_gt}  (rest need the manual/vision GT pass)")
    print(f"wrote           : {args.out}")
    # TODO: print a per-submode breakdown once the human pass fills `submode`.
    # TODO downstream (separate script): promote keep==True rows -> v11 corrective
    #       seed schema [jp, en=human_en, src=corrective_v11:cleanocr_<submode>:<id>,
    #       register_tag, gold_flag=True]; euphemism-audit submode=="vocab" first.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

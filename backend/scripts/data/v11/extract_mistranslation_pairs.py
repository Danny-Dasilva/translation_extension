#!/usr/bin/env python3
"""Extract (jp, our_en, human_en) seed rows for the fix6 clean-OCR corrective set.

FUNCTIONAL for the (jp, our_en) extraction half. The human-GT side
(`human_en`) lives in the page IMAGES, not in any structured field — it is left
EMPTY here and recovered in a separate manual/vision pass (see big NOTE below).

Plan: thoughts/shared/plans/fix6-clean-ocr-mistranslation-finetune.md

What this does
--------------
The Ikenie 4 audit found the most frequent model fault is "mistranslation on
CLEAN OCR": the Japanese was read correctly but the English is wrong (158
bubbles, avg severity 2.25). This script walks our pipeline outputs
(`bubbles.json`), keeps only the CLEAN-OCR bubbles (high `ocr_conf`, not garbled,
not filtered/gated), and emits one JSONL seed row per kept bubble pairing the
correct Japanese against our (possibly wrong) English:

    {"page": "050", "idx": 0, "jp": <ocr_jp>, "our_en": <translation_en>,
     "human_en": "", "src": "corrective_v11:cleanocr:050_0", "ocr_conf": <float>,
     "submode_guess": "vocab", "submode": "", "register_tag": "", "keep": null}

Those triples are then human-reviewed and promoted (separately) into the v11
corrective seed schema [jp, en=human_en, src, register_tag, gold_flag] consumed
by build_v11_corrective_seed.py / build_v11_dataset.py. We TRAIN on (jp ->
human_en) SFT only; `our_en` is kept as a curation trigger + held-out contrastive
probe, NOT as a DPO rejected signal (see plan §3 — DPO toward NSFW caused the v12
euphemism regression).

bubbles.json schema (per-page list; one dict per bubble)
--------------------------------------------------------
    idx              int     bubble index on the page
    bbox             {minX,minY,maxX,maxY}
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
  from --gt-mode manual-merge, if a side file is provided) and flags it for the
  human pass. GT human-EN recovery is the remaining manual/vision step before
  these triples become training rows — a vision-OCR draft must NEVER be
  auto-promoted to a gold target.

Usage
-----
    python backend/scripts/data/v11/extract_mistranslation_pairs.py \
        --bench-dir /home/danny/Documents/personal/extension/backend/.bench/ikenie4_bf16_insp \
        --gt-dir "/mnt/nas/drive_2/onlyfans/external_content/nhentai/616137_Ikenie no Haha 4" \
        --out backend/scripts/data/v11/seed_cleanocr_pairs.jsonl \
        --ocr-conf-min 0.85 \
        --gt-mode none           # none | vision | manual-merge

Output: JSONL of seed triples (one per kept clean-OCR bubble) for human review.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

# Import the production linguistic-plausibility garble detector so the clean-OCR
# filter matches the serve-side gate (catches duplicated-bigram / doubled-kana /
# latin-intrusion garble that carries falsely-high ocr_conf).
try:
    from app.utils.ocr_confidence_gate import is_implausible_japanese
except Exception:  # pragma: no cover - import guard for standalone runs
    def is_implausible_japanese(text: str) -> bool:  # type: ignore[misc]
        return False


# --------------------------------------------------------------------------- #
# Defaults / constants
# --------------------------------------------------------------------------- #
DEFAULT_BENCH_DIR = Path(
    "/home/danny/Documents/personal/extension/backend/.bench/ikenie4_bf16_insp"
)
DEFAULT_GT_DIR = Path(
    "/mnt/nas/drive_2/onlyfans/external_content/nhentai/616137_Ikenie no Haha 4"
)
DEFAULT_OCR_CONF_MIN = 0.85

# A bubble whose ocr_jp carries >= this many latin letters is a romaji/garbage
# leak, not clean Japanese (e.g. "yo」oubnnnhhoo", "kenie nohhaha"). This
# complements is_implausible_japanese (which targets latin *intrusion* into
# otherwise-Japanese text); this one catches mostly-latin OCR noise.
_LATIN_LEAK_MIN = 3
_LATIN_RE = re.compile(r"[A-Za-z]")
_JP_RE = re.compile(r"[぀-ゟ゠-ヿ一-鿿]")


# --------------------------------------------------------------------------- #
# Coarse sub-mode classification (plan §1: sense / negation / idiom / vocab)
# --------------------------------------------------------------------------- #
# Adult-domain / commonly-mistranslated nouns from the plan §5 gloss. Presence
# of any of these in the JP biases the row toward the "vocab" sub-mode. This is
# ONLY a guess to pre-sort the human review queue — the human pass overrides
# `submode`. We do NOT use these as a runtime dictionary.
_VOCAB_TERMS = (
    "尻", "お尻", "マンコ", "おまんこ", "ちんこ", "チンポ", "ちんぽ", "オチンチン",
    "おちんちん", "バイブ", "ローター", "洗濯バサミ", "風俗", "顔射", "中出し",
    "アナル", "尻穴", "搾り", "締ま", "ザーメン", "精液", "射精", "イク", "イっ",
    "クリ", "乳首", "おっぱい", "ペニス", "チンチン", "フェラ", "パイズリ",
)
# Idiom / set-phrase markers that tend to be over-literalized.
_IDIOM_TERMS = (
    "頂きましょう", "いただきましょう", "頂きます", "いただきます",
    "お疲れ", "仕方", "しょうが", "つもり", "わけ", "せい", "おかげ",
    "気が", "手が", "目が", "口が", "腹が",
)
# Negation markers (sub-mode b = polarity flips). Japanese negation is varied;
# this is a coarse net for the human-review queue.
_NEG_RE = re.compile(
    r"(ない|ねえ|ねぇ|ぬ|ん(?:な|じゃ)|なよ|まい|ず|ざる|なきゃ|なくちゃ|"
    r"でない|じゃない|くない|せず|わけがない|はずがない)"
)


def guess_submode(jp: str) -> str:
    """Coarse sub-mode guess to pre-sort the human-review queue.

    Returns one of: "vocab" | "negation" | "idiom" | "other".
    Precedence (vocab > negation > idiom > other) is arbitrary but stable; the
    human pass owns the authoritative `submode`. We never train on this guess.
    """
    if any(t in jp for t in _VOCAB_TERMS):
        return "vocab"
    if _NEG_RE.search(jp):
        return "negation"
    if any(t in jp for t in _IDIOM_TERMS):
        return "idiom"
    return "other"


@dataclass
class SeedRow:
    """One (jp, our_en, human_en) triple awaiting human review/promotion."""

    page: str
    idx: int
    jp: str
    our_en: str
    human_en: str            # "" until the human/vision GT pass fills it
    src: str                 # provenance id (becomes corrective_v11:cleanocr:<id>)
    ocr_conf: float
    submode_guess: str = ""  # coarse auto-guess: vocab|negation|idiom|other
    # to be filled / overridden in the human pass:
    submode: str = ""        # authoritative: "vocab" | "negation" | "idiom" | "other"
    register_tag: str = ""   # "manga_dialog" | "vn_eroge" (must exist in data_v10)
    keep: Optional[bool] = None  # null = undecided; human sets True/False


# --------------------------------------------------------------------------- #
# Clean-OCR filter
# --------------------------------------------------------------------------- #
def is_garbled(jp: Optional[str]) -> bool:
    """Heuristic: empty, mostly-latin noise, or production-flagged implausible.

    Layers (any one => garbled):
      * empty / whitespace-only,
      * >= _LATIN_LEAK_MIN latin letters AND no Japanese chars (romaji/url noise),
      * production linguistic-plausibility gate (duplicated-bigram, doubled
        kana/kanji, latin intrusion into Japanese) — shared with the serve-side
        OCR gate so we reject the same garble the pipeline would.
    """
    if not jp or not jp.strip():
        return True
    # Mostly-latin OCR noise (e.g. "yo」oubnnnhhoo"): latin letters present and
    # no Japanese script at all.
    if len(_LATIN_RE.findall(jp)) >= _LATIN_LEAK_MIN and not _JP_RE.search(jp):
        return True
    # Production garble detector (catches latin-intrusion-in-JP + duplication
    # garble that carries falsely-high ocr_conf).
    if is_implausible_japanese(jp):
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
    if not str(b.get("translation_en", "")).strip():
        return False
    if float(b.get("ocr_conf", 0.0)) < ocr_conf_min:
        return False
    if is_garbled(b.get("ocr_jp")):
        return False
    return True


# --------------------------------------------------------------------------- #
# Human-GT (typeset English) recovery — the REMAINING MANUAL/VISION STEP
# --------------------------------------------------------------------------- #
def load_human_gt_for_page(gt_dir: Path, page: str, mode: str) -> dict[int, str]:
    """Return {bubble_idx -> human_en} for a page.

    >>> human_en is NOT in the structured pipeline output. <<<
    The GT is TYPESET INTO THE IMAGE at gt_dir/<page>.webp — there is no
    structured English to read. This is the remaining manual/vision step before
    these triples become training rows. Strategies:

      mode == "none":   return {} (emit empty human_en; fill in the manual pass).
      mode == "manual-merge": read a side human-transcribed file
                        (gt_dir/<page>.gt.json: {idx: "english"}) and merge.
      mode == "vision": NOT IMPLEMENTED. A vision/OCR model over the GT webp +
                        bbox-IoU alignment to our bubble idx. Its output would be
                        a DRAFT for human verification — never auto-gold.

    Alignment note: our bbox order/idx will NOT match the GT image's reading
    order automatically; matching needs bbox IoU between our detected bubbles and
    the GT English regions, or a manual idx mapping.
    """
    if mode == "none":
        return {}
    if mode == "manual-merge":
        gt_file = gt_dir / f"{page}.gt.json"
        if gt_file.exists():
            data = json.loads(gt_file.read_text())
            return {int(k): str(v) for k, v in data.items()}
        return {}
    if mode == "vision":
        raise NotImplementedError(
            "vision GT extraction not implemented — human_en recovery is the "
            "remaining manual/vision step (see plan §7 / §9 step 2)"
        )
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
            jp = b["ocr_jp"]
            rows.append(
                SeedRow(
                    page=page,
                    idx=idx,
                    jp=jp,
                    our_en=str(b.get("translation_en") or ""),
                    human_en=human.get(idx, ""),
                    src=f"corrective_v11:cleanocr:{page}_{idx}",
                    ocr_conf=float(b.get("ocr_conf", 0.0)),
                    submode_guess=guess_submode(jp),
                )
            )
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract clean-OCR (jp, our_en) seed pairs for fix6.",
    )
    ap.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH_DIR)
    ap.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("backend/scripts/data/v11/seed_cleanocr_pairs.jsonl"),
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
    submode_counts = Counter(r.submode_guess for r in rows)
    print(f"bench-dir       : {args.bench_dir}")
    print(f"pages scanned   : {sum(1 for _ in iter_bubbles_files(args.bench_dir))}")
    print(f"ocr-conf-min    : {args.ocr_conf_min}")
    print(f"gt-mode         : {args.gt_mode}")
    print(f"clean-OCR rows  : {len(rows)}")
    print(f"  with human_en : {have_gt}  (rest need the manual/vision GT pass)")
    print("submode_guess   :")
    for sm in ("vocab", "negation", "idiom", "other"):
        print(f"    {sm:<9}: {submode_counts.get(sm, 0)}")
    print(f"wrote           : {args.out}")
    # REMAINING WORK (not this script — see plan §9):
    #   1. human_en recovery: GT is TYPESET in the page .webp; fill via manual
    #      transcription or a reviewed vision-OCR draft (never auto-gold).
    #   2. human review: verify submode + register_tag, euphemism-audit vocab
    #      rows, set keep=True/False, drop garbled/ambiguous-polarity rows.
    #   3. promote keep==True rows -> v11 corrective seed schema
    #      [jp, en=human_en, src, register_tag, gold_flag=True].
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

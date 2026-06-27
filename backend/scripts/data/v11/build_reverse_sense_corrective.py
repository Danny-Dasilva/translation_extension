#!/usr/bin/env python3
"""ITEM 2: reverse-sense corrective DATA set for the largest pure-model bucket.

The Ikenie 4 audit's biggest pure-model bucket (55 of the 160 mistranslation
hits) is REVERSE-SENSE lexical errors on CLEAN OCR: the Japanese was read
correctly, but the model rendered the OPPOSITE / a wrong sense of a known
lexeme. Documented cases:

    締まる   tightens          -> our wrong: 'closing'
    吸い出せ  suck OUT          -> our wrong: 'spit out'
    果てた   climaxed          -> our wrong: 'passed away'
    風俗     brothel / sex-trade-> our wrong: 'rumor'
    騎乗位   cowgirl (position) -> our wrong: 'coworking'
    割る     dilute (a drink)   -> our wrong: 'break'
    尻       butt               -> our wrong: 'balls'
    マンコ   pussy              -> our wrong: 'butthole'

This builder emits, for EACH lexeme, 2-3 VARIED JP carriers (distinct lines, so
the model learns the SENSE rather than memorizing one surface form), in BOTH
plain and page-context shape, each with:

  * ``human_en``  : the GOLD English — left EMPTY (TODO). human_en for NEW
                    carriers needs the gold eval set from the eval workstream.
                    We NEVER fabricate a gold target.
  * ``our_wrong`` : the wrong sense the model currently emits (curation trigger +
                    held-out contrastive probe; NOT a DPO rejected signal — DPO
                    toward NSFW caused the v12 euphemism regression).
  * ``contrastive_margin`` : chrF++(human_en) - chrF++(our_wrong), computed
                    DOWNSTREAM once human_en exists. None here.

NSFW handling: the NSFW fraction is kept FLAT. Every lexeme contributes the SAME
number of rows (one plain + one pagectx per carrier); there is NO per-NSFW
multiplier. Oversampling NSFW backfired into euphemism in v12 (see MEMORY.md
v12_nsfw_oversampling_regression) — do NOT oversample here.

The page-context carriers reuse build_v11_dataset.build_context_prompt so the
prompt is byte-identical to the trained v11 page-context template.

Run:  .venv/bin/python backend/scripts/data/v11/build_reverse_sense_corrective.py
Output: JSONL of structured reverse-sense corrective rows (human_en TODO).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Sibling import: build_context_prompt / PAGE_INSTR are the byte-exact trained
# page-context template. conftest puts backend/scripts/data on sys.path, and the
# CLI run-from-here case is handled in __main__.
try:
    from v11.build_v11_dataset import PAGE_INSTR, build_context_prompt, build_plain_prompt
except Exception:  # pragma: no cover - standalone CLI fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from v11.build_v11_dataset import PAGE_INSTR, build_context_prompt, build_plain_prompt


REGISTER_DEFAULT = "vn_eroge"  # adult-domain carriers; must exist in data_v10


@dataclass
class Carrier:
    """One VARIED JP line carrying the target lexeme + its page-context window.

    ``context_jp`` is the ordered page/window (including this line); ``context_k``
    is the 0-based index of this carrier within it. ``register_tag`` lets a
    carrier override the entry default (e.g. a manga_dialog carrier).
    """

    jp: str
    context_jp: list[str]
    context_k: int
    register_tag: str = REGISTER_DEFAULT


@dataclass
class ReverseSenseEntry:
    """A reverse-sense lexeme + its 2-3 varied carriers and sense annotations."""

    lexeme: str
    right_sense: str          # the CORRECT sense (human-readable note)
    our_wrong_sense: str      # the WRONG sense the model emits
    nsfw: bool
    carriers: list[Carrier]


# --------------------------------------------------------------------------- #
# The audit set. Carriers are DISTINCT surface forms so the model learns the
# SENSE, not one memorized line. human_en (gold) is supplied DOWNSTREAM.
# --------------------------------------------------------------------------- #
REVERSE_SENSE_ENTRIES: list[ReverseSenseEntry] = [
    ReverseSenseEntry(
        lexeme="締まる",
        right_sense="tightens / clenches (muscle, grip)",
        our_wrong_sense="'closing' (mistaken for shutting a shop/door)",
        nsfw=True,
        carriers=[
            Carrier("中がすごく締まってる", ["どう、気持ちいい？", "中がすごく締まってる", "うん…"], 1),
            Carrier("ぎゅっと締まってきた", ["もうイキそう？", "ぎゅっと締まってきた"], 1),
            Carrier("そんなに締めないで", ["ああっ", "そんなに締めないで"], 1),
        ],
    ),
    ReverseSenseEntry(
        lexeme="吸い出せ",
        right_sense="suck OUT / draw out (extract)",
        our_wrong_sense="'spit out' (reversed direction)",
        nsfw=True,
        carriers=[
            Carrier("全部吸い出せ", ["まだ残ってる", "全部吸い出せ"], 1),
            Carrier("中のを吸い出してやる", ["動かないで", "中のを吸い出してやる"], 1),
        ],
    ),
    ReverseSenseEntry(
        lexeme="果てた",
        right_sense="climaxed / came (sexual)",
        our_wrong_sense="'passed away' (mistaken for 'died/expired')",
        nsfw=True,
        carriers=[
            Carrier("とうとう果ててしまった", ["もうダメ…", "とうとう果ててしまった"], 1),
            Carrier("彼女の中で果てた", ["うっ", "彼女の中で果てた"], 1),
            Carrier("何度も果てさせられた", ["まだ終わらない", "何度も果てさせられた"], 1),
        ],
    ),
    ReverseSenseEntry(
        lexeme="風俗",
        right_sense="sex-trade / brothel (commercial)",
        our_wrong_sense="'rumor' (confused with 噂/風説)",
        nsfw=True,
        carriers=[
            Carrier("風俗で働いてるの", ["仕事は何？", "風俗で働いてるの"], 1),
            Carrier("風俗に行ったことある？", ["なあ", "風俗に行ったことある？"], 1),
        ],
    ),
    ReverseSenseEntry(
        lexeme="騎乗位",
        right_sense="cowgirl / woman-on-top position",
        our_wrong_sense="'coworking' (phonetic garble)",
        nsfw=True,
        carriers=[
            Carrier("騎乗位で動いて", ["上に乗って", "騎乗位で動いて"], 1),
            Carrier("今度は騎乗位がいい", ["どうする？", "今度は騎乗位がいい"], 1),
        ],
    ),
    ReverseSenseEntry(
        lexeme="割る",
        right_sense="dilute / cut (a drink with water/mixer)",
        our_wrong_sense="'break' (literal smashing)",
        nsfw=False,
        carriers=[
            Carrier("水で割って飲む", ["濃いから", "水で割って飲む"], 1),
            Carrier("お湯で割ろうか", ["焼酎どうする？", "お湯で割ろうか"], 1),
        ],
    ),
    ReverseSenseEntry(
        lexeme="尻",
        right_sense="butt / buttocks",
        our_wrong_sense="'balls' (wrong body part)",
        nsfw=True,
        carriers=[
            Carrier("尻を突き出して", ["こっち向いて", "尻を突き出して"], 1),
            Carrier("尻がエロい", ["うわ", "尻がエロい"], 1),
        ],
    ),
    ReverseSenseEntry(
        lexeme="マンコ",
        right_sense="pussy / vagina",
        our_wrong_sense="'butthole' (wrong orifice)",
        nsfw=True,
        carriers=[
            Carrier("マンコが濡れてる", ["ほら", "マンコが濡れてる"], 1),
            Carrier("マンコに挿れて", ["早く", "マンコに挿れて"], 1),
        ],
    ),
]


def build_rows() -> list[dict]:
    """Emit structured reverse-sense corrective rows (human_en is TODO).

    Per carrier: ONE plain row + ONE page-context row (flat across NSFW/SFW; no
    oversampling). Each row carries the prompt, the empty gold ``human_en``, the
    ``our_wrong`` sense annotation, a ``contrastive_margin`` placeholder (None),
    and ``needs_gold=True``.
    """
    rows: list[dict] = []
    for e in REVERSE_SENSE_ENTRIES:
        for ci, c in enumerate(e.carriers):
            base = {
                "lexeme": e.lexeme,
                "right_sense": e.right_sense,
                "our_wrong": e.our_wrong_sense,
                "human_en": "",            # GOLD — TODO from the eval workstream
                "needs_gold": True,
                "contrastive_margin": None,  # chrF++(human) - chrF++(our_wrong)
                "nsfw": e.nsfw,
                "register_tag": c.register_tag,
                "gold_flag": True,
                "context_jp": c.context_jp,
                "context_k": c.context_k,
                "jp": c.jp,
            }
            src = f"reverse_sense:{e.lexeme}:{ci}"
            # plain shape
            rows.append({
                **base,
                "shape": "plain",
                "src": src + ":plain",
                "prompt": build_plain_prompt(c.jp),
            })
            # page-context shape (byte-identical to the trained template)
            rows.append({
                **base,
                "shape": "pagectx",
                "src": src + ":pagectx",
                "prompt": build_context_prompt(PAGE_INSTR, c.context_jp, c.context_k),
            })
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build reverse-sense corrective rows.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "reverse_sense_corrective.jsonl",
    )
    args = ap.parse_args(argv)

    rows = build_rows()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_lex = len(REVERSE_SENSE_ENTRIES)
    n_carriers = sum(len(e.carriers) for e in REVERSE_SENSE_ENTRIES)
    n_nsfw = sum(1 for e in REVERSE_SENSE_ENTRIES if e.nsfw)
    print(f"lexemes        : {n_lex}  (nsfw {n_nsfw} / sfw {n_lex - n_nsfw})")
    print(f"carriers       : {n_carriers}  (2-3 varied per lexeme)")
    print(f"rows           : {len(rows)}  (plain + pagectx per carrier)")
    print(f"human_en TODO  : {sum(1 for r in rows if r['needs_gold'])} (gold from eval workstream)")
    print(f"wrote          : {args.out}")
    print("DOWNSTREAM: fill human_en from the gold set, then compute")
    print("            contrastive_margin = chrF++(human_en) - chrF++(our_wrong).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

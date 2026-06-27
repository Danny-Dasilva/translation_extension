#!/usr/bin/env python3
"""ITEM 3: GRAMMATICAL VOICE / ADDRESSEE probe + data set.

Neither fix6 (clean-OCR lexical corrective) nor fix8 covers GRAMMATICAL VOICE.
Two distinct, recurring voice failures the eval surfaced:

  1. causative-passive ``させられる`` ('be MADE to do X')
     The model collapses the causative-passive into a plain past: it renders
     "I did X" instead of "I was MADE to do X" — losing the coerced-agent
     meaning that matters in this (dubcon) domain.

  2. 2nd<->1st-person COMMAND inversion
     A 2nd-person imperative/command ('keep them on') is inverted into a
     1st-person past report ('I kept it on') — the addressee and the speaker
     are swapped.

This builds a SMALL, structured probe/data set of these patterns WITH gold
targets, so a future SFT can target voice-correctness and the eval can MEASURE
it (voice-correct vs voice-inverted), independent of surface chrF++.

Each entry carries:
  jp        : the Japanese line
  gold_en   : the CORRECT English (voice/addressee preserved)
  wrong_en  : the characteristic WRONG output (the inversion we guard against;
              a curation/eval contrastive signal, NOT a DPO rejected sample)
  category  : "causative_passive" | "command_addressee"
  note      : what the probe checks

Emitted in BOTH plain and page-context shape (page-context reuses
build_v11_dataset.build_context_prompt, byte-identical to the trained template).

Run:  .venv/bin/python backend/scripts/data/v11/build_voice_addressee_probe.py
Output: a JSONL probe set (gold targets) + this module's docstring as the doc.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from v11.build_v11_dataset import PAGE_INSTR, build_context_prompt, build_plain_prompt
except Exception:  # pragma: no cover - standalone CLI fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from v11.build_v11_dataset import PAGE_INSTR, build_context_prompt, build_plain_prompt


REGISTER_DEFAULT = "vn_eroge"


@dataclass
class VoiceProbeEntry:
    """One voice/addressee pattern with a GOLD target + the inversion to guard."""

    category: str          # "causative_passive" | "command_addressee"
    jp: str
    gold_en: str           # CORRECT (voice/addressee preserved)
    wrong_en: str          # the characteristic inversion (contrastive signal)
    context_jp: list[str]  # page/window for the page-context shape (incl. line)
    context_k: int
    note: str = ""
    register_tag: str = REGISTER_DEFAULT


# --------------------------------------------------------------------------- #
# The probe set. GOLD targets are human-authored canonical translations of
# textbook voice patterns (NOT recovered from a specific page) — these encode
# the grammatical contract (causative-passive => 'made to'; 2nd-person command
# stays 2nd-person), so they are safe gold for a voice probe.
# --------------------------------------------------------------------------- #
VOICE_PROBE_ENTRIES: list[VoiceProbeEntry] = [
    # ---- causative-passive させられる: 'be MADE to do' --------------------- #
    VoiceProbeEntry(
        category="causative_passive",
        jp="無理やり飲まされた",
        gold_en="I was made to drink it against my will.",
        wrong_en="I drank it.",
        context_jp=["やめて", "無理やり飲まされた"],
        context_k=1,
        note="causative-passive 飲まされた = 'was made to drink', not 'drank'",
    ),
    VoiceProbeEntry(
        category="causative_passive",
        jp="何度も言わされた",
        gold_en="I was made to say it over and over.",
        wrong_en="I said it many times.",
        context_jp=["ほら、もう一回", "何度も言わされた"],
        context_k=1,
        note="言わされた = 'was made to say', preserves coerced agent",
    ),
    VoiceProbeEntry(
        category="causative_passive",
        jp="服を脱がされた",
        gold_en="I was made to take my clothes off.",
        wrong_en="I took off my clothes.",
        context_jp=["動かないで", "服を脱がされた"],
        context_k=1,
        note="脱がされた = 'was made to undress', not the volitional 'I undressed'",
    ),
    VoiceProbeEntry(
        category="causative_passive",
        jp="ずっと待たされてた",
        gold_en="I was being kept waiting the whole time.",
        wrong_en="I waited the whole time.",
        context_jp=["遅いよ", "ずっと待たされてた"],
        context_k=1,
        note="待たされてた = 'was kept waiting' (causative-passive continuous)",
    ),
    VoiceProbeEntry(
        category="causative_passive",
        jp="正座させられた",
        gold_en="I was made to sit on my knees.",
        wrong_en="I sat on my knees.",
        context_jp=["そこに", "正座させられた"],
        context_k=1,
        note="canonical させられる causative-passive = 'was made to', not 'I did'",
    ),
    # ---- 2nd<->1st-person COMMAND inversion ------------------------------- #
    VoiceProbeEntry(
        category="command_addressee",
        jp="つけたままにして",
        gold_en="Keep them on.",
        wrong_en="I kept it on.",
        context_jp=["脱がないで", "つけたままにして"],
        context_k=1,
        note="2nd-person imperative -て command; must NOT become 1st-person past",
    ),
    VoiceProbeEntry(
        category="command_addressee",
        jp="そこに座ってて",
        gold_en="Stay sitting there.",
        wrong_en="I sat there.",
        context_jp=["ちょっと待って", "そこに座ってて"],
        context_k=1,
        note="-てて command to the addressee, not a 1st-person report",
    ),
    VoiceProbeEntry(
        category="command_addressee",
        jp="こっちを見て",
        gold_en="Look at me.",
        wrong_en="I looked over here.",
        context_jp=["ねえ", "こっちを見て"],
        context_k=1,
        note="2nd-person command; addressee looks, speaker does not report",
    ),
    VoiceProbeEntry(
        category="command_addressee",
        jp="声を出さないで",
        gold_en="Don't make a sound.",
        wrong_en="I didn't make a sound.",
        context_jp=["静かに", "声を出さないで"],
        context_k=1,
        note="negative 2nd-person command, not a 1st-person past negation",
    ),
]


def build_rows() -> list[dict]:
    """Emit the probe rows: per entry, ONE plain + ONE page-context row."""
    rows: list[dict] = []
    for i, e in enumerate(VOICE_PROBE_ENTRIES):
        base = {
            "category": e.category,
            "jp": e.jp,
            "gold_en": e.gold_en,
            "wrong_en": e.wrong_en,
            "note": e.note,
            "register_tag": e.register_tag,
            "gold_flag": True,
            "context_jp": e.context_jp,
            "context_k": e.context_k,
        }
        src = f"voice_probe:{e.category}:{i}"
        rows.append({
            **base,
            "shape": "plain",
            "src": src + ":plain",
            "prompt": build_plain_prompt(e.jp),
        })
        rows.append({
            **base,
            "shape": "pagectx",
            "src": src + ":pagectx",
            "prompt": build_context_prompt(PAGE_INSTR, e.context_jp, e.context_k),
        })
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build voice/addressee probe rows.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "voice_addressee_probe.jsonl",
    )
    args = ap.parse_args(argv)

    rows = build_rows()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_cat: dict[str, int] = {}
    for e in VOICE_PROBE_ENTRIES:
        by_cat[e.category] = by_cat.get(e.category, 0) + 1
    print(f"entries        : {len(VOICE_PROBE_ENTRIES)}")
    for cat, n in sorted(by_cat.items()):
        print(f"    {cat:<18}: {n}")
    print(f"rows           : {len(rows)}  (plain + pagectx per entry)")
    print(f"wrote          : {args.out}")
    print("USAGE: gold_en is the voice-correct target for SFT; wrong_en is the")
    print("       contrastive inversion the eval measures against (NOT a DPO neg).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

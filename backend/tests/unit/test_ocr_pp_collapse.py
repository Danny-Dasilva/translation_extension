"""FIX P3-1: collapse exact whole-phrase P+P OCR duplicates instead of dropping.

~20% of dup-gate drops are CLEAN high-confidence Japanese whose two halves are
an exact repeat (お母さんお母さん@0.93). These must COLLAPSE to one copy and be
KEPT (so the partner bubble is not orphaned), not dropped as omissions.
"""
from __future__ import annotations

from app.utils.ocr_confidence_gate import (
    collapse_immediate_dup,
    is_garbled_low_conf,
)


# --- clean P+P MUST collapse+keep -----------------------------------------
def test_collapse_whole_string_repeat():
    assert collapse_immediate_dup("お母さんお母さん") == "お母さん"


def test_collapse_jp_glyph_only_repeat_with_punct():
    # JP glyphs form P+P even though punctuation differs between halves.
    out = collapse_immediate_dup("また昨日みたいな…また昨日みたいな…")
    assert out is not None
    assert out.startswith("また昨日みたいな")
    # collapsed to roughly one half (much shorter than the doubled input)
    assert len(out) < len("また昨日みたいな…また昨日みたいな…")


def test_collapse_keeps_line_not_dropped_high_conf():
    # The whole point: a clean collapsible dup is NOT a gate drop.
    assert is_garbled_low_conf("お母さんお母さん", 0.93) is False
    assert is_garbled_low_conf("洗濯バサミ洗濯バサミ", 0.91) is False


def test_collapse_keeps_line_not_dropped_mid_conf():
    # Even at a confidence below the dup-ceiling, a clean P+P collapses+keeps.
    assert is_garbled_low_conf("また昨日みたいなまた昨日みたいな", 0.78) is False


# --- garble doubles MUST NOT be "rescued" ---------------------------------
def test_garble_double_not_collapsed():
    # A doubled GARBLE half is implausible, so collapse refuses (returns None);
    # the dup-gate still drops it.
    assert collapse_immediate_dup("身身わわ身身わわ") is None


def test_non_repeat_unchanged():
    assert collapse_immediate_dup("こんにちは元気ですか") is None


def test_legit_short_double_not_collapsed():
    # ますます is a legit reduplication, not a P+P garble — no collapse.
    assert collapse_immediate_dup("ますます") is None

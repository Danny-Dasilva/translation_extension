"""Unit tests for the OCR-confidence garble gate (FIX 2).

Garbled / low-confidence OCR on stylized SFX must NOT reach the LLM (it
hallucinates non-English captions). The gate drops a bubble only when OCR
recognition confidence is low AND the text looks garbled. Real dialogue
(measured OCR conf ~0.9+) must always pass, even when short.
"""
from __future__ import annotations

from app.utils.ocr_confidence_gate import is_garbled_low_conf


# --- must DROP: low confidence AND garbled --------------------------------
def test_drops_low_conf_replacement_chars():
    assert is_garbled_low_conf("も...]]]..", 0.61) is True


def test_drops_low_conf_bracket_scrawl():
    # 045 idx6 illegible scrawl, low OCR conf.
    assert is_garbled_low_conf("]] [[ //", 0.45) is True


# --- must KEEP: high confidence even if short -----------------------------
def test_keeps_high_conf_dialogue():
    assert is_garbled_low_conf("イキたいんなら自分で動きなさい!", 0.95) is False


def test_keeps_high_conf_short_sfx():
    # Real katakana SFX with good confidence stays (translated to onomatopoeia).
    assert is_garbled_low_conf("ドン", 0.90) is False


def test_drops_short_lowconf_kana_sfx():
    # OCR conf cleanly separates real dialogue (>=0.85) from garbled short SFX
    # (<0.65). A short low-conf kana scrawl is dropped (these hallucinate).
    assert is_garbled_low_conf("よっピ", 0.57) is True
    assert is_garbled_low_conf("こちにちこち", 0.62) is True


def test_keeps_long_lowconf_dialogue():
    # A long, mostly-Japanese line at low conf is genuine hard dialogue — keep
    # it (better to translate a hard line than silently lose it).
    assert is_garbled_low_conf("お母さんの匂いがたぁ〜っぷり染みついたブラ", 0.55) is False


# --- threshold boundary ----------------------------------------------------
def test_above_threshold_never_dropped_even_if_weird():
    # Above the conf gate, we never drop regardless of text quality.
    assert is_garbled_low_conf("]]]///", 0.80) is False


def test_low_conf_low_jp_ratio_dropped():
    # Low conf + mostly non-Japanese chars => garbled => drop.
    assert is_garbled_low_conf("abc]] xy", 0.55) is True

"""FIX P3-2 (dup-conf ceiling) + P3-3 (substitution/perplexity garble guard).

P3-2: dup-only signals are skipped at/above the high-confidence ceiling so a
clean conf-0.93 bubble that trips a dup heuristic is NOT dropped.

P3-3: a corpus-grounded char-bigram guard flags LONG high-entropy noise the
dup-predicates are blind to, routing it to erase-only. Calibrated to flag ZERO
clean dialogue / NSFW lines (see report) — substitution garbles like もっ張って
are documented as out of reach for a bigram model and are NOT expected to flag.
"""
from __future__ import annotations

from app.utils.ocr_confidence_gate import (
    DUP_CONF_CEILING,
    _is_substitution_garble,
    is_implausible_japanese,
)


# --- P3-2: high-confidence dup ceiling ------------------------------------
def test_high_conf_dup_not_flagged():
    # お母さんお母さん would trip _immediate_substring_dup, but at conf >= ceiling
    # the dup-only signals are skipped (recognizer is trusted).
    assert is_implausible_japanese("お母さんお母さん", DUP_CONF_CEILING) is False


def test_high_conf_adjacent_dup_kanji_flagged_fix2():
    # FIX-2 recalibration: adjacent doubled kanji (止止) is now UNCONDITIONAL, so
    # it is flagged even at conf 0.90 (>= the 0.88 ceiling). The ceiling only
    # spares the length/bigram dup signals (see test_high_conf_dup_not_flagged).
    assert is_implausible_japanese("止止まらない頃", 0.90) is True


def test_low_conf_dup_still_flagged():
    # Below the ceiling the dup signal still fires (back-compat).
    assert is_implausible_japanese("身身わわ", 0.60) is True


def test_no_conf_keeps_legacy_behavior():
    # Text-only call (no confidence) keeps the original always-on dup behavior.
    assert is_implausible_japanese("身身わわ") is True


# --- P3-2: unconditional signals stay ON even at high confidence -----------
def test_latin_intrusion_unconditional_at_high_conf():
    assert is_implausible_japanese("oisxoovitterさん", 0.95) is True


def test_garbled_leading_tsu_unconditional_at_high_conf():
    assert is_implausible_japanese("っく混みますよ", 0.95) is True


# --- P3-3: substitution/perplexity guard ----------------------------------
def test_long_noise_scramble_flagged():
    # Long high-entropy noise the dup-predicates miss -> flagged (erase-only).
    assert _is_substitution_garble("ゴム昔化湖ゴム首次角") is True


# Legit reduplication MUST NOT be flagged by the perplexity guard.
def test_legit_redup_not_flagged_by_guard():
    for s in ("様々", "段々", "ますます", "はは"):
        assert _is_substitution_garble(s) is False, s


# Clean dialogue + explicit NSFW vocab MUST NOT be flagged (false-drop guard).
def test_clean_and_nsfw_not_flagged_by_guard():
    spare = [
        "あゆむ",                       # legit name (task: do NOT touch)
        "洗濯バサミ",
        "毎日チンポ",                   # explicit NSFW
        "おばさんの膣筋",               # explicit NSFW
        "口も膣も穴は",                 # explicit NSFW
        "オラ全部脱げ",
        "顔射決めたー",                 # explicit NSFW
        "ハメまくってる",               # explicit NSFW
        "イキたいんなら自分で動きなさい",
        "お母さんお母さん",
        "また昨日みたいな",
        "本当に大丈夫か",
    ]
    for s in spare:
        assert _is_substitution_garble(s) is False, s


def test_short_substitution_garbles_documented_blind():
    # CALIBRATION FINDING: single-char substitution garbles sit inside the legit
    # bigram distribution (もっ張って ppl ~= 引っ張って ppl), so a bigram model
    # cannot catch them without false-dropping NSFW vocab. This test PINS that
    # documented limitation — the real fix is the OCR fine-tune (fix7), not this
    # guard. If a future model lets these flag safely, update this test.
    for s in ("もっ張って", "控え込んでた"):
        assert _is_substitution_garble(s) is False, s

"""Unit tests for the OCR-confidence garble gate.

Garbled / low-confidence OCR on stylized SFX must NOT reach the LLM (it
hallucinates non-English captions). The gate drops a bubble only when OCR
recognition confidence is low AND the text looks garbled, OR when the text
carries a hard structural garble-signature at ANY confidence.

Recalibrated on the 650-row conf x sim-to-gold table
(``scripts/eval/scorecards/ikenie4/preds_for_gold_v1_fair.jsonl``):
  * short strings (< 5 chars) are ~98% correct vs gold at any confidence -> kept
    (short-text carve-out; the old length-12 rule silently dropped correct SFX);
  * multi-char lines need conf >= 0.80 (the 0.65-0.80 band is a ~40-55%-correct
    noise trough) -> raised long-text threshold;
  * adjacent doubled kanji/kana (身身わわ@0.92) are dropped unconditionally,
    no longer exempted by the DUP_CONF_CEILING;
  * BUG FIX: the doubled-katakana check (``_adjacent_dup_kana``) ran BEFORE the
    short-text carve-out even had a chance to apply, so a short doubled-kana
    giggle/SFX (ヒヒ.., アア, ヴヴ, シュウウ) was always dropped, at any
    confidence, with no whitelist escape hatch (audit: p114 idx9 "ヒヒ.." @
    0.9082, re-tested at 0.99, still dropped). Re-verified on the 650-row
    table: every false drop this rule produced there was a short (< 5 char)
    gold-exact SFX, while every genuine dup-garble catch (アソコアア,
    ...チチンの, ...ババブブババブ.., ...濯ササ, ...セッッスく) was 5+ chars. So
    the rule is now length-gated (mirrors the module's short-text carve-out)
    instead of widening the katakana whitelist, which would have silently
    un-caught real garble like アソコアア.
"""
from __future__ import annotations

from app.utils.ocr_confidence_gate import (
    DUP_CONF_CEILING,
    is_garbled_low_conf,
    is_implausible_japanese,
)


# --- must DROP: low confidence AND garbled --------------------------------
def test_drops_low_conf_replacement_chars():
    assert is_garbled_low_conf("も...]]]..", 0.61) is True


def test_drops_low_conf_bracket_scrawl():
    # 045 idx6 illegible scrawl, low OCR conf.
    assert is_garbled_low_conf("]] [[ //", 0.45) is True


# --- must KEEP: high confidence -------------------------------------------
def test_keeps_high_conf_dialogue():
    assert is_garbled_low_conf("イキたいんなら自分で動きなさい!", 0.95) is False


def test_keeps_high_conf_short_sfx():
    # Real katakana SFX with good confidence stays (translated to onomatopoeia).
    assert is_garbled_low_conf("ドン", 0.90) is False


# --- FIX-2: adjacent-dup garble is caught UNCONDITIONALLY ------------------
def test_drops_dup_kanji_at_falsely_high_conf():
    # 身代わり -> 身身わわ misread carrying FALSELY HIGH confidence (0.92 > the
    # 0.88 DUP_CONF_CEILING). Adjacent doubled kanji is now unconditional, so
    # this confidently-wrong dup-garble is dropped instead of sailing through.
    assert 0.92 > DUP_CONF_CEILING
    assert is_implausible_japanese("身身わわ", 0.92) is True
    assert is_garbled_low_conf("身身わわ", 0.92) is True


def test_drops_dup_katakana_at_high_conf():
    # Doubled non-laughter katakana (ヌヌ) is dup-garble even at high conf, as
    # long as the line clears the short-SFX length gate (>= 5 chars) — see
    # test_keeps_short_giggle_sfx below for the short-string exemption.
    assert is_implausible_japanese("ヌヌー界だよ", 0.91) is True


# --- BUG FIX: short doubled-katakana giggle/SFX is no longer false-dropped --
def test_keeps_short_giggle_sfx():
    # p114 idx9 (audit case): ヒヒ (giggle) was dropped as dup-garble at conf
    # 0.9082 and even at 0.99 — the check was unconditional and had no length
    # carve-out. A short doubled-kana giggle/SFX is now kept.
    assert is_implausible_japanese("ヒヒ..", 0.9082) is False
    assert is_implausible_japanese("ヒヒ..", 0.99) is False
    assert is_garbled_low_conf("ヒヒ..", 0.9082) is False
    assert is_garbled_low_conf("ヒヒ..", 0.99) is False


def test_keeps_other_short_katakana_dup_sfx():
    # Other common short doubled-katakana SFX/interjections from the 650-row
    # calib table, previously false-dropped by the unconditional rule.
    assert is_implausible_japanese("アア", 0.7351) is False    # moan/interjection
    assert is_implausible_japanese("ヴヴ", 0.9378) is False    # buzz/vibration SFX
    assert is_implausible_japanese("シュウウ", 0.811) is False  # trailing ウウ, hiss/spray SFX


def test_dup_kana_length_gate_does_not_reopen_real_garble():
    # アソコアア (gold: アソコ) is a genuine PARSeq dup-garble — a bogus "アア"
    # appended to a clean word — at conf 0.9213 in the calib table. It is
    # exactly 5 chars (the length-gate boundary) and must stay caught: this
    # is the concrete case that rules out simply widening the whitelist to
    # include ア (which would have silently un-caught it).
    assert is_implausible_japanese("アソコアア", 0.9213) is True
    assert is_garbled_low_conf("アソコアア", 0.9213) is True


def test_dup_kanji_garble_still_caught_unaffected_by_kana_fix():
    # 濃濃くへ / 押押えて style doubled-KANJI garble is handled by a separate,
    # untouched rule (_adjacent_dup_kanji) and is unaffected by the katakana
    # length gate above — must still be dropped at any confidence.
    assert is_implausible_japanese("濃濃くへ", 0.9) is True
    assert is_implausible_japanese("押押えて", 0.9) is True
    assert is_garbled_low_conf("濃濃くへ", 0.9) is True
    assert is_garbled_low_conf("押押えて", 0.9) is True


def test_dup_ceiling_still_spares_phrase_repeat_signal():
    # The length/bigram dup signals still honour the ceiling: a clean whole-phrase
    # repeat at high conf is collapsed/kept, not dropped.
    assert is_garbled_low_conf("お母さんお母さん", 0.93) is False


# --- FIX-1: short-text carve-out (SFX / moans / numbers now KEPT) ----------
def test_keeps_short_lowconf_sfx_carveout():
    # Calib table: sub-0.65 short strings were EXACT gold matches yet silently
    # dropped by the old rule. A clean short SFX/moan is now kept.
    assert is_garbled_low_conf("もみせ", 0.42) is False   # len 3, exact-gold SFX
    assert is_garbled_low_conf("濃厚", 0.60) is False      # len 2


def test_keeps_short_lowconf_number():
    # A short number bubble (OCR '56' @0.61, exact gold) is now kept.
    assert is_garbled_low_conf("56", 0.61) is False


def test_short_carveout_does_not_rescue_garble_chars():
    # The carve-out never overrides garble-char / structural checks: a short
    # bracket scrawl at low conf still drops.
    assert is_garbled_low_conf("]]/", 0.40) is True


# --- FIX-1: raised long-text threshold (mid-conf long lines now DROP) ------
def test_drops_long_lowconf_dialogue():
    # A long, clean-Japanese line at mid confidence sits in the 0.65-0.80 noise
    # trough (~40-55% correct vs gold). Recalibration DROPS it (the old rule kept
    # it); it survives downstream as page CONTEXT, not a confident-wrong render.
    assert is_garbled_low_conf("お母さんの匂いがたぁ〜っぷり染みついたブラ", 0.70) is True


def test_drops_long_lowconf_dialogue_just_below_threshold():
    # Just under the 0.80 knee -> still dropped.
    assert is_garbled_low_conf("昨日あんな事をしていたなんて信じられない", 0.78) is True


def test_keeps_long_dialogue_above_threshold():
    # At/above the 0.80 knee a clean long line is kept (a genuine 0.80-conf
    # balloon column must survive so multi-column balloons stay intact).
    assert is_garbled_low_conf("昨日あんな事をしていたなんて信じられない", 0.80) is False


# --- threshold boundary ----------------------------------------------------
def test_above_threshold_never_dropped_even_if_weird():
    # Well above the (raised) conf gate we never drop regardless of text quality.
    assert is_garbled_low_conf("]]]///", 0.90) is False


def test_low_conf_low_jp_ratio_dropped():
    # Low conf + mostly non-Japanese chars => garbled => drop.
    assert is_garbled_low_conf("abc]] xy", 0.55) is True

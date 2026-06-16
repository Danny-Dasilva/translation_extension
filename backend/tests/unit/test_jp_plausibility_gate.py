"""Unit tests for the JP-plausibility OCR gate (FIX P1-1).

The confidence gate alone is blind to garbled OCR that carries FALSELY HIGH
confidence — e.g. page 070's vertical title "3へ続く…" misread as
"..?っく混みますよ" at ocr_conf 0.91, which then translated to a confident-wrong
"It's going to be crowded.". This adds a linguistic-plausibility signal so
confidently-garbled OCR is caught even at high confidence, WITHOUT dropping
genuine dialogue.

Design constraint: false-dropping real dialogue is worse than letting some
garble through. The signals are deliberately narrow and were calibrated so they
fire on ZERO of 600+ real dialogue lines in the replay corpus.
"""
from __future__ import annotations

from app.utils.ocr_confidence_gate import (
    is_garbled_low_conf,
    is_implausible_japanese,
)


# --- is_implausible_japanese: must flag the garbles -----------------------
def test_implausible_bad_leading_small_tsu():
    # page 070: a line cannot legitimately begin with small-tsu + a kana other
    # than て/た. "っく..." is a garbled prefix glued onto "混みますよ".
    assert is_implausible_japanese("..?っく混みますよ") is True


def test_implausible_latin_intrusion_credit_line():
    # page 071 idx1: garbled URL/credit line with heavy latin intrusion.
    s = (
        "連絡先:(kakyLuvishi 8のemau.conXCTすrnterン:(00kalkマエ Dushile"
        "この作品はフィクションです。実在の人物.団体.事件とは一切関係がありません"
    )
    assert is_implausible_japanese(s) is True


def test_implausible_latin_garble_handle():
    # page 074 idx1: "oisxo、((ovitter..." — OCR latched onto a logo/handle.
    assert is_implausible_japanese("oisxo、((ovitterにイラスト投稿してます。。是非フォローお願いします!") is True


# --- is_implausible_japanese: must NOT flag real dialogue -----------------
def test_plausible_real_dialogue():
    for s in [
        "イキたいんなら自分で動きなさい!",
        "いつの話よそんなの",
        "あぁっ!あの時かっ!クリスマスの!懐かしい..",
        "かわいい娘さんっすねー一緒写ってたの奥さんっすか?",
        "熱っ..ある...みたい...でっ..",
        "子供がっ..",
        "お母さんの匂いがたぁ〜っぷり染みついたブラ",
    ]:
        assert is_implausible_japanese(s) is False, s


def test_plausible_legit_leading_tsu_forms():
    # って (quotation), ったく (=まったく), trailing っ — all real speech.
    for s in ["...ってあれ?", "ったく..", "ったくもっと稼いでこいよな", "っ..", "ってなんだよ.."]:
        assert is_implausible_japanese(s) is False, s


def test_plausible_single_digit_in_japanese():
    # A lone digit embedded in Japanese is normal (年号, counters).
    for s in ["10年経って人妻になった女を", "おいおいまだ1セット目だぞ加奈子", "じゃあ今から3回イったら決定な"]:
        assert is_implausible_japanese(s) is False, s


# --- gate wiring: high-conf garble now DROPS ------------------------------
def test_gate_drops_high_conf_implausible_070():
    # The headline P1-1 case: high confidence (0.91) but garbled -> now dropped.
    assert is_garbled_low_conf("..?っく混みますよ", 0.91) is True


def test_gate_keeps_high_conf_real_dialogue():
    # Real dialogue at high conf must still pass.
    assert is_garbled_low_conf("いつの話よそんなの", 0.93) is False
    assert is_garbled_low_conf("かわいい娘さんっすねー一緒写ってたの奥さんっすか?", 0.93) is False


def test_plausibility_check_can_be_disabled():
    # The plausibility signal is gated so behavior stays tunable. With it off,
    # a high-conf implausible line is NOT dropped (pre-P1-1 behavior).
    assert is_garbled_low_conf("..?っく混みますよ", 0.91, check_plausibility=False) is False

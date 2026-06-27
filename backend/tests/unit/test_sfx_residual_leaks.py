"""FIX P3-4: residual SFX romaji leaks.

New onomatopoeia added to SFX_MAP, and a romaji-leak fallback in
clean_sfx_output (a pure-kana marker box that round-trips to raw romaji is
replaced with a glossary SFX / *...*, never rendered as raw romaji).
(あゆむ -> Ayumu and *Haa* are CORRECT and must NOT be touched.)
"""
from __future__ import annotations

from app.services.sfx_glossary import (
    SFX_MAP,
    clean_sfx_output,
    sfx_pre_translate,
)


# --- new SFX map entries ---------------------------------------------------
def test_new_sfx_keys_present_and_mapped():
    expected = {
        "ばぽ": "Plop",
        "たらん": "Plink",
        "おご": "Gulp",
        "ぼろおろ": "Bworp",
        "ボロオロ": "Bworp",
        "めぷ": "Squish",
        "ぽん": "Pop",
    }
    for key, val in expected.items():
        assert SFX_MAP.get(key) == val, key


def test_new_sfx_pre_translate_bypass():
    # A whole-box pure-kana SFX box short-circuits to its glossary English.
    assert sfx_pre_translate("ばっぽ") == "Plop"
    assert sfx_pre_translate("ぽん") == "Pop"
    assert sfx_pre_translate("ボロオロ") == "Bworp"


# --- romaji-leak fallback --------------------------------------------------
def test_romaji_leak_replaced_with_glossary():
    # A kana SFX whose English round-tripped to raw romaji is replaced.
    assert clean_sfx_output("Bappo", "ばっぽ") == "Plop"
    assert clean_sfx_output("Pon", "ぽん") == "Pop"


def test_romaji_leak_unknown_falls_back_to_ellipsis():
    # A kana SFX source not in the glossary, leaked as romaji -> *...*.
    out = clean_sfx_output("Tarurun", "たるるん")
    assert out in ("*...*", "Plink") or out.startswith("*")
    assert out != "Tarurun"


# --- must NOT touch correct outputs ---------------------------------------
def test_ayumu_name_not_touched():
    # あゆむ -> Ayumu is a CORRECT name romanisation, not a leak.
    assert clean_sfx_output("Ayumu", "あゆむ") == "Ayumu"


def test_haa_not_touched():
    # *Haa* is a CORRECT breath SFX, not a leak.
    assert clean_sfx_output("*Haa*", "はぁ") == "*Haa*"


def test_real_english_sfx_not_touched():
    # A genuine one-word English SFX is not a romaji leak (doesn't equal the JP
    # romaji), so it passes through.
    assert clean_sfx_output("Boom", "ドン") in ("Boom",)


def test_real_dialogue_not_touched():
    assert (
        clean_sfx_output("I'm really fine.", "本当に大丈夫だよ")
        == "I'm really fine."
    )

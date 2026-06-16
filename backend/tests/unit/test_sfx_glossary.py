"""Unit tests for SFX onomatopoeia post-processing (Fix P2-2).

Covers three problems found in QA of automated manga translation:
  1. Mistranslated SFX onomatopoeia (パシッ -> "splash" should be "Smack").
  2. Verbose / low-value English SFX labels.
  3. WORST: model emits a META-DESCRIPTION of the sound instead of a
     translation ("SFX for a grunt of surprise...") and it gets typeset.

All handling is PURE POST-PROCESSING on output text (the v11 prompt is
train/serve format-sensitive and MUST NOT be touched).

Critical safety property: real dialogue must NEVER be rewritten, even when
it happens to contain the words "sound", "represents", or "SFX".
"""

import pytest

from app.services.sfx_glossary import (
    clean_sfx_output,
    is_sfx_meta_description,
    suppress_or_transliterate,
)


# ---------------------------------------------------------------------------
# is_sfx_meta_description — the META-DESCRIPTION leak detector
# ---------------------------------------------------------------------------

META_LEAKS = [
    "SFX for a quick, forceful movement, like a quick splash cover",
    "SFX for a grunt of surprise, or any other short grunt call.",
    "SFX for a grunt of surprise, or any other short grunt sound.",
    "SFX for a big, sudden revelation, like a revelation that has huge implications",
    "SFX for a tearing, ripping, or breaking sound, like tearing something thick",
    "SFX for a big, powerful movement",
    "SFX for a very fast, powerful movement",
    "SFX for a momentary shock or realization",
    "Smokey lunging, SFX for a grunt while getting down to serious action.",
]


@pytest.mark.parametrize("en", META_LEAKS)
def test_detects_real_meta_leaks(en):
    assert is_sfx_meta_description(en) is True


@pytest.mark.parametrize(
    "en",
    [
        "This onomatopoeia means a wet sound",
        "represents a loud crash",
        "represents the moment of impact",
        "the sound of a quick, forceful movement",
        "a wet squelching sound",
    ],
)
def test_detects_meta_pattern_variants(en):
    assert is_sfx_meta_description(en) is True


# --- NEGATIVE cases: real dialogue must NOT be flagged --------------------

NORMAL_DIALOGUE = [
    "It really doesn't work! It's over!!",
    "What is?",
    "Heh, give me more!",
    "I can't... I said I can't!!",
    "Wake up, wake up...",
    "This...!! This is bad",
    "Cum while I spray!",
    # words that overlap with the patterns but are genuine dialogue:
    "I love the sound of your voice.",
    "That sound scared me half to death.",
    "This necklace represents our love.",
    "He represents the whole company at the meeting.",
    "What a beautiful sound!",
    "Did you hear that sound?",
]


@pytest.mark.parametrize("en", NORMAL_DIALOGUE)
def test_does_not_flag_normal_dialogue(en):
    assert is_sfx_meta_description(en) is False


@pytest.mark.parametrize("en", ["", None, "   "])
def test_empty_is_not_meta(en):
    assert is_sfx_meta_description(en) is False


# ---------------------------------------------------------------------------
# clean_sfx_output — end-to-end behaviour
# ---------------------------------------------------------------------------


def test_meta_leak_with_jp_sfx_transliterates():
    # ゴボッ -> known map "Glug"/transliteration, not the meta sentence.
    out = clean_sfx_output(
        "SFX for a quick, forceful movement, like a quick splash cover", "ゴボッ"
    )
    assert "SFX for" not in out
    assert "movement" not in out
    assert out  # non-empty (transliteration or mapped SFX)


def test_meta_leak_without_jp_becomes_ellipsis():
    out = clean_sfx_output("SFX for a big, powerful movement", None)
    assert out in ("...", "")
    assert "SFX for" not in out


def test_meta_leak_with_unmappable_jp_transliterates():
    # がっ is not in SFX_MAP -> should romanize the katakana/kana, not leak meta.
    out = clean_sfx_output("SFX for a big, powerful movement", "がっ")
    assert "SFX for" not in out
    assert "movement" not in out


# --- known mistranslations corrected --------------------------------------


def test_pashii_splash_corrected_to_smack():
    assert clean_sfx_output("*Splash*", "パシッ") == "Smack"


def test_pashii_plain_splash_corrected():
    assert clean_sfx_output("splash", "パシッ") == "Smack"


def test_nuchu_munch_corrected_to_squelch():
    assert clean_sfx_output("Munch", "ヌチュ") == "Squelch"


def test_don_maps_to_boom():
    assert clean_sfx_output("thud", "ドン") == "Boom"


def test_zaazaa_maps_to_shhh():
    assert clean_sfx_output("rain", "ザアザア") == "Shhh"


def test_hiragana_zan_corrected_from_plunging():
    # ざんっ is hiragana; corrected only because it is an EXACT glossary key.
    assert clean_sfx_output("plunging", "ざんっ") == "Slash"


def test_hiragana_dialogue_not_mapped():
    # Short hiragana that is NOT an exact SFX key must pass through untouched.
    assert clean_sfx_output("Wait...", "まっ..") == "Wait..."
    assert clean_sfx_output("Ugh...", "あうー") == "Ugh..."


# --- conservative: SFX map only fires on short SFX-y jp + wrong common noun -


def test_correct_existing_sfx_left_alone():
    # Already a reasonable comic SFX; do not churn it.
    assert clean_sfx_output("Boom!", "ドンッ") == "Boom!"


def test_real_dialogue_not_touched_even_if_jp_short():
    # jp is dialogue (has hiragana / not katakana SFX), en is dialogue.
    en = "It really doesn't work! It's over!!"
    assert clean_sfx_output(en, "ホントにっだめっ!おわりっ!!") == en


def test_long_english_never_treated_as_sfx_mapping():
    # Even if jp were katakana, a long English sentence is dialogue, not SFX.
    en = "I can't believe you actually splashed water all over me again!"
    assert clean_sfx_output(en, "パシッ") == en


def test_none_translation_passthrough():
    assert clean_sfx_output(None, "タッ") in (None, "")


def test_empty_translation_passthrough():
    assert clean_sfx_output("", "タッ") == ""


def test_no_jp_normal_dialogue_passthrough():
    en = "Wake up, wake up..."
    assert clean_sfx_output(en, None) == en


# ---------------------------------------------------------------------------
# suppress_or_transliterate — direct unit
# ---------------------------------------------------------------------------


def test_suppress_or_transliterate_prefers_map():
    assert suppress_or_transliterate("パシッ") == "Smack"


def test_suppress_or_transliterate_romanizes_unmapped_katakana():
    out = suppress_or_transliterate("ガシャ")
    assert out and "SFX" not in out
    # romaji-ish, ascii only
    assert out.isascii()


def test_suppress_or_transliterate_none_returns_ellipsis():
    assert suppress_or_transliterate(None) == "..."


def test_suppress_or_transliterate_empty_returns_ellipsis():
    assert suppress_or_transliterate("") == "..."

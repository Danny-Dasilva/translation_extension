"""FIX B: SFX wet/jiggle/twitch/slurp/thrust glossary + PRE-LLM bypass gate.

Adds the missing onomatopoeia families to the existing sfx_glossary, plus a
PRE-LLM gate (`sfx_pre_translate`) that short-circuits a known-SFX box to its
glossary English so it NEVER reaches the LLM. Real failures fixed here:

    ぬちょ   -> "menace"      (now "Squelch")
    たぷん   -> "Maybe"       (now "Jiggle")
    ビクン   -> "Twinkle"     (now "Twitch")
    ガバガバに-> "Gabagabani"  (now "so loose"  — adjectival, NOT romaji)

Conservative: real dialogue (kanji/grammar, or only a prefix match) returns
None and is translated normally.
"""

import asyncio

import pytest

from app.services.sfx_glossary import (
    SFX_ADJ_MAP,
    SFX_MAP,
    sfx_pre_translate,
    suppress_or_transliterate,
)
from app.services.vllm_openai_translation_service import VLLMOpenAITranslationService


# ---------------------------------------------------------------------------
# New onomatopoeia families resolve through the PRE-LLM gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "jp,expected",
    [
        # wet / squelch
        ("ぬちょ", "Squelch"), ("にちゃ", "Squelch"), ("ぐちゅ", "Squelch"),
        ("じゅぷ", "Squelch"), ("じゃぽ", "Squelch"), ("ねちょ", "Squelch"),
        ("びちゃ", "Squelch"), ("ずちゅ", "Squelch"),
        # slurp / suck
        ("ちゅぱ", "Slurp"), ("じゅぽ", "Slurp"), ("れろ", "Lick"),
        ("べろ", "Lick"), ("ちゅう", "Suck"), ("ぢゅる", "Slurp"),
        # jiggle
        ("たぷん", "Jiggle"), ("ぷるん", "Jiggle"), ("ぶるん", "Jiggle"),
        ("たゆん", "Jiggle"), ("ぼいん", "Boing"),
        # twitch / throb
        ("ビクン", "Twitch"), ("ピクン", "Twitch"), ("ドクン", "Throb"),
        ("ズキン", "Throb"), ("びくっ", "Twitch"),
        # thrust
        ("ずぶ", "Thrust"), ("ぬぷ", "Thrust"), ("ぐぽ", "Thrust"),
    ],
)
def test_new_families_bypass_to_glossary(jp, expected):
    assert sfx_pre_translate(jp) == expected


# ---------------------------------------------------------------------------
# Adjectival ガバガバ -> "so loose" (NOT a romaji transliteration)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("jp", ["ガバガバ", "がばがば", "ガバガバに", "ガバガバと"])
def test_gabagaba_is_adjectival_not_romaji(jp):
    out = sfx_pre_translate(jp)
    assert out in ("so loose", "gaping")
    assert "gaba" not in (out or "").lower()  # never the romaji leak


def test_gabagaba_in_adj_map_not_sfx_map():
    assert "ガバガバ" in SFX_ADJ_MAP
    assert "ガバガバ" not in SFX_MAP


# ---------------------------------------------------------------------------
# Repeat / small-kana / elongation / ♡ ☆ decoration variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "jp,expected",
    [
        ("ぬちょぬちょ", "Squelch"),   # whole-SFX repeat collapses
        ("たぷんたぷん", "Jiggle"),
        ("ビクンビクン", "Twitch"),
        ("たぷん♡", "Jiggle"),          # heart decoration stripped
        ("ビクン☆", "Twitch"),          # star decoration stripped
        ("ぬちょ♡♡", "Squelch"),
        ("ビクンッ", "Twitch"),         # trailing sokuon stripped
        ("たぷ〜ん", "Jiggle"),          # chōonpu-ish tilde stripped
        ("ちゅぱ♪", "Slurp"),           # music note stripped
    ],
)
def test_decoration_and_repeat_variants(jp, expected):
    assert sfx_pre_translate(jp) == expected


# ---------------------------------------------------------------------------
# NEGATIVE: real dialogue must NOT be bypassed (gate returns None)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "jp",
    [
        "おばさん",        # "lady" — NOT an SFX
        "ありがとう",      # thank you
        "ごめんね",        # sorry
        "だいすき",        # I love you
        "こんにちは",      # hello
        "やめて",          # stop it
        "おはよう",        # good morning
        "これはなに",      # "what is this" (has grammar)
        "彼は来た",        # has kanji
        "Hello",           # latin
        "123",             # digits
    ],
)
def test_dialogue_not_bypassed(jp):
    assert sfx_pre_translate(jp) is None


@pytest.mark.parametrize("jp", ["", "   ", None])
def test_empty_returns_none(jp):
    assert sfx_pre_translate(jp) is None


# ---------------------------------------------------------------------------
# suppress_or_transliterate still benefits from the new families + collapse
# ---------------------------------------------------------------------------

def test_suppress_or_transliterate_new_family():
    assert suppress_or_transliterate("ぬちょ") == "Squelch"
    assert suppress_or_transliterate("ぬちょぬちょ") == "Squelch"


def test_suppress_or_transliterate_gabagaba():
    assert suppress_or_transliterate("ガバガバ") == "so loose"


# ---------------------------------------------------------------------------
# Service integration: SFX boxes bypass the model on BOTH paths
# ---------------------------------------------------------------------------

class _CountingService(VLLMOpenAITranslationService):
    """Counts model calls; any _chat call on a pure-SFX box is a bug."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def _chat(self, messages, max_tokens, temperature=0.0, repetition_penalty=None):  # type: ignore[override]
        self.calls += 1
        return "SHOULD-NOT-BE-CALLED"


def test_translate_single_sfx_never_reaches_llm():
    svc = _CountingService()
    assert asyncio.run(svc.translate_single("ぬちょ")) == "Squelch"
    assert asyncio.run(svc.translate_single("ビクン")) == "Twitch"
    assert asyncio.run(svc.translate_single("ガバガバに")) == "so loose"
    assert svc.calls == 0  # the model was NEVER invoked for any SFX box


def test_page_context_sfx_never_reaches_llm():
    svc = _CountingService()
    out = asyncio.run(svc.translate_page_context(["ぬちょ", "たぷん", "ビクン"]))
    assert out == ["Squelch", "Jiggle", "Twitch"]
    assert svc.calls == 0  # all three SFX bypassed the model


def test_page_context_mixes_sfx_and_dialogue():
    # SFX boxes bypass; a real dialogue box (with grammar) hits the model.
    class _Svc(VLLMOpenAITranslationService):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def _chat(self, messages, max_tokens, temperature=0.0, repetition_penalty=None):  # type: ignore[override]
            self.calls += 1
            return "What is this?"

    svc = _Svc()
    out = asyncio.run(svc.translate_page_context(["ぬちょ", "これはなに？"]))
    assert out[0] == "Squelch"            # bypassed
    assert out[1] == "What is this?"      # translated
    assert svc.calls == 1                  # only the dialogue box hit the model

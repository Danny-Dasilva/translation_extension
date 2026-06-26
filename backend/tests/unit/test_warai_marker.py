"""FIX A: net-slang 笑 ("lol"/"haha") marker normalizer + Korean laugh strip.

The Japanese sentence-final 笑 means "lol"/"haha", NOT the verb "to laugh".
The small model rendered おばさん笑 -> "Laugh, lady!", カレー笑 -> "Curry
laughter", etc. We strip a TRAILING standalone 笑 from the JP before prompting,
translate the remainder, then append ", haha" to the cleaned English. A bare
笑-only bubble -> "haha". Stray Korean ㅋㅋ/ㅎㅎ trailing an English line is also
stripped (all backends, via clean_translation_output).

CRITICAL safety: 笑 that is part of a word (笑顔, 笑う, 爆笑, ...) is NEVER
stripped, and the NON-笑 prompt path stays byte-identical to the v11 template.
"""

import asyncio

import pytest

from app.services.translation_text_utils import (
    clean_translation_output,
    strip_trailing_foreign_laugh,
)
from app.services.vllm_openai_translation_service import (
    VLLMOpenAITranslationService,
    append_haha,
    build_v11_plain_prompt,
    build_v11_context_prompt,
    strip_warai_marker,
)


# ---------------------------------------------------------------------------
# strip_warai_marker — POSITIVE: trailing standalone 笑 is split off
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "jp,body",
    [
        ("おばさん笑", "おばさん"),
        ("デブ笑", "デブ"),
        ("カレー笑", "カレー"),
        ("テスト笑笑", "テスト"),       # 笑笑
        ("すごい笑！", "すごい"),         # punctuation after the marker
        ("やばい笑笑笑", "やばい"),       # 笑笑笑
        ("マジで笑 ", "マジで"),          # trailing whitespace
        ("いいね w笑", "いいね"),         # emphatic w glue before the marker
    ],
)
def test_strips_trailing_warai(jp, body):
    out_body, had = strip_warai_marker(jp)
    assert had is True
    assert out_body == body


@pytest.mark.parametrize("jp", ["笑", "笑笑", "笑笑笑"])
def test_bare_warai_only_has_empty_body(jp):
    body, had = strip_warai_marker(jp)
    assert had is True
    assert body == ""


# ---------------------------------------------------------------------------
# strip_warai_marker — NEGATIVE: 笑 inside a word must NEVER be stripped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "jp",
    [
        "笑顔",        # smile (noun)
        "笑う",        # to laugh (verb)
        "笑った",      # laughed
        "笑える",      # can laugh
        "爆笑",        # burst of laughter (kanji-preceded)
        "微笑",        # smile
        "苦笑",        # bitter smile
        "嘲笑",        # sneer
        "彼は笑顔だ",  # "he is smiling" — 笑 mid-word, not trailing
        "笑顔で",      # 笑 word-initial, more text after
    ],
)
def test_word_internal_warai_not_stripped(jp):
    body, had = strip_warai_marker(jp)
    assert had is False
    assert body == jp


def test_no_warai_passthrough():
    body, had = strip_warai_marker("こんにちは")
    assert had is False
    assert body == "こんにちは"


# ---------------------------------------------------------------------------
# append_haha
# ---------------------------------------------------------------------------

def test_append_haha_basic():
    assert append_haha("Hey lady") == "Hey lady, haha"


def test_append_haha_empty_is_bare_haha():
    assert append_haha("") == "haha"
    assert append_haha("   ") == "haha"


def test_append_haha_does_not_duplicate():
    assert append_haha("That's funny haha") == "That's funny haha"
    assert append_haha("lol") == "lol"


# ---------------------------------------------------------------------------
# strip_trailing_foreign_laugh — Korean ㅋㅋ / ㅎㅎ / Hangul
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Stop it ㅋㅋ", "Stop it"),
        ("That's so funny ㅋㅋㅋ", "That's so funny"),
        ("No way ㅎㅎ", "No way"),
        ("Whatever~ ㅋㅋ", "Whatever~"),
    ],
)
def test_strips_trailing_korean_laugh(raw, expected):
    assert strip_trailing_foreign_laugh(raw) == expected


def test_interior_hangul_not_stripped():
    # A real Korean quote mid-sentence (has Latin AFTER it) is left intact.
    s = "He said 안녕 to me"
    assert strip_trailing_foreign_laugh(s) == s


def test_clean_output_strips_trailing_korean_instead_of_garbling():
    # Without the strip, the Hangul would trip the garble guard -> "...".
    assert clean_translation_output("Stop it ㅋㅋ") == "Stop it"
    # A line that is ONLY Hangul still becomes "..." (no Latin to keep).
    assert clean_translation_output("ㅋㅋㅋ") == "..."


# ---------------------------------------------------------------------------
# v11 prompt byte-identity for the NON-笑 path (the trained template contract)
# ---------------------------------------------------------------------------

def test_plain_prompt_unchanged_for_non_warai():
    # The plain prompt for a normal line must be byte-identical to the template.
    jp = "おはよう"
    assert build_v11_plain_prompt(jp) == (
        "Translate the following Japanese to English. Output only the "
        "translation.\n\nJapanese: おはよう"
    )


def test_context_prompt_unchanged_for_non_warai():
    lines = ["おはよう", "げんき？", "うん"]
    got = build_v11_context_prompt(lines, 1)
    assert got == (
        "Translate the marked line of this manga page from Japanese to English. "
        "Use the page context for speakers, pronouns, and continuity. "
        "Output only the translation of the marked line.\n\n"
        "Page:\n1. おはよう\n2. げんき？\n3. うん\n\n"
        "Translate line 2: げんき？"
    )


# ---------------------------------------------------------------------------
# Service integration: translate_single / translate_page_context wiring
# ---------------------------------------------------------------------------

class _PromptCaptureService(VLLMOpenAITranslationService):
    """Stub that records the prompt and returns a fixed English translation."""

    def __init__(self, fixed_reply: str = "Hey lady"):
        super().__init__()
        self.prompts: list[str] = []
        self.calls = 0
        self.fixed_reply = fixed_reply

    async def _chat(self, messages, max_tokens, temperature=0.0, repetition_penalty=None):  # type: ignore[override]
        self.calls += 1
        self.prompts.append(messages[-1]["content"])
        return self.fixed_reply


def test_translate_single_warai_appends_haha_and_strips_marker():
    svc = _PromptCaptureService("Hey lady")
    out = asyncio.run(svc.translate_single("おばさん笑"))
    assert out == "Hey lady, haha"
    # The prompt the model saw must NOT contain the 笑 marker.
    assert svc.calls == 1
    assert "笑" not in svc.prompts[0]
    assert "おばさん" in svc.prompts[0]


def test_translate_single_bare_warai_skips_model():
    svc = _PromptCaptureService("unused")
    out = asyncio.run(svc.translate_single("笑"))
    assert out == "haha"
    assert svc.calls == 0  # never reached the model


def test_translate_single_non_warai_prompt_byte_identical():
    # The exact prompt for a normal line equals the standalone template build.
    svc = _PromptCaptureService("Good morning")
    asyncio.run(svc.translate_single("おはよう"))
    assert svc.prompts[0] == build_v11_plain_prompt("おはよう")


def test_page_context_warai_per_line():
    svc = _PromptCaptureService("Hey lady")
    lines = ["おばさん笑", "おはよう"]
    out = asyncio.run(svc.translate_page_context(lines))
    assert out[0] == "Hey lady, haha"
    assert out[1] == "Hey lady"  # fixed reply, no marker -> unchanged
    # The 笑-bubble's OWN marked-line prompt (line 1) must have the marker
    # stripped from its target line: "Translate line 1: おばさん" — no 笑 in the
    # target. (Other lines keep it verbatim in their shared context block.)
    marked_prompt = next(p for p in svc.prompts if p.endswith("Translate line 1: おばさん"))
    assert "笑" not in marked_prompt.split("Translate line 1:")[-1]


def test_page_context_non_warai_prompt_identical():
    svc = _PromptCaptureService("X")
    lines = ["おはよう", "げんき"]
    asyncio.run(svc.translate_page_context(lines))
    # Both prompts equal the unmodified context build (byte-identical template).
    assert svc.prompts[0] == build_v11_context_prompt(lines, 0)
    assert svc.prompts[1] == build_v11_context_prompt(lines, 1)

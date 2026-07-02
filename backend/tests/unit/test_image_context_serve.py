"""IMAGE-CONTEXT serve path for the v1 (Qwen3-VL-8B text-SFT) page-context prompt.

v1 is text-trained but measurably exploits a page image supplied at INFERENCE.
The serve path (gated by ``settings.translation_serve_image_context``, default
OFF) sends each marked page-context call as an OpenAI MULTIMODAL user message:

    content = [ {image_url block}, {text block == build_v11_context_prompt(...)} ]

with the IMAGE FIRST so its (expensive) KV sits in the byte-identical prefix
every marked call on a page shares (vLLM multimodal prefix caching pays it once
per page). A one-shot ``warm_page_image`` call pre-warms that shared prefix.

Train/serve-safety contract (the whole reason these tests exist):

  * The TEXT block MUST stay byte-for-byte ``build_v11_context_prompt``'s output
    (drift here is the ~95% chrF++-collapse risk class). This module asserts the
    image path never mutates the text.
  * Flag OFF (no image URL threaded) => plain STRING content == exact prior
    behavior.
  * The warm call's content is the LONGEST prefix safely shared with a real
    marked call (image block + ``V11_PAGE_INSTR``), so it actually pre-warms the
    shared KV.
  * Warm errors are opportunistic and never propagate.

All HTTP is mocked (a capturing ``_chat`` subclass); no live vLLM calls.
"""
from __future__ import annotations

import asyncio

import pytest

from app.config import settings
from app.services.vllm_openai_translation_service import (
    V11_PAGE_INSTR,
    VLLMOpenAITranslationService,
    build_image_text_content,
    build_v11_context_prompt,
)

PAGE = ["お母さん、ただいま", "おかえり", "今日は早かったね"]
IMG_URL = "data:image/jpeg;base64,QUJDREVGRw=="


class _CapturingService(VLLMOpenAITranslationService):
    """Capture the raw ``content`` (str or list) + max_tokens of every _chat call."""

    def __init__(self):
        # Skip network client init from the base __init__.
        self.captured: list = []
        self.captured_max_tokens: list = []
        self._healthy = True

    async def _ensure_healthy(self):
        return None

    async def _chat(self, messages, max_tokens, temperature=0.0, repetition_penalty=None):
        content = messages[-1]["content"]
        self.captured.append(content)
        self.captured_max_tokens.append(max_tokens)
        # Extract the text block so we can echo a deterministic 1:1 marker,
        # whether content is a plain string or a [image, text] list.
        if isinstance(content, list):
            text = next(b["text"] for b in content if b.get("type") == "text")
        else:
            text = content
        if "Translate line " not in text:  # e.g. the warm call (instruction only)
            return ""
        k = text.rsplit("Translate line ", 1)[1].split(":", 1)[0].strip()
        return f"EN[line {k}]"


class _RaisingService(_CapturingService):
    """Every _chat raises — to prove warm errors are swallowed, not propagated."""

    async def _chat(self, messages, max_tokens, temperature=0.0, repetition_penalty=None):
        raise RuntimeError("simulated vLLM failure")


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _text_block(content) -> str:
    assert isinstance(content, list), "expected multimodal list content"
    return next(b["text"] for b in content if b.get("type") == "text")


# ---------------------------------------------------------------------------
# (a) BYTE-IDENTITY: image path leaves the text block == builder output exactly
# ---------------------------------------------------------------------------

def test_image_path_text_block_is_byte_identical_to_builder():
    svc = _CapturingService()
    out = _run(
        svc.translate_page_context_marked(PAGE, [1], "English", page_image_data_url=IMG_URL)
    )
    assert out == ["EN[line 2]"]
    content = svc.captured[0]
    assert isinstance(content, list)
    # Image block FIRST, then text.
    assert content[0] == {"type": "image_url", "image_url": {"url": IMG_URL}}
    assert content[1]["type"] == "text"
    golden = build_v11_context_prompt(PAGE, 1)
    assert content[1]["text"] == golden
    assert content[1]["text"].encode("utf-8") == golden.encode("utf-8")


def test_image_path_content_shape_is_two_block_image_then_text():
    svc = _CapturingService()
    _run(
        svc.translate_page_context_marked(PAGE, [0, 2], "English", page_image_data_url=IMG_URL)
    )
    assert len(svc.captured) == 2
    for content in svc.captured:
        assert isinstance(content, list) and len(content) == 2
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == IMG_URL
        assert content[1]["type"] == "text"


def test_build_image_text_content_helper_shape_and_passthrough():
    text = build_v11_context_prompt(PAGE, 0)
    content = build_image_text_content(IMG_URL, text)
    assert content == [
        {"type": "image_url", "image_url": {"url": IMG_URL}},
        {"type": "text", "text": text},
    ]
    # Text passed through verbatim (never mutated).
    assert content[1]["text"] is text


# ---------------------------------------------------------------------------
# (b) FLAG-OFF default: plain STRING content (no list), exact prior behavior
# ---------------------------------------------------------------------------

def test_default_no_image_builds_plain_string_content():
    svc = _CapturingService()
    out = _run(svc.translate_page_context_marked(PAGE, [1]))
    assert out == ["EN[line 2]"]
    content = svc.captured[0]
    assert isinstance(content, str)
    assert content == build_v11_context_prompt(PAGE, 1)


def test_none_image_url_is_plain_string_content():
    svc = _CapturingService()
    _run(svc.translate_page_context_marked(PAGE, [0], "English", page_image_data_url=None))
    assert isinstance(svc.captured[0], str)


def test_setting_defaults_off():
    from app.config import Settings

    assert Settings().translation_serve_image_context is False


# ---------------------------------------------------------------------------
# (c) WARM-PREFIX construction shares the exact prefix bytes with a real call
# ---------------------------------------------------------------------------

def test_warm_call_shares_image_block_and_instruction_prefix_with_real_call():
    warm = _CapturingService()
    _run(warm.warm_page_image(IMG_URL))
    real = _CapturingService()
    _run(
        real.translate_page_context_marked(PAGE, [1], "English", page_image_data_url=IMG_URL)
    )

    warm_content = warm.captured[0]
    real_content = real.captured[0]

    # Same IMAGE block (byte-identical) => the image KV is shared/cacheable.
    assert warm_content[0] == real_content[0]

    # The warm text is EXACTLY the shared instruction opening...
    assert _text_block(warm_content) == V11_PAGE_INSTR
    # ...and is a true byte-prefix of the real call's text (the page/target lines
    # are the only bytes that diverge), so the shared prefix is really shared.
    real_text = _text_block(real_content)
    assert real_text.startswith(V11_PAGE_INSTR)
    assert real_text.encode("utf-8").startswith(V11_PAGE_INSTR.encode("utf-8"))


def test_warm_call_uses_single_token_budget():
    svc = _CapturingService()
    _run(svc.warm_page_image(IMG_URL))
    assert svc.captured_max_tokens == [1]


def test_warm_call_fabricates_no_page_block():
    svc = _CapturingService()
    _run(svc.warm_page_image(IMG_URL))
    assert "Page:" not in _text_block(svc.captured[0])
    assert "Translate line" not in _text_block(svc.captured[0])


def test_warm_empty_url_is_a_noop():
    svc = _CapturingService()
    _run(svc.warm_page_image(""))
    assert svc.captured == []


# ---------------------------------------------------------------------------
# (d) WARM errors are opportunistic — logged, never propagated
# ---------------------------------------------------------------------------

def test_warm_errors_do_not_propagate():
    svc = _RaisingService()
    # Must NOT raise despite the underlying _chat blowing up.
    result = _run(svc.warm_page_image(IMG_URL))
    assert result is None


# ---------------------------------------------------------------------------
# Cross-check: the plain single-line + plain page-context paths are unaffected
# ---------------------------------------------------------------------------

def test_plain_page_context_still_string_when_flag_semantics_absent():
    # translate_page_context (all-lines-are-targets) never threads an image =>
    # every content stays a plain string (byte-identical to prior behavior).
    svc = _CapturingService()
    _run(svc.translate_page_context(PAGE))
    assert all(isinstance(c, str) for c in svc.captured)

"""Unit tests: WHOLE-PAGE context for v11 page-context translation.

The v11 LoRA was trained to translate ONE marked line while seeing the WHOLE
page (all dialogue lines, in reading order) as the numbered "Page:" context.
At serve time we previously passed only the KEPT lines (post OCR-gate / JP
filter) as context, so dropped/garbled DIALOGUE lines left GAPS in the page the
model saw — breaking the speaker/pronoun continuity the retraining bought.

`translate_page_context_marked(page_lines, target_indices)` must:
  * send the FULL ordered page (every entry of `page_lines`) as the numbered
    context, byte-compatible with `build_v11_context_prompt`,
  * mark the TARGET line at its index within that full page,
  * return ONE translation per `target_indices` entry (1:1, in order),
  * never request a translation for a non-target (context-only) line.
"""
from __future__ import annotations

import asyncio

from app.services.vllm_openai_translation_service import (
    VLLMOpenAITranslationService,
    build_v11_context_prompt,
)


class _CapturingService(VLLMOpenAITranslationService):
    """Capture every prompt sent to _chat; echo the marked line index back."""

    def __init__(self):
        # Skip network client init from the base __init__.
        self.captured = []
        self._healthy = True

    async def _ensure_healthy(self):
        return None

    async def _chat(self, messages, max_tokens, temperature=0.0, repetition_penalty=None):
        content = messages[-1]["content"]
        self.captured.append(content)
        # Return a deterministic marker so we can assert 1:1 mapping.
        # The prompt ends with "Translate line {k}: {jp}".
        tail = content.rsplit("Translate line ", 1)[1]
        k = tail.split(":", 1)[0].strip()
        return f"EN[line {k}]"


def _run(coro):
    # Fresh event loop per call so the shared default loop (used by other
    # async tests in the suite) is never closed/reused out from under them.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_marked_full_page_context_only_targets_returned():
    page = ["右1", "右2", "左1", "左2", "SFX_drop"]
    targets = [0, 1, 2, 3]  # index 4 is a dropped SFX -> context only
    svc = _CapturingService()
    out = _run(svc.translate_page_context_marked(page, targets))

    # One output per target, in order.
    assert len(out) == len(targets)
    # 4 model calls (one per target); the SFX line is never a marked target.
    assert len(svc.captured) == 4
    # Every captured prompt carries the FULL 5-line numbered page as context.
    for prompt in svc.captured:
        assert "1. 右1" in prompt
        assert "5. SFX_drop" in prompt  # dropped line still present as context
        # numbered count == full page length
        assert "\n5. " in prompt

    # The marked line numbers are exactly the targets (1-based), in order.
    marked = [int(p.rsplit("Translate line ", 1)[1].split(":", 1)[0]) for p in svc.captured]
    assert marked == [1, 2, 3, 4]


def test_marked_prompt_is_byte_compatible_with_builder():
    page = ["あ", "い", "う"]
    targets = [1]
    svc = _CapturingService()
    _run(svc.translate_page_context_marked(page, targets))
    assert len(svc.captured) == 1
    # Identical to build_v11_context_prompt(full_page, target_idx).
    assert svc.captured[0] == build_v11_context_prompt(page, 1)


def test_marked_empty_target_line_skips_model():
    page = ["text", "", "more"]
    targets = [0, 1, 2]
    svc = _CapturingService()
    out = _run(svc.translate_page_context_marked(page, targets))
    assert len(out) == 3
    assert out[1] == ""  # blank target -> "" without a model call
    # only 2 model calls (indices 0 and 2)
    assert len(svc.captured) == 2


def test_marked_out_of_order_targets_preserve_target_order():
    page = ["a", "b", "c", "d"]
    targets = [3, 1]  # request d then b
    svc = _CapturingService()
    out = _run(svc.translate_page_context_marked(page, targets))
    assert out == ["EN[line 4]", "EN[line 2]"]


def test_plain_page_context_equiv_when_all_kept():
    # translate_page_context(texts) must be equivalent to _marked with all
    # indices as targets (backwards-compatible whole-page == kept-page).
    page = ["x", "y", "z"]
    svc = _CapturingService()
    out_marked = _run(svc.translate_page_context_marked(page, [0, 1, 2]))
    svc2 = _CapturingService()
    out_plain = _run(svc2.translate_page_context(page))
    assert out_marked == out_plain
    assert svc.captured == svc2.captured

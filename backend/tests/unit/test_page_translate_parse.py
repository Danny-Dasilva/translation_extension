"""Unit tests for robust page-level numbered-block parsing (FIX 1).

The page-level translate path must HOLD (keep intra-page context) instead of
falling back to per-bubble isolation. These tests pin the parser contract:

- Accept ``[N]``-tagged output.
- Accept plain one-line-per-item output (what the v10it fine-tune emits).
- Strip a leading preamble line / trailing chatter and still align by count.
- Best-effort index alignment when the model adds/loses a stray blank line.
- Return None only as a TRUE last resort (count cannot be reconciled).
"""
from __future__ import annotations

from app.services.vllm_openai_translation_service import (
    VLLMOpenAITranslationService as S,
)


def test_tagged_exact():
    raw = "[1] Hello.\n[2] thump thump\n[3] Huh?!"
    assert S._parse_page_output(raw, 3) == ["Hello.", "thump thump", "Huh?!"]


def test_plain_lines_exact():
    raw = "Hello.\nthump thump\nHuh?!"
    assert S._parse_page_output(raw, 3) == ["Hello.", "thump thump", "Huh?!"]


def test_tagged_with_preamble_is_stripped():
    raw = "Here are the translations:\n[1] Hello.\n[2] Bye."
    assert S._parse_page_output(raw, 2) == ["Hello.", "Bye."]


def test_plain_lines_with_preamble_is_stripped():
    # One extra non-translation preamble line; count then matches n.
    raw = "Sure! Here you go:\nHello.\nBye."
    assert S._parse_page_output(raw, 2) == ["Hello.", "Bye."]


def test_tagged_out_of_order_realigns_by_index():
    raw = "[2] Bye.\n[1] Hello.\n[3] Yo."
    assert S._parse_page_output(raw, 3) == ["Hello.", "Bye.", "Yo."]


def test_tagged_missing_one_tag_best_effort_pads():
    # Model emitted only 2 of 3 tags; parser returns a 3-length list (one blank)
    # rather than None so the caller can per-bubble-fill only the gap.
    raw = "[1] Hello.\n[3] Yo."
    out = S._parse_page_output(raw, 3)
    assert out is not None
    assert len(out) == 3
    assert out[0] == "Hello."
    assert out[2] == "Yo."
    assert out[1] == ""  # gap, to be individually filled (never rendered as "...")


def test_blank_lines_between_items_ignored():
    raw = "Hello.\n\nthump thump\n\nHuh?!"
    assert S._parse_page_output(raw, 3) == ["Hello.", "thump thump", "Huh?!"]


def test_unreconcilable_returns_none():
    # Far too few lines and no tags -> genuinely cannot align.
    raw = "just one line"
    assert S._parse_page_output(raw, 5) is None


def test_runaway_repetition_tail_is_stripped():
    # The v10it loop on garbled SFX: real prefix then `||||...`.
    raw = "Hah!\nIf you want to cum, move yourself!\nHgh...||||||||||||||||||"
    out = S._parse_page_output(raw, 3)
    assert out == ["Hah!", "If you want to cum, move yourself!", "Hgh..."]


def test_truncated_tail_salvaged_and_padded():
    # Model emitted 6 of 8 lines then a repetition loop ate the budget; salvage
    # the 6 (page context held) and pad to 8 so the caller fills only the gap.
    raw = (
        "Hah!\n"
        "If you want to cum, move yourself!\n"
        "You're such a coward!!\n"
        "There you go!!\n"
        "Yah...\n"
        "Hgh...||||||||||||||||||||||||||||||||||||||"
    )
    out = S._parse_page_output(raw, 8)
    assert out is not None
    assert len(out) == 8
    assert out[0] == "Hah!"
    assert out[4] == "Yah..."
    assert out[5] == "Hgh..."
    assert out[6] == "" and out[7] == ""  # gap, caller fills individually

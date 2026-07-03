"""ITEM 5: NARRATION-CAPTION 3rd-person conditioning for the v11 page-context
prompt.

Manga carries two box kinds: spoken dialogue bubbles and NARRATION captions
(a narrator's aside, not a character speaking). The box kind is detected
upstream but discarded before the prompt is built, so narration inherits the
dialogue path's first/second-person pressure. Item 5 adds an OPT-IN, no-retrain
serve lever: an in-body 3rd-person directive line inserted AFTER the ``Page:``
block, behind ``settings.translation_render_narration_3rd_person`` (default
FALSE) AND only for a line the caller marks as narration.

Same train/serve-safety contract as the cast anchor (see MEMORY.md
chat-template-mismatch): the directive is an IN-BODY context line, NEVER a
``system`` message. These tests assert the flag-off / non-narration path is
BYTE-IDENTICAL to the trained template, and the flag-on narration path inserts
exactly one directive line between the ``Page:`` block and the ``Translate
line`` directive (nothing else changes).
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.services.vllm_openai_translation_service import (
    NARRATION_3RD_PERSON_DIRECTIVE,
    build_narration_directive_line,
    build_v11_context_prompt,
)


GOLDEN_FLAG_OFF = (
    "Translate the marked line of this manga page from Japanese to English. "
    "Use the page context for speakers, pronouns, and continuity. "
    "Output only the translation of the marked line.\n\n"
    "Page:\n"
    "1. お母さん、ただいま\n"
    "2. おかえり\n"
    "3. 今日は早かったね\n\n"
    "Translate line 2: おかえり"
)
PAGE = ["お母さん、ただいま", "おかえり", "今日は早かったね"]


@pytest.fixture(autouse=True)
def _reset_flags():
    """Restore BOTH flags so test order can't leak state."""
    orig_narr = getattr(settings, "translation_render_narration_3rd_person", False)
    orig_cast = getattr(settings, "translation_cast_anchor", False)
    # Keep the cast flag off so these tests isolate the narration lever.
    settings.translation_cast_anchor = False
    yield
    settings.translation_render_narration_3rd_person = orig_narr
    settings.translation_cast_anchor = orig_cast


# ---------------------------------------------------------------------------
# Default: byte-identical to the trained template
# ---------------------------------------------------------------------------

def test_default_flag_is_false():
    from app.config import Settings

    assert Settings().translation_render_narration_3rd_person is False


def test_flag_off_is_byte_identical_even_for_narration_line():
    # Flag OFF: even a line explicitly marked as narration changes nothing.
    settings.translation_render_narration_3rd_person = False
    got = build_v11_context_prompt(PAGE, 1, is_narration=True)
    assert got == GOLDEN_FLAG_OFF
    assert got.encode("utf-8") == GOLDEN_FLAG_OFF.encode("utf-8")


def test_flag_on_non_narration_line_is_byte_identical():
    # Flag ON but the marked line is a normal dialogue bubble (is_narration
    # False) => still byte-identical; the directive is opt-in PER LINE.
    settings.translation_render_narration_3rd_person = True
    got = build_v11_context_prompt(PAGE, 1, is_narration=False)
    assert got == GOLDEN_FLAG_OFF


def test_default_call_has_no_directive():
    settings.translation_render_narration_3rd_person = True
    got = build_v11_context_prompt(PAGE, 1)  # is_narration defaults False
    assert "narration caption" not in got
    assert got == GOLDEN_FLAG_OFF


# ---------------------------------------------------------------------------
# Flag ON + narration line: exactly one directive, after Page:, before Translate
# ---------------------------------------------------------------------------

def test_flag_on_narration_line_inserts_directive():
    settings.translation_render_narration_3rd_person = True
    got = build_v11_context_prompt(PAGE, 1, is_narration=True)
    directive = build_narration_directive_line()
    expected = (
        "Translate the marked line of this manga page from Japanese to English. "
        "Use the page context for speakers, pronouns, and continuity. "
        "Output only the translation of the marked line.\n\n"
        "Page:\n"
        "1. お母さん、ただいま\n"
        "2. おかえり\n"
        "3. 今日は早かったね\n\n"
        f"{directive}\n\n"
        "Translate line 2: おかえり"
    )
    assert got == expected


def test_directive_sits_between_page_and_translate():
    settings.translation_render_narration_3rd_person = True
    got = build_v11_context_prompt(PAGE, 1, is_narration=True)
    page_end = got.index("今日は早かったね")
    directive_at = got.index(NARRATION_3RD_PERSON_DIRECTIVE)
    translate_at = got.index("Translate line 2:")
    assert page_end < directive_at < translate_at


def test_directive_only_adds_the_directive_block():
    # Diff between narration-on and off must be EXACTLY the inserted directive.
    settings.translation_render_narration_3rd_person = True
    on = build_v11_context_prompt(PAGE, 1, is_narration=True)
    off = build_v11_context_prompt(PAGE, 1, is_narration=False)
    block = f"{build_narration_directive_line()}\n\n"
    assert on.replace(block, "", 1) == off


def test_directive_is_single_line_and_third_person():
    line = build_narration_directive_line()
    assert "\n" not in line
    assert "third person" in line
    assert "narration" in line.lower()


def test_directive_preserves_page_numbering_and_target():
    settings.translation_render_narration_3rd_person = True
    got = build_v11_context_prompt(PAGE, 1, is_narration=True)
    assert "1. お母さん、ただいま" in got
    assert "2. おかえり" in got
    assert got.rstrip().endswith("Translate line 2: おかえり")


def test_no_system_message_surface():
    # This builder only ever returns the single user-message body string; the
    # directive lives in-body (the collapse-safe surface), never a system role.
    settings.translation_render_narration_3rd_person = True
    got = build_v11_context_prompt(PAGE, 1, is_narration=True)
    assert isinstance(got, str)
    assert got.startswith(
        "Translate the marked line of this manga page from Japanese to English."
    )

"""ITEM 4: CAST/ROLE-ANCHOR A/B serve mechanism for the v11 page-context prompt.

The dominant remaining ceiling is the model bucket (mistranslation + pronoun/
gender). Item 4 adds an OPTIONAL, cheap, no-retrain serve-time lever: an in-body
``Cast:`` line inserted BEFORE the ``Page:`` block of the page-context prompt,
behind ``settings.translation_cast_anchor`` (default FALSE).

CRITICAL train/serve-safety contract (see MEMORY.md chat-template-mismatch +
v12 NSFW-oversampling notes):

  * The cast hint MUST be an IN-BODY context line, never a ``system`` message.
    A system message on this page-context path is the ~95% chrF++-collapse risk
    class. These tests assert the flag-off path is BYTE-IDENTICAL to the trained
    template, and the flag-on path inserts exactly one ``Cast:`` line between the
    instruction and the ``Page:`` block (nothing else changes).
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.services.vllm_openai_translation_service import (
    DEFAULT_CAST_ANCHOR,
    build_cast_anchor_line,
    build_v11_context_prompt,
)


# The byte-exact template the v11 LoRA was trained on (flag OFF). If this string
# ever changes, the serve/train contract is broken — this is the golden.
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
def _reset_cast_flag():
    """Always restore the flag so test order can't leak state."""
    orig = getattr(settings, "translation_cast_anchor", False)
    yield
    settings.translation_cast_anchor = orig


# ---------------------------------------------------------------------------
# Flag OFF: byte-identical to the trained template (the safety contract)
# ---------------------------------------------------------------------------

def test_flag_off_is_byte_identical_to_golden():
    settings.translation_cast_anchor = False
    got = build_v11_context_prompt(PAGE, 1)
    assert got == GOLDEN_FLAG_OFF
    assert got.encode("utf-8") == GOLDEN_FLAG_OFF.encode("utf-8")


def test_flag_off_has_no_cast_line():
    settings.translation_cast_anchor = False
    got = build_v11_context_prompt(PAGE, 1)
    assert "Cast:" not in got


def test_flag_off_default_setting_is_false():
    # The shipped default MUST be off (no behavior change without opt-in).
    # We read a freshly-imported attribute, not the per-test fixture value.
    from app.config import Settings

    assert Settings().translation_cast_anchor is False


# ---------------------------------------------------------------------------
# Flag ON: exactly one in-body Cast line, inserted before Page:, nothing else
# ---------------------------------------------------------------------------

def test_flag_on_inserts_cast_line_before_page_block():
    settings.translation_cast_anchor = True
    got = build_v11_context_prompt(PAGE, 1)
    cast_line = build_cast_anchor_line()
    expected = (
        "Translate the marked line of this manga page from Japanese to English. "
        "Use the page context for speakers, pronouns, and continuity. "
        "Output only the translation of the marked line.\n\n"
        f"{cast_line}\n\n"
        "Page:\n"
        "1. お母さん、ただいま\n"
        "2. おかえり\n"
        "3. 今日は早かったね\n\n"
        "Translate line 2: おかえり"
    )
    assert got == expected


def test_flag_on_cast_line_is_in_body_not_system():
    # The Cast hint must live in the single user-message body, between the
    # instruction and Page:. This function only ever returns the user string;
    # there is no system message on this path (that's the collapse-risk class).
    settings.translation_cast_anchor = True
    got = build_v11_context_prompt(PAGE, 1)
    assert got.startswith(
        "Translate the marked line of this manga page from Japanese to English."
    )
    # Cast line sits AFTER the instruction's blank line and BEFORE Page:.
    instr_end = got.index("\n\nCast:")
    page_start = got.index("\n\nPage:")
    assert instr_end < page_start


def test_flag_on_only_adds_the_cast_line():
    # Diff between on and off must be EXACTLY the inserted "Cast: ...\n\n".
    settings.translation_cast_anchor = False
    off = build_v11_context_prompt(PAGE, 1)
    settings.translation_cast_anchor = True
    on = build_v11_context_prompt(PAGE, 1)
    cast_block = f"{build_cast_anchor_line()}\n\n"
    assert on.replace(cast_block, "", 1) == off


def test_flag_on_preserves_page_numbering_and_target():
    settings.translation_cast_anchor = True
    got = build_v11_context_prompt(PAGE, 1)
    assert "1. お母さん、ただいま" in got
    assert "2. おかえり" in got
    assert got.rstrip().endswith("Translate line 2: おかえり")


# ---------------------------------------------------------------------------
# Cast line content: known cast populated, extension point present
# ---------------------------------------------------------------------------

def test_default_cast_anchor_contains_known_mother_role():
    # Yurie is the documented mother. Roles are conservative.
    assert "Yurie" in DEFAULT_CAST_ANCHOR
    assert "mother" in DEFAULT_CAST_ANCHOR.lower()
    assert "she/her" in DEFAULT_CAST_ANCHOR


def test_cast_line_has_cast_prefix_and_no_newline():
    # A single context LINE: prefixed "Cast:" and free of embedded newlines so it
    # cannot accidentally inject extra numbered/Page structure.
    line = build_cast_anchor_line()
    assert line.startswith("Cast:")
    assert "\n" not in line


def test_cast_line_pronoun_tags_present():
    # Pronoun anchoring is the whole point (pronoun_gender bucket): every named
    # role carries an explicit pronoun tag.
    line = build_cast_anchor_line()
    assert "she/her" in line
    assert "he/him" in line


def test_extension_point_marked_in_source_constant():
    # The known cast is small; the full cast is a clearly-marked TODO extension
    # point in the module so a future pass can populate it without guessing the
    # format.
    import app.services.vllm_openai_translation_service as svc

    assert hasattr(svc, "CAST_ANCHOR_EXTENSION_NOTE")
    assert "extend" in svc.CAST_ANCHOR_EXTENSION_NOTE.lower()


# ---------------------------------------------------------------------------
# Plain path is untouched by the cast flag (cast only applies to page context)
# ---------------------------------------------------------------------------

def test_plain_prompt_unaffected_by_cast_flag():
    from app.services.vllm_openai_translation_service import build_v11_plain_prompt

    settings.translation_cast_anchor = True
    on = build_v11_plain_prompt("おかえり")
    settings.translation_cast_anchor = False
    off = build_v11_plain_prompt("おかえり")
    assert on == off
    assert "Cast:" not in on

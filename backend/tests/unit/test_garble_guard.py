"""FIX #6: ASCII garble guard classes + short-utterance normalization."""

import pytest

from app.services.translation_text_utils import (
    _dedup_repeated_phrase,
    _is_garbled,
    clean_translation_output,
)
from app.services.vllm_openai_translation_service import normalize_short_utterance


# --------------------------------------------------------------------------- #
# Garble classes that MUST be caught
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "garbled",
    [
        "See also *po, ka, and ku*",   # dict-gloss
        "It is pronounced like that",  # dict-gloss marker (short)
        "S.h.i.+p",                    # dotted/spaced emphasis w/ `+`
        "KIKIUUUUUUUU",                # single-letter run-on
        "Hello こんにちは",              # existing CJK guard still works
    ],
)
def test_garble_caught(garbled):
    assert _is_garbled(garbled) is True


def test_garble_caught_via_clean_output():
    assert clean_translation_output("S.h.i.+p") == "..."
    assert clean_translation_output("KIKIUUUUUUUU") == "..."


# --------------------------------------------------------------------------- #
# Legit text that must NOT be flagged (false-positive guard)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ok",
    [
        "U.S.A.",                                   # known acronym, dot-sep
        "Hello there",                              # plain phrase
        "No! No!",                                  # short emphatic repeat
        "very very good",                           # short word repeat
        "Nooooo",                                   # 5-run survives
        "She pronounced the word slowly and the whole class listened in silence",  # long (>60)
        "This is a perfectly normal sentence.",
    ],
)
def test_legit_not_flagged(ok):
    assert _is_garbled(ok) is False
    # And clean_translation_output returns the (possibly de-duped) text, not "...".
    assert clean_translation_output(ok) != "..."


# --------------------------------------------------------------------------- #
# _dedup_repeated_phrase
# --------------------------------------------------------------------------- #
def test_dedup_collapses_long_dup():
    assert _dedup_repeated_phrase("in this world, in this world") == "in this world"


def test_dedup_collapses_many_copies():
    assert _dedup_repeated_phrase("go go go") == "go"


def test_dedup_leaves_short_emphasis():
    assert _dedup_repeated_phrase("No! No!") == "No! No!"
    assert _dedup_repeated_phrase("very very good") == "very very good"


# --------------------------------------------------------------------------- #
# normalize_short_utterance
# --------------------------------------------------------------------------- #
def test_normalize_strips_dot_between_kana():
    assert normalize_short_utterance("バ.カ") == "バカ"


def test_normalize_strips_space_between_kana():
    assert normalize_short_utterance("わ け") == "わけ"


def test_normalize_collapses_runaway_kana():
    # Single-char runaway: (.)\1{3,} -> two copies.
    assert normalize_short_utterance("あーーーー") == "あーー"


def test_normalize_leaves_long_sentence_unchanged():
    long_jp = "これは長い文章なので正規化されないはずです"
    assert normalize_short_utterance(long_jp) == long_jp

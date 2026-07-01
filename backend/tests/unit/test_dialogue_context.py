"""Unit tests for dialogue-context candidacy of GATE-DROPPED lines.

When a DIALOGUE line is dropped before translation (OCR-gate / garble), the v11
page-context model still benefits from seeing it in the numbered "Page:" context
(speaker/pronoun continuity) — but a PURE-SFX box must NOT pollute the dialogue
context. ``is_dialogue_context_candidate`` decides whether a dropped line is
real-enough dialogue to keep as CONTEXT-ONLY (never translated/rendered).

Policy (conservative): keep as dialogue context when the line is dialogue-LENGTH
mostly-Japanese text; drop short SFX-ish / glossary-SFX / non-Japanese scrawl.
"""
from __future__ import annotations

from app.utils.ocr_confidence_gate import is_dialogue_context_candidate


def test_long_japanese_dialogue_is_context():
    # A real (if garbled) dialogue-length JP line — keep as context.
    assert is_dialogue_context_candidate("昨日あんな事をしていたのに今日は平気なの") is True


def test_short_sfx_is_not_context():
    # Short onomatopoeia / SFX — not dialogue context.
    assert is_dialogue_context_candidate("ドン") is False
    assert is_dialogue_context_candidate("ビクッ") is False


def test_glossary_sfx_is_not_context():
    # A glossary-matched SFX (handled out-of-band) is not dialogue context.
    assert is_dialogue_context_candidate("ぬちょ") is False


def test_empty_is_not_context():
    assert is_dialogue_context_candidate("") is False
    assert is_dialogue_context_candidate("   ") is False


def test_garble_only_is_not_context():
    # Recognizer breakdown / non-JP scrawl — not context.
    assert is_dialogue_context_candidate("|||::::") is False


def test_medium_clean_japanese_is_context():
    # A plain dialogue-length line passes.
    assert is_dialogue_context_candidate("これは一体どういうことなんですか") is True


def test_short_speaker_reference_is_context():
    # A SHORT line that names a speaker / carries a pronoun is exactly the
    # continuity context the page is for — keep it even though it is short.
    # (IK4 page 5: "お母さんは僕の…" establishes the mother as the subject and
    # must inform the He/She pronoun on the marked line.)
    assert is_dialogue_context_candidate("お母さんは僕の") is True
    assert is_dialogue_context_candidate("私のことが") is True


def test_short_generic_japanese_excluded():
    # A short line with NO speaker reference and below dialogue length is more
    # likely an exclamation / SFX-ish fragment — exclude.
    assert is_dialogue_context_candidate("そうか") is False


def test_low_conf_garble_excluded_even_if_long():
    # A LONG but low-OCR-confidence garbled line is noise, not useful context —
    # exclude when confidence is supplied and low.
    garble = "平速ととのちま何き然然と家族で朝ごんん"
    assert is_dialogue_context_candidate(garble, ocr_confidence=0.49) is False
    # Without confidence info, length+JP-ratio still admit it (back-compat).
    assert is_dialogue_context_candidate(garble) is True

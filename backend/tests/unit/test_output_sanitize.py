"""Tests for the model-artifact output sanitizer (markdown + refusal leaks)."""
from app.services.output_sanitize import (
    strip_model_artifacts,
    strip_romaji_honorifics,
)


def test_markdown_bold_stripped():
    assert strip_model_artifacts("I'm on a **Lumber** ride") == "I'm on a Lumber ride"
    assert strip_model_artifacts("**Cowgirl** position") == "Cowgirl position"


def test_stray_bold_removed():
    assert strip_model_artifacts("a **Sliding Ruler") == "a Sliding Ruler"


def test_refusal_blanked():
    assert strip_model_artifacts("I'm sorry, I'm not sure what you want me to do with this") == ""
    assert strip_model_artifacts("As an AI, I cannot translate this.") == ""
    assert strip_model_artifacts("I'm unable to translate that.") == ""


def test_clean_text_unchanged():
    assert strip_model_artifacts("Whose cock do you want first?") == "Whose cock do you want first?"
    # a single asterisk (SFX-style) is left alone; only ** is markdown.
    assert strip_model_artifacts("*pant*") == "*pant*"


def test_passthrough():
    assert strip_model_artifacts(None) is None
    assert strip_model_artifacts("") == ""


# --------------------------------------------------------------------------- #
# Romaji-honorific leaks.
# --------------------------------------------------------------------------- #
def test_honorific_stripped_from_name():
    assert strip_romaji_honorifics("Yui-chan, look!") == "Yui, look!"
    assert strip_romaji_honorifics("Thanks, No-kun...") == "Thanks, No..."
    assert strip_romaji_honorifics("I told Tanaka-san already.") == "I told Tanaka already."
    assert strip_romaji_honorifics("Yes, Kaede-sama.") == "Yes, Kaede."
    assert strip_romaji_honorifics("Watch out, Rin-senpai!") == "Watch out, Rin!"


def test_honorific_strip_keeps_possessive():
    assert strip_romaji_honorifics("That's Yui-chan's bag.") == "That's Yui's bag."


def test_kinship_mapped_to_english():
    assert strip_romaji_honorifics("Onii-chan, wait!") == "Big brother, wait!"
    assert strip_romaji_honorifics("onee-chan...") == "big sister..."
    assert strip_romaji_honorifics("Stop it, nii-san.") == "Stop it, big brother."


def test_english_words_never_touched():
    # Hyphenated English compounds must survive (honorific suffix list is tight).
    assert strip_romaji_honorifics("I bought a T-shirt.") == "I bought a T-shirt."
    assert strip_romaji_honorifics("my mother-in-law") == "my mother-in-law"
    # An honorific that is merely a word PREFIX (not the whole suffix token).
    assert strip_romaji_honorifics("the old-sanctuary gate") == "the old-sanctuary gate"
    # No hyphenated honorific at all.
    assert strip_romaji_honorifics("Whose cock do you want first?") == "Whose cock do you want first?"


def test_honorific_strip_idempotent():
    once = strip_romaji_honorifics("Onii-chan and Yui-chan")
    assert once == strip_romaji_honorifics(once) == "Big brother and Yui"


def test_honorific_passthrough():
    assert strip_romaji_honorifics(None) is None
    assert strip_romaji_honorifics("") == ""

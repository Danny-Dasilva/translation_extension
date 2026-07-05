"""Tests for honorific preservation (post-edit, discourse-fidelity mitigation).

See app/services/honorific_glossary.py for the design: TIER A (append a
suffix to an already-present bare name) never invents text and is the
default-safe path; TIER B (insert a vocative when the name is fully absent)
is opt-in via ``allow_vocative_insertion`` and covered separately.
"""
from app.services.honorific_glossary import restore_honorifics
from app.services.translation_postedit import postedit_one


# --------------------------------------------------------------------------- #
# TIER A: name present, honorific dropped -> restored.
# --------------------------------------------------------------------------- #
def test_registry_name_honorific_restored():
    # あゆむ is a verified NAME_LOCKS/IKENIE_CAST entry (canonical "Ayumu").
    assert restore_honorifics("Ayumu!", "あゆむくんの") == "Ayumu-kun!"


def test_registry_name_honorific_restored_possessive():
    assert restore_honorifics("Ayumu's", "あゆむくんの") == "Ayumu-kun's"


def test_mechanical_katakana_name_honorific_restored():
    # クリちゃん -> katakana "Kuri" + "-chan"; bench-validated (furube p6/012).
    assert restore_honorifics("And Kuri too.", "クリちゃんも") == "And Kuri-chan too."


def test_mechanical_long_vowel_name_honorific_restored():
    # Real bench case (ikenie4/065): ロナさん -> "Rona-san".
    assert (
        restore_honorifics("Rona doesn't come to this room...", "ロナさんこの部屋に来たりしないのに..")
        == "Rona-san doesn't come to this room..."
    )


def test_case_insensitive_match_preserves_original_casing():
    # Lowercase "ayumu" in the model output is matched case-insensitively;
    # only the suffix is appended, original casing of the match is untouched.
    assert restore_honorifics("turn on ayumu!", "あゆむくんの") == "turn on ayumu-kun!"


# --------------------------------------------------------------------------- #
# No-op: source has no honorific, or name is absent (Tier B off by default).
# --------------------------------------------------------------------------- #
def test_no_source_honorific_is_noop():
    assert restore_honorifics("Yui-chan, look!", "みて") == "Yui-chan, look!"


def test_name_absent_is_noop_by_default():
    # Ikenie5 p017: source addresses "Maki-sama" but the EN dropped it
    # entirely. TIER B (invention) is OFF by default -> no change.
    jp = "そ...それがまーきさまちょうど切れてて明日買いに"
    en = "U-um... I was just about to go buy some tomorrow, since it's running out."
    assert restore_honorifics(en, jp) == en


def test_name_absent_with_vocative_insertion_enabled():
    jp = "そ...それがまーきさまちょうど切れてて明日買いに"
    en = "U-um... I was just about to go buy some tomorrow, since it's running out."
    out = restore_honorifics(en, jp, allow_vocative_insertion=True)
    assert out == f"Maaki-sama, {en}"


def test_generic_kinship_vocative_not_treated_as_name():
    # おばさん (auntie) must never be romanized/treated as a personal name.
    assert restore_honorifics("Old lady", "おばさん") == "Old lady"
    assert restore_honorifics("Old lady", "おばさん", allow_vocative_insertion=True) == "Old lady"


def test_plain_hiragana_without_signal_is_noop():
    # No long-vowel mark / katakana -> conservative reject (precision over
    # recall; avoids false positives on ordinary sentence hiragana).
    assert restore_honorifics("Shinobu... help...", "しのぶさまあ..") == "Shinobu... help..."


def test_passthrough_none_and_empty():
    assert restore_honorifics(None, "あゆむくんの") is None
    assert restore_honorifics("", "あゆむくんの") == ""
    assert restore_honorifics("Ayumu!", None) == "Ayumu!"
    assert restore_honorifics("Ayumu!", "") == "Ayumu!"


# --------------------------------------------------------------------------- #
# Idempotency: never double-append or double-prepend.
# --------------------------------------------------------------------------- #
def test_idempotent_tier_a_no_double_append():
    once = restore_honorifics("Ayumu!", "あゆむくんの")
    assert once == "Ayumu-kun!"
    twice = restore_honorifics(once, "あゆむくんの")
    assert twice == once


def test_idempotent_tier_b_no_double_prepend():
    jp = "そ...それがまーきさまちょうど切れてて明日買いに"
    en = "U-um... I was just about to go buy some tomorrow, since it's running out."
    once = restore_honorifics(en, jp, allow_vocative_insertion=True)
    twice = restore_honorifics(once, jp, allow_vocative_insertion=True)
    assert twice == once


def test_already_honorific_marked_name_untouched():
    # A name that already carries a (different or same) hyphenated honorific
    # is never re-suffixed -- avoids "Ayumu-kun-kun" / "Ayumu-chan-kun".
    assert restore_honorifics("Ayumu-chan!", "あゆむくんの") == "Ayumu-chan!"


# --------------------------------------------------------------------------- #
# Full chain (translation_postedit.postedit_one) composition.
# --------------------------------------------------------------------------- #
def test_restores_honorific_that_the_strip_pass_removed():
    # The model KEEPS the source-confirmed honorific; strip_romaji_honorifics
    # removes it as a generic "romaji leak"; restore_honorifics (source-
    # conditioned, running last) puts it back.
    out = postedit_one("Ayumu-kun!", "あゆむくんの")
    assert out == "Ayumu-kun!"


def test_hallucinated_honorific_not_restored_through_chain():
    # No source honorific at all -> strip removes the leak and restore has no
    # candidate to reinstate (the case strip_romaji_honorifics targets).
    out = postedit_one("Yui-chan, look!", "みて")
    assert out == "Yui, look!"


def test_dropped_honorific_restored_through_full_chain():
    out = postedit_one("And Kuri too.", "クリちゃんも")
    assert out == "And Kuri-chan too."


def test_flag_disabled_master_switch_is_noop():
    import app.config as cfgmod

    prev = cfgmod.settings.postedit_restore_honorifics
    try:
        cfgmod.settings.postedit_restore_honorifics = False
        out = postedit_one("And Kuri too.", "クリちゃんも")
        assert out == "And Kuri too."
    finally:
        cfgmod.settings.postedit_restore_honorifics = prev

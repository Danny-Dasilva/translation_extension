"""Tests for the おばさん->Auntie address-term lock (post-edit)."""
from app.services.address_glossary import lock_address_terms


def test_old_lady_to_auntie_case_preserved():
    assert lock_address_terms("Old lady", "おばさん") == "Auntie"
    assert lock_address_terms("old lady", "おばさん") == "auntie"
    assert lock_address_terms("OLD LADY", "おばさん") == "AUNTIE"


def test_wrong_vocatives_normalized():
    assert lock_address_terms("ma'am?", "おばさん") == "auntie?"
    assert lock_address_terms("Your muscles, sweetie?", "おばさんの膣筋?") == "Your muscles, auntie?"
    assert lock_address_terms("Old woman", "おばちゃん") == "Auntie"


def test_possessive_handled():
    assert lock_address_terms("Old lady's vibrator", "おばさんがバイブ") == "Auntie's vibrator"


def test_grandma_not_touched():
    # おばあさん (grandmother) must NOT trigger; substring おばさん does not match it.
    assert lock_address_terms("old woman", "おばあさん") == "old woman"
    assert lock_address_terms("granny", "お婆さん") == "granny"


def test_requires_source_term():
    # No おばさん in source -> leave "old lady" alone (could be a legit line).
    assert lock_address_terms("the old lady left", "彼女は帰った") == "the old lady left"


def test_bare_noun_use_untouched():
    # Bare "lady"/"woman" as a *noun* (not a direct address) stays put.
    assert lock_address_terms("that woman", "おばさん") == "that woman"
    # Relative-clause noun phrase: descriptive, not vocative -> untouched.
    assert (
        lock_address_terms("the woman who lives next door", "おばさん")
        == "the woman who lives next door"
    )
    assert (
        lock_address_terms("the lady who saw us left", "おばさん")
        == "the lady who saw us left"
    )
    # Mid-sentence noun use (no vocative comma / not "the lady") stays put.
    assert lock_address_terms("I gave the woman a hand", "おばさん") == "I gave the woman a hand"


def test_vocative_terminal_to_auntie():
    # The model's most common wrong vocative: bare "lady"/"woman" in address.
    assert lock_address_terms("What do you want, lady?", "おばさん") == "What do you want, auntie?"
    assert lock_address_terms("Wake up, lady!", "おばさん") == "Wake up, auntie!"
    assert lock_address_terms("Thanks, lady", "おばさん") == "Thanks, auntie"
    assert lock_address_terms("Yes, woman.", "おばちゃん") == "Yes, auntie."
    # Medial vocative (comma on both sides).
    assert lock_address_terms("Well, lady, listen up", "おばさん") == "Well, auntie, listen up"


def test_vocative_initial_to_auntie():
    assert lock_address_terms("Lady, wake up!", "おばさん") == "Auntie, wake up!"
    assert lock_address_terms("Hey. Lady, listen.", "おばさん") == "Hey. Auntie, listen."
    # A capitalised proper noun ("Lady Gaga") is NOT a vocative -> untouched.
    assert lock_address_terms("Lady Gaga is here", "おばさん") == "Lady Gaga is here"


def test_vocative_standalone_to_auntie():
    assert lock_address_terms("Lady!", "おばさん") == "Auntie!"
    assert lock_address_terms("Woman", "おばちゃん") == "Auntie"
    assert lock_address_terms("lady?", "おばさん") == "auntie?"


def test_the_lady_reference_to_auntie():
    # Definite "the lady" reference -> "Auntie" (article consumed). Case follows
    # the matched span, matching the module's case-preservation convention.
    assert lock_address_terms("The lady is worried.", "おばさん") == "Auntie is worried."
    assert lock_address_terms("the lady", "おばさん") == "auntie"


def test_vocative_requires_source_term():
    # No おばさん in source -> a legit vocative line is left alone.
    assert lock_address_terms("What do you want, lady?", "何の用?") == "What do you want, lady?"
    assert lock_address_terms("Lady!", "助けて") == "Lady!"


def test_vocative_idempotent():
    once = lock_address_terms("Wake up, lady!", "おばさん")
    twice = lock_address_terms(once, "おばさん")
    assert once == twice == "Wake up, auntie!"


def test_passthrough():
    assert lock_address_terms(None, "おばさん") is None
    assert lock_address_terms("", "おばさん") == ""
    assert lock_address_terms("Auntie", None) == "Auntie"

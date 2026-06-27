"""Unit tests for the deterministic, seedless gold-set probes added to
``backend.scripts.eval.probes`` for the Ikenie4 MT regression harness.

Probes under test (all seedless, scored on the frozen gold set):
    * reverse_sense   - banned/required EN substrings (negation/sense reversal)
    * pronoun_gender  - referent:she rows fail on he/his/him w/o she/her
    * name_invention  - fail if a banned hallucinated name appears
    * sfx_meta_leak   - fail if a gloss/explainer shape leaks onto the page
    * number_romaji   - fail if a JP number-word is left romanized
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR.parent))

from backend.scripts.eval.probes import (  # noqa: E402
    BANNED_INVENTED_NAMES,
    check_name_invention,
    check_number_romaji,
    check_pronoun_gender,
    check_reverse_sense,
    check_sfx_meta_leak,
    run_probes,
)


# ---------------------------------------------------------------------------
# reverse_sense
# ---------------------------------------------------------------------------


def test_reverse_sense_pass_when_required_present_and_banned_absent() -> None:
    assert check_reverse_sense(
        jp="吸い出せよ",
        en_pred="Make sure you suck it all out, auntie",
        banned_en_substrings=["spit"],
        required_en_substrings=["suck", "out"],
    ) is True


def test_reverse_sense_fail_on_banned_substring() -> None:
    # "Spit it out" is the reversed sense of 吸い出せ (suck OUT).
    assert check_reverse_sense(
        jp="吸い出せよ",
        en_pred="Spit it out.",
        banned_en_substrings=["spit"],
        required_en_substrings=[],
    ) is False


def test_reverse_sense_fail_when_required_missing() -> None:
    assert check_reverse_sense(
        jp="吸い出せよ",
        en_pred="Take it.",
        banned_en_substrings=[],
        required_en_substrings=["suck"],
    ) is False


def test_reverse_sense_required_is_any_not_all() -> None:
    # Any one of the required alternatives suffices.
    assert check_reverse_sense(
        jp="ユリエは果てた",
        en_pred="Yurie came.",
        banned_en_substrings=["passed away", "died"],
        required_en_substrings=["came", "climaxed"],
    ) is True


# ---------------------------------------------------------------------------
# pronoun_gender
# ---------------------------------------------------------------------------


def test_pronoun_gender_she_pass_when_she() -> None:
    assert check_pronoun_gender(
        jp="朝の服のままだ",
        en_pred="She's still wearing her clothes from this morning.",
        referent="she",
    ) is True


def test_pronoun_gender_she_fail_on_he_without_she() -> None:
    assert check_pronoun_gender(
        jp="朝の服のままだ",
        en_pred="He's still in his morning clothes.",
        referent="she",
    ) is False


def test_pronoun_gender_she_pass_when_no_pronoun_at_all() -> None:
    # No gendered pronoun present -> nothing to get wrong -> pass.
    assert check_pronoun_gender(
        jp="朝の服のままだ",
        en_pred="Still wearing the morning clothes.",
        referent="she",
    ) is True


def test_pronoun_gender_he_fail_on_she_without_he() -> None:
    assert check_pronoun_gender(
        jp="彼は走った",
        en_pred="She ran away.",
        referent="he",
    ) is False


def test_pronoun_gender_word_boundary_not_substring() -> None:
    # "the" contains "he" but must NOT trip the he-detector.
    assert check_pronoun_gender(
        jp="朝の服のままだ",
        en_pred="She grabbed the clothes.",
        referent="she",
    ) is True


# ---------------------------------------------------------------------------
# name_invention
# ---------------------------------------------------------------------------


def test_name_invention_fail_on_banned_name() -> None:
    assert check_name_invention(
        jp="ロナさん",
        en_pred="Lona-san pointed at it.",
    ) is False


def test_name_invention_fail_on_torachance() -> None:
    assert check_name_invention(
        jp="次男のトラチャンスだぜ",
        en_pred="The second son, Torachance.",
    ) is False


def test_name_invention_pass_on_clean_output() -> None:
    assert check_name_invention(
        jp="ロナさん",
        en_pred="This is your chance.",
    ) is True


def test_name_invention_word_boundary() -> None:
    # "Aki" must match as a word, not inside "making".
    assert check_name_invention(
        jp="あいかー",
        en_pred="We are making dinner.",
    ) is True


def test_banned_invented_names_contains_known_hallucinations() -> None:
    lowered = {n.lower() for n in BANNED_INVENTED_NAMES}
    for expected in ("lona", "kinomiya", "torachance", "aki", "zuri"):
        assert expected in lowered


# ---------------------------------------------------------------------------
# sfx_meta_leak
# ---------------------------------------------------------------------------


def test_sfx_meta_leak_fail_on_lighter_version_of() -> None:
    assert check_sfx_meta_leak(
        jp="ベロチュー",
        en_pred="Lighter version of *beso* or *ero-ch* sound effects",
    ) is False


def test_sfx_meta_leak_fail_on_yelled_by() -> None:
    assert check_sfx_meta_leak(
        jp="ふぇ?",
        en_pred="Yelled by a crowd when a revelation is made",
    ) is False


def test_sfx_meta_leak_fail_on_sfx_for_a() -> None:
    assert check_sfx_meta_leak(
        jp="どきっ",
        en_pred="SFX for a sudden surprise",
    ) is False


def test_sfx_meta_leak_pass_on_real_translation() -> None:
    assert check_sfx_meta_leak(
        jp="ベロチュー",
        en_pred="French kiss me, auntie!",
    ) is True


# ---------------------------------------------------------------------------
# number_romaji
# ---------------------------------------------------------------------------


def test_number_romaji_fail_on_left_romanized_hyaku() -> None:
    assert check_number_romaji(
        jp="ひゃく!!",
        en_pred="H-hyaku!!",
    ) is False


def test_number_romaji_pass_on_translated_number() -> None:
    assert check_number_romaji(
        jp="ひゃく!!",
        en_pred="One hun...dred!!",
    ) is True


def test_number_romaji_pass_when_no_number_word_in_jp() -> None:
    assert check_number_romaji(
        jp="こんにちは",
        en_pred="ichi sounds fine",  # romaji present but no number word in jp
    ) is True


# ---------------------------------------------------------------------------
# run_probes integration with the new probe_types + per-row config
# ---------------------------------------------------------------------------


def test_run_probes_dispatches_gold_probes_with_row_config() -> None:
    rows = [
        {
            "probe_type": "reverse_sense",
            "jp": "吸い出せよ",
            "en_pred": "Spit it out.",
            "banned_en_substrings": ["spit"],
            "required_en_substrings": ["suck"],
        },  # fail
        {
            "probe_type": "reverse_sense",
            "jp": "吸い出せよ",
            "en_pred": "Suck it all out.",
            "banned_en_substrings": ["spit"],
            "required_en_substrings": ["suck"],
        },  # pass
        {
            "probe_type": "pronoun_gender",
            "jp": "朝の服のままだ",
            "en_pred": "He's still in his clothes.",
            "referent": "she",
        },  # fail
        {
            "probe_type": "name_invention",
            "jp": "ロナさん",
            "en_pred": "Lona-san waved.",
        },  # fail
        {
            "probe_type": "sfx_meta_leak",
            "jp": "ふぇ?",
            "en_pred": "Yelled by a crowd when a revelation is made",
        },  # fail
        {
            "probe_type": "number_romaji",
            "jp": "ひゃく!!",
            "en_pred": "H-hyaku!!",
        },  # fail
    ]
    result = run_probes(rows)
    c = result.per_probe_counts
    assert c["reverse_sense"]["n"] == 2
    assert c["reverse_sense"]["pass"] == 1
    assert c["reverse_sense"]["fail"] == 1
    assert c["pronoun_gender"] == {"n": 1, "pass": 0, "fail": 1}
    assert c["name_invention"] == {"n": 1, "pass": 0, "fail": 1}
    assert c["sfx_meta_leak"] == {"n": 1, "pass": 0, "fail": 1}
    assert c["number_romaji"] == {"n": 1, "pass": 0, "fail": 1}

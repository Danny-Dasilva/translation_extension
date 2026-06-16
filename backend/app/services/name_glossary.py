"""Character-name canonicalizer (P0-3) — pure post-processing.

The recurring cast in an automated manga translation gets corrupted
inconsistently across pages (the model translates each bubble in isolation
at temp 0, so a name with no surrounding context drifts spelling page to
page). This module repairs the *output* text after translation.

IMPORTANT — DO NOT add a glossary to the model prompt. The v11 translation
model is acutely train/serve format-sensitive (a documented ~95% quality
collapse when the serving format drifts from training). All correction here
is a pure post-pass on the model's EN output, optionally conditioned on the
source ``jp`` for the cases that need disambiguation.

Design constraints:
  * Conservative: only act on the *specific* known corruptions below; never
    invent a name and never rewrite a substring of an unrelated word.
  * Word-boundary regexes so "Kana" fixes the standalone name but leaves
    "Kanazawa" / "banana" intact.
  * Idempotent: running twice == running once.
  * Source-conditioned fixes (``jp`` present) only fire when the relevant
    kanji is in the source, so a sentence that legitimately mentions "milk"
    with no 愛菜 is left alone.

To extend per-series, add an entry to ``CANONICAL_NAMES`` (EN-only word
variants) or to ``SOURCE_CONDITIONED`` (fixes that need the jp source).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Pattern


# --------------------------------------------------------------------------- #
# 1. EN-only canonicalisation table (no source needed)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CanonicalName:
    """A canonical EN name and the corrupted variants to normalise to it.

    ``variants`` are matched as whole words (``\\b...\\b``), case-sensitively,
    so they only hit capitalised name tokens — not lowercase common words.
    The canonical form itself is intentionally NOT listed as a variant
    (matching it would be a no-op but wastes a substitution).
    """

    canonical: str
    variants: tuple[str, ...]


# Per-series cast. Extend here for new series / new characters.
CANONICAL_NAMES: tuple[CanonicalName, ...] = (
    # 加奈子 (Kanako) — seen as Kana / Kanan / Kana-ji / Kanakao.
    # NOTE: "Kana-ji" is handled by allowing an optional "-ji"/"-chan" name
    # suffix after the variant stem (see _build_variant_pattern).
    CanonicalName(
        canonical="Kanako",
        variants=("Kanakao", "Kanan", "Kana"),
    ),
    # 康介 (Kousuke) — three mis-spellings collapse to one.
    CanonicalName(
        canonical="Kousuke",
        variants=("Yousuke", "Kansuke", "Kosuke"),
    ),
)

# Junk romaji suffixes the model hallucinates onto a name stem and that are
# NOT real honorifics (e.g. "Kana-ji"). These are stripped along with the
# correction so "Kana-ji" -> "Kanako" (not "Kanako-ji"). Genuine honorifics
# the model chose to keep ("-san", "-chan", "-kun") are preserved, so
# "Yousuke-san" -> "Kousuke-san".
_JUNK_SUFFIXES = ("ji",)


def _build_variant_pattern(variant: str) -> Pattern[str]:
    """Whole-word regex for one variant, consuming a trailing *junk* suffix.

    Matches e.g. ``Kana``, ``Kana-ji`` but NOT ``Kanazawa`` (no boundary
    after "Kana") nor ``banana`` (different casing / boundary). A real
    honorific like ``-san`` is left in place (it is not in _JUNK_SUFFIXES),
    so the variant stem is replaced and the honorific survives.
    """
    junk = "|".join(_JUNK_SUFFIXES)
    # \b<variant>(?:-(?:junk))?\b — the optional junk suffix, when present,
    # is consumed so it disappears in the replacement.
    return re.compile(rf"\b{re.escape(variant)}(?:-(?:{junk}))?\b")


# Pre-compile (canonical, compiled-variant) pairs once at import time.
# Variants are ordered longest-first within each name so e.g. "Kanakao" is
# tried before "Kana" (avoids leaving a "ko"/"n" tail behind).
_COMPILED_EN: list[tuple[str, Pattern[str]]] = []
for _entry in CANONICAL_NAMES:
    for _variant in sorted(_entry.variants, key=len, reverse=True):
        _COMPILED_EN.append((_entry.canonical, _build_variant_pattern(_variant)))


# --------------------------------------------------------------------------- #
# 2. Source-conditioned fixes (need the jp source to disambiguate)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SourceConditionedFix:
    """A correction that only fires when the jp source contains a trigger.

    ``en_pattern`` is searched against the EN output; if it matches AND the
    (separator-normalised) jp source contains ANY string in ``jp_triggers``,
    every match is replaced with ``replacement``. This keeps us from "fixing"
    a bubble that legitimately contains the trigger word (e.g. real milk, or
    a person actually named Chen) when the source kanji is absent.

    Multiple triggers cover OCR variation — e.g. お姉ちゃん is frequently
    OCR'd as the all-hiragana おねえちゃん when the 姉 kanji is mis-read.
    """

    name: str
    jp_triggers: tuple[str, ...]
    en_pattern: Pattern[str]
    replacement: str


SOURCE_CONDITIONED: tuple[SourceConditionedFix, ...] = (
    # 愛菜 (Aina, a little girl) once mistranslated as "the milk" / "milk".
    # Only correct when 愛菜 is actually in the source bubble.
    SourceConditionedFix(
        name="Aina",
        jp_triggers=("愛菜",),
        # Eat an optional leading article so "the milk" -> "Aina" reads cleanly.
        en_pattern=re.compile(r"\b(?:the\s+)?milk\b", re.IGNORECASE),
        replacement="Aina",
    ),
    # お姉ちゃん / お姉さん (onee-chan = big sis) phoneticised to "Chen",
    # breaking the sister reveal. Only correct when the source is onee-chan.
    # Includes the hiragana OCR form おねえちゃん / おねえさん (the 姉 kanji is
    # commonly mis-read as ねえ, as on bench page 069).
    SourceConditionedFix(
        name="Sis",
        jp_triggers=("お姉ちゃん", "お姉さん", "おねえちゃん", "おねえさん"),
        en_pattern=re.compile(r"\bChen\b"),
        replacement="Sis",
    ),
)


def _normalize_jp(jp: str) -> str:
    """Strip OCR separators (dots, spaces, punctuation) for trigger matching.

    Bench page 069 OCRs お姉ちゃん as ``お...ねえ..ちゃん...?`` — collapsing the
    separators yields ``おねえちゃん`` which matches the hiragana trigger.
    """
    return re.sub(r"[.。、・…\s!?！？]+", "", jp)


# --------------------------------------------------------------------------- #
# 3. Public API
# --------------------------------------------------------------------------- #
def canonicalize_names(en: str, jp: Optional[str] = None) -> str:
    """Repair known character-name corruptions in a translated EN bubble.

    Args:
        en: the model's English output for one bubble.
        jp: the OCR'd Japanese source for the same bubble, if available.
            Enables source-conditioned fixes (愛菜→Aina, お姉ちゃん→Sis).

    Returns:
        The EN text with known name corruptions normalised. Conservative and
        idempotent: text with no known corruption is returned unchanged.
    """
    if not en:
        return en

    out = en

    # Pass 1: EN-only word-boundary variant normalisation.
    for canonical, pattern in _COMPILED_EN:
        out = pattern.sub(canonical, out)

    # Pass 2: source-conditioned fixes (require a jp trigger to be present).
    if jp:
        jp_norm = _normalize_jp(jp)
        for fix in SOURCE_CONDITIONED:
            if any(t in jp_norm for t in fix.jp_triggers):
                out = fix.en_pattern.sub(fix.replacement, out)

    return out

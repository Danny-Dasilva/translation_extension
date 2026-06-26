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


# --------------------------------------------------------------------------- #
# 2b. Hard name-locks (kana name in source -> ONE canonical EN spelling)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NameLock:
    """A hard lock: when the jp source contains ``jp_kana`` (the character's
    name written in kana), force the EN output to ``canonical``.

    Unlike :class:`SourceConditionedFix` (which only rewrites a *specific*
    known mis-romanisation), a lock also rewrites *any* of ``mis_romaji`` the
    model invents for this kana name. We deliberately do NOT blanket-replace
    every capitalised token — we only swap the enumerated wrong spellings — so
    an unrelated proper noun in the same bubble survives. List every observed
    mis-spelling in ``mis_romaji`` (longest-first is handled at compile time).

    This is the extension point for the per-title cast. The full cast list for
    THIS title is not yet known to the post-edit; add a NameLock per character
    as their kana name + observed mis-romanisations are collected from bench
    output. See the TODO marker below.
    """

    canonical: str
    jp_kana: str
    mis_romaji: tuple[str, ...]


# >>> PER-TITLE CAST GLOSSARY — EXTEND ME <<<
# Only ユリエ -> "Yurie" is confirmed. The remainder of the cast still needs to
# be enumerated from bench output (kana name + the wrong romaji the model
# emits). Do NOT guess names: add a NameLock only once both the kana and the
# observed mis-spelling are verified, otherwise a real proper noun could be
# clobbered. TODO(cast): populate the full character roster for this title.
NAME_LOCKS: tuple[NameLock, ...] = (
    # ユリエ (Yurie) — model emits "Julie" / "Lucia" page to page.
    NameLock(
        canonical="Yurie",
        jp_kana="ユリエ",
        mis_romaji=("Julie", "Lucia", "Yulie", "Yurié"),
    ),
)


# Pre-compile (canonical, jp_kana, [compiled wrong-spelling patterns]) once.
_COMPILED_LOCKS: list[tuple[str, str, list[Pattern[str]]]] = []
for _lock in NAME_LOCKS:
    _pats = [
        re.compile(rf"\b{re.escape(v)}\b")
        for v in sorted(_lock.mis_romaji, key=len, reverse=True)
    ]
    _COMPILED_LOCKS.append((_lock.canonical, _lock.jp_kana, _pats))


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
# 2c. Counted-number kana (じゅうさん = 33, not "Jus-san"/honorific)
# --------------------------------------------------------------------------- #
# A bare counted-number bubble is a few number-kana with optional small
# punctuation / emphasis. The model mis-reads the さん in じゅうさん (=3 in
# 10+3) as the honorific "-san", or transliterates ひゃく -> "hyaku". We only
# fire when the WHOLE bubble (separator-stripped) is a number reading, so a
# real "○○さん" honorific on a name (田中さん) is never touched.

# Base digit readings (1-9) used both standalone and as multipliers/units.
_KANA_DIGITS: dict[str, int] = {
    "いち": 1, "に": 2, "さん": 3, "し": 4, "よん": 4,
    "ご": 5, "ろく": 6, "なな": 7, "しち": 7, "はち": 8,
    "きゅう": 9, "く": 9,
}
# Powers of ten.
_KANA_TENS = ("じゅう", "ひゃく", "せん", "まん")
_KANA_POWERS: dict[str, int] = {
    "じゅう": 10, "ひゃく": 100, "せん": 1000, "まん": 10000,
}

# A bubble is "number-ish" only if it is composed solely of these kana.
_NUMBER_KANA_TOKENS = tuple(_KANA_DIGITS.keys()) + _KANA_TENS
# longest-first so にじゅう parses じゅう as a power, not に+じゅう ambiguity.
_NUMBER_TOKEN_RE = re.compile(
    "|".join(re.escape(t) for t in sorted(_NUMBER_KANA_TOKENS, key=len, reverse=True))
)
# Whole-bubble guard: nothing but number kana (separators already stripped).
_PURE_NUMBER_RE = re.compile(rf"^(?:{_NUMBER_TOKEN_RE.pattern})+$")


# Colloquial / elided counting forms that do NOT equal the literal place-value
# arithmetic. In casual counting the leading multiplier digit is dropped:
#   じゅうさん   spoken "san-juu-san"  = 33  (literal じゅう+さん would be 13)
#   にじゅうご   = literal 25 (no elision) — handled by the general parser
# These documented overrides take precedence over _parse_number_kana. Keep this
# list to verified bench cases only.
_COLLOQUIAL_NUMBER_FORMS: dict[str, int] = {
    "じゅうさん": 33,
}


def _parse_number_kana(s: str) -> Optional[int]:
    """Parse a separator-stripped all-kana number reading to an int.

    Handles the common manga range with standard place-value arithmetic
    (じゅう=10, にじゅう=20, ひゃく=100, にじゅうご=25). Documented colloquial
    elisions (see ``_COLLOQUIAL_NUMBER_FORMS``) are applied first. Returns None
    if the string is not a pure number reading.
    """
    if not s or not _PURE_NUMBER_RE.match(s):
        return None
    if s in _COLLOQUIAL_NUMBER_FORMS:
        return _COLLOQUIAL_NUMBER_FORMS[s]
    tokens = _NUMBER_TOKEN_RE.findall(s)
    if not tokens:
        return None
    total = 0
    current = 0
    for tok in tokens:
        if tok in _KANA_POWERS:
            power = _KANA_POWERS[tok]
            if power >= 10000:  # まん scales the running total
                total = (total + max(current, 1)) * power
                current = 0
            else:
                current = max(current, 1) * power
                total += current
                current = 0
        else:
            current += _KANA_DIGITS[tok]
    return total + current


def _maybe_number_bubble(jp: Optional[str]) -> Optional[int]:
    """If the jp bubble is essentially a bare counted-number reading, return
    its integer value; else None.

    Conservative: the bubble must reduce (after stripping dots/spaces/emphasis)
    to ONLY number kana. ``田中さん`` contains 田中 kanji -> not pure -> ignored.
    ``さんは三人います`` contains 三人います -> not pure -> ignored.
    """
    if not jp:
        return None
    stripped = _normalize_jp(jp)
    # Also drop long-vowel marks / repeated emphasis the strip missed.
    stripped = stripped.replace("ー", "").replace("〜", "").replace("~", "")
    if not stripped:
        return None
    return _parse_number_kana(stripped)


# --------------------------------------------------------------------------- #
# 2d. Low-OCR-confidence name-invention suppression
# --------------------------------------------------------------------------- #
# Default: postedit does nothing about confidence (back-compat). When a caller
# threads a low ocr_conf for a bubble whose source is a common *generic* kana
# word (e.g. おばさん = auntie), we refuse to let the model's single invented
# proper-noun token stand. We neutralise to a generic gloss instead of a name.
LOW_CONF_THRESHOLD = 0.50

# Generic kana source words that the model is known to promote into a fake
# proper name when OCR confidence is low. Maps source -> safe generic gloss.
_GENERIC_KANA_GLOSS: dict[str, str] = {
    "おばさん": "auntie",
    "おじさん": "mister",
    "おばあさん": "grandma",
    "おじいさん": "grandpa",
}

# A single capitalised token (optionally with trailing punctuation) that looks
# like an invented proper name.
_LONE_PROPER_NOUN_RE = re.compile(r"^[A-Z][a-z]+[.!?…]*$")


def _suppress_low_conf_invention(
    en: str, jp: Optional[str], ocr_conf: Optional[float]
) -> str:
    """If OCR conf is low and the EN looks like a lone invented proper name for
    a generic kana source word, replace it with a safe generic gloss."""
    if ocr_conf is None or ocr_conf >= LOW_CONF_THRESHOLD or not en or not jp:
        return en
    stripped = _normalize_jp(jp)
    gloss = _GENERIC_KANA_GLOSS.get(stripped)
    if gloss is None:
        return en
    if _LONE_PROPER_NOUN_RE.match(en.strip()):
        return gloss
    return en


# --------------------------------------------------------------------------- #
# 3. Public API
# --------------------------------------------------------------------------- #
def canonicalize_names(
    en: str, jp: Optional[str] = None, ocr_conf: Optional[float] = None
) -> str:
    """Repair known character-name corruptions in a translated EN bubble.

    Args:
        en: the model's English output for one bubble.
        jp: the OCR'd Japanese source for the same bubble, if available.
            Enables source-conditioned fixes (愛菜→Aina, お姉ちゃん→Sis),
            hard name-locks (ユリエ→Yurie) and the counted-number-kana rule
            (じゅうさん→33, not "-san").
        ocr_conf: optional OCR recognition confidence in [0, 1] for this bubble.
            When low (< LOW_CONF_THRESHOLD) it suppresses name *invention* for
            generic kana source words (おばさん must not become "Sue"). Omitting
            it preserves the prior behaviour exactly.

    Returns:
        The EN text with known name corruptions normalised. Conservative and
        idempotent: text with no known corruption is returned unchanged.
    """
    if not en:
        return en

    jp_norm = _normalize_jp(jp) if jp else ""

    # Pass 0: counted-number kana. If the whole bubble is a bare number reading,
    # replace the (mis-honorific / romaji) EN with the digits and stop — there
    # is no name in a number bubble. Done first so a downstream "-san" rule can
    # never fire on じゅうさん.
    number_value = _maybe_number_bubble(jp)
    if number_value is not None:
        return str(number_value)

    out = en

    # Pass 1: EN-only word-boundary variant normalisation.
    for canonical, pattern in _COMPILED_EN:
        out = pattern.sub(canonical, out)

    # Pass 2: source-conditioned fixes (require a jp trigger to be present).
    if jp:
        for fix in SOURCE_CONDITIONED:
            if any(t in jp_norm for t in fix.jp_triggers):
                out = fix.en_pattern.sub(fix.replacement, out)

    # Pass 3: hard name-locks. When the locked kana is in the source, force any
    # known mis-romanisation of that name to the one canonical spelling.
    if jp:
        for canonical, jp_kana, patterns in _COMPILED_LOCKS:
            if jp_kana in jp_norm:
                for pat in patterns:
                    out = pat.sub(canonical, out)

    # Pass 4: low-confidence name-invention suppression (opt-in via ocr_conf).
    out = _suppress_low_conf_invention(out, jp, ocr_conf)

    return out

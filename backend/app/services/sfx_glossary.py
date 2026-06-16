"""SFX onomatopoeia post-processing for manga translation (Fix P2-2).

PURE POST-PROCESSING on translation output text. The v11 translation model is
acutely train/serve format-sensitive (documented ~95% chrF++ collapse on
format drift), so the model PROMPT must NOT be touched. Instead we clean the
*output* text here, using the source Japanese (``jp``) where helpful.

Three QA problems addressed:

1. **Meta-description leak (WORST).** The model sometimes emits a description
   of the sound instead of a translation, and it gets typeset onto the page::

       "SFX for a grunt of surprise, or any other short grunt call."
       "SFX for a big, sudden revelation, like a revelation that has ..."
       "SFX for a quick, forceful movement, like a quick splash cover"

   These are DETECTED (:func:`is_sfx_meta_description`) and SUPPRESSED, then
   replaced with a short transliteration of the JP SFX when available, else
   ``"..."`` (:func:`suppress_or_transliterate`).

2. **Mistranslated onomatopoeia.** A short katakana SFX gets turned into the
   wrong common English noun (パシッ -> "splash" should be a smack; ヌチュ ->
   "Munch" should be a wet "Squelch"). A conservative :data:`SFX_MAP` corrects
   these *only* when the JP is a short katakana SFX and the English is a single
   clearly-wrong common noun.

Safety: real dialogue is never rewritten. Mapping/suppression only fires on
short, punctuation-heavy, or SFX-shaped lines, or when the meta-description
pattern matches.

Wiring (orchestrator applies this at merge time — do NOT edit translate.py
here): in ``app/routers/translate.py``, immediately after a bubble's
``translation_en`` is produced::

    from app.services.sfx_glossary import clean_sfx_output
    bubble.translation_en = clean_sfx_output(bubble.translation_en, bubble.ocr_jp)
"""

from __future__ import annotations

import re
from typing import Optional

__all__ = [
    "clean_sfx_output",
    "is_sfx_meta_description",
    "suppress_or_transliterate",
    "SFX_MAP",
]


# ---------------------------------------------------------------------------
# 1. Meta-description leak detection
# ---------------------------------------------------------------------------

# Patterns that indicate the model described the sound instead of translating
# it. These are intentionally specific so genuine dialogue containing "sound"
# or "represents" is NOT caught (see negative tests).
_META_PATTERNS: tuple[re.Pattern[str], ...] = (
    # The signature giveaway: a line that literally talks ABOUT an SFX.
    re.compile(r"\bSFX for\b", re.IGNORECASE),
    re.compile(r"\bonomatopoeia\b", re.IGNORECASE),
    # "represents a/the ..." at the START of the line = describing, not speaking.
    re.compile(r"^\s*represents\s+(?:a|an|the)\b", re.IGNORECASE),
    # "(the) sound of a ..." describing a sound as a noun phrase, line-initial.
    re.compile(r"^\s*(?:the\s+)?sound of\b", re.IGNORECASE),
    # "a <short modifier> sound" used as the WHOLE line (a label, not a clause
    # inside dialogue). Anchored to start + must be a short, noun-phrase line.
    re.compile(r"^\s*(?:a|an)\b[^.!?]{0,30}\bsound\b", re.IGNORECASE),
    # "grunt of surprise" / "grunt ... call" descriptive phrasing.
    re.compile(r"\bgrunt of surprise\b", re.IGNORECASE),
)


def is_sfx_meta_description(en: Optional[str]) -> bool:
    """Return True if ``en`` is a meta-description of a sound, not a translation.

    Conservative by design: clauses like "I love the sound of your voice" or
    "This necklace represents our love" must return False, because the
    descriptive patterns are anchored to the start of the line (a label) and
    the start-anchored noun-phrase forms require the line to *be* the phrase.
    """
    if not en:
        return False
    text = en.strip()
    if not text:
        return False
    return any(p.search(text) for p in _META_PATTERNS)


# ---------------------------------------------------------------------------
# 2. SFX glossary + transliteration
# ---------------------------------------------------------------------------

# Common manga onomatopoeia -> conventional English comic SFX. Keys are the
# bare katakana/kana SFX (small kana / sokuon / chōonpu stripped before lookup).
SFX_MAP: dict[str, str] = {
    "パシ": "Smack",     # パシッ — a slap / smack (was mistranslated "splash")
    "ヌチュ": "Squelch",  # wet squelch (was mistranslated "Munch")
    "ニチャ": "Squelch",  # sticky/wet
    "グチュ": "Squelch",
    "ドン": "Boom",      # heavy impact
    "ドカ": "Boom",
    "ドス": "Thud",
    "バン": "Bang",
    "ガシャン": "Crash",
    "ガシャ": "Crash",
    "ガチャ": "Clack",
    "ザアザア": "Shhh",   # heavy rain / water
    "ザア": "Shhh",
    "ザン": "Slash",     # cutting / slashing sound
    "ざん": "Slash",     # ざんっ (hiragana) — a cutting/slashing sound (was "plunging")
    "ゴボ": "Glug",      # underwater / bubbling (ゴボッ)
    "ドキドキ": "Thump thump",
    "ドキ": "Thump",
    "ゴゴゴ": "Rrrrumble",
    "バキ": "Crack",
    "ボキ": "Crack",
    "ピシ": "Crack",
    "ズドン": "Boom",
    "ガッ": "Grab",
}


# Katakana -> romaji table (Hepburn-ish, ASCII only). Covers the SFX domain;
# falls through to the raw char for anything unmapped (then we ASCII-filter).
_KATAKANA_ROMAJI: dict[str, str] = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ヲ": "wo", "ン": "n",
    "ヴ": "vu",
}

# Small (sutegana) kana used as digraphs (ャ ュ ョ) and as standalone small.
_SMALL_KANA_COMBINE = {"ャ": "ya", "ュ": "yu", "ョ": "yo", "ェ": "e", "ァ": "a", "ィ": "i", "ゥ": "u", "ォ": "o"}

# Chōonpu, small tsu (sokuon) and small kana to strip when normalising a SFX
# key for SFX_MAP lookup.
_NORMALISE_STRIP = "ーッｯっ"


def _to_romaji(jp: str) -> str:
    """Best-effort ASCII transliteration of a short katakana/kana SFX."""
    out: list[str] = []
    prev_base = ""
    for ch in jp:
        if ch in ("ー",):  # chōonpu: lengthen previous vowel (drop = ascii-safe)
            continue
        if ch in ("ッ", "ｯ", "っ"):  # sokuon: gemination — keep it light, skip
            continue
        if ch in _SMALL_KANA_COMBINE:
            # combine with previous consonant: shi+yu -> shu, etc. Light touch:
            base = _SMALL_KANA_COMBINE[ch]
            if prev_base.endswith("i") and base.startswith("y"):
                out[-1] = prev_base[:-1] + base[1:]
                prev_base = out[-1]
            else:
                out.append(base)
                prev_base = base
            continue
        rom = _KATAKANA_ROMAJI.get(ch)
        if rom:
            out.append(rom)
            prev_base = rom
        # unmapped (hiragana / punctuation): skipped so result stays ASCII
    word = "".join(out)
    if not word:
        return ""
    return word.capitalize()


def _normalise_sfx_key(jp: str) -> str:
    """Strip elongation / sokuon / trailing small kana for SFX_MAP lookup."""
    key = "".join(c for c in jp if c not in _NORMALISE_STRIP)
    return key


def suppress_or_transliterate(jp: Optional[str]) -> str:
    """Produce a short replacement for a suppressed meta-description.

    Prefers a mapped comic SFX, falls back to a romaji transliteration of the
    JP SFX, and finally ``"..."`` when nothing usable is available.
    """
    if not jp:
        return "..."
    stripped = jp.strip()
    if not stripped:
        return "..."
    key = _normalise_sfx_key(stripped)
    if key in SFX_MAP:
        return SFX_MAP[key]
    # progressive prefix match (e.g. パシ from パシッ already handled by strip,
    # but ザアザア vs ザア etc.)
    for k, v in SFX_MAP.items():
        if key.startswith(k):
            return v
    rom = _to_romaji(stripped)
    if rom:
        return rom
    return "..."


# ---------------------------------------------------------------------------
# 3. Heuristics for "is this line SFX-shaped?" (conservative dialogue guard)
# ---------------------------------------------------------------------------

_KATAKANA_RE = re.compile(r"[゠-ヿ]")
_KANA_KANJI_RE = re.compile(r"[぀-ヿ一-鿿]")
_HIRAGANA_RE = re.compile(r"[぀-ゟ]")


def _jp_is_short_katakana_sfx(jp: Optional[str]) -> bool:
    """True when ``jp`` looks like a short katakana onomatopoeia.

    Used to gate SFX_MAP correction so we never touch real dialogue (which is
    longer and contains hiragana / kanji).
    """
    if not jp:
        return False
    s = jp.strip()
    if not s:
        return False
    # mostly katakana, short, no hiragana-driven grammar
    core = [c for c in s if _KANA_KANJI_RE.match(c)]
    if not core or len(core) > 6:
        return False
    if _HIRAGANA_RE.search(s):
        return False
    return bool(_KATAKANA_RE.search(s))


def _jp_is_exact_mapped_sfx(jp: Optional[str]) -> bool:
    """True when ``jp`` (any kana) normalises to an EXACT :data:`SFX_MAP` key.

    This is the only path by which a *hiragana* SFX (e.g. ざんっ) is corrected.
    It is deliberately exact-match-only (no prefix guessing) so that ordinary
    hiragana dialogue is never caught.
    """
    if not jp:
        return False
    s = jp.strip()
    if not s or len(s) > 6:
        return False
    return _normalise_sfx_key(s) in SFX_MAP


def _en_is_sfx_shaped(en: str) -> bool:
    """True when an English line is short / punctuation-heavy enough to be SFX.

    Examples that qualify: "*Splash*", "splash", "Munch", "plunging".
    A multi-word sentence with real dialogue structure does NOT qualify.
    """
    core = en.strip().strip("*").strip()
    if not core:
        return False
    words = re.findall(r"[A-Za-z']+", core)
    # SFX rendered as 1-2 short words, possibly wrapped in * or punctuation.
    return 0 < len(words) <= 2


# ---------------------------------------------------------------------------
# 4. Public entry point
# ---------------------------------------------------------------------------


def clean_sfx_output(en: Optional[str], jp: Optional[str] = None) -> Optional[str]:
    """Clean a single bubble's translation output.

    Order of operations:
      1. Pass through empty / None unchanged.
      2. If the English is a META-DESCRIPTION leak -> suppress + transliterate.
      3. Else, if JP is a short katakana SFX and English is a short
         (SFX-shaped) clearly-wrong common noun -> apply :data:`SFX_MAP`.
      4. Else leave the text untouched (real dialogue is preserved).
    """
    if en is None:
        return None
    if not en.strip():
        return en

    # (2) Meta-description leak — the highest-priority, page-breaking case.
    if is_sfx_meta_description(en):
        return suppress_or_transliterate(jp)

    # (3) Conservative SFX correction. Fires when the English is short /
    # SFX-shaped AND the JP is either (a) a short katakana SFX, or (b) any kana
    # SFX that EXACTLY matches a glossary key (the only route for hiragana SFX
    # such as ざんっ, kept exact-match-only so real dialogue is never touched).
    jp_is_sfx = _jp_is_short_katakana_sfx(jp) or _jp_is_exact_mapped_sfx(jp)
    if jp_is_sfx and _en_is_sfx_shaped(en):
        key = _normalise_sfx_key(jp.strip())  # type: ignore[union-attr]
        mapped = SFX_MAP.get(key)
        if mapped is None:
            for k, v in SFX_MAP.items():
                if key.startswith(k):
                    mapped = v
                    break
        if mapped is not None:
            # Don't churn an already-correct SFX: if the English already reads
            # as the mapped word (ignoring case / *...* / punctuation), keep it.
            existing = re.sub(r"[^A-Za-z]+", " ", en).strip().lower()
            if existing == mapped.lower():
                return en
            return mapped

    # (4) Leave real dialogue (and already-good SFX) untouched.
    return en

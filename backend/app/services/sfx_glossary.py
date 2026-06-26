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
    "sfx_pre_translate",
    "SFX_MAP",
    "SFX_ADJ_MAP",
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
    # --- wet / squelch family (NSFW). ぬちょ->"menace" etc. were real misses. ---
    # (ニチャ / グチュ katakana forms already mapped above; add hiragana forms.)
    "ぬちょ": "Squelch", "ヌチョ": "Squelch",
    "にちゃ": "Squelch",
    "ぐちゅ": "Squelch",
    "じゅぷ": "Squelch", "ジュプ": "Squelch",
    "じゃぽ": "Squelch", "ジャポ": "Squelch",
    "ねちょ": "Squelch", "ネチョ": "Squelch",
    "びちゃ": "Squelch", "ビチャ": "Squelch",
    "ずちゅ": "Squelch", "ズチュ": "Squelch",
    # --- slurp / suck family ---
    "ちゅぱ": "Slurp", "チュパ": "Slurp",
    "じゅぽ": "Slurp", "ジュポ": "Slurp",
    "れろ": "Lick", "レロ": "Lick",
    "べろ": "Lick", "ベロ": "Lick",
    "ちゅう": "Suck", "チュウ": "Suck",
    "ぢゅる": "Slurp", "ヂュル": "Slurp",
    # --- jiggle family ---
    "たぷん": "Jiggle", "タプン": "Jiggle",  # たぷん->"Maybe" was a real miss
    "ぷるん": "Jiggle", "プルン": "Jiggle",
    "ぶるん": "Jiggle", "ブルン": "Jiggle",
    "たゆん": "Jiggle", "タユン": "Jiggle",
    "ぼいん": "Boing", "ボイン": "Boing",
    # --- twitch / throb family ---
    "ビクン": "Twitch", "びくん": "Twitch",  # ビクン->"Twinkle" was a real miss
    "ピクン": "Twitch", "ぴくん": "Twitch",
    # びくっ / ビクッ normalise (sokuon stripped) to びく / ビク, so map the bases.
    "びく": "Twitch", "ビク": "Twitch",
    "ドクン": "Throb", "どくん": "Throb",
    "ズキン": "Throb", "ずきん": "Throb",
    # --- thrust family ---
    "ずぶ": "Thrust", "ズブ": "Thrust",
    "ぬぷ": "Thrust", "ヌプ": "Thrust",
    "ぐぽ": "Thrust", "グポ": "Thrust",
}

# Adjectival NSFW onomatopoeia: NOT a transliteration. ガバガバに -> "so loose"
# was rendered "Gabagabani" (raw romaji leak). These map a normalised JP key to
# a short English adjective/phrase (rendered verbatim, no romaji fallback).
SFX_ADJ_MAP: dict[str, str] = {
    "ガバガバ": "so loose",
    "がばがば": "so loose",
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

# Decorative / emphatic marks manga SFX trail with: hearts, stars, music notes,
# tildes, ellipses, sentence punctuation, full/half spaces. Stripped before key
# lookup so たぷん♡ and ビクン☆ normalise to their base SFX.
_DECOR_STRIP_RE = re.compile(r"[♡♥❤☆★※♪♫〜~!！?？.。、,…・\s　　]+")


def _collapse_sfx_repeat(key: str) -> str:
    """Collapse a fully-repeated SFX (ぬちょぬちょ -> ぬちょ; ビクンビクン -> ビクン).

    Only collapses when the WHOLE key is an integer number of copies of a unit,
    so it never mangles a distinct two-part SFX. Single units pass through.
    """
    n = len(key)
    if n < 2:
        return key
    for unit_len in range(1, n // 2 + 1):
        if n % unit_len:
            continue
        unit = key[:unit_len]
        if unit * (n // unit_len) == key:
            return unit
    return key


def _to_romaji(jp: str) -> str:
    """Best-effort ASCII transliteration of a short katakana/kana SFX."""
    out: list[str] = []
    prev_base = ""
    for ch in jp:
        if "ぁ" <= ch <= "ゟ":  # hiragana block -> katakana (so kana SFX romanise)
            ch = chr(ord(ch) + 0x60)
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
    """Strip elongation / sokuon / trailing small kana for SFX_MAP lookup.

    Also strips decorative/emphatic trailing marks (♡ ☆ ♪ ～ ! ? … etc.) so a
    SFX written as たぷん♡ or ビクン☆ normalises to its base key. Whole-SFX
    repeats are NOT collapsed here (ドキドキ vs ドキ map to different values);
    repeat-collapse is a separate fallback in the pre-LLM gate.
    """
    jp = _DECOR_STRIP_RE.sub("", jp)
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
    # Adjectival NSFW terms (ガバガバ -> "so loose") take precedence and are
    # NEVER romaji-transliterated.
    if key in SFX_ADJ_MAP:
        return SFX_ADJ_MAP[key]
    if _collapse_sfx_repeat(key) in SFX_ADJ_MAP:
        return SFX_ADJ_MAP[_collapse_sfx_repeat(key)]
    if key in SFX_MAP:
        return SFX_MAP[key]
    # Collapse a whole-SFX repeat (ぬちょぬちょ -> ぬちょ) and retry exact match
    # before falling back to prefix/romaji.
    collapsed = _collapse_sfx_repeat(key)
    if collapsed != key and collapsed in SFX_MAP:
        return SFX_MAP[collapsed]
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


# Dialogue tells: a real spoken line almost always has a personal pronoun or a
# sentence-terminal mark. A bare descriptive word-list ("angerous, grumble,
# rumble rumbling") has neither — that's the model describing the sound.
_PRONOUN_RE = re.compile(
    r"\b(i|i'm|i'll|i've|i'd|you|you're|me|my|mine|your|yours|we|we're|us|our|"
    r"he|he's|she|she's|him|her|his|they|them|their|it|it's|that's|this)\b",
    re.IGNORECASE,
)
_TERMINAL_PUNCT_RE = re.compile(r"[.!?]")


def _jp_is_kana_sfx_source(jp: Optional[str]) -> bool:
    """True when ``jp`` looks like a kana onomatopoeia SOURCE (katakana OR
    hiragana), used only in conjunction with :func:`_en_is_descriptive_leak`.

    Pure kana (no kanji/latin/digits), short (<=5 base kana), and carrying an
    SFX marker — a sokuon (っ/ッ), chōonpu (ー), a repeated kana, or katakana.
    Plain short hiragana words (ありがとう, ごめんね) carry no marker -> False, so
    real dialogue is never caught.
    """
    if not jp:
        return False
    s = jp.strip()
    if not s or re.search(r"[一-鿿A-Za-z0-9]", s):
        return False
    base = [c for c in s if _KANA_KANJI_RE.match(c)]
    if not base or len(base) > 5:
        return False
    if any(m in s for m in ("っ", "ッ", "ー")):
        return True
    if _KATAKANA_RE.search(s):
        return True
    return any(base[i] == base[i + 1] for i in range(len(base) - 1))


def _en_is_descriptive_leak(en: str) -> bool:
    """True when ``en`` is a bare multi-word description of a sound rather than
    a spoken line: >=3 words, NO personal pronoun, and NO sentence-terminal
    punctuation. Catches comma word-lists like "angerous, grumble, rumble
    rumbling". Only ever used gated behind a kana-SFX source, so the few real
    pronoun-less, punctuation-less fragments are still protected by that gate.
    """
    core = en.strip()
    words = re.findall(r"[A-Za-z']+", core)
    if len(words) < 3:
        return False
    if _TERMINAL_PUNCT_RE.search(core) or _PRONOUN_RE.search(core):
        return False
    return True


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

    # (2b) Word-list / descriptive leak that doesn't match the "SFX for ..."
    # patterns (e.g. "angerous, grumble, rumble rumbling" for ぶぶっ). Gated on
    # the JP being a kana SFX source AND the English having no dialogue tells.
    if _jp_is_kana_sfx_source(jp) and _en_is_descriptive_leak(en):
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


# ---------------------------------------------------------------------------
# 5. PRE-LLM gate — bypass the model entirely for glossary-matched SFX
# ---------------------------------------------------------------------------


def _jp_matches_sfx_glossary(jp: str) -> Optional[str]:
    """Return the mapped English for a JP box that is a KNOWN glossary SFX/term.

    Returns None when the JP is not a confident glossary match (so the caller
    falls through to the LLM). Tries, in order, on the normalised key and on a
    whole-repeat-collapsed key:
      1. SFX_ADJ_MAP  (adjectival: ガバガバ -> "so loose")
      2. SFX_MAP      (impact + wet/jiggle/twitch/etc onomatopoeia)
    Exact-key only — NO prefix/romaji guessing here, so ordinary kana dialogue
    that merely starts like a SFX is never bypassed.
    """
    key = _normalise_sfx_key(jp.strip())
    if not key:
        return None
    collapsed = _collapse_sfx_repeat(key)
    for k in (key, collapsed):
        if k in SFX_ADJ_MAP:
            return SFX_ADJ_MAP[k]
    for k in (key, collapsed):
        if k in SFX_MAP:
            return SFX_MAP[k]
    return None


# A trailing grammatical particle (に / と / で / の / が / は / を / って) that
# can follow an adjectival SFX term: ガバガバに -> "so loose". Stripped only for
# the ADJ-term match, never for the onomatopoeia map.
_TRAILING_PARTICLE_RE = re.compile(r"(?:に|と|で|の|が|は|を|って|だ|です)$")


def sfx_pre_translate(jp: Optional[str]) -> Optional[str]:
    """PRE-LLM gate: short-circuit a pure-SFX box to its glossary English.

    Returns the English string to render when ``jp`` is confidently a known
    SFX / adjectival onomatopoeia (so the box NEVER reaches the LLM); returns
    None when the box should be translated normally.

    Gating (conservative, in priority order):
      1. Adjectival term (ガバガバ / ガバガバに -> "so loose") — whole-box exact
         key after stripping ONE trailing grammatical particle.
      2. WHOLE-BOX exact onomatopoeia key (ぬちょ, たぷん, ビクン, ちゅぱ, ...):
         the normalised / repeat-collapsed key EXACTLY equals a SFX_MAP key.
         Glossary keys are non-lexical onomatopoeia, so a whole-box exact match
         is safe — ordinary hiragana dialogue never equals one in full.
      3. SFX-SHAPED katakana / marker-bearing kana box that also exact-matches.

    Real dialogue (longer, with grammar/kanji, or only a PREFIX match) returns
    None and is translated by the model as before.
    """
    if not jp:
        return None
    s = jp.strip()
    if not s:
        return None
    # Reject anything with kanji / latin / digits outright — not a pure SFX box.
    if re.search(r"[一-鿿A-Za-z0-9]", s):
        return None

    # (1) Adjectival term, tolerating one trailing particle (ガバガバに).
    adj_key = _normalise_sfx_key(s)
    adj_key = _TRAILING_PARTICLE_RE.sub("", adj_key)
    if adj_key in SFX_ADJ_MAP:
        return SFX_ADJ_MAP[adj_key]
    if _collapse_sfx_repeat(adj_key) in SFX_ADJ_MAP:
        return SFX_ADJ_MAP[_collapse_sfx_repeat(adj_key)]

    # (2)+(3) Whole-box exact onomatopoeia match (no prefix/romaji guessing).
    return _jp_matches_sfx_glossary(s)

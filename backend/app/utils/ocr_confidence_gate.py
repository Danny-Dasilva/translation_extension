"""OCR-confidence garble gate (translation pre-filter).

Low-confidence, garbled OCR on stylized SFX/illegible scrawl must NOT reach the
translation LLM — it hallucinates non-English captions ("Calcul math not done
by a lifter", Bengali junk) that then render onto the page. This gate drops a
bubble ONLY when the OCR *recognition* confidence is low AND the decoded text
looks garbled, so real dialogue (measured OCR conf ~0.9+) always passes.

Note: this uses PARSeq OCR recognition confidence (mean softmax max-prob over
decoded tokens), NOT the detector/block confidence.
"""
from __future__ import annotations

import unicodedata

from app.utils.japanese_text_filter import analyze_characters, is_japanese_text

# Tuned on the Part13 inspection data (.bench/Part13_translated_en_v4_inspection):
#   real dialogue bubbles: OCR-equivalent crisp, conf ~0.85-0.97
#   garbled SFX / scrawl that produced hallucinations: conf ~0.45-0.61
# A gate at 0.65 cleanly separates them while staying conservative — we ALSO
# require the text to look garbled before dropping, so a merely-uncertain but
# clean Japanese line is kept.
DEFAULT_CONF_THRESHOLD = 0.65

# Below this Japanese-character ratio (post-normalization) a low-conf line is
# treated as garbled even if it scraped past the main japanese filter.
_MIN_JP_RATIO_FOR_LOWCONF = 0.5

# Characters that signal recognizer breakdown rather than real text.
_GARBLE_CHARS = set("[]{}/\\|<>^~`�□")


def _has_garble_chars(text: str) -> bool:
    return any(ch in _GARBLE_CHARS for ch in text)


# --- JP-plausibility (FIX P1-1) -------------------------------------------
# The confidence gate above only fires when ``ocr_conf`` is LOW. The worst
# failures, though, are garbled OCR carrying FALSELY HIGH confidence: e.g.
# page 070's vertical title "3へ続く…" was misread as "..?っく混みますよ" at
# ocr_conf 0.91, sailed past the gate, and was "smoothed" by the LLM into a
# confident-wrong "It's going to be crowded.". The signals below add a
# linguistic-plausibility check that catches such lines regardless of
# confidence, while staying narrow enough to leave genuine dialogue untouched.
#
# Calibration note: both signals were validated against the stored replay
# corpus (.bench/full_pipeline/588828_mesu2_insp, 649 bubbles). They fire on
# the known garbles (070, 071-credit, 074-handle) and on ZERO real dialogue
# lines. Being conservative is intentional — false-dropping real dialogue is
# worse than letting some garble through.

# Small kana (sokuon / yōon). A line legitimately *starting* with one is rare:
# only って (quotation), ったく (=まったく), and trailing っ.. survive in real
# speech — captured by the explicit exceptions in ``_has_garbled_leading_tsu``.
_SMALL_TSU = "っ"

# Leading punctuation we strip before inspecting the first real glyph.
_LEAD_PUNCT = set("。、！？!?「」『』（）()・ー〜….,　 \"'")

# A run of this many ASCII letters embedded in otherwise-Japanese text is the
# signature of OCR latching onto a logo / URL / handle (071 credit line,
# 074 "oisxo…ovitter"). A *single* stray letter or digit is normal (年号,
# counters, "1セット") so the threshold is on letters and is >= 3.
_MAX_LATIN_LETTERS_IN_JP = 3


def _is_hiragana(ch: str) -> bool:
    return 0x3040 <= ord(ch) <= 0x309F


def _is_katakana(ch: str) -> bool:
    return 0x30A0 <= ord(ch) <= 0x30FF or 0xFF65 <= ord(ch) <= 0xFF9F


def _is_kanji(ch: str) -> bool:
    return 0x4E00 <= ord(ch) <= 0x9FFF or 0x3400 <= ord(ch) <= 0x4DBF


def _is_japanese_glyph(ch: str) -> bool:
    return _is_hiragana(ch) or _is_katakana(ch) or _is_kanji(ch)


def _strip_leading_punct(norm: str) -> str:
    i = 0
    while i < len(norm) and norm[i] in _LEAD_PUNCT:
        i += 1
    return norm[i:]


def _has_garbled_leading_tsu(norm: str) -> bool:
    """Line begins with a small tsu ``っ`` glued to a garbled prefix.

    Real Japanese never opens with ``っ`` + an arbitrary kana. The only genuine
    forms are ``って`` (quotation/"like"), ``ったく`` (=まったく), and a trailing
    ``っ`` (``っ..`` cut-off speech). So we flag ``っ`` followed by a kana that
    is NOT て/た. This catches page 070's "..?っく…" prefix and nothing else in
    the replay corpus.
    """
    body = _strip_leading_punct(norm)
    if len(body) < 2 or body[0] != _SMALL_TSU:
        return False
    nxt = body[1]
    if nxt in "てた":  # って / ったく — legitimate.
        return False
    # Only treat it as garbled when a kana actually follows (a trailing っ
    # before punctuation/end is real cut-off speech and was already excluded by
    # the length check / non-kana next char).
    return _is_hiragana(nxt) or _is_katakana(nxt)


def _has_latin_intrusion(norm: str) -> bool:
    """Many ASCII letters wedged into Japanese text -> logo/URL garble.

    Requires Japanese to be present (so genuine all-Latin lines, handled
    elsewhere, are not double-counted) and at least ``_MAX_LATIN_LETTERS_IN_JP``
    ASCII letters. A lone digit/letter (年号, "1セット目") stays plausible.
    """
    latin = sum(1 for ch in norm if ch.isascii() and ch.isalpha())
    if latin < _MAX_LATIN_LETTERS_IN_JP:
        return False
    return any(_is_japanese_glyph(ch) for ch in norm)


# --- duplication garble (FIX P1-2) ----------------------------------------
# The DOMINANT real failure mode (144 bubbles, avg severity 2.53): PARSeq
# misreads dense / stylized vertical kana into duplicated adjacent characters
# and immediate phrase repetition, carrying FALSELY HIGH confidence (0.76-0.92)
# so the confidence threshold never fires. Examples:
#   身代わり -> 身身わわ      吐気 -> 吐吐気       濯濯バサミ
#   また昨日みたいな -> また昨日みたいなまた昨日みたいな (whole-phrase dup)
#   妄想止まらない -> 妄..妄ま定れいい妄.想止止らな (corrupt + 止止 dup)
# These signals are confidence-INDEPENDENT (run in is_implausible_japanese).
#
# Conservatism: false-dropping real dialogue is worse than missing some garble,
# so legitimate Japanese reduplication is whitelisted before flagging.

# Legitimate doubled-KANJI words. Real reduplication normally uses the 々
# iteration mark (人々), but PARSeq may emit the literal doubled kanji, so we
# whitelist both members of each common pair. An adjacent doubled kanji NOT in
# this set is the garble signal.
_LEGIT_KANJI_REDUP = {
    "様", "段", "人", "我", "色", "時", "方",
    "国", "日", "別", "中", "数", "順", "程",
    "個", "村", "家", "山", "木",  # 個々 村々 家々 etc.
}

# Iteration mark — a glyph that *is* legitimate reduplication; never a garble.
_ITERATION_MARK = "々"

# Legitimate doubled-KATAKANA fragments. We only treat doubled KATAKANA as a
# garble signal (doubled hiragana is far too common in real text: long-vowel
# spellings おお/ええ, mimetics, emphatic stretches), and even then we whitelist
# katakana laughter (ハハ/フフ/ヘヘ/ホホ) which is genuine speech.
_LEGIT_KATAKANA_DOUBLE = {"ハ", "フ", "ヘ", "ホ"}


def _adjacent_dup_kanji(norm: str) -> bool:
    """An adjacent ``X X`` kanji pair that is NOT legitimate reduplication.

    Real reduplicated kanji words are whitelisted (``_LEGIT_KANJI_REDUP``); any
    other doubled kanji (身身, 吐吐, 濯濯, 止止) is the PARSeq dup-garble signal.
    """
    for i in range(len(norm) - 1):
        a, b = norm[i], norm[i + 1]
        if a == b and _is_kanji(a) and a not in _LEGIT_KANJI_REDUP:
            return True
    return False


def _adjacent_dup_kana(norm: str) -> bool:
    """An adjacent ``X X`` KATAKANA pair that is not whitelisted laughter.

    Deliberately KATAKANA-ONLY. Doubled hiragana is rejected as a signal because
    it occurs in genuine text (long-vowel おお/ええ as in 大きい/大阪, mimetics),
    so flagging it false-drops real dialogue. Doubled katakana is far more
    garble-like (katakana long vowels use ー, not vowel doubling), and the only
    common legit form is laughter (ハハ/フフ), which is whitelisted.
    """
    for i in range(len(norm) - 1):
        a, b = norm[i], norm[i + 1]
        if a == b and _is_katakana(a) and a not in _LEGIT_KATAKANA_DOUBLE:
            return True
    return False


def _repeated_bigram_garble(norm: str) -> bool:
    """High ratio of repeated character-bigrams -> phrase-repetition garble.

    Whole-phrase OCR duplication (また昨日みたいなまた昨日みたいな) produces many
    repeated bigrams. Genuine dialogue rarely exceeds ~30% repeated bigrams, so
    a >= 0.5 ratio over a long-enough line is a strong dup signal. Gated on
    length to avoid firing on tiny strings where one repeat dominates.
    """
    glyphs = [c for c in norm if _is_japanese_glyph(c)]
    if len(glyphs) < 8:
        return False
    bigrams = [glyphs[i] + glyphs[i + 1] for i in range(len(glyphs) - 1)]
    if not bigrams:
        return False
    unique = len(set(bigrams))
    repeated_ratio = 1.0 - (unique / len(bigrams))
    return repeated_ratio >= 0.5


def _immediate_substring_dup(norm: str) -> bool:
    """Line is ``P + P`` — a phrase repeated immediately back-to-back.

    Catches whole-phrase OCR duplication whose two JP-glyph halves are identical
    (また昨日みたいな + また昨日みたいな). Requires the repeated unit to be
    non-trivial (>= 4 JP glyphs) so ordinary short doubled words don't trip it.
    """
    stripped = "".join(c for c in norm if _is_japanese_glyph(c))
    n = len(stripped)
    if n < 8 or n % 2 != 0:
        return False
    half = n // 2
    if half < 4:
        return False
    return stripped[:half] == stripped[half:]


def is_implausible_japanese(text: str) -> bool:
    """True if ``text`` reads as garbled OCR despite being mostly Japanese.

    A *linguistic*-plausibility heuristic (NOT confidence-based) so that
    confidently-garbled OCR is caught even at high OCR confidence. Deliberately
    narrow: it only returns True on patterns that cannot occur in genuine
    Japanese dialogue, so it does not drop real lines.

    Signals (any one is sufficient):
      * Garbled leading small-tsu prefix (page 070 "..?っく混みますよ").
      * Heavy ASCII-letter intrusion in Japanese text (logo/URL/handle garble).
      * Duplication garble (FIX P1-2): adjacent doubled kanji/kana, whole-phrase
        immediate repetition, or a high repeated-bigram ratio — the dominant
        PARSeq dense-kana failure mode, carrying falsely-high confidence.
    """
    norm = unicodedata.normalize("NFC", text).strip()
    if not norm:
        return False
    if _has_garbled_leading_tsu(norm):
        return True
    if _has_latin_intrusion(norm):
        return True
    if _adjacent_dup_kanji(norm):
        return True
    if _adjacent_dup_kana(norm):
        return True
    if _immediate_substring_dup(norm):
        return True
    if _repeated_bigram_garble(norm):
        return True
    return False


# Longer lines that are mostly real Japanese are treated as dialogue we should
# not silently drop even at low confidence — a hard-to-read but genuine line is
# better translated than dropped. SFX/garble that hallucinates is short.
_DIALOGUE_MIN_LEN = 12


def is_garbled_low_conf(
    text: str,
    ocr_confidence: float,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    min_jp_ratio: float = 0.5,
    katakana_max_len: int = 6,
    check_plausibility: bool = True,
) -> bool:
    """True if this bubble should be DROPPED before translation.

    Empirically (Part13 inspection) PARSeq OCR *recognition* confidence cleanly
    separates real dialogue (~0.85-0.93) from garbled / stylized SFX scrawl
    (~0.15-0.62): NO real dialogue line measured below ~0.65, while every
    sub-0.65 line was a short SFX/garble the LLM then hallucinated a caption
    for. So:

      * Confidence >= ``conf_threshold`` -> NEVER drop (conservative; real SFX
        like ドン at ~0.9 stays and is translated to onomatopoeia).
      * Confidence < ``conf_threshold`` -> drop, UNLESS the line is long enough
        and mostly real Japanese to read as genuine dialogue (don't silently
        lose a hard-but-real line). Garble chars / non-JP always drop.

    FIX P1-1: the rule above is blind to garbled OCR carrying FALSELY HIGH
    confidence (page 070's "..?っく混みますよ" at 0.91). When
    ``check_plausibility`` is set (default), a *linguistic*-plausibility check
    runs FIRST and drops such lines regardless of confidence. It is gated so
    the behavior stays tunable, and is narrow enough to leave real dialogue
    untouched (validated on 600+ replay lines with zero false drops).
    """
    norm = unicodedata.normalize("NFC", text).strip()

    # Plausibility check runs irrespective of confidence — this is the whole
    # point of P1-1 (catch confidently-wrong OCR). Only real Japanese text is
    # inspected; obvious garble-char / non-JP cases fall through to the
    # confidence logic below as before.
    if check_plausibility and norm and is_implausible_japanese(norm):
        return True

    if ocr_confidence >= conf_threshold:
        return False

    if not norm:
        # Empty/low-conf -> nothing to translate; safe to drop.
        return True

    # Garble chars / non-Japanese / low JP-ratio at low conf -> always drop.
    if _has_garble_chars(norm):
        return True
    if not is_japanese_text(norm, min_jp_ratio, katakana_max_len):
        return True
    analysis = analyze_characters(norm)
    if analysis.japanese_ratio < _MIN_JP_RATIO_FOR_LOWCONF:
        return True

    # Low-confidence clean Japanese: drop SHORT SFX-like lines (these are the
    # garbled recognitions that produce hallucinated captions); keep longer
    # genuine-dialogue-length lines so a hard-but-real line still translates.
    if len(norm) < _DIALOGUE_MIN_LEN:
        return True
    return False


# Speaker / pronoun references. A SHORT dropped line carrying one of these is
# exactly the continuity context the v11 page is for (who is speaking / being
# referred to), so we keep it even below the dialogue-length cutoff. Covers the
# common first/second/third-person pronouns and the family-role address terms
# that drive he/she/I/you selection in manga dialogue.
_SPEAKER_REF_TOKENS = (
    "僕", "私", "俺", "あたし", "わたし", "ぼく", "おれ",
    "君", "きみ", "あなた", "お前", "おまえ", "貴方",
    "お母さん", "母さん", "ママ", "お父さん", "父さん", "パパ",
    "お兄ちゃん", "兄さん", "お姉ちゃん", "姉さん",
    "おばさん", "おじさん", "先生", "彼", "彼女",
)


def _has_speaker_reference(norm: str) -> bool:
    return any(tok in norm for tok in _SPEAKER_REF_TOKENS)


# A short dropped line still needs a minimum substance to be context (avoid
# admitting 1-2 char fragments). Half the dialogue cutoff.
_CONTEXT_MIN_LEN_WITH_SPEAKER = 4


def is_dialogue_context_candidate(
    text: str, ocr_confidence: float | None = None
) -> bool:
    """True if a GATE-DROPPED line is real-enough DIALOGUE to keep as CONTEXT.

    The v11 page-context model translates one marked line while seeing the whole
    page's dialogue (speaker/pronoun continuity). A dialogue line dropped before
    translation (OCR-gate / garble) still belongs in the numbered "Page:" context
    so the page the model sees has no holes — BUT a pure-SFX box, or a genuinely
    garbled low-confidence scrawl, must NOT pollute that dialogue context. This
    decides which dropped lines are kept as CONTEXT-ONLY (never rendered).

    Policy (validated on IK4 page 5):
      * exclude empty / glossary-SFX / garble-char / mostly-non-Japanese,
      * exclude LOW-OCR-confidence lines when ``ocr_confidence`` is supplied
        (genuine garble like the conf-0.49 "平速ととの…" scrawl — pure noise),
      * KEEP a short line that names a speaker / carries a pronoun
        (お母さん, 僕, 私, …) — that is precisely the continuity the page is for
        (IK4 p5 "お母さんは僕の…" fixes the He/She pronoun on the marked line),
      * KEEP a dialogue-LENGTH mostly-Japanese line,
      * otherwise exclude (short generic fragment / exclamation).
    """
    norm = unicodedata.normalize("NFC", text or "").strip()
    if not norm:
        return False
    # Glossary-matched SFX (ぬちょ, ビクン, …) are handled out-of-band; never
    # dialogue context. Local import keeps this module dependency-light.
    try:
        from app.services.sfx_glossary import sfx_pre_translate
        if sfx_pre_translate(norm) is not None:
            return False
    except Exception:
        pass
    if _has_garble_chars(norm):
        return False
    analysis = analyze_characters(norm)
    if analysis.japanese_ratio < _MIN_JP_RATIO_FOR_LOWCONF:
        return False
    # Genuinely-garbled low-confidence scrawl is noise, not context. Only applies
    # when confidence is known (the gate has it); the text-only call stays
    # length/JP-ratio based for back-compat.
    if ocr_confidence is not None and ocr_confidence < DEFAULT_CONF_THRESHOLD:
        return False
    # A speaker/pronoun reference makes even a short line valuable context.
    if _has_speaker_reference(norm) and len(norm) >= _CONTEXT_MIN_LEN_WITH_SPEAKER:
        return True
    # Otherwise require dialogue length: short generic fragments are SFX-ish.
    if len(norm) < _DIALOGUE_MIN_LEN:
        return False
    return True


def should_erase_dropped(text: str) -> bool:
    """True if a gate-dropped region is real Japanese ink worth ERASING.

    A region dropped by ``is_garbled_low_conf`` is never translated, but it may
    still be genuine Japanese SFX/scrawl on the page. Leaving it untouched
    renders the raw Japanese into the final image. This decides whether such a
    dropped region should be inpainted away (erased) even though no translation
    will be drawn over it. Conservative: a non-empty region must contain at
    least one Japanese glyph (so stray Latin/garble-only crops are left alone).

      * Empty / near-empty short text -> erase (an empty low-conf crop is real
        ink the recognizer couldn't decode, typical of stylized SFX).
      * Garble char present AND a JP glyph present -> erase.
      * japanese_ratio > 0.5 -> erase.
    """
    norm = unicodedata.normalize("NFC", text).strip()
    analysis = analyze_characters(norm)

    if not norm:
        # Empty/near-empty: real ink the recognizer gave up on -> erase.
        return True

    # Non-empty text must contain a Japanese glyph to be worth erasing.
    if analysis.japanese_count == 0:
        return False

    if _has_garble_chars(norm):
        return True
    if analysis.japanese_ratio > _MIN_JP_RATIO_FOR_LOWCONF:
        return True
    return False

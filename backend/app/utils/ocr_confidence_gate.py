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


def is_implausible_japanese(text: str) -> bool:
    """True if ``text`` reads as garbled OCR despite being mostly Japanese.

    A *linguistic*-plausibility heuristic (NOT confidence-based) so that
    confidently-garbled OCR is caught even at high OCR confidence. Deliberately
    narrow: it only returns True on patterns that cannot occur in genuine
    Japanese dialogue, so it does not drop real lines.

    Signals (any one is sufficient):
      * Garbled leading small-tsu prefix (page 070 "..?っく混みますよ").
      * Heavy ASCII-letter intrusion in Japanese text (logo/URL/handle garble).
    """
    norm = unicodedata.normalize("NFC", text).strip()
    if not norm:
        return False
    if _has_garbled_leading_tsu(norm):
        return True
    if _has_latin_intrusion(norm):
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

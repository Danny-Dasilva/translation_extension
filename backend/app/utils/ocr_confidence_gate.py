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
    """
    if ocr_confidence >= conf_threshold:
        return False

    norm = unicodedata.normalize("NFC", text).strip()
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

"""Post-translation glossary chain (shared by the API router and the batch
pipeline so they never diverge).

These are PURE post-edits on the model OUTPUT (optionally conditioned on the
Japanese source). The v11 translation model is acutely train/serve
format-sensitive, so we must NOT inject glossaries/instructions into the prompt
— every correction happens here, after the model returns.

Chain order (each is conservative and idempotent):
  1. restore_register   — un-euphemize explicit terms (e.g. 潮 -> "squirt")
  2. canonicalize_names — fix corrupted cast names (Kana/Kana-ji -> Kanako, ...)
  3. lock_address_terms — おばさん vocative -> "Auntie" (audit's #1 register error)
  4. clean_sfx_output   — suppress "SFX for a ..." meta-leaks; fix onomatopoeia
  5. strip_model_artifacts — drop markdown-bold + assistant-refusal leaks
  6. strip_romaji_honorifics — drop leaked "-chan"/"-san" honorifics; onii->big bro
  7. restore_honorifics — put back a SOURCE-CONFIRMED honorific step 6 removed
     (or that the model dropped outright), when the source unambiguously
     carries it. See app/services/honorific_glossary.py for the design.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence

from app.config import settings
from app.services.register_glossary import restore_register
from app.services.name_glossary import canonicalize_names
from app.services.address_glossary import lock_address_terms
from app.services.sfx_glossary import clean_sfx_output
from app.services.output_sanitize import (
    strip_model_artifacts,
    strip_romaji_honorifics,
)
from app.services.honorific_glossary import restore_honorifics


# --------------------------------------------------------------------------- #
# Over-expansion faithfulness gate (anti-hallucination, fix #2)
# --------------------------------------------------------------------------- #
# On short, context-light bubbles the small model CONFIDENTLY HALLUCINATES — it
# answers a 4-char source (お母さんSサイズ) or a grunt (うくっ) with a full,
# plausible-but-wrong sentence. This pure post-gate flags an English output
# whose word count FAR exceeds a generous linear function of the Japanese source
# length, and lets the service retry via the no-scene-pressure PLAIN prompt or
# fall back to an ellipsis. It is the faithfulness *floor* that runs in BOTH the
# API router and the batch pipeline (via postedit_one), independent of the
# generation-time token cap in vllm_openai_translation_service.
#
# Tuning rationale (deliberately SLACK so legit short->meaningful survives):
#   * OVER_EXPANSION_WORDS_PER_JP_CHAR (slope): Japanese is dense — a normal
#     JP->EN rendering runs well under ~2.5 English words per source char. We
#     allow up to 2.5 words/char before the slope even begins to bite, so a
#     genuine 2.5x expansion (and anything tighter) always passes.
#   * OVER_EXPANSION_CONST (intercept): a flat +3 words of headroom so very short
#     but meaningful bubbles (ええ -> "What are you saying?!", 4 words on a 2-char
#     source) clear the bar comfortably.
#   * OVER_EXPANSION_ABS_MIN_WORDS: a hard floor — outputs with fewer than this
#     many words are NEVER flagged, so no short, faithful line is ever blanked.
# Net effect: a 4-char source flagged only past ceil(4*2.5)+3 = 13 words (the
# ~40-word runaway the audit found is far over this), while a 2-char meaningful
# reply (<8 words) and any <=2.5 words/char expansion pass untouched.
OVER_EXPANSION_WORDS_PER_JP_CHAR = 2.5
OVER_EXPANSION_CONST = 3
OVER_EXPANSION_ABS_MIN_WORDS = 8
# The faithfulness floor when an over-expansion survives retry: render an
# ellipsis (the codebase's established "no confident meaning" marker) rather
# than a confidently-invented sentence.
OVER_EXPANSION_FALLBACK = "..."

# Japanese characters that carry no length signal (punctuation / spacing) and so
# must NOT inflate the per-source budget.
_JP_NON_CONTENT_RE = re.compile(
    r"[\s　。、！？!?.,…‥・〜~「」『』（）()\"'“”‘’♪♡☆※]+"
)


def jp_content_len(jp: Optional[str]) -> int:
    """Length of a Japanese source in *content* chars (punctuation/space removed).

    Used to size the over-expansion budget so a line padded with 。、！ cannot
    buy itself extra English-word headroom.
    """
    if not jp:
        return 0
    return len(_JP_NON_CONTENT_RE.sub("", jp))


def en_word_count(en: Optional[str]) -> int:
    """Whitespace-delimited word count of an English translation."""
    if not en:
        return 0
    return len(en.split())


def is_over_expanded(en: Optional[str], jp: Optional[str]) -> bool:
    """True when ``en`` is far longer than the JP source can faithfully justify.

    Conservative by construction (see module constants): never flags an output
    under OVER_EXPANSION_ABS_MIN_WORDS words, and otherwise only fires past
    ``ceil(jp_content_len * WORDS_PER_JP_CHAR) + CONST`` words. Missing source or
    empty output -> never flagged.
    """
    words = en_word_count(en)
    if words < OVER_EXPANSION_ABS_MIN_WORDS:
        return False
    jp_len = jp_content_len(jp)
    if jp_len <= 0:
        return False
    threshold = math.ceil(jp_len * OVER_EXPANSION_WORDS_PER_JP_CHAR) + OVER_EXPANSION_CONST
    return words > threshold


def gate_over_expansion(
    en: Optional[str], jp: Optional[str], fallback: str = OVER_EXPANSION_FALLBACK
) -> Optional[str]:
    """Pure faithfulness floor: blank a confidently over-expanded line.

    Returns ``en`` unchanged unless :func:`is_over_expanded` fires, in which case
    the (model-invented) sentence is replaced by ``fallback`` (an ellipsis). This
    is the last-resort guard wired into :func:`postedit_one`; the service applies
    a single PLAIN-prompt retry BEFORE reaching this floor.
    """
    if en is None:
        return None
    return fallback if is_over_expanded(en, jp) else en


def postedit_one(
    en: Optional[str],
    jp: Optional[str],
    ocr_conf: Optional[float] = None,
) -> Optional[str]:
    """Apply the full glossary chain to a single (translation, source) pair.

    ``ocr_conf`` (optional, in [0, 1]) is threaded into the name canonicalizer
    so low-confidence bubbles do not get a model-invented proper name. Omitting
    it preserves prior behaviour exactly."""
    en = restore_register(en, jp or "")
    en = canonicalize_names(en, jp, ocr_conf=ocr_conf)
    en = lock_address_terms(en, jp)
    en = clean_sfx_output(en, jp)
    en = strip_model_artifacts(en)
    en = strip_romaji_honorifics(en)
    # FAITHFULNESS FLOOR (fix #2): if the output is still a confidently
    # over-expanded hallucination relative to the JP source, blank it to an
    # ellipsis. Runs last so all glossary fixes apply first; conservative
    # thresholds mean faithful short->meaningful lines are never touched.
    en = gate_over_expansion(en, jp)
    # HONORIFIC RESTORATION (discourse-fidelity safe mitigation): put back a
    # source-confirmed honorific (e.g. "Maki-sama") that step 6 stripped or
    # that the model dropped outright. Gated: TIER A (append a suffix to an
    # already-present bare name) always runs when the master flag is on;
    # TIER B (invent a vocative prefix when the name is fully absent) is a
    # separate, higher-risk opt-in. Runs absolute last so nothing downstream
    # can re-strip it. See app/services/honorific_glossary.py.
    if getattr(settings, "postedit_restore_honorifics", True):
        en = restore_honorifics(
            en,
            jp,
            allow_vocative_insertion=getattr(
                settings, "postedit_restore_dropped_honorific_vocative", False
            ),
        )
    return en


def apply_postedit_glossaries(
    translations: Sequence[Optional[str]],
    jp_texts: Sequence[Optional[str]],
    ocr_confs: Optional[Sequence[Optional[float]]] = None,
) -> List[Optional[str]]:
    """Apply the glossary chain to a list of translations aligned 1:1 with
    their Japanese sources. Extra translations (no paired source) are passed
    through the EN-only edits with jp=None.

    ``ocr_confs`` (optional) is aligned 1:1 with ``jp_texts`` and threaded into
    the name canonicalizer to suppress low-confidence name invention. When
    omitted, behaviour is unchanged."""
    out: List[Optional[str]] = []
    n = len(jp_texts)
    nc = len(ocr_confs) if ocr_confs is not None else 0
    for i, en in enumerate(translations):
        jp = jp_texts[i] if i < n else None
        conf = ocr_confs[i] if (ocr_confs is not None and i < nc) else None
        out.append(postedit_one(en, jp, ocr_conf=conf))
    return out

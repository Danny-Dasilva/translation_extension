"""Post-translation glossary chain (shared by the API router and the batch
pipeline so they never diverge).

These are PURE post-edits on the model OUTPUT (optionally conditioned on the
Japanese source). The v11 translation model is acutely train/serve
format-sensitive, so we must NOT inject glossaries/instructions into the prompt
— every correction happens here, after the model returns.

Chain order (each is conservative and idempotent):
  1. restore_register   — un-euphemize explicit terms (e.g. 潮 -> "squirt")
  2. canonicalize_names — fix corrupted cast names (Kana/Kana-ji -> Kanako, ...)
  3. clean_sfx_output   — suppress "SFX for a ..." meta-leaks; fix onomatopoeia
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from app.services.register_glossary import restore_register
from app.services.name_glossary import canonicalize_names
from app.services.sfx_glossary import clean_sfx_output


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
    en = clean_sfx_output(en, jp)
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

"""Strip model-artifact leaks from translation output (post-translation).

The full-chapter audit found two non-translation artifacts leaking into rendered
bubbles:

  1. MARKDOWN bold — e.g. ``"...I'm on a **Lumber** with a **Sliding Ruler**..."``
     Manga dialogue never contains markdown, so ``**`` is always a leak.

  2. ASSISTANT-REFUSAL leaks — on a garbled bubble the instruction-tuned base
     occasionally emits a chat-assistant refusal instead of a translation, e.g.
     ``"I'm sorry, I'm not sure what you want me to do with this..."``. Rendering
     that on the page is worse than an empty bubble, so we blank it.

  3. ROMAJI-HONORIFIC leaks — the model sometimes keeps a Japanese honorific on
     a romanised name ("Yui-chan", "No-kun", "-san"/"-sama") or leaves a kinship
     vocative untranslated ("Onii-chan"). ``strip_romaji_honorifics`` drops the
     trailing ``-chan/-kun/-san/-sama/-senpai`` from hyphenated romaji forms and
     maps the standalone ``onii-/onee-`` kinship terms to natural English.

All are pure OUTPUT edits (no source needed) and run last in the post-edit
chain, after the glossary corrections.
"""
from __future__ import annotations

import re
from typing import Optional

# ``**bold**`` -> ``bold``; then drop any stray ``**`` markers left over.
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_STRAY_BOLD = re.compile(r"\*\*")

# Assistant-refusal / meta signatures. Kept TIGHT and assistant-specific so we
# never blank a legitimate in-character line. Matched case-insensitively as a
# substring of the whole output.
_REFUSAL_SIGNATURES = (
    "as an ai",
    "i cannot translate",
    "i can't translate",
    "i am unable to translate",
    "i'm unable to translate",
    "i'm not sure what you want me to",
    "i am not sure what you want me to",
    "i cannot fulfill",
    "i can't assist",
    "i'm sorry, but i can",
    "i'm sorry, i can",
    "i don't have enough information",
)


# --------------------------------------------------------------------------- #
# Romaji-honorific leaks.
# --------------------------------------------------------------------------- #
# The Japanese honorific suffixes the model leaves attached to a romanised name.
# ``sempai`` is the alternate romanisation of ``senpai``. None of these is a
# common English word-ending after a hyphen, so a ``\b<stem>-<honorific>\b``
# match cannot fire on real English ("T-shirt", "mother-in-law").
_HONORIFIC = r"(?:chan|kun|san|sama|senpai|sempai)"

# Kinship vocatives left untranslated -> natural English. Run BEFORE the generic
# strip so "onii-chan" becomes "big brother", not the bare stem "onii". Leading
# ``o`` is optional (nii-/nee- also appear). Honorific suffix required (hyphen
# form only) to stay conservative.
_KINSHIP = re.compile(r"\b(o?nii|o?nee)-" + _HONORIFIC + r"\b", re.IGNORECASE)
_KINSHIP_EN = {"nii": "big brother", "nee": "big sister"}

# Generic strip: a romaji name stem + a hyphenated honorific -> just the stem.
# "Yui-chan" -> "Yui", "No-kun" -> "No". Whole-word boundaries on both ends so a
# trailing honorific that is actually part of a longer word ("…-sanctuary") is
# never touched.
_ROMAJI_HONORIFIC = re.compile(
    r"\b([A-Za-z][A-Za-z']*)-" + _HONORIFIC + r"\b", re.IGNORECASE
)


def _kinship_repl(m: "re.Match[str]") -> str:
    en = _KINSHIP_EN["nii" if m.group(1).lower().endswith("nii") else "nee"]
    # Preserve sentence-initial capitalisation ("Onii-chan," -> "Big brother,").
    return en[:1].upper() + en[1:] if m.group(0)[:1].isupper() else en


def strip_romaji_honorifics(en: Optional[str]) -> Optional[str]:
    """Drop leaked Japanese honorifics from hyphenated romaji forms.

    Conservative — only fires on ``<romaji-stem>-<honorific>`` (the hyphen +
    a known honorific suffix is the unambiguous leak signal), so plain English
    words and English hyphenated compounds are never altered. Idempotent.
    ``None``/empty passes through unchanged.
    """
    if not en:
        return en
    out = _KINSHIP.sub(_kinship_repl, en)
    out = _ROMAJI_HONORIFIC.sub(r"\1", out)
    return out


def strip_model_artifacts(en: Optional[str]) -> Optional[str]:
    """Remove markdown-bold leaks and blank assistant-refusal leaks.

    Returns the cleaned string, or "" when the whole output is an assistant
    refusal (better an empty bubble than a rendered refusal). ``None``/empty
    passes through unchanged.
    """
    if not en:
        return en
    # 1. assistant-refusal -> blank (check before stripping so markers don't hide it)
    low = en.lower()
    if any(sig in low for sig in _REFUSAL_SIGNATURES):
        return ""
    # 2. markdown bold -> plain
    out = _BOLD.sub(r"\1", en)
    out = _STRAY_BOLD.sub("", out)
    return out

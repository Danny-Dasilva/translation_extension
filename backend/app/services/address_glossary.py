"""Source-conditioned address-term lock (post-translation glossary).

PROBLEM
-------
The full-chapter quality audit found the abuser's vocative ``おばさん`` (the
mother / "Auntie") is the SINGLE most frequent error: the v11 model renders it
inconsistently as "Old lady" / "Old woman" / "Woman" / "ma'am" / "sweetie"
instead of the human-scanlation convention "Auntie" — ~10× per 34-page range,
and ~87% wrong in the worst range. The model CAN produce "Auntie" (it does so
sometimes), so this is pure INCONSISTENCY, fixable with a deterministic
post-edit and no retrain.

WHY A POST-EDIT (not a prompt/retrain change)
---------------------------------------------
Same reason as ``register_glossary``: the v11 model is acutely train/serve
format-sensitive (drifting the format caused a documented ~95% chrF collapse),
so we never touch the prompt. We normalize the OUTPUT after the model returns.

DESIGN: precision over recall
-----------------------------
Rewrite ONLY when BOTH hold:
  (a) the Japanese SOURCE contains the auntie vocative ``おばさん`` / ``おばちゃん``
      (NOT ``おばあさん`` / ``お婆さん`` = grandmother — those legitimately map to
      "old woman"/"granny" and the substring ``おばさん`` does not match them), AND
  (b) the English OUTPUT contains a known WRONG rendering (whole-word).

We do NOT touch a bare "woman"/"lady" used as a *noun* mid-sentence ("the woman
who lives next door") — too common in legitimate lines. We DO rewrite bare
"lady"/"woman" when it is a clear VOCATIVE (direct address): standalone bubble
("Lady!"), sentence-initial ("Lady, …"), sentence-terminal/medial ("…, lady?"),
or the definite reference "the lady". These were the model's single most common
wrong vocative for おばさん ("What do you want, lady?"). Casing of the matched
span is preserved so "Old lady" -> "Auntie" but "old lady" -> "auntie".
"""
from __future__ import annotations

import re
from typing import Optional

# The auntie vocative. NOTE: ``おばさん`` is a substring-safe key — it does NOT
# occur inside ``おばあさん`` (grandmother, extra あ) so we never rewrite grandma.
_OBASAN_TERMS = ("おばさん", "おばちゃん")

# Wrong renderings -> "Auntie". Multi-word demeaning forms + wrong-register
# vocatives observed in the audit. Bare "woman"/"lady" are INTENTIONALLY absent
# (too risky mid-sentence).
_WRONG_OBASAN = (
    "old lady",
    "old woman",
    "old hag",
    "granny",
    "grandma",
    "ma'am",
    "sweetie",
)
_CORRECT = "auntie"

_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", re.IGNORECASE)
    for w in _WRONG_OBASAN
]

# Bare "lady"/"woman" -> "Auntie", but ONLY as a VOCATIVE (direct address),
# never as a descriptive noun. These are the model's most frequent wrong
# vocative for おばさん. The word to swap is captured in group "w" so casing is
# preserved exactly like the multi-word forms above; ``pre``/``post`` re-emit
# the surrounding context untouched. Relative-clause noun phrases
# ("the woman who…", "the lady who…") and plain mid-sentence noun use are left
# alone.
_VOC_WORD = r"lady|woman"
# 1. sentence-initial:  "Lady, …"  /  after a sentence break ". Woman, …"
_VOC_INITIAL = re.compile(
    r"(?P<pre>^|[.!?…]\s+)(?P<w>" + _VOC_WORD + r")(?P<post>,)", re.IGNORECASE
)
# 2. sentence-terminal / medial:  "…, lady?"  "…, woman!"  "Thanks, lady"
_VOC_TERMINAL = re.compile(
    r"(?P<pre>,\s)(?P<w>" + _VOC_WORD + r")(?P<post>\s*(?:[,.!?…]|$))", re.IGNORECASE
)
# 3. standalone bubble: the WHOLE string is the vocative ("Lady!", "Woman").
#    Empty ``pre`` group keeps it compatible with the shared replacement helper.
_VOC_STANDALONE = re.compile(
    r"^(?P<pre>)(?P<w>" + _VOC_WORD + r")(?P<post>[.!?…]*)$", re.IGNORECASE
)
# 4. definite "the lady" reference -> "Auntie" (consumes the article), unless it
#    heads a relative clause ("the lady who…") = descriptive, not address. Only
#    "the lady" (not "the woman", which is too often a plain noun phrase).
_VOC_THE_LADY = re.compile(
    r"\bthe\s+lady\b(?!\s+(?:who|that|which|whom|whose)\b)", re.IGNORECASE
)


def _match_case(template: str, replacement: str) -> str:
    """Re-apply the casing of ``template`` (matched span) to ``replacement``."""
    if template.isupper():
        return replacement.upper()
    if template[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _voc_word_repl(m: "re.Match[str]") -> str:
    """Swap only the captured vocative word for case-matched "auntie", keeping
    the surrounding ``pre``/``post`` context exactly."""
    return m.group("pre") + _match_case(m.group("w"), _CORRECT) + m.group("post")


def _voc_the_lady_repl(m: "re.Match[str]") -> str:
    """Collapse "the lady" -> one case-matched "auntie" (drops the article)."""
    return _match_case(m.group(0), _CORRECT)


def lock_address_terms(en: Optional[str], jp: Optional[str]) -> Optional[str]:
    """Lock the ``おばさん`` vocative to "Auntie" in ``en`` given source ``jp``.

    Source-conditioned, precision-over-recall: only fires when the JP contains
    ``おばさん``/``おばちゃん`` AND the English contains a known wrong rendering.
    Casing preserved. ``None``/empty passes through unchanged.
    """
    if not en or not jp:
        return en
    if not any(term in jp for term in _OBASAN_TERMS):
        return en
    out = en
    for pat in _PATTERNS:
        out = pat.sub(lambda m: _match_case(m.group(0), _CORRECT), out)
    # Bare "lady"/"woman" — vocative-only (precision over recall).
    out = _VOC_INITIAL.sub(_voc_word_repl, out)
    out = _VOC_TERMINAL.sub(_voc_word_repl, out)
    out = _VOC_STANDALONE.sub(_voc_word_repl, out)
    out = _VOC_THE_LADY.sub(_voc_the_lady_repl, out)
    return out

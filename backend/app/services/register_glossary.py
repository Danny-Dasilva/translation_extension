"""Source-conditioned explicit-register restoration (post-translation glossary).

P1-2 — Explicit-register preservation.

PROBLEM
-------
This corpus is explicit adult (NSFW) manga. QA against a human reference shows
the v11 translation model systematically EUPHEMIZES a handful of domain-specific
explicit Japanese terms, losing meaning. The clearest example:

    潮  (in a sexual context = female ejaculate / "squirt")
        -> the model emits "seawater" / "sea water" / "tide" / "salt water"
        e.g. page 031: リビングが潮まみれじゃん
             model:  "The living room is covered in seawater."
             human:  "FLOOD YOUR FUCKING LIVING ROOM, LOL!"  (squirt-implied)

WHY A POST-EDIT (and NOT a prompt/retrain change)
-------------------------------------------------
The v11 model is acutely train/serve format-sensitive: drifting the serving
format from the training format has produced a documented ~95% quality collapse,
and a prior NSFW-oversampling retrain REGRESSED register (the model learned
euphemism). So we must NOT touch the prompt or retrain. Instead we do a narrow,
deterministic post-edit on the (jp, en) pair after translation.

DESIGN: precision over recall
-----------------------------
We rewrite the English ONLY when BOTH conditions hold:

  (a) the Japanese SOURCE contains a known explicit term (e.g. 潮), AND
  (b) the English OUTPUT contains a known euphemism / wrong-word for that term
      (e.g. "seawater"), matched as a WHOLE WORD.

Both gates are required so we never re-explicitate a clean line, and never touch
a line where "seawater" is a legitimate literal reading (e.g. 潮干狩り = clam
digging at the beach, where there is no sexual 潮 and "seawater" is fine).

SCOPE / HONESTY
---------------
This only fixes WRONG-WORD substitutions we can do safely. It does NOT and
cannot re-insert content the model DROPPED. For example page 064
(...チンコのが... = a "your dick is better than his" comparison) is flattened to
"This is the one that actually feels like sex!!" — the word "dick" never appears
in the English, so there is nothing to substitute and we leave it untouched.
Register is the hardest fix and is only partly recoverable by post-edit.

The table is intentionally SMALL and high-precision. Add entries only when you
have a clear (jp_term, wrong_en -> correct_en) mapping confirmed against a human
reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class RegisterEntry:
    """One source-conditioned substitution rule.

    Attributes:
        jp_terms:  JP substrings; the rule arms only if the source contains one.
        wrong_en:  English euphemisms/wrong words to replace (whole-word match,
                   case-insensitive). Multi-word phrases (e.g. "sea water") are
                   matched as whole phrases.
        correct_en: the explicit replacement, written in lowercase. The original
                   casing of the matched span is re-applied (lower / Title /
                   UPPER) so it blends into the surrounding line.
        note:      human-readable rationale (kept for documentation/audit).
    """

    jp_terms: List[str]
    wrong_en: List[str]
    correct_en: str
    note: str = ""
    # JP substrings that DISARM the rule even though a jp_term matched. These
    # are non-sexual compounds where the literal reading is correct, e.g.
    # 潮干狩り (clam digging) / 潮風 (sea breeze) — here "seawater" is fine and
    # must NOT be rewritten to "squirt". Precision guard.
    jp_excludes: List[str] = field(default_factory=list)
    # Pre-compiled whole-"word" patterns for each wrong_en token. We allow the
    # phrase to be bounded by non-alphanumeric chars rather than \b alone so
    # that "sea water" (with the space) still anchors correctly.
    _patterns: List[re.Pattern] = field(default_factory=list, compare=False, repr=False)

    def __post_init__(self):
        pats = []
        for w in self.wrong_en:
            # (?<![A-Za-z0-9]) ... (?![A-Za-z0-9]) = whole-token boundaries that
            # also work for multi-word phrases. Prevents "tide" matching inside
            # "untidy"/"tides".
            pat = re.compile(
                r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            pats.append(pat)
        # frozen dataclass: bypass the immutability guard to cache patterns.
        object.__setattr__(self, "_patterns", pats)


def _match_case(template: str, replacement: str) -> str:
    """Re-apply the casing of ``template`` (the matched span) to ``replacement``.

    - "SEAWATER" -> "SQUIRT"   (all upper)
    - "Seawater" -> "Squirt"   (title / leading cap)
    - "seawater" -> "squirt"   (lower / mixed -> lower)
    """
    if template.isupper():
        return replacement.upper()
    if template[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


# ---------------------------------------------------------------------------
# The glossary. SMALL and HIGH-PRECISION on purpose. See module docstring.
# ---------------------------------------------------------------------------
GLOSSARY: List[RegisterEntry] = [
    RegisterEntry(
        jp_terms=["潮"],
        wrong_en=["seawater", "sea water", "salt water", "saltwater", "tide"],
        correct_en="squirt",
        # Non-sexual 潮 compounds: the literal seawater/tide reading is correct.
        jp_excludes=["潮干狩", "潮風", "潮目", "潮流", "潮汐", "黒潮", "親潮", "満潮", "干潮", "潮位"],
        note=(
            "潮 in this NSFW context = female ejaculate ('squirt'). v11 "
            "literalizes it to seawater/tide. Confirmed vs human ref p031: "
            "リビングが潮まみれじゃん -> squirt, not seawater."
        ),
    ),
]


def restore_register(en: Optional[str], jp: str) -> Optional[str]:
    """Conservatively restore explicit register in ``en`` given source ``jp``.

    Source-conditioned post-edit: for each glossary entry, substitute a
    wrong_en token with correct_en ONLY when the JP source contains one of the
    entry's jp_terms AND the English contains the wrong_en token (whole-word,
    case-insensitive). Casing of the matched span is preserved.

    This is precision-over-recall: lines that don't match BOTH gates are
    returned unchanged. ``None`` (dropped/empty bubble) passes through.

    Args:
        en: the model's English translation (may be None or empty).
        jp: the OCR'd Japanese source for the same bubble.

    Returns:
        The (possibly) register-corrected English, or the input unchanged.
    """
    if not en:
        return en
    if not jp:
        return en

    out = en
    for entry in GLOSSARY:
        # Gate (a): source must contain a known explicit JP term...
        if not any(term in jp for term in entry.jp_terms):
            continue
        # ...but NOT inside a non-sexual compound (潮干狩り etc.).
        if any(ex in jp for ex in entry.jp_excludes):
            continue
        # Gate (b): English must contain a known wrong word; substitute it.
        for pat in entry._patterns:
            out = pat.sub(
                lambda m: _match_case(m.group(0), entry.correct_en),
                out,
            )
    return out

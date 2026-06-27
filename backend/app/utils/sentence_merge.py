"""Cross-bubble JP sentence-continuation merge (PRE-translation re-segmentation).

A single Japanese sentence is frequently typeset across 2-3 stacked bubbles in
the SAME vertical column (e.g. a connective dangles at the end of the upper
bubble and the predicate / sentence-final particle lands in the bubble below).
The detector + OCR treat each bubble as an independent line, so the v11
page-context translator renders the halves as separate utterances — and the
halves then CONTRADICT (p8 "今朝はあの子達が" mistranslated as an affirmative that
negates the paired "didn't come"; "からな" -> "It's from you"; "ごめん"/"なさい" ->
"Sorry"/"Be quiet").

This module detects such continuations and groups the member lines into ONE
translation unit BEFORE translation, so the model sees the whole sentence at
once. It is RE-SEGMENTATION ONLY — the merged JP is passed to the EXISTING
``translate_page_context_marked`` as a single marked line, with no prompt /
template change (so NO train/serve format risk). The English is then RE-SPLIT
back to the member bubbles for typesetting: the simplest low-risk mapping renders
the full merged English in the FIRST (reading-order-earliest) member bubble and
blanks the continuation bubbles.

Guards (deliberately conservative — fusing two DISTINCT utterances is worse than
leaving a split sentence):
  * members must be STRICTLY ADJACENT in page reading order,
  * members must sit in the SAME vertical column (geometry),
  * merge span is capped at ``MAX_MERGE_SPAN`` (2-3 bubbles),
  * the LEADING line MUST carry a dangling signal (a trailing connective with no
    terminal punctuation) OR the trailing line is a bare sentence-final
    particle/auxiliary — a continuation is only asserted on a positive signal.

These are PURE helpers (geometry + string heuristics); no model/service
dependency, so they unit-test in isolation.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# Max bubbles fused into one sentence unit. A real cross-bubble sentence spans
# 2 (occasionally 3) balloons; beyond that we are almost certainly fusing
# distinct utterances, so cap hard.
MAX_MERGE_SPAN = 3

# Terminal punctuation: presence of any of these at the end of the LEADING line
# means the sentence already closed there — do NOT treat it as dangling.
_TERMINAL_PUNCT = "。．.!！?？…‥〜~、,"

# Dangling connectives / particles: when the LEADING line ENDS with one of these
# (and has no terminal punctuation) the sentence clearly continues into the next
# bubble. Kept narrow to the connectives that genuinely dangle mid-sentence in
# manga dialogue (te-form / de / ga / no / kedo and their kana spellings).
_DANGLING_TRAILERS = (
    "けど", "けれど", "けれども",
    "から", "ので", "のに",
    "って",
    "て", "で", "が", "の", "し", "ば", "と",
)

# Bare sentence-final particles / auxiliaries that, when they make up the WHOLE
# trailing line, are the tail of the sentence that started in the bubble above
# (they cannot stand as an utterance on their own). A trailing line that is ONLY
# one of these is a strong continuation signal even if the leading-line trailer
# check is borderline.
_BARE_SENTENCE_FINAL = (
    "からな", "からね", "のに", "なさい", "だろう", "でしょう",
    "させられる", "られる", "させる", "ている", "でいる",
    "んだ", "のよ", "のね", "わよ", "かな", "かしら",
)


def _norm(text: Optional[str]) -> str:
    return unicodedata.normalize("NFC", text or "").strip()


def _ends_with_terminal_punct(norm: str) -> bool:
    return bool(norm) and norm[-1] in _TERMINAL_PUNCT


def has_dangling_connective(text: Optional[str]) -> bool:
    """True if ``text`` ends with a dangling connective and is NOT terminated.

    The LEADING-line signal: a te-form / で / が / の / けど / … trailer with no
    closing punctuation means the clause runs into the next bubble. A line that
    already ends in 。!?… closed its sentence and is never dangling.
    """
    norm = _norm(text)
    if not norm:
        return False
    if _ends_with_terminal_punct(norm):
        return False
    # Longest trailer first so けど is preferred over a bare と etc.
    for trailer in sorted(_DANGLING_TRAILERS, key=len, reverse=True):
        if norm.endswith(trailer):
            # A 1-char particle trailer (て/で/が/の…) on a line that is ITSELF
            # only that particle is not a dangling connective — it is just a
            # fragment; require some lead-in so we only fuse genuine clauses.
            if len(trailer) == 1 and len(norm) <= 1:
                return False
            return True
    return False


def is_bare_sentence_final(text: Optional[str]) -> bool:
    """True if the WHOLE line is a bare sentence-final particle / auxiliary.

    The TRAILING-line signal: a line consisting only of からな / のに / なさい /
    だろう / させられる … is the tail of a sentence opened in the bubble above —
    it cannot be an utterance on its own, so its parent is the previous bubble.
    """
    norm = _norm(text)
    if not norm:
        return False
    # Strip a single trailing terminal punct so "なさい。" still matches.
    core = norm
    while core and core[-1] in _TERMINAL_PUNCT:
        core = core[:-1]
    return core in _BARE_SENTENCE_FINAL


def _column_of(block: Dict) -> float:
    return (float(block["minX"]) + float(block["maxX"])) / 2.0


def _same_column(a: Dict, b: Dict, *, tol: float) -> bool:
    return abs(_column_of(a) - _column_of(b)) <= tol


def _vertically_adjacent(upper: Dict, lower: Dict) -> bool:
    """``lower`` reads directly below ``upper`` (top-to-bottom in the column).

    Within a manga column reading order is top-to-bottom, so the continuation
    bubble's vertical center sits below the leading bubble's. We require the
    lower block's center to be below the upper block's center (no large gap test
    — same-column reading-order adjacency already constrains this).
    """
    upper_cy = (float(upper["minY"]) + float(upper["maxY"])) / 2.0
    lower_cy = (float(lower["minY"]) + float(lower["maxY"])) / 2.0
    return lower_cy >= upper_cy


@dataclass
class MergeGroup:
    """One translation unit after cross-bubble merge.

    ``member_positions`` are the 0-based indices INTO the reading-order page
    (``page_context_lines``) that were fused, in reading order. ``merged_text``
    is the concatenated JP to translate as one unit. ``lead_position`` is the
    member that renders the full English (the rest are blanked on re-split).
    """

    member_positions: List[int]
    merged_text: str

    @property
    def lead_position(self) -> int:
        return self.member_positions[0]

    @property
    def continuation_positions(self) -> List[int]:
        return self.member_positions[1:]

    @property
    def is_merged(self) -> bool:
        return len(self.member_positions) > 1


@dataclass
class SentenceMergePlan:
    """Result of :func:`detect_sentence_continuations`.

    ``groups`` covers EVERY page position exactly once, in reading order — a
    solo (un-merged) line is a single-member group. ``merged_page_lines`` is the
    page context after re-segmentation (one entry per group), and
    ``position_to_group`` maps each original page position to its group index.
    """

    groups: List[MergeGroup] = field(default_factory=list)
    merged_page_lines: List[str] = field(default_factory=list)
    position_to_group: Dict[int, int] = field(default_factory=dict)

    @property
    def num_merges(self) -> int:
        return sum(1 for g in self.groups if g.is_merged)


def detect_sentence_continuations(
    page_context_lines: Sequence[str],
    blocks: Sequence[Dict],
    *,
    column_tol: Optional[float] = None,
    max_span: int = MAX_MERGE_SPAN,
) -> SentenceMergePlan:
    """Group cross-bubble sentence continuations into single translation units.

    ``page_context_lines`` and ``blocks`` are PARALLEL lists already in PAGE
    READING ORDER (column-major RTL, top-to-bottom — the output of
    ``orphan_lines.reading_order_sort``). Reading-order adjacency is therefore
    just index adjacency; same-column adjacency is confirmed by block geometry.

    Two (up to ``max_span``) strictly-adjacent same-column lines are fused when:
      * the LEADING line ends with a dangling connective (no terminal punct), OR
      * the TRAILING line is a bare sentence-final particle / auxiliary.

    Returns a :class:`SentenceMergePlan`; when no merge fires every line is its
    own single-member group (an identity plan), so callers can wire it
    unconditionally.
    """
    n = len(page_context_lines)
    plan = SentenceMergePlan()
    if n == 0:
        return plan
    if len(blocks) != n:
        # Defensive: misaligned inputs -> identity plan (no merge), never crash.
        for i, line in enumerate(page_context_lines):
            plan.groups.append(MergeGroup([i], _norm(line)))
            plan.position_to_group[i] = len(plan.groups) - 1
        plan.merged_page_lines = [g.merged_text for g in plan.groups]
        return plan

    # Column tolerance: derive from page width like reading_order_sort so the
    # "same column" test matches the order the lines were placed in.
    if column_tol is None:
        try:
            page_w = max(float(b["maxX"]) for b in blocks) - min(
                float(b["minX"]) for b in blocks
            )
        except (ValueError, KeyError):
            page_w = 0.0
        column_tol = max(40.0, page_w * 0.06)

    used = [False] * n
    for i in range(n):
        if used[i]:
            continue
        members = [i]
        # Greedily extend the group downward while the chain keeps signalling a
        # continuation and stays in the same column, capped at max_span.
        j = i
        while len(members) < max_span and j + 1 < n and not used[j + 1]:
            lead = page_context_lines[j]
            nxt = page_context_lines[j + 1]
            if not _same_column(blocks[j], blocks[j + 1], tol=column_tol):
                break
            if not _vertically_adjacent(blocks[j], blocks[j + 1]):
                break
            cont = has_dangling_connective(lead) or is_bare_sentence_final(nxt)
            if not cont:
                break
            members.append(j + 1)
            j += 1

        for m in members:
            used[m] = True
        merged_text = "".join(_norm(page_context_lines[m]) for m in members)
        gidx = len(plan.groups)
        plan.groups.append(MergeGroup(list(members), merged_text))
        for m in members:
            plan.position_to_group[m] = gidx

    plan.merged_page_lines = [g.merged_text for g in plan.groups]
    return plan


def resplit_translation_to_members(
    group: MergeGroup,
    translated_text: str,
    *,
    continuation_filler: str = "",
) -> Dict[int, str]:
    """Map ONE merged group's English back onto its member page positions.

    Lowest-risk re-split: the full merged English renders in the LEADING (first
    reading-order) member; every continuation member is blanked (or set to a
    caller-supplied ellipsis filler). Returns ``{page_position: english}`` for
    EVERY member so the caller can place each on its own bubble.
    """
    out: Dict[int, str] = {group.lead_position: translated_text}
    for pos in group.continuation_positions:
        out[pos] = continuation_filler
    return out

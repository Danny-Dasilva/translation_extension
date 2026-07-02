"""Column -> parent-speech-bubble grouping (PRE-translation re-segmentation).

ROOT-CAUSE FIX (pipeline audit). A single multi-column vertical speech balloon
is detected/OCR'd as ONE box PER COLUMN (CTD emits a tight box per text column;
koharu's merge only fuses vertically-stacked SAME-column lines, never the
side-by-side columns of one balloon). So a balloon that reads
``そ…それが／まーきさまちょうど切れてて／明日買いに`` arrives downstream as THREE
independent translation units. The v11 page-context model then either

  * consolidates the whole sentence onto ONE fragment and BLANKS the others
    (silent omissions), or
  * independently reconstructs the whole sentence on EACH fragment
    (identical EN duplicated across adjacent bubbles),

and every fragment also gets its own render box (clutter).

This module groups the column-fragments of ONE balloon into a SINGLE translation
unit BEFORE the marked-translate call, so one balloon = one marked JP line = one
render box (union bbox), EN rendered once. It produces a
:class:`~app.utils.sentence_merge.SentenceMergePlan` — the SAME structure the
existing cross-bubble sentence merge emits — so it flows through the EXISTING
``build_merged_translation_request`` / ``apply_resplit`` machinery unchanged (no
prompt / template change => NO train/serve format risk).

Grouping signal, in priority order:

  1. **CTD parent-bubble membership** (preferred). When the YOLO speech-bubble
     detector ran (production has it on both pipelines), each text block is
     mapped to the smallest bubble interior that contains its center. Sharing a
     bubble is a NECESSARY condition (different parents never fuse) but it is NOT
     sufficient: at scale on real YOLO interiors many DISTINCT utterances — and
     SFX — fall inside ONE loose / panel-sized box, so a same-id + weak-Y-overlap
     test fuses genuinely separate balloons (and the v11 model then blanks the
     rest — Ikenie4 omissions 44 -> 385). Membership therefore REPLACES the
     fallback's geometry GUESS, it does NOT bypass its geometry GUARDS: two
     fragments fuse only when they share a bubble AND pass the SAME
     same-balloon-column geometry the fallback uses (column adjacency, similar
     glyph width, strong Y-overlap, RTL directional step), AND the shared bubble
     is not implausibly large relative to them (a panel is not a balloon).

  2. **Geometric column adjacency** (fallback). When no bubble detector ran, or a
     fragment falls in no bubble, side-by-side columns are grouped by geometry:
     adjacent in page reading order, X-centers within a few glyph-widths, and
     their Y-ranges OVERLAP (columns of one balloon share vertical extent;
     vertically-stacked separate balloons do not).

Conservative guards (fusing two DISTINCT balloons is worse than leaving a split):
  * only STRICTLY-ADJACENT page-reading-order fragments fuse (contiguous runs),
  * different parent bubbles NEVER fuse,
  * even in membership mode the pair must pass the column-adjacency geometry
    (X-proximity within ``glyph_mult`` glyph-widths, similar glyph width, strong
    Y-overlap ``membership_y_overlap_min``, RTL directional step) — so a chain of
    same-bubble fragments cannot fuse two distinct utterances, and a wide SFX
    glyph is never absorbed as a "column",
  * a fragment is NOT grouped into a bubble whose area is implausibly large
    relative to the joined fragments (``membership_bubble_area_mult``) — a
    panel-sized container must not authorize fusing everything inside it,
  * the geometric fallback caps the X gap at ``glyph_mult`` glyph-widths,
  * span is capped at ``max_span``.

These are PURE helpers (geometry + the shared dangling-connective string
heuristics); no model/service dependency, so they unit-test in isolation.
"""
from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Sequence

from app.utils.sentence_merge import (
    MergeGroup,
    SentenceMergePlan,
    has_dangling_connective,
    is_bare_sentence_final,
)

logger = logging.getLogger(__name__)


def _norm(text: Optional[str]) -> str:
    return unicodedata.normalize("NFC", text or "").strip()


def _cx(block: Dict) -> float:
    return (float(block["minX"]) + float(block["maxX"])) / 2.0


def _cy(block: Dict) -> float:
    return (float(block["minY"]) + float(block["maxY"])) / 2.0


def _glyph_width(block: Dict) -> float:
    """Approx glyph/column width = the SHORT axis of the box.

    A vertical-Japanese column is one glyph wide and many tall, so its short
    axis is the column width — the natural unit for "one column step".
    """
    w = float(block["maxX"]) - float(block["minX"])
    h = float(block["maxY"]) - float(block["minY"])
    return max(min(w, h), 1.0)


def _y_overlap_frac(a: Dict, b: Dict) -> float:
    """Fraction of the SHORTER box's height that the two Y-ranges share."""
    top = max(float(a["minY"]), float(b["minY"]))
    bot = min(float(a["maxY"]), float(b["maxY"]))
    inter = bot - top
    if inter <= 0:
        return 0.0
    ha = float(a["maxY"]) - float(a["minY"])
    hb = float(b["maxY"]) - float(b["minY"])
    shorter = min(ha, hb)
    return inter / shorter if shorter > 0 else 0.0


def _adjacent_columns(
    a: Dict,
    b: Dict,
    *,
    glyph_mult: float,
    y_overlap_min: float,
    directional: bool = False,
    max_width_ratio: Optional[float] = None,
) -> bool:
    """Geometric "side-by-side columns of one balloon" test.

    True when ``a`` and ``b`` are within ``glyph_mult`` glyph-widths horizontally
    AND their Y-ranges overlap by at least ``y_overlap_min`` of the shorter box.

    ``directional`` (column-grouping path): in a vertical RIGHT-TO-LEFT balloon
    each successive reading-order column sits to the LEFT of the previous, so a
    candidate ``b`` whose center is significantly to the RIGHT of ``a`` is almost
    certainly a DIFFERENT balloon (the reading-order sort lumped two overlapping
    balloons into one band). Rejecting the rightward jump removes the dominant
    geometric over-merge (e.g. two distinct utterances that merely overlap in Y)
    while keeping every genuine left-stepping multi-column balloon.

    ``max_width_ratio`` (membership path): two adjacent COLUMNS of one balloon are
    each ~one glyph wide (the font size is constant within a balloon), so their
    short-axis widths are near-equal. A box whose width is more than
    ``max_width_ratio`` x the other's is therefore NOT a sibling column — most
    often a fat SFX glyph or a horizontal label that merely Y-overlaps the
    dialogue. Rejecting the width mismatch stops a wide SFX from being absorbed as
    a "column" (the no-X-guard membership bug). It also stops a wide box from
    inflating the X-step tolerance (``glyph_mult * max(gw)``) so two genuinely
    separated columns are not bridged.
    """
    gw_a, gw_b = _glyph_width(a), _glyph_width(b)
    if max_width_ratio is not None:
        wide, narrow = max(gw_a, gw_b), min(gw_a, gw_b)
        if narrow <= 0 or wide / narrow > max_width_ratio:
            return False
    gw = max(gw_a, gw_b)
    if abs(_cx(a) - _cx(b)) > glyph_mult * gw:
        return False
    if directional and _cx(b) > _cx(a) + 0.5 * gw:
        return False
    return _y_overlap_frac(a, b) >= y_overlap_min


def bubble_id_of(block: Dict, bubbles: Optional[Sequence[Dict]]) -> Optional[int]:
    """Index of the smallest bubble interior whose box contains ``block``'s center.

    Mirrors the membership half of ``match_blocks_to_bubbles`` (smallest /
    tightest enclosure wins) but WITHOUT the typeset-only ``min_expand`` area
    gate — here we only want to know which balloon a fragment belongs to.
    Returns ``None`` when no bubble contains the center (SFX over art, or no
    bubble detector).
    """
    if not bubbles:
        return None
    cx, cy = _cx(block), _cy(block)
    best_i: Optional[int] = None
    best_area: Optional[float] = None
    for i, bub in enumerate(bubbles):
        ux0, uy0, ux1, uy1 = bub["minX"], bub["minY"], bub["maxX"], bub["maxY"]
        if not (ux0 <= cx <= ux1 and uy0 <= cy <= uy1):
            continue
        area = (float(ux1) - float(ux0)) * (float(uy1) - float(uy0))
        if best_area is None or area < best_area:
            best_i, best_area = i, area
    return best_i


def _same_column_union(a: Dict, b: Dict, *, tol: float) -> bool:
    """X-center proximity test for two (group-union) boxes — same vertical band."""
    return abs(_cx(a) - _cx(b)) <= tol


def _union_block(blocks: Sequence[Dict], positions: Sequence[int]) -> Dict:
    return {
        "minX": min(float(blocks[p]["minX"]) for p in positions),
        "minY": min(float(blocks[p]["minY"]) for p in positions),
        "maxX": max(float(blocks[p]["maxX"]) for p in positions),
        "maxY": max(float(blocks[p]["maxY"]) for p in positions),
    }


def _bubble_implausibly_large(
    bubble: Dict, a: Dict, b: Dict, *, area_mult: float
) -> bool:
    """True when ``bubble`` is a PANEL-sized container, not a tight balloon.

    The union bbox of two column-adjacent fragments is a STABLE fraction of even a
    6-column balloon (~1/3 of its area: two of N columns spanning the full height),
    so a genuine balloon is only a few x that union. A panel-sized YOLO region that
    happens to enclose the fragments is many TENS of times larger. When the bubble
    area exceeds ``area_mult`` x the joined-fragment union it is treated as a panel
    and must NOT authorise membership fusion of its contents (the conservative
    principle: leaving a split is better than fusing two distinct balloons that a
    loose detector lumped into one box).
    """
    try:
        bub_area = (float(bubble["maxX"]) - float(bubble["minX"])) * (
            float(bubble["maxY"]) - float(bubble["minY"])
        )
    except (KeyError, TypeError, ValueError):
        return False
    u = _union_block([a, b], [0, 1])
    u_area = (u["maxX"] - u["minX"]) * (u["maxY"] - u["minY"])
    if u_area <= 0:
        return False
    return bub_area > area_mult * u_area


def group_columns_into_bubbles(
    page_context_lines: Sequence[str],
    blocks: Sequence[Dict],
    *,
    bubbles: Optional[Sequence[Dict]] = None,
    max_span: int = 6,
    glyph_mult: float = 1.8,
    y_overlap_min: float = 0.30,
    membership_y_overlap_min: float = 0.50,
    membership_width_ratio: float = 2.2,
    membership_bubble_area_mult: float = 8.0,
    fuse_dangling: bool = False,
) -> SentenceMergePlan:
    """Group the column-fragments of one balloon into single translation units.

    ``page_context_lines`` and ``blocks`` are PARALLEL lists already in PAGE
    READING ORDER (column-major RTL, top-to-bottom — the output of
    ``orphan_lines.reading_order_sort``). Reading-order adjacency is therefore
    index adjacency.

    Two (up to ``max_span``) strictly-adjacent fragments fuse when:
      * they share a CTD parent bubble that is NOT panel-sized relative to them
        AND they pass the column-adjacency geometry (X-proximity, similar glyph
        width, ``membership_y_overlap_min`` Y-overlap, RTL directional step) —
        membership is NECESSARY but not sufficient, it replaces the fallback's
        geometry GUESS without bypassing its GUARDS, OR
      * (geometric fallback, no/!partial bubble) they are column-adjacent and
        Y-overlap by ``y_overlap_min``.

    When ``fuse_dangling`` is set, a SECOND bounded pass fuses two adjacent
    balloon GROUPS that form one cross-bubble sentence (the lead group dangles on
    a connective or the trailing group is a bare sentence-final particle, and the
    two union boxes sit in the same vertical band) — this preserves the existing
    cross-bubble sentence-continuation behaviour on top of the column grouping.

    Returns a :class:`SentenceMergePlan` whose ``groups`` cover EVERY page
    position exactly once (a solo fragment is a single-member group), so callers
    can wire it through ``build_merged_translation_request`` unconditionally. The
    grouped JP (fragments concatenated in reading order) is each group's
    ``merged_text``; the lead (first reading-order) member renders the EN.
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

    bids = [bubble_id_of(blocks[i], bubbles) for i in range(n)]

    def _can_join(j: int, k: int) -> bool:
        a, b = blocks[j], blocks[k]
        ba, bb = bids[j], bids[k]
        if ba is not None and bb is not None:
            # MEMBERSHIP mode. Same balloon is NECESSARY but NOT sufficient: at
            # scale many DISTINCT utterances (and SFX) map to ONE loose / panel
            # box, so a same-id + weak-Y-overlap test fused them and the v11 model
            # blanked the rest (Ikenie4 omissions 44 -> 385). Require the SAME
            # same-balloon-column geometry the no-bubble fallback uses — column
            # adjacency, similar glyph width, STRONG Y-overlap, RTL directional
            # step — so membership replaces the fallback's geometry GUESS without
            # bypassing its GUARDS. This is strictly TIGHTER than the old test and
            # makes chaining safe (each consecutive pair must itself be a sibling
            # column, so A-B-C cannot bridge two distinct utterances).
            if ba != bb:
                return False
            # PANEL guard: an implausibly large container is not a balloon and
            # must not authorise fusing its contents.
            if _bubble_implausibly_large(
                bubbles[ba], a, b, area_mult=membership_bubble_area_mult
            ):
                return False
            return _adjacent_columns(
                a, b,
                glyph_mult=glyph_mult,
                y_overlap_min=membership_y_overlap_min,
                directional=True,
                max_width_ratio=membership_width_ratio,
            )
        # Geometric fallback (at least one fragment matched no bubble). RTL
        # directional guard rejects a candidate column to the RIGHT (a different
        # balloon the reading-order sort merged into the band).
        return _adjacent_columns(
            a, b, glyph_mult=glyph_mult, y_overlap_min=y_overlap_min,
            directional=True,
        )

    used = [False] * n
    raw_groups: List[List[int]] = []
    for i in range(n):
        if used[i]:
            continue
        members = [i]
        j = i
        while len(members) < max_span and j + 1 < n and not used[j + 1]:
            if not _can_join(j, j + 1):
                break
            members.append(j + 1)
            j += 1
        for m in members:
            used[m] = True
        raw_groups.append(members)

    if fuse_dangling and len(raw_groups) > 1:
        raw_groups = _fuse_dangling_groups(
            raw_groups, page_context_lines, blocks, max_span=max_span
        )

    for members in raw_groups:
        merged_text = "".join(_norm(page_context_lines[m]) for m in members)
        gidx = len(plan.groups)
        plan.groups.append(MergeGroup(list(members), merged_text))
        for m in members:
            plan.position_to_group[m] = gidx
    plan.merged_page_lines = [g.merged_text for g in plan.groups]
    return plan


def _fuse_dangling_groups(
    raw_groups: List[List[int]],
    page_context_lines: Sequence[str],
    blocks: Sequence[Dict],
    *,
    max_span: int,
) -> List[List[int]]:
    """Second pass: fuse two adjacent balloon groups that form one sentence.

    A cross-bubble sentence (e.g. a dangling connective in the upper balloon and
    its predicate in the balloon below) is fused when the upper group's text
    dangles on a connective OR the lower group is a bare sentence-final particle,
    AND the two union boxes share a vertical band AND the lower group reads below
    the upper. Bounded: only consecutive groups, combined member count capped at
    ``max_span`` (so two small balloons fuse, two big multi-column balloons do
    not).
    """
    try:
        page_w = max(float(b["maxX"]) for b in blocks) - min(
            float(b["minX"]) for b in blocks
        )
    except (ValueError, KeyError):
        page_w = 0.0
    tol = max(40.0, page_w * 0.06)

    out: List[List[int]] = []
    gi = 0
    while gi < len(raw_groups):
        cur = list(raw_groups[gi])
        while gi + 1 < len(raw_groups):
            nxt = raw_groups[gi + 1]
            if len(cur) + len(nxt) > max_span:
                break
            cur_u = _union_block(blocks, cur)
            nxt_u = _union_block(blocks, nxt)
            if not _same_column_union(cur_u, nxt_u, tol=tol):
                break
            if _cy(nxt_u) < _cy(cur_u):  # continuation must read below the lead
                break
            cur_text = "".join(_norm(page_context_lines[m]) for m in cur)
            nxt_text = "".join(_norm(page_context_lines[m]) for m in nxt)
            if not (
                has_dangling_connective(cur_text) or is_bare_sentence_final(nxt_text)
            ):
                break
            cur = cur + list(nxt)
            gi += 1
        out.append(cur)
        gi += 1
    return out


# --- P2.1: empty-bubble backfill ---------------------------------------------

def select_backfill_targets(
    kept_texts: Sequence[str],
    translations: Sequence[Optional[str]],
    kept_confs: Optional[Sequence[Optional[float]]],
    bubble_resplit: Optional[Sequence[tuple]],
    *,
    is_japanese_fn: Callable[[str], bool],
    conf_threshold: float = 0.65,
    lead_truncation_ratio: float = 0.5,
) -> List[int]:
    """Indices into the kept list that should be re-translated via the plain path.

    UNCONDITIONAL BACKFILL FLOOR for the consolidation-omission symptom: the
    marked page-context call sometimes blanks a KEPT high-confidence dialogue
    bubble (it folded the sentence onto a neighbour). Such a bubble is recovered
    with a deterministic single-line plain translate. A kept bubble qualifies
    when ALL hold:

      * its current translation is empty / whitespace,
      * its JP source is non-empty and passes the Japanese filter,
      * its OCR confidence is >= ``conf_threshold`` (``None`` treated as high —
        no confidence info means we do not suppress), and
      * it is NOT a legitimate merge continuation WHOSE LEAD CARRIES THE FULL
        SENTENCE. ``bubble_resplit`` (from
        :func:`build_merged_translation_request`) marks each kept bubble
        ``(req_idx, is_lead)``; a continuation (``is_lead`` False) is normally
        blank (its EN is on the lead bubble), so it is left alone.

    THE FLOOR. In production EVERY blanked dialogue line is marked an (is_lead
    False) continuation, so a blanket continuation-skip suppresses the safety net
    entirely (it fired zero times). So the skip is qualified: a continuation is
    only left blank when its lead actually carries the sentence. When the lead EN
    is suspiciously SHORT relative to the fused JP (see
    :func:`_lead_appears_truncated`) the lead was truncated and dropped this tail,
    so the continuation is recovered with a standalone translate instead of being
    silently omitted. A bare (<2-char) sentence-final particle continuation is
    never revived (too trivial to translate out of context).

    Pure: returns indices; the caller issues the (async) plain translate so this
    stays model-free and unit-testable.
    """
    targets: List[int] = []
    nconf = len(kept_confs) if kept_confs is not None else 0
    nrs = len(bubble_resplit) if bubble_resplit is not None else 0
    for i, en in enumerate(translations):
        if en is not None and str(en).strip():
            continue  # already has a translation
        jp = kept_texts[i] if i < len(kept_texts) else ""
        if not _norm(jp) or not is_japanese_fn(jp):
            continue
        # Merge continuation: normally blank (its EN is on the lead bubble). Only
        # recover it when the lead was TRUNCATED (lead EN too short vs fused JP)
        # AND this continuation carries >=2 JP chars worth recovering standalone.
        if bubble_resplit is not None and i < nrs and not bubble_resplit[i][1]:
            if not (
                len(_norm(jp)) >= 2
                and _lead_appears_truncated(
                    i, translations, kept_texts, bubble_resplit,
                    ratio=lead_truncation_ratio,
                )
            ):
                continue
        conf = kept_confs[i] if (kept_confs is not None and i < nconf) else None
        if conf is not None and conf < conf_threshold:
            continue
        targets.append(i)
    return targets


def _lead_appears_truncated(
    i: int,
    translations: Sequence[Optional[str]],
    kept_texts: Sequence[str],
    bubble_resplit: Sequence[tuple],
    *,
    ratio: float,
) -> bool:
    """True when the lead of ``i``'s merge group carries far less EN than its
    fused JP implies — a TRUNCATION signal (the tail, this continuation, was
    dropped rather than legitimately folded into the lead).

    Compares the lead bubble's EN char count to the GROUP's total JP char count.
    English of a Japanese sentence is normally AT LEAST the JP char count (kanji
    pack dense content), so a lead under ``ratio`` x the fused-JP length has
    almost certainly lost content. A blank lead (whole group dropped) counts as
    truncated. Conservative on purpose: ``ratio`` defaults low (0.5) so a terse
    but COMPLETE lead is not mistaken for a truncated one (which would create a
    duplicate render across a cross-bubble merge).
    """
    req_idx = bubble_resplit[i][0]
    fused_jp_len = 0
    lead_en: Optional[str] = None
    for k, rs in enumerate(bubble_resplit):
        if rs[0] != req_idx:
            continue
        if k < len(kept_texts):
            fused_jp_len += len(_norm(kept_texts[k]))
        if rs[1]:  # lead member of this group
            lead_en = translations[k] if k < len(translations) else None
    if fused_jp_len <= 0:
        return False
    lead_len = len(_norm_en(lead_en)) if lead_en else 0
    return lead_len < ratio * fused_jp_len


# --- P2.2: adjacent identical-EN de-dup --------------------------------------

def _norm_en(text: Optional[str]) -> str:
    """Lowercase, collapse whitespace, drop trailing terminal punctuation."""
    s = unicodedata.normalize("NFC", text or "").strip().lower()
    s = " ".join(s.split())
    while s and s[-1] in ".!?…。、,":
        s = s[:-1].rstrip()
    return s


def dedup_adjacent_identical(
    translations: Sequence[Optional[str]],
    blocks: Sequence[Dict],
    *,
    glyph_mult: float = 2.2,
    y_overlap_min: float = 0.25,
    min_chars: int = 8,
) -> List[Optional[str]]:
    """Collapse adjacent same-balloon bubbles whose EN is identical / substring.

    SAFETY NET for the duplicated-text symptom: when P1 did NOT group two
    column-fragments of one balloon, the model can independently reconstruct the
    WHOLE sentence on each, so adjacent bubbles render the SAME English. This
    mirrors the merge contract — keep the FULL EN on the lead (earlier
    reading-order) bubble and blank the continuation.

    A pair ``(i, i+1)`` collapses when their blocks are column-adjacent (same
    balloon geometry) AND both normalized EN are non-empty, >= ``min_chars``, and
    equal OR one contains the other. The longer text is kept on the lead bubble;
    the other is blanked. Conservative: short interjections (``< min_chars``) and
    non-adjacent / empty bubbles are never touched.

    Returns a NEW list 1:1 with ``translations``.
    """
    out: List[Optional[str]] = list(translations)
    n = min(len(out), len(blocks))
    for i in range(n - 1):
        a_en, b_en = out[i], out[i + 1]
        na, nb = _norm_en(a_en), _norm_en(b_en)
        if not na or not nb:
            continue
        if len(na) < min_chars or len(nb) < min_chars:
            continue
        if not (na == nb or na in nb or nb in na):
            continue
        if not _adjacent_columns(
            blocks[i], blocks[i + 1], glyph_mult=glyph_mult, y_overlap_min=y_overlap_min
        ):
            continue
        # Keep the FULL (longer) EN on the lead bubble; blank the continuation.
        full = a_en if len(na) >= len(nb) else b_en
        out[i] = full
        out[i + 1] = ""
    return out


# --- P2.3: bubble-keyed final dedup ("1 balloon = 1 string") ------------------

def _block_area(block: Dict) -> float:
    w = max(0.0, float(block["maxX"]) - float(block["minX"]))
    h = max(0.0, float(block["maxY"]) - float(block["minY"]))
    return w * h


def _en_token_overlap(a: Optional[str], b: Optional[str]) -> float:
    """Jaccard token overlap of two normalized ENs (0.0-1.0).

    Cheap divergence proxy — no external dep. Word-set intersection over union
    on :func:`_norm_en` tokens. Used by :func:`en_near_duplicate` to tell the
    TRUE dedup case (the model reconstructed the same utterance on each
    column-fragment => high overlap) from a genuinely DISTINCT sibling line
    (low overlap => its content must be preserved, not blanked).
    """
    na, nb = _norm_en(a), _norm_en(b)
    if not na or not nb:
        return 0.0
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def en_near_duplicate(
    a: Optional[str], b: Optional[str], *, min_overlap: float = 0.6
) -> bool:
    """True when two ENs are near-duplicates (the TRUE in-balloon dup case).

    Near-duplicate == the surface strings are equal, one contains the other
    (mirrors :func:`dedup_adjacent_identical`'s substring test), OR their token
    overlap (:func:`_en_token_overlap`) is at least ``min_overlap``. Anything
    below that is a MEANINGFULLY DIVERGENT sibling: distinct content the balloon
    must keep (via the fused-balloon retranslate), not silently drop.

    An empty side is treated as a near-duplicate (nothing to preserve).
    """
    na, nb = _norm_en(a), _norm_en(b)
    if not na or not nb:
        return True
    if na == nb or na in nb or nb in na:
        return True
    return _en_token_overlap(a, b) >= min_overlap


@dataclass
class FusedBalloonGroup:
    """One multi-block balloon whose blanked siblings DIVERGE from the winner.

    ``winner`` is the kept-list index that renders the fused EN (largest-area
    block — same choice :func:`dedup_by_bubble` keeps). ``members`` are ALL the
    balloon's non-empty kept-list indices in READING ORDER (ascending index ==
    page reading order), used to build the fused JP. ``blanked`` are the sibling
    indices (members minus winner) that stay blank after the fused EN lands on
    the winner.
    """

    winner: int
    members: List[int]
    blanked: List[int]


@dataclass
class BubbleDedupPlan:
    """Result of :func:`plan_bubble_dedup`.

    ``deduped`` is the 1:1 translations list AFTER blanking all-but-winner per
    balloon — byte-identical to the legacy :func:`dedup_by_bubble` output.
    ``retranslate`` lists the balloons whose siblings meaningfully DIVERGE from
    the winner (distinct content, not near-duplicates); the caller issues ONE
    fused marked call per group to preserve their content. When
    ``translation_balloon_fused_retranslate`` is off (or no OCR text was
    threaded) the caller simply ignores ``retranslate`` and behavior == legacy.

    ``effective_jp`` (populated by :func:`apply_fused_balloon_retranslate`) maps a
    winner's kept-list index to the FUSED JP its rewritten EN now covers, so the
    caller's post-edit over-expansion gate compares the fused EN against the fused
    JP (not the winner's short single fragment — which would false-positive and
    blank a faithful translation). ``retranslate_errors`` counts fused calls that
    raised (each logged) so callers can surface the failure count.
    """

    deduped: List[Optional[str]] = field(default_factory=list)
    retranslate: List[FusedBalloonGroup] = field(default_factory=list)
    effective_jp: Dict[int, str] = field(default_factory=dict)
    retranslate_errors: int = 0


def plan_bubble_dedup(
    translations: Sequence[Optional[str]],
    blocks: Sequence[Dict],
    bubbles: Optional[Sequence[Dict]],
    *,
    near_duplicate: Optional[Callable[[Optional[str], Optional[str]], bool]] = None,
) -> BubbleDedupPlan:
    """Plan the "1 balloon = 1 string" dedup AND flag divergent balloons.

    Same bucketing / winner selection as :func:`dedup_by_bubble` (see its
    docstring). Additionally, for each multi-block balloon it classifies the
    blanked siblings: when EVERY sibling is a near-duplicate of the winner
    (``near_duplicate``, default :func:`en_near_duplicate`) the balloon is a true
    duplication and only ``deduped`` is affected. When ANY sibling meaningfully
    DIVERGES, the balloon is added to ``retranslate`` so the caller can fuse the
    members' JP and retranslate as one unit (content preservation) — the
    root-cause fix for the multi-column-balloon content drop.

    Pure planner: no model / OCR dependency. The fused retranslate itself is a
    caller-side pass (:func:`apply_fused_balloon_retranslate`) with the translator
    injected, so this module never imports the translator.
    """
    check = near_duplicate or en_near_duplicate
    out: List[Optional[str]] = list(translations)
    plan = BubbleDedupPlan(deduped=out)
    if not bubbles:
        return plan
    n = min(len(out), len(blocks))
    buckets: Dict[int, List[int]] = {}
    for i in range(n):
        bid = bubble_id_of(blocks[i], bubbles)
        if bid is None:
            continue
        buckets.setdefault(bid, []).append(i)
    for positions in buckets.values():
        nonempty = [
            p for p in positions if out[p] is not None and str(out[p]).strip()
        ]
        if len(nonempty) <= 1:
            continue
        # Winner: largest-area block (the bubble renderer), then longest EN, then
        # earliest reading-order position. ``-p`` makes ``max`` prefer the
        # smaller index on a full tie.
        winner = max(
            nonempty,
            key=lambda p: (_block_area(blocks[p]), len(_norm_en(out[p])), -p),
        )
        # Classify BEFORE blanking (need the siblings' EN to compare).
        siblings = [p for p in nonempty if p != winner]
        diverges = any(not check(out[winner], out[p]) for p in siblings)
        if diverges:
            plan.retranslate.append(
                FusedBalloonGroup(
                    winner=winner,
                    members=sorted(nonempty),  # ascending index == reading order
                    blanked=sorted(siblings),
                )
            )
        for p in siblings:
            out[p] = ""
    return plan


def dedup_by_bubble(
    translations: Sequence[Optional[str]],
    blocks: Sequence[Dict],
    bubbles: Optional[Sequence[Dict]],
) -> List[Optional[str]]:
    """Final "1 balloon = 1 string" dedup keyed on CTD speech-bubble membership.

    Buckets every kept block by the speech balloon it sits in (:func:`bubble_id_of`
    — the smallest enclosing detected bubble). Within each balloon keep exactly
    ONE non-empty EN — the bubble-matched / longest winner (largest-area block,
    then longest EN, then earliest reading order) — and blank the rest. The
    largest-area tie-break MIRRORS :func:`match_blocks_to_bubbles`, which typesets
    the bubble interior to the largest block, so the surviving EN lands on the box
    that actually renders the balloon.

    Unlike :func:`dedup_adjacent_identical` this is INDEPENDENT of adjacency,
    orientation, length and string equality: a speech balloon holds exactly one
    utterance, so any extra EN the v11 model reconstructed on a sibling
    column-fragment is a duplication and is removed even when the surface strings
    DIVERGE (the "ghosted"/duplicated-EN symptom the narrow adjacency dedup
    misses). It therefore SUPERSEDES the adjacency dedup for the in-balloon case;
    :func:`dedup_adjacent_identical` remains the fallback when no bubble detector
    ran.

    Blocks whose center falls in NO detected bubble (SFX over art) keep
    ``bubble_id`` ``None`` and are left untouched. Returns a NEW list 1:1 with
    ``translations``; a no-op copy when no bubbles were detected.

    NOTE: this blanks divergent siblings unconditionally (the legacy contract).
    When their content must be PRESERVED, the caller uses :func:`plan_bubble_dedup`
    + :func:`apply_fused_balloon_retranslate` instead (gated by
    ``translation_balloon_fused_retranslate``). This function is kept as the
    thin, dependency-free path (and for callers/tests that only want the blank).
    """
    return plan_bubble_dedup(translations, blocks, bubbles).deduped


async def apply_fused_balloon_retranslate(
    translations: Sequence[Optional[str]],
    kept_texts: Sequence[str],
    plan: "BubbleDedupPlan",
    page_context_lines: Sequence[str],
    target_positions: Sequence[int],
    marked_translate: Callable[
        [List[str], List[int], str], Awaitable[List[str]]
    ],
    *,
    target_language: str = "English",
) -> List[Optional[str]]:
    """Preserve divergent balloons' content via ONE fused marked call each.

    For every balloon in ``plan.retranslate`` the members' OCR JP is fused (joined
    in reading order — mirrors :class:`~app.utils.sentence_merge.MergeGroup`'s
    ``merged_text``) into a SINGLE line that REPLACES the members in an otherwise
    unchanged copy of ``page_context_lines`` (the lead member's page slot holds
    the fused JP, the other member slots are dropped — the same re-segmentation
    :func:`build_merged_translation_request` performs). ``marked_translate`` is
    the injected page-context translator (``translate_page_context_marked``), so
    the call goes through the SAME ``build_v11_context_prompt`` template as every
    other call — no new prompt. The returned fused EN lands on the winner block;
    the siblings stay blank (they are intentionally-blank continuations, the
    sentence_merge contract).

    ``translations`` is the ALREADY-deduped list (``plan.deduped``); this returns
    a NEW list 1:1 with it, with each divergent balloon's winner overwritten by
    the fused EN. Cost is bounded: one marked call per divergent balloon.

    SIDE EFFECT: for every winner actually overwritten with a fused EN this records
    ``plan.effective_jp[winner] = fused_jp`` so the caller's post-edit over-expansion
    gate sees the FUSED JP the EN now covers (using the winner's short single
    fragment JP would false-positive and blank the faithful fused translation —
    silently re-introducing the very content drop this pass exists to fix).
    """
    out: List[Optional[str]] = list(translations)
    if not plan.retranslate:
        return out
    n_ctx = len(page_context_lines)
    for group in plan.retranslate:
        # Page positions of the members (target_positions maps kept-idx -> page
        # position in page_context_lines). Guard against any misalignment.
        try:
            member_page_pos = sorted(target_positions[k] for k in group.members)
        except (IndexError, TypeError):
            continue
        if not member_page_pos or any(p >= n_ctx for p in member_page_pos):
            continue
        fused_jp = "".join(
            _norm_en_source(kept_texts[k])
            for k in group.members
            if k < len(kept_texts)
        )
        if not fused_jp.strip():
            continue
        lead_pp = member_page_pos[0]
        drop = set(member_page_pos[1:])
        merged_ctx: List[str] = []
        fused_target: Optional[int] = None
        for pp in range(n_ctx):
            if pp in drop:
                continue
            if pp == lead_pp:
                fused_target = len(merged_ctx)
                merged_ctx.append(fused_jp)
            else:
                line = page_context_lines[pp]
                merged_ctx.append(line if line is not None else "")
        if fused_target is None:
            continue
        try:
            marked = await marked_translate(
                merged_ctx, [fused_target], target_language
            )
        except Exception as exc:
            # Retranslate is a best-effort content-recovery pass; on failure keep
            # the deduped (winner-only) output rather than crash the page. Never
            # swallow silently: log the original exception + traceback and count
            # it so callers can surface the failure rate.
            plan.retranslate_errors += 1
            logger.warning(
                "fused-balloon retranslate failed for winner %d (members=%s): %s",
                group.winner, group.members, exc, exc_info=True,
            )
            continue
        if marked and marked[0] and str(marked[0]).strip():
            out[group.winner] = marked[0]
            # Record the EFFECTIVE JP the winner's rewritten EN now covers so the
            # caller's post-edit over-expansion gate compares against the fused JP.
            plan.effective_jp[group.winner] = fused_jp
            for p in group.blanked:
                out[p] = ""
    return out


def _norm_en_source(text: Optional[str]) -> str:
    """NFC-normalize + strip a JP source line (mirrors sentence_merge._norm)."""
    return unicodedata.normalize("NFC", text or "").strip()

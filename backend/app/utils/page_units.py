"""Shared gate -> page-context assembly for the manga-translation pipeline.

THREE hand-maintained copies of the same "filter / gate / build whole-page v11
context" logic had drifted apart:

  (a) ``app.routers.translate.process_single_image`` PIPELINED branch — the LIVE
      path (``use_pipeline_overlap`` defaults True and pages have >1 crop),
  (b) the same function's BATCH else-branch (effectively dead for live traffic),
  (c) ``scripts.batch_translate_chapter._process`` — the eval path.

The drift caused real bugs: the OCR-confidence threading fix had to be applied
THREE times (and was missing from copy (a)), and ``is_leave_intact_label`` lived
ONLY in copy (c) — the live pipelined path leaked editorial margin labels.

``build_page_translation_units`` is the ONE shared, PURE (no model / no I/O)
data-shaping helper. Each caller does its own OCR (with its own timing /
overlap) and then hands the already-OCR'd parallel lists here to get the kept
subset, the whole-page v11 context, the target positions, and the inpaint-only
erase blocks — identical decisions on every path.

The helper also wires in the cross-bubble sentence-continuation merge
(``app.utils.sentence_merge``): once it has the page in reading order it
re-segments dangling-connective continuations into single translation units, so
ALL THREE callers benefit. The merge is OPTIONAL and gated by ``settings`` so it
can be turned off without touching call sites.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.utils.ocr_confidence_gate import (
    is_dialogue_context_candidate,
    is_garbled_low_conf,
    should_erase_dropped,
)
from app.utils.bubble_grouping import group_columns_into_bubbles
from app.utils.sentence_merge import (
    SentenceMergePlan,
    detect_sentence_continuations,
)


def _norm(text: Optional[str]) -> str:
    return unicodedata.normalize("NFC", text or "").strip()


@dataclass
class PageTranslationUnits:
    """Result of :func:`build_page_translation_units`.

    The core tuple every caller needs:
      * ``kept_blocks`` / ``kept_texts`` / ``kept_confs`` — the rendered subset,
        parallel and 1:1 (kept_confs holds the REAL OCR recognition confidence so
        the post-edit name-invention suppressor fires on EVERY path),
      * ``page_context_lines`` — the WHOLE page's dialogue (kept + dropped
        dialogue) in reading order, for the v11 page-context prompt,
      * ``target_positions`` — 0-based index of each kept line INTO
        ``page_context_lines`` (what ``translate_page_context_marked`` marks),
      * ``erase_only_blocks`` — gate-dropped blocks that are real JP ink to erase
        (inpaint-only; never rendered).

    ``kept_indices`` are the indices INTO the input parallel lists that were kept
    (so a caller can also subset other parallel lists, e.g. crops / regions).

    ``merge_plan`` (when sentence-merge ran) maps page positions to merged
    translation units; ``num_targets`` is the count actually sent to the model.
    """

    kept_blocks: List[Dict] = field(default_factory=list)
    kept_texts: List[str] = field(default_factory=list)
    kept_confs: List[Optional[float]] = field(default_factory=list)
    page_context_lines: List[str] = field(default_factory=list)
    target_positions: List[int] = field(default_factory=list)
    erase_only_blocks: List[Dict] = field(default_factory=list)
    kept_indices: List[int] = field(default_factory=list)
    # Editorial / margin labels (表紙用イラスト, 奥付, ...) the human reference
    # leaves as ORIGINAL art: never translated, never rendered, never erased.
    # Surfaced so callers can pass them to ``build_inpaint_mask`` and punch them
    # OUT of the erase mask now that ALL detected text ink is otherwise erased.
    leave_intact_blocks: List[Dict] = field(default_factory=list)
    merge_plan: Optional[SentenceMergePlan] = None

    def as_tuple(self):
        """The documented 6-tuple contract, for callers that prefer unpacking."""
        return (
            self.kept_blocks,
            self.kept_texts,
            self.kept_confs,
            self.page_context_lines,
            self.target_positions,
            self.erase_only_blocks,
        )


@dataclass
class MergedTranslationRequest:
    """Re-segmented page ready for ``translate_page_context_marked``.

    ``merged_page_lines`` is the page context AFTER cross-bubble merge (one entry
    per sentence unit, in reading order). ``merged_target_positions`` are the
    GROUP positions to translate (deduplicated — one request per kept sentence
    unit). ``bubble_resplit`` maps each ORIGINAL kept-bubble index (0-based into
    the caller's kept list) to ``(group_request_index, is_lead)``: the kept
    bubble pulls the model output for ``group_request_index`` and renders it only
    when ``is_lead`` (continuation bubbles are blanked).
    """

    merged_page_lines: List[str] = field(default_factory=list)
    merged_target_positions: List[int] = field(default_factory=list)
    # kept-bubble index -> (index into merged_target_positions output, is_lead)
    bubble_resplit: List[tuple] = field(default_factory=list)


def build_merged_translation_request(
    units: "PageTranslationUnits",
    *,
    continuation_filler: str = "",
) -> Optional[MergedTranslationRequest]:
    """Fold a merge plan into a single-unit-per-sentence translation request.

    Returns ``None`` when no merge plan is present (caller uses the un-merged
    ``page_context_lines`` / ``target_positions`` path unchanged). Otherwise it
    collapses each merged sentence group into ONE marked line, marks only the
    groups that contain a KEPT (rendered) bubble, and builds the per-kept-bubble
    re-split map so each kept bubble renders the merged English on its LEAD bubble
    and ``continuation_filler`` ("" by default) on the continuation bubbles.

    The output is aligned so that, given ``out = translate_page_context_marked(
    merged_page_lines, merged_target_positions, lang)``, the kept bubble ``k``
    renders ``out[group_request_index]`` when ``is_lead`` else
    ``continuation_filler``.
    """
    plan = units.merge_plan
    if plan is None:
        return None

    merged_page_lines = list(plan.merged_page_lines)

    # Each kept bubble lives at a page position (target_positions[k]) that belongs
    # to exactly one merged group. A group is a TARGET (sent to the model) iff it
    # contains at least one kept bubble. Build the dedup'd list of target groups.
    kept_page_positions = list(units.target_positions)  # one per kept bubble
    target_group_indices: List[int] = []
    group_to_request: Dict[int, int] = {}
    for p in kept_page_positions:
        g = plan.position_to_group[p]
        if g not in group_to_request:
            group_to_request[g] = len(target_group_indices)
            target_group_indices.append(g)

    # The LEAD page position of each group is its first member position; the kept
    # bubble whose page position == lead renders the EN, the rest are blanked.
    group_lead_position = {
        gi: g.lead_position for gi, g in enumerate(plan.groups)
    }

    bubble_resplit: List[tuple] = []
    for p in kept_page_positions:
        g = plan.position_to_group[p]
        req_idx = group_to_request[g]
        is_lead = p == group_lead_position[g]
        bubble_resplit.append((req_idx, is_lead))

    return MergedTranslationRequest(
        merged_page_lines=merged_page_lines,
        merged_target_positions=target_group_indices,
        bubble_resplit=bubble_resplit,
    )


def apply_resplit(
    merged_output: Sequence[str],
    bubble_resplit: Sequence[tuple],
    *,
    continuation_filler: str = "",
) -> List[str]:
    """Map per-group model output back to per-kept-bubble translations (1:1).

    ``merged_output[req_idx]`` is the English for sentence unit ``req_idx``; each
    kept bubble renders it on its LEAD bubble and ``continuation_filler`` on the
    continuation bubbles — so the result is 1:1 with the caller's kept list and
    each translation lands on its own bubble.
    """
    out: List[str] = []
    for req_idx, is_lead in bubble_resplit:
        if is_lead and req_idx < len(merged_output):
            out.append(merged_output[req_idx])
        else:
            out.append(continuation_filler)
    return out


def combined_effective_jp(
    dedup_plan: Any,
    merge_plan: Optional[SentenceMergePlan],
    target_positions: Sequence[int],
) -> Dict[int, str]:
    """kept-index -> the EFFECTIVE JP a kept bubble's rendered EN actually covers.

    Two independent re-segmentation passes make a kept bubble's EN span MORE JP
    than its own single OCR fragment. The post-edit over-expansion gate
    (``translation_postedit.gate_over_expansion``) compares EN word count against
    a budget derived from the JP source; fed the SHORT single fragment it
    false-positives on a faithful fused/merged EN and blanks it to ``...`` —
    silently re-introducing the content drop these passes exist to fix. This maps
    each affected kept bubble to the JP its EN really covers so the gate sees it:

      * SENTENCE-MERGE lead (``merge_plan``): a cross-bubble sentence's lead bubble
        renders the whole group's EN; its effective JP is the group ``merged_text``.
      * FUSED-BALLOON winner (``dedup_plan.effective_jp``, populated by
        :func:`~app.utils.bubble_grouping.apply_fused_balloon_retranslate`): the
        winner's EN covers the balloon's fused JP.

    Fused wins on the rare overlap (it is the later, wider rewrite). Both key
    spaces are the caller's kept-list index; ``target_positions`` maps kept-index
    to page position for the merge lookup.
    """
    eff: Dict[int, str] = {}
    if merge_plan is not None:
        for k, pp in enumerate(target_positions):
            g_idx = merge_plan.position_to_group.get(pp)
            if g_idx is None:
                continue
            g = merge_plan.groups[g_idx]
            if g.is_merged and pp == g.lead_position:
                eff[k] = g.merged_text
    if dedup_plan is not None:
        for k, jp in getattr(dedup_plan, "effective_jp", {}).items():
            eff[k] = jp
    return eff


def build_page_translation_units(
    blocks: Sequence[Dict],
    ocr_texts: Sequence[str],
    ocr_confs: Optional[Sequence[float]],
    text_lines: Optional[Sequence[Dict]],
    settings: Any,
    *,
    is_japanese_fn: Callable[[str], bool],
    is_leave_intact_fn: Optional[Callable[[str], bool]] = None,
    should_skip_as_english_fn: Optional[
        Callable[[Dict, Optional[Sequence[Dict]], str, Callable[[str], bool]], bool]
    ] = None,
    on_drop: Optional[Callable[[int, str, float, str], None]] = None,
    bubbles: Optional[Sequence[Dict]] = None,
) -> PageTranslationUnits:
    """Shape an OCR'd page into the kept subset + whole-page v11 context.

    ``blocks`` / ``ocr_texts`` / ``ocr_confs`` are PARALLEL lists already in PAGE
    READING ORDER (the caller must have run ``reading_order_sort`` first — the
    v11 page-context contract). ``text_lines`` feeds the English-early-exit
    geometry check. ``settings`` supplies the same feature flags all three
    callers read.

    Decision order (the union of the three former copies, canonicalised on the
    batch-script copy which had every feature):

      1. ``is_japanese_text`` filter (when ``japanese_filter_enabled``),
      2. ``is_leave_intact_label`` — editorial / margin label, left as original
         art (was MISSING from the live pipelined path — a real gap),
      3. ``should_skip_as_english`` — horizontal Latin region left intact,
      4. OCR-confidence garble gate (``is_garbled_low_conf``): dropped lines that
         are real JP ink go to ``erase_only_blocks`` (``should_erase_dropped``);
         dropped lines that are real DIALOGUE re-enter the page CONTEXT
         (``is_dialogue_context_candidate``) so a sentence partner is not
         orphaned — even though they are NOT rendered (#3 decouple
         drop-from-render vs drop-from-context).

    The kept + context-only lines are assembled into ``page_context_lines`` in
    reading order with ``target_positions`` indexing the kept lines, then the
    cross-bubble sentence-continuation merge re-segments danglers into single
    translation units (when ``translation_sentence_merge`` is on).

    The injected ``is_japanese_fn`` / ``is_leave_intact_fn`` /
    ``should_skip_as_english_fn`` keep this util free of the heavy
    service-module imports the callers already hold.

    ``on_drop(index, text, conf, reason)`` is an optional logging hook so each
    caller can keep its own log format.
    """
    n = len(ocr_texts)
    confs: List[float] = list(ocr_confs) if ocr_confs is not None else [1.0] * n
    # Defensive alignment so a short conf list never indexes out of range.
    if len(confs) < n:
        confs = confs + [1.0] * (n - len(confs))

    jp_filter_on = bool(getattr(settings, "japanese_filter_enabled", True))
    jp_min_ratio = getattr(settings, "japanese_filter_min_ratio", 0.5)
    jp_kata_max = getattr(settings, "japanese_filter_katakana_max_length", 6)
    english_exit_on = bool(getattr(settings, "english_early_exit_enabled", True))
    gate_on = bool(getattr(settings, "ocr_confidence_gate_enabled", False)) and (
        getattr(settings, "ocr_confidence_gate_threshold", 0.0) > 0
    )
    gate_thresh = getattr(settings, "ocr_confidence_gate_threshold", 0.65)
    whole_page = bool(getattr(settings, "translation_pagecontext_whole_page", True))

    kept_indices: List[int] = []
    context_indices: List[int] = []
    erase_only_blocks: List[Dict] = []
    leave_intact_blocks: List[Dict] = []

    def _emit(idx: int, text: str, conf: float, reason: str) -> None:
        if on_drop is not None:
            try:
                on_drop(idx, text, conf, reason)
            except Exception:
                pass

    for i in range(n):
        text = ocr_texts[i]
        conf = confs[i]
        block = blocks[i]

        # 1. Japanese filter — non-Japanese is left as original pixels.
        if jp_filter_on and not is_japanese_fn(text):
            _emit(i, text, conf, "non_japanese")
            continue

        # 2. Leave-intact editorial / margin label (表紙用イラスト, 奥付, ...):
        #    keep as original art — never translate / erase / typeset over.
        if is_leave_intact_fn is not None and is_leave_intact_fn(text):
            # Kept as original art (not translated / not erased), but surfaced so
            # the inpaint mask builder can punch this region OUT of the erase mask.
            leave_intact_blocks.append(block)
            _emit(i, text, conf, "leave_intact_label")
            continue

        # 3. English early-exit: horizontal Latin (non-Japanese) region left as
        #    ORIGINAL pixels (NOT erased, NOT context).
        if (
            english_exit_on
            and should_skip_as_english_fn is not None
            and should_skip_as_english_fn(block, text_lines, text, is_japanese_fn)
        ):
            _emit(i, text, conf, "english_early_exit")
            continue

        # 4. OCR-confidence garble gate.
        if gate_on and is_garbled_low_conf(text, conf, conf_threshold=gate_thresh):
            # Real JP ink we drop (not translate) must still be ERASED so raw
            # Japanese does not survive into the render — inpaint-only.
            if should_erase_dropped(text):
                erase_only_blocks.append(block)
            # #3 DECOUPLE drop-from-render vs drop-from-context: a dropped
            # DIALOGUE line still feeds the page CONTEXT (so its sentence partner
            # is not orphaned -> 'InNo' nonsense) even though it is not rendered.
            if whole_page and is_dialogue_context_candidate(text, ocr_confidence=conf):
                context_indices.append(i)
            _emit(i, text, conf, "ocr_gate_garbled")
            continue

        # Kept: rendered AND context.
        kept_indices.append(i)
        context_indices.append(i)

    # WHOLE-PAGE v11 context: numbered page = all context indices (kept + dropped
    # dialogue) in reading (block) order; targets = kept lines' positions in it.
    context_order = sorted(set(context_indices))
    page_context_lines = [_norm(ocr_texts[i]) for i in context_order]
    ctx_pos = {orig: p for p, orig in enumerate(context_order)}
    target_positions = [ctx_pos[i] for i in kept_indices]

    kept_blocks = [blocks[i] for i in kept_indices]
    kept_texts = [ocr_texts[i] for i in kept_indices]
    kept_confs = [confs[i] for i in kept_indices]

    units = PageTranslationUnits(
        kept_blocks=kept_blocks,
        kept_texts=kept_texts,
        kept_confs=kept_confs,
        page_context_lines=page_context_lines,
        target_positions=target_positions,
        erase_only_blocks=erase_only_blocks,
        kept_indices=kept_indices,
        leave_intact_blocks=leave_intact_blocks,
    )

    # Pre-translation re-segmentation. Operates on the page in reading order; the
    # context blocks are the page-context lines' source blocks, in the same order
    # as page_context_lines.
    #
    #   P1 (translation_bubble_grouping): group the column-fragments of ONE
    #   speech balloon into a single translation unit (root-cause fix for
    #   over-segmented multi-column bubbles -> omissions / duplicated EN /
    #   render clutter). Prefers CTD parent-bubble membership (``bubbles``),
    #   falls back to geometric column adjacency. When sentence-merge is ALSO on,
    #   a bounded second pass fuses adjacent balloon GROUPS that form one
    #   cross-bubble sentence (preserving the #2 behaviour).
    #
    #   #2 (translation_sentence_merge ONLY, P1 off): the original cross-bubble
    #   dangling-continuation merge, unchanged.
    #
    # Both emit a SentenceMergePlan consumed identically by
    # build_merged_translation_request / apply_resplit.
    bubble_group_on = getattr(settings, "translation_bubble_grouping", False)
    sentence_merge_on = getattr(settings, "translation_sentence_merge", False)
    if (bubble_group_on or sentence_merge_on) and len(page_context_lines) > 1:
        context_blocks = [blocks[i] for i in context_order]
        if bubble_group_on:
            plan = group_columns_into_bubbles(
                page_context_lines,
                context_blocks,
                bubbles=bubbles,
                fuse_dangling=sentence_merge_on,
            )
        else:
            plan = detect_sentence_continuations(page_context_lines, context_blocks)
        if plan.num_merges > 0:
            units.merge_plan = plan

    return units

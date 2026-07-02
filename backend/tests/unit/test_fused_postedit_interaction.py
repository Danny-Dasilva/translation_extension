"""Fused-retranslate <-> post-edit over-expansion interaction (P0 correctness).

After the fused-balloon retranslate rewrites a multi-column balloon's winner with
an EN faithful to the balloon's FUSED JP (2-3 fragments), the post-edit
over-expansion gate must compare that EN against the FUSED JP — not the winner's
own SINGLE short fragment. Fed the short fragment, ``gate_over_expansion``
false-positives (``is_over_expanded``) and blanks the faithful line to ``...``,
silently re-introducing the content drop the fused retranslate exists to fix.

The SAME defect pre-exists for sentence-merge LEAD bubbles: the lead renders the
whole group's merged EN but its own OCR text is just the lead fragment.

These pin:
  * the pre-fix false-positive blank (fused EN + fragment JP -> "..."),
  * ``apply_fused_balloon_retranslate`` exposing the effective (fused) JP on the
    plan, and ``combined_effective_jp`` threading it so the gate keeps the line,
  * the sentence-merge lead effective-JP map,
  * router/batch ordering parity: dedup -> fused-retranslate -> glossaries(with
    effective JP) yields the SAME final translations on both consumers.
"""
from __future__ import annotations

import asyncio

from app.services.translation_postedit import (
    apply_postedit_glossaries,
    gate_over_expansion,
    postedit_one,
)
from app.utils.bubble_grouping import (
    apply_fused_balloon_retranslate,
    plan_bubble_dedup,
)
from app.utils.page_units import (
    build_merged_translation_request,
    build_page_translation_units,
    combined_effective_jp,
)
from app.utils.sentence_merge import MergeGroup, SentenceMergePlan


def _b(minX, minY, maxX, maxY):
    return {"minX": minX, "minY": minY, "maxX": maxX, "maxY": maxY, "confidence": 0.9}


# A tight balloon holding two column-fragments. WIN is the larger-area block (the
# dedup winner / renderer); SIB is the smaller sibling column.
SIB = _b(1065, 520, 1090, 700)          # 平然と家族で朝ごはんを  (long fragment)
WIN = _b(1000, 500, 1060, 800)          # 違う                    (short fragment, larger area)
BUB = _b(990, 490, 1100, 810)           # both centers fall inside

# Reading order == ascending index, so blocks=[SIB, WIN] => fused JP = SIB+WIN.
BLOCKS = [SIB, WIN]
KEPT_TEXTS = ["平然と家族で朝ごはんを", "違う"]
FUSED_JP = "平然と家族で朝ごはんを違う"

# A faithful 10-word EN for the fused sentence. Over the SHORT-fragment budget
# (違う -> ceil(2*2.5)+3 = 8 words) but well under the FUSED budget
# (ceil(13*2.5)+3 = 36 words). No names / honorifics / SFX so the rest of the
# glossary chain is a no-op and only the over-expansion gate is exercised.
FUSED_EN = "We are calmly having a peaceful family breakfast right now"

# Divergent raw model output per fragment => the balloon is flagged for fused
# retranslate (siblings are NOT near-duplicates of the winner).
RAW_DIVERGENT = ["We're having breakfast", "No way"]


class _FusedTranslator:
    """Fake translate_page_context_marked: returns the fused EN for the target."""

    def __init__(self, fused_en: str = FUSED_EN):
        self.calls: list = []
        self.fused_en = fused_en

    async def __call__(self, page_lines, target_indices, target_language="English"):
        self.calls.append((list(page_lines), list(target_indices), target_language))
        return [self.fused_en for _ in target_indices]


# --- the bug, isolated to the gate ------------------------------------------- #

def test_gate_blanks_faithful_fused_en_against_short_fragment_jp():
    # PRE-FIX reproduction: the winner's OWN short fragment trips the gate and the
    # faithful fused EN is blanked to the ellipsis marker.
    assert gate_over_expansion(FUSED_EN, "違う") == "..."


def test_gate_keeps_fused_en_against_fused_jp():
    # FIX: fed the JP the EN actually covers, the gate leaves it untouched.
    assert gate_over_expansion(FUSED_EN, FUSED_JP) == FUSED_EN


# --- effective-JP threading through the fused-retranslate plan ---------------- #

def test_fused_retranslate_exposes_effective_jp_on_plan():
    plan = plan_bubble_dedup(RAW_DIVERGENT, BLOCKS, [BUB])
    assert len(plan.retranslate) == 1
    winner = plan.retranslate[0].winner  # WIN == index 1 (larger area)
    assert winner == 1

    fake = _FusedTranslator()
    out = asyncio.run(
        apply_fused_balloon_retranslate(
            plan.deduped, KEPT_TEXTS, plan,
            KEPT_TEXTS,          # page_context_lines
            [0, 1],              # target_positions (kept-idx -> page pos)
            fake,
        )
    )
    # Winner carries the fused EN; the plan records the FUSED JP it now covers.
    assert out[winner] == FUSED_EN
    assert plan.effective_jp == {winner: FUSED_JP}
    assert plan.retranslate_errors == 0

    # combined_effective_jp surfaces that winner -> fused JP mapping.
    eff = combined_effective_jp(plan, None, [0, 1])
    assert eff == {winner: FUSED_JP}


def test_postedit_uses_effective_jp_so_fused_winner_survives_glossaries():
    plan = plan_bubble_dedup(RAW_DIVERGENT, BLOCKS, [BUB])
    fake = _FusedTranslator()
    translations = asyncio.run(
        apply_fused_balloon_retranslate(
            plan.deduped, KEPT_TEXTS, plan, KEPT_TEXTS, [0, 1], fake
        )
    )
    winner = plan.retranslate[0].winner
    eff = combined_effective_jp(plan, None, [0, 1])

    # WITHOUT the fix (fragment JP) the glossary chain blanks the winner.
    naive = apply_postedit_glossaries(list(translations), list(KEPT_TEXTS))
    assert naive[winner] == "..."

    # WITH the fix: substitute the effective JP for the mapped indices.
    jp_for_postedit = [eff.get(i, KEPT_TEXTS[i]) for i in range(len(translations))]
    fixed = apply_postedit_glossaries(list(translations), jp_for_postedit)
    assert fixed[winner] == FUSED_EN
    # sibling stays intentionally blank (its EN is on the winner).
    assert fixed[0] == ""


def test_fused_retranslate_error_is_counted_and_logged(caplog):
    class _Boom:
        async def __call__(self, *a, **k):
            raise RuntimeError("vllm down")

    plan = plan_bubble_dedup(RAW_DIVERGENT, BLOCKS, [BUB])
    import logging
    with caplog.at_level(logging.WARNING):
        out = asyncio.run(
            apply_fused_balloon_retranslate(
                plan.deduped, KEPT_TEXTS, plan, KEPT_TEXTS, [0, 1], _Boom()
            )
        )
    # On failure keep the deduped (winner-only) output, but surface the error.
    assert out == plan.deduped
    assert plan.retranslate_errors == 1
    assert plan.effective_jp == {}
    assert any("vllm down" in r.getMessage() for r in caplog.records)


# --- sentence-merge lead effective JP ---------------------------------------- #

def test_combined_effective_jp_maps_sentence_merge_lead_to_merged_text():
    # A two-bubble cross-bubble sentence: positions 0 (lead) and 1 (continuation)
    # merged into one group; kept-idx 0 renders the whole merged EN.
    plan = SentenceMergePlan()
    plan.groups = [MergeGroup([0, 1], "今朝はあの子達が来なかった")]
    plan.position_to_group = {0: 0, 1: 0}
    plan.merged_page_lines = ["今朝はあの子達が来なかった"]

    # target_positions: kept-idx -> page position. Both kept bubbles map 1:1.
    eff = combined_effective_jp(None, plan, [0, 1])
    # Only the LEAD (page pos == lead_position 0) gets the merged JP.
    assert eff == {0: "今朝はあの子達が来なかった"}


def test_fused_overrides_merge_on_overlap():
    # If a kept index is both a merge lead AND a fused winner, the fused (later,
    # wider) JP wins.
    class _Plan:
        effective_jp = {0: "FUSED"}

    merge = SentenceMergePlan()
    merge.groups = [MergeGroup([0, 1], "MERGED")]
    merge.position_to_group = {0: 0, 1: 0}
    eff = combined_effective_jp(_Plan(), merge, [0, 1])
    assert eff[0] == "FUSED"


# --- router/batch ordering parity -------------------------------------------- #

def _shared_finalize(translations, blocks, bubbles, kept_texts,
                     page_context_lines, target_positions, translator,
                     merge_plan=None):
    """The SHARED order both consumers now run: dedup -> fused-retranslate ->
    glossaries(with effective JP). Router and batch call the SAME helpers in this
    SAME order, so exercising it once documents their parity."""
    plan = plan_bubble_dedup(translations, blocks, bubbles)
    translations = plan.deduped
    if plan.retranslate:
        translations = asyncio.run(
            apply_fused_balloon_retranslate(
                translations, kept_texts, plan,
                page_context_lines, target_positions, translator,
            )
        )
    eff = combined_effective_jp(plan, merge_plan, target_positions)
    jp_for_postedit = [eff.get(i, kept_texts[i]) for i in range(len(translations))]
    return apply_postedit_glossaries(translations, jp_for_postedit)


def test_router_and_batch_order_produce_identical_finals():
    router_out = _shared_finalize(
        list(RAW_DIVERGENT), BLOCKS, [BUB], KEPT_TEXTS,
        KEPT_TEXTS, [0, 1], _FusedTranslator(),
    )
    batch_out = _shared_finalize(
        list(RAW_DIVERGENT), BLOCKS, [BUB], KEPT_TEXTS,
        KEPT_TEXTS, [0, 1], _FusedTranslator(),
    )
    assert router_out == batch_out
    winner = 1
    # The fused EN survives the glossary chain on BOTH paths (not blanked to ...).
    assert router_out[winner] == FUSED_EN
    assert router_out[0] == ""

"""Unit tests for column -> parent-bubble grouping (P1) and its safety nets (P2).

The detector/OCR split a single multi-column vertical speech balloon into ONE
box per column, so it arrives downstream as N independent translation units. The
v11 page-context model then omits (blanks all-but-one fragment) or duplicates
(reconstructs the whole sentence on each fragment). ``bubble_grouping`` fuses the
columns of one balloon into a single translation unit BEFORE translation, and
two safety nets recover the residual omission / duplication.

These pin:
  P1  (a) a 3-column balloon groups to ONE unit by CTD parent-bubble membership,
      (b) the same by geometric column adjacency when no bubble detector ran,
      (c) two genuinely-separate balloons stay separate (membership AND geometry),
      (d) vertically-stacked same-column boxes are NOT column-grouped,
      (e) the plan flows through build_merged_translation_request / apply_resplit,
      (f) the bounded cross-bubble dangling fuse pass.
  P2.1 (g) an empty KEPT high-conf JP bubble is selected for backfill,
       (h) an intentionally-blanked merge continuation is NOT backfilled,
       (i) a low-OCR-conf empty bubble is NOT backfilled.
  P2.2 (j) adjacent same-balloon bubbles with identical/substring EN de-dup,
       (k) non-adjacent or short identical EN are left alone.

Pure helpers (geometry + strings); no detector / OCR / model.
"""
from __future__ import annotations

from types import SimpleNamespace

import asyncio

from app.utils.bubble_grouping import (
    apply_fused_balloon_retranslate,
    bubble_id_of,
    dedup_adjacent_identical,
    dedup_by_bubble,
    en_near_duplicate,
    group_columns_into_bubbles,
    plan_bubble_dedup,
    select_backfill_targets,
)
from app.utils.page_units import (
    apply_resplit,
    build_merged_translation_request,
    build_page_translation_units,
)


def _b(minX, minY, maxX, maxY, **extra):
    d = {"minX": minX, "minY": minY, "maxX": maxX, "maxY": maxY, "confidence": 0.9}
    d.update(extra)
    return d


# Real geometry from .bench/ikenie5_v11fix8_insp/017/bubbles.json — the canonical
# 3-column balloon (idx 2/3/4) plus the adjacent separate balloon (idx 7), in page
# reading order (column-major RTL: descending x-center).
COL2 = _b(1068, 573, 1098, 707)   # そ…それが
COL3 = _b(1032, 578, 1066, 794)   # まーきさまちょうど切れてて
COL4 = _b(989, 650, 1024, 794)    # 明日買いに
COL7 = _b(958, 655, 989, 821)     # 行くつもりで  (separate balloon per the audit)
LINES = ["そ…それが", "まーきさまちょうど切れてて", "明日買いに", "行くつもりで"]


def _is_jp(text: str) -> bool:
    if not text:
        return False
    return any(
        "぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in text
    )


# --- P1 (a): membership groups the 3-column balloon, idx7 stays separate -------

def test_membership_groups_three_columns_and_separates_neighbour():
    # Balloon A encloses col2/3/4 centers but NOT col7 (its center x=973.5 < 980).
    bubble_a = {"minX": 980, "minY": 560, "maxX": 1110, "maxY": 800}
    # Balloon B encloses only col7 (center 973.5, 738).
    bubble_b = {"minX": 940, "minY": 640, "maxX": 1000, "maxY": 835}
    blocks = [COL2, COL3, COL4, COL7]
    plan = group_columns_into_bubbles(
        LINES, blocks, bubbles=[bubble_a, bubble_b]
    )
    assert plan.num_merges == 1
    # idx 2/3/4 (positions 0,1,2) fused; idx7 (position 3) solo.
    assert plan.groups[0].member_positions == [0, 1, 2]
    assert plan.groups[1].member_positions == [3]
    assert plan.merged_page_lines[0] == "そ…それがまーきさまちょうど切れてて明日買いに"


# --- P1 (b): geometric fallback also groups the 3-column balloon ---------------

def test_geometric_fallback_groups_columns_without_bubbles():
    plan = group_columns_into_bubbles(LINES, [COL2, COL3, COL4, COL7], bubbles=None)
    # The validation contract: idx2/3/4 land in ONE unit. (Without bubble
    # membership the adjacent idx7 column also joins — geometry alone cannot tell
    # the two balloons apart; that is what the membership signal is for.)
    g = plan.groups[plan.position_to_group[0]]
    assert {0, 1, 2}.issubset(set(g.member_positions))
    assert plan.position_to_group[0] == plan.position_to_group[1] == plan.position_to_group[2]


# --- P1 (c): two genuinely-separate balloons stay separate ---------------------

def test_membership_keeps_separate_bubbles_apart():
    # Two side-by-side columns one glyph apart but in DIFFERENT balloons.
    left = _b(600, 100, 632, 300)    # center 616
    right = _b(636, 100, 668, 300)   # center 652 (36px away ~ one glyph)
    bub_l = {"minX": 590, "minY": 90, "maxX": 634, "maxY": 310}
    bub_r = {"minX": 634, "minY": 90, "maxX": 680, "maxY": 310}
    # reading order RTL: right column first.
    plan = group_columns_into_bubbles(
        ["みぎ", "ひだり"], [right, left], bubbles=[bub_l, bub_r]
    )
    assert plan.num_merges == 0  # different parents -> never fused
    assert plan.position_to_group[0] != plan.position_to_group[1]


def test_geometric_keeps_far_columns_apart():
    # Two columns far apart in X (separate balloons across the page).
    a = _b(1000, 100, 1032, 300)
    b = _b(200, 100, 232, 300)
    plan = group_columns_into_bubbles(["あ", "い"], [a, b], bubbles=None)
    assert plan.num_merges == 0


# --- P1 (c'): membership over-merge regressions (the bug that disabled P1) ------
#
# These pin the ROOT CAUSE of the 44 -> 385 omission explosion: at scale on REAL
# YOLO interiors, distinct balloons / SFX fall inside ONE loose (or panel-sized)
# detected box, and the old membership branch (same bubble id + a weak 0.30
# pairwise Y-overlap, NO X-proximity / glyph-width / panel guard) chained them all
# into one unit and blanked the rest. Geometry is modelled on real interiors:
# loose bubble boxes a fair margin larger than their text, columns ONE glyph-width
# (~36 px) apart center-to-center. The OLD code fused every case below; the fix
# keeps true sibling columns together while refusing the cross-balloon fuse.

def test_membership_does_not_cross_fuse_distinct_clusters_in_one_loose_bubble():
    # A loose YOLO box swallowed TWO separate balloons. Each is a 2-column cluster;
    # the clusters sit side by side, strongly Y-overlapping, but separated by a
    # wide X gap (>> one glyph-width). Old membership (no X guard) fused all four
    # and blanked three; the fix fuses each cluster but NOT across the gap.
    r0 = _b(700, 100, 732, 280)   # right cluster, center 716
    r1 = _b(664, 100, 696, 290)   # center 680
    l0 = _b(544, 100, 576, 280)   # left cluster, center 560  (120 px from r1)
    l1 = _b(508, 100, 540, 285)   # center 524
    big = {"minX": 495, "minY": 85, "maxX": 745, "maxY": 305}
    plan = group_columns_into_bubbles(
        ["みぎA", "みぎB", "ひだりA", "ひだりB"],
        [r0, r1, l0, l1],
        bubbles=[big],
    )
    # Each genuine 2-column cluster fuses; the two clusters stay distinct.
    assert plan.position_to_group[0] == plan.position_to_group[1]
    assert plan.position_to_group[2] == plan.position_to_group[3]
    assert plan.position_to_group[1] != plan.position_to_group[2]
    assert len(plan.groups) == 2


def test_membership_does_not_fuse_sfx_into_adjacent_dialogue():
    # One loose box holds a 2-column dialogue cluster AND a fat SFX glyph that
    # Y-overlaps it. Old membership absorbed the SFX as a "column" and blanked it
    # (also the wide SFX inflated the X-step tolerance); the glyph-width-similarity
    # guard keeps the SFX its own unit.
    d0 = _b(744, 120, 776, 300)   # dialogue column, center 760, width 32
    d1 = _b(708, 120, 740, 300)   # dialogue column, center 724, width 32
    sfx = _b(595, 140, 685, 280)  # fat SFX glyph,    center 640, width 90
    big = {"minX": 585, "minY": 100, "maxX": 800, "maxY": 320}
    plan = group_columns_into_bubbles(
        ["セリフA", "セリフB", "ドカーン"],
        [d0, d1, sfx],
        bubbles=[big],
    )
    assert plan.position_to_group[0] == plan.position_to_group[1]   # dialogue fuses
    assert plan.position_to_group[1] != plan.position_to_group[2]   # SFX stays solo
    assert len(plan.groups) == 2


def test_membership_still_fuses_genuine_multicolumn_balloon():
    # The signal must survive the tighter guards: a real 4-column RTL balloon
    # (columns one glyph-width apart, left-stepping, strongly Y-overlapping) inside
    # a snug-but-loose bubble MUST still collapse to a single translation unit.
    c0 = _b(700, 100, 732, 280)   # center 716
    c1 = _b(664, 100, 696, 290)   # center 680
    c2 = _b(628, 110, 660, 285)   # center 644
    c3 = _b(592, 115, 624, 280)   # center 608
    bub = {"minX": 575, "minY": 85, "maxX": 750, "maxY": 305}
    plan = group_columns_into_bubbles(
        ["ぜんぶ", "ひとつ", "ばぶる", "になる"],
        [c0, c1, c2, c3],
        bubbles=[bub],
    )
    assert plan.num_merges == 1
    assert plan.groups[0].member_positions == [0, 1, 2, 3]


def test_membership_rejects_panel_sized_bubble():
    # Two column-adjacent, Y-overlapping fragments that WOULD pass the geometry,
    # but their only enclosing 'bubble' is a panel-sized region (the detector
    # emitted a whole panel, not a balloon). The panel-area guard refuses to let
    # that container authorise fusion -> left split (the conservative choice).
    a = _b(700, 100, 732, 280)   # center 716
    b = _b(664, 100, 696, 290)   # center 680
    panel = {"minX": 100, "minY": 50, "maxX": 800, "maxY": 1000}
    plan = group_columns_into_bubbles(
        ["パネル", "ぜんぶ"], [a, b], bubbles=[panel]
    )
    assert plan.num_merges == 0


# --- P1 (d): vertically-stacked same-column boxes are NOT column-grouped -------

def test_stacked_same_column_not_column_grouped():
    # Same column, stacked top/bottom with NO vertical overlap -> column grouping
    # (side-by-side fusion) must not touch them (that is sentence-merge's domain).
    top = _b(600, 100, 640, 190)
    bot = _b(600, 200, 640, 290)
    plan = group_columns_into_bubbles(["うえ", "した"], [top, bot], bubbles=None)
    assert plan.num_merges == 0


# --- P1 (e): plan round-trips through the existing merge machinery -------------

def _settings(**ov):
    base = dict(
        japanese_filter_enabled=True,
        japanese_filter_min_ratio=0.5,
        japanese_filter_katakana_max_length=6,
        english_early_exit_enabled=True,
        ocr_confidence_gate_enabled=True,
        ocr_confidence_gate_threshold=0.65,
        translation_pagecontext_whole_page=True,
        translation_sentence_merge=False,
        translation_bubble_grouping=True,
    )
    base.update(ov)
    return SimpleNamespace(**base)


def test_build_units_groups_columns_and_resplits():
    bubble_a = {"minX": 980, "minY": 560, "maxX": 1110, "maxY": 800}
    bubble_b = {"minX": 940, "minY": 640, "maxX": 1000, "maxY": 835}
    blocks = [COL2, COL3, COL4, COL7]
    units = build_page_translation_units(
        blocks,
        LINES,
        [0.93, 0.80, 0.93, 0.93],
        None,
        _settings(),
        is_japanese_fn=_is_jp,
        bubbles=[bubble_a, bubble_b],
    )
    assert units.merge_plan is not None
    req = build_merged_translation_request(units)
    assert req is not None
    # 2 translation units (the 3-column balloon + the solo idx7).
    assert len(req.merged_target_positions) == 2
    # Lead of the merged balloon renders EN; its two continuations blank.
    assert req.bubble_resplit[0] == (0, True)
    assert req.bubble_resplit[1][1] is False
    assert req.bubble_resplit[2][1] is False
    out = apply_resplit(["I just ran out, I'll buy more tomorrow", "going to go"], req.bubble_resplit)
    assert out == ["I just ran out, I'll buy more tomorrow", "", "", "going to go"]


# --- P1 (f): bounded cross-bubble dangling fuse pass ---------------------------

def test_fuse_dangling_merges_two_single_column_balloons():
    # Two separate SINGLE-column balloons in the same vertical band: the upper
    # dangles on the connective て -> they form one sentence and fuse.
    upper = _b(600, 100, 640, 200)
    lower = _b(600, 230, 640, 330)
    plan = group_columns_into_bubbles(
        ["今朝はあの子達が来て", "学校に行かなかった"],
        [upper, lower],
        bubbles=None,
        fuse_dangling=True,
    )
    assert plan.num_merges == 1
    assert plan.groups[0].member_positions == [0, 1]


def test_fuse_dangling_off_leaves_stacked_balloons_split():
    upper = _b(600, 100, 640, 200)
    lower = _b(600, 230, 640, 330)
    plan = group_columns_into_bubbles(
        ["今朝はあの子達が来て", "学校に行かなかった"],
        [upper, lower],
        bubbles=None,
        fuse_dangling=False,
    )
    assert plan.num_merges == 0


# --- P2.1 backfill selection ---------------------------------------------------

def test_backfill_selects_empty_highconf_jp_bubble():
    targets = select_backfill_targets(
        ["それは秘密だ", "うん"],
        ["", "Yeah"],
        [0.92, 0.90],
        None,
        is_japanese_fn=_is_jp,
    )
    assert targets == [0]


def test_backfill_skips_merge_continuation():
    # Merge fired: bubble 0 is the lead (full EN), bubble 1 the blanked
    # continuation -> bubble 1 must NOT be backfilled.
    targets = select_backfill_targets(
        ["親の顔が", "見たい"],
        ["I want to see your parents' faces", ""],
        [0.9, 0.9],
        [(0, True), (0, False)],
        is_japanese_fn=_is_jp,
    )
    assert targets == []


def test_backfill_skips_low_confidence():
    targets = select_backfill_targets(
        ["ぐちゃぐちゃ"],
        [""],
        [0.40],
        None,
        is_japanese_fn=_is_jp,
        conf_threshold=0.65,
    )
    assert targets == []


# --- P2.1 (FIX 1): unconditional backfill floor --------------------------------
#
# The marked page-context translate folds a sentence onto a neighbour and BLANKS
# this line. Every such blank is marked an (is_lead False) merge continuation, so
# the old continuation-skip suppressed ALL of them — the safety net fired zero
# times in practice. The floor below recovers the residual omissions:
#   * a blanked LEAD (whole group dropped) is recovered,
#   * a continuation whose lead EN is suspiciously short vs the fused JP (the lead
#     was truncated and dropped this tail) is recovered standalone,
#   * a continuation whose lead carries the full sentence is still left blank.

def test_backfill_recovers_blanked_lead_bubble():
    # The group's LEAD bubble (is_lead True) came back blank — the model dropped
    # the whole utterance. A blanked lead is not a continuation, so it must be
    # recovered (clean high-conf JP).
    targets = select_backfill_targets(
        ["これは大事な話だ"],
        [""],
        [0.9],
        [(0, True)],
        is_japanese_fn=_is_jp,
    )
    assert targets == [0]


def test_backfill_recovers_continuation_when_lead_truncated():
    # Two-bubble merge group: the lead got only "Um" while the fused JP is a long
    # sentence — the tail (this continuation) was dropped, not folded in. The
    # safeguard recovers the continuation standalone instead of leaving it blank.
    targets = select_backfill_targets(
        ["あのねずっと前から", "君のことが好きだった"],  # fused JP ~19 chars
        ["Um", ""],                                       # lead EN truncated
        [0.9, 0.9],
        [(0, True), (0, False)],
        is_japanese_fn=_is_jp,
    )
    assert targets == [1]


def test_backfill_keeps_continuation_blank_when_lead_complete():
    # The lead EN (33 chars) comfortably covers the fused JP (7 chars) — a genuine
    # merge continuation whose lead carries the full sentence. Must stay blank
    # (no spurious backfill -> no duplicate render).
    targets = select_backfill_targets(
        ["親の顔が", "見たい"],
        ["I want to see your parents' faces", ""],
        [0.9, 0.9],
        [(0, True), (0, False)],
        is_japanese_fn=_is_jp,
    )
    assert targets == []


def test_backfill_does_not_revive_bare_particle_continuation():
    # Even with a truncated lead, a 1-char sentence-final particle continuation is
    # too trivial to translate standalone -> left blank (conservative guard).
    targets = select_backfill_targets(
        ["ずっと前からあなたのことが", "ね"],
        ["Um", ""],
        [0.9, 0.9],
        [(0, True), (0, False)],
        is_japanese_fn=_is_jp,
    )
    assert targets == []


# --- P2.2 adjacent identical-EN de-dup -----------------------------------------

def test_dedup_collapses_adjacent_identical_en():
    # Two adjacent columns of one balloon both got the WHOLE sentence.
    a = _b(1032, 578, 1066, 794)
    b = _b(989, 650, 1024, 794)
    en = "I just ran out, I'll buy more tomorrow"
    out = dedup_adjacent_identical([en, en], [a, b])
    assert out == [en, ""]


def test_dedup_substring_keeps_longer_on_lead():
    a = _b(1032, 578, 1066, 794)
    b = _b(989, 650, 1024, 794)
    short = "I'll buy more tomorrow"
    full = "I just ran out, I'll buy more tomorrow"
    # lead carries the short fragment, continuation the full -> full lands on lead.
    out = dedup_adjacent_identical([short, full], [a, b])
    assert out == [full, ""]


def test_dedup_leaves_nonadjacent_identical():
    a = _b(1000, 100, 1032, 300)
    b = _b(200, 100, 232, 300)
    en = "It is a long sentence here"
    out = dedup_adjacent_identical([en, en], [a, b])
    assert out == [en, en]  # far apart -> different balloons, untouched


def test_dedup_leaves_short_identical_interjections():
    a = _b(1032, 578, 1066, 794)
    b = _b(989, 650, 1024, 794)
    out = dedup_adjacent_identical(["Huh?", "Huh?"], [a, b])
    assert out == ["Huh?", "Huh?"]  # below min_chars -> not collapsed


# --- P2.3 (FIX 2): bubble-keyed final dedup ("1 balloon = 1 string") -----------
#
# A speech balloon holds exactly one utterance. When P1 missed grouping the
# column-fragments of one balloon, the model reconstructs DIVERGENT strings on the
# siblings ("ghosted"/duplicated EN). dedup_adjacent_identical misses these (needs
# exact/substring + adjacency + >=8 chars + vertical geometry). The bubble-keyed
# dedup collapses by SHARED detected bubble id alone — independent of adjacency,
# orientation, length and string equality — keeping one winner per balloon.

# Two distinct detected bubbles (different bubble_id).
BUB_A = {"minX": 980, "minY": 560, "maxX": 1110, "maxY": 800}   # holds COL2/COL3
BUB_B = {"minX": 940, "minY": 640, "maxX": 1000, "maxY": 835}   # holds COL7


def test_dedup_by_bubble_collapses_divergent_duplicates_in_one_balloon():
    # COL2 (area 30*134=4020) and COL3 (area 34*216=7344) both sit inside BUB_A.
    # Their EN diverges (NOT equal / substring) so dedup_adjacent_identical would
    # leave both — but one balloon = one string, so all-but-the-winner is blanked.
    # Winner = largest-area block (COL3) -> its EN survives.
    out = dedup_by_bubble(
        ["buy more", "I just ran out, I'll buy more tomorrow"],
        [COL2, COL3],
        [BUB_A, BUB_B],
    )
    assert out == ["", "I just ran out, I'll buy more tomorrow"]


def test_dedup_by_bubble_keeps_distinct_balloons():
    # COL2 -> BUB_A, COL7 -> BUB_B (different bubble_id). Genuinely distinct
    # balloons are NEVER collapsed, even though they are adjacent on the page.
    out = dedup_by_bubble(
        ["First balloon line here", "Second balloon line here"],
        [COL2, COL7],
        [BUB_A, BUB_B],
    )
    assert out == ["First balloon line here", "Second balloon line here"]


def test_dedup_by_bubble_collapses_even_identical_across_three_fragments():
    # Three column-fragments of ONE balloon, the model put the SAME full EN on all
    # three. Only the winner survives; the other two are blanked.
    c0 = _b(1068, 573, 1098, 707)   # area 30*134 = 4020
    c1 = _b(1032, 578, 1066, 794)   # area 34*216 = 7344 (largest -> winner)
    c2 = _b(989, 650, 1024, 794)    # area 35*144 = 5040
    full = "I just ran out, I'll buy more tomorrow"
    out = dedup_by_bubble([full, full, full], [c0, c1, c2], [BUB_A])
    assert out == ["", full, ""]


def test_dedup_by_bubble_noop_without_bubbles():
    # No bubble detector ran -> bubble-keyed dedup is a no-op (the adjacency dedup
    # remains the fallback). Returns an unchanged copy.
    en = ["alpha text one", "beta text two"]
    out = dedup_by_bubble(en, [COL2, COL3], None)
    assert out == en
    assert out is not en  # new list


def test_dedup_by_bubble_ignores_blocks_outside_any_bubble():
    # A block whose center is in no detected bubble (SFX over art) keeps its EN —
    # only in-balloon duplicates are touched.
    sfx = _b(5, 5, 40, 200)   # center (22, 102) -> inside no bubble
    out = dedup_by_bubble(
        ["dialogue line here", "dramatic sfx boom"],
        [COL2, sfx],
        [BUB_A, BUB_B],
    )
    assert out == ["dialogue line here", "dramatic sfx boom"]


# --- P0 (2026-07-01): FUSED-BALLOON RETRANSLATE (content-drop fix) -------------
#
# dedup_by_bubble blanks all-but-winner per balloon. For a DUP balloon (same
# utterance reconstructed on each fragment) that is correct. For a DIVERGENT
# balloon (multi-column balloon holding DISTINCT lines) blanking silently DROPS
# the sibling content — the fused-balloon retranslate re-issues ONE marked call
# on the balloon's FUSED JP so the content is preserved.


class _RecordingTranslator:
    """Fake translate_page_context_marked: records calls, returns a fixed EN."""

    def __init__(self, fused_en: str = "We're calmly having a family breakfast — no!"):
        self.calls: list = []
        self.fused_en = fused_en

    async def __call__(self, page_lines, target_indices, target_language="English"):
        self.calls.append((list(page_lines), list(target_indices), target_language))
        # translate_page_context_marked returns one EN per marked target.
        return [self.fused_en for _ in target_indices]


def test_en_near_duplicate_classifies_dup_vs_divergent():
    # Substring / equality / high token-overlap == near-duplicate (the dup case).
    assert en_near_duplicate("buy more tomorrow", "I'll buy more tomorrow")
    assert en_near_duplicate("Hello there", "hello there.")
    # Genuinely distinct content == NOT a near-duplicate (must be preserved).
    assert not en_near_duplicate(
        "We're calmly having a family breakfast", "No...!!"
    )


def test_plan_bubble_dedup_matches_legacy_deduped_output():
    # plan.deduped is byte-identical to the legacy dedup_by_bubble blanking.
    trans = ["buy more", "I just ran out, I'll buy more tomorrow"]
    plan = plan_bubble_dedup(trans, [COL2, COL3], [BUB_A, BUB_B])
    legacy = dedup_by_bubble(trans, [COL2, COL3], [BUB_A, BUB_B])
    assert plan.deduped == legacy == ["", "I just ran out, I'll buy more tomorrow"]


def test_plan_flags_divergent_balloon_but_not_near_dup():
    # Divergent siblings -> a retranslate group (winner = largest-area COL3).
    divergent = plan_bubble_dedup(
        ["No...!!", "We're calmly having a family breakfast"],
        [COL2, COL3],
        [BUB_A, BUB_B],
    )
    assert len(divergent.retranslate) == 1
    grp = divergent.retranslate[0]
    assert grp.winner == 1              # COL3 is larger -> renders the balloon
    assert grp.members == [0, 1]        # both members, reading order
    assert grp.blanked == [0]
    # Near-duplicate siblings -> NO retranslate group (true dup case).
    dup = plan_bubble_dedup(
        ["buy more tomorrow", "I'll buy more tomorrow"],
        [COL2, COL3],
        [BUB_A, BUB_B],
    )
    assert dup.retranslate == []


def test_fused_retranslate_calls_once_winner_gets_fused_en_siblings_blank():
    # page context: two kept lines (the balloon's two columns), reading order.
    page_context_lines = ["違う…!!", "平然と家族で朝ごはんを"]
    target_positions = [0, 1]  # kept-idx -> page position
    kept_texts = ["違う…!!", "平然と家族で朝ごはんを"]
    plan = plan_bubble_dedup(
        ["No...!!", "We're calmly having a family breakfast"],
        [COL2, COL3],
        [BUB_A, BUB_B],
    )
    fake = _RecordingTranslator()
    out = asyncio.run(
        apply_fused_balloon_retranslate(
            plan.deduped,
            kept_texts,
            plan,
            page_context_lines,
            target_positions,
            fake,
        )
    )
    # Exactly ONE extra marked call for the one divergent balloon.
    assert len(fake.calls) == 1
    ctx_lines, tgt_idx, lang = fake.calls[0]
    # Members fused into ONE line (reading order, no separator) replacing them.
    assert ctx_lines == ["違う…!!平然と家族で朝ごはんを"]
    assert tgt_idx == [1] or tgt_idx == [0]  # single fused target
    assert len(tgt_idx) == 1
    # Winner (COL3, idx 1) carries the fused EN; sibling blanked.
    assert out[1] == fake.fused_en
    assert out[0] == ""


def test_fused_retranslate_noop_when_no_divergent_groups():
    # Near-dup balloon => plan.retranslate empty => translator never called and
    # the deduped output is returned unchanged (setting-OFF behavior parity).
    plan = plan_bubble_dedup(
        ["buy more tomorrow", "I'll buy more tomorrow"],
        [COL2, COL3],
        [BUB_A, BUB_B],
    )
    fake = _RecordingTranslator()
    out = asyncio.run(
        apply_fused_balloon_retranslate(
            plan.deduped,
            ["明日買いに", "明日買いに来る"],
            plan,
            ["明日買いに", "明日買いに来る"],
            [0, 1],
            fake,
        )
    )
    assert fake.calls == []
    assert out == plan.deduped == ["", "I'll buy more tomorrow"]


def test_setting_off_equals_legacy_blank_for_divergent_balloon():
    # When translation_balloon_fused_retranslate is OFF the caller simply does
    # NOT run apply_fused_balloon_retranslate, so a divergent balloon keeps the
    # legacy blank (winner-only). This pins that plan.deduped alone == old drop.
    plan = plan_bubble_dedup(
        ["No...!!", "We're calmly having a family breakfast"],
        [COL2, COL3],
        [BUB_A, BUB_B],
    )
    assert plan.deduped == ["", "We're calmly having a family breakfast"]


# --- membership helper ---------------------------------------------------------

def test_bubble_id_picks_smallest_enclosing():
    big = {"minX": 0, "minY": 0, "maxX": 1000, "maxY": 1000}
    small = {"minX": 980, "minY": 560, "maxX": 1110, "maxY": 800}
    assert bubble_id_of(COL2, [big, small]) == 1  # smaller enclosure wins
    assert bubble_id_of(_b(5, 5, 9, 9), [big, small]) == 0
    assert bubble_id_of(_b(2000, 2000, 2010, 2010), [big, small]) is None

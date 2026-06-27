"""Unit tests for cross-bubble JP sentence-continuation merge.

The detector + OCR split one JP sentence typeset across 2-3 stacked bubbles into
independent lines, so the v11 translator renders contradicting halves
(p8 "今朝はあの子達が" -> affirmative negating the paired "didn't come";
"からな" -> "It's from you"). ``sentence_merge`` re-segments those continuations
into ONE translation unit BEFORE translation (no prompt change), then re-splits
the English back to the member bubbles for typesetting.

These pin: (a) dangling-connective detection on the LEADING line, (b) bare
sentence-final particle detection on the TRAILING line, (c) terminal-punctuation
guard (closed sentences never dangle), (d) same-column / adjacency geometry
guard, (e) max-merge-span cap, (f) re-split member mapping.
"""
from __future__ import annotations

from app.utils.sentence_merge import (
    MAX_MERGE_SPAN,
    detect_sentence_continuations,
    has_dangling_connective,
    is_bare_sentence_final,
    resplit_translation_to_members,
)


def _b(minX, minY, maxX, maxY):
    return {"minX": minX, "minY": minY, "maxX": maxX, "maxY": maxY}


# A single vertical column: same X band, descending Y (top-to-bottom reading).
def _column(n, *, x0=600, x1=660, y_start=100, h=90, gap=10):
    blocks = []
    y = y_start
    for _ in range(n):
        blocks.append(_b(x0, y, x1, y + h))
        y += h + gap
    return blocks


# --- leading-line dangling-connective detection -------------------------------

def test_dangling_te_form_no_terminal_punct():
    assert has_dangling_connective("今朝はあの子達が来て")
    assert has_dangling_connective("そう言われても困るんだけど")
    assert has_dangling_connective("行くつもりだったんだが")


def test_dangling_blocked_by_terminal_punct():
    # Same trailers but the sentence already closed -> not dangling.
    assert not has_dangling_connective("今日は来て。")
    assert not has_dangling_connective("困るんだけど！")
    assert not has_dangling_connective("行くんだが…")


def test_non_dangling_normal_sentence():
    assert not has_dangling_connective("おはようございます")
    assert not has_dangling_connective("元気です")


def test_bare_one_char_particle_is_not_dangling():
    # A line that is ONLY "て" is a fragment, not a danging clause.
    assert not has_dangling_connective("て")
    assert not has_dangling_connective("の")


# --- trailing-line bare sentence-final detection ------------------------------

def test_bare_sentence_final_particles():
    assert is_bare_sentence_final("からな")
    assert is_bare_sentence_final("のに")
    assert is_bare_sentence_final("なさい")
    assert is_bare_sentence_final("だろう")
    assert is_bare_sentence_final("させられる")


def test_bare_sentence_final_tolerates_trailing_punct():
    assert is_bare_sentence_final("なさい。")
    assert is_bare_sentence_final("からな！")


def test_not_bare_sentence_final_when_embedded():
    # からな as a suffix of a longer real line is NOT a bare tail line.
    assert not is_bare_sentence_final("それは秘密だからな")
    assert not is_bare_sentence_final("ごめんなさい")  # full apology, stands alone


# --- merge: leading dangling connective fuses into next bubble -----------------

def test_merge_two_bubbles_leading_dangling():
    blocks = _column(2)
    lines = ["今朝はあの子達が来て", "学校に行かなかった"]
    plan = detect_sentence_continuations(lines, blocks)
    assert plan.num_merges == 1
    assert len(plan.groups) == 1
    g = plan.groups[0]
    assert g.member_positions == [0, 1]
    assert g.merged_text == "今朝はあの子達が来て学校に行かなかった"
    assert g.lead_position == 0
    assert g.continuation_positions == [1]
    assert plan.merged_page_lines == ["今朝はあの子達が来て学校に行かなかった"]


def test_merge_two_bubbles_trailing_bare_final():
    blocks = _column(2)
    lines = ["危ないからやめておけ", "からな"]
    plan = detect_sentence_continuations(lines, blocks)
    assert plan.num_merges == 1
    assert plan.groups[0].member_positions == [0, 1]


def test_no_merge_two_independent_sentences():
    blocks = _column(2)
    lines = ["おはよう。", "今日はいい天気だね"]
    plan = detect_sentence_continuations(lines, blocks)
    assert plan.num_merges == 0
    assert [g.member_positions for g in plan.groups] == [[0], [1]]
    assert plan.merged_page_lines == lines


# --- geometry guards ----------------------------------------------------------

def test_no_merge_when_different_columns():
    # Leading line dangles, but the next line is in a FAR column (RTL neighbour),
    # not the bubble directly below -> must NOT fuse.
    blocks = [_b(600, 100, 660, 200), _b(300, 100, 360, 200)]
    lines = ["来て", "別の話だよ"]
    plan = detect_sentence_continuations(lines, blocks)
    assert plan.num_merges == 0


def test_merge_respects_same_column_tolerance():
    # Slight X jitter within tolerance still merges (same column).
    blocks = [_b(600, 100, 660, 200), _b(606, 210, 666, 310)]
    lines = ["行くつもりだったんだが", "やめた"]
    plan = detect_sentence_continuations(lines, blocks)
    assert plan.num_merges == 1


# --- span cap -----------------------------------------------------------------

def test_merge_three_bubbles_chain():
    blocks = _column(3)
    # The first TWO leading lines dangle (て, けど) -> 3-bubble chain (== MAX span).
    lines = ["あの時は信じていて", "今も変わらないけど", "もう遅い"]
    plan = detect_sentence_continuations(lines, blocks)
    assert plan.num_merges == 1
    assert plan.groups[0].member_positions == [0, 1, 2]


def test_merge_span_capped_at_max():
    # Four chained dangling lines must NOT fuse into one >MAX_MERGE_SPAN group.
    # Every leading line dangles (て / が / で) so the chain would run forever
    # without the cap.
    blocks = _column(4)
    lines = ["あの時は信じていて", "ずっと待っていたが", "今も変わらないで", "そばにいる"]
    plan = detect_sentence_continuations(lines, blocks)
    for g in plan.groups:
        assert len(g.member_positions) <= MAX_MERGE_SPAN
    # The first group consumes exactly MAX_MERGE_SPAN, the 4th is its own group.
    assert plan.groups[0].member_positions == [0, 1, 2]
    assert plan.groups[-1].member_positions == [3]


# --- re-split mapping ---------------------------------------------------------

def test_resplit_renders_in_lead_blanks_continuations():
    g = detect_sentence_continuations(
        ["今朝はあの子達が来て", "学校に行かなかった"], _column(2)
    ).groups[0]
    mapping = resplit_translation_to_members(g, "They came this morning but didn't go to school")
    assert mapping[0] == "They came this morning but didn't go to school"
    assert mapping[1] == ""
    assert set(mapping.keys()) == {0, 1}


def test_resplit_continuation_filler_ellipsis():
    g = detect_sentence_continuations(["危ないからやめておけ", "からな"], _column(2)).groups[0]
    mapping = resplit_translation_to_members(g, "Stop, it's dangerous", continuation_filler="…")
    assert mapping[0] == "Stop, it's dangerous"
    assert mapping[1] == "…"


def test_solo_group_resplit_is_identity():
    plan = detect_sentence_continuations(["おはよう。", "元気？"], _column(2))
    g0 = plan.groups[0]
    mapping = resplit_translation_to_members(g0, "Morning.")
    assert mapping == {0: "Morning."}


# --- robustness ---------------------------------------------------------------

def test_empty_input():
    plan = detect_sentence_continuations([], [])
    assert plan.groups == []
    assert plan.merged_page_lines == []


def test_misaligned_inputs_identity_plan():
    plan = detect_sentence_continuations(["a", "b", "c"], _column(2))
    # blocks shorter than lines -> identity (no merge), no crash
    assert plan.num_merges == 0
    assert len(plan.groups) == 3


def test_position_to_group_covers_every_position():
    blocks = _column(4)
    lines = ["今朝はあの子達が来て", "学校に行かなかった", "おはよう。", "元気？"]
    plan = detect_sentence_continuations(lines, blocks)
    # positions 0,1 merge; 2,3 solo -> 3 groups, every pos mapped exactly once
    assert set(plan.position_to_group.keys()) == {0, 1, 2, 3}
    assert plan.position_to_group[0] == plan.position_to_group[1]
    assert plan.position_to_group[2] != plan.position_to_group[0]
    assert len(plan.groups) == 3

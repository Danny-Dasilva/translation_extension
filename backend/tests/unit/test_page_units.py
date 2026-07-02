"""Unit tests for the SHARED page-translation-units builder.

``build_page_translation_units`` replaces three hand-maintained copies of the
gate -> whole-page-context assembly (the live pipelined branch, the batch
else-branch, and the eval batch script). These tests pin the contract the three
call sites depend on:

  * the kept subset (blocks / texts / confs) stays parallel and 1:1,
  * REAL OCR confidence is threaded into ``kept_confs`` (the #-fix that had to be
    applied three times),
  * the whole-page v11 context keeps dropped DIALOGUE lines (so a sentence
    partner is not orphaned) but EXCLUDES pure-SFX / garble drops,
  * ``target_positions`` index the kept lines INTO ``page_context_lines``,
  * ``is_leave_intact_label`` fires on EVERY path (was missing from the live
    pipelined branch),
  * gate-dropped real JP ink lands in ``erase_only_blocks`` (inpaint-only),
  * the three call SHAPES (with/without confs, gate on/off, jp-filter off)
    produce identical kept/context decisions.

These run the helper with FAKE injected predicates (no detector / OCR / model),
so they stay fast and import-safe in the worktree.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.utils.page_units import (
    apply_resplit,
    build_merged_translation_request,
    build_page_translation_units,
)


def _b(minX, minY, maxX, maxY, **extra):
    d = {"minX": minX, "minY": minY, "maxX": maxX, "maxY": maxY, "confidence": 0.9}
    d.update(extra)
    return d


def _settings(**overrides):
    base = dict(
        japanese_filter_enabled=True,
        japanese_filter_min_ratio=0.5,
        japanese_filter_katakana_max_length=6,
        english_early_exit_enabled=True,
        ocr_confidence_gate_enabled=True,
        ocr_confidence_gate_threshold=0.65,
        translation_pagecontext_whole_page=True,
        translation_sentence_merge=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# Simple JP predicate: True if the text contains a hiragana/katakana/kanji glyph
# and is not a pure-Latin string. Good enough for these structural tests.
def _is_jp(text: str) -> bool:
    if not text:
        return False
    return any(
        "぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in text
    )


def _never_skip_english(block, text_lines, ocr_text, is_jp_fn):
    return False


def _never_leave_intact(text):
    return False


# A column of n vertically-stacked same-column blocks in reading order.
def _column(n, *, x0=600, x1=660, y0=100, h=90, gap=10):
    blocks = []
    y = y0
    for _ in range(n):
        blocks.append(_b(x0, y, x1, y + h))
        y += h + gap
    return blocks


# --- basic kept-subset / parallelism -----------------------------------------

def test_all_kept_high_conf():
    blocks = _column(3)
    texts = ["おはよう", "元気ですか", "そうだね"]
    confs = [0.95, 0.90, 0.88]
    units = build_page_translation_units(
        blocks, texts, confs, None, _settings(),
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_never_skip_english,
    )
    assert units.kept_texts == texts
    assert units.kept_confs == confs
    assert units.kept_indices == [0, 1, 2]
    assert units.page_context_lines == texts
    assert units.target_positions == [0, 1, 2]
    assert units.erase_only_blocks == []


def test_kept_confs_threaded_real_confidence():
    # The #-fix: real per-bubble OCR confidence must reach kept_confs (not None).
    blocks = _column(2)
    texts = ["これは本物の台詞です", "もうひとつの台詞"]
    # Both confs above the recalibrated 0.85 long-text gate so the threading check
    # (real per-bubble conf reaching kept_confs) is not confounded by a drop.
    confs = [0.91, 0.88]
    units = build_page_translation_units(
        blocks, texts, confs, None, _settings(),
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_never_skip_english,
    )
    assert units.kept_confs == [0.91, 0.88]


# --- #3 decouple drop-from-render vs drop-from-context ------------------------

def test_dropped_dialogue_stays_in_context_not_rendered():
    # #3 DECOUPLE: a line dropped from RENDER (garble) that is still a DIALOGUE
    # context candidate must stay in the page CONTEXT so its sentence partner is
    # not orphaned. "お母さん身身わわ" is substitution-garble (身代わり misread) at a
    # conf the gate still drops (is_garbled_low_conf True => not rendered) but it
    # carries a speaker reference (is_dialogue_context_candidate True => kept as
    # context-only). (An exact P+P お母さんお母さん is now COLLAPSE-RECOVERED, not
    # dropped, so it would render — hence a substitution garble here.)
    blocks = _column(2)
    texts = ["お母さんは僕のことを心配して", "お母さん身身わわ"]
    confs = [0.92, 0.66]
    units = build_page_translation_units(
        blocks, texts, confs, None, _settings(),
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_never_skip_english,
    )
    # Only line 0 rendered, but BOTH lines in the page context.
    assert units.kept_texts == ["お母さんは僕のことを心配して"]
    assert units.kept_indices == [0]
    assert len(units.page_context_lines) == 2
    assert units.page_context_lines[1] == "お母さん身身わわ"
    # Target marks the kept line at its position in the full page.
    assert units.target_positions == [0]
    # The dropped garble is real JP ink -> erase-only.
    assert len(units.erase_only_blocks) == 1


def test_pure_sfx_drop_excluded_from_context():
    # A low-conf multi-char katakana SFX scrawl is dropped AND excluded from
    # context (not dialogue), but is real JP ink -> erase-only.
    # NOTE: the OCR-gate short-text carve-out now KEEPS very short (< 5 char) SFX
    # like "ドカ" (SFX/moans/numbers are ~98% correct vs gold), so this pins the
    # still-dropped case: a longer low-conf SFX scrawl above the carve-out length.
    blocks = _column(2)
    texts = ["ちゃんと聞いて欲しいんだ", "バキュンゴバッ"]
    confs = [0.90, 0.30]
    units = build_page_translation_units(
        blocks, texts, confs, None, _settings(),
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_never_skip_english,
    )
    assert units.kept_texts == ["ちゃんと聞いて欲しいんだ"]
    # SFX not in dialogue context.
    assert units.page_context_lines == ["ちゃんと聞いて欲しいんだ"]
    # Real JP ink -> erase-only block collected.
    assert len(units.erase_only_blocks) == 1


# --- #1 is_leave_intact fires on every path ----------------------------------

def test_leave_intact_label_skipped_everywhere():
    blocks = _column(2)
    texts = ["表紙用イラスト", "本物の台詞です"]
    confs = [0.95, 0.95]

    def _leave_intact(text):
        return "表紙用イラスト" in text

    units = build_page_translation_units(
        blocks, texts, confs, None, _settings(),
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=_leave_intact,
        should_skip_as_english_fn=_never_skip_english,
    )
    # Editorial label is NOT rendered, NOT context, NOT erased.
    assert units.kept_texts == ["本物の台詞です"]
    assert units.page_context_lines == ["本物の台詞です"]
    assert units.erase_only_blocks == []


def test_leave_intact_none_means_no_label_filtering():
    # When no leave-intact predicate is injected, labels are NOT skipped.
    blocks = _column(1)
    texts = ["表紙用イラスト"]
    units = build_page_translation_units(
        blocks, texts, [0.95], None, _settings(),
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=None,
        should_skip_as_english_fn=_never_skip_english,
    )
    assert units.kept_texts == ["表紙用イラスト"]


# --- english early-exit -------------------------------------------------------

def test_english_early_exit_skips_horizontal_latin():
    blocks = _column(2)
    texts = ["WATERMARK", "本物の台詞です"]
    confs = [0.95, 0.95]

    def _skip_english(block, text_lines, ocr_text, is_jp_fn):
        return ocr_text == "WATERMARK"

    units = build_page_translation_units(
        blocks, texts, confs, None, _settings(),
        is_japanese_fn=lambda t: t != "WATERMARK" and _is_jp(t),
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_skip_english,
    )
    assert units.kept_texts == ["本物の台詞です"]
    assert units.page_context_lines == ["本物の台詞です"]
    assert units.erase_only_blocks == []  # english region NOT erased


# --- call-shape parity --------------------------------------------------------

def test_parity_confs_none_defaults_to_one():
    # Call SHAPE without confidences (e.g. japanese_filter-disabled batch path):
    # every conf defaults to 1.0, nothing gate-dropped.
    blocks = _column(2)
    texts = ["おはよう", "元気ですか"]
    units = build_page_translation_units(
        blocks, texts, None, None, _settings(),
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_never_skip_english,
    )
    assert units.kept_texts == texts
    assert units.kept_confs == [1.0, 1.0]


def test_parity_gate_off_keeps_lowconf():
    # With the gate OFF, a low-conf line is NOT dropped (matches the pipelined
    # and batch branches when ocr_confidence_gate_enabled is False).
    blocks = _column(2)
    texts = ["お母さんは僕のことを", "心配してくれているんだよね"]
    confs = [0.92, 0.30]
    units = build_page_translation_units(
        blocks, texts, confs, None,
        _settings(ocr_confidence_gate_enabled=False),
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_never_skip_english,
    )
    assert units.kept_texts == texts
    assert units.page_context_lines == texts
    assert units.target_positions == [0, 1]


def test_parity_jp_filter_off_keeps_all():
    blocks = _column(2)
    texts = ["hello", "world"]  # non-JP, but filter is OFF
    units = build_page_translation_units(
        blocks, texts, [1.0, 1.0], None,
        _settings(japanese_filter_enabled=False),
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_never_skip_english,
    )
    assert units.kept_texts == texts
    assert units.page_context_lines == texts


def test_target_positions_index_into_page_context():
    # Mixed: kept, dropped-dialogue (context), kept. target_positions must index
    # the kept lines' slots in the FULL page context. The middle line is
    # substitution-garble at a conf the gate still drops -> dropped from render,
    # kept as dialogue context (speaker reference). (An exact P+P is now
    # collapse-recovered, not dropped.)
    blocks = _column(3)
    texts = ["ちゃんと聞いてほしいの", "お母さん身身わわ", "だから言ったでしょう"]
    confs = [0.95, 0.66, 0.93]
    units = build_page_translation_units(
        blocks, texts, confs, None, _settings(),
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_never_skip_english,
    )
    assert units.kept_indices == [0, 2]
    assert len(units.page_context_lines) == 3
    # kept lines sit at positions 0 and 2 in the full page context.
    assert units.target_positions == [0, 2]


# --- sentence-merge wiring ----------------------------------------------------

def test_sentence_merge_plan_populated_when_enabled():
    blocks = _column(2)
    texts = ["今朝はあの子達が来て", "学校に行かなかった"]
    confs = [0.95, 0.95]
    units = build_page_translation_units(
        blocks, texts, confs, None,
        _settings(translation_sentence_merge=True),
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_never_skip_english,
    )
    assert units.merge_plan is not None
    assert units.merge_plan.num_merges == 1
    assert units.merge_plan.groups[0].member_positions == [0, 1]


def test_sentence_merge_plan_none_when_disabled():
    blocks = _column(2)
    texts = ["今朝はあの子達が来て", "学校に行かなかった"]
    units = build_page_translation_units(
        blocks, texts, [0.95, 0.95], None,
        _settings(translation_sentence_merge=False),
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_never_skip_english,
    )
    assert units.merge_plan is None


def test_merged_request_collapses_kept_sentence_to_one_unit():
    # Two kept bubbles that form ONE sentence (leading dangles into trailer) must
    # collapse to ONE marked translation unit; re-split renders the EN on the
    # lead bubble and blanks the continuation bubble.
    blocks = _column(2)
    texts = ["今朝はあの子達が来て", "学校に行かなかった"]
    confs = [0.95, 0.95]
    units = build_page_translation_units(
        blocks, texts, confs, None,
        _settings(translation_sentence_merge=True),
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_never_skip_english,
    )
    req = build_merged_translation_request(units)
    assert req is not None
    # One sentence unit -> one marked line and one target.
    assert req.merged_page_lines == ["今朝はあの子達が来て学校に行かなかった"]
    assert req.merged_target_positions == [0]
    # Bubble 0 is the lead; bubble 1 is a continuation.
    assert req.bubble_resplit == [(0, True), (0, False)]

    # Model returns one EN for the merged unit; re-split -> per-bubble 1:1.
    out = apply_resplit(["They came this morning but didn't go to school"], req.bubble_resplit)
    assert out == ["They came this morning but didn't go to school", ""]


def test_merged_request_none_without_plan():
    blocks = _column(2)
    texts = ["おはよう。", "元気ですか"]
    units = build_page_translation_units(
        blocks, texts, [0.95, 0.95], None,
        _settings(translation_sentence_merge=True),  # on, but no danglers
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_never_skip_english,
    )
    # No continuation signal -> no plan -> caller uses the un-merged path.
    assert units.merge_plan is None
    assert build_merged_translation_request(units) is None


def test_merged_request_mixed_solo_and_merged():
    # bubble0+bubble1 merge; bubble2 stays solo. Two translation units; re-split
    # maps bubble2 to its own unit, lead, full EN.
    blocks = _column(3)
    texts = ["今朝はあの子達が来て", "学校に行かなかった", "おはよう。"]
    units = build_page_translation_units(
        blocks, texts, [0.95, 0.95, 0.95], None,
        _settings(translation_sentence_merge=True),
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_never_skip_english,
    )
    req = build_merged_translation_request(units)
    assert req is not None
    assert len(req.merged_page_lines) == 2
    assert req.merged_target_positions == [0, 1]
    assert req.bubble_resplit == [(0, True), (0, False), (1, True)]
    out = apply_resplit(["merged EN", "Morning."], req.bubble_resplit)
    assert out == ["merged EN", "", "Morning."]


def test_resplit_continuation_filler():
    blocks = _column(2)
    texts = ["危ないからやめておけ", "からな"]
    units = build_page_translation_units(
        blocks, texts, [0.95, 0.95], None,
        _settings(translation_sentence_merge=True),
        is_japanese_fn=_is_jp,
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_never_skip_english,
    )
    req = build_merged_translation_request(units)
    out = apply_resplit(["Stop, it's dangerous"], req.bubble_resplit, continuation_filler="…")
    assert out == ["Stop, it's dangerous", "…"]


def test_on_drop_hook_called_with_reason():
    # A horizontal region whose OCR passes the (loose) JP filter but is caught by
    # the english early-exit geometry check -> drop reason "english_early_exit".
    blocks = _column(2)
    texts = ["横書きSFX", "本物の台詞"]
    drops = []

    def _skip_english(block, text_lines, ocr_text, is_jp_fn):
        return ocr_text == "横書きSFX"

    build_page_translation_units(
        blocks, texts, [0.95, 0.95], None, _settings(),
        is_japanese_fn=_is_jp,  # both pass the JP filter
        is_leave_intact_fn=_never_leave_intact,
        should_skip_as_english_fn=_skip_english,
        on_drop=lambda i, t, c, r: drops.append((i, r)),
    )
    assert (0, "english_early_exit") in drops

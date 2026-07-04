"""Unit tests for DETECTION-TIME balloon-column fusion.

``ComicTextDetectorService.fuse_balloon_columns`` fuses the side-by-side text
columns of ONE speech balloon into a SINGLE block BEFORE OCR, so OCR sees one
crop and translation one JP string per balloon (eliminating the per-column
duplication / omission the page-context model otherwise produces). This is the
detection-time twin of the disabled ``bubble_grouping`` re-segmentation — but
there is NO merge->translate->resplit roundtrip, so the 2026-06-29 regression
(resplit blanking continuations) cannot recur.

These pin the predicate directly on the fused BLOCK output:
  (a) the columns of one balloon fuse to ONE union block; a neighbour balloon
      stays separate (real Ikenie geometry),
  (b) two side-by-side columns in DIFFERENT parent bubbles NEVER fuse,
  (c) a fat SFX glyph that merely Y-overlaps dialogue is NOT absorbed,
  (d) a panel-sized container does not authorise fusion,
  (e) two distinct clusters inside ONE loose box each fuse but do NOT bridge,
  (f) vertically-stacked same-column boxes are NOT column-fused,
  (g) no bubble detector => no-op (input returned unchanged),
  (h) fused block geometry = member union, confidence = max member,
  (i) the span cap bounds a run.

Pure geometry; the staticmethod loads no ONNX model.
"""
from __future__ import annotations

from app.services.ctd_service import ComicTextDetectorService

fuse = ComicTextDetectorService.fuse_balloon_columns


def _b(minX, minY, maxX, maxY, **extra):
    d = {"minX": minX, "minY": minY, "maxX": maxX, "maxY": maxY, "confidence": 0.9}
    d.update(extra)
    return d


# Real geometry from .bench/ikenie5_v11fix8_insp/017/bubbles.json — the canonical
# 3-column balloon (idx 2/3/4) plus the adjacent SEPARATE balloon (idx 7).
COL2 = _b(1068, 573, 1098, 707)   # そ…それが
COL3 = _b(1032, 578, 1066, 794)   # まーきさまちょうど切れてて
COL4 = _b(989, 650, 1024, 794)    # 明日買いに
COL7 = _b(958, 655, 989, 821)     # 行くつもりで  (separate balloon per the audit)
BUB_A = {"minX": 980, "minY": 560, "maxX": 1110, "maxY": 800}
BUB_B = {"minX": 940, "minY": 640, "maxX": 1000, "maxY": 835}


def _bbox(block):
    return (block["minX"], block["minY"], block["maxX"], block["maxY"])


# --- (a) three columns of one balloon fuse; neighbour balloon stays separate ---

def test_fuse_three_columns_and_separate_neighbour():
    out = fuse([COL2, COL3, COL4, COL7], [BUB_A, BUB_B])
    assert len(out) == 2
    fused, solo = out[0], out[1]
    # Fused block is the union of the three columns.
    assert _bbox(fused) == (989, 573, 1098, 794)
    # The neighbour balloon is untouched (same object identity).
    assert solo is COL7


# --- (h) union geometry + confidence -------------------------------------------

def test_fused_block_confidence_is_max_member():
    c0 = _b(700, 100, 732, 280, confidence=0.7)
    c1 = _b(664, 100, 696, 290, confidence=0.95)
    bub = {"minX": 655, "minY": 90, "maxX": 745, "maxY": 305}
    out = fuse([c0, c1], [bub])
    assert len(out) == 1
    assert out[0]["confidence"] == 0.95
    assert _bbox(out[0]) == (664, 100, 732, 290)


# --- (b) distinct parent bubbles never fuse ------------------------------------

def test_distinct_parent_bubbles_never_fuse():
    left = _b(600, 100, 632, 300)    # center 616
    right = _b(636, 100, 668, 300)   # center 652 (~one glyph away)
    bub_l = {"minX": 590, "minY": 90, "maxX": 634, "maxY": 310}
    bub_r = {"minX": 634, "minY": 90, "maxX": 680, "maxY": 310}
    out = fuse([right, left], [bub_l, bub_r])
    assert len(out) == 2  # different parents -> untouched


# --- (c) fat SFX is not absorbed as a column -----------------------------------

def test_wide_sfx_not_fused_into_dialogue():
    d0 = _b(744, 120, 776, 300)   # dialogue column, width 32
    d1 = _b(708, 120, 740, 300)   # dialogue column, width 32
    sfx = _b(595, 140, 685, 280)  # fat SFX glyph,    width 90
    big = {"minX": 585, "minY": 100, "maxX": 800, "maxY": 320}
    out = fuse([d0, d1, sfx], [big])
    assert len(out) == 2  # d0+d1 fuse; sfx stays its own block
    # The SFX object survives untouched.
    assert sfx in out


# --- (d) panel-sized container does not authorise fusion -----------------------

def test_panel_sized_bubble_blocks_fusion():
    a = _b(700, 100, 732, 280)
    b = _b(664, 100, 696, 290)
    panel = {"minX": 100, "minY": 50, "maxX": 800, "maxY": 1000}
    out = fuse([a, b], [panel])
    assert len(out) == 2  # panel is not a balloon


# --- (e) two distinct clusters in one loose box each fuse but do NOT bridge -----

def test_two_clusters_in_loose_box_do_not_bridge():
    r0 = _b(700, 100, 732, 280)   # right cluster, center 716
    r1 = _b(664, 100, 696, 290)   # center 680
    l0 = _b(544, 100, 576, 280)   # left cluster, center 560 (120px from r1)
    l1 = _b(508, 100, 540, 285)   # center 524
    big = {"minX": 495, "minY": 85, "maxX": 745, "maxY": 305}
    out = fuse([r0, r1, l0, l1], [big])
    assert len(out) == 2  # each 2-col cluster fuses; the wide X-gap is not bridged


# --- genuine 4-column balloon still fuses despite the tighter guards ------------

def test_four_column_balloon_capped_at_default_span_3():
    # Default max_span=3 (2026-07-04 audit hardening): a 4-column balloon fuses
    # its first 3 columns and leaves the 4th solo — 4+-col fused JP choked the
    # model (all 4 audit regressions were >=4-col). Explicit larger span still
    # fuses all four (fusion mechanics intact).
    c0 = _b(700, 100, 732, 280)
    c1 = _b(664, 100, 696, 290)
    c2 = _b(628, 110, 660, 285)
    c3 = _b(592, 115, 624, 280)
    bub = {"minX": 575, "minY": 85, "maxX": 750, "maxY": 305}
    out = fuse([c0, c1, c2, c3], [bub])
    boxes = [_bbox(b) for b in out]
    assert len(out) == 2, boxes
    assert (628, 100, 732, 290) in boxes   # c0+c1+c2 fused (first 3)
    assert (592, 115, 624, 280) in boxes   # c3 solo, unchanged
    out_all = fuse([c0, c1, c2, c3], [bub], max_span=4)
    assert len(out_all) == 1
    assert _bbox(out_all[0]) == (592, 100, 732, 290)


# --- (f) vertically-stacked same-column boxes are NOT column-fused --------------

def test_stacked_same_column_not_fused():
    top = _b(600, 100, 640, 190)
    bot = _b(600, 200, 640, 290)  # no Y-overlap with top
    bub = {"minX": 590, "minY": 90, "maxX": 650, "maxY": 300}
    out = fuse([top, bot], [bub])
    assert len(out) == 2


# --- (g) no bubbles => no-op ---------------------------------------------------

def test_no_bubbles_is_noop():
    blocks = [COL2, COL3, COL4]
    assert fuse(blocks, None) == blocks
    assert fuse(blocks, []) == blocks


def test_single_block_is_noop():
    assert fuse([COL2], [BUB_A]) == [COL2]


# --- (i) span cap bounds a run -------------------------------------------------

def test_span_cap_bounds_run():
    c0 = _b(700, 100, 732, 280)
    c1 = _b(664, 100, 696, 290)
    c2 = _b(628, 110, 660, 285)
    c3 = _b(592, 115, 624, 280)
    bub = {"minX": 575, "minY": 85, "maxX": 750, "maxY": 305}
    # max_span=2 -> the 4-column balloon fuses in two pairs, not one block.
    out = fuse([c0, c1, c2, c3], [bub], max_span=2)
    assert len(out) == 2
    assert _bbox(out[0]) == (664, 100, 732, 290)  # c0+c1
    assert _bbox(out[1]) == (592, 110, 660, 285)  # c2+c3


# --- BUG D2: a near-identical duplicate raw box does not survive as a phantom --
#
# Real geometry from .bench/_flagab_on3_insp/082 (audit): a raw CTD column
# (idx6's bbox, 息子を視界に) was detected TWICE at (near-)identical bbox; one
# instance fused normally into the balloon's union block, the exact duplicate
# survived fusion as its own standalone box with a blank translation -- a
# phantom blank/duplicate render artifact.

def test_duplicate_block_does_not_survive_as_phantom():
    # A 4-column balloon at the default span cap (max_span=3): c0/c1/c2 fuse
    # into one union, c3 stays its own separate block (see
    # test_four_column_balloon_capped_at_default_span_3). An EXACT duplicate
    # detection of c2 -- the real p082 case: idx6 duplicates a member
    # already absorbed into the fused block -- must have ZERO effect on this
    # outcome. Without dedup the duplicate (sharing c2's bbox, which
    # legitimately passes adjacency with c3) wrongly fuses with c3 INSTEAD OF
    # c3 staying solo -- corrupting c3's bbox/crop with a ghost repeat of
    # c2's glyphs (verified: differs from the no-duplicate baseline pre-fix,
    # identical post-fix).
    c0 = _b(700, 100, 732, 280)
    c1 = _b(664, 100, 696, 290)
    c2 = _b(628, 110, 660, 285)
    c3 = _b(592, 115, 624, 280)
    dup_of_c2 = _b(628, 110, 660, 285)  # exact-duplicate raw CTD detection of c2
    bub = {"minX": 575, "minY": 85, "maxX": 750, "maxY": 305}
    baseline = fuse([c0, c1, c2, c3], [bub])
    out = fuse([c0, c1, c2, dup_of_c2, c3], [bub])
    boxes = [_bbox(b) for b in out]
    baseline_boxes = [_bbox(b) for b in baseline]
    # The duplicate must be dropped BEFORE fusion: byte-identical to the
    # no-duplicate baseline, never a phantom extra entry or a corrupted c3.
    assert boxes == baseline_boxes, boxes
    assert (628, 100, 732, 290) in boxes   # c0+c1+c2 fused (unchanged)
    assert (592, 115, 624, 280) in boxes   # c3 solo, uncorrupted


def test_near_duplicate_block_iou_above_threshold_is_dropped():
    c0 = _b(700, 100, 732, 280)
    c1 = _b(664, 100, 696, 290)
    c2 = _b(628, 110, 660, 285)
    c3 = _b(592, 115, 624, 280)
    # ~98% IoU with c2 (shifted a few px, not byte-exact) -- still a duplicate.
    near_dup_of_c2 = _b(628, 113, 660, 285)
    bub = {"minX": 575, "minY": 85, "maxX": 750, "maxY": 305}
    baseline = fuse([c0, c1, c2, c3], [bub])
    out = fuse([c0, c1, c2, near_dup_of_c2, c3], [bub])
    assert [_bbox(b) for b in out] == [_bbox(b) for b in baseline]


# --- BUG D3: a narrow ruby/furigana box flush against a tall kanji column is ---
# --- not interleaved into fused reading order as its own column ---------------
#
# Real geometry from the audit: ruby gloss "あゆむ" (~22x60px) sits flush
# beside the taller kanji column "息子を視界に" (~33x159px, height ratio
# ~0.38) it annotates. Naive fusion treats the ruby as an ordinary column and
# fuses its bbox INTO the kanji union -- OCR then reads both regions of the
# resulting crop and interleaves the ruby reading ahead of the sentence
# (``あゆむ`` + ``息子を視界に`` + ...), garbling the JP. Because bbox union
# is associative, merging ruby+kanji together first is a NO-OP (produces the
# same final crop as not special-casing it at all) -- the only fix that
# actually changes the crop is EXCLUDING the pairing so the ruby's pixels
# never enter the kanji's fused crop; the ruby surfaces as its own tiny
# isolated block instead (a low-value/blank crop, but not sentence-garbling).

def test_ruby_gloss_excluded_from_kanji_fusion():
    ruby = _b(495, 480, 517, 540)     # あゆむ -- short (60px), narrow (22px)
    kanji = _b(469, 497, 502, 656)    # 息子を視界に -- tall (159px), width 33px
    bub = {"minX": 460, "minY": 470, "maxX": 525, "maxY": 665}
    out = fuse([ruby, kanji], [bub])
    boxes = [_bbox(b) for b in out]
    # Two SEPARATE blocks: the ruby untouched on its own, the kanji untouched
    # on its own -- crucially NOT one union block (which would still hand OCR
    # a crop containing both regions).
    assert len(out) == 2, boxes
    assert _bbox(ruby) in boxes
    assert _bbox(kanji) in boxes


def test_ruby_gloss_does_not_block_kanji_fusing_with_real_neighbour():
    # The kanji column must still fuse normally with a genuine FURTHER
    # dialogue column of the SAME balloon -- excluding the ruby pairing must
    # not also prevent the kanji's OTHER legitimate fusion.
    ruby = _b(495, 480, 517, 540)
    kanji = _b(469, 497, 502, 656)
    next_col = _b(433, 500, 467, 660)  # further dialogue column, same balloon
    bub = {"minX": 420, "minY": 470, "maxX": 525, "maxY": 665}
    out = fuse([ruby, kanji, next_col], [bub])
    boxes = [_bbox(b) for b in out]
    # Ruby stays solo; kanji+next_col fuse into ONE union WITHOUT the ruby.
    assert len(out) == 2, boxes
    assert _bbox(ruby) in boxes
    assert (433, 497, 502, 660) in boxes


def test_real_short_trailing_column_is_not_treated_as_ruby():
    # Sanity/conservatism check: a genuine (merely shorter) trailing column
    # -- height ratio ~0.62, well above the ruby cutoff (0.5) -- still fuses
    # normally as an ordinary column, not specially "absorbed".
    col_a = _b(1068, 573, 1098, 707)   # height 134
    col_b = _b(1032, 578, 1066, 794)   # height 216 (ratio 134/216 ~ 0.62)
    bub = {"minX": 1000, "minY": 560, "maxX": 1110, "maxY": 800}
    out = fuse([col_a, col_b], [bub])
    assert len(out) == 1
    assert _bbox(out[0]) == (1032, 573, 1098, 794)

"""Unit tests for P2-3: erase-without-replace guard + over-broad mask clamp.

Two QA failures motivate these guards (both in lama_inpaint_service):

  1. Page 073 (cover-illustration page). The ONLY detected text is the editorial
     margin label "表紙用イラスト" ("for cover use"). The pipeline translated it
     ("Cover Illustration") and ERASED it, but typeset nothing back, leaving a
     gray smear in the top margin. The human reference LEAVES such editorial /
     margin labels intact. So a region must only be erased when real replacement
     English will be drawn over it (translation non-empty AND not gate-dropped
     AND not a leave-intact label), OR when it is genuine JP ink we explicitly
     want gone (``should_erase_dropped`` reference).

  2. Page 001 (cover). An over-broad inpaint box produced a large rectangular
     gray/blurred smear over artwork. The clamp skips a mask component whose
     bbox area is wildly larger than the ink it actually contains, or larger
     than a sane fraction of the page.

``should_inpaint_region`` is the erase decision; ``_mask_box_too_broad`` is the
per-component over-large clamp used inside ``inpaint()``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.services.lama_inpaint_service import (
    _MASK_BOX_PAGE_FRACTION_CAP,
    _MASK_BOX_INK_RATIO_CAP,
    _mask_box_too_broad,
    should_inpaint_region,
)

_BENCH = Path(
    "/home/danny/Documents/personal/extension/backend/.bench/full_pipeline/588828_mesu2_insp"
)


# --------------------------------------------------------------------------- #
# should_inpaint_region — erase-without-replace guard
# --------------------------------------------------------------------------- #


def test_normal_dialogue_with_translation_is_erased():
    """Real dialogue with a translation that WILL be typeset -> erase."""
    assert should_inpaint_region(
        translation="I'm the wife.", jp="私は妻で", was_dropped=False
    ) is True


def test_073_editorial_label_is_preserved_even_with_translation():
    """073 case: 表紙用イラスト was translated to "Cover Illustration" but is an
    editorial margin label the human reference leaves intact -> do NOT erase."""
    assert should_inpaint_region(
        translation="Cover Illustration", jp="表紙用イラスト", was_dropped=False
    ) is False


def test_empty_translation_non_ink_is_not_erased():
    """No replacement text and the JP is not real ink worth erasing -> preserve
    (erasing without replacing damages the page)."""
    assert should_inpaint_region(translation="", jp="!?", was_dropped=False) is False
    assert should_inpaint_region(translation=None, jp="...", was_dropped=False) is False


def test_dropped_real_jp_ink_is_still_erased():
    """A gate-dropped region with no translation but genuine JP ink should still
    be erased (matches should_erase_dropped) so raw Japanese is not rendered."""
    # Long real-JP line, dropped -> should_erase_dropped returns True.
    assert should_inpaint_region(
        translation="", jp="うるさいですね本当に", was_dropped=True
    ) is True


def test_dropped_garble_only_is_not_erased():
    """Dropped Latin/garble-only crop (no JP glyph) -> leave alone."""
    assert should_inpaint_region(
        translation="", jp="[]/\\", was_dropped=True
    ) is False


def test_whitespace_translation_treated_as_empty():
    assert should_inpaint_region(
        translation="   ", jp="表紙用イラスト", was_dropped=False
    ) is False


@pytest.mark.skipif(not _BENCH.exists(), reason="bench data not present")
def test_073_label_from_real_bubbles_json():
    data = json.loads((_BENCH / "073" / "bubbles.json").read_text())
    b = data[0]
    assert b["ocr_jp"] == "表紙用イラスト"
    # Real pipeline state: translated, NOT gate-dropped. Must still be preserved.
    assert should_inpaint_region(
        translation=b["translation_en"],
        jp=b["ocr_jp"],
        was_dropped=b["ocr_gate_dropped"],
    ) is False


# --------------------------------------------------------------------------- #
# _mask_box_too_broad — over-broad inpaint clamp (page 001 smear)
# --------------------------------------------------------------------------- #


def test_normal_text_box_not_flagged():
    """Tight dialogue box whose ink fills a normal fraction -> keep (erase)."""
    # 150x800 box, ink fraction ~25% (vertical strokes) -> not too broad.
    assert _mask_box_too_broad(
        box_area=150 * 800, ink_area=int(0.25 * 150 * 800), page_area=1280 * 1807
    ) is False


def test_box_with_tiny_ink_is_flagged():
    """A box far larger than the ink it contains -> too broad (smear risk)."""
    assert _mask_box_too_broad(
        box_area=600 * 600, ink_area=200, page_area=1280 * 1807
    ) is True


def test_box_covering_large_page_fraction_is_flagged():
    """A single box covering > the page-fraction cap -> too broad."""
    page = 1280 * 1807
    big = int((_MASK_BOX_PAGE_FRACTION_CAP + 0.05) * page)
    # Even with plenty of ink, a box this large is pathological.
    assert _mask_box_too_broad(
        box_area=big, ink_area=int(0.5 * big), page_area=page
    ) is True


def test_ink_ratio_cap_boundary():
    """Just below the ink-ratio cap is flagged; just above is kept."""
    page = 1280 * 1807
    box = 100 * 100
    below = int((_MASK_BOX_INK_RATIO_CAP * 0.5) * box)  # very sparse
    above = int(min(1.0, _MASK_BOX_INK_RATIO_CAP * 2.0) * box)
    assert _mask_box_too_broad(box_area=box, ink_area=below, page_area=page) is True
    assert _mask_box_too_broad(box_area=box, ink_area=above, page_area=page) is False


@pytest.mark.skipif(not _BENCH.exists(), reason="bench data not present")
def test_001_real_dialogue_boxes_not_flagged_by_page_fraction():
    """The real 001 vertical-dialogue boxes are each well under the page-fraction
    cap, so the clamp must not reject legitimate dialogue on page area alone."""
    data = json.loads((_BENCH / "001" / "bubbles.json").read_text())
    page = 1280 * 1807
    for b in data:
        bb = b["bbox"]
        box_area = (bb["maxX"] - bb["minX"]) * (bb["maxY"] - bb["minY"])
        # Page-fraction alone (assume box fully inked) must not flag these.
        assert _mask_box_too_broad(
            box_area=box_area, ink_area=box_area, page_area=page
        ) is False

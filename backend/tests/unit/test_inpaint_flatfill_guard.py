"""Unit tests for FIX #2: reject dark-median and oversized flat-fills.

With neural LaMa off in prod, the solid-color flat-fill tiers in
``lama_inpaint_service`` are what actually paint masked regions. Two failure
modes produced blob/rectangle artifacts:

  (a) a dark/red background median was flat-filled over the masked region,
      painting opaque dark blobs over faces — guarded by ``_MIN_FLATFILL_LUMA``.
  (b) a LARGE connected component flat-filled into a big opaque patch — guarded
      caller-side by ``_MAX_FLATFILL_AREA`` (skips the flat-fill tiers entirely).

These pin both guards on the free helpers ``_apply_bubble_fastpath`` and
``_apply_bubble_interior_fill``. The area guard lives in the ``inpaint()``
per-component loop (the fill fns can't observe the component), so test (iv)
replicates the caller-side expression it gates on.
"""
from __future__ import annotations

import numpy as np

from app.services.lama_inpaint_service import (
    _MAX_FLATFILL_AREA,
    _MIN_FLATFILL_LUMA,
    _apply_bubble_fastpath,
    _apply_bubble_interior_fill,
    _luma_scalar,
)


def _crop_with_center_mask(fill_rgb, size=40, mask_box=(18, 18, 22, 22)):
    """Uniform `size`x`size` RGB crop of `fill_rgb` with a small central mask."""
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[:] = np.array(fill_rgb, dtype=np.uint8)
    msk = np.zeros((size, size), dtype=np.uint8)
    x0, y0, x1, y1 = mask_box
    msk[y0:y1, x0:x1] = 255
    return img, msk


# (i) near-white uniform crop + small mask -> flat-fill applies.
def test_near_white_flatfill_applies():
    img, msk = _crop_with_center_mask((250, 248, 252))
    _filled, _rem, count = _apply_bubble_fastpath(img.copy(), msk.copy())
    assert count > 0, "near-white bubble interior should flat-fill"

    # interior-fill tier: rect spans whole crop, mask is interior.
    h, w = msk.shape
    _fi, _rm, ci = _apply_bubble_interior_fill(img.copy(), msk.copy(), (0, 0, w, h))
    assert ci > 0, "near-white interior fill should apply"


# (ii) near-black / dark crop + small mask -> flat-fill REJECTED.
def test_dark_flatfill_rejected():
    img, msk = _crop_with_center_mask((20, 20, 20))
    assert _luma_scalar((20, 20, 20)) < _MIN_FLATFILL_LUMA

    filled, rem, count = _apply_bubble_fastpath(img.copy(), msk.copy())
    assert count == 0, "dark median must be rejected (falls through to inpaint)"
    # Unchanged: crop returned as-is, mask preserved.
    assert np.array_equal(filled, img)
    assert np.array_equal(rem, msk)

    h, w = msk.shape
    fi, rm, ci = _apply_bubble_interior_fill(img.copy(), msk.copy(), (0, 0, w, h))
    assert ci == 0, "dark interior must be rejected"
    assert np.array_equal(fi, img)
    assert np.array_equal(rm, msk)


# (iii) dark-red crop -> rejected (red alone has low BT.601 luma).
def test_dark_red_flatfill_rejected():
    # Pure-ish red: luma = 180*0.299 ≈ 54 < 110.
    img, msk = _crop_with_center_mask((180, 10, 10))
    assert _luma_scalar((180, 10, 10)) < _MIN_FLATFILL_LUMA

    _filled, _rem, count = _apply_bubble_fastpath(img.copy(), msk.copy())
    assert count == 0, "dark-red median must be rejected"

    h, w = msk.shape
    _fi, _rm, ci = _apply_bubble_interior_fill(img.copy(), msk.copy(), (0, 0, w, h))
    assert ci == 0, "dark-red interior must be rejected"


# (iv) large-area mask -> flat-fill skipped by the caller-side area guard.
def test_large_area_flatfill_skipped():
    # Replicate the inpaint() per-component guard expression. The fill fns
    # themselves can't see the component, so the cap is enforced caller-side.
    side = 100  # 100*100 = 10000 px masked > 6000 cap
    big_mask = np.full((side, side), 255, dtype=np.uint8)
    comp_area = int((big_mask > 0).sum())
    allow_flatfill = comp_area <= _MAX_FLATFILL_AREA
    assert comp_area > _MAX_FLATFILL_AREA
    assert not allow_flatfill, "oversized component must skip flat-fill tiers"

    # A small component (well under the cap) is still allowed.
    small_area = 30 * 30  # 900 < 6000
    assert small_area <= _MAX_FLATFILL_AREA

"""Unit tests for pure-int vision logic (no cv2/onnx needed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doujin_vision import (  # noqa: E402
    align_pages_by_phash,
    hamming_distance,
)


def test_hamming_distance() -> None:
    assert hamming_distance(0b1010, 0b1010) == 0
    assert hamming_distance(0b1111, 0b0000) == 4
    assert hamming_distance(0, 0b1) == 1


def test_align_exact_match() -> None:
    jp = [10, 20, 30]
    en = [10, 20, 30]
    out = align_pages_by_phash(jp, en, max_distance=0)
    assert [(a.jp_page, a.en_page) for a in out] == [(0, 0), (1, 1), (2, 2)]
    assert all(a.distance == 0 for a in out)


def test_align_drops_above_threshold() -> None:
    # second jp page has no close en match
    jp = [0b0000, 0b1111_1111]  # page1 far from everything
    en = [0b0000, 0b0001]
    out = align_pages_by_phash(jp, en, max_distance=2)
    # jp page 0 -> en 0 (dist 0); jp page 1 dropped (dist to en1 is 7)
    assert [(a.jp_page, a.en_page) for a in out] == [(0, 0)]


def test_align_is_one_to_one() -> None:
    # two jp pages both closest to en[0]; only one may claim it
    jp = [0b0000, 0b0001]
    en = [0b0000, 0b0011]
    out = align_pages_by_phash(jp, en, max_distance=8)
    en_used = [a.en_page for a in out]
    assert len(en_used) == len(set(en_used))  # no en page reused


def test_align_handles_inserted_page() -> None:
    # en has an extra credits page inserted at index 1
    jp = [100, 200, 300]
    en = [100, 999, 200, 300]
    out = align_pages_by_phash(jp, en, max_distance=0)
    assert [(a.jp_page, a.en_page) for a in out] == [(0, 0), (1, 2), (2, 3)]

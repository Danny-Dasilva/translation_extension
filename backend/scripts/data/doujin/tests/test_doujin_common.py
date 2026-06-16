"""Unit tests for the PURE logic of the doujin pipeline.

Run:
    backend/.venv/bin/python -m pytest scripts/data/doujin/tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# make `doujin_common` importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doujin_common import (  # noqa: E402
    DoujinMeta,
    SrcRef,
    format_src,
    manga_reading_order,
    parse_gallery_name,
    parse_src,
    parse_ubuca_filename,
    slugify,
)


# --------------------------------------------------------------------------- #
# src format round-trip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "workid,page,idx",
    [
        ("g123-456", 0, 0),
        ("sono-koe-wo-kikitai-pikohan", 12, 7),
        ("untitled", 999, 3),
    ],
)
def test_src_round_trip(workid: str, page: int, idx: int) -> None:
    s = format_src(workid, page, idx)
    ref = parse_src(s)
    assert ref == SrcRef(workid=workid, page=page, idx=idx)


def test_src_string_shape() -> None:
    assert format_src("g1-2", 3, 4) == "doujin:g1-2:p3:b4"


def test_src_rejects_colon_in_workid() -> None:
    with pytest.raises(ValueError):
        format_src("bad:id", 0, 0)


def test_src_rejects_negative() -> None:
    with pytest.raises(ValueError):
        format_src("ok", -1, 0)
    with pytest.raises(ValueError):
        format_src("ok", 0, -1)


@pytest.mark.parametrize(
    "bad",
    ["", "doujin:x", "doujin:x:p1", "doujin:x:p1:b", "nope:x:p1:b2", "doujin:x:pa:b2"],
)
def test_parse_src_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_src(bad)


# --------------------------------------------------------------------------- #
# slugify
# --------------------------------------------------------------------------- #


def test_slugify_basic() -> None:
    assert slugify("  Sono Koe wo Kikitai. ") == "sono-koe-wo-kikitai"
    assert slugify("32 Year-Old Unsatisfied Wife") == "32-year-old-unsatisfied-wife"
    assert slugify("___") == ""


# --------------------------------------------------------------------------- #
# ubuca filename parsing  (real corpus samples)
# --------------------------------------------------------------------------- #


def test_parse_ubuca_english_flag() -> None:
    m = parse_ubuca_filename("A Maiden's Heart [English].zip")
    assert m.is_english is True
    assert m.pairing_key == "a-maiden-s-heart"


def test_parse_ubuca_eng_variants() -> None:
    for fn in (
        "F-NERD [ENG].zip",
        "First Impression [ENG].zip",
        "For Beautys Sake - Karasu [ English ].zip",
    ):
        assert parse_ubuca_filename(fn).is_english is True


def test_parse_ubuca_circle_artist_parody() -> None:
    fn = "(Kimi no Sentaku.) [Yojihan! (Pikohan)] Sono Koe wo Kikitai. (Undertale) [English] [Uncle Bane].zip"
    m = parse_ubuca_filename(fn)
    assert m.is_english is True
    assert "sono-koe-wo-kikitai" in m.pairing_key
    # language token must NOT leak into the title/key
    assert "english" not in m.pairing_key


def test_parse_ubuca_artist_dash_title() -> None:
    m = parse_ubuca_filename("Ashiomi Masato - Pink Links[QBtranslations].zip")
    assert m.artist.lower().startswith("ashiomi")
    assert "pink-links" in m.pairing_key


def test_parse_ubuca_non_english() -> None:
    m = parse_ubuca_filename("Casual Inevitability Contact [Kirie Masanobu].zip")
    assert m.is_english is False


def test_pairing_key_matches_across_language_tags() -> None:
    en = parse_ubuca_filename("Pink Links [English].zip")
    jp = parse_ubuca_filename("Pink Links.zip")
    assert en.pairing_key == jp.pairing_key


def test_workid_has_no_colon() -> None:
    m = parse_ubuca_filename("Some Title - Artist Name [English].zip")
    assert ":" not in m.workid
    # workid must be a valid src component
    format_src(m.workid, 0, 0)


# --------------------------------------------------------------------------- #
# gallery name parsing
# --------------------------------------------------------------------------- #


def test_parse_gallery_name() -> None:
    g = parse_gallery_name("1000018_en")
    assert g is not None
    assert g.gallery_id == "1000018"
    assert g.lang == "en"


def test_parse_gallery_name_jp() -> None:
    g = parse_gallery_name("1000086_jp")
    assert g.lang == "jp"


def test_parse_gallery_name_invalid() -> None:
    assert parse_gallery_name("not_a_gallery") is None
    assert parse_gallery_name("1000018_fr") is None
    assert parse_gallery_name("1000018") is None


# --------------------------------------------------------------------------- #
# manga reading order
# --------------------------------------------------------------------------- #


def _box(xmin, ymin, xmax, ymax, tag):
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "tag": tag}


def test_reading_order_single() -> None:
    b = [_box(0, 0, 10, 10, "a")]
    assert manga_reading_order(b) == b


def test_reading_order_right_to_left() -> None:
    # two columns: right column should come first
    left = _box(10, 10, 60, 60, "left")
    right = _box(500, 10, 560, 60, "right")
    out = manga_reading_order([left, right])
    assert [b["tag"] for b in out] == ["right", "left"]


def test_reading_order_top_to_bottom_within_column() -> None:
    top = _box(500, 10, 560, 60, "top")
    bot = _box(505, 400, 565, 460, "bot")
    out = manga_reading_order([bot, top])
    assert [b["tag"] for b in out] == ["top", "bot"]


def test_reading_order_full_page() -> None:
    # right column (top,bottom) then left column (top,bottom)
    boxes = [
        _box(10, 400, 60, 460, "L2"),
        _box(500, 10, 560, 60, "R1"),
        _box(10, 10, 60, 60, "L1"),
        _box(505, 400, 565, 460, "R2"),
    ]
    out = manga_reading_order(boxes)
    assert [b["tag"] for b in out] == ["R1", "R2", "L1", "L2"]


def test_reading_order_preserves_objects() -> None:
    b1 = _box(500, 10, 560, 60, "a")
    b2 = _box(10, 10, 60, 60, "b")
    out = manga_reading_order([b1, b2])
    # identity preserved (caller's dicts returned, not copies)
    assert out[0] is b1
    assert out[1] is b2

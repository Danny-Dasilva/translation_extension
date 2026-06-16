"""Light wiring tests: modules import, BubblePair.src round-trips, row schema.

These avoid loading any model — they only exercise the pure dataclass/row logic
so they run in CI without GPU/onnx.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align_and_ocr import BubblePair  # noqa: E402
from build_doujin_pairs import REGISTER_TAG, bubble_to_row  # noqa: E402
from doujin_common import parse_src  # noqa: E402


def _bubble(workid, page, idx, jp="あ", en="ah"):
    return BubblePair(
        workid=workid,
        page=page,
        idx=idx,
        jp=jp,
        en=en,
        jp_bbox=(0, 0, 10, 10),
        en_bbox=(0, 0, 10, 10),
    )


def test_bubble_src_round_trip() -> None:
    b = _bubble("g1-2", 3, 5)
    ref = parse_src(b.src)
    assert (ref.workid, ref.page, ref.idx) == ("g1-2", 3, 5)


def test_bubble_to_row_schema() -> None:
    row = bubble_to_row(_bubble("g7-8", 0, 0), qe_score=0.83)
    assert set(row) == {"jp", "en", "src", "register_tag", "gold_flag", "qe_score"}
    assert row["register_tag"] == REGISTER_TAG
    assert row["gold_flag"] is False
    assert row["qe_score"] == 0.83
    assert row["src"] == "doujin:g7-8:p0:b0"


def test_page_grouping_recoverable_from_src() -> None:
    # emit 2 pages x 2 bubbles; downstream must regroup by (workid,page) order idx
    bubbles = [
        _bubble("w", 0, 1),
        _bubble("w", 0, 0),
        _bubble("w", 1, 0),
    ]
    rows = [bubble_to_row(b, None) for b in bubbles]
    groups: dict[tuple[str, int], list[int]] = {}
    for r in rows:
        ref = parse_src(r["src"])
        groups.setdefault((ref.workid, ref.page), []).append(ref.idx)
    assert sorted(groups[("w", 0)]) == [0, 1]
    assert groups[("w", 1)] == [0]

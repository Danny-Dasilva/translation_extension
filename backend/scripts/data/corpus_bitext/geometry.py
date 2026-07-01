"""Bbox geometry + reading-order helpers for cross-page bubble alignment.

A *bubble* in this pipeline is a dict::

    {"idx": int, "text": str, "bbox": {"minX","minY","maxX","maxY"}, "conf": float}

JP and EN bubbles live on DIFFERENT page images (raw vs English redraw), so all
spatial comparison is done on NORMALIZED coordinates (each page's own pixel size
mapped to the unit square). The redraw approximately preserves panel/bubble
layout, so a bubble's normalized centroid is the primary cross-page anchor.

Reading order reuses the EXACT training-side ``manga_reading_order`` from
``build_v11_dataset`` (column-major, right-to-left, top-to-bottom) so the page
context we emit is ordered identically to the v11 training data.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[3]  # .../backend
sys.path.insert(0, str(_BACKEND / "scripts" / "data" / "v11"))

from build_v11_dataset import manga_reading_order  # noqa: E402  (training-exact)


# --------------------------------------------------------------------------- #
# bbox primitives  (all take a {"minX","minY","maxX","maxY"} dict)
# --------------------------------------------------------------------------- #
def bbox_wh(b: dict) -> tuple[float, float]:
    bb = b["bbox"]
    return (max(0.0, bb["maxX"] - bb["minX"]), max(0.0, bb["maxY"] - bb["minY"]))


def bbox_center(b: dict) -> tuple[float, float]:
    bb = b["bbox"]
    return ((bb["minX"] + bb["maxX"]) / 2.0, (bb["minY"] + bb["maxY"]) / 2.0)


def bbox_area(b: dict) -> float:
    w, h = bbox_wh(b)
    return w * h


def iou(a: dict, b: dict) -> float:
    """Pixel-space IoU of two bubbles' bboxes (assumes same coordinate frame).

    Used for the auxiliary SAME-FRAME sanity check and for de-duplication; NOT
    for cross-page matching (cross-page uses normalized centroid distance).
    """
    A, B = a["bbox"], b["bbox"]
    ix0, iy0 = max(A["minX"], B["minX"]), max(A["minY"], B["minY"])
    ix1, iy1 = min(A["maxX"], B["maxX"]), min(A["maxY"], B["maxY"])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = bbox_area(a) + bbox_area(b) - inter
    return inter / ua if ua > 0 else 0.0


def page_dims(bubbles: list[dict], fallback: tuple[float, float] | None = None) -> tuple[float, float]:
    """Estimate (W, H) for a page from its bubbles when the true image size is
    unknown. Uses the max extent; a small margin avoids a centroid sitting on the
    unit-square edge. Falls back to ``fallback`` (or (1,1)) for an empty page.
    """
    if not bubbles:
        return fallback or (1.0, 1.0)
    W = max(b["bbox"]["maxX"] for b in bubbles)
    H = max(b["bbox"]["maxY"] for b in bubbles)
    return (float(max(W, 1.0)), float(max(H, 1.0)))


def norm_center(b: dict, W: float, H: float) -> tuple[float, float]:
    cx, cy = bbox_center(b)
    return (cx / W, cy / H)


def norm_wh(b: dict, W: float, H: float) -> tuple[float, float]:
    w, h = bbox_wh(b)
    return (w / W, h / H)


def centroid_dist_norm(a: dict, Wa: float, Ha: float, b: dict, Wb: float, Hb: float) -> float:
    """Euclidean distance between two bubbles' centroids in normalized space.

    Each bubble is normalized by ITS OWN page dimensions, so this compares the
    relative position on each page (robust to the JP/EN pages being scanned at
    different resolutions).
    """
    ax, ay = norm_center(a, Wa, Ha)
    bx, by = norm_center(b, Wb, Hb)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


# --------------------------------------------------------------------------- #
# reading order  (training-exact, via build_v11_dataset.manga_reading_order)
# --------------------------------------------------------------------------- #
def reading_order(bubbles: list[dict]) -> list[dict]:
    """Return ``bubbles`` reordered into manga reading order (column-major RTL).

    Delegates to the training-side ``manga_reading_order`` so the page context
    order is byte-identical to the v11 training data. The bubbles keep their
    original dicts (we proxy the xmin/ymin/xmax/ymax keys it expects)."""
    if len(bubbles) <= 1:
        return list(bubbles)
    proxies = [
        {
            "xmin": b["bbox"]["minX"],
            "ymin": b["bbox"]["minY"],
            "xmax": b["bbox"]["maxX"],
            "ymax": b["bbox"]["maxY"],
            "_orig": b,
        }
        for b in bubbles
    ]
    ordered = manga_reading_order(proxies)
    return [p["_orig"] for p in ordered]


def reading_order_ranks(bubbles: list[dict]) -> dict[int, int]:
    """Map each bubble's id() to its 0-based rank in reading order."""
    ordered = reading_order(bubbles)
    return {id(b): r for r, b in enumerate(ordered)}

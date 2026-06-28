"""Skeleton: aspect-ratio-based vertical→AR OCR routing.

PLAN-STAGE REFERENCE IMPLEMENTATION (fix7-parseq-vertical). This file documents
*where* and *how* the vertical→AR routing hooks into the production OCR service.
It is intentionally a thin, self-contained reference: the production change lands
inside ``ParseqOCRService._recognize_batch_with_conf``
(``backend/app/services/parseq_ocr_service.py``); this script mirrors that logic
so it can be unit-tested and ablated in isolation.

--------------------------------------------------------------------------------
WHY
--------------------------------------------------------------------------------
The dominant Ikenie-4 failure is dense/stylized VERTICAL kana garble produced by
the **non-autoregressive (NAR)** PARSeq decode: duplicated/substituted adjacent
glyphs (身代わり→身身わわ, 吐気→吐吐気) at falsely-HIGH confidence (0.76-0.92),
so the existing low-confidence AR-RETRY never fires. The **autoregressive (AR)**
dynbatch export (``models/parseq_manga_ep60_AR_dynbatch.onnx``, same weights,
dynamic batch) does not exhibit this — its sequential decode cannot fall into the
parallel-decode duplication loops. See:
  thoughts/shared/plans/fix7-parseq-vertical-kana-hardening.md  (§2, §3)

So instead of "NAR-all → AR-retry only the low-confidence crops", we route
**tall/narrow vertical crops to AR up front (by default)** and keep the cheaper
NAR path for horizontal crops.

--------------------------------------------------------------------------------
ROUTING TRIGGER (crop geometry, pre-rotate)
--------------------------------------------------------------------------------
    is_vertical(crop) := (h / w) >= VERTICAL_AR_ASPECT

VERTICAL_AR_ASPECT defaults to 1.5 — the SAME threshold ParseqOCRService uses in
``_maybe_rotate_vertical`` — so the set "rotated as vertical text" is identical to
the set "routed to AR". Exposed as config knobs:
    settings.parseq_vertical_ar_routing : bool   (master switch)
    settings.parseq_vertical_ar_aspect  : float  (default 1.5)

--------------------------------------------------------------------------------
INTEGRATION POINT
--------------------------------------------------------------------------------
``ParseqOCRService._recognize_batch_with_conf(image_crops, batch_size)`` currently:
    1. NAR-batch over ALL crops              -> out[i] = (text, conf)
    2. AR-retry over {i : conf < threshold}  -> replace out[i]
    3. (downstream) garble gate drops residual illegible SFX

This skeleton's ``partition_by_aspect`` + ``plan_routes`` replace step 1-2 with:
    1. partition crops -> vertical_idx / horizontal_idx
    2. AR-batch  vertical_idx     (the garble-prone set, recovered not dropped)
    3. NAR-batch horizontal_idx   (clean & cheap)
    4. (optional, retained) AR-retry any horizontal crop still < threshold
    5. stitch back into original order; garble gate runs unchanged

Run (smoke test, no model/GPU needed):
    backend/.venv/bin/python backend/scripts/ocr/route_vertical_to_ar.py --demo
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

# Mirror ParseqOCRService._maybe_rotate_vertical's threshold so the
# "rotate-as-vertical" set and the "route-to-AR" set are identical.
DEFAULT_VERTICAL_AR_ASPECT: float = 1.5


def is_vertical_crop(crop: np.ndarray, aspect: float = DEFAULT_VERTICAL_AR_ASPECT) -> bool:
    """True if ``crop`` is tall/narrow enough to route to the AR model.

    Uses the RAW (pre-rotate) crop geometry: a manga vertical-text line is taller
    than it is wide. ``aspect`` is height/width; >= 1.5 matches
    ``ParseqOCRService._maybe_rotate_vertical(thresh_aspect=1.5)``.

    A zero/degenerate-width crop is treated as vertical (safer to send the harder
    model). Empty crops should be filtered upstream.
    """
    if crop is None or crop.ndim < 2:
        return False
    h, w = crop.shape[:2]
    if w <= 0:
        return True
    return (h / float(w)) >= aspect


def partition_by_aspect(
    crops: Sequence[np.ndarray],
    aspect: float = DEFAULT_VERTICAL_AR_ASPECT,
) -> tuple[list[int], list[int]]:
    """Split crop indices into (vertical_idx, horizontal_idx).

    ``vertical_idx``  -> route to AR  (garble-prone dense vertical kana).
    ``horizontal_idx``-> route to NAR (clean, fast).
    Order within each list preserves the original crop order so results can be
    stitched back deterministically.
    """
    vertical_idx: list[int] = []
    horizontal_idx: list[int] = []
    for i, crop in enumerate(crops):
        (vertical_idx if is_vertical_crop(crop, aspect) else horizontal_idx).append(i)
    return vertical_idx, horizontal_idx


@dataclass
class RoutePlan:
    """A computed routing decision over a page's crops (no inference run yet)."""

    n_crops: int
    aspect: float
    vertical_idx: list[int] = field(default_factory=list)    # -> AR
    horizontal_idx: list[int] = field(default_factory=list)  # -> NAR

    @property
    def n_ar(self) -> int:
        return len(self.vertical_idx)

    @property
    def n_nar(self) -> int:
        return len(self.horizontal_idx)

    def summary(self) -> str:
        return (
            f"{self.n_crops} crops -> AR(vertical)={self.n_ar} "
            f"NAR(horizontal)={self.n_nar} @aspect>={self.aspect}"
        )


def plan_routes(
    crops: Sequence[np.ndarray],
    enabled: bool = True,
    aspect: float = DEFAULT_VERTICAL_AR_ASPECT,
) -> RoutePlan:
    """Compute the AR/NAR split for a batch of crops.

    When ``enabled`` is False (``settings.parseq_vertical_ar_routing == False``)
    every crop is sent to NAR — i.e. exactly the legacy "NAR-all" behavior, so the
    flag is a safe no-op kill switch.
    """
    if not enabled:
        return RoutePlan(n_crops=len(crops), aspect=aspect,
                         vertical_idx=[], horizontal_idx=list(range(len(crops))))
    v, h = partition_by_aspect(crops, aspect)
    return RoutePlan(n_crops=len(crops), aspect=aspect, vertical_idx=v, horizontal_idx=h)


# ---------------------------------------------------------------------------
# Reference recognize flow (decode-agnostic). The real version lives in
# ParseqOCRService._recognize_batch_with_conf and reuses _preprocess / _run_sync
# / _run_ar_sync / _decode_with_conf / the AR OOM-halving guard already present
# in the service.
# ---------------------------------------------------------------------------
def route_and_recognize(
    crops: Sequence[np.ndarray],
    nar_recognize: Callable[[Sequence[np.ndarray]], list[tuple[str, float]]],
    ar_recognize: Callable[[Sequence[np.ndarray]], list[tuple[str, float]]],
    *,
    enabled: bool = True,
    aspect: float = DEFAULT_VERTICAL_AR_ASPECT,
) -> list[tuple[str, float]]:
    """Partition -> AR(vertical) / NAR(horizontal) -> stitch into original order.

    ``nar_recognize`` / ``ar_recognize`` are injected so this is testable without
    a GPU. In production they are bound to the NAR and AR ONNX sessions.

    Returns ``out[i] = (text, conf)`` aligned to the input crop order. The
    downstream garble gate (ocr_confidence_gate.is_garbled_low_conf /
    is_implausible_japanese) runs unchanged on these results, so genuinely
    illegible SFX still drop after the AR pass.

    TODO(fix7): port this body into ParseqOCRService._recognize_batch_with_conf:
      - TODO: gate on settings.parseq_vertical_ar_routing / parseq_vertical_ar_aspect.
      - TODO: AR batch must go through _ensure_ar_session(); on AR-load failure,
              fall back to NAR for the vertical set (do NOT drop the page).
      - TODO: reuse the existing AR OOM-halving guard (_ar_retry's batch//2 loop)
              for the larger AR-default batch.
      - TODO: keep the optional low-conf AR-retry for horizontal crops (step 4).
      - TODO: emit a per-page log line: "OCR route: AR(vert)=N NAR(horiz)=M
              ar_ms=.. nar_ms=.." for the §5 latency budget.
      - TODO: ensure orphan-line / per-line stitching order in
              recognize_blocks_with_lines is unaffected (indices are absolute).
    """
    plan = plan_routes(crops, enabled=enabled, aspect=aspect)
    out: list[tuple[str, float]] = [("", 0.0)] * plan.n_crops

    if plan.horizontal_idx:
        nar_res = nar_recognize([crops[i] for i in plan.horizontal_idx])
        for i, r in zip(plan.horizontal_idx, nar_res):
            out[i] = r

    if plan.vertical_idx:
        # TODO(fix7): wrap in try/except -> on AR failure, NAR-recognize this set.
        ar_res = ar_recognize([crops[i] for i in plan.vertical_idx])
        for i, r in zip(plan.vertical_idx, ar_res):
            out[i] = r

    return out


def _demo() -> None:
    """Smoke test with synthetic crops (no ONNX, no GPU)."""
    rng = np.random.default_rng(0)
    # 3 tall vertical crops (h>w -> AR), 2 wide horizontal crops (NAR).
    crops = [
        rng.integers(0, 255, (200, 60, 3), dtype=np.uint8),   # vertical
        rng.integers(0, 255, (300, 50, 3), dtype=np.uint8),   # vertical
        rng.integers(0, 255, (180, 40, 3), dtype=np.uint8),   # vertical
        rng.integers(0, 255, (40, 200, 3), dtype=np.uint8),   # horizontal
        rng.integers(0, 255, (60, 240, 3), dtype=np.uint8),   # horizontal
    ]
    plan = plan_routes(crops)
    print(plan.summary())
    assert plan.vertical_idx == [0, 1, 2], plan.vertical_idx
    assert plan.horizontal_idx == [3, 4], plan.horizontal_idx

    out = route_and_recognize(
        crops,
        nar_recognize=lambda cs: [("NAR", 0.9)] * len(cs),
        ar_recognize=lambda cs: [("AR", 0.95)] * len(cs),
    )
    print("routed results:", out)
    assert [t for t, _ in out] == ["AR", "AR", "AR", "NAR", "NAR"], out

    # Kill switch -> all NAR (legacy behavior).
    plan_off = plan_routes(crops, enabled=False)
    assert plan_off.n_ar == 0 and plan_off.n_nar == len(crops)
    print("kill-switch OK:", plan_off.summary())
    print("demo OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="run the no-GPU smoke test")
    ap.add_argument("--aspect", type=float, default=DEFAULT_VERTICAL_AR_ASPECT,
                    help="height/width threshold to route a crop to AR (default 1.5)")
    args = ap.parse_args()
    if args.demo:
        _demo()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

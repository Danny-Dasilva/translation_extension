"""Cross-page JP <-> EN bubble alignment.

The JP page (raw) and EN page (English redraw) are DIFFERENT images. The redraw
approximately preserves the panel / bubble layout, so a JP bubble's normalized
centroid is close to its EN counterpart's. We pose alignment as an optimal
bipartite assignment:

  cost(jp_i, en_j) = centroid_dist                 (primary: normalized position)
                   + size_weight * size_diff       (bubble shape agreement)
                   + order_weight * |rank_i-rank_j| (reading-order tie-break)

solved with the Hungarian algorithm (scipy.linear_sum_assignment). A pair is
KEPT only when its normalized centroid distance is within ``tol`` — count
mismatches (merged / split bubbles, untranslated SFX present on only one side)
therefore drop out as unmatched, which favors PRECISION (the roadmap says
translation-SFT plateaus at ~1-10k pairs, so wrong pairs hurt more than missing
ones).

A greedy mutual-nearest-neighbour fallback is provided if scipy is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass

from geometry import (  # type: ignore  (sibling module; run from this dir)
    centroid_dist_norm,
    norm_wh,
    page_dims,
    reading_order_ranks,
)

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


@dataclass
class AlignConfig:
    tol: float = 0.08          # max normalized centroid distance to KEEP a match
    size_weight: float = 0.25  # weight on |Δw|+|Δh| (normalized) in the cost
    order_weight: float = 0.10  # weight on normalized reading-order rank gap
    big_cost: float = 9.0      # sentinel for the rectangular padding


@dataclass
class Match:
    jp_idx: int       # index into the JP bubble list
    en_idx: int       # index into the EN bubble list
    match_dist: float  # NORMALIZED centroid distance (the interpretable gate)
    cost: float        # full assignment cost (dist + size + order)


def _cost_matrix(jp, en, cfg: AlignConfig):
    Wj, Hj = page_dims(jp)
    We, He = page_dims(en)
    jp_rank = reading_order_ranks(jp)
    en_rank = reading_order_ranks(en)
    nj, ne = len(jp), len(en)
    rank_norm_j = (nj - 1) or 1
    rank_norm_e = (ne - 1) or 1

    import math

    dist = [[0.0] * ne for _ in range(nj)]
    cost = [[0.0] * ne for _ in range(nj)]
    for i, bj in enumerate(jp):
        wj, hj = norm_wh(bj, Wj, Hj)
        ri = jp_rank[id(bj)] / rank_norm_j
        for j, be in enumerate(en):
            d = centroid_dist_norm(bj, Wj, Hj, be, We, He)
            we, he = norm_wh(be, We, He)
            size_diff = abs(wj - we) + abs(hj - he)
            rj = en_rank[id(be)] / rank_norm_e
            order_diff = abs(ri - rj)
            c = d + cfg.size_weight * size_diff + cfg.order_weight * order_diff
            dist[i][j] = d
            cost[i][j] = c
    return dist, cost


def _assign_hungarian(cost):
    import numpy as np

    C = np.asarray(cost, dtype=float)
    rows, cols = linear_sum_assignment(C)
    return list(zip(rows.tolist(), cols.tolist()))


def _assign_greedy(cost):
    """Mutual-nearest greedy assignment (fallback when scipy is absent)."""
    nj = len(cost)
    ne = len(cost[0]) if nj else 0
    pairs = sorted(
        ((cost[i][j], i, j) for i in range(nj) for j in range(ne)),
        key=lambda t: t[0],
    )
    used_i, used_j, out = set(), set(), []
    for _, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j))
    return out


def align_pages(jp: list[dict], en: list[dict], cfg: AlignConfig | None = None):
    """Align JP bubbles to EN bubbles across two pages.

    Returns ``(matches, jp_unmatched, en_unmatched)`` where ``matches`` is a list
    of :class:`Match` with NORMALIZED centroid ``match_dist <= cfg.tol``. JP/EN
    indices that did not get a within-tolerance partner are returned as the
    unmatched index lists (left out of the bitext — precision over recall).
    """
    cfg = cfg or AlignConfig()
    if not jp or not en:
        return [], list(range(len(jp))), list(range(len(en)))

    dist, cost = _cost_matrix(jp, en, cfg)
    assignment = _assign_hungarian(cost) if _HAVE_SCIPY else _assign_greedy(cost)

    matches, matched_i, matched_j = [], set(), set()
    for i, j in assignment:
        if i >= len(jp) or j >= len(en):
            continue
        d = dist[i][j]
        if d <= cfg.tol:
            matches.append(Match(jp_idx=i, en_idx=j, match_dist=d, cost=cost[i][j]))
            matched_i.add(i)
            matched_j.add(j)
    jp_un = [i for i in range(len(jp)) if i not in matched_i]
    en_un = [j for j in range(len(en)) if j not in matched_j]
    matches.sort(key=lambda m: m.match_dist)
    return matches, jp_un, en_un


def coverage(matches, jp: list[dict], en: list[dict]) -> float:
    """Fraction of the SMALLER side that got matched (page-level alignment
    quality; a low value means the redraw layout diverged or one side is mostly
    SFX -> the whole page is dropped during curation)."""
    denom = max(1, min(len(jp), len(en)))
    return len(matches) / denom

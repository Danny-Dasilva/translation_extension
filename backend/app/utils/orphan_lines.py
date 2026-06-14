"""Orphan-line recovery for the manga-translation pipeline.

A `text_line` whose center sits inside NO detected block is silently dropped
before OCR (the OCR assignment rule keeps only lines a block contains). On a
normal page those orphans — SMS/chat balloons, vertical narration columns,
dense paragraphs the block detector merges away — would otherwise render as
raw Japanese to the reader.

These are PURE geometry helpers (no model/service dependency) shared by the
production router, the chapter renderer, and the e2e visualizer. The OCR step
itself is left to the caller (it owns the OCR service + crops).

Lifted from scripts/visualize_e2e_pipeline.py so the production path and the
visualizer share ONE implementation instead of diverging copies.
"""
from __future__ import annotations

from typing import Dict, List


def find_orphan_lines(
    blocks: List[Dict], text_lines: List[Dict]
) -> List[Dict]:
    """Return text_lines whose center no block contains.

    Mirrors the OCR line->block assignment rule (a line belongs to the first
    block whose bbox contains the line center). Any line not claimed by some
    block is an orphan.
    """
    orphans: List[Dict] = []
    for ln in text_lines:
        cx = (ln["minX"] + ln["maxX"]) / 2
        cy = (ln["minY"] + ln["maxY"]) / 2
        if not any(
            b["minX"] <= cx <= b["maxX"] and b["minY"] <= cy <= b["maxY"]
            for b in blocks
        ):
            orphans.append(ln)
    return orphans


def cluster_orphan_lines(orphans: List[Dict]) -> List[List[Dict]]:
    """Union-find clustering: lines whose bboxes (expanded ~1.2 line-heights)
    intersect belong to one paragraph.

    Merges nearby orphan lines so multi-line vertical narration / a multi-line
    chat balloon becomes ONE synthetic block instead of N single-line blocks.
    """
    n = len(orphans)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def expand(ln: Dict):
        h = ln["maxY"] - ln["minY"]
        w = ln["maxX"] - ln["minX"]
        pad = 1.2 * min(h, w) if min(h, w) > 0 else 12
        return (ln["minX"] - pad, ln["minY"] - pad, ln["maxX"] + pad, ln["maxY"] + pad)

    boxes = [expand(ln) for ln in orphans]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                parent[find(i)] = find(j)

    groups: Dict[int, List[Dict]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(orphans[i])
    return list(groups.values())


def order_cluster_lines(cluster: List[Dict]) -> List[Dict]:
    """Reading order. Horizontal lines (w > h): top-to-bottom. Vertical
    columns: right-to-left, then top-to-bottom — matches manga convention.
    """
    horiz = sum(
        1
        for ln in cluster
        if (ln["maxX"] - ln["minX"]) > (ln["maxY"] - ln["minY"])
    )
    if horiz >= len(cluster) / 2:
        return sorted(cluster, key=lambda ln: (ln["minY"], ln["minX"]))
    return sorted(cluster, key=lambda ln: (-ln["minX"], ln["minY"]))


def cluster_bbox(cluster: List[Dict]) -> Dict:
    """Bounding box (as a synthetic block) covering all lines in a cluster.

    Marked confidence 0.5 + ``orphan`` so downstream rendering / debugging can
    tell recovered blocks apart from detector blocks.
    """
    return {
        "minX": int(min(ln["minX"] for ln in cluster)),
        "minY": int(min(ln["minY"] for ln in cluster)),
        "maxX": int(max(ln["maxX"] for ln in cluster)),
        "maxY": int(max(ln["maxY"] for ln in cluster)),
        "confidence": 0.5,
        "orphan": True,
    }


async def ocr_orphan_clusters(
    ocr_service,
    image_np,
    clusters: List[List[Dict]],
    batch_size: int = 8,
) -> List[str]:
    """OCR each cluster's lines (reading order) and join into one text/cluster.

    The crops are gathered into ONE flat batch across all clusters so the OCR
    service still runs a single batched forward. ``ocr_service`` must expose
    ``recognize_text_batch(crops, batch_size=...)`` (PARSeq / manga-ocr both
    do). Returns a list aligned to ``clusters`` (empty string for empty crops).
    """
    h, w = image_np.shape[:2]
    flat = []
    owner: List[int] = []
    for ci, cluster in enumerate(clusters):
        for ln in order_cluster_lines(cluster):
            x0 = max(0, int(ln["minX"]) - 2)
            y0 = max(0, int(ln["minY"]) - 2)
            x1 = min(w, int(ln["maxX"]) + 2)
            y1 = min(h, int(ln["maxY"]) + 2)
            if x1 > x0 and y1 > y0:
                flat.append(image_np[y0:y1, x0:x1])
                owner.append(ci)
    if not flat:
        return ["" for _ in clusters]
    texts = await ocr_service.recognize_text_batch(flat, batch_size=batch_size)
    joined: List[List[str]] = [[] for _ in clusters]
    for ci, t in zip(owner, texts):
        if t:
            joined[ci].append(t)
    return ["".join(parts) for parts in joined]

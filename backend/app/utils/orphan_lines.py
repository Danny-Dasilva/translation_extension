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


def _join_sep() -> str:
    """Separator for joining per-cluster OCR lines.

    Mirrors ``settings.ocr_line_join_newline`` (defaults to ""); defensive
    local import so these pure-geometry helpers stay import-safe in isolation.
    """
    try:
        from app.config import settings
        return "\n" if getattr(settings, "ocr_line_join_newline", False) else ""
    except Exception:
        return ""


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
        short = min(h, w) if min(h, w) > 0 else 12
        # Text lines in one balloon stack vertically with gaps that can exceed a
        # single line-height (leading/inter-line spacing). Pad the SHORT axis (the
        # stacking direction for both horizontal rows and vertical columns) more
        # generously so consecutive lines of the SAME balloon bridge into ONE
        # cluster instead of rendering as two overlapping synthetic blocks. The
        # long axis keeps a tight pad so distinct adjacent balloons stay apart.
        pad_short = 3.0 * short
        pad_long = 1.2 * short
        if h <= w:  # horizontal line -> stack vertically
            px, py = pad_long, pad_short
        else:       # vertical column -> stack horizontally
            px, py = pad_short, pad_long
        return (ln["minX"] - px, ln["minY"] - py, ln["maxX"] + px, ln["maxY"] + py)

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


def _iou(a: Dict, b: Dict) -> float:
    """Intersection-over-union of two bbox dicts (minX/minY/maxX/maxY)."""
    ix0 = max(a["minX"], b["minX"])
    iy0 = max(a["minY"], b["minY"])
    ix1 = min(a["maxX"], b["maxX"])
    iy1 = min(a["maxY"], b["maxY"])
    iw = ix1 - ix0
    ih = iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0, a["maxX"] - a["minX"]) * max(0, a["maxY"] - a["minY"])
    area_b = max(0, b["maxX"] - b["minX"]) * max(0, b["maxY"] - b["minY"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_in(inner: Dict, outer: Dict) -> bool:
    """True if the center of ``inner`` lies inside ``outer``'s bbox."""
    cx = (inner["minX"] + inner["maxX"]) / 2
    cy = (inner["minY"] + inner["maxY"]) / 2
    return outer["minX"] <= cx <= outer["maxX"] and outer["minY"] <= cy <= outer["maxY"]


def _overlaps(s: Dict, o: Dict, iou_thresh: float = 0.25) -> bool:
    """Significant overlap: IoU above threshold OR either center contained."""
    return _iou(s, o) > iou_thresh or _center_in(s, o) or _center_in(o, s)


def merge_orphans_into_blocks(
    blocks: List[Dict],
    texts: List[str],
    synth_blocks: List[Dict],
    synth_texts: List[str],
    iou_thresh: float = 0.25,
):
    """Resolve synthetic orphan blocks against existing original blocks.

    ``blocks``/``texts`` are the ORIGINAL detector blocks and their OCR'd text
    (parallel lists). ``synth_blocks``/``synth_texts`` are freshly built orphan
    blocks (each carrying ``orphan: True``) and their OCR'd text.

    For each synthetic block S:
      * If S significantly overlaps some original block O (IoU > ``iou_thresh``
        OR either center contained) — MERGE: expand O's bbox to the union of
        O and S, concatenate the two texts in manga reading order (right-to-left
        / top-to-bottom over the union), and DROP S. ONE block then renders the
        combined text over the union region (kills the double-render overlap).
      * Otherwise keep S as a new block (preserves the coverage win for truly
        isolated narration / SMS bubbles). The ``orphan: True`` marker is kept.

    Only synthetic-vs-original overlaps are resolved here; synthetic blocks are
    never merged into each other. Each S merges into AT MOST the first matching
    O. Returns ``(blocks, texts)`` as new appended-to copies of the inputs.

    Reading order for a merged pair uses ``order_cluster_lines`` on two pseudo
    "lines" (the two bboxes) so vertical-JP pages order right-to-left then
    top-to-bottom; the text whose box comes first in that order leads.
    """
    blocks = list(blocks)
    texts = list(texts)
    n_orig = len(blocks)

    for s, st in zip(synth_blocks, synth_texts):
        match_i = -1
        for i in range(n_orig):
            if _overlaps(s, blocks[i], iou_thresh):
                match_i = i
                break
        if match_i < 0:
            # No overlap — keep as a new block (orphan marker preserved).
            blocks.append(s)
            texts.append(st)
            continue

        o = blocks[match_i]
        ot = texts[match_i]
        # Dedup before concatenating: overlapping detections often re-OCR the
        # same text, so an exact/substring match would yield "X X". If one
        # normalized string contains the other (or they are equal), keep the
        # longer/containing one. Normalized exact/substring, not fuzzy.
        no, ns = "".join(ot.split()), "".join(st.split())
        if no and ns and (no == ns or ns in no or no in ns):
            merged_text = ot if len(no) >= len(ns) else st
        else:
            # Order the two boxes in manga reading order; lead text comes first.
            ordered = order_cluster_lines([o, s])
            if ordered and ordered[0] is s:
                merged_text = (st + ot)
            else:
                merged_text = (ot + st)
        blocks[match_i] = {
            **o,
            "minX": int(min(o["minX"], s["minX"])),
            "minY": int(min(o["minY"], s["minY"])),
            "maxX": int(max(o["maxX"], s["maxX"])),
            "maxY": int(max(o["maxY"], s["maxY"])),
            "confidence": o.get("confidence", s.get("confidence", 0.5)),
        }
        texts[match_i] = merged_text

    return blocks, texts


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
    return [_join_sep().join(parts) for parts in joined]


async def ocr_orphan_clusters_with_conf(
    ocr_service,
    image_np,
    clusters: List[List[Dict]],
    batch_size: int = 8,
):
    """Like ``ocr_orphan_clusters`` but also returns per-cluster OCR confidence.

    Returns ``(texts, confs)`` aligned to ``clusters``. Per-cluster confidence
    is the MIN over the cluster's line crops (a single garbled line poisons the
    cluster), so the OCR-confidence garble gate can drop garbled orphan SFX
    instead of defaulting them to "trusted". Empty clusters get conf 0.0.

    Requires ``ocr_service.recognize_text_batch_with_conf``; falls back to
    text-only (conf 1.0) when that method is unavailable (e.g. manga-ocr).
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
        return ["" for _ in clusters], [0.0 for _ in clusters]

    if hasattr(ocr_service, "recognize_text_batch_with_conf"):
        tc = await ocr_service.recognize_text_batch_with_conf(flat, batch_size=batch_size)
    else:
        texts = await ocr_service.recognize_text_batch(flat, batch_size=batch_size)
        tc = [(t, 1.0) for t in texts]

    joined: List[List[str]] = [[] for _ in clusters]
    confs: List[List[float]] = [[] for _ in clusters]
    for ci, (t, c) in zip(owner, tc):
        confs[ci].append(c)
        if t:
            joined[ci].append(t)
    texts_out = [_join_sep().join(parts) for parts in joined]
    confs_out = [min(cs) if cs else 0.0 for cs in confs]
    return texts_out, confs_out

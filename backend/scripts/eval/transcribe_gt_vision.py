#!/usr/bin/env python
"""One-time vision-OCR pass over the human GT scanlation images.

PURPOSE (read this first)
-------------------------
The frozen ``data/ikenie4/gold.jsonl`` produced by ``build_ikenie4_gold.py`` is
seeded from the 24-agent LLM-judge comparison and therefore only covers the
~77 bubbles the judge flagged.  To grow the gold set to ~300 rows we need the
human English for the REST of the bubbles -- the ones the judge did not flag,
including the many *correct* ones (a regression set needs correct rows too, so a
change that breaks a currently-correct bubble is caught).

The only source of that human English is the typeset text baked into the GT
scanlation webp images.  This script is the ONE-TIME human_en recovery step: it
runs a vision model over each GT page, reads the rendered English in each
bubble, and aligns it to our OCR'd ``jp`` bubbles (by page + reading-order /
bbox overlap), then appends the new (jp, human_en) rows to the gold set.

This is a *one-time* pass: once the extended gold.jsonl is committed and
reviewed, it is frozen exactly like the judge-seeded rows.  You re-run this only
to re-derive the gold from scratch, not on every eval.

THE p41 OFFSET (baked in permanently)
-------------------------------------
Our bench pipeline emitted 134 pages (001..134).  The GT directory has 133 webp
(001..133).  The comparison reported ``missing_gt_page: 41`` -- GT page 41 does
not exist, so from bench page 41 onward the GT image index is shifted by +1.

    gt_webp(our_page) = our_page          if our_page < 41
                      = our_page - 1       if our_page >= 41

``resolve_gt_image_path()`` below encodes this so the vision pass reads the
CORRECT GT image for every bench bubble.  This is the single source of truth for
the offset; do not re-derive it ad hoc elsewhere.

RUNNABLE SHAPE
--------------
The script is runnable-shaped but the actual vision call is stubbed
(``_vision_transcribe_page`` raises NotImplementedError with a clear TODO).  You
do NOT need to run the vision model to use the rest of the harness; this is the
documented extension path.  When you do wire a model, fill in that one function
and the alignment in ``align_page``.

Usage (when the vision model is wired)
--------------------------------------
    PYTHONPATH=. python backend/scripts/eval/transcribe_gt_vision.py \
        --bubbles-root /home/danny/Documents/personal/extension/backend/.bench/ikenie4_final_insp \
        --gt-images-root "/mnt/nas/drive_2/onlyfans/external_content/nhentai/616137_Ikenie no Haha 4" \
        --existing-gold backend/scripts/eval/data/ikenie4/gold.jsonl \
        --out backend/scripts/eval/data/ikenie4/gold_vision_extended.jsonl \
        --pages 1-134
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Single source of truth for the offset, shared with the gold builder.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
BACKEND_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(BACKEND_DIR))
from build_ikenie4_gold import (  # noqa: E402
    MISSING_GT_PAGE,
    our_page_to_gt_page,
)

# is_implausible_japanese is the linguistic-plausibility garble guard used by the
# OCR confidence gate; reuse it so vision-gold ocr_clean is consistent with prod.
try:
    from app.utils.ocr_confidence_gate import is_implausible_japanese  # noqa: E402
except Exception:  # pragma: no cover - fallback when app import path unavailable

    def is_implausible_japanese(text: str, ocr_conf: float | None = None) -> bool:  # type: ignore[misc]
        return False


# Box VLM (Qwen2.5-VL-7B served on the training box, model name "qwenvl").
VLM_ENDPOINT = "http://100.64.235.63:8001/v1/chat/completions"
VLM_MODEL = "qwenvl"
VLM_PROMPT = (
    "Transcribe the English text in this manga page, grouped by speech bubble / "
    "caption box. For EACH bubble output ONE object with the FULL bubble text "
    "(join its words/lines with spaces) and its bounding box in pixel coordinates "
    "of this image. Return ONLY a JSON array of "
    '{"text": "<full bubble text>", "bbox": [x0,y0,x1,y1]} in reading order. '
    "Do not split a bubble into words."
)
# Qwen3-VL (and Gemini-style grounders) emit grounding coords NORMALISED to
# 0-1000, not pixels. Ask for that explicitly and rescale with --coord-norm 1000.
VLM_PROMPT_NORM1000 = (
    "Transcribe the English text in this manga page, grouped by speech bubble / "
    "caption box. For EACH bubble output ONE object with the FULL bubble text "
    "(join its words/lines with spaces) and its bounding box as integer "
    "coordinates normalised to a 0-1000 grid over this image (0,0 = top-left, "
    "1000,1000 = bottom-right). Return ONLY a JSON array of "
    '{"text": "<full bubble text>", "bbox": [x0,y0,x1,y1]} in reading order. '
    "Do not split a bubble into words."
)

# A bubble is scoreable (ocr_clean) only when OCR was confident AND the OCR'd
# Japanese is not linguistically garbled.
OCR_CLEAN_CONF = 0.85


# ---------------------------------------------------------------------------
# GT image path resolution (p41 offset)
# ---------------------------------------------------------------------------


def resolve_gt_image_path(
    gt_images_root: Path,
    our_page: int,
    *,
    missing_gt_page: int = MISSING_GT_PAGE,
    ext: str = "webp",
    width: int = 3,
) -> Path | None:
    """Return the GT webp for a bench/our page, applying the p41 offset.

    Returns None if the resolved GT page would be the missing page or out of
    range on disk.
    """
    if our_page == missing_gt_page:
        # The bench page that has no GT counterpart at all.
        return None
    gt_page = our_page_to_gt_page(our_page, missing_gt_page=missing_gt_page)
    if gt_page < 1:
        return None
    # Try the requested ext first, then the common manga page formats. The
    # Ikenie GT is webp; the generalization-benchmark scanlations are jpg (and
    # behindmoon mixes jpg + png). Probing extensions keeps one code path.
    for e in [ext, "webp", "jpg", "jpeg", "png"]:
        p = gt_images_root / f"{gt_page:0{width}d}.{e}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Vision transcription (STUB)
# ---------------------------------------------------------------------------


@dataclass
class VisionBubble:
    """A single bubble of English text read off the GT image by the vision model."""

    text: str
    bbox: dict[str, int] | None = None  # {minX,minY,maxX,maxY} if the model returns one
    reading_order: int | None = None


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_fences(raw: str) -> str:
    """Remove a leading ```json / ``` fence and the trailing ``` fence."""
    s = raw.strip()
    if s.startswith("```"):
        # Drop the opening fence line (``` or ```json) and the closing ```.
        s = re.sub(r"^```(?:json)?\s*\n?", "", s, count=1, flags=re.IGNORECASE)
        s = re.sub(r"\n?```\s*$", "", s, count=1)
    return s.strip()


# A degenerate VLM run on an SFX/scream bubble loops one char (e.g. "RYYYY...")
# and exhausts max_tokens, truncating the JSON.  Cap any single run of a repeated
# character so the surrounding (valid) bubbles are still recoverable.
_RUNAWAY_RE = re.compile(r"(.)\1{9,}")
# Salvage individual ``{"text": "...", "bbox": [...]}`` objects from a truncated
# or malformed array (recover the good bubbles that precede a runaway SFX).
_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _squash_runaway(text: str) -> str:
    """Collapse a runaway repeated-character loop (>=10 repeats -> 3)."""
    return _RUNAWAY_RE.sub(lambda m: m.group(1) * 3, text)


def _parse_vision_response(
    raw: str, *, img_w: int, img_h: int, coord_norm: int = 0
) -> list[VisionBubble]:
    """Parse the VLM JSON array into VisionBubble objects.

    The VLM returns ``[{"text": "...", "bbox": [x0,y0,x1,y1]}, ...]`` in reading
    order (optionally wrapped in a ```json fence``).  bbox is in pixel
    coordinates of the GT image; we convert ``[x0,y0,x1,y1]`` -> the
    ``{minX,minY,maxX,maxY}`` dict the rest of the harness uses, clamping to the
    image bounds.

    Robustness:
      * Strips ```json fences``.
      * If full-array JSON parse fails (the VLM sometimes loops on an SFX bubble
        and truncates the array at max_tokens), SALVAGES every complete
        ``{...}`` object individually so the good bubbles before the runaway are
        still recovered.
      * Collapses runaway repeated-character loops in ``text``.

    Rows with empty text are dropped; a malformed bbox yields ``bbox=None``.  A
    response with no recoverable object yields ``[]`` (caller skips the page).
    """
    s = _strip_fences(raw)
    # Be forgiving: grab the first top-level [...] array if there's extra prose.
    if not s.startswith("["):
        m = re.search(r"\[.*\]", s, re.DOTALL)
        if m:
            s = m.group(0)

    items: list[Any] = []
    parsed = None
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        items = parsed
    else:
        # Salvage path: pull each complete {...} object out of the (possibly
        # truncated) text and parse it on its own.  Squash runaway loops first
        # so a half-finished SFX string doesn't break its enclosing object.
        for blob in _OBJ_RE.findall(_squash_runaway(s)):
            try:
                obj = json.loads(blob)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict):
                items.append(obj)

    out: list[VisionBubble] = []
    for order, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        text = _squash_runaway(str(item.get("text", "") or "")).strip()
        if not text:
            continue
        bbox = None
        raw_box = item.get("bbox")
        if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
            try:
                x0, y0, x1, y1 = (float(v) for v in raw_box)
                # Qwen3-VL (and other Gemini-style grounders) emit coords
                # NORMALISED to a fixed range (e.g. 0-1000) rather than pixels.
                # Rescale to GT-image pixels so the rest of the harness (which
                # works in pixel space + normalises by image dims) aligns.
                if coord_norm and coord_norm > 0:
                    x0 *= img_w / coord_norm
                    x1 *= img_w / coord_norm
                    y0 *= img_h / coord_norm
                    y1 *= img_h / coord_norm
                # tolerate reversed corners
                x0, x1 = sorted((x0, x1))
                y0, y1 = sorted((y0, y1))
                bbox = {
                    "minX": int(max(0, min(x0, img_w))),
                    "minY": int(max(0, min(y0, img_h))),
                    "maxX": int(max(0, min(x1, img_w))),
                    "maxY": int(max(0, min(y1, img_h))),
                }
            except (TypeError, ValueError):
                bbox = None
        out.append(VisionBubble(text=text, bbox=bbox, reading_order=order))
    return out


_MIME_BY_SUFFIX = {
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _image_to_data_url(image_path: Path) -> str:
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    mime = _MIME_BY_SUFFIX.get(image_path.suffix.lower(), "image/webp")
    return f"data:{mime};base64,{b64}"


def _vision_transcribe_page(
    image_path: Path,
    *,
    img_w: int,
    img_h: int,
    endpoint: str = VLM_ENDPOINT,
    model: str = VLM_MODEL,
    prompt: str = VLM_PROMPT,
    coord_norm: int = 0,
    max_tokens: int = 1500,
    retries: int = 3,
    timeout: float = 180.0,
) -> list[VisionBubble]:
    """Call the box VLM on one GT image; return one VisionBubble per bubble.

    Deterministic (temperature 0).  Retries with exponential backoff on network
    / 5xx errors.  Returns ``[]`` if every attempt fails so the caller can skip
    the page gracefully without aborting the whole pass.

    ``coord_norm`` > 0 rescales normalised (e.g. 0-1000) bbox coords to pixels.
    """
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(image_path)},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                endpoint, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            content = obj["choices"][0]["message"]["content"]
            return _parse_vision_response(
                content, img_w=img_w, img_h=img_h, coord_norm=coord_norm
            )
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2.0 * (2**attempt))  # 2s, 4s, ...
    print(f"    !! VLM failed for {image_path.name} after {retries} tries: {last_err}")
    return []


# ---------------------------------------------------------------------------
# Alignment: our JP bubbles <-> GT English bubbles
# ---------------------------------------------------------------------------


def _norm_box(b: dict[str, Any], w: int, h: int) -> tuple[float, float, float, float]:
    """Normalise a ``{minX,minY,maxX,maxY}`` box to [0,1] by image (w,h)."""
    w = max(1, w)
    h = max(1, h)
    return (b["minX"] / w, b["minY"] / h, b["maxX"] / w, b["maxY"] / h)


def _norm_iou(
    a: dict[str, Any], aw: int, ah: int, b: dict[str, Any], bw: int, bh: int
) -> float:
    """IoU of two boxes that may live in DIFFERENT pixel spaces.

    Both boxes are normalised to [0,1] by their own image dimensions first, so a
    GT box (GT image space) and our box (raw image space) overlap correctly even
    when the two images differ in resolution.
    """
    ax0, ay0, ax1, ay1 = _norm_box(a, aw, ah)
    bx0, by0, bx1, by1 = _norm_box(b, bw, bh)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _norm_center(b: dict[str, Any], w: int, h: int) -> tuple[float, float]:
    x0, y0, x1, y1 = _norm_box(b, w, h)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _center_in_box(
    a: dict[str, Any], aw: int, ah: int, b: dict[str, Any], bw: int, bh: int
) -> bool:
    """True if the (normalised) centre of box ``a`` lies inside box ``b``."""
    cx, cy = _norm_center(a, aw, ah)
    bx0, by0, bx1, by1 = _norm_box(b, bw, bh)
    return bx0 <= cx <= bx1 and by0 <= cy <= by1


def _match_score(
    obx: dict[str, Any], ow: int, oh: int, gbx: dict[str, Any], gw: int, gh: int
) -> float:
    """Spatial-match score in [0,1] tolerant of the vertical-vs-horizontal box
    aspect-ratio mismatch between OUR (tall, vertical JP text) and GT (wide,
    horizontal EN text) boxes for the SAME bubble.

    Manga JP boxes and the typeset EN boxes for the same bubble share a CENTRE
    but have orthogonal aspect ratios, so plain IoU under-scores true matches.
    We therefore combine three signals and take the strongest:

      * normalised IoU (works when shapes happen to be similar),
      * mutual centre-containment (our centre in GT box, or vice-versa) — a
        strong same-bubble signal robust to aspect ratio,
      * a centre-distance kernel (1 - dist/diag) so near-but-not-contained
        centres still score, decaying smoothly with separation.

    Returns 0.0 when the centres are far apart and neither box contains the
    other's centre.
    """
    iou = _norm_iou(obx, ow, oh, gbx, gw, gh)
    contain = _center_in_box(obx, ow, oh, gbx, gw, gh) or _center_in_box(
        gbx, gw, gh, obx, ow, oh
    )
    ocx, ocy = _norm_center(obx, ow, oh)
    gcx, gcy = _norm_center(gbx, gw, gh)
    dist = ((ocx - gcx) ** 2 + (ocy - gcy) ** 2) ** 0.5
    # Normalised diagonal is sqrt(2); a centre kernel that decays to 0 by ~0.18
    # of the page diagonal (bubbles in a panel are well separated).
    centre_kernel = max(0.0, 1.0 - dist / 0.18)
    score = max(iou, centre_kernel)
    if contain:
        score = max(score, 0.5 + 0.5 * centre_kernel)
    return score


def align_page(
    our_bubbles: list[dict[str, Any]],
    gt_bubbles: list[VisionBubble],
    *,
    our_w: int,
    our_h: int,
    gt_w: int,
    gt_h: int,
    iou_threshold: float = 0.2,
) -> list[tuple[dict[str, Any], VisionBubble | None, float]]:
    """Pair each of OUR JP bubbles with a GT English bubble.

    Boxes are normalised to [0,1] (GT in GT-image space, ours in raw-image
    space) so differing pixel dimensions don't break the match.

    Matching uses an aspect-ratio-tolerant spatial score (``_match_score``:
    IoU OR centre-containment OR a centre-distance kernel) because our tall
    vertical-JP boxes and the wide horizontal-EN boxes for the SAME bubble share
    a centre but have orthogonal shapes — plain IoU under-scores true matches.

    Strategy: greedy globally-best assignment.  Repeatedly take the highest
    (our, gt) score over all unused pairs and bind them, until no pair scores
    >= ``iou_threshold``.  There is NO reading-order zip fallback: a blind
    positional zip mis-pairs bubbles (our idx order != VLM reading order, and
    the VLM merges/splits bubbles), which would poison the gold.  Our-bubbles
    with no spatial match stay ``None`` and are dropped by the caller.

    Returns ``(our_bubble, gt_bubble_or_None, score)`` triples, one per
    our-bubble; ``score`` is the spatial-match score (0.0 when unmatched).
    """
    n = len(our_bubbles)
    matched: list[VisionBubble | None] = [None] * n
    matched_score: list[float] = [0.0] * n

    score_grid: dict[tuple[int, int], float] = {}
    candidates: list[tuple[float, int, int]] = []
    for i, ob in enumerate(our_bubbles):
        obx = ob.get("bbox")
        if not obx:
            continue
        for j, gb in enumerate(gt_bubbles):
            if gb.bbox is None:
                continue
            s = _match_score(obx, our_w, our_h, gb.bbox, gt_w, gt_h)
            score_grid[(i, j)] = s
            if s >= iou_threshold:
                candidates.append((s, i, j))
    candidates.sort(key=lambda t: t[0], reverse=True)  # globally best first

    used_our: set[int] = set()
    used_gt: set[int] = set()
    match_j: list[int | None] = [None] * n
    for s, i, j in candidates:
        if i in used_our or j in used_gt:
            continue
        match_j[i] = j
        matched_score[i] = s
        used_our.add(i)
        used_gt.add(j)

    # --- reading-order 2-opt: fix stacked-bubble swaps -----------------------
    # Manga JP columns are read RIGHT-to-LEFT, top-to-bottom; the typeset EN
    # boxes for the SAME cluster are stacked TOP-to-BOTTOM in that same reading
    # order.  Greedy centre-matching can swap two adjacent stacked bubbles
    # (our tall columns don't vertically line up with the short EN boxes).  For
    # any matched pair (i->j),(k->l) whose boxes are spatially close (a cluster),
    # swap the GT assignment iff swapping does NOT lower total spatial score by
    # more than a small tolerance AND it makes the our-reading-order agree with
    # the gt-reading-order.  This corrects order without inventing matches.
    _reading_order_2opt(
        our_bubbles, gt_bubbles, match_j, matched_score, score_grid,
        our_w, our_h, gt_w, gt_h,
    )

    for i in range(n):
        j = match_j[i]
        matched[i] = gt_bubbles[j] if j is not None else None

    return [(our_bubbles[i], matched[i], matched_score[i]) for i in range(n)]


def _our_reading_key(b: dict[str, Any], w: int, h: int) -> tuple[float, float]:
    """Manga reading-order key for OUR (vertical-JP) box: right-to-left columns,
    then top-to-bottom.  Larger x => earlier, so sort by (-cx, top)."""
    x0, y0, x1, y1 = _norm_box(b, w, h)
    cx = (x0 + x1) / 2.0
    return (-cx, y0)


def _gt_reading_key(b: dict[str, Any], w: int, h: int) -> tuple[float, float]:
    """Reading-order key for a GT (horizontal-EN) box: top-to-bottom, then
    right-to-left (English typeset stacks top-down within a bubble cluster)."""
    x0, y0, x1, y1 = _norm_box(b, w, h)
    cx = (x0 + x1) / 2.0
    return (y0, -cx)


def _clusters_close(
    a: dict[str, Any], aw: int, ah: int, b: dict[str, Any], bw: int, bh: int
) -> bool:
    """True if two GT boxes are in the same stacked cluster (centres within a
    small normalised radius -> candidates for a reading-order swap)."""
    acx, acy = _norm_center(a, aw, ah)
    bcx, bcy = _norm_center(b, bw, bh)
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 < 0.12


def _reading_order_2opt(
    our_bubbles: list[dict[str, Any]],
    gt_bubbles: list[VisionBubble],
    match_j: list[int | None],
    matched_score: list[float],
    score_grid: dict[tuple[int, int], float],
    our_w: int,
    our_h: int,
    gt_w: int,
    gt_h: int,
    *,
    swap_tolerance: float = 0.25,
) -> None:
    """In-place 2-opt over matched pairs to enforce reading-order consistency
    within a stacked cluster (see caller).  Swaps GT assignments of two matched
    pairs when (a) their GT boxes form a cluster, (b) our reading order disagrees
    with the gt reading order, and (c) swapping costs at most ``swap_tolerance``
    total spatial score (so we never trade a strong match for a weak one)."""
    matched_idx = [i for i in range(len(our_bubbles)) if match_j[i] is not None]
    improved = True
    guard = 0
    while improved and guard < 50:
        improved = False
        guard += 1
        for a_pos in range(len(matched_idx)):
            for b_pos in range(a_pos + 1, len(matched_idx)):
                i, k = matched_idx[a_pos], matched_idx[b_pos]
                ji, jk = match_j[i], match_j[k]
                if ji is None or jk is None:
                    continue
                gi, gk = gt_bubbles[ji].bbox, gt_bubbles[jk].bbox
                if gi is None or gk is None:
                    continue
                if not _clusters_close(gi, gt_w, gt_h, gk, gt_w, gt_h):
                    continue
                obi, obk = our_bubbles[i].get("bbox"), our_bubbles[k].get("bbox")
                if not obi or not obk:
                    continue
                # current vs swapped reading-order agreement
                our_order = _our_reading_key(obi, our_w, our_h) < _our_reading_key(
                    obk, our_w, our_h
                )
                gt_order = _gt_reading_key(gi, gt_w, gt_h) < _gt_reading_key(
                    gk, gt_w, gt_h
                )
                consistent = our_order == gt_order
                if consistent:
                    continue  # order already agrees -> leave as-is
                cur = score_grid.get((i, ji), 0.0) + score_grid.get((k, jk), 0.0)
                swp = score_grid.get((i, jk), 0.0) + score_grid.get((k, ji), 0.0)
                if swp >= cur - swap_tolerance:
                    match_j[i], match_j[k] = jk, ji
                    matched_score[i] = score_grid.get((i, jk), 0.0)
                    matched_score[k] = score_grid.get((k, ji), 0.0)
                    improved = True


# ---------------------------------------------------------------------------
# Gold-row emission
# ---------------------------------------------------------------------------


def _ocr_clean(ob: dict[str, Any]) -> bool:
    """A bubble is scoreable iff OCR was confident AND not garbled/filtered.

    Mirrors the prod gate: ocr_conf >= 0.85, the pipeline did not gate-drop or
    filter it, and ``is_implausible_japanese`` does not flag the OCR'd JP.
    """
    jp = (ob.get("ocr_jp") or "").strip()
    if not jp:
        return False
    conf = ob.get("ocr_conf")
    if conf is None or float(conf) < OCR_CLEAN_CONF:
        return False
    if ob.get("ocr_gate_dropped") or ob.get("filtered"):
        return False
    if is_implausible_japanese(jp, conf):
        return False
    return True


def build_gold_row(
    ob: dict[str, Any],
    gb: VisionBubble,
    *,
    page: int,
    iou: float,
    src_prefix: str = "ikenie4",
) -> dict[str, Any]:
    """Build a gold.jsonl row keyed by OUR bubble with the matched human GT EN.

    Schema matches the score harness gold side (needs ``jp`` + ``en`` + ``src`` +
    ``bbox`` for the bbox-IoU re-join) plus the vision provenance fields.
    """
    idx = ob.get("idx")
    return {
        "jp": (ob.get("ocr_jp") or "").strip(),
        "en": gb.text.strip(),
        "src": f"{src_prefix}:p{page:02d}:idx{idx}",
        "register_tag": "manga_nsfw",
        "category": "",
        "severity": 1,
        "ocr_clean": _ocr_clean(ob),
        "ocr_conf": ob.get("ocr_conf"),
        "bbox": ob.get("bbox"),
        "our_en": (ob.get("translation_en") or "").strip(),
        "source_field": "vision_gt",
        "iou": round(float(iou), 4),
    }


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _load_existing_jp_keys(existing_gold: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not existing_gold.exists():
        return keys
    for line in existing_gold.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        keys.add((r.get("src", ""), r.get("jp", "")))
    return keys


def _load_page_bubbles(bubbles_root: Path, page: int) -> list[dict[str, Any]]:
    p = bubbles_root / f"{page:03d}" / "bubbles.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def _parse_pages(spec: str, default_max: int = 134) -> list[int]:
    if not spec:
        return list(range(1, default_max + 1))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _image_size(path: Path) -> tuple[int, int] | None:
    """Return (width, height) of an image, or None if it can't be read."""
    try:
        from PIL import Image  # local import: only needed for the real pass

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None


def _resolve_our_raw_path(raw_root: Path, our_page: int) -> Path | None:
    """The raw page our bbox coordinates live in (flat ``NNN.<ext>``).

    Ikenie raws are webp; the generalization-benchmark JP raws are jpg. Probe
    the common formats so bbox normalisation reads the right source dimensions.
    """
    for e in ("webp", "jpg", "jpeg", "png"):
        p = raw_root / f"{our_page:03d}.{e}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_BUBBLES_ROOT = (
    "/home/danny/Documents/personal/extension/backend/.bench/ikenie4_merged_insp"
)
DEFAULT_GT_ROOT = (
    "/mnt/nas/drive_2/onlyfans/external_content/nhentai/616137_Ikenie no Haha 4"
)
# Raw pages our bbox coordinates are expressed in (same source as the inspect
# 01_source.webp).  Used to normalise our bbox before IoU vs the GT box.
DEFAULT_RAW_ROOT = (
    "/mnt/nas/drive_2/onlyfans/external_content/nhentai/583875_Ikenie no Haha 4"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bubbles-root", default=DEFAULT_BUBBLES_ROOT)
    ap.add_argument("--gt-images-root", default=DEFAULT_GT_ROOT)
    ap.add_argument(
        "--raw-images-root",
        default=DEFAULT_RAW_ROOT,
        help="Pages our bbox coordinates are in (583875). Used to read our "
        "image size for bbox normalisation. Falls back to GT size if missing.",
    )
    ap.add_argument(
        "--existing-gold",
        default=str(SCRIPT_DIR / "data" / "ikenie4" / "gold.jsonl"),
        help="Judge-seeded gold; rows already present (by src) are skipped.",
    )
    ap.add_argument(
        "--out",
        default=str(SCRIPT_DIR / "data" / "ikenie4" / "gold_full.jsonl"),
    )
    ap.add_argument("--pages", default="1-134", help="e.g. '1-134' or '1,2,5-9'")
    ap.add_argument("--iou-threshold", type=float, default=0.2)
    ap.add_argument(
        "--src-prefix",
        default="ikenie4",
        help="Chapter slug for the gold `src` key (e.g. 'ikenie5'). Must match "
        "the prefix the eval expects when scoring this chapter.",
    )
    ap.add_argument(
        "--missing-gt-page",
        type=int,
        default=MISSING_GT_PAGE,
        help="Bench page with no GT counterpart (the +1 offset pivot). Default 41 "
        "(ch4). For a 1:1 chapter (e.g. ch5, 102==102) pass a value past the last "
        "page (e.g. 99999) so GT mapping is identity and no page is skipped.",
    )
    ap.add_argument("--vlm-endpoint", default=VLM_ENDPOINT,
                    help="OpenAI-compatible /chat/completions URL of the VLM.")
    ap.add_argument("--vlm-model", default=VLM_MODEL,
                    help="served-model-name of the VLM (e.g. qwenvl, qwen3vl).")
    ap.add_argument("--coord-norm", type=int, default=0,
                    help="If >0, bbox coords are normalised to this range (e.g. "
                    "1000 for Qwen3-VL) and are rescaled to pixels. 0 = pixels "
                    "(Qwen2.5-VL).")
    ap.add_argument("--vlm-prompt", default=None,
                    help="Override the VLM prompt. Defaults to the pixel-coord "
                    "prompt, or the 0-1000 normalised prompt when --coord-norm=1000.")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve GT image paths + count bubbles WITHOUT calling the vision "
        "model (verifies the p41 offset wiring end-to-end).",
    )
    args = ap.parse_args(argv)
    vlm_prompt = args.vlm_prompt or (
        VLM_PROMPT_NORM1000 if args.coord_norm == 1000 else VLM_PROMPT
    )

    bubbles_root = Path(args.bubbles_root)
    gt_root = Path(args.gt_images_root)
    raw_root = Path(args.raw_images_root)
    existing_keys = _load_existing_jp_keys(Path(args.existing_gold))
    pages = _parse_pages(args.pages)

    new_rows: list[dict[str, Any]] = []
    resolved, missing = 0, 0
    vlm_pages, vlm_failed = 0, 0
    matched_iou, matched_zip, unmatched = 0, 0, 0
    iou_values: list[float] = []

    for page in pages:
        our_bubbles = _load_page_bubbles(bubbles_root, page)
        if not our_bubbles:
            continue
        gt_path = resolve_gt_image_path(
            gt_root, page, missing_gt_page=args.missing_gt_page
        )
        if gt_path is None:
            missing += 1
            if page == args.missing_gt_page:
                print(f"  p{page:03d}: no GT page (missing_gt_page) -- skipped")
            else:
                print(f"  p{page:03d}: GT image not found on disk -- skipped")
            continue
        resolved += 1

        if args.dry_run:
            print(
                f"  p{page:03d} -> GT {gt_path.name}  "
                f"(our_bubbles={len(our_bubbles)})"
            )
            continue

        # --- image dimensions for normalised IoU ---
        gt_size = _image_size(gt_path)
        if gt_size is None:
            print(f"  p{page:03d}: cannot read GT image size -- skipped")
            missing += 1
            continue
        gt_w, gt_h = gt_size
        raw_path = _resolve_our_raw_path(raw_root, page)
        our_size = _image_size(raw_path) if raw_path else None
        # Our bbox lives in the source-webp space; that is the same source as the
        # GT (both nhentai), so default to GT size when the raw page is absent.
        our_w, our_h = our_size if our_size else (gt_w, gt_h)

        # --- real path: transcribe the GT image with the box VLM ---
        gt_bubbles = _vision_transcribe_page(
            gt_path, img_w=gt_w, img_h=gt_h,
            endpoint=args.vlm_endpoint, model=args.vlm_model,
            prompt=vlm_prompt, coord_norm=args.coord_norm,
        )
        if not gt_bubbles:
            vlm_failed += 1
            print(f"  p{page:03d} -> GT {gt_path.name}: VLM returned 0 bubbles -- skipped")
            continue
        vlm_pages += 1

        pairs = align_page(
            our_bubbles,
            gt_bubbles,
            our_w=our_w,
            our_h=our_h,
            gt_w=gt_w,
            gt_h=gt_h,
            iou_threshold=args.iou_threshold,
        )
        page_emitted = 0
        for ob, gb, iou in pairs:
            if gb is None or not gb.text.strip():
                unmatched += 1
                continue
            jp = (ob.get("ocr_jp") or "").strip()
            if not jp:
                unmatched += 1
                continue
            idx = ob.get("idx")
            src = f"{args.src_prefix}:p{page:02d}:idx{idx}"
            if src in existing_keys:
                # Covered by the richer judge-seeded gold; don't overwrite.
                continue
            row = build_gold_row(ob, gb, page=page, iou=iou, src_prefix=args.src_prefix)
            new_rows.append(row)
            page_emitted += 1
            iou_values.append(iou)
            if iou > 0.0:
                matched_iou += 1
            else:
                matched_zip += 1
        print(
            f"  p{page:03d} -> GT {gt_path.name}: "
            f"gt={len(gt_bubbles)} our={len(our_bubbles)} emitted={page_emitted}"
        )

    print(f"\nGT image resolution: {resolved} resolved, {missing} skipped")
    if args.dry_run:
        print("dry-run: vision model NOT called; offset wiring verified above.")
        return 0

    # --- alignment summary ---
    total_matched = matched_iou + matched_zip
    print(
        f"VLM pages: {vlm_pages} transcribed, {vlm_failed} failed/empty\n"
        f"Alignment: {total_matched} matched "
        f"({matched_iou} by IoU>= {args.iou_threshold}, {matched_zip} zip-fallback), "
        f"{unmatched} unmatched"
    )
    if iou_values:
        nz = [v for v in iou_values if v > 0]
        if nz:
            nz.sort()
            import statistics

            print(
                "  IoU(matched, nonzero): "
                f"min={nz[0]:.3f} med={statistics.median(nz):.3f} "
                f"max={nz[-1]:.3f} n={len(nz)}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in new_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    clean = sum(1 for r in new_rows if r["ocr_clean"])
    print(f"\nwrote {len(new_rows)} vision-gold rows ({clean} ocr_clean) -> {out_path}")
    print("NEXT: merge into gold.jsonl (keep the 77 judge rows), then FREEZE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

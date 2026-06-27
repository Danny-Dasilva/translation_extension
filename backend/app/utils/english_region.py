"""English early-exit: detect horizontal/Latin (English) regions to leave intact.

The CTD detector already classifies each text *line* as "vertical" or
"horizontal" (`direction`, set in CTDService._expand_text_lines:
``"vertical" if bh > bw*1.2 else "horizontal"``). Japanese manga dialogue is
near-universally vertical; English UI text, watermarks, and already-localized
captions are horizontal Latin. We use that geometry — AND the OCR content —
to skip translating/inpainting English regions so their original pixels stay
untouched (no OCR -> translate -> inpaint -> typeset, no TextBox emitted).

Two pure functions, no side effects, no service deps beyond the existing JP
detector (app.utils.japanese_text_filter), so they are trivially unit-testable
and safe to call inside the post-OCR filter loop of both pipeline paths.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional


def _line_is_horizontal(line: Dict) -> bool:
    """True if a text_line is horizontal.

    Prefer the detector-provided ``direction`` field; fall back to line
    geometry ((maxX-minX) > (maxY-minY)) when it is absent.
    """
    direction = line.get("direction")
    if direction is not None:
        return direction == "horizontal"
    return (line["maxX"] - line["minX"]) > (line["maxY"] - line["minY"])


def _line_center_in_block(line: Dict, block: Dict) -> bool:
    """Center-containment test (same rule build_text_regions / bubble-match use)."""
    cx = (line["minX"] + line["maxX"]) / 2
    cy = (line["minY"] + line["maxY"]) / 2
    return (
        block["minX"] <= cx <= block["maxX"]
        and block["minY"] <= cy <= block["maxY"]
    )


def is_horizontal_region(
    block: Dict,
    text_lines: Optional[List[Dict]],
    threshold: float = 0.6,
) -> bool:
    """Decide whether a detected block reads horizontally (Latin layout).

    Collect the text_lines whose CENTER falls inside ``block`` and return True
    when at least ``threshold`` fraction of them are horizontal. When no line is
    contained (e.g. a synthetic/orphan block, or geometry-only), fall back to the
    block's own aspect ratio: width/height > 1.5 => horizontal.

    Args:
        block: dict with minX, minY, maxX, maxY (and optionally confidence).
        text_lines: per-line dicts with minX/minY/maxX/maxY and optional
            ``direction``; may be None/empty.
        threshold: minimum fraction of contained lines that must be horizontal.

    Returns:
        True if the region is horizontal (English-layout candidate).
    """
    contained: List[Dict] = []
    if text_lines:
        for line in text_lines:
            try:
                if _line_center_in_block(line, block):
                    contained.append(line)
            except (KeyError, TypeError):
                # Malformed line dict: ignore it rather than crash the filter.
                continue

    if contained:
        horizontal = sum(1 for ln in contained if _line_is_horizontal(ln))
        return (horizontal / len(contained)) >= threshold

    # Fallback: no contained lines -> use the block's aspect ratio.
    bw = block["maxX"] - block["minX"]
    bh = block["maxY"] - block["minY"]
    if bw <= 0 or bh <= 0:
        return False
    return (bw / bh) > 1.5


def should_skip_as_english(
    block: Dict,
    text_lines: Optional[List[Dict]],
    ocr_text: str,
    is_japanese_fn: Callable[[str], bool],
) -> bool:
    """True if a region should be left intact as English (skip the pipeline).

    A region is skipped only when BOTH hold:
      1. geometry is horizontal (is_horizontal_region), AND
      2. the OCR text is NOT recognized as Japanese (is_japanese_fn False).

    The content check guards against horizontal Japanese SFX (e.g. ドドド laid
    out left-to-right): those are horizontal but ARE Japanese, so they are NOT
    skipped. Conversely a vertical region is never skipped here — stray Latin in
    a vertical column is left to the existing japanese_filter.

    Args:
        block: detected block dict.
        text_lines: per-line dicts (may be None/empty).
        ocr_text: the recognized OCR text for this block.
        is_japanese_fn: predicate returning True when text is valid Japanese
            (e.g. app.utils.japanese_text_filter.is_japanese_text).

    Returns:
        True -> skip this region (leave original pixels untouched).
    """
    if not is_horizontal_region(block, text_lines):
        return False
    return not is_japanese_fn(ocr_text or "")

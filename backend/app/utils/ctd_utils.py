"""Helpers for mapping CTD outputs to text regions."""

from __future__ import annotations

from typing import Dict, List


def build_text_regions(
    blocks: List[Dict],
    text_lines: List[Dict] = None,
    inset_percent: float = 0.05  # 5% inset on each side = 90% final size
) -> List[List[Dict]]:
    """
    Build per-block text regions with inset for white box masking.

    Args:
        blocks: List of detected text blocks with minX, minY, maxX, maxY
        text_lines: Unused (kept for API compatibility)
        inset_percent: Percentage to shrink on each side (0.05 = 5% = 90% final size)

    Returns:
        List of text regions, each slightly smaller than the original block
    """
    regions: List[List[Dict]] = []

    for block in blocks:
        w = block["maxX"] - block["minX"]
        h = block["maxY"] - block["minY"]

        # Apply inset (shrink box by inset_percent on each side)
        inset_x = int(w * inset_percent)
        inset_y = int(h * inset_percent)

        regions.append([{
            "minX": block["minX"] + inset_x,
            "minY": block["minY"] + inset_y,
            "maxX": block["maxX"] - inset_x,
            "maxY": block["maxY"] - inset_y,
        }])

    return regions

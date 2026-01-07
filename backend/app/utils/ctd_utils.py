"""Helpers for mapping CTD outputs to text regions."""

from __future__ import annotations

from typing import Dict, List


def build_text_regions(
    blocks: List[Dict],
    text_lines: List[Dict] = None,
    inset_percent: float = 0.0
) -> List[List[Dict]]:
    """
    Build per-block text regions from CTD outputs.

    Returns block bounds as the text region (for full white box coverage).
    """
    regions: List[List[Dict]] = []

    for block in blocks:
        regions.append([{
            "minX": block["minX"],
            "minY": block["minY"],
            "maxX": block["maxX"],
            "maxY": block["maxY"],
        }])

    return regions

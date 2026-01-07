"""Z-index assignment utilities for text box layering."""

from __future__ import annotations

from typing import Dict, List, Union


def assign_smart_zindex(
    text_boxes: List[Union[Dict, object]],
    use_dict: bool = False
) -> List[Union[Dict, object]]:
    """
    Assign zIndex based on box area - smaller boxes get higher zIndex.

    This ensures smaller boxes render on top of larger overlapping boxes,
    preventing larger boxes' backgrounds from covering smaller boxes' text.

    Args:
        text_boxes: List of text box objects or dicts with minX/minY/maxX/maxY
        use_dict: If True, treat items as dicts; if False, treat as objects

    Returns:
        Same list with updated zIndex values (modified in place)
    """
    if not text_boxes:
        return text_boxes

    # Calculate area for each box
    areas = []
    for box in text_boxes:
        if use_dict:
            w = box["maxX"] - box["minX"]
            h = box["maxY"] - box["minY"]
        else:
            w = box.maxX - box.minX
            h = box.maxY - box.minY
        areas.append(w * h)

    # Create index-area pairs and sort by area descending (largest first)
    indexed_areas = sorted(enumerate(areas), key=lambda x: -x[1])

    # Assign zIndex: 1 (largest) to N (smallest)
    # Smaller boxes get higher zIndex = rendered last = on top
    for rank, (idx, _) in enumerate(indexed_areas):
        new_zindex = rank + 1
        if use_dict:
            text_boxes[idx]["zIndex"] = new_zindex
        else:
            text_boxes[idx].zIndex = new_zindex

    return text_boxes

"""Text box grouping utilities for manga translation"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class UnionFind:
    """Union-Find (Disjoint Set Union) data structure for grouping text boxes"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root of set containing x"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int):
        """Merge sets containing x and y"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            # Union by rank
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

    def get_groups(self) -> Dict[int, List[int]]:
        """Get all groups as dict of root -> [members]"""
        groups = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        return groups


def calculate_y_overlap(box1: Dict, box2: Dict) -> float:
    """
    Calculate Y-axis overlap percentage between two boxes.
    Returns value between 0.0 (no overlap) and 1.0 (complete overlap).
    """
    y1_min, y1_max = box1['minY'], box1['maxY']
    y2_min, y2_max = box2['minY'], box2['maxY']

    # Calculate overlap region
    overlap_start = max(y1_min, y2_min)
    overlap_end = min(y1_max, y2_max)
    overlap = max(0, overlap_end - overlap_start)

    # Calculate percentage relative to smaller box height
    height1 = y1_max - y1_min
    height2 = y2_max - y2_min
    min_height = min(height1, height2)

    if min_height == 0:
        return 0.0

    return overlap / min_height


def calculate_min_distance(box1: Dict, box2: Dict) -> float:
    """
    Calculate minimum distance between two bounding boxes.
    Returns 0 if boxes overlap, otherwise returns minimum edge distance.
    """
    x1_min, x1_max = box1['minX'], box1['maxX']
    y1_min, y1_max = box1['minY'], box1['maxY']
    x2_min, x2_max = box2['minX'], box2['maxX']
    y2_min, y2_max = box2['minY'], box2['maxY']

    # Check for overlap
    if not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min):
        return 0.0  # Boxes overlap

    # Calculate horizontal and vertical distances
    if x1_max < x2_min:
        dx = x2_min - x1_max
    elif x2_max < x1_min:
        dx = x1_min - x2_max
    else:
        dx = 0

    if y1_max < y2_min:
        dy = y2_min - y1_max
    elif y2_max < y1_min:
        dy = y1_min - y2_max
    else:
        dy = 0

    # Return Euclidean distance
    return (dx ** 2 + dy ** 2) ** 0.5


def should_group_boxes(box1: Dict, box2: Dict,
                       distance_threshold: float = 80.0,
                       y_overlap_threshold: float = 0.5) -> bool:
    """
    Determine if two boxes should be grouped together for manga translation.

    Boxes are grouped if:
    1. They overlap (distance = 0), OR
    2. They're close together (< distance_threshold) AND have significant Y-overlap

    Args:
        box1, box2: Text boxes with minX, minY, maxX, maxY
        distance_threshold: Maximum distance (px) between boxes to group
        y_overlap_threshold: Minimum Y-axis overlap ratio (0-1) for vertical text

    Returns:
        True if boxes should be grouped together
    """
    distance = calculate_min_distance(box1, box2)

    # Always group overlapping boxes
    if distance == 0:
        return True

    # Check distance threshold
    if distance > distance_threshold:
        return False

    # For nearby boxes, check Y-overlap (indicates vertical text column)
    y_overlap = calculate_y_overlap(box1, box2)
    return y_overlap >= y_overlap_threshold


def group_text_boxes(boxes: List[Dict[str, Any]],
                     distance_threshold: float = 80.0,
                     y_overlap_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Group nearby text boxes into logical paragraphs for manga translation.

    This function implements intelligent grouping for Japanese manga:
    - Vertical text columns (boxes vertically aligned with overlapping Y ranges)
    - Speech bubbles (boxes within close proximity)
    - Reading order: Right-to-left, top-to-bottom

    Algorithm:
    1. Use Union-Find to group "connected" boxes
    2. Boxes are connected if distance < threshold AND Y-overlap > threshold
    3. For each group:
       - Sort by manga reading order (X descending, Y ascending)
       - Concatenate text
       - Create union bounding box
       - Store original boxes as subtextBoxes

    Args:
        boxes: List of OCR text boxes with text, minX, minY, maxX, maxY, confidence
        distance_threshold: Maximum distance (px) between boxes to group (default: 80)
        y_overlap_threshold: Minimum Y-overlap ratio for grouping (default: 0.5)

    Returns:
        List of grouped text boxes with merged text and union bounding boxes
    """
    if not boxes:
        return []

    n = len(boxes)
    logger.info(f"Grouping {n} text boxes for manga translation...")

    # Edge case: single box
    if n == 1:
        box = boxes[0].copy()
        box['subtextBoxes'] = [boxes[0]]
        return [box]

    # Initialize Union-Find
    uf = UnionFind(n)

    # Find all pairs of boxes that should be grouped
    for i in range(n):
        for j in range(i + 1, n):
            if should_group_boxes(boxes[i], boxes[j], distance_threshold, y_overlap_threshold):
                uf.union(i, j)
                logger.debug(f"Grouping box {i} and box {j} (distance: {calculate_min_distance(boxes[i], boxes[j]):.1f}px)")

    # Get groups
    groups = uf.get_groups()
    logger.info(f"Created {len(groups)} groups from {n} boxes")

    # Process each group
    grouped_boxes = []
    for group_id, indices in groups.items():
        group_boxes = [boxes[idx] for idx in indices]

        # Sort by manga reading order: right-to-left (X desc), then top-to-bottom (Y asc)
        # Use center coordinates for sorting
        group_boxes_sorted = sorted(
            group_boxes,
            key=lambda b: (
                -((b['minX'] + b['maxX']) / 2),  # Right to left (negative for descending)
                (b['minY'] + b['maxY']) / 2       # Top to bottom
            )
        )

        # Concatenate text
        combined_text = ''.join(b['text'] for b in group_boxes_sorted)

        # Calculate union bounding box
        all_minX = [b['minX'] for b in group_boxes]
        all_minY = [b['minY'] for b in group_boxes]
        all_maxX = [b['maxX'] for b in group_boxes]
        all_maxY = [b['maxY'] for b in group_boxes]

        union_box = {
            'text': combined_text,
            'minX': min(all_minX),
            'minY': min(all_minY),
            'maxX': max(all_maxX),
            'maxY': max(all_maxY),
            'confidence': sum(b['confidence'] for b in group_boxes) / len(group_boxes),  # Average confidence
            'subtextBoxes': group_boxes  # Preserve original boxes
        }

        grouped_boxes.append(union_box)
        logger.debug(
            f"Group {group_id}: Combined {len(group_boxes)} boxes -> '{combined_text[:50]}...' "
            f"at ({union_box['minX']},{union_box['minY']})-({union_box['maxX']},{union_box['maxY']})"
        )

    # Sort groups by reading order (right-to-left, top-to-bottom)
    grouped_boxes_sorted = sorted(
        grouped_boxes,
        key=lambda b: (
            -((b['minX'] + b['maxX']) / 2),  # Right to left
            (b['minY'] + b['maxY']) / 2       # Top to bottom
        )
    )

    logger.info(f"Finished grouping: {n} boxes -> {len(grouped_boxes_sorted)} groups")
    return grouped_boxes_sorted

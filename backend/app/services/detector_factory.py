"""Factory for creating text detector instances based on configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from app.services.animetext_service import AnimeTextDetectorService
    from app.services.ctd_service import ComicTextDetectorService

logger = logging.getLogger(__name__)

# Union type for detector services
DetectorService = "AnimeTextDetectorService | ComicTextDetectorService"


def create_detector() -> DetectorService:
    """
    Create appropriate text detector based on configuration.

    detector_type options:
    - "animetext": AnimeText YOLO12s FP16 (fast, 414 FPS, blocks-only)
    - "ctd": Comic Text Detector (133 FPS, blocks + text_lines + mask)

    Returns:
        Detector service instance
    """
    detector_type = settings.detector_type.lower()

    if detector_type == "animetext":
        from app.services.animetext_service import AnimeTextDetectorService
        logger.info("Using AnimeText YOLO12s detector (fast, blocks-only)")
        return AnimeTextDetectorService()

    elif detector_type == "ctd":
        from app.services.ctd_service import ComicTextDetectorService
        logger.info("Using Comic Text Detector (CTD) with text_lines support")
        return ComicTextDetectorService()

    else:
        raise ValueError(
            f"Unknown detector type: '{detector_type}'. "
            "Use 'animetext' or 'ctd'"
        )

"""Response models for the translation API"""
from pydantic import BaseModel
from typing import List, Optional


class TextRegion(BaseModel):
    """Precise text bounding box within a bubble for targeted masking"""

    minX: int
    minY: int
    maxX: int
    maxY: int


class TextBox(BaseModel):
    """Represents a single text box with OCR result and translation"""

    ocrText: str
    originalLanguage: str = ""
    minX: int
    minY: int
    maxX: int
    maxY: int
    # Per-box source-crop JPEG was removed from the pipeline (frontend renderer
    # never read it); kept as an empty-string default for API compatibility.
    background: str = ""
    fontHeightPx: int
    fontColor: str
    fontStrokeColor: str
    zIndex: int = 1
    translatedText: str
    subtextBoxes: List = []
    textRegions: List[TextRegion] = []
    # Speech-bubble interior this box was matched to (YOLOv10n + match_blocks_to_bubbles).
    # None when the block sits in no qualifying bubble (e.g. SFX over art) — the
    # frontend should fall back to the tight block bbox / textRegions in that case.
    bubbleRect: Optional[TextRegion] = None
    confidence: float = 0.0
    ocrTimeMs: float = 0.0
    translateTimeMs: float = 0.0

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "ocrText": "こんにちは",
                "originalLanguage": "ja",
                "minX": 100,
                "minY": 200,
                "maxX": 300,
                "maxY": 250,
                "background": "data:image/jpeg;base64,...",
                "fontHeightPx": 20,
                "fontColor": "#000000",
                "fontStrokeColor": "#FFFFFF",
                "zIndex": 1,
                "translatedText": "Hello",
                "subtextBoxes": []
            }]
        }
    }


class TranslateResponse(BaseModel):
    """Response model for /translate endpoint"""

    images: List[List[TextBox]]
    inpainted_image_base64: List[Optional[str]] = []
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "images": [[
                    {
                        "ocrText": "こんにちは",
                        "originalLanguage": "ja",
                        "minX": 100,
                        "minY": 200,
                        "maxX": 300,
                        "maxY": 250,
                        "background": "data:image/jpeg;base64,...",
                        "fontHeightPx": 20,
                        "fontColor": "#000000",
                        "fontStrokeColor": "#FFFFFF",
                        "zIndex": 1,
                        "translatedText": "Hello",
                        "subtextBoxes": []
                    }
                ]]
            }]
        }
    }

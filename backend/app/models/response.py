"""Response models for the translation API"""
from pydantic import BaseModel
from typing import List


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
    background: str
    fontHeightPx: int
    fontColor: str
    fontStrokeColor: str
    zIndex: int = 1
    translatedText: str
    subtextBoxes: List = []
    textRegions: List[TextRegion] = []
    
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

"""Response models for the translation API"""
from pydantic import BaseModel, Field
from typing import List


class TextBox(BaseModel):
    """Represents a single text box with OCR result and translation"""
    
    ocrText: str = Field(description="Original text detected by OCR")
    originalLanguage: str = Field(default="", description="Detected source language")
    minX: int = Field(description="Bounding box minimum X coordinate")
    minY: int = Field(description="Bounding box minimum Y coordinate")
    maxX: int = Field(description="Bounding box maximum X coordinate")
    maxY: int = Field(description="Bounding box maximum Y coordinate")
    background: str = Field(description="Base64 of text region background")
    fontHeightPx: int = Field(description="Suggested font size in pixels")
    fontColor: str = Field(description="Text color (hex)")
    fontStrokeColor: str = Field(description="Text stroke color (hex)")
    zIndex: int = Field(default=1, description="Layer order")
    translatedText: str = Field(description="Translated text")
    subtextBoxes: List = Field(default_factory=list, description="Nested text boxes")
    
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
    
    images: List[List[TextBox]] = Field(
        description="Array of text boxes for each image"
    )
    
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

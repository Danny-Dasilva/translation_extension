"""Request models for the translation API"""
from pydantic import BaseModel, Field
from typing import List


class TranslateRequest(BaseModel):
    """Request model for /translate endpoint"""
    
    base64Images: List[str] = Field(
        ...,
        description="Array of base64-encoded images (data:image/jpeg;base64,...)",
        min_length=1,
        max_length=5
    )
    
    targetLanguage: str = Field(
        default="English",
        description="Target language for translation"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "base64Images": ["data:image/jpeg;base64,/9j/4AAQ..."],
                "targetLanguage": "English"
            }]
        }
    }

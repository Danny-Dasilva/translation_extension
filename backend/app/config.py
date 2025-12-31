"""Configuration management using Pydantic Settings"""
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Keys (optional - only needed for cloud fallback)
    gemini_api_key: Optional[str] = None

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # CORS
    allowed_origins: str = "*"

    # Rate Limiting
    max_requests_per_minute: int = 60
    max_images_per_request: int = 5

    # Translation
    default_target_language: str = "English"

    # Local AI Model Paths
    yolo_model_path: str = "app/models/yolov10n_manga.pt"
    ocr_model_id: str = "jzhang533/PaddleOCR-VL-For-Manga"
    translation_model_path: str = "app/weights/HY-MT1.5-1.8B-Q8_0.gguf"

    # Weights directory (for downloaded models)
    weights_dir: str = "app/weights"

    # Performance Tuning
    use_flash_attention: bool = True
    ocr_batch_size: int = 4
    translation_batch_mode: str = "concatenated"  # "individual" or "concatenated"
    detection_confidence: float = 0.25
    detection_image_size: int = 640

    class Config:
        env_file = ".env"
        case_sensitive = False

    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def yolo_model_exists(self) -> bool:
        """Check if YOLOv10 model file exists"""
        return Path(self.yolo_model_path).exists()

    @property
    def translation_model_exists(self) -> bool:
        """Check if translation model file exists"""
        return Path(self.translation_model_path).exists()


# Global settings instance
settings = Settings()

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
    # Note: manga-ocr auto-downloads its model, no path config needed

    # Translation model
    translation_model_filename: str = "HY-MT1.5-1.8B-Q8_0.gguf"

    # Weights directory (for downloaded models)
    weights_dir: str = "app/weights"

    @property
    def translation_model_path(self) -> str:
        """Get translation model path."""
        return f"{self.weights_dir}/{self.translation_model_filename}"

    # Performance Tuning
    detection_confidence: float = 0.25
    detection_image_size: int = 640
    parallel_image_processing: bool = True  # Process multiple images in parallel
    max_parallel_images: int = 3  # Max concurrent image processing

    # Translation parallelization
    translation_use_parallel: bool = True  # Use parallel translation with asyncio.gather
    translation_num_instances: int = 3  # Number of translation model instances

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

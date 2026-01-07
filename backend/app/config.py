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

    # Detector Selection: "animetext" (fast) or "ctd" (full-featured)
    detector_type: str = "animetext"

    # AnimeText YOLO12s FP16 (3.1x faster than CTD: 414 FPS vs 133 FPS)
    animetext_model_path: str = "models/animetext_yolo12s_fp16.onnx"
    animetext_input_size: int = 640
    animetext_confidence_threshold: float = 0.272  # From model's threshold.json

    # Comic Text Detector (CTD) - includes text_lines and mask
    ctd_model_path: str = "models/comictextdetector.onnx"
    ctd_input_size: int = 1024
    ctd_text_threshold: float = 0.3
    ctd_block_confidence: float = 0.4
    ctd_min_text_area: int = 100
    ctd_nms_free: bool = False  # Enable NMS to filter duplicate overlapping boxes

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
    translation_num_instances: int = 6  # Number of translation model instances (6 for single-round 6-bubble pages)

    # Translation model tuning
    translation_n_ctx: int = 1024  # Context window (reduced from 2048, but 512 was too tight)
    translation_n_batch: int = 256  # Prompt processing batch size
    translation_n_ubatch: int = 128  # Physical batch size for GPU
    translation_max_tokens: int = 96  # Max output tokens (reduced from 256 - manga dialogue is short)

    # Pipeline optimization
    use_pipeline_overlap: bool = True  # Start translation as each OCR completes (overlap OCR+translation)

    # Japanese text filter (post-OCR)
    # Filters out non-Japanese text that MangaOCR may hallucinate from English
    japanese_filter_enabled: bool = True
    japanese_filter_min_ratio: float = 0.5  # Min Japanese char ratio (0.0-1.0)
    japanese_filter_katakana_max_length: int = 6  # Max length for katakana-only text

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

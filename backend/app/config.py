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

    # OCR backend selection: "parseq" (local trained model) or "manga-ocr"
    ocr_backend: str = "parseq"
    parseq_model_path: str = "models/parseq_manga_best_ep60_AR_single.onnx"
    # AR_single ONNX has hardcoded batch=1 in its Reshape node, so we force
    # per-sample inference. Batched non-AR exports use larger sizes.
    parseq_batch_size: int = 1

    # Detector Selection: "animetext" (fast) or "ctd" (full-featured).
    # CTD is recommended when ocr_backend="parseq" because PARSeq is a
    # line-level STR model and needs per-line crops (text_lines) for best
    # quality; AnimeText only produces block-level bboxes.
    detector_type: str = "ctd"

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

    # Translation backend: "vllm-openai" (vLLM serving an OpenAI-compatible
    # chat endpoint — the v10-it Gemma 4 E4B merged model + Google's MTP
    # drafter) or "transformers" (HF transformers, used for Hy-MT1.5-2bit).
    #
    # Default is the vLLM + Google MTP path: it's the fastest production config
    # benched (112 tok/s, 29% draft acceptance, lossless) and gives the best
    # translation quality. Requires the server from
    # `backend/scripts/eval/serve_v10it_vllm.sh` to be running on vllm_base_url;
    # VLLMOpenAITranslationService raises a clear "start it with…" error if not.
    translation_backend: str = "vllm-openai"
    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    vllm_model_name: str = "v10it"

    # Translation model (transformers backend)
    hymt_transformers_model_dir: str = "app/weights/hymt15-2bit"

    # Weights directory (for downloaded models)
    weights_dir: str = "app/weights"

    # Performance Tuning
    detection_confidence: float = 0.25
    detection_image_size: int = 640
    parallel_image_processing: bool = True  # Process multiple images in parallel
    max_parallel_images: int = 3  # Max concurrent image processing

    # Translation parallelization
    translation_use_parallel: bool = True  # Use parallel translation with asyncio.gather

    # Pipeline optimization
    use_pipeline_overlap: bool = True  # Start translation as each OCR completes (overlap OCR+translation)

    # Koharu-inspired stages
    # When enabled, run LaMa inpainting after OCR/translate and return inpainted PNG
    enable_inpainting: bool = True
    lama_model_path: str = "models/lama.onnx"
    # When enabled, use page-level [N]-tagged batched translation (coherence win)
    # instead of per-bubble parallel calls. Fallback to parallel on failure.
    use_batched_translation: bool = True

    # Japanese text filter (post-OCR)
    # Filters out non-Japanese text that MangaOCR may hallucinate from English
    japanese_filter_enabled: bool = True
    japanese_filter_min_ratio: float = 0.5  # Min Japanese char ratio (0.0-1.0)
    japanese_filter_katakana_max_length: int = 6  # Max length for katakana-only text

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def yolo_model_exists(self) -> bool:
        """Check if YOLOv10 model file exists"""
        return Path(self.yolo_model_path).exists()


# Global settings instance
settings = Settings()

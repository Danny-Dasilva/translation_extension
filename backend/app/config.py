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
    parseq_model_path: str = "models/parseq_manga_ep60_nonAR_dynbatch.fp16.onnx"
    # Batched non-autoregressive export with a dynamic batch axis: one forward
    # pass OCRs all lines on a page (~10x faster than the old AR_single model,
    # which had a hardcoded batch=1 Reshape and required N sequential forwards).
    #
    # NOTE: this non-AR export was evaluated 2026-06-14 against the AR_single model
    # (parseq_manga_best_ep60_AR_single.onnx) on a labeled per-line GT set. RAW, it
    # REGRESSED accuracy (+1.04pp mean CER over the +0.5pp bar) and emitted non-AR
    # repeat artifacts on single-line crops ('体体体体体', '...。..', 'うっ!!!ー!!').
    # We now run it with the repeat-collapse postprocess (collapse_cjk_runs +
    # collapse_trailing_loop in app/utils/ocr_postprocess.py, wired into apply_all)
    # which neutralizes those artifacts; the speed win is taken pending the
    # postprocess-on per-line A/B re-run. To fall back, set this to the AR_single
    # path and parseq_batch_size=1. See
    # thoughts/shared/research/translation-perf-display/2026-06-13_parseq-dynamic-batch-proposal.md
    parseq_batch_size: int = 8

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
    # Max concurrent image pipelines. RTX 5090 (32GB) comfortably holds CTD +
    # YOLO + PARSeq + LaMa working sets for 4 in-flight pages; vLLM runs
    # out-of-process so it is not bounded by this. Raise further only with a VRAM
    # headroom check.
    max_parallel_images: int = 4

    # Translation parallelization
    translation_use_parallel: bool = True  # Use parallel translation with asyncio.gather

    # Pipeline optimization
    use_pipeline_overlap: bool = True  # Start translation as each OCR completes (overlap OCR+translation)

    # Koharu-inspired stages
    # When enabled, run LaMa inpainting after OCR/translate and return inpainted PNG
    enable_inpainting: bool = True
    lama_model_path: str = "models/lama.onnx"
    # Encode the inpainted "plate" as WebP (lossy, q=82) instead of uncompressed
    # PNG base64. Cuts the per-page plate payload ~91% (PNG 3.38MB -> WebP ~0.28MB)
    # with no visible quality loss on manga line-art. WebP decodes natively in the
    # browser canvas, so no frontend change is needed. Set False to restore PNG.
    plate_encode_webp: bool = True
    plate_webp_quality: int = 82
    # bubbleRect-gated interior solid-fill inpaint tier (R1 hybrid). When on, the
    # LaMa service fills flat speech-bubble interiors with their robust median
    # background and skips the neural forward for those components. Purely
    # additive + gated; False instantly restores the prior 3-tier behaviour.
    enable_bubble_solid_fill: bool = True
    # Final inpaint tier: when True, the textured/screentone residual that used to
    # go through the LaMa neural forward is instead handled by cv2.inpaint
    # (Navier-Stokes, r=3) — a purely classical (no-AI) reconstruction. The
    # bubble solid-fill / ring fast-path / classical-NS tiers are unchanged; this
    # only swaps tier-3. Audit (2026-06-13_noai-inpaint-audit.md) over 11
    # benchmark pages found 85% of inpainted pixels are hidden by the re-rendered
    # translation and the visible residual is imperceptible on dialogue; only
    # large SFX-over-detailed-art (which is largely left un-erased in production
    # anyway) is mildly softer than LaMa. Removing the neural tier drops the
    # 208MB ONNX model load + GPU working set and the ~28ms/forward on the ~40%
    # of components that previously hit the model. Set False to restore LaMa.
    use_neural_inpaint: bool = False
    # Overlap LaMa inpaint with OCR+translate. Inpainting only needs the detection
    # mask (not translated text), so it can run concurrently with the OCR/translate
    # stage instead of serially after it. Runs in a worker thread so the event loop
    # stays free to drive the vLLM translate calls.
    overlap_inpaint: bool = True
    # When enabled, detect speech bubbles (YOLOv10n) and expose the matched
    # bubble interior per text box (bubbleRect) so the frontend can typeset the
    # translation to the bubble rather than the tight (vertical-JP) text column.
    enable_bubble_fit: bool = True
    # When enabled, use page-level [N]-tagged batched translation (coherence win)
    # instead of per-bubble parallel calls. Fallback to parallel on failure.
    # NOTE: the current vLLM translate_batched fans out concurrent single-bubble
    # calls — it does NOT pack bubbles into one prompt. See batch_translate below
    # for the true single-call numbered-block path.
    use_batched_translation: bool = True

    # TRUE single-call numbered-block translation: pack all of a page's bubbles
    # into ONE vLLM generate call (1.,2.,3.… prompt, numbered output parsed back).
    # QUALITY-GATED + OFF by default — production behaviour is unchanged. Enable
    # only after a chrF++ holdout A/B confirms no regression vs the per-bubble path.
    batch_translate: bool = False

    # Per-bubble translation generation budget. Manga lines are short; 64 tokens
    # comfortably covers a translated bubble (~measured outputs well under this).
    # Lower = fewer decode steps on overlong generations. Raise if truncation seen.
    translate_max_tokens: int = 64

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

"""ONNX Manga OCR service using l0wgear/manga-ocr-2025-onnx."""

import asyncio
import logging
import re
import time
from typing import List

import numpy as np
from PIL import Image
import onnxruntime as ort

# Suppress ONNX runtime warnings about node assignments to execution providers
ort.set_default_logger_severity(3)  # 3 = ERROR only

logger = logging.getLogger(__name__)


def normalize_japanese_text(text: str) -> str:
    """
    Remove spaces between Japanese characters (hiragana, katakana, kanji).

    Japanese doesn't use spaces between words, but OCR models often
    output spaces between tokens. This fixes: ゴ ー ル!! → ゴール!!

    Args:
        text: Raw OCR output text

    Returns:
        Normalized text with spurious spaces removed
    """
    # Pattern: space between two Japanese characters
    # Japanese ranges: Hiragana (\u3040-\u309F), Katakana (\u30A0-\u30FF),
    #                  Kanji (\u4E00-\u9FFF), Extended (\u3400-\u4DBF)
    japanese_char = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]'

    # Remove space between two Japanese characters
    pattern = f'({japanese_char})\\s+({japanese_char})'
    while re.search(pattern, text):
        text = re.sub(pattern, r'\1\2', text)

    return text.strip()

# Global GPU inference semaphore - serializes OCR GPU operations
_ocr_gpu_semaphore = asyncio.Semaphore(1)


class MangaOCRService:
    """
    OCR service using ONNX-optimized manga-ocr model.

    Uses l0wgear/manga-ocr-2025-onnx with CUDAExecutionProvider for
    GPU-accelerated inference with native batching support.
    """

    def __init__(self, model_id: str | None = None):
        """
        Initialize the ONNX manga OCR model.

        Args:
            model_id: HuggingFace model ID. Defaults to l0wgear/manga-ocr-2025-onnx.
        """
        from transformers import AutoTokenizer, AutoImageProcessor
        from optimum.onnxruntime import ORTModelForVision2Seq

        if model_id is None:
            model_id = "l0wgear/manga-ocr-2025-onnx"

        logger.info(f"Loading ONNX manga-ocr model: {model_id}")
        start_time = time.perf_counter()

        # Load image processor and tokenizer separately
        self.image_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load ONNX model with CUDA provider
        # use_cache=False because model doesn't have decoder_with_past_model.onnx
        try:
            self.model = ORTModelForVision2Seq.from_pretrained(
                model_id,
                provider="CUDAExecutionProvider",
                use_cache=False,
                decoder_file_name="decoder_model.onnx",
                encoder_file_name="encoder_model.onnx"
            )
            self.device = "cuda"
            logger.info("ONNX model loaded with CUDAExecutionProvider")
        except Exception as e:
            logger.warning(f"CUDA failed ({e}), falling back to CPU")
            self.model = ORTModelForVision2Seq.from_pretrained(
                model_id,
                provider="CPUExecutionProvider",
                use_cache=False,
                decoder_file_name="decoder_model.onnx",
                encoder_file_name="encoder_model.onnx"
            )
            self.device = "cpu"

        load_time = time.perf_counter() - start_time
        logger.info(f"ONNX manga-ocr loaded in {load_time:.2f}s on {self.device}")

    async def recognize_text_batch(
        self,
        image_crops: List[np.ndarray],
        batch_size: int = 8
    ) -> List[str]:
        """
        Perform batched OCR on image crops.

        Args:
            image_crops: List of image regions as numpy arrays (RGB)
            batch_size: Not used - processes all at once for simplicity

        Returns:
            List of recognized text strings, one per crop
        """
        if not image_crops:
            return []

        total_start = time.perf_counter()

        # Convert to PIL images
        pil_images = [
            Image.fromarray(crop).convert("RGB")
            for crop in image_crops
        ]

        # Batch preprocess images
        inputs = self.image_processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs.pixel_values

        # Move to device BEFORE thread (CUDA ops must stay in main async thread)
        if self.device == "cuda":
            pixel_values = pixel_values.to("cuda")

        # Batched generation with GPU semaphore
        async with _ocr_gpu_semaphore:
            generated_ids = await asyncio.to_thread(
                self.model.generate,
                pixel_values
            )

        # Batch decode using tokenizer
        texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        # Normalize: remove spaces between Japanese characters (e.g., ゴ ー ル!! → ゴール!!)
        texts = [normalize_japanese_text(t) for t in texts]

        total_time = (time.perf_counter() - total_start) * 1000
        avg_time = total_time / len(image_crops)
        logger.info(
            f"ONNX OCR batch: {len(image_crops)} crops in {total_time:.1f}ms "
            f"(avg {avg_time:.1f}ms/crop)"
        )

        return texts

    async def recognize_single(self, image_crop: np.ndarray) -> str:
        """
        Perform OCR on a single image crop.

        Useful for pipeline overlap where OCR crops are processed one at a time
        and translation can start immediately on each result.

        Args:
            image_crop: Single image region as numpy array (RGB)

        Returns:
            Recognized text string
        """
        results = await self.recognize_text_batch([image_crop])
        return results[0] if results else ""

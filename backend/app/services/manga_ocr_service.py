"""Manga OCR service using kha-white/manga-ocr."""

import asyncio
import logging
import time
from typing import List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Global GPU inference semaphore - serializes ALL OCR GPU operations
# Prevents GPU contention when multiple inferences run simultaneously
_ocr_gpu_semaphore = asyncio.Semaphore(1)


class MangaOCRService:
    """
    OCR service using kha-white/manga-ocr.

    This model is specifically trained for Japanese manga text recognition
    and handles vertical text, stylized fonts, and various text orientations.
    It's fast (~100-200ms per image) and accurate.
    """

    def __init__(self, model_id: str | None = None):
        """
        Initialize the Manga OCR model.

        Args:
            model_id: Not used - manga-ocr auto-downloads its model.
                      Kept for interface compatibility.
        """
        from manga_ocr import MangaOcr

        logger.info("Loading manga-ocr model...")
        start_time = time.perf_counter()

        # MangaOcr auto-detects CUDA and handles device placement
        self._mocr = MangaOcr()

        # Try to apply torch.compile for optimization (PyTorch 2.0+)
        self._compiled = False
        try:
            import torch
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                logger.info("Applying torch.compile() optimization...")
                self._mocr.model = torch.compile(self._mocr.model, mode="reduce-overhead")
                self._compiled = True
                logger.info("torch.compile() applied successfully")
        except Exception as e:
            logger.warning(f"torch.compile() not available or failed: {e}")

        load_time = time.perf_counter() - start_time
        device = "cuda" if self._is_cuda_available() else "cpu"
        logger.info(f"manga-ocr loaded in {load_time:.2f}s on {device}")

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _recognize_sync(self, pil_image: Image.Image) -> str:
        """
        Synchronous OCR recognition on a single PIL image.

        Args:
            pil_image: PIL Image to recognize text from

        Returns:
            Recognized text string
        """
        return self._mocr(pil_image)

    async def recognize_text_batch(
        self,
        image_crops: List[np.ndarray],
        batch_size: int = 4
    ) -> List[str]:
        """
        Perform OCR on a list of image crops.

        Processes images sequentially since manga-ocr is already fast (~100-200ms).
        Uses asyncio.to_thread() to avoid blocking the event loop.

        Args:
            image_crops: List of image regions as numpy arrays (RGB)
            batch_size: Not used - kept for interface compatibility.
                       manga-ocr processes one image at a time.

        Returns:
            List of recognized text strings, one per crop
        """
        if not image_crops:
            return []

        results = []
        total_start = time.perf_counter()

        for i, crop in enumerate(image_crops):
            crop_start = time.perf_counter()

            try:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(crop)

                # Acquire semaphore to serialize GPU access
                async with _ocr_gpu_semaphore:
                    # Run sync OCR in thread pool to avoid blocking
                    text = await asyncio.to_thread(self._recognize_sync, pil_image)

                crop_time = (time.perf_counter() - crop_start) * 1000
                text_preview = text[:50] + "..." if len(text) > 50 else text
                logger.debug(f"OCR crop {i+1}/{len(image_crops)}: '{text_preview}' ({crop_time:.1f}ms)")

                results.append(text)

            except Exception as e:
                logger.warning(f"OCR failed for crop {i+1}: {e}")
                results.append("")

        total_time = (time.perf_counter() - total_start) * 1000
        avg_time = total_time / len(image_crops) if image_crops else 0
        logger.info(
            f"OCR batch complete: {len(image_crops)} crops in {total_time:.1f}ms "
            f"(avg {avg_time:.1f}ms/crop)"
        )

        return results

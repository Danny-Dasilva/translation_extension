"""PaddleOCR-VL manga OCR service using Vision-Language model."""

import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from app.config import settings

logger = logging.getLogger(__name__)


class MangaOCRService:
    """
    OCR service using PaddleOCR-VL-For-Manga Vision-Language model.

    This model is specifically trained for manga text recognition and
    handles vertical text, stylized fonts, and sound effects better
    than traditional OCR.
    """

    def __init__(self, model_id: str | None = None):
        """
        Initialize the OCR model.

        Args:
            model_id: HuggingFace model ID or local path. Defaults to settings.
        """
        if model_id is None:
            model_id = settings.ocr_model_id

        # Check for local model first
        local_path = Path(settings.weights_dir) / "paddleocr-vl"
        if local_path.exists():
            model_id = str(local_path)
            logger.info(f"Using local OCR model from {model_id}")
        else:
            logger.info(f"Using HuggingFace model: {model_id}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"OCR device: {self.device}")

        # Determine attention implementation
        attn_impl = None
        if self.device == "cuda":
            try:
                # Try Flash Attention 2 for speed
                attn_impl = "flash_attention_2"
            except Exception:
                attn_impl = "sdpa"  # Fallback to scaled dot product attention

        logger.info(f"Loading PaddleOCR-VL model...")

        # Load model with optimizations
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
        }
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        logger.info("PaddleOCR-VL model loaded successfully")

    async def recognize_text(self, image_crops: List[np.ndarray]) -> List[str]:
        """
        Perform OCR on a list of image crops.

        Args:
            image_crops: List of image regions as numpy arrays (RGB)

        Returns:
            List of recognized text strings, one per crop
        """
        if not image_crops:
            return []

        texts = []
        for i, crop in enumerate(image_crops):
            try:
                text = await self._recognize_single(crop)
                texts.append(text)
                logger.debug(f"OCR crop {i+1}/{len(image_crops)}: '{text[:50]}...' " if len(text) > 50 else f"OCR crop {i+1}/{len(image_crops)}: '{text}'")
            except Exception as e:
                logger.warning(f"OCR failed for crop {i+1}: {e}")
                texts.append("")

        return texts

    async def _recognize_single(self, image: np.ndarray) -> str:
        """
        Recognize text from a single image crop.

        Args:
            image: Image as numpy array (RGB)

        Returns:
            Recognized text string
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Prepare chat message for OCR
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "OCR:"}
            ]
        }]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            images=[pil_image],
            text=[text],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False  # Deterministic for OCR
            )

        # Decode output
        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # Extract text after "OCR:" prompt
        clean_text = output.split("OCR:")[-1].strip()

        return clean_text

    async def recognize_text_batch(
        self,
        image_crops: List[np.ndarray],
        batch_size: int = 4
    ) -> List[str]:
        """
        Batch OCR for better throughput (experimental).

        Note: Current implementation processes sequentially.
        True batching requires model-specific optimizations.
        """
        # For now, delegate to sequential processing
        # TODO: Implement true batching when model supports it
        return await self.recognize_text(image_crops)

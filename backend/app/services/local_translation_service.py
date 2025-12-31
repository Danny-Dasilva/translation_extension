"""Local translation service using HY-MT1.5 GGUF model via llama-cpp."""

import logging
from pathlib import Path
from typing import List

from llama_cpp import Llama

from app.config import settings

logger = logging.getLogger(__name__)


class LocalTranslationService:
    """
    Translation service using HY-MT1.5-1.8B GGUF model.

    HY-MT1.5 is a machine translation model optimized for Japanese to English.
    Uses llama-cpp-python for efficient GGUF inference with GPU acceleration.
    """

    def __init__(self, model_path: str | None = None):
        """
        Initialize the translation model.

        Args:
            model_path: Path to the GGUF model file. Defaults to settings.
        """
        if model_path is None:
            model_path = settings.translation_model_path

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Translation model not found at {model_path}. "
                "Run: uv run python scripts/download_models.py --translation"
            )

        logger.info(f"Loading HY-MT1.5 model from {model_path}")

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,           # Context window
            n_gpu_layers=-1,      # All layers on GPU (use 0 for CPU-only)
            n_threads=4,          # CPU threads for non-GPU operations
            verbose=False
        )

        logger.info("HY-MT1.5 model loaded successfully")

    async def translate_single(
        self,
        text: str,
        target_language: str = "English"
    ) -> str:
        """
        Translate a single text string.

        Args:
            text: Japanese text to translate
            target_language: Target language (default: English)

        Returns:
            Translated text
        """
        if not text.strip():
            return ""

        prompt = f"""Translate the following segment into {target_language}, without additional explanation.

{text}"""

        response = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.3,
            stop=["\n\n"],  # Stop at double newline
            echo=False
        )

        translation = response["choices"][0]["text"].strip()
        return translation

    async def translate_batch(
        self,
        texts: List[str],
        target_language: str = "English"
    ) -> List[str]:
        """
        Translate a list of texts individually.

        Args:
            texts: List of Japanese texts to translate
            target_language: Target language (default: English)

        Returns:
            List of translated texts
        """
        translations = []

        for i, text in enumerate(texts):
            try:
                translation = await self.translate_single(text, target_language)
                translations.append(translation)
                logger.debug(f"Translated {i+1}/{len(texts)}: '{text[:30]}...' -> '{translation[:30]}...'" if len(text) > 30 else f"Translated {i+1}/{len(texts)}: '{text}' -> '{translation}'")
            except Exception as e:
                logger.warning(f"Translation failed for text {i+1}: {e}")
                translations.append(text)  # Return original on failure

        return translations

    async def translate_batch_concatenated(
        self,
        texts: List[str],
        target_language: str = "English"
    ) -> List[str]:
        """
        Optimized batch translation using concatenated prompt.

        Combines multiple texts into a single numbered prompt for faster
        processing when dealing with many short texts (typical for manga bubbles).

        Args:
            texts: List of Japanese texts to translate
            target_language: Target language (default: English)

        Returns:
            List of translated texts
        """
        if not texts:
            return []

        # Filter empty texts but track their positions
        indexed_texts = [(i, t) for i, t in enumerate(texts) if t.strip()]

        if not indexed_texts:
            return [""] * len(texts)

        # Build numbered prompt
        numbered = "\n".join(f"{i+1}. {t}" for i, (_, t) in enumerate(indexed_texts))

        prompt = f"""Translate each numbered line from Japanese to {target_language}. Keep the numbering. Only output translations, no explanations.

{numbered}"""

        response = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.3,
            echo=False
        )

        # Parse numbered response
        output = response["choices"][0]["text"].strip()
        lines = output.split("\n")

        # Extract translations from numbered lines
        parsed_translations = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Strip number prefix: "1. Hello" -> "Hello"
            if ". " in line:
                parsed_translations.append(line.split(". ", 1)[-1].strip())
            else:
                parsed_translations.append(line)

        # Build result list with empty strings for originally empty texts
        result = [""] * len(texts)
        for (orig_idx, _), translation in zip(indexed_texts, parsed_translations):
            result[orig_idx] = translation

        # Fill any missing translations with original text
        for orig_idx, orig_text in indexed_texts:
            if not result[orig_idx]:
                result[orig_idx] = orig_text

        return result

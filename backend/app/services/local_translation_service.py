"""Local translation service using HY-MT1.5-1.8B-Q4_K_M via llama-cpp."""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Tuple

from llama_cpp import Llama

from app.config import settings

logger = logging.getLogger(__name__)

# Global GPU inference semaphore - DISABLED: serializes ALL translation GPU operations
# Prevents GPU contention that causes slowdown when multiple inferences run simultaneously
# This was defeating the purpose of having multiple instances - now each instance can run in parallel
# _translation_gpu_semaphore = asyncio.Semaphore(1)


class LocalTranslationService:
    """
    Translation service using HY-MT1.5-1.8B-Q4_K_M via llama-cpp-python.

    Uses llama-cpp-python for efficient GGUF inference with GPU acceleration.
    """

    def __init__(self, model_path: str | None = None):
        """
        Initialize the HY-MT1.5 translation model.

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
            n_ctx=2048,
            n_gpu_layers=-1,
            n_threads=4,
            verbose=False,
        )

        logger.info("HY-MT1.5 model loaded")

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

        # Format prompt directly (bypass create_chat_completion overhead)
        prompt = f"<|im_start|>user\nTranslate the following segment into {target_language}, without additional explanation.\n\n{text}<|im_end|>\n<|im_start|>assistant\n"

        # Run synchronously - llama-cpp GPU offload means most time is GPU-bound anyway
        # Avoiding asyncio.to_thread eliminates thread pool overhead and lock contention
        # that was causing progressive slowdown (31ms → 315ms over 6 translations)
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=256,
            temperature=0.7,
            top_k=20,
            top_p=0.6,
            repeat_penalty=1.05,
            stop=["<|im_end|>"]
        )
        translation = response["choices"][0]["text"].strip()

        # Remove "Assistant:" prefix if present (model chat template artifact)
        if translation.startswith("Assistant:"):
            translation = translation[len("Assistant:"):].strip()

        # Strip any special end tokens that may have leaked through
        for token in ["<|im_end|>", "<|im_end>", "</s>", "<|eot_id|>"]:
            translation = translation.replace(token, "")
        translation = translation.strip()

        return translation


class LocalTranslationPool:
    """
    Pool of Llama instances for true parallel translation.

    Loads multiple model instances, each with its own semaphore, allowing
    parallel translation without lock contention that caused progressive slowdown.

    VRAM Usage: ~1.5GB per instance
    Recommended: 3 instances for 32GB VRAM
    """

    def __init__(self, num_instances: int | None = None, model_path: str | None = None):
        """
        Initialize multiple translation model instances.

        Args:
            num_instances: Number of model instances to load. Defaults to settings.
            model_path: Path to the GGUF model file. Defaults to settings.
        """
        if num_instances is None:
            num_instances = settings.translation_num_instances
        if model_path is None:
            model_path = settings.translation_model_path

        self.num_instances = num_instances
        self.instances: List[Llama] = []
        self.semaphores: List[asyncio.Semaphore] = []

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Translation model not found at {model_path}. "
                "Run: uv run python scripts/download_models.py --translation"
            )

        logger.info(f"Translation Pool: Loading {num_instances} instances from {model_path}")

        for i in range(num_instances):
            logger.info(f"Loading translation instance {i+1}/{num_instances}...")

            llm = Llama(
                model_path=str(model_path),
                n_ctx=2048,
                n_gpu_layers=-1,
                n_threads=2,  # Reduced threads per instance
                verbose=False,
            )

            self.instances.append(llm)
            self.semaphores.append(asyncio.Semaphore(1))

        logger.info(f"Translation Pool ready: {num_instances} instances loaded")

    def _translate_sync(
        self, llm: Llama, text: str, target_language: str,
        instance_id: int = -1, text_idx: int = -1
    ) -> str:
        """
        Synchronous translation on a specific Llama instance.

        Args:
            llm: The Llama instance to use
            text: Japanese text to translate
            target_language: Target language (default: English)
            instance_id: Instance ID for logging
            text_idx: Text index for logging

        Returns:
            Translated text
        """
        logger.info(f"[DEBUG-SYNC] Instance {instance_id}, Text {text_idx}: ENTERED _translate_sync")
        t0 = time.perf_counter()

        if not text.strip():
            logger.info(f"[DEBUG-SYNC] Instance {instance_id}, Text {text_idx}: empty text, returning")
            return ""

        # Format prompt
        prompt = f"<|im_start|>user\nTranslate the following segment into {target_language}, without additional explanation.\n\n{text}<|im_end|>\n<|im_start|>assistant\n"
        t_prompt = time.perf_counter()
        logger.info(f"[DEBUG-SYNC] Instance {instance_id}, Text {text_idx}: calling llm.create_completion")

        # Inference
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=256,
            temperature=0.7,
            top_k=20,
            top_p=0.6,
            repeat_penalty=1.05,
            stop=["<|im_end|>"]
        )
        logger.info(f"[DEBUG-SYNC] Instance {instance_id}, Text {text_idx}: llm.create_completion returned")
        t_inference = time.perf_counter()

        # Cleanup
        translation = response["choices"][0]["text"].strip()
        if translation.startswith("Assistant:"):
            translation = translation[len("Assistant:"):].strip()

        # Strip any special end tokens that may have leaked through
        for token in ["<|im_end|>", "<|im_end>", "</s>", "<|eot_id|>"]:
            translation = translation.replace(token, "")
        translation = translation.strip()

        t_cleanup = time.perf_counter()

        # Log detailed timing breakdown
        logger.info(
            f"[Trans-{instance_id}] Text {text_idx}: "
            f"prompt={(t_prompt-t0)*1000:.1f}ms, "
            f"INFER={(t_inference-t_prompt)*1000:.1f}ms, "
            f"clean={(t_cleanup-t_inference)*1000:.1f}ms, "
            f"TOTAL={(t_cleanup-t0)*1000:.1f}ms"
        )

        return translation

    async def translate_parallel(
        self,
        texts: List[str],
        target_language: str = "English"
    ) -> List[str]:
        """
        Translate all texts in parallel across model instances.

        Each instance processes texts assigned to it round-robin style.
        Per-instance semaphores prevent contention within each instance.

        Args:
            texts: List of texts to translate
            target_language: Target language for translation

        Returns:
            List of translated texts (order preserved)
        """
        if not texts:
            return []

        async def translate_one(idx: int, text: str) -> Tuple[int, str]:
            """Translate single text on assigned instance."""
            instance_id = idx % self.num_instances
            llm = self.instances[instance_id]

            # Per-instance semaphore: prevents concurrent calls to same llama instance
            # (llama-cpp is NOT thread-safe for concurrent calls to same instance)
            async with self.semaphores[instance_id]:
                try:
                    # Run sync function in thread pool to avoid blocking event loop
                    translation = await asyncio.to_thread(
                        self._translate_sync, llm, text, target_language, instance_id, idx
                    )
                    logger.debug(f"Trans[{instance_id}] text {idx+1}: '{translation[:30]}...'" if len(translation) > 30 else f"Trans[{instance_id}] text {idx+1}: '{translation}'")
                    return (idx, translation)
                except Exception as e:
                    logger.warning(f"Trans[{instance_id}] text {idx+1} failed: {e}")
                    return (idx, "")

        # Create tasks for all texts (distributed across instances)
        tasks = [translate_one(i, text) for i, text in enumerate(texts)]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Sort by index to preserve order
        results.sort(key=lambda x: x[0])
        return [trans for _, trans in results]

    async def translate_single(
        self,
        text: str,
        target_language: str = "English"
    ) -> str:
        """
        Translate a single text using next available instance.

        For backward compatibility with code expecting single translate calls.

        Args:
            text: Text to translate
            target_language: Target language

        Returns:
            Translated text
        """
        # Use instance 0 for single translations
        return await asyncio.to_thread(
            self._translate_sync, self.instances[0], text, target_language, 0, 0
        )

    async def translate_streaming(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        target_language: str = "English",
        num_workers: int | None = None
    ) -> None:
        """
        Stream translations from input queue to output queue.

        Used for pipeline overlap where OCR feeds translations as results arrive.

        Args:
            input_queue: Queue providing (index, text) tuples, None signals end
            output_queue: Queue to put (index, ocr_text, translation) tuples
            target_language: Target language for translation
            num_workers: Number of concurrent workers (defaults to num_instances)
        """
        if num_workers is None:
            num_workers = self.num_instances

        async def worker(worker_id: int):
            """Translation worker consuming from queue."""
            while True:
                item = await input_queue.get()
                if item is None:
                    # Put None back for other workers
                    await input_queue.put(None)
                    break

                idx, text = item
                instance_id = worker_id % self.num_instances

                # Per-instance execution allows true parallel translation across instances
                try:
                    # Run sync function in thread pool to avoid blocking event loop
                    translation = await asyncio.to_thread(
                        self._translate_sync,
                        self.instances[instance_id], text, target_language, instance_id, idx
                    )
                    await output_queue.put((idx, text, translation))
                except Exception as e:
                    logger.warning(f"Trans[{instance_id}] text {idx+1} failed: {e}")
                    await output_queue.put((idx, text, ""))

        # Start worker tasks
        workers = [worker(i) for i in range(num_workers)]
        await asyncio.gather(*workers)

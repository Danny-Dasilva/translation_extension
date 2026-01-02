"""Local translation service using HY-MT1.5-1.8B-Q4_K_M via llama-cpp."""

import asyncio
import concurrent.futures
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

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

    @staticmethod
    def _clean_translation_output(translation: str) -> str:
        """
        Clean up translation output by removing model artifacts.

        Removes "Assistant:" prefix and special end tokens that may leak through.

        Args:
            translation: Raw translation text from model

        Returns:
            Cleaned translation text
        """
        translation = translation.strip()

        # Remove "Assistant:" prefix if present (model chat template artifact)
        if translation.startswith("Assistant:"):
            translation = translation[len("Assistant:"):].strip()

        # Strip any special end tokens that may have leaked through
        for token in ["<|im_end|>", "<|im_end>", "</s>", "<|eot_id|>"]:
            translation = translation.replace(token, "")

        return translation.strip()

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
        translation = response["choices"][0]["text"]
        return self._clean_translation_output(translation)


class LocalTranslationPool:
    """
    Pool of Llama instances for true parallel translation.

    Loads multiple model instances, each with its own semaphore, allowing
    parallel translation without lock contention that caused progressive slowdown.

    VRAM Usage: ~1.5GB per instance
    Recommended: 6 instances for 32GB VRAM (handles 6-bubble pages in single round)
    """

    def __init__(self, num_instances: int | None = None, model_path: str | None = None):
        """
        Initialize multiple translation model instances in parallel.

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

        logger.info(f"Translation Pool: Loading {num_instances} instances in parallel from {model_path}")
        load_start = time.perf_counter()

        def load_single_instance(instance_id: int) -> Tuple[int, Llama]:
            """Load a single Llama instance (for parallel loading)."""
            instance_start = time.perf_counter()
            llm = Llama(
                model_path=str(model_path),
                n_ctx=settings.translation_n_ctx,
                n_batch=settings.translation_n_batch,
                n_ubatch=settings.translation_n_ubatch,
                n_gpu_layers=-1,
                n_threads=2,  # Reduced threads per instance
                verbose=False,
            )
            elapsed = (time.perf_counter() - instance_start) * 1000
            logger.info(f"Translation instance {instance_id+1}/{num_instances} loaded in {elapsed:.0f}ms")
            return (instance_id, llm)

        # Load all instances in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_instances) as executor:
            futures = [executor.submit(load_single_instance, i) for i in range(num_instances)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Sort by instance_id to maintain order and extract instances
        results.sort(key=lambda x: x[0])
        self.instances = [llm for _, llm in results]
        self.semaphores = [asyncio.Semaphore(1) for _ in self.instances]

        load_time = (time.perf_counter() - load_start) * 1000
        logger.info(f"Translation Pool ready: {num_instances} instances loaded in {load_time:.0f}ms")

    async def warmup(self) -> Dict[str, Any]:
        """
        Warm up all translation instances with dummy inference.

        First inference on each instance has cold-start latency. This warms up
        all instances at startup to ensure consistent latency.

        Returns:
            dict with warmup timing statistics
        """
        warmup_text = "テスト"
        timings = []

        async def warmup_one(instance_id: int) -> float:
            """Warmup single instance."""
            start = time.perf_counter()
            await asyncio.to_thread(
                self._translate_sync,
                self.instances[instance_id], warmup_text, "English", instance_id, -1
            )
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug(f"Instance {instance_id} warmup: {elapsed:.1f}ms")
            return elapsed

        # Warmup all instances in parallel
        tasks = [warmup_one(i) for i in range(self.num_instances)]
        timings = await asyncio.gather(*tasks)

        stats = {
            'num_instances': self.num_instances,
            'avg_warmup_ms': sum(timings) / len(timings),
            'max_warmup_ms': max(timings),
            'total_warmup_ms': max(timings),  # Parallel = max time
        }

        logger.info(
            f"Translation warmup complete: {self.num_instances} instances, "
            f"avg={stats['avg_warmup_ms']:.1f}ms, max={stats['max_warmup_ms']:.1f}ms"
        )

        return stats

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
        t0 = time.perf_counter()

        if not text.strip():
            logger.debug(f"Instance {instance_id}, text {text_idx}: empty text, skipping")
            return ""

        # Format prompt
        prompt = f"<|im_start|>user\nTranslate the following segment into {target_language}, without additional explanation.\n\n{text}<|im_end|>\n<|im_start|>assistant\n"

        # Inference
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=settings.translation_max_tokens,
            temperature=0.7,
            top_k=20,
            top_p=0.6,
            repeat_penalty=1.05,
            stop=["<|im_end|>"]
        )

        translation = LocalTranslationService._clean_translation_output(
            response["choices"][0]["text"]
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Instance {instance_id}, text {text_idx}: completed in {elapsed_ms:.1f}ms")

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
        Translate a single text using instance 0 with semaphore protection.

        For backward compatibility with code expecting single translate calls.
        Uses semaphore to prevent concurrent access to the same Llama instance.

        Args:
            text: Text to translate
            target_language: Target language

        Returns:
            Translated text
        """
        # Use instance 0 with semaphore protection (llama-cpp is NOT thread-safe)
        async with self.semaphores[0]:
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

"""Local translation service using HY-MT1.5-1.8B-Q4_K_M via llama-cpp.

TODO(router-integration): The router currently calls `translate_parallel` on
`LocalTranslationPool`. Once the batched path below has been validated end-to-end,
`backend/app/routers/translate.py` should be updated to prefer
`translate_batched(texts, target_language)` to get page-level coherence plus the
speedup from a single prompt-processing pass. The legacy parallel path is kept
intact for fallback / A-B comparison.
"""

import asyncio
import concurrent.futures
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Disable CUDA graph capture before importing llama-cpp to prevent conflicts with onnxruntime
os.environ.setdefault("GGML_CUDA_NO_GRAPHS", "1")

from llama_cpp import Llama

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Koharu-compatible batched translation protocol
# Ported from:
#   /tmp/koharu/koharu-llm/src/prompt.rs:50-57    (system prompt)
#   /tmp/koharu/koharu-app/src/llm.rs:439-524     (format/parse/strip helpers)
# ---------------------------------------------------------------------------

BATCHED_SYSTEM_PROMPT = (
    "You are a professional manga translator. Translate manga dialogue into "
    "natural {target} that fits inside speech bubbles. Preserve character voice, "
    "emotional tone, relationship nuance, emphasis, and sound effects naturally. "
    "Keep the wording concise. Do not add notes, explanations, or romanization. "
    "The input uses numbered tags like [1], [2], etc. to mark each text block. "
    "Translate only the text after each tag. Keep every tag exactly unchanged, "
    "including numbers and order. Output the same tags followed by the "
    "translated text. Do not merge, split, or reorder blocks."
)


def format_sources(texts: List[str]) -> str:
    """Format a list of source strings into koharu's tagged-block body.

    Port of `format_sources` at `/tmp/koharu/koharu-app/src/llm.rs:439-446`.
    Produces: `[1]text1\\n[2]text2\\n...[N]textN`.
    """
    return "\n".join(f"[{i + 1}]{text}" for i, text in enumerate(texts))


def strip_thinking_block(text: str) -> str:
    """Remove any ``<think>...</think>`` wrapper from model output.

    Port of `strip_thinking_block` at `/tmp/koharu/koharu-app/src/llm.rs:517-524`.
    """
    start = text.find("<think>")
    if start == -1:
        return text
    end_rel = text[start:].find("</think>")
    if end_rel == -1:
        return text
    return text[start + end_rel + len("</think>"):].lstrip()


def strip_wrapping_quotes(text: str) -> str:
    """Strip matching single or double quotes wrapping a string.

    Port of `strip_wrapping_quotes` at `/tmp/koharu/koharu-app/src/llm.rs:526-538`.
    """
    trimmed = text.strip()
    if len(trimmed) >= 2:
        first = trimmed[0]
        last = trimmed[-1]
        if (first == '"' and last == '"') or (first == "'" and last == "'"):
            return trimmed[1:-1]
    return trimmed


_TAG_RE = re.compile(r"\[(\d+)\]\s*([^\[]*)")


def parse_tagged_blocks(output: str, n: int) -> Optional[List[str]]:
    """Parse tagged-block translation output into an ordered list of length n.

    Port of `parse_tagged_blocks` at `/tmp/koharu/koharu-app/src/llm.rs:483-503`.
    Returns None if no tags were found (caller should fall back to legacy split).
    """
    matches = _TAG_RE.findall(output)
    if not matches:
        return None
    blocks = [""] * n
    for num_str, content in matches:
        try:
            idx_1based = int(num_str)
        except ValueError:
            continue
        if idx_1based <= 0:
            continue
        idx = idx_1based - 1
        if idx < n:
            blocks[idx] = content.strip()
    return blocks


def split_legacy_lines(output: str, n: int) -> List[str]:
    """Legacy fallback: split by newlines and pad/truncate to length n.

    Port of `split_legacy_lines` at `/tmp/koharu/koharu-app/src/llm.rs:505-515`.
    """
    lines = [line.rstrip("\r") for line in output.splitlines()]
    if len(lines) > n:
        lines = lines[:n]
    while len(lines) < n:
        lines.append("")
    return lines


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

        # Strip any special tokens that may have leaked through
        # Use regex to catch all variants (e.g. <|im_end|>, <|im_end+], <|im_end/>, etc.)
        translation = re.sub(r'<\|im_\w*[^>]*[>\]|/]+', '', translation)
        for token in ["</s>", "<|eot_id|>"]:
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

        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=256,
            temperature=0.3,
            top_k=20,
            top_p=0.9,
            repeat_penalty=1.05,
            stop=["<|im_end|>"]
        )
        translation = response["choices"][0]["text"]
        return self._clean_translation_output(translation)

    async def translate_batched(
        self,
        texts: List[str],
        target_language: str = "English"
    ) -> List[str]:
        """
        Page-level batched translation using koharu's tagged-block protocol.

        Sends all bubble texts in a single LLM call formatted as
        ``[1]text1\\n[2]text2\\n...[N]textN`` and parses the same structure from
        the model response. This gives the model intra-page coherence (tone,
        pronouns, names) and usually runs faster than N independent calls
        because only one prompt-processing pass is needed.

        Args:
            texts: List of source texts.
            target_language: Target language (default: English).

        Returns:
            List of translations with ``len(out) == len(texts)``. On any
            failure, returns a list of empty strings of the correct length.
        """
        return await _batched_translate_on_instance(
            self.llm, texts, target_language
        )


async def _batched_translate_on_instance(
    llm: Llama,
    texts: List[str],
    target_language: str,
) -> List[str]:
    """Run one batched tagged-block translation on a given Llama instance.

    Guarantees ``len(out) == len(texts)``. On any exception returns
    ``[""] * len(texts)`` and logs the error.
    """
    if not texts:
        return []

    def _run_sync() -> List[str]:
        return _batched_translate_sync(llm, texts, target_language)

    try:
        return await asyncio.to_thread(_run_sync)
    except Exception as e:  # pragma: no cover - safety net
        logger.warning(f"Batched translate failed: {e!r}")
        return [""] * len(texts)


def _batched_translate_sync(
    llm: Llama,
    texts: List[str],
    target_language: str,
) -> List[str]:
    """Synchronous core of batched translation. See `translate_batched` doc.

    Pipeline (matches koharu ordering):
      1. Build chat messages with `BATCHED_SYSTEM_PROMPT` + formatted sources.
      2. Prefer `llama.create_chat_completion` (uses GGUF's embedded chat template).
      3. Fall back to hand-built ChatML prompt + `create_completion` if that errors.
      4. `strip_thinking_block` → `parse_tagged_blocks` → `split_legacy_lines` fallback
         → per-block `strip_wrapping_quotes`.
    """
    n = len(texts)
    if n == 0:
        return []

    system_content = BATCHED_SYSTEM_PROMPT.format(target=target_language)
    user_content = format_sources(texts)

    # Hunyuan-family models in koharu's path (prompt.rs:90-92) use a single user
    # message with the system prompt prepended. We pass both a two-message
    # (system + user) layout AND a single-user combined layout; if the first
    # produces unparseable output we retry with the combined layout.
    gen_kwargs = dict(
        temperature=0.1,
        top_k=40,
        top_p=0.9,
        repeat_penalty=1.05,
        max_tokens=1500,
    )

    raw = ""
    used_path = "chat_completion(system+user)"
    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            **gen_kwargs,
        )
        raw = resp["choices"][0]["message"]["content"] or ""
    except Exception as e:
        logger.debug(f"create_chat_completion(system+user) failed: {e!r} — retrying combined")
        try:
            used_path = "chat_completion(user-combined)"
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "user", "content": f"{system_content}\n\n{user_content}"},
                ],
                **gen_kwargs,
            )
            raw = resp["choices"][0]["message"]["content"] or ""
        except Exception as e2:
            logger.debug(f"create_chat_completion(user-combined) failed: {e2!r} — falling back to ChatML")
            # Hand-built ChatML fallback (matches legacy prompt style in this file)
            used_path = "chatml_fallback"
            prompt = (
                f"<|im_start|>system\n{system_content}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            resp = llm.create_completion(
                prompt=prompt,
                stop=["<|im_end|>"],
                **gen_kwargs,
            )
            raw = resp["choices"][0]["text"] or ""

    logger.debug(f"Batched translate path={used_path}, raw_len={len(raw)}")

    # Clean known artifacts (Assistant: prefix, special tokens) then parse.
    cleaned = LocalTranslationService._clean_translation_output(raw)
    cleaned = strip_thinking_block(cleaned)

    parsed = parse_tagged_blocks(cleaned, n)
    if parsed is None:
        parsed = split_legacy_lines(cleaned, n)

    # Post-process each block and guarantee length.
    out: List[str] = [strip_wrapping_quotes(block.strip()) for block in parsed]
    if len(out) != n:
        # Defensive — should not happen since helpers pad/truncate.
        if len(out) < n:
            out = out + [""] * (n - len(out))
        else:
            out = out[:n]
    return out


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

        self._next_instance = 0  # Round-robin counter for translate_single()

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

        # Inference — lower temperature for more consistent translation output
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=settings.translation_max_tokens,
            temperature=0.3,
            top_k=20,
            top_p=0.9,
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
        Translate a single text using round-robin instance selection.

        For backward compatibility with code expecting single translate calls.
        Round-robins across instances to avoid bottleneck on instance 0.

        Args:
            text: Text to translate
            target_language: Target language

        Returns:
            Translated text
        """
        # Round-robin across instances to avoid bottleneck on instance 0
        instance_id = self._next_instance % self.num_instances
        self._next_instance += 1

        async with self.semaphores[instance_id]:
            return await asyncio.to_thread(
                self._translate_sync, self.instances[instance_id], text, target_language, instance_id, 0
            )

    async def translate_batched(
        self,
        texts: List[str],
        target_language: str = "English"
    ) -> List[str]:
        """
        Page-level batched translation using koharu's tagged-block protocol.

        Unlike ``translate_parallel`` which dispatches N calls across the
        instance pool, this sends ALL ``texts`` in a single call to ONE
        instance so that the model sees every bubble on the page at once. This
        gives intra-page coherence (tone / pronouns / name consistency) and
        usually reduces total latency because only one prompt-processing pass
        is needed instead of N.

        The instance is selected round-robin across the pool so repeated calls
        don't bottleneck on instance 0.

        Args:
            texts: Per-bubble source texts for a single page.
            target_language: Target language (default: English).

        Returns:
            List of translations with ``len(out) == len(texts)``. Always safe:
            on any internal exception a list of empty strings is returned.
        """
        if not texts:
            return []

        # Round-robin across instances so batched calls don't stack on 0.
        instance_id = self._next_instance % self.num_instances
        self._next_instance += 1
        llm = self.instances[instance_id]

        async with self.semaphores[instance_id]:
            logger.debug(
                f"Trans[{instance_id}] batched: n={len(texts)} target={target_language}"
            )
            return await _batched_translate_on_instance(
                llm, texts, target_language
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

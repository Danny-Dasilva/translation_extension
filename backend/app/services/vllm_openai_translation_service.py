"""vLLM (OpenAI-compatible) translation service.

Talks to a local vLLM `vllm serve` instance over the OpenAI Chat Completions
API. Designed for the v10-it Gemma 4 E4B + MTP setup launched by
``backend/scripts/eval/serve_v10it_vllm.sh`` (default port 8000,
served-model-name "v10it").

Drop-in translation service — same `translate_single` /
`translate_batched` async surface used by the e2e visualizer and routers.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import List

import httpx

from app.services.translation_text_utils import clean_translation_output

logger = logging.getLogger(__name__)


class VLLMOpenAITranslationService:
    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str = "EMPTY",
        request_timeout_s: float = 120.0,
        concurrency: int = 8,
    ):
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.model_name = model_name or os.environ.get("VLLM_MODEL", "v10it")
        self.api_key = api_key
        self.timeout = request_timeout_s
        self._client = httpx.AsyncClient(timeout=request_timeout_s)
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._healthy = False
        logger.info(f"vLLM client targeting {self.base_url} model={self.model_name}")

    async def _ensure_healthy(self) -> None:
        if self._healthy:
            return
        try:
            r = await self._client.get(f"{self.base_url}/models")
            r.raise_for_status()
            ids = [m.get("id") for m in r.json().get("data", [])]
            if self.model_name not in ids:
                logger.warning(
                    f"vLLM /v1/models returned {ids}; expected {self.model_name}"
                )
            self._healthy = True
        except Exception as e:
            raise RuntimeError(
                f"vLLM server at {self.base_url} not reachable. "
                f"Start it with: bash backend/scripts/eval/serve_v10it_vllm.sh "
                f"(orig error: {e!r})"
            )

    async def _chat(
        self,
        messages: List[dict],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        await self._ensure_healthy()
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        async with self._sem:
            r = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"] or ""

    async def translate_single(
        self, text: str, target_language: str = "English"
    ) -> str:
        if not text.strip():
            return ""
        # Mirror the prompt format used by the dedicated translation models.
        msg = [{
            "role": "user",
            "content": (
                f"Translate the following segment into {target_language}, "
                f"without additional explanation.\n\n{text}"
            ),
        }]
        try:
            raw = await self._chat(msg, max_tokens=128, temperature=0.0)
        except Exception as e:
            logger.warning(f"vLLM translate_single failed: {e!r}")
            return ""
        return clean_translation_output(raw)

    async def translate_batched(
        self, texts: List[str], target_language: str = "English"
    ) -> List[str]:
        """Fan out to per-bubble translate_single concurrently.

        vLLM with continuous batching handles concurrent requests well, so
        N parallel single-bubble calls is typically faster than one giant
        tagged-block prompt — and it avoids the small-model regression
        observed with the few-shot tagged prompt on dedicated translation
        models.
        """
        if not texts:
            return []
        return await asyncio.gather(
            *(self.translate_single(t, target_language) for t in texts)
        )

    async def warmup(self) -> dict:
        t0 = time.perf_counter()
        try:
            await self.translate_single("テスト", "English")
        except Exception as e:
            logger.warning(f"vLLM warmup failed: {e!r}")
        return {"warmup_ms": (time.perf_counter() - t0) * 1000}

    @property
    def num_instances(self) -> int:
        return 1

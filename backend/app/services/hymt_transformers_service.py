"""Transformers-backed Hy-MT1.5-1.8B-2bit translation service.

Runs the Tencent Hy-MT1.5 2-bit safetensors model through HuggingFace
transformers on CUDA. The "2-bit" weights are stored at BF16 in the
safetensors file (each weight is one of {-1.5, -0.5, 0.5, 1.5}); the
custom 2-bit kernel is ARM SME2/Neon-only, so on x86_64+CUDA we just
load the BF16 tensors and run normal matmul. Output quality matches the
2-bit kernel since the weight values themselves are identical.

This service exposes the same async `translate_single` / `translate_batched`
surface as the other translation backends so it can drop in wherever those
are used.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings
from app.services.translation_text_utils import clean_translation_output

logger = logging.getLogger(__name__)


def _resolve_model_dir() -> Path:
    p = Path(settings.hymt_transformers_model_dir)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    return p


class HyMTTransformersService:
    """Transformers-based translation service compatible with the
    translation-service interface used by the visualizer and routers.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str | None = None,
        min_free_gpu_gb: float = 4.5,
    ):
        path = Path(model_dir) if model_dir else _resolve_model_dir()
        if not path.exists():
            raise FileNotFoundError(f"Hy-MT model dir not found: {path}")

        if device is None:
            device = "cpu"
            if torch.cuda.is_available():
                try:
                    free_b, _ = torch.cuda.mem_get_info(0)
                    free_gb = free_b / 1e9
                    if free_gb >= min_free_gpu_gb:
                        device = "cuda"
                    else:
                        logger.warning(
                            f"Only {free_gb:.2f} GB free on cuda:0; need ≥{min_free_gpu_gb} GB. "
                            f"Falling back to CPU."
                        )
                except Exception as e:
                    logger.warning(f"CUDA probe failed ({e!r}); falling back to CPU.")
        logger.info(f"Loading Hy-MT1.5-2bit (transformers) from {path} on {device}")
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path),
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        self.device = device
        self.dtype = dtype
        # End-of-assistant marker for hy chat template
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(
            "<｜hy_place▁holder▁no▁2｜>"
        )
        if self.eos_token_id == self.tokenizer.unk_token_id or self.eos_token_id is None:
            self.eos_token_id = self.tokenizer.eos_token_id
        logger.info(
            f"Hy-MT loaded in {(time.perf_counter()-t0)*1000:.0f}ms "
            f"(eos={self.eos_token_id})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_inputs(self, messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # Hy-MT doesn't accept token_type_ids; drop it
        enc.pop("token_type_ids", None)
        return enc

    def _generate(self, inputs, max_new_tokens: int, **gen_kwargs) -> str:
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.eos_token_id,
                **gen_kwargs,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Public API (mirrors the other translation services)
    # ------------------------------------------------------------------
    async def translate_single(
        self, text: str, target_language: str = "English"
    ) -> str:
        if not text.strip():
            return ""

        def _run() -> str:
            messages = [{
                "role": "user",
                "content": (
                    f"Translate the following segment into {target_language}, "
                    f"without additional explanation.\n\n{text}"
                ),
            }]
            inputs = self._build_inputs(messages)
            raw = self._generate(
                inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.05,
            )
            return clean_translation_output(raw)

        return await asyncio.to_thread(_run)

    async def translate_batched(
        self, texts: List[str], target_language: str = "English"
    ) -> List[str]:
        """Translate a list of texts.

        Hy-MT1.5-1.8B is a *small dedicated translation model* — it doesn't
        follow elaborate few-shot instructions well, so the [N]-tagged
        batched prompt used by the GGUF service produces garbage (it copies
        example outputs verbatim). Instead, fan out to per-text translation
        sequentially. The model is fast enough that this is fine for E2E.
        """
        n = len(texts)
        if n == 0:
            return []
        results: List[str] = []
        for i, t in enumerate(texts):
            try:
                results.append(await self.translate_single(t, target_language))
            except Exception as e:
                logger.warning(f"Hy-MT translate_single[{i}] failed: {e!r}")
                results.append("")
        return results

    async def warmup(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        await self.translate_single("テスト", "English")
        return {"warmup_ms": (time.perf_counter() - t0) * 1000}

    @property
    def num_instances(self) -> int:
        return 1

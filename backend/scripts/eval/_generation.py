"""Shared generation helper for L1/L2/L4 runners.

Supports:

- A merged HF-format checkpoint directory, OR
- A merged base directory + a LoRA adapter directory (auto-detected via
  presence of ``adapter_config.json`` in the checkpoint path).

Uses vLLM if importable, otherwise falls back to ``transformers.generate``.
All heavy imports are lazy - importing this module alone does nothing.

Prompt format (exact, per VNTL leaderboard protocol):

    Translate the following Japanese to English. Output only the translation.

    Japanese: {jp}
    English:

The Qwen3 chat template is applied with ``enable_thinking=False`` so the
model emits only the final translation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

PROMPT_TEMPLATE = (
    "Translate the following Japanese to English. Output only the translation.\n\n"
    "Japanese: {jp}\n"
    "English:"
)

DEFAULT_SAMPLING = {
    "temperature": 0.2,
    "top_p": 0.9,
    "min_p": 0.1,
    "max_new_tokens": 256,
}


def build_prompts(jp_list: list[str], tokenizer: Any | None = None) -> list[str]:
    """Build chat-templated prompts.

    If ``tokenizer`` has ``apply_chat_template`` and the template supports
    ``enable_thinking``, we use it with thinking disabled; otherwise we fall
    back to the raw PROMPT_TEMPLATE.
    """
    prompts: list[str] = []
    for jp in jp_list:
        user_msg = PROMPT_TEMPLATE.format(jp=jp)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompts.append(rendered)
                continue
            except TypeError:
                # Older tokenizer without enable_thinking arg.
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(rendered)
                continue
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("apply_chat_template failed ({}); using raw prompt", e)
        prompts.append(user_msg)
    return prompts


def _resolve_checkpoint(checkpoint: Path) -> tuple[Path, Path | None]:
    """Return (base_path, adapter_path_or_None).

    If the checkpoint dir contains ``adapter_config.json`` we treat it as a
    LoRA adapter and require a sibling ``base/`` directory (or a symlink),
    or a ``base_model_name_or_path`` entry in the adapter config.
    """
    cfg = checkpoint / "adapter_config.json"
    if not cfg.exists():
        return checkpoint, None

    import json as _json

    with cfg.open("r", encoding="utf-8") as fh:
        adapter_cfg = _json.load(fh)
    base_name = adapter_cfg.get("base_model_name_or_path")
    if base_name is None:
        raise RuntimeError(
            f"{checkpoint} is a LoRA adapter but has no base_model_name_or_path set."
        )
    return Path(base_name), checkpoint


def load_hf_model(checkpoint: Path) -> tuple[Any, Any]:
    """Load a HF model + tokenizer, merging a LoRA adapter if present."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_path, adapter_path = _resolve_checkpoint(checkpoint)

    logger.info("Loading base model from {}", base_path)
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path is not None:
        logger.info("Merging LoRA adapter from {}", adapter_path)
        from peft import PeftModel  # lazy

        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model, tokenizer


def _try_vllm(checkpoint: Path) -> Any | None:
    """Return a vLLM LLM instance if importable + supports this checkpoint."""
    try:
        from vllm import LLM  # noqa: F401
    except Exception:
        return None
    base_path, adapter_path = _resolve_checkpoint(checkpoint)
    if adapter_path is not None:
        # vLLM LoRA support needs separate flags; fall back to HF for safety.
        logger.info("vLLM available but checkpoint is LoRA - using HF fallback.")
        return None
    from vllm import LLM

    logger.info("Loading vLLM engine from {}", base_path)
    return LLM(model=str(base_path), dtype="bfloat16", trust_remote_code=True)


def generate(
    checkpoint: Path,
    jp_list: list[str],
    *,
    batch_size: int = 32,
    sampling: dict[str, Any] | None = None,
    prefer_vllm: bool = True,
) -> list[str]:
    """Generate translations for ``jp_list``.  Returns English strings."""
    sp = dict(DEFAULT_SAMPLING)
    if sampling:
        sp.update(sampling)

    if prefer_vllm:
        engine = _try_vllm(checkpoint)
        if engine is not None:
            from vllm import SamplingParams  # lazy

            # We still need a tokenizer for chat templating.
            from transformers import AutoTokenizer

            base_path, _ = _resolve_checkpoint(checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)

            prompts = build_prompts(jp_list, tokenizer)
            params = SamplingParams(
                temperature=sp["temperature"],
                top_p=sp["top_p"],
                min_p=sp.get("min_p", 0.0),
                max_tokens=sp["max_new_tokens"],
            )
            outputs = engine.generate(prompts, params)
            return [o.outputs[0].text.strip() for o in outputs]

    # HF fallback
    import torch  # noqa: F401

    model, tokenizer = load_hf_model(checkpoint)
    prompts = build_prompts(jp_list, tokenizer)

    results: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(
            model.device
        )
        import torch as _torch

        with _torch.inference_mode():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=sp["temperature"],
                top_p=sp["top_p"],
                min_p=sp.get("min_p", 0.0),
                max_new_tokens=sp["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        for j in range(out.shape[0]):
            gen = out[j, enc["input_ids"].shape[1] :]
            text = tokenizer.decode(gen, skip_special_tokens=True).strip()
            results.append(text)

    return results

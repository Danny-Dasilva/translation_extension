"""Merge a Gemma 4 LoRA adapter into the base model cleanly, on CPU.

Why this exists
---------------
Two failure modes have to be handled:

1. **Unsloth path (broken):** ``save_pretrained_merged`` corrupts Gemma 4 LoRA
   merges. Output of an Unsloth-merged model is mangled (TeX, Cyrillic, Korean).

2. **Plain PEFT path (also broken for Gemma 4):** ``PeftModel.from_pretrained``
   walks ``target_modules`` from ``adapter_config.json`` and looks for them in
   the base model. In Gemma 4, ``vision_tower`` and ``audio_tower`` projections
   are wrapped in ``Gemma4ClippableLinear`` (a plain ``nn.Module``, not
   ``nn.Linear``), so PEFT raises::

       ValueError: Target module Gemma4ClippableLinear(...) is not supported.

The v9c adapter actually contains LoRA pairs for *all three* sub-models:

  * ``language_model.layers.N.{self_attn|mlp}.<proj>``           -> bare nn.Linear
  * ``vision_tower.encoder.layers.N.{self_attn|mlp}.<proj>.linear`` -> nn.Linear inside ClippableLinear
  * ``audio_tower.layers.N.self_attn.<proj>.linear``                -> nn.Linear inside ClippableLinear

The adapter's safetensors keys already encode the *correct* path: the trainer
wrote ``...q_proj.linear.lora_A.weight`` for vision/audio entries and
``...q_proj.lora_A.weight`` for the language tower. So we side-step PEFT's
module-class check entirely and merge LoRA weights directly into the matching
``Linear.weight`` tensor by walking the safetensors keys.

Approach
--------

For each adapter pair ``base_model.model.<path>.lora_A.weight`` /
``base_model.model.<path>.lora_B.weight``:

    target = model.<path>.weight
    target += (B @ A) * scaling

where ``scaling = lora_alpha / r`` (constant across the v9c adapter).

After merge, save with plain HF ``save_pretrained`` (NOT Unsloth's variant) and
copy the tokenizer/processor side-files. Save runs on CPU to avoid VRAM blow-up
on the single-GPU rig (RTX 5090 / 32 GB).

Usage
-----

    uv run python backend/scripts/eval/merge_gemma4_lora_clean.py \\
        --adapter backend/training/runs/manga-bubbles/gemma4_e4b_v9c/final \\
        --output  backend/training/runs/manga-bubbles/gemma4_e4b_v9c/merged_clean

The base model id is read from the adapter's ``adapter_config.json``
(``base_model_name_or_path``); override with ``--base`` if you trained against
a different snapshot.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open


PROCESSOR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "processor_config.json",
    "preprocessor_config.json",
    "chat_template.json",
    "added_tokens.json",
    "spiece.model",
)

LORA_KEY_PREFIX = "base_model.model."
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #

def _load_adapter_config(adapter_dir: Path) -> dict:
    cfg_p = adapter_dir / "adapter_config.json"
    if not cfg_p.exists():
        raise FileNotFoundError(f"adapter_config.json missing at {cfg_p}")
    with open(cfg_p) as f:
        return json.load(f)


def _resolve_base_name(adapter_cfg: dict, override: str | None) -> str:
    if override:
        return override
    base = adapter_cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError("adapter_config.json missing 'base_model_name_or_path'")
    return base


def _resolve_scaling(adapter_cfg: dict, module_path: str) -> float:
    """Return per-module LoRA scaling = alpha / rank.

    v9c uses uniform alpha=32, r=16, but the adapter format supports per-pattern
    overrides via ``rank_pattern`` / ``alpha_pattern`` (regex on module name).
    Honor those if present.
    """
    r = int(adapter_cfg.get("r", 16))
    alpha = float(adapter_cfg.get("lora_alpha", r))
    rank_pattern = adapter_cfg.get("rank_pattern") or {}
    alpha_pattern = adapter_cfg.get("alpha_pattern") or {}
    # PEFT applies the first matching pattern as a substring/key match.
    # v9c has these dicts empty, so the loop is a no-op there.
    for k, v in rank_pattern.items():
        if k in module_path:
            r = int(v); break
    for k, v in alpha_pattern.items():
        if k in module_path:
            alpha = float(v); break
    if r == 0:
        raise ZeroDivisionError(f"r=0 for module {module_path}")
    if adapter_cfg.get("use_rslora"):
        # rsLoRA scales by alpha / sqrt(r)
        import math
        return alpha / math.sqrt(r)
    return alpha / r


# --------------------------------------------------------------------------- #
# manual merge
# --------------------------------------------------------------------------- #

def _walk_to_weight(model, dotted_path: str) -> torch.nn.Parameter:
    """Resolve ``module.<path>.weight`` to the actual parameter tensor.

    ``dotted_path`` is the part between ``base_model.model.`` and ``.lora_A.weight``,
    e.g. ``model.language_model.layers.0.self_attn.q_proj``.
    Returns the ``.weight`` Parameter of the resolved leaf module.
    """
    parts = dotted_path.split(".")
    obj = model
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    if not hasattr(obj, "weight"):
        raise AttributeError(f"resolved module at {dotted_path!r} has no .weight ({type(obj).__name__})")
    return obj.weight


def _collect_lora_pairs(adapter_dir: Path) -> dict[str, dict[str, torch.Tensor]]:
    """Return ``{module_path: {'A': tensor, 'B': tensor}}``.

    ``module_path`` is the dotted path stripped of ``base_model.model.`` prefix
    and the ``.lora_A.weight`` / ``.lora_B.weight`` suffix.
    """
    st = adapter_dir / "adapter_model.safetensors"
    if not st.exists():
        raise FileNotFoundError(f"adapter weights missing: {st}")
    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    with safe_open(str(st), framework="pt") as f:
        for k in f.keys():
            if not k.startswith(LORA_KEY_PREFIX):
                logger.warning("skipping non-LoRA key: {}", k)
                continue
            stripped = k[len(LORA_KEY_PREFIX):]
            if stripped.endswith(LORA_A_SUFFIX):
                mod, side = stripped[: -len(LORA_A_SUFFIX)], "A"
            elif stripped.endswith(LORA_B_SUFFIX):
                mod, side = stripped[: -len(LORA_B_SUFFIX)], "B"
            else:
                # E.g. lora_embedding_A/B -- v9c doesn't have these. Skip + log.
                logger.warning("skipping unrecognized LoRA key: {}", k)
                continue
            pairs[mod][side] = f.get_tensor(k)
    return pairs


def manual_merge(model: torch.nn.Module, adapter_dir: Path, adapter_cfg: dict) -> dict:
    """Merge LoRA pairs from ``adapter_dir`` into ``model`` in place.

    Returns counts: ``{'merged': N, 'skipped': M, 'sections': {sect: N, ...}}``.
    """
    pairs = _collect_lora_pairs(adapter_dir)
    logger.info("found {} LoRA pairs in adapter", len(pairs))

    merged = 0
    skipped = 0
    by_section: dict[str, int] = defaultdict(int)

    for module_path, ab in sorted(pairs.items()):
        if "A" not in ab or "B" not in ab:
            logger.warning("incomplete pair at {} -- skipping", module_path)
            skipped += 1
            continue
        A = ab["A"]  # [r, in_features]
        B = ab["B"]  # [out_features, r]
        try:
            W = _walk_to_weight(model, module_path)
        except AttributeError as e:
            logger.warning("could not resolve {!r}: {} -- skipping", module_path, e)
            skipped += 1
            continue
        scaling = _resolve_scaling(adapter_cfg, module_path)
        # Promote to fp32 for the matmul to keep precision, then cast back.
        delta = (B.to(torch.float32) @ A.to(torch.float32)) * scaling
        with torch.no_grad():
            W.add_(delta.to(W.dtype))
        merged += 1
        # Section bookkeeping: language_model / vision_tower / audio_tower
        for tag in ("language_model", "vision_tower", "audio_tower"):
            if tag in module_path:
                by_section[tag] += 1
                break
        else:
            by_section["other"] += 1

    logger.info("manual merge done: merged={}, skipped={}, by_section={}",
                merged, skipped, dict(by_section))
    return {"merged": merged, "skipped": skipped, "sections": dict(by_section)}


# --------------------------------------------------------------------------- #
# tokenizer + processor copy
# --------------------------------------------------------------------------- #

def _copy_processor_files(src: Path, dst: Path) -> list[str]:
    copied: list[str] = []
    for name in PROCESSOR_FILES:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)
            copied.append(name)
    return copied


# --------------------------------------------------------------------------- #
# main flow
# --------------------------------------------------------------------------- #

def merge(adapter_dir: Path, output_dir: Path, base_name: str | None) -> dict:
    from transformers import AutoProcessor, AutoTokenizer, Gemma4ForConditionalGeneration

    adapter_cfg = _load_adapter_config(adapter_dir)
    base = _resolve_base_name(adapter_cfg, base_name)
    logger.info("base model:   {}", base)
    logger.info("adapter:      {}", adapter_dir)
    logger.info("output:       {}", output_dir)
    logger.info("LoRA r/alpha: r={}, alpha={}, rslora={}",
                adapter_cfg.get("r"), adapter_cfg.get("lora_alpha"),
                adapter_cfg.get("use_rslora"))
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("loading base on CPU (bfloat16)...")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        base,
        device_map="cpu",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    t_load_base = time.time() - t0
    logger.info("base loaded in {:.1f}s", t_load_base)

    t0 = time.time()
    merge_stats = manual_merge(model, adapter_dir, adapter_cfg)
    t_merge = time.time() - t0
    logger.info("manual merge in {:.1f}s", t_merge)

    t0 = time.time()
    logger.info("save_pretrained -> {}", output_dir)
    model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size="5GB",
    )
    t_save = time.time() - t0
    logger.info("saved in {:.1f}s", t_save)

    # Tokenizer/processor: prefer adapter's bundled files, then re-emit via
    # AutoTokenizer to make sure the result is complete.
    copied = _copy_processor_files(adapter_dir, output_dir)
    logger.info("copied {} side-files from adapter dir: {}", len(copied), copied)

    try:
        tok = AutoTokenizer.from_pretrained(str(adapter_dir))
        tok.save_pretrained(str(output_dir))
        logger.info("tokenizer re-saved via AutoTokenizer (from adapter dir)")
    except Exception as e:  # noqa: BLE001
        logger.warning("AutoTokenizer from adapter failed ({}); falling back to base", e)
        tok = AutoTokenizer.from_pretrained(base)
        tok.save_pretrained(str(output_dir))

    # Multimodal processor: optional. Adapter dir is text-only; pull from base.
    try:
        proc = AutoProcessor.from_pretrained(base)
        proc.save_pretrained(str(output_dir))
        logger.info("processor saved from base ({})", base)
    except Exception as e:  # noqa: BLE001
        logger.warning("AutoProcessor.save_pretrained skipped: {}", e)

    return {
        "load_base_s": t_load_base,
        "merge_s": t_merge,
        "save_s": t_save,
        "merge_stats": merge_stats,
    }


def _dir_size_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--adapter", required=True, type=Path,
                    help="Path to LoRA adapter dir (must contain adapter_config.json)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Where to write the merged model")
    ap.add_argument("--base", default=None,
                    help="Override base model id (default: read from adapter_config.json)")
    args = ap.parse_args()

    adapter_dir = args.adapter.resolve()
    output_dir = args.output.resolve()

    if not adapter_dir.is_dir():
        logger.error("adapter dir not found: {}", adapter_dir)
        return 2

    t_total = time.time()
    timings = merge(adapter_dir, output_dir, args.base)
    wall = time.time() - t_total

    size_b = _dir_size_bytes(output_dir)
    summary = {
        "adapter": str(adapter_dir),
        "output": str(output_dir),
        "wall_s": round(wall, 1),
        "load_base_s": round(timings["load_base_s"], 1),
        "merge_s": round(timings["merge_s"], 1),
        "save_s": round(timings["save_s"], 1),
        "merge_stats": timings["merge_stats"],
        "output_size_bytes": size_b,
        "output_size_human": f"{size_b/1e9:.2f} GB",
    }
    print(json.dumps(summary, indent=2))
    logger.info("DONE in {:.1f}s -> {} ({:.2f} GB)", wall, output_dir, size_b/1e9)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Patch the v10-it merged checkpoint to include the 18 k_norm tensors that
HF transformers 5.8.0 drops for KV-shared Gemma4 layers (24-41).

vLLM's Gemma4 model class instantiates a k_norm RMSNorm for ALL 42 layers
(including the KV-shared ones), but HF's `Gemma4ForConditionalGeneration` only
materializes 24 of them (layers 0-23, the non-KV-shared subset). This causes
`merge_gemma4_lora_clean.py` (which loads via HF and calls `save_pretrained`)
to write a merged checkpoint missing layers 24-41 of `k_norm.weight`. vLLM
then aborts at load time with::

    ValueError: Following weights were not initialized from checkpoint:
        {'language_model.model.layers.24.self_attn.k_norm.weight', ...}

Fix: copy the 18 missing tensors verbatim from the base model safetensors and
write a new shard alongside the existing 4 shards, then update
`model.safetensors.index.json` to point at it. The LoRA adapter never targeted
k_norm (only q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj), so the
base values are correct.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file


BASE_KEY_FMT = "model.language_model.layers.{layer}.self_attn.k_norm.weight"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--merged",
        required=True,
        type=Path,
        help="Path to merged checkpoint dir to patch in-place",
    )
    ap.add_argument(
        "--base-safetensors",
        default=Path(
            "/home/danny/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it/"
            "snapshots/0d5a7f9ba73eda1616e58344f7025fae44914675/model.safetensors"
        ),
        type=Path,
    )
    args = ap.parse_args()

    idx_p = args.merged / "model.safetensors.index.json"
    if not idx_p.exists():
        logger.error("no index at {} — is this a sharded checkpoint?", idx_p)
        return 2

    idx = json.loads(idx_p.read_text())
    weight_map: dict[str, str] = idx["weight_map"]
    metadata: dict = idx.get("metadata", {})

    # Identify missing layers
    have = {k for k in weight_map if "language_model" in k and "k_norm" in k}
    missing = []
    for layer in range(42):
        k = BASE_KEY_FMT.format(layer=layer)
        if k not in weight_map:
            missing.append((layer, k))
    logger.info("merged has {} lm k_norm keys, missing {} (KV-shared layers)", len(have), len(missing))
    if not missing:
        logger.info("no k_norm patch needed")
        return 0

    # Pull missing tensors from base safetensors
    logger.info("opening base safetensors: {}", args.base_safetensors)
    patched: dict[str, torch.Tensor] = {}
    with safe_open(args.base_safetensors, framework="pt") as f:
        base_keys = set(f.keys())
        for layer, key in missing:
            if key not in base_keys:
                logger.error("missing key in base too: {}", key)
                return 3
            t = f.get_tensor(key)
            patched[key] = t.to(torch.bfloat16) if t.dtype != torch.bfloat16 else t
            logger.info("  copied {} dtype={} shape={}", key, t.dtype, tuple(t.shape))

    # Write a new shard (00005) — naming follows existing shard scheme
    existing_shards = sorted({Path(v).name for v in weight_map.values()})
    last = existing_shards[-1]  # e.g. model-00004-of-00004.safetensors
    n_total = int(last.split("-of-")[1].split(".")[0])
    new_total = n_total + 1
    new_shard = f"model-{new_total:05d}-of-{new_total:05d}.safetensors"

    # We also need to rename existing shards to reflect the new total in their suffix
    # Actually: HF tolerates index.json with shard names that don't match the "X-of-Y"
    # scheme as long as weight_map points to existing files. To minimize risk we'll
    # use a non-conflicting filename like "model-knorm-extra.safetensors".
    extra_name = "model-knorm-extra.safetensors"
    extra_path = args.merged / extra_name
    logger.info("writing {} ({} tensors)", extra_path, len(patched))
    save_file(patched, str(extra_path))

    # Update weight_map and total_size in index
    total_size = metadata.get("total_size", 0)
    add_size = sum(t.numel() * t.element_size() for t in patched.values())
    metadata["total_size"] = int(total_size) + int(add_size)
    for key in patched:
        weight_map[key] = extra_name

    new_idx = {"metadata": metadata, "weight_map": weight_map}
    idx_p.write_text(json.dumps(new_idx, indent=2))
    logger.info("updated index.json (+{} bytes)", add_size)
    logger.info("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fold the 18 KV-shared k_norm tensors (layers 24-41) into a single-file
quantized W4A16 checkpoint (model.safetensors) so vLLM's Gemma4 class can load.

The RTN/llmcompressor save writes one model.safetensors (no shard index). HF /
save_pretrained omitted layers 24-41 of self_attn.k_norm.weight (KV-shared).
k_norm is an RMSNorm (never quantized), so the bf16 values from merged_fixed's
model-knorm-extra.safetensors are valid verbatim. We read the quant tensors,
add the 18 k_norm tensors, and rewrite model.safetensors in place.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant-dir", required=True, type=Path)
    ap.add_argument("--knorm-extra", required=True, type=Path)
    args = ap.parse_args()

    model_p = args.quant_dir / "model.safetensors"
    if not model_p.exists():
        print(f"ERROR: {model_p} not found")
        return 2

    tensors: dict = {}
    with safe_open(str(model_p), framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    print(f"loaded {len(tensors)} tensors from quant checkpoint")

    added = 0
    with safe_open(str(args.knorm_extra), framework="pt") as f:
        for k in f.keys():
            if k not in tensors:
                tensors[k] = f.get_tensor(k)
                added += 1
    print(f"adding {added} k_norm tensors")

    if added == 0:
        print("nothing to add — already complete")
        return 0

    save_file(tensors, str(model_p), metadata=meta)
    print(f"rewrote {model_p} with {len(tensors)} tensors")

    # sanity: confirm all 42 k_norm layers now present
    with safe_open(str(model_p), framework="pt") as f:
        kn = [k for k in f.keys() if "k_norm" in k and "language_model" in k]
    print(f"k_norm language_model keys now: {len(kn)} (expect 42)")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

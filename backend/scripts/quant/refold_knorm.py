"""Re-fold the 18 KV-shared k_norm tensors (layers 24-41) into a freshly
quantized W4A16 checkpoint so vLLM's Gemma4 model class can load it.

Background: HF transformers materializes k_norm only for the 24 non-KV-shared
layers; save_pretrained (and llmcompressor's compressed save) therefore omit
layers 24-41 of `...self_attn.k_norm.weight`. vLLM instantiates k_norm for ALL
42 layers and aborts at load with "weights were not initialized".

The merged_fixed checkpoint already carries these 18 tensors in
`model-knorm-extra.safetensors` under the HF key scheme
`model.language_model.layers.{L}.self_attn.k_norm.weight`. k_norm is NEVER
quantized (it is an RMSNorm, not a Linear), so the bf16 values are valid as-is
for the quantized model. We copy that extra shard into the quant dir and add its
keys to the quant index.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant-dir", required=True, type=Path,
                    help="W4A16 output dir to patch in place")
    ap.add_argument("--knorm-extra", required=True, type=Path,
                    help="model-knorm-extra.safetensors with the 18 KV-shared k_norm tensors")
    args = ap.parse_args()

    idx_p = args.quant_dir / "model.safetensors.index.json"
    if not idx_p.exists():
        print(f"ERROR: no index at {idx_p}")
        return 2
    idx = json.loads(idx_p.read_text())
    weight_map: dict[str, str] = idx["weight_map"]
    metadata: dict = idx.get("metadata", {})

    # Which k_norm keys does the quant checkpoint already have?
    have_knorm = sorted(k for k in weight_map if "k_norm" in k)
    print(f"quant checkpoint has {len(have_knorm)} k_norm keys")

    extra_name = "model-knorm-extra.safetensors"
    dst = args.quant_dir / extra_name

    # Copy the extra shard in and register only the MISSING keys.
    added = 0
    add_bytes = 0
    with safe_open(str(args.knorm_extra), framework="pt") as f:
        extra_keys = list(f.keys())
        for k in extra_keys:
            if k not in weight_map:
                weight_map[k] = extra_name
                t = f.get_tensor(k)
                add_bytes += t.numel() * t.element_size()
                added += 1

    if added == 0:
        print("no missing k_norm keys — checkpoint already complete, nothing to do")
        return 0

    shutil.copy2(args.knorm_extra, dst)
    metadata["total_size"] = int(metadata.get("total_size", 0)) + add_bytes
    idx_p.write_text(json.dumps({"metadata": metadata, "weight_map": weight_map}, indent=2))
    print(f"folded {added} k_norm tensors (+{add_bytes} bytes) -> {dst}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

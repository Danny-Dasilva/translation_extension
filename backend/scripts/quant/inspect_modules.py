"""Print the model's module tree so we can craft exact W4A16 ignore patterns.

We must quantize ONLY transformer Linear layers and keep ALL embeddings
(per-layer + tied 262k vocab) and the lm_head high-precision. This script lists:
  * every Linear module path (candidates to quantize)
  * every Embedding / per-layer-input / norm module (MUST be ignored)
so we can verify our ignore regexes match the real names.
"""
from __future__ import annotations

import argparse
import collections

import torch
from transformers import AutoModelForCausalLM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    print("loading (meta-ish, cpu, bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    linear_samples: list[str] = []
    embed_like: list[str] = []
    other_param_modules: collections.Counter = collections.Counter()

    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if isinstance(mod, torch.nn.Linear) or cls.endswith("Linear"):
            linear_samples.append(f"{name}  [{cls}]")
        elif isinstance(mod, torch.nn.Embedding) or "embed" in name.lower() or "Embedding" in cls:
            embed_like.append(f"{name}  [{cls}]")
        else:
            # count leaf module class names that hold parameters
            if any(True for _ in mod.parameters(recurse=False)):
                other_param_modules[cls] += 1

    print("\n==== EMBEDDING-LIKE MODULES (MUST IGNORE) ====")
    for s in embed_like:
        print("  ", s)

    print("\n==== sample LINEAR MODULES (to quantize), first 40 distinct suffixes ====")
    seen = set()
    for s in linear_samples:
        # collapse layer index to show distinct suffix shapes
        suffix = s.split("layers.")[-1]
        suffix = suffix.split(".", 1)[-1] if suffix[0].isdigit() else s
        if suffix not in seen:
            seen.add(suffix)
            print("  ", s)
        if len(seen) >= 40:
            break
    print(f"\n  total Linear modules: {len(linear_samples)}")

    print("\n==== OTHER param-holding leaf module classes (norms etc.) ====")
    for cls, c in other_param_modules.most_common():
        print(f"   {cls}: {c}")

    # explicit check: is lm_head a Linear? tied?
    print("\n==== lm_head / tie check ====")
    print("  tie_word_embeddings:", getattr(model.config, "tie_word_embeddings", None))
    for name, mod in model.named_modules():
        if name.endswith("lm_head"):
            print(f"  {name}  [{type(mod).__name__}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

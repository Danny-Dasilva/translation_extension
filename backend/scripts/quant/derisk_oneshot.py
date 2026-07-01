"""DE-RISK GATE: run llmcompressor oneshot W4A16 on just N (default 6) calibration
samples to prove the gemma4 PLE arch can be traced/quantized BEFORE the full run.

If this fails (arch unsupported, tracing error, embedding-quant blowup), STOP and
report the exact error. If it succeeds, the full run reuses the same recipe.

Embeddings + lm_head are IGNORED so only transformer Linear layers go to 4-bit.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch


def load_calib_texts(path: Path, n: int) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
            if len(texts) >= n:
                break
    return texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True, type=Path)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--ignore", nargs="*", default=None,
                    help="override ignore patterns (default: lm_head + all embed/per-layer)")
    ap.add_argument("--pipeline", default="basic",
                    help="llmcompressor pipeline. MUST be 'basic' for gemma4: the "
                         "'sequential' pipeline partitions the graph and breaks "
                         "Gemma4's cross-layer KV sharing (shared_kv_states dict is "
                         "empty in later partitions -> KeyError 'sliding_attention').")
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    # Default ignore: lm_head + every embedding / per-layer-input module.
    # Patterns are intentionally broad; inspect_modules.py output validates them.
    # Verified against inspect_modules.py on this exact checkpoint:
    #   * tied lm_head + all embeddings (embed_tokens, embed_tokens_per_layer,
    #     embed_vision, embed_audio) MUST stay high-precision.
    #   * vision_tower / audio_tower are unused for text translation and wrap
    #     inner Linears in Gemma4ClippableLinear (GPTQ-hostile) -> ignore.
    # What remains quantized = the 343 language_model transformer Linears
    #   (q/k/v/o_proj, gate/up/down_proj, per_layer_input_gate,
    #    per_layer_projection, per_layer_model_projection).
    ignore = args.ignore if args.ignore is not None else [
        "lm_head",
        "re:.*embed_tokens.*",        # embed_tokens + embed_tokens_per_layer
        "re:.*embed_vision.*",
        "re:.*embed_audio.*",
        "re:model\\.vision_tower\\..*",
        "re:model\\.audio_tower\\..*",
        "re:.*multi_modal_projector.*",
        # PLE per-layer-input projections: these feed the per-layer embedding
        # path (which we keep high-precision). GPTQ's sequential hooks never
        # register activation stats for per_layer_model_projection (it bypasses
        # the main residual stream), so quantizing it aborts with
        # "No statistics available". They are tiny (-> 256 dim) so ignoring them
        # costs ~no VRAM. Matches task guidance: ignore re:.*per_layer.*input.*
        "re:.*per_layer_model_projection.*",
        "re:.*per_layer_projection.*",
        "re:.*per_layer_input_gate.*",
    ]

    print(f"[derisk] loading model {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    texts = load_calib_texts(args.calib, args.n)
    print(f"[derisk] {len(texts)} calib samples, group_size={args.group_size}")

    def tokenize(sample):
        return tok(
            sample["text"],
            truncation=True,
            max_length=args.max_seq_len,
            add_special_tokens=False,  # chat template already added them
        )

    ds = Dataset.from_dict({"text": texts}).map(tokenize, remove_columns=["text"])

    # W4A16 preset = 4-bit, group_size 128, symmetric group quant.
    # For a non-128 group_size, build an explicit config_group instead.
    if args.group_size == 128:
        recipe = GPTQModifier(targets=["Linear"], scheme="W4A16", ignore=ignore)
    else:
        from compressed_tensors.quantization import (
            QuantizationArgs, QuantizationScheme, QuantizationStrategy,
        )
        scheme = QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4, type="int", symmetric=True,
                strategy=QuantizationStrategy.GROUP, group_size=args.group_size,
            ),
        )
        recipe = GPTQModifier(
            config_groups={"group_0": scheme}, ignore=ignore,
        )

    print("[derisk] running oneshot (this is the make-or-break trace)...")
    # Pass the tokenizer explicitly as the processor: the model is multimodal
    # (gemma4) so llmcompressor's AutoProcessor path tries to build the vision
    # image processor (needs torchvision). For text-only calibration we hand it
    # the plain tokenizer and skip that entirely.
    oneshot(
        model=model,
        processor=tok,
        dataset=ds,
        recipe=recipe,
        pipeline=args.pipeline,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=len(texts),
        output_dir=args.out,
    )
    print(f"[derisk] SUCCESS — wrote {args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n========== DE-RISK FAILED ==========", file=sys.stderr)
        traceback.print_exc()
        print("====================================", file=sys.stderr)
        sys.exit(1)

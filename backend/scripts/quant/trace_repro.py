"""Reproduce + diagnose the gemma4 sequential-GPTQ tracing failure in isolation.

Loads the model on CPU (meta-ish, bf16), builds a tiny sample input, and runs
llmcompressor's `trace_subgraphs` directly with a chosen `sequential_targets`.
This is the make-or-break symbolic trace — it does NO real compute, so it runs
without GPU headroom. Prints the full traceback on failure and the subgraph
count on success.

Usage:
  trace_repro.py --model <dir> [--targets Gemma4TextDecoderLayer] [--seqlen 64]
"""
from __future__ import annotations

import argparse
import sys
import traceback

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", nargs="*", default=None,
                    help="sequential_targets (default: auto from model)")
    ap.add_argument("--ignore", nargs="*", default=[],
                    help="function/method names to autowrap (non-traceable)")
    ap.add_argument("--seqlen", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor.pipelines.sequential.helpers import trace_subgraphs
    from llmcompressor.utils.pytorch.module import get_no_split_params

    print(f"[trace] loading {args.model} on CPU bf16 ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
        device_map="cpu", low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.targets is not None:
        targets = args.targets
    else:
        targets = get_no_split_params(model)
        print(f"[trace] auto no_split_params = {targets}", flush=True)

    # tiny sample input matching the calibration forward signature
    enc = tok("Translate: こんにちは", return_tensors="pt",
              truncation=True, max_length=args.seqlen, add_special_tokens=False)
    sample_input = {k: v for k, v in enc.items()}

    print(f"[trace] sequential_targets={targets}", flush=True)
    print("[trace] calling trace_subgraphs ... (the make-or-break)", flush=True)
    print(f"[trace] ignore(autowrap)={args.ignore}", flush=True)
    subgraphs = trace_subgraphs(
        model=model,
        sample_input=sample_input,
        sequential_targets=targets,
        ignore=args.ignore,
    )
    print(f"[trace] SUCCESS — produced {len(subgraphs)} subgraphs "
          f"(expect ~#decoder_layers + 1)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n========== TRACE FAILED ==========", file=sys.stderr)
        traceback.print_exc()
        print("==================================", file=sys.stderr)
        sys.exit(1)

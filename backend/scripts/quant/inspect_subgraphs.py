"""Inspect how the sequential subgraphs thread `shared_kv_states` across the
KV-sharing boundary (layers 0-23 produce, 24-41 consume). Prints, per subgraph,
the input placeholder names and output names, so we can see whether
shared_kv_states is threaded as an intermediate (it should be, for KV sharing to
survive partitioning).
"""
from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor.pipelines.sequential.helpers import trace_subgraphs

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
        device_map="cpu", low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    enc = tok("Translate: こんにちは", return_tensors="pt",
              truncation=True, max_length=64, add_special_tokens=False)
    sample_input = dict(enc)

    from llmcompressor.args.dataset_arguments import DatasetArguments
    tracing_ignore = list(DatasetArguments().tracing_ignore)

    subgraphs = trace_subgraphs(
        model=model, sample_input=sample_input,
        sequential_targets=["Gemma4TextDecoderLayer"],
        ignore=tracing_ignore,
    )
    print(f"\n=== {len(subgraphs)} subgraphs ===")
    for i, sg in enumerate(subgraphs):
        out_nodes = [n for n in sg.graph.nodes if n.op == "output"]
        out_names = []
        for o in out_nodes:
            if o.args and isinstance(o.args[0], dict):
                out_names = list(o.args[0].keys())
        has_skv_in = any("shared_kv" in n for n in sg.input_names)
        has_skv_out = any("shared_kv" in n for n in out_names)
        # detect call_module to a decoder layer
        callmods = [n.target for n in sg.graph.nodes if n.op == "call_module"]
        layer_calls = [t for t in callmods if "language_model.layers." in str(t)
                       and str(t).count(".") == 3]
        tag = layer_calls[0] if layer_calls else "(head)"
        print(f"sg[{i:2d}] {tag:30s} skv_in={has_skv_in} skv_out={has_skv_out} "
              f"in={sorted(sg.input_names)[:4]}... out={out_names[:4]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())

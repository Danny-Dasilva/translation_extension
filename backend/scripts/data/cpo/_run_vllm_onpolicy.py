#!/usr/bin/env python
"""Internal: generate on-policy v10-it samples for v10.5 CPO triplets.

This runs under /home/danny/.venvs/vllm because vLLM and the rest of the
backend use incompatible transformers versions. Do not import from the
backend package here.
"""
from __future__ import annotations
import argparse
import re
import sys
import time
from pathlib import Path

import polars as pl
from vllm import LLM, SamplingParams


USER_TEMPLATE = (
    "Translate the following Japanese to English. Output only the translation.\n\n"
    "Japanese: {jp}"
)

NEWLINE_RE = re.compile(r"[\r\n]")
NEXT_PROMPT_RE = re.compile(r"\s*(?:Japanese:|JP:|English:|EN:).*$", re.S)


def clean(text: str) -> str:
    if not text:
        return ""
    text = text.lstrip()
    text = NEWLINE_RE.split(text, 1)[0]
    text = NEXT_PROMPT_RE.sub("", text)
    # Strip residual chat-template markers
    for cut in ["<turn|>", "<|turn>", "<start_of_turn>", "<end_of_turn>"]:
        j = text.find(cut)
        if j >= 0:
            text = text[:j]
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=1024)
    args = ap.parse_args()

    df = pl.read_parquet(args.input)
    print(f"[vllm-onpolicy] {len(df)} rows from {args.input}", flush=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()
    print(f"[vllm-onpolicy] model loaded, building prompts...", flush=True)

    prompts = []
    for r in df.iter_rows(named=True):
        jp = (r.get("jp") or "").strip()
        msg = USER_TEMPLATE.format(jp=jp)
        text = tok.apply_chat_template(
            [{"role": "user", "content": msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(text)

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<end_of_turn>", "<|end|>"],
        seed=42,
    )

    print(f"[vllm-onpolicy] generating {len(prompts)} prompts...", flush=True)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    print(f"[vllm-onpolicy] done in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} prompts/s)", flush=True)

    raw_texts = []
    cleaned = []
    for o in outs:
        raw = o.outputs[0].text if o.outputs else ""
        raw_texts.append(raw)
        cleaned.append(clean(raw))

    out_df = df.with_columns([
        pl.Series("onpolicy_raw", raw_texts, dtype=pl.String),
        pl.Series("onpolicy_en", cleaned, dtype=pl.String),
    ])
    out_df.write_parquet(args.output)
    print(f"[vllm-onpolicy] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

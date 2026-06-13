"""Benchmark Gemma 4 E4B IT (with and without official MTP drafter) on RTX 5090.

Measures decode tok/s on a manga-style JP→EN translation prompt.

Without vLLM (transformers fallback), MTP isn't available — so this script
benchmarks transformers baseline first. vLLM benchmark requires
`uv run vllm benchmark` separately or installing vllm into a sibling venv.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT = (
    "Translate the following Japanese to English. "
    "Output only the translation.\n\nJapanese: 「一緒にお風呂、入ろう？」\nEnglish:"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--n-runs", type=int, default=3)
    args = ap.parse_args()

    logger.info("loading {}", args.model)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    logger.info("model load: {:.1f}s, VRAM {:.2f} GB",
                time.time() - t0, torch.cuda.max_memory_allocated() / 1e9)

    def _enc(s: str):
        out = tok(text=s, return_tensors="pt", add_special_tokens=False)
        return {k: v.to("cuda") for k, v in out.items()}

    # Warmup
    enc = _enc(PROMPT)
    with torch.inference_mode():
        _ = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)

    # Time runs
    decode_toks = []
    decode_times = []
    for i in range(args.n_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        torch.cuda.synchronize()
        dt = time.time() - t0
        n_new = out.shape[1] - enc["input_ids"].shape[1]
        decode_toks.append(n_new)
        decode_times.append(dt)
        text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        logger.info("run {}: {} tok in {:.2f}s = {:.1f} tok/s | sample: {}",
                    i + 1, n_new, dt, n_new / dt, text[:60])

    avg_tps = sum(t / dt for t, dt in zip(decode_toks, decode_times)) / len(decode_toks)
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print()
    print(f"=== {args.model} (transformers, bf16, no MTP) ===")
    print(f"avg decode: {avg_tps:.1f} tok/s")
    print(f"peak VRAM:  {peak_vram:.2f} GB")
    print(f"per-run:    {[f'{t/dt:.1f}' for t, dt in zip(decode_toks, decode_times)]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

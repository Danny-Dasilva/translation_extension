"""Quick 5-prompt sample translator for v10-it adapter post-training.

Outputs JSON of {prompt, translation} for each prompt. Used in the v10-it report
to spot-check the trained model against v9c on identical inputs.

Usage:
    .venv/bin/python scripts/eval/v10it_sample_translate.py \
        --adapter backend/training/runs/manga-bubbles/gemma4_e4b_v10it/final \
        --out backend/training/runs/manga-bubbles/gemma4_e4b_v10it/sample_translations.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import unsloth  # must come first
import torch
from loguru import logger


# Five canonical manga JP prompts used in the existing 3-way v7/v9c HTML comparison.
# Picking varied registers: short shout, polite, casual female, narration, slang.
SAMPLE_PROMPTS = [
    "ゴメン森･･･痛かったろ？",                      # apology, casual
    "明日は雨が降りそうですね。",                      # polite, weather
    "なんで僕がこんな目に遭わなきゃいけないんだ！",      # outburst
    "ふふ、君って本当に面白い人ね。",                  # flirtatious female
    "おい、ちょっと待ってくれよ！",                    # casual male shout
]

USER_TEMPLATE = (
    "Translate the following Japanese to English. Output only the translation.\n\n"
    "Japanese: {jp}"
)


def _ids(tok, s: str) -> list[int]:
    out = tok(text=s, add_special_tokens=False)["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    help="Path to LoRA adapter dir (Unsloth resolves base+adapter)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if not adapter.exists():
        logger.error("adapter dir not found: {}", adapter)
        return 2
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from unsloth import FastLanguageModel
    logger.info("loading {} via Unsloth", adapter)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=1024,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    results = []
    for jp in SAMPLE_PROMPTS:
        user_msg = USER_TEMPLATE.format(jp=jp)
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = _ids(tok, prompt_text)
        input_ids = torch.tensor([ids], device="cuda")
        attn = torch.ones_like(input_ids)
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.9,
                min_p=0.1,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        elapsed = time.time() - t0
        gen = out[0, input_ids.shape[1]:]
        raw = tok.decode(gen, skip_special_tokens=True).strip()
        # Clean: cut at any trailing chat marker
        for cut in ["<turn|>", "<|turn>", "Japanese:", "English:"]:
            i = raw.find(cut)
            if i >= 0:
                raw = raw[:i].strip()
        results.append({
            "jp": jp,
            "translation": raw,
            "elapsed_seconds": round(elapsed, 3),
        })
        logger.info("[{:>4}] JP: {}  →  EN: {}", "ok", jp, raw)

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("wrote → {}", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

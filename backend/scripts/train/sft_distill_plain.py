"""Minimal plain-transformers SFT for v8 distill.

Skips unsloth (broken under transformers 5.x); 1740 pairs × 3 epochs is small
enough that we don't need the unsloth speedup. Uses peft for LoRA.

Loads v7 merged model + new LoRA adapter, trains on Gemma distill pairs with
EOS-appended response-only masking.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl
import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def tokenize_with_labels(jp: str, en: str, tok, prompt_tmpl: str, max_len: int) -> dict:
    prompt = prompt_tmpl.format(jp=jp)
    completion = f" {en}{tok.eos_token}"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    input_ids = p_ids + c_ids
    labels = [-100] * len(p_ids) + list(c_ids)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="backend/training/datasets/filtered/gemma_distill_v8.parquet")
    ap.add_argument("--base", default="backend/training/weights/qwen3-mt-v7-merged")
    ap.add_argument("--out", default="backend/training/runs/manga-bubbles/qwen3_4b_v8_distill")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    PROMPT = "Translate the following Japanese to English. Output only the translation.\n\nJapanese: {jp}\nEnglish:"

    logger.info("loading data from {}", args.data)
    df = pl.read_parquet(args.data)
    logger.info("rows: {}", len(df))

    logger.info("loading {}", args.base)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if not hasattr(tok, "warnings_issued"):
        tok.warnings_issued = {}

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # required when gradient_checkpointing + LoRA

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model.print_trainable_parameters()

    logger.info("tokenizing...")
    rows = []
    for r in df.iter_rows(named=True):
        rows.append(tokenize_with_labels(r["jp"], r["en"], tok, PROMPT, args.max_len))
    ds = Dataset.from_list(rows)

    def collate(features):
        from torch.nn.utils.rnn import pad_sequence
        pad_id = tok.pad_token_id or tok.eos_token_id
        ii = [torch.tensor(f["input_ids"]) for f in features]
        ll = [torch.tensor(f["labels"]) for f in features]
        am = [torch.ones(len(f["input_ids"]), dtype=torch.long) for f in features]
        return {
            "input_ids": pad_sequence(ii, batch_first=True, padding_value=pad_id),
            "labels": pad_sequence(ll, batch_first=True, padding_value=-100),
            "attention_mask": pad_sequence(am, batch_first=True, padding_value=0),
        }

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model, args=targs, train_dataset=ds, data_collator=collate,
        processing_class=tok,
    )
    t0 = time.time()
    trainer.train()
    logger.info("training done in {:.1f}s", time.time()-t0)

    final = out_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    logger.info("saved → {}", final)
    return 0


if __name__ == "__main__":
    sys.exit(main())

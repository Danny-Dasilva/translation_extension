"""v9: Gemma 4 E4B-pt SFT on the v7.1 mix (144k pairs + UUF NSFW SFX).

Recipe (synthesized from oracle agents):
- LoRA r=16, alpha=32, dropout=0.05
- LR 1e-4 cosine, warmup 3%
- 1 epoch (144k pairs = plenty without forgetting)
- bs=2, grad_accum=8 → eff batch 16
- max_seq 1024
- Vision tower frozen (Gemma 4 E4B has vision but we don't train it)
- EOS append + response-only masking (proven from v7)

Avoids v8's failure mode: 144k diverse pairs, single epoch, regularized.
"""
from __future__ import annotations

import argparse
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


PROMPT_TEMPLATE = (
    "Translate the following Japanese to English. "
    "Output only the translation.\n\nJapanese: {jp}\nEnglish:"
)


def tokenize_with_labels(jp: str, en: str, tok, max_len: int) -> dict:
    prompt = PROMPT_TEMPLATE.format(jp=jp)
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
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="backend/training/runs/manga-bubbles/data_v71.parquet")
    ap.add_argument("--base", default="google/gemma-4-E4B")  # -pt (base, no IT chat template)
    ap.add_argument("--out", default="backend/training/runs/manga-bubbles/gemma4_e4b_v9")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Cap data for quick smoke tests")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading data from {}", args.data)
    df = pl.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit)
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
        attn_implementation="eager",  # Gemma hybrid attention safer with eager
    )
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model.print_trainable_parameters()

    logger.info("tokenizing {} rows...", len(df))
    rows = []
    for r in df.iter_rows(named=True):
        rows.append(tokenize_with_labels(r["jp"], r["en"], tok, args.max_len))
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
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=20,
        save_steps=500,
        save_total_limit=3,
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
    logger.info("training done in {:.1f}s", time.time() - t0)

    final = out_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    logger.info("saved → {}", final)
    return 0


if __name__ == "__main__":
    sys.exit(main())

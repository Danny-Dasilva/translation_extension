"""CPO-train v7 LoRA on (chosen=Gemma, rejected=v7) triplets to produce v8.

Loads our v7 merged model, attaches a fresh LoRA adapter, runs TRL's CPOTrainer
on the triplets parquet from build_cpo_triplets.py.

Output: backend/training/runs/manga-bubbles/qwen3_4b_v8_cpo/final/

Per ALMA-R: CPO with beta=0.1, lr=5e-7, 1 epoch, ~10-30k triplets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import CPOConfig, CPOTrainer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", default="backend/training/datasets/filtered/cpo_triplets.parquet")
    ap.add_argument("--base", default="backend/training/weights/qwen3-mt-v7-merged",
                    help="v7 merged model (start point for CPO)")
    ap.add_argument("--out", default="backend/training/runs/manga-bubbles/qwen3_4b_v8_cpo")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading triplets from {}", args.triplets)
    df = pl.read_parquet(args.triplets)
    logger.info("triplets: {}", len(df))
    if len(df) < 500:
        logger.warning("only {} triplets — CPO typically needs ≥2k for stable training",
                       len(df))

    ds = Dataset.from_pandas(df.to_pandas())

    logger.info("loading base + LoRA from {}", args.base)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    # trl 0.23 expects transformers <5 model.warnings_issued attr; patch for compat.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    if not hasattr(model.base_model.model, "warnings_issued"):
        model.base_model.model.warnings_issued = {}

    cpo_cfg = CPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        beta=args.beta,
        loss_type="sigmoid",
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        bf16=True,
        logging_steps=20,
        save_steps=200,
        save_total_limit=3,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = CPOTrainer(
        model=model,
        args=cpo_cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    logger.info("saved CPO adapter -> {}", final_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

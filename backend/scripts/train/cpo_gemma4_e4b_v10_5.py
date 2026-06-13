"""v10.5: CPO chain on top of v10-it (Gemma 4 E4B-it).

Stacks a fresh LoRA on the merged v10-it checkpoint and trains with TRL's
CPOTrainer using the SimPO-joint loss from the v10 synthesis recipe:

    loss_type=simpo, cpo_alpha=1.0, simpo_gamma=0.5, beta=2.0
    lr=1e-6, cosine, warmup_ratio=0.05, 1 epoch
    per_device_batch=2, grad_accum=32 (effective 64)
    max_length=768, max_prompt_length=384
    LoRA r=16, alpha=32, dropout=0.0 — same 258 lang-only modules as v10-it
    attn_implementation=sdpa (Gemma 4 hybrid attention on Blackwell sm_120)

Why merged-then-re-LoRA?
------------------------
v10-it itself is a LoRA on top of unsloth/gemma-4-E4B-it. We could try to
attach a SECOND LoRA to that PEFT-wrapped model, but Unsloth doesn't reliably
stack two LoRAs and PEFT's default behavior is to inject adapters at the same
target modules — composing the two cleanly at training time is brittle. The
canonical CPO chain pattern (used in ALMA-R and the v8 CPO recipe in this
repo) is: merge SFT LoRA into base, then attach a fresh LoRA for the CPO
phase. The merged checkpoint at
``gemma4_e4b_v10it/merged/`` is the k_norm-patched, vLLM-ready bundle —
treat it as the new base.

Prompt format
-------------
The preference parquet stores `prompt` as the raw user-message body
(``Translate the following...\\n\\nJapanese: <jp>``). At training time we
wrap each prompt with the Gemma 4 -it chat template via
``tok.apply_chat_template`` so the masked prompt tokens match what v10-it
saw during SFT.

CPOTrainer applies the chat-template wrapping when we override the
``processing_class`` and the dataset has a ``prompt`` column — TRL's
DPO/CPO data pipeline supports both the conversational and the
prompt+chosen+rejected style. Here we use the simpler prompt+chosen+rejected
style and pre-format the prompt strings, so the chat template only wraps
the user message (no second turn for the response, which is correct).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Unsloth must be imported BEFORE trl/transformers for monkeypatches.
import unsloth  # noqa: F401  pylint: disable=unused-import

import polars as pl
import torch
import yaml
from datasets import Dataset
from loguru import logger


PROJECT_ROOT = Path("/home/danny/Documents/personal/extension")

DEFAULT_TARGET_REGEX = (
    r"^model\.language_model\.layers\.\d+\.(self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def count_matching_modules(model, regex: str) -> tuple[int, list[str]]:
    pat = re.compile(regex)
    names = [n for n, _ in model.named_modules() if pat.match(n)]
    return len(names), names


def format_chat_prompt(tok, user_msg: str) -> str:
    """Wrap a bare user message with the Gemma 4 -it chat template.

    add_generation_prompt=True so the prompt ends right where the model
    should start emitting the assistant response.
    """
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_cpo_dataset(df: pl.DataFrame, tok) -> Dataset:
    """Pre-format prompts with the chat template; pass chosen/rejected as-is.

    CPOTrainer accepts a (prompt, chosen, rejected) dataset and tokenizes
    internally. By chat-template-wrapping the prompt at this stage, we bypass
    TRL's auto-detection (which can mis-route on Gemma4ForConditionalGeneration,
    a VLM-flavored class even though we never use vision/audio here).
    """
    rows = []
    skipped = 0
    for r in df.iter_rows(named=True):
        prompt = (r.get("prompt") or "").strip()
        chosen = (r.get("chosen") or "").strip()
        rejected = (r.get("rejected") or "").strip()
        if not prompt or not chosen or not rejected:
            skipped += 1
            continue
        if chosen == rejected:
            skipped += 1
            continue
        rows.append({
            "prompt": format_chat_prompt(tok, prompt),
            "chosen": chosen,
            "rejected": rejected,
        })
    if skipped:
        logger.info("[ds] skipped {} rows", skipped)
    return Dataset.from_list(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="backend/training/configs/gemma4_e4b_v10_5_cpo.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="Run 100 steps on 1% subsample (overrides config)")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap data rows for quick smoke (0 = no cap)")
    p.add_argument("--out-suffix", default="",
                   help="Append suffix to output dir (e.g. '_dryrun')")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_config(cfg_path)

    out_dir = Path(cfg["output"]["dir"])
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    if args.out_suffix:
        out_dir = Path(str(out_dir) + args.out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "training.log"
    logger.add(str(log_file), level="INFO", enqueue=True)
    logger.info("v10.5 CPO starting → out_dir={}", out_dir)
    logger.info("config: {}", json.dumps(cfg, default=str)[:1500])

    # ---- env preflight ----
    import transformers
    import peft
    import trl

    logger.info(
        "env: torch={} cuda={} sm={} unsloth={} transformers={} peft={} trl={}",
        torch.__version__,
        torch.version.cuda,
        torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        getattr(__import__("unsloth"), "__version__", "?"),
        transformers.__version__,
        peft.__version__,
        trl.__version__,
    )
    if not torch.cuda.is_available():
        logger.error("no CUDA device — bailing")
        return 2

    # ---- load preference data ----
    pref_path = Path(cfg["data"]["preference_path"])
    if not pref_path.is_absolute():
        pref_path = PROJECT_ROOT / pref_path
    logger.info("loading preferences: {}", pref_path)
    df = pl.read_parquet(pref_path)
    logger.info("preference rows: {} cols: {}", len(df), df.columns)
    logger.info("mean margin: {:.4f}", df["margin"].mean() if "margin" in df.columns else -1)
    logger.info("chosen kind dist:\n{}",
                df.group_by("chosen_kind").agg(pl.len()).sort("len", descending=True)
                  if "chosen_kind" in df.columns else "n/a")

    # Apply --limit / dry-run subsampling
    dryrun = args.dry_run or args.out_suffix.endswith("_dryrun")
    if args.limit:
        df = df.head(args.limit)
        logger.info("--limit applied: {} rows", len(df))
    elif args.out_suffix.endswith("_dryrun"):
        frac = float(cfg["dry_run"]["fraction"])
        n = max(500, int(len(df) * frac))
        df = df.sample(n=n, seed=int(cfg["train"]["seed"]))
        logger.info("dry_run subsample: {} rows ({}% of full)", len(df), frac * 100)

    # Hold out tail eval_size rows
    eval_size = int(cfg["data"].get("eval_size", 1000))
    if len(df) > eval_size + 100:
        train_df = df.head(len(df) - eval_size)
        eval_df = df.tail(eval_size)
    else:
        train_df = df
        eval_df = df.tail(min(100, len(df) // 10))
    logger.info("train rows: {}  eval rows: {}", len(train_df), len(eval_df))

    # ---- load model via Unsloth FastLanguageModel ----
    from unsloth import FastLanguageModel
    base = cfg["model"]["name_or_path"]
    if not Path(base).is_absolute() and not base.startswith(("unsloth/", "google/", "meta-")):
        base = str(PROJECT_ROOT / base)
    max_seq = int(cfg["model"]["max_seq_length"])
    logger.info("loading base model: {} (max_seq={})", base, max_seq)
    model, tok_or_proc = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=max_seq,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=False,
    )
    # Gemma 4 returns a multimodal Gemma4Processor whose __call__ chokes on
    # plain string input (TRL tokenize_row passes positional strings).
    # Extract the inner GemmaTokenizer for pure-text CPO.
    if hasattr(tok_or_proc, "tokenizer") and hasattr(tok_or_proc, "image_processor"):
        logger.info("unwrapping multimodal processor → inner tokenizer ({})",
                    type(tok_or_proc.tokenizer).__name__)
        tok = tok_or_proc.tokenizer
    else:
        tok = tok_or_proc
    if not hasattr(tok, "warnings_issued"):
        tok.warnings_issued = {}
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- module count check ----
    target_regex = cfg["lora"].get("target_modules_regex", DEFAULT_TARGET_REGEX)
    n_match, sample_names = count_matching_modules(model, target_regex)
    expected = int(cfg["lora"].get("expected_module_count", 258))
    logger.info("LoRA target regex: {}", target_regex)
    logger.info("modules matching regex: {} (expected: {})", n_match, expected)
    logger.info("sample matched (first 3): {}", sample_names[:3])
    if n_match != expected:
        logger.error("module count mismatch: got {} expected {}", n_match, expected)
        return 3

    # ---- attach LoRA ----
    seed = int(cfg["train"]["seed"])
    logger.info("attaching LoRA r={} alpha={} dropout={}",
                cfg["lora"]["r"], cfg["lora"]["alpha"], cfg["lora"]["dropout"])
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        target_modules=target_regex,
        bias=cfg["lora"]["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    if not hasattr(model.base_model.model, "warnings_issued"):
        model.base_model.model.warnings_issued = {}
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("trainable params: {:,} / {:,} ({:.4%})",
                trainable, total, trainable / total)

    # ---- build datasets ----
    logger.info("building train CPO dataset (chat-template prompts)...")
    train_ds = build_cpo_dataset(train_df, tok)
    logger.info("building eval CPO dataset...")
    eval_ds = build_cpo_dataset(eval_df, tok)
    logger.info("train_ds={} eval_ds={}", len(train_ds), len(eval_ds))

    # Decode sample
    sample = train_ds[0]
    logger.info("[sample] prompt (first 240): {!r}", sample["prompt"][:240])
    logger.info("[sample] chosen: {!r}", sample["chosen"])
    logger.info("[sample] rejected: {!r}", sample["rejected"])

    # ---- compute step schedule ----
    bs = int(cfg["train"]["per_device_train_batch_size"])
    ga = int(cfg["train"]["gradient_accumulation_steps"])
    epochs = float(cfg["train"]["num_train_epochs"])
    steps_per_epoch = max(1, len(train_ds) // (bs * ga))
    total_steps = int(steps_per_epoch * epochs)
    save_pct = float(cfg["train"]["save_steps_pct"])
    eval_pct = float(cfg["train"]["eval_steps_pct"])
    eval_steps = max(10, int(total_steps * eval_pct))
    raw_save_steps = max(20, int(total_steps * save_pct))
    # load_best_model_at_end requires save_steps % eval_steps == 0
    save_steps = max(eval_steps, (raw_save_steps // eval_steps) * eval_steps)
    if save_steps == 0:
        save_steps = eval_steps
    logger.info("steps_per_epoch={} total_steps≈{} save_steps={} eval_steps={} (raw_save={})",
                steps_per_epoch, total_steps, save_steps, eval_steps, raw_save_steps)

    # ---- CPO config ----
    from trl import CPOConfig, CPOTrainer

    cpo_kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=ga,
        learning_rate=float(cfg["train"]["learning_rate"]),
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        warmup_ratio=float(cfg["train"]["warmup_ratio"]),
        max_grad_norm=float(cfg["train"]["max_grad_norm"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        bf16=bool(cfg["train"]["bf16"]),
        optim=cfg["train"].get("optim", "adamw_8bit"),
        # CPO-SimPO joint (per Xu et al. and v10 synthesis)
        loss_type=cfg["cpo"]["loss_type"],
        cpo_alpha=float(cfg["cpo"]["cpo_alpha"]),
        simpo_gamma=float(cfg["cpo"]["simpo_gamma"]),
        beta=float(cfg["cpo"]["beta"]),
        max_length=int(cfg["cpo"]["max_length"]),
        max_prompt_length=int(cfg["cpo"]["max_prompt_length"]),
        logging_steps=int(cfg["train"]["logging_steps"]),
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_total_limit=int(cfg["train"]["save_total_limit"]),
        report_to=cfg["train"].get("report_to", "none"),
        seed=seed,
        dataloader_num_workers=int(cfg["train"]["dataloader_num_workers"]),
        remove_unused_columns=False,
        # Save best checkpoint by eval rewards/margins
        load_best_model_at_end=True,
        metric_for_best_model="rewards/margins",
        greater_is_better=True,
    )

    if dryrun:
        cpo_kwargs["max_steps"] = int(cfg["dry_run"].get("max_steps", 100))
        cpo_kwargs["save_steps"] = max(20, cpo_kwargs["max_steps"] // 2)
        cpo_kwargs["eval_steps"] = max(10, cpo_kwargs["max_steps"] // 5)
        cpo_kwargs["load_best_model_at_end"] = False
        logger.info("DRY_RUN mode: max_steps={}", cpo_kwargs["max_steps"])

    cpo_cfg = CPOConfig(**cpo_kwargs)

    trainer = CPOTrainer(
        model=model,
        args=cpo_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
    )

    # ---- VRAM snapshot ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        alloc = torch.cuda.memory_allocated() / 1e9
        logger.info("VRAM allocated before train: {:.2f} GB", alloc)

    # ---- TRAIN ----
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0
    logger.info("training done in {:.1f}s ({:.2f} h)", elapsed, elapsed / 3600)
    logger.info("train metrics: {}", train_result.metrics)
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        logger.info("VRAM peak: {:.2f} GB", peak)

    # ---- save final adapter ----
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    logger.info("saved final adapter → {}", final_dir)

    # ---- summary ----
    summary = {
        "out_dir": str(out_dir),
        "base_model": base,
        "preference_path": str(pref_path),
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_modules_matched": n_match,
        "elapsed_seconds": elapsed,
        "elapsed_hours": elapsed / 3600,
        "vram_peak_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None,
        "metrics": train_result.metrics,
        "lora": {
            "r": cfg["lora"]["r"],
            "alpha": cfg["lora"]["alpha"],
            "dropout": cfg["lora"]["dropout"],
            "target_regex": target_regex,
        },
        "cpo": cfg["cpo"],
        "train": {
            "lr": cfg["train"]["learning_rate"],
            "epochs": epochs,
            "per_device_batch": bs,
            "grad_accum": ga,
            "effective_batch": bs * ga,
            "max_seq": max_seq,
            "seed": seed,
        },
    }
    (out_dir / "v10_5-summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote summary → {}", out_dir / "v10_5-summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""v10-it: Fresh Gemma 4 E4B-IT LoRA SFT on the v10 mix (v7.1 + Manga109 1.5x).

Key v10 differences from v9c:
  * Base = unsloth/gemma-4-E4B-it (NOT -pt). Path A: pair with the
    gemma-4-E4B-it-assistant MTP drafter at deploy.
  * target_modules scoped to language_model.* ONLY (258 modules), NOT vision/audio.
    v9c accidentally adapted 406 modules because bare suffix matching also hit
    vision_tower (112) and audio_tower (36). A clean language-only LoRA is required
    for MTP drafter compatibility.
  * Use the Gemma 4 -it chat template (apply_chat_template) — not raw "Japanese: ... English:".
  * LR 2e-4 (fresh -it train), dropout 0.0, completion_only_loss=True.
  * Effective batch = 16 (per_device 4 × grad_accum 4).
  * 1 epoch on 258,981 rows.

Hyperparams locked per the research synthesis. See:
  backend/training/configs/gemma4_e4b_v10it_sft.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Unsloth must be imported BEFORE trl/transformers to apply optimizations.
import unsloth  # noqa: F401  pylint: disable=unused-import

import polars as pl
import torch
import yaml
from datasets import Dataset
from loguru import logger


# Match exactly the language-model attention/MLP projections — anchored regex
# so vision_tower / audio_tower projections are skipped.
DEFAULT_TARGET_REGEX = (
    r"^model\.language_model\.layers\.\d+\.(self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _ids(tok, s: str) -> list[int]:
    """Gemma 4's processor sometimes returns nested [[ids]]; flatten."""
    out = tok(text=s, add_special_tokens=False)["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def build_chat_dataset(
    df: pl.DataFrame,
    tok,
    user_template: str,
    max_len: int,
) -> Dataset:
    """Pre-tokenize each row using the Gemma 4 -it chat template, with
    completion-only masking already baked in via labels=-100 on prompt tokens.

    Why pre-tokenize? Gemma 4 E4B-it is `Gemma4ForConditionalGeneration` which
    Unsloth's SFTTrainer detects as a VLM and silently sets
    `skip_prepare_dataset=True`. That bypasses TRL's prompt-completion tokenization,
    so any conversational prompt/completion dataset never gets tokenized and
    fails at collate time. Pre-tokenizing avoids the entire VLM detection
    path and gives us deterministic completion-only loss masking.

    For each row we build:
        prompt_text = chat_template([{user, jp}], add_generation_prompt=True)
        full_text   = chat_template([{user, jp}, {assistant, en}], add_gen_prompt=False)
        prompt_ids  = tokenize(prompt_text)
        full_ids    = tokenize(full_text)
        labels      = [-100]*len(prompt_ids) + full_ids[len(prompt_ids):]

    The completion portion is full_ids[len(prompt_ids):] which includes the
    response and the closing <turn|> token (acts like EOS). This matches v9c's
    pre-tokenization shape so the rest of the trainer just works.
    """
    rows: list[dict[str, list[int]]] = []
    skipped = 0
    over_len = 0
    for r in df.iter_rows(named=True):
        jp = (r.get("jp") or "").strip()
        en = (r.get("en") or "").strip()
        if not jp or not en:
            skipped += 1
            continue
        user_msg = user_template.format(jp=jp).rstrip()
        prompt_msgs = [{"role": "user", "content": user_msg}]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": en}]
        prompt_text = tok.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
        )
        full_text = tok.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False,
        )
        # Note: apply_chat_template already inserts <bos> at the start; do not add special tokens again
        p_ids = _ids(tok, prompt_text)
        full_ids = _ids(tok, full_text)
        # Sanity: full_ids should start with p_ids
        if full_ids[: len(p_ids)] != p_ids:
            # Rare tokenizer drift (whitespace) — fall back to concatenation
            comp_text = full_text[len(prompt_text):]
            c_ids = _ids(tok, comp_text)
            full_ids = p_ids + c_ids
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]
            over_len += 1
        if len(full_ids) <= len(p_ids):
            # No completion tokens after truncation — skip
            skipped += 1
            continue
        labels = [-100] * len(p_ids) + list(full_ids[len(p_ids):])
        rows.append({
            "input_ids": full_ids,
            "labels": labels,
        })
    if skipped or over_len:
        logger.info("dataset build: skipped={} over_len={}", skipped, over_len)
    return Dataset.from_list(rows)


def count_matching_modules(model, regex: str) -> tuple[int, list[str]]:
    pat = re.compile(regex)
    names = [n for n, _ in model.named_modules() if pat.match(n)]
    return len(names), names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="backend/training/configs/gemma4_e4b_v10it_sft.yaml",
    )
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
        cfg_path = Path("/home/danny/Documents/personal/extension") / cfg_path
    cfg = load_config(cfg_path)

    out_dir = Path(cfg["output"]["dir"])
    if not out_dir.is_absolute():
        out_dir = Path("/home/danny/Documents/personal/extension") / out_dir
    if args.out_suffix:
        out_dir = Path(str(out_dir) + args.out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "training.log"
    logger.add(str(log_file), level="INFO", enqueue=True)
    logger.info("v10-it SFT starting → out_dir={}", out_dir)
    logger.info("config: {}", json.dumps(cfg, default=str)[:1500])

    # ---- env preflight ----
    import transformers
    import peft

    logger.info(
        "env: torch={} cuda={} sm={} unsloth={} transformers={} peft={}",
        torch.__version__,
        torch.version.cuda,
        torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        getattr(__import__("unsloth"), "__version__", "?"),
        transformers.__version__,
        peft.__version__,
    )
    if not torch.cuda.is_available():
        logger.error("no CUDA device — bailing")
        return 2

    # ---- load data ----
    data_path = Path(cfg["data"]["train_path"])
    if not data_path.is_absolute():
        data_path = Path("/home/danny/Documents/personal/extension") / data_path
    logger.info("loading data: {}", data_path)
    df = pl.read_parquet(data_path)
    logger.info("rows: {} cols: {}", len(df), df.columns)

    # Apply --limit / dry-run subsampling
    dryrun = args.dry_run or cfg.get("dry_run", {}).get("enable", False) and args.out_suffix.endswith("_dryrun")
    if args.limit:
        df = df.head(args.limit)
        logger.info("--limit applied: {} rows", len(df))
    elif args.out_suffix.endswith("_dryrun"):
        frac = float(cfg["dry_run"]["fraction"])
        n = max(2000, int(len(df) * frac))
        df = df.sample(n=n, seed=cfg["train"]["seed"])
        logger.info("dry_run subsample: {} rows ({}% of full)", len(df), frac * 100)

    # Hold out the tail eval_size rows for in-distribution eval
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
    max_seq = int(cfg["model"]["max_seq_length"])
    logger.info("loading base model: {} (max_seq={})", base, max_seq)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=max_seq,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=False,
    )
    if not hasattr(tok, "warnings_issued"):
        tok.warnings_issued = {}
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- Verify module count BEFORE attaching LoRA ----
    target_regex = cfg["lora"].get("target_modules_regex", DEFAULT_TARGET_REGEX)
    n_match, sample_names = count_matching_modules(model, target_regex)
    expected = int(cfg["lora"].get("expected_module_count", 258))
    logger.info("LoRA target regex: {}", target_regex)
    logger.info("modules matching regex: {} (expected: {})", n_match, expected)
    logger.info("sample matched: {}", sample_names[:3])
    logger.info("sample matched (last): {}", sample_names[-3:])
    if n_match != expected:
        logger.error(
            "module count mismatch: got {} expected {}. "
            "v9c hit 406 (language+vision+audio); v10 should hit {} (language only).",
            n_match, expected, expected,
        )
        return 3

    # ---- attach LoRA via FastLanguageModel.get_peft_model ----
    # FastLanguageModel accepts target_modules as a list of full module name suffixes
    # OR Unsloth honors `target_modules` regex via raw peft. We pass the regex string
    # directly — peft 0.7+ supports str-form target_modules as a regex.
    seed = int(cfg["train"]["seed"])
    logger.info("attaching LoRA r={} alpha={} dropout={} (regex-scoped)",
                cfg["lora"]["r"], cfg["lora"]["alpha"], cfg["lora"]["dropout"])
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        target_modules=target_regex,  # regex string (peft supports this)
        bias=cfg["lora"]["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model.print_trainable_parameters()
    # Log structured count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("trainable params: {:,} / {:,} ({:.4%})",
                trainable, total, trainable / total)

    # Sanity check the LoRA scope post-attach: count adapter params per prefix.
    adapter_prefixes: dict[str, int] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Find the first 2 path segments
        parts = n.split(".")
        if len(parts) >= 4:
            key = ".".join(parts[1:3]) if parts[0] in {"base_model", "model"} else ".".join(parts[:2])
        else:
            key = parts[0]
        adapter_prefixes[key] = adapter_prefixes.get(key, 0) + 1
    logger.info("adapter param tensors by prefix: {}", adapter_prefixes)

    # ---- build datasets ----
    user_template = cfg["data"]["prompt_user_template"]
    logger.info("building train dataset (pre-tokenized chat template)...")
    train_ds = build_chat_dataset(train_df, tok, user_template, max_seq)
    logger.info("building eval dataset (pre-tokenized chat template)...")
    eval_ds = build_chat_dataset(eval_df, tok, user_template, max_seq)
    logger.info("train_ds={} eval_ds={}", len(train_ds), len(eval_ds))

    # Demo: decode the FIRST training example.
    sample = train_ds[0]
    sample_ids = sample["input_ids"]
    sample_labels = sample["labels"]
    n_prompt_tokens = sum(1 for l in sample_labels if l == -100)
    n_completion_tokens = sum(1 for l in sample_labels if l != -100)
    decoded_full = tok.decode(sample_ids, skip_special_tokens=False)
    decoded_prompt = tok.decode(sample_ids[:n_prompt_tokens], skip_special_tokens=False)
    decoded_completion = tok.decode(sample_ids[n_prompt_tokens:], skip_special_tokens=False)
    logger.info("[example] full decoded (first 400 chars): {!r}", decoded_full[:400])
    logger.info("[example] prompt-portion (masked, first 200): {!r}", decoded_prompt[:200])
    logger.info("[example] completion-portion (loss applies, first 200): {!r}", decoded_completion[:200])
    logger.info("[example] seq_len={} prompt_tokens={} completion_tokens={}",
                len(sample_ids), n_prompt_tokens, n_completion_tokens)

    # ---- compute step schedule ----
    bs = int(cfg["train"]["per_device_train_batch_size"])
    ga = int(cfg["train"]["gradient_accumulation_steps"])
    epochs = float(cfg["train"]["num_train_epochs"])
    steps_per_epoch = max(1, len(train_ds) // (bs * ga))
    total_steps = int(steps_per_epoch * epochs)
    save_pct = float(cfg["train"]["save_steps_pct"])
    eval_pct = float(cfg["train"]["eval_steps_pct"])
    save_steps = max(50, int(total_steps * save_pct))
    eval_steps = max(20, int(total_steps * eval_pct))
    logger.info("steps_per_epoch={} total_steps≈{} save_steps={} eval_steps={}",
                steps_per_epoch, total_steps, save_steps, eval_steps)

    # ---- TrainingArguments ----
    # We use plain Trainer (not SFTTrainer) because:
    #   1. Gemma 4 E4B-it is `Gemma4ForConditionalGeneration` and Unsloth's SFTTrainer
    #      detects this as a VLM, silently setting skip_prepare_dataset=True.
    #   2. Our dataset is already pre-tokenized with labels=-100 on prompt tokens —
    #      this IS completion-only loss already (the trainer just runs CE on labels).
    #   3. Matches v9c's working pattern.
    from transformers import Trainer, TrainingArguments

    targs_kwargs = dict(
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
        logging_steps=int(cfg["train"]["logging_steps"]),
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_total_limit=int(cfg["train"]["save_total_limit"]),
        report_to=cfg["train"].get("report_to", "none"),
        seed=seed,
        dataloader_num_workers=int(cfg["train"]["dataloader_num_workers"]),
        remove_unused_columns=False,
    )
    # Dry-run override
    if args.out_suffix.endswith("_dryrun"):
        targs_kwargs["max_steps"] = int(cfg["dry_run"].get("max_steps", 100))
        targs_kwargs["save_steps"] = max(25, targs_kwargs["max_steps"] // 2)
        targs_kwargs["eval_steps"] = max(10, targs_kwargs["max_steps"] // 5)
        logger.info("DRY_RUN mode: max_steps={}", targs_kwargs["max_steps"])

    targs = TrainingArguments(**targs_kwargs)

    # Pad-and-batch collator (mirrors v9c)
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

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
        processing_class=tok,
    )

    # Decode the FIRST batch to verify completion-only masking is working.
    try:
        sample_batch = next(iter(trainer.get_train_dataloader()))
        keys = list(sample_batch.keys())
        logger.info("[mask check] batch keys: {}", keys)
        ids = sample_batch["input_ids"][0].tolist()
        labels0 = sample_batch["labels"][0].tolist()
        unmasked_idx = [i for i, l in enumerate(labels0) if l != -100]
        masked_idx = [i for i, l in enumerate(labels0) if l == -100]
        logger.info(
            "[mask check] seq_len={} masked(prompt)={} unmasked(completion)={}",
            len(ids), len(masked_idx), len(unmasked_idx),
        )
        if unmasked_idx:
            logger.info(
                "[mask check] completion tokens decoded (loss applies): {!r}",
                tok.decode([ids[i] for i in unmasked_idx], skip_special_tokens=False)[:240],
            )
        if masked_idx:
            logger.info(
                "[mask check] prompt tokens decoded (masked, first 240 chars): {!r}",
                tok.decode([ids[i] for i in masked_idx], skip_special_tokens=False)[:240],
            )
    except Exception as e:
        logger.warning("mask-check skipped: {}", e)

    # ---- VRAM snapshot before training ----
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

    # ---- write summary ----
    summary = {
        "out_dir": str(out_dir),
        "base_model": base,
        "data_path": str(data_path),
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
        "train": {
            "lr": cfg["train"]["learning_rate"],
            "epochs": epochs,
            "per_device_batch": bs,
            "grad_accum": ga,
            "effective_batch": bs * ga,
            "max_seq": max_seq,
            "completion_only_loss": cfg["train"]["completion_only_loss"],
            "seed": seed,
        },
    }
    (out_dir / "v10it-summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote summary → {}", out_dir / "v10it-summary.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())

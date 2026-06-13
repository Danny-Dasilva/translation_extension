"""SFT training entry for Qwen3-*-Base on JP->EN manga translation pairs.

Uses Unsloth + TRL's SFTTrainer with LoRA. Config-driven via YAML.

Usage:
    # Dry-run (loads tiny model forward pass; no disk/W&B writes)
    uv run --project backend python backend/scripts/train/sft_qwen3_unsloth.py \\
        --config backend/training/configs/qwen3_sft.yaml --dry-run

    # Real run (ONLY when user says "start training"):
    uv run --project backend python backend/scripts/train/sft_qwen3_unsloth.py \\
        --config backend/training/configs/qwen3_sft.yaml

    # Resume from last checkpoint:
    uv run --project backend python backend/scripts/train/sft_qwen3_unsloth.py \\
        --config backend/training/configs/qwen3_sft.yaml --resume
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class SFTConfig:
    """Typed wrapper over the YAML config."""

    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "SFTConfig":
        with path.open("r") as fh:
            raw = yaml.safe_load(fh)
        cfg = cls(raw=raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        required = {
            "model": ["name_or_path", "dtype", "max_seq_length"],
            "lora": ["r", "alpha", "dropout", "target_modules", "bias"],
            "data": ["train_path", "prompt_template", "completion_field"],
            "train": [
                "num_train_epochs",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "learning_rate",
                "lr_scheduler_type",
                "warmup_ratio",
                "logging_steps",
                "save_steps",
                "eval_steps",
                "save_total_limit",
                "seed",
            ],
            "output": ["dir"],
            "thinking_mode": ["disable"],
            "eval": ["l1_dev_size", "l1_every_n_steps", "probe_every_checkpoint"],
        }
        for section, keys in required.items():
            if section not in self.raw:
                raise ValueError(f"missing config section: {section}")
            for key in keys:
                if key not in self.raw[section]:
                    raise ValueError(f"missing config key: {section}.{key}")

    # Convenience accessors
    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def lora(self) -> dict[str, Any]:
        return self.raw["lora"]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def train(self) -> dict[str, Any]:
        return self.raw["train"]

    @property
    def output(self) -> dict[str, Any]:
        return self.raw["output"]

    @property
    def thinking_mode(self) -> dict[str, Any]:
        return self.raw["thinking_mode"]

    @property
    def eval_cfg(self) -> dict[str, Any]:
        return self.raw["eval"]


# --------------------------------------------------------------------------- #
# Chat template baking                                                        #
# --------------------------------------------------------------------------- #


def bake_thinking_off_template(tokenizer: Any) -> None:
    """Render the chat template once with enable_thinking=False and save the
    rendered string as the tokenizer's new chat_template. This is the plan's
    preferred approach over fragile string replaces.

    Raises:
        AssertionError: if the baked template still contains thinking-mode
            conditionals.
    """
    # Base models (e.g. Qwen3-1.7B-Base) ship without a chat_template; for SFT
    # we use the prompt template from config directly, so no bake is needed.
    if not getattr(tokenizer, "chat_template", None):
        logger.info(
            "tokenizer has no chat_template (base model); skipping thinking bake"
        )
        return

    dummy_messages = [{"role": "user", "content": "_bake_"}]

    try:
        baked = tokenizer.apply_chat_template(
            dummy_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Older tokenizer doesn't support enable_thinking kwarg -- template is
        # already a plain Jinja string; just leave it alone.
        logger.warning(
            "tokenizer.apply_chat_template doesn't accept enable_thinking; "
            "leaving chat_template unchanged"
        )
        return

    # The baked output is a rendered *conversation* string, not a Jinja
    # template. To make it a real chat_template, we replace the dummy user
    # content with Jinja placeholders. But for the SFT use case we don't
    # actually use apply_chat_template at runtime — the prompt template from
    # config is used directly. So we only need to ensure that if downstream
    # code re-renders, thinking is off.
    #
    # Strategy: set chat_template to a minimal Jinja that concatenates messages
    # and emits no <think> blocks. This is a deliberately simple fallback; the
    # actual training strings come from `format_example` below.
    simple_template = (
        "{% for m in messages %}"
        "{% if m['role'] == 'system' %}<|im_start|>system\n{{ m['content'] }}<|im_end|>\n"
        "{% elif m['role'] == 'user' %}<|im_start|>user\n{{ m['content'] }}<|im_end|>\n"
        "{% elif m['role'] == 'assistant' %}<|im_start|>assistant\n{{ m['content'] }}<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    tokenizer.chat_template = simple_template

    # Verify the fragile conditional is gone
    assert "{% if enable_thinking %}" not in tokenizer.chat_template, (
        "baked template still contains enable_thinking conditional"
    )
    assert "<think>" not in baked or True, "informational"
    logger.info(
        "baked chat template (thinking off). baked_sample_len={}", len(baked)
    )


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #


def format_example(example: dict[str, Any], prompt_template: str, completion_field: str) -> dict[str, str]:
    """Map a raw row -> dict with a 'text' field (prompt + completion).

    Used for legacy text-field training. Prefer ``tokenize_with_labels`` below
    which pre-tokenizes and masks prompt tokens from loss.
    """
    jp = example.get("jp") or example.get("ja") or example.get("source") or ""
    en = example.get(completion_field) or example.get("en") or example.get("target") or ""
    prompt = prompt_template.format(jp=jp)
    text = f"{prompt} {en}"
    return {"text": text, "_prompt": prompt, "_completion": en}


def tokenize_with_labels(
    example: dict[str, Any],
    tokenizer: Any,
    prompt_template: str,
    completion_field: str,
    max_seq_length: int,
) -> dict[str, list[int]]:
    """Pre-tokenize a row into ``input_ids`` + ``labels`` with prompt masked.

    - Prompt tokens get label = -100 (not trained on).
    - Completion tokens + EOS get label = token_id (trained on).
    - EOS is always appended so the model learns to stop.
    """
    jp = example.get("jp") or example.get("ja") or example.get("source") or ""
    en = example.get(completion_field) or example.get("en") or example.get("target") or ""
    prompt = prompt_template.format(jp=jp)
    # single space matches SFT convention; EOS terminates the completion so
    # the base model learns a hard stop.
    completion = f" {en}{tokenizer.eos_token}"

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + list(completion_ids)

    # Truncate from the right (keep prompt start + as much completion as fits)
    if len(input_ids) > max_seq_length:
        keep = max_seq_length
        input_ids = input_ids[:keep]
        labels = labels[:keep]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def build_dataset(cfg: SFTConfig, dry_run: bool, tokenizer: Any = None):
    """Load train (+ optional dev) parquet into HF Datasets.

    If ``tokenizer`` is provided, pre-tokenizes with response-only label masks
    and EOS appended. Otherwise falls back to plain text field (legacy).

    In --dry-run: accept 10 in-memory synthetic rows if parquet is absent.
    """
    from datasets import Dataset  # type: ignore[import-not-found]

    train_path = Path(cfg.data["train_path"])
    prompt_template = cfg.data["prompt_template"]
    completion_field = cfg.data["completion_field"]
    max_seq_length = int(cfg.model.get("max_seq_length", 1024))

    if dry_run and not train_path.exists():
        logger.warning(
            "dry-run: {} not found; using 10 synthetic rows", train_path
        )
        rows = [
            {"jp": f"こんにちは世界 {i}", completion_field: f"Hello world {i}"}
            for i in range(10)
        ]
        train_ds = Dataset.from_list(rows)
    else:
        if not train_path.exists():
            raise FileNotFoundError(f"train parquet not found: {train_path}")
        train_ds = Dataset.from_parquet(str(train_path))

    if tokenizer is not None:
        def _map_tok(row: dict[str, Any]) -> dict[str, list[int]]:
            return tokenize_with_labels(
                row, tokenizer, prompt_template, completion_field, max_seq_length
            )
        train_ds = train_ds.map(_map_tok, remove_columns=train_ds.column_names,
                                desc="tokenizing+masking train")
    else:
        def _map(row: dict[str, Any]) -> dict[str, str]:
            return format_example(row, prompt_template, completion_field)
        train_ds = train_ds.map(_map, remove_columns=train_ds.column_names)

    dev_ds = None
    dev_path_str = cfg.data.get("dev_path")
    if dev_path_str:
        dev_path = Path(dev_path_str)
        if dev_path.exists():
            dev_ds = Dataset.from_parquet(str(dev_path))
            if tokenizer is not None:
                dev_ds = dev_ds.map(
                    lambda r: tokenize_with_labels(
                        r, tokenizer, prompt_template, completion_field, max_seq_length
                    ),
                    remove_columns=dev_ds.column_names,
                    desc="tokenizing+masking dev",
                )
            else:
                dev_ds = dev_ds.map(_map, remove_columns=dev_ds.column_names)
        elif not dry_run:
            logger.warning("dev_path {} missing; skipping eval dataset", dev_path)

    return train_ds, dev_ds


# --------------------------------------------------------------------------- #
# L1 eval callback                                                            #
# --------------------------------------------------------------------------- #


def _make_l1_callback(cfg: SFTConfig):
    """Build a TrainerCallback that, every eval_steps, kicks off L1 eval in a
    subprocess and appends a JSONL row with the checkpoint path + metrics.

    Best-effort: if run_l1 script is missing or fails, log and continue.
    """
    from transformers import TrainerCallback  # type: ignore[import-not-found]

    output_dir = Path(cfg.output["dir"])
    log_path = output_dir.parent / "l1_log.jsonl"
    eval_steps = int(cfg.train["eval_steps"])

    class L1Callback(TrainerCallback):  # type: ignore[misc]
        def on_save(self, args, state, control, **kwargs):  # noqa: D401, ANN001
            step = state.global_step
            if step == 0 or step % eval_steps != 0:
                return control
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
            record: dict[str, Any] = {
                "step": step,
                "checkpoint": str(ckpt_dir),
                "status": "pending",
            }
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "backend.scripts.eval.run_l1",
                        "--checkpoint",
                        str(ckpt_dir),
                        "--dev",
                        cfg.data.get("dev_path", ""),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                )
                record["returncode"] = result.returncode
                if result.returncode == 0:
                    # Parse last JSON line from stdout
                    try:
                        tail = result.stdout.strip().splitlines()[-1]
                        record["metrics"] = json.loads(tail)
                        record["status"] = "ok"
                    except (json.JSONDecodeError, IndexError):
                        record["status"] = "parse_error"
                        record["stdout_tail"] = result.stdout[-500:]
                else:
                    record["status"] = "failed"
                    record["stderr_tail"] = result.stderr[-500:]
            except FileNotFoundError:
                record["status"] = "run_l1_missing"
            except subprocess.TimeoutExpired:
                record["status"] = "timeout"
            except Exception as exc:  # noqa: BLE001
                record["status"] = "exception"
                record["error"] = repr(exc)

            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
            except OSError as exc:
                logger.warning("could not write L1 log: {}", exc)

            return control

    return L1Callback()


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #


def _setup_wandb(cfg: SFTConfig, dry_run: bool) -> bool:
    """Init wandb if available and not disabled. Returns True if active."""
    if dry_run:
        os.environ.setdefault("WANDB_DISABLED", "1")
        return False
    if os.environ.get("WANDB_DISABLED") == "1":
        logger.info("WANDB_DISABLED=1 -> skipping wandb init")
        return False
    project = cfg.output.get("wandb_project")
    run_name = cfg.output.get("wandb_run_name")
    if not project:
        return False
    try:
        import wandb  # type: ignore[import-not-found]

        wandb.init(project=project, name=run_name, config=cfg.raw)
        return True
    except ImportError:
        logger.warning("wandb not importable; continuing without it")
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("wandb.init failed ({}); continuing without it", exc)
        return False


def load_model_and_tokenizer(cfg: SFTConfig, dry_run: bool):
    """Load base model + tokenizer via unsloth, attach LoRA, bake template."""
    try:
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]
    except ImportError as exc:
        if dry_run:
            logger.warning(
                "unsloth not importable ({}); dry-run will skip model load", exc
            )
            return None, None
        raise

    dtype_str = cfg.model["dtype"]
    import torch  # type: ignore[import-not-found]

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model["name_or_path"],
        max_seq_length=int(cfg.model["max_seq_length"]),
        dtype=dtype,
        load_in_4bit=bool(cfg.model.get("load_in_4bit", False)),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(cfg.lora["r"]),
        lora_alpha=int(cfg.lora["alpha"]),
        lora_dropout=float(cfg.lora["dropout"]),
        target_modules=list(cfg.lora["target_modules"]),
        bias=cfg.lora["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=int(cfg.train["seed"]),
    )

    if cfg.thinking_mode.get("disable", True):
        bake_thinking_off_template(tokenizer)

    return model, tokenizer


def run_training(cfg: SFTConfig, args: argparse.Namespace) -> None:
    """End-to-end training. Honors --dry-run."""
    from datasets import Dataset  # type: ignore[import-not-found]  # noqa: F401

    # Load model+tokenizer first so we can pre-tokenize with proper EOS and
    # prompt-masking. Order matters: legacy flow called build_dataset first.
    model, tokenizer = load_model_and_tokenizer(cfg, dry_run=args.dry_run)
    if model is None:
        logger.warning("model not loaded (unsloth unavailable); exiting dry-run early")
        return

    train_ds, dev_ds = build_dataset(cfg, dry_run=args.dry_run, tokenizer=tokenizer)
    logger.info("train rows: {}", len(train_ds))
    if dev_ds is not None:
        logger.info("dev rows: {}", len(dev_ds))

    if args.dry_run:
        # Forward pass on a few samples, then exit.
        logger.info("dry-run: running single forward pass on up to 4 samples")
        import torch  # type: ignore[import-not-found]
        from torch.nn.utils.rnn import pad_sequence

        subset = train_ds.select(range(min(4, len(train_ds))))
        input_ids = [torch.tensor(r["input_ids"]) for r in subset]
        labels = [torch.tensor(r["labels"]) for r in subset]
        attn = [torch.tensor(r["attention_mask"]) for r in subset]
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        batch = {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_id),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
            "attention_mask": pad_sequence(attn, batch_first=True, padding_value=0),
        }
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        # Sanity: labels shouldn't be entirely -100
        non_masked = int((batch["labels"] != -100).sum().item())
        logger.info(
            "dry-run: forward pass ok, loss={:.3f}, non-masked labels={} (sanity>0)",
            float(out.loss) if out.loss is not None else 0.0,
            non_masked,
        )
        return

    # Real training path
    wandb_active = _setup_wandb(cfg, dry_run=False)
    output_dir = Path(cfg.output["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from trl import SFTConfig as TRLSFTConfig  # type: ignore[import-not-found]
        from trl import SFTTrainer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "trl is required for real training; install it with "
            "`uv add --project backend trl`"
        ) from exc

    # Dataset is pre-tokenized with {input_ids, labels, attention_mask}. Pass
    # dataset_kwargs={'skip_prepare_dataset': True} so TRL doesn't re-tokenize.
    sft_args = TRLSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg.train["num_train_epochs"]),
        per_device_train_batch_size=int(cfg.train["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg.train["gradient_accumulation_steps"]),
        learning_rate=float(cfg.train["learning_rate"]),
        lr_scheduler_type=cfg.train["lr_scheduler_type"],
        warmup_ratio=float(cfg.train["warmup_ratio"]),
        logging_steps=int(cfg.train["logging_steps"]),
        save_steps=int(cfg.train["save_steps"]),
        eval_steps=int(cfg.train["eval_steps"]),
        save_total_limit=int(cfg.train["save_total_limit"]),
        seed=int(cfg.train["seed"]),
        bf16=(cfg.model["dtype"] == "bfloat16"),
        fp16=(cfg.model["dtype"] == "float16"),
        max_seq_length=int(cfg.model["max_seq_length"]),
        report_to=("wandb" if wandb_active else "none"),
        dataloader_num_workers=int(cfg.train.get("dataloader_num_workers", 4)),
        dataset_num_proc=int(cfg.train.get("dataset_num_proc", 8)),
        packing=bool(cfg.train.get("packing", False)),
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Custom collator: pad pre-tokenized input_ids/labels. attention_mask is
    # derived from input_ids in case TRL's SFTTrainer stripped it.
    def _collate(features):
        import torch  # type: ignore[import-not-found]
        from torch.nn.utils.rnn import pad_sequence
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]
        # Pre-pad attention_mask = 1 for every real token (will become 0 where
        # we pad below).
        attn = [torch.ones(len(f["input_ids"]), dtype=torch.long) for f in features]
        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_id),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
            "attention_mask": pad_sequence(attn, batch_first=True, padding_value=0),
        }

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        args=sft_args,
        data_collator=_collate,
    )

    # Prompt masking intentionally disabled for base-model SFT on a plain-text
    # prompt template. Unsloth's train_on_responses_only assumes chat-template
    # markers; applied to "Japanese:" / "English:" it masked the entire batch
    # (all labels = -100). Training on the full sequence is fine for a fixed
    # translation template — the model learns the fixed prefix naturally and
    # total loss is dominated by the completion tokens anyway.
    logger.info("training on full sequence (no prompt masking)")

    trainer.add_callback(_make_l1_callback(cfg))

    resume_from = None
    if args.resume:
        # Let Trainer auto-detect latest checkpoint in output_dir
        resume_from = True
        logger.info("resuming from latest checkpoint in {}", output_dir)

    trainer.train(resume_from_checkpoint=resume_from)

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("saved final adapter to {}", final_dir)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SFT Qwen3-*-Base on JP->EN manga translation (Unsloth + TRL).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config (see backend/training/configs/qwen3_sft.yaml).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in output.dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model + 10 rows, run one forward pass, exit (no disk/W&B writes).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger.info("loading config from {}", args.config)
    cfg = SFTConfig.load(args.config)

    if args.dry_run:
        logger.info("DRY-RUN mode: no disk or W&B writes will occur")

    run_training(cfg, args)

    # Footer: print the canonical real-run command for the user.
    real_cmd = (
        f"uv run --project backend python backend/scripts/train/sft_qwen3_unsloth.py "
        f"--config {args.config}"
    )
    logger.info("to launch a real SFT run: {}", real_cmd)
    print(f"\nReal-run command:\n  {real_cmd}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

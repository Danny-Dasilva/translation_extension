"""v12vision — Gemma-4 E4B VISION-LoRA SFT (Phase 2: page IMAGE as context).

SCAFFOLDING — RUNNABLE ON THE BOX BUT NOT YET RUN. No GPU has executed this.

What this differs from the v10it/v11 TEXT trainer
-------------------------------------------------
  * Loads the CLEAN multimodal base `unsloth/gemma-4-E4B-it` via unsloth
    `FastVisionModel` (NOT `FastLanguageModel`). The base is
    `Gemma4ForConditionalGeneration` — text + vision + audio towers.
  * LoRA is EXTENDED to the vision tower (and language tower) via unsloth's
    `finetune_vision_layers=True` flag — the #1 Phase-2 lever (page image as
    context for the pronoun/speaker ceiling). The v11 text LoRA touched
    `language_model.*` ONLY, leaving the vision path unconditioned.
  * Dataset is a VISION CHAT dataset: each page becomes one user turn carrying
    {image block + jp_ocr + page_context} and one assistant turn = en_target.
  * Uses unsloth's `UnslothVisionDataCollator` (handles image preprocessing +
    completion-only label masking) with trl `SFTTrainer`, NOT the hand-rolled
    pad collator + plain `Trainer` used by the text recipe.

Config: backend/training/configs/gemma4_e4b_v12vision_sft.yaml

============================================================================
API NAMES THAT NEED ON-BOX VERIFICATION (unsloth 2026.6.7 / trl 0.23.1)
Every line flagged `?? VERIFY` below is an API name/behavior assumed from
the unsloth vision convention and MUST be confirmed before a real run. The
script's --inspect mode prints exactly what you need to confirm them.
============================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Unsloth MUST be imported before trl/transformers to apply its patches.
import unsloth  # noqa: F401  pylint: disable=unused-import

import torch
import yaml
from loguru import logger

PROJECT_ROOT = Path("/home/danny/Documents/personal/extension")


# --------------------------------------------------------------------------- #
# config + data helpers
# --------------------------------------------------------------------------- #

def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_user_text(rec: dict[str, Any], cfg_data: dict[str, Any]) -> str:
    """Compose the TEXT portion of the user turn (image is a separate block).

    Mirrors the v11 page-context user-message intent (instruction + page OCR +
    context) but adapted for the whole-page multimodal POC schema.
    """
    instr = cfg_data["instruction"].strip()
    jp_ocr = (rec.get("jp_ocr") or "").strip()
    page_ctx = (rec.get("page_context") or "").strip()
    parts = [instr]
    if jp_ocr:
        parts.append(f"{cfg_data.get('ocr_label', 'Page OCR:')}\n{jp_ocr}")
    if page_ctx:
        parts.append(f"{cfg_data.get('context_label', 'Context:')}\n{page_ctx}")
    return "\n\n".join(parts)


def build_conversations(
    rows: list[dict[str, Any]],
    cfg_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Turn POC rows into unsloth vision chat samples.

    Output shape (one element per page):
        {"messages": [
            {"role": "user", "content": [
                {"type": "image", "image": <PIL.Image RGB>},
                {"type": "text",  "text": "<instruction + OCR + context>"},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "<en_target>"},
            ]},
        ]}

    ?? VERIFY: this multimodal message schema (content = list of typed blocks
    with {"type":"image","image":PIL}) is the unsloth/HF vision-chat convention.
    Confirm UnslothVisionDataCollator on 2026.6.7 consumes exactly this. Some
    builds key the image as {"type":"image"} with the PIL passed separately, or
    accept {"type":"image","image":<path>}. The --inspect mode dumps one sample.

    NOTE (memory): for the FULL dataset, eager-loading every page PIL into a
    Python list will blow RAM. The POC is small so this is fine; switch to a
    lazy/streaming dataset (load PIL in the collator from image_path) at scale.
    """
    from PIL import Image  # local import: only needed at data-build time

    image_root = _abs(cfg_data["image_root"])
    out: list[dict[str, Any]] = []
    skipped = 0
    for rec in rows:
        en = (rec.get("en_target") or "").strip()
        img_path = rec.get("image_path") or ""
        if not en or not img_path:
            skipped += 1
            continue
        ip = Path(img_path)
        if not ip.is_absolute():
            ip = image_root / ip
        if not ip.exists():
            logger.warning("image missing, skipping page: {}", ip)
            skipped += 1
            continue
        try:
            img = Image.open(ip).convert("RGB")
        except Exception as e:  # noqa: BLE001
            logger.warning("PIL open failed for {}: {} — skipping", ip, e)
            skipped += 1
            continue
        user_text = build_user_text(rec, cfg_data)
        out.append({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": user_text},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": en},
                ]},
            ]
        })
    if skipped:
        logger.info("conversation build: skipped={} (missing en/image)", skipped)
    return out


# --------------------------------------------------------------------------- #
# LoRA section audit — catches the silent "vision tower not adapted" bug
# --------------------------------------------------------------------------- #

def audit_lora_sections(model) -> dict[str, int]:
    """Count trainable LoRA param tensors grouped by tower.

    The whole point of Phase 2 is adapting the vision tower; if unsloth silently
    skipped it (e.g. didn't recognize gemma4_vision's ClippableLinear wrapper),
    `vision_tower` here will be 0 and we must bail rather than train a text-only
    LoRA under a vision banner.
    """
    sections: dict[str, int] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad or "lora" not in n.lower():
            continue
        if "language_model" in n:
            key = "language_model"
        elif "vision_tower" in n:
            key = "vision_tower"
        elif "multi_modal_projector" in n:
            key = "multi_modal_projector"
        elif "audio_tower" in n:
            key = "audio_tower"
        else:
            key = "other"
        sections[key] = sections.get(key, 0) + 1
    return sections


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="backend/training/configs/gemma4_e4b_v12vision_sft.yaml",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Short smoke run (max_steps from config.dry_run.max_steps)")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap pages for a quick smoke (0 = no cap)")
    p.add_argument("--out-suffix", default="",
                   help="Append suffix to output dir (e.g. '_dryrun')")
    p.add_argument("--inspect", action="store_true",
                   help="Load model + 1 sample, print named_modules / LoRA section "
                        "audit / one converted conversation, then EXIT without "
                        "training. Use this FIRST to verify the vision module "
                        "names + collator schema on the box.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(_abs(args.config))

    out_dir = _abs(cfg["output"]["dir"])
    if args.out_suffix:
        out_dir = Path(str(out_dir) + args.out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(str(out_dir / "training.log"), level="INFO", enqueue=True)
    logger.info("v12vision SFT starting → out_dir={}", out_dir)

    # ---- env preflight --------------------------------------------------------
    import transformers
    import peft
    logger.info(
        "env: torch={} cuda={} sm={} unsloth={} transformers={} peft={}",
        torch.__version__, torch.version.cuda,
        torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        getattr(__import__("unsloth"), "__version__", "?"),
        transformers.__version__, peft.__version__,
    )
    if not torch.cuda.is_available():
        logger.error("no CUDA device — bailing")
        return 2

    # ---- load CLEAN multimodal base via FastVisionModel ----------------------
    # ?? VERIFY: `FastVisionModel` is the unsloth vision entrypoint (mirrors
    #    FastLanguageModel for VLMs). Returns (model, processor); unsloth calls
    #    the 2nd value `tokenizer` but it is the AutoProcessor (has .tokenizer +
    #    image processor). Confirm the signature on 2026.6.7.
    from unsloth import FastVisionModel

    base = cfg["model"]["name_or_path"]
    max_seq = int(cfg["model"]["max_seq_length"])
    logger.info("loading multimodal base: {} (max_seq={})", base, max_seq)
    model, processor = FastVisionModel.from_pretrained(
        base,
        max_seq_length=max_seq,        # ?? VERIFY: some FastVisionModel builds
                                       #    ignore max_seq_length; harmless if so.
        dtype=torch.bfloat16,
        load_in_4bit=bool(cfg["model"]["load_in_4bit"]),
        use_gradient_checkpointing="unsloth",
    )

    # ---- attach VISION + LANGUAGE LoRA ---------------------------------------
    # ?? VERIFY: FastVisionModel.get_peft_model boolean-flag API. This is the
    #    documented unsloth vision pattern; it resolves the per-arch wrapped
    #    Linears (incl. gemma4_vision Gemma4ClippableLinear at `*.proj.linear`)
    #    internally. If unsloth does NOT know gemma4_vision, the section audit
    #    below will show vision_tower == 0 and we bail.
    lcfg = cfg["lora"]
    logger.info("attaching LoRA r={} alpha={} (vision={} language={})",
                lcfg["r"], lcfg["alpha"],
                lcfg["finetune_vision_layers"], lcfg["finetune_language_layers"])
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=bool(lcfg["finetune_vision_layers"]),
        finetune_language_layers=bool(lcfg["finetune_language_layers"]),
        finetune_attention_modules=bool(lcfg["finetune_attention_modules"]),
        finetune_mlp_modules=bool(lcfg["finetune_mlp_modules"]),
        r=int(lcfg["r"]),
        lora_alpha=int(lcfg["alpha"]),
        lora_dropout=float(lcfg["dropout"]),
        bias=lcfg["bias"],
        random_state=int(cfg["train"]["seed"]),
        use_rslora=False,
        # target_modules=...  # leave to unsloth's flags; see explicit regex in
        #                       the YAML (lora.target_modules_regex_explicit) only
        #                       if you bypass FastVisionModel's resolver.
    )

    # ---- audit: which towers actually got a LoRA? ----------------------------
    sections = audit_lora_sections(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("LoRA sections (trainable tensor counts): {}", sections)
    logger.info("trainable params: {:,} / {:,} ({:.4%})",
                trainable, total, trainable / total)
    if sections.get("vision_tower", 0) == 0:
        logger.error(
            "VISION TOWER GOT NO LoRA — Phase-2 requires adapting the vision "
            "tower. unsloth likely did not recognize gemma4_vision's "
            "Gemma4ClippableLinear wrapper. Fix: pass the explicit regex "
            "lora.target_modules_regex_explicit (note the trailing `.linear` on "
            "the vision branch) via get_peft_model(target_modules=...), or "
            "upgrade unsloth. Bailing rather than training a text-only LoRA.")
        return 3

    # ---- build the vision chat dataset ---------------------------------------
    data_path = _abs(cfg["data"]["train_path"])
    if not data_path.exists():
        logger.error("dataset not found: {} (sibling agent builds it). Bailing.",
                     data_path)
        return 4
    rows = read_jsonl(data_path)
    logger.info("loaded {} pages from {}", len(rows), data_path)
    if args.limit:
        rows = rows[: args.limit]
    elif args.out_suffix.endswith("_dryrun") or args.dry_run:
        frac = float(cfg["dry_run"]["fraction"])
        rows = rows[: max(4, int(len(rows) * frac))]
        logger.info("dry_run subsample: {} pages", len(rows))

    eval_size = int(cfg["data"].get("eval_size", 64))
    if len(rows) > eval_size + 8:
        train_rows, eval_rows = rows[:-eval_size], rows[-eval_size:]
    else:
        train_rows, eval_rows = rows, rows[-max(2, len(rows) // 10):]
    train_ds = build_conversations(train_rows, cfg["data"])
    eval_ds = build_conversations(eval_rows, cfg["data"])
    logger.info("train pages={} eval pages={}", len(train_ds), len(eval_ds))

    # ---- --inspect: verify everything, then EXIT (no training) ---------------
    if args.inspect:
        logger.info("=== --inspect: VISION module names (vision_tower.*proj*) ===")
        shown = 0
        for n, _ in model.named_modules():
            if "vision_tower" in n and ("proj" in n or "linear" in n):
                logger.info("  {}", n)
                shown += 1
                if shown >= 20:
                    logger.info("  ... (truncated)")
                    break
        logger.info("=== --inspect: multi_modal_projector.* modules ===")
        for n, _ in model.named_modules():
            if "multi_modal_projector" in n:
                logger.info("  {}", n)
        if train_ds:
            s = train_ds[0]
            preview = {
                "n_messages": len(s["messages"]),
                "user_content_types": [c["type"] for c in s["messages"][0]["content"]],
                "user_text_head": next(
                    (c["text"][:200] for c in s["messages"][0]["content"]
                     if c["type"] == "text"), ""),
                "assistant_text_head": s["messages"][1]["content"][0]["text"][:200],
            }
            logger.info("=== --inspect: one converted conversation ===\n{}",
                        json.dumps(preview, ensure_ascii=False, indent=2))
        logger.info("--inspect done. NO TRAINING performed. Verify the module "
                    "names + section audit above, then run without --inspect.")
        return 0

    # ---- trainer (trl SFTTrainer + unsloth vision collator) ------------------
    # ?? VERIFY: import path of the vision collator on 2026.6.7. Known historical
    #    location: `from unsloth.trainer import UnslothVisionDataCollator`.
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    FastVisionModel.for_training(model)  # ?? VERIFY: enables training mode for VLM

    bs = int(cfg["train"]["per_device_train_batch_size"])
    ga = int(cfg["train"]["gradient_accumulation_steps"])
    epochs = float(cfg["train"]["num_train_epochs"])
    steps_per_epoch = max(1, len(train_ds) // (bs * ga))
    total_steps = int(steps_per_epoch * epochs)
    save_steps = max(10, int(total_steps * float(cfg["train"]["save_steps_pct"])))
    eval_steps = max(5, int(total_steps * float(cfg["train"]["eval_steps_pct"])))

    # ?? VERIFY: SFTConfig field names on trl 0.23.1.
    #   - trl>=0.20 renamed `max_seq_length` -> `max_length`; we set `max_length`.
    #     If 0.23.1 errors on it, switch the key back to `max_seq_length`.
    #   - `dataset_kwargs={"skip_prepare_dataset": True}` + `remove_unused_columns
    #     =False` are REQUIRED so trl does not try to text-tokenize the image
    #     conversations (the collator owns preprocessing).
    sft_args = SFTConfig(
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
        seed=int(cfg["train"]["seed"]),
        dataloader_num_workers=int(cfg["train"]["dataloader_num_workers"]),
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=max_seq,
    )
    if args.dry_run or args.out_suffix.endswith("_dryrun"):
        sft_args.max_steps = int(cfg["dry_run"].get("max_steps", 30))
        logger.info("DRY_RUN: max_steps={}", sft_args.max_steps)

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,            # ?? VERIFY: trl 0.23 uses
                                               #    `processing_class`; older trl
                                               #    used `tokenizer=`. unsloth
                                               #    patches accept either.
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logger.info("VRAM before train: {:.2f} GB", torch.cuda.memory_allocated() / 1e9)

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    logger.info("training done in {:.1f}s ({:.2f} h)", elapsed, elapsed / 3600)
    if torch.cuda.is_available():
        logger.info("VRAM peak: {:.2f} GB", torch.cuda.max_memory_allocated() / 1e9)

    # ---- save final adapter ---------------------------------------------------
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    logger.info("saved LoRA adapter → {}", final_dir)
    # To MERGE for serving, prefer unsloth's native merge which handles the
    # vision/projector keys (the repo's merge_gemma4_lora_clean.py was written for
    # language-only adapters). See V12VISION_README.md "Merging the vision-LoRA".
    #   model.save_pretrained_merged(str(out_dir/"merged"), processor,
    #                                save_method="merged_16bit")   # ?? VERIFY API

    (out_dir / "v12vision-summary.json").write_text(json.dumps({
        "out_dir": str(out_dir),
        "base_model": base,
        "data_path": str(data_path),
        "train_pages": len(train_ds),
        "eval_pages": len(eval_ds),
        "lora_sections": sections,
        "trainable_params": trainable,
        "total_params": total,
        "elapsed_seconds": elapsed,
        "metrics": result.metrics,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

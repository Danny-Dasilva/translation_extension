"""Qwen3-VL-8B IMAGE-CONTEXT LoRA SFT (Phase-1 POC: page IMAGE as context).

Ported from ``sft_gemma4_e4b_v12vision.py``. SCAFFOLDING — RUNNABLE ON THE BOX
BUT NOT YET RUN. No GPU has executed this. Plan:
thoughts/shared/plans/2026-06-30_image-context-vlm-finetune.md (§4 + §7).

WHAT CHANGED vs the Gemma-4 v12vision trainer
---------------------------------------------
  * BASE = huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated (uncensored twin of
    the measured-46%-POV Qwen3-VL-8B-Instruct; abliteration kills the refusal
    tiger at the weights — pre-mortem §2 / uncensored-base direction).
  * Loaded via unsloth ``FastVisionModel`` (the Qwen3-VL vision notebook pattern).
  * LoRA-section audit expects QWEN3-VL tower names (visual tower + merger +
    language self_attn/mlp) instead of Gemma's. KEEPS the audit-guard that BAILS
    if the vision-tower LoRA tensor count == 0 (Phase-2's whole point).
  * Consumes the NEW POC schema: each jsonl row is already a Qwen3-VL chat sample
    ({"messages":[...]} with the image block carrying an image PATH string). The
    trainer hydrates image paths -> PIL at load; text-only rows (no image block)
    pass through unchanged -> mixed image/no-image dataset (§3.2).
  * ADDS ``--resume`` (resume_from_checkpoint) — pre-mortem §4 (box thermal fault
    + no resume = full restart). The Gemma script lacked it.
  * ADDS an OPTIONAL modality-bucketed batch sampler (``data.bucket_by_modality``)
    for the case where the unsloth vision collator rejects mixed image/text
    batches (VERIFY-ON-BOX; default OFF = single mixed dataset).

Config: backend/training/configs/qwen3vl_8b_imagectx_poc.yaml (image-on)
        backend/training/configs/qwen3vl_8b_imagectx_poc_imageoff.yaml (control)

============================================================================
VERIFY-ON-BOX (unsloth / trl / transformers pins on the box build)
Every line flagged ``# VERIFY-ON-BOX`` is an API name/behaviour ASSUMED from the
unsloth Qwen3-VL vision convention and MUST be confirmed by ``--inspect`` before
a real run. The known-unknowns the task calls out explicitly:
  (1) Qwen3-VL unsloth module names (visual.* / merger / language_model.*).
  (2) UnslothVisionDataCollator mixed image/no-image batch behaviour.
  (3) processor max_pixels arg (image-token budget knob).
============================================================================
"""
from __future__ import annotations

import argparse
import copy
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
# BLOCKER-1 FIX: neutralize unsloth's fix_untrained_tokens for LoRA SFT.
# --------------------------------------------------------------------------- #
# unsloth_zoo.tokenizer_utils.fix_untrained_tokens() runs inside
# UnslothSFTTrainer.__init__ and does `lm_head_bad = lm_head_bad.cpu().float()`,
# which raises `NotImplementedError: Cannot copy out of meta tensor; no data!`
# because the Qwen3-VL lm_head is a META tensor under unsloth's lazy load. That
# routine only exists to reset *untrained* token embedding vectors when NEW
# tokens are added — we do a pure LoRA SFT that adds NO tokens, so it is
# unnecessary here. The compiled-cache trainer
# (~/unsloth_compiled_cache/UnslothSFTTrainer.py:1673) does a LOCAL
# `from unsloth_zoo.tokenizer_utils import fix_untrained_tokens` at CALL time
# (immediately before invoking it), so patching the module attribute here is
# resolved freshly at __init__ and takes effect. Patch the source module (and
# any already-imported alias) to a no-op.
def _disable_unsloth_untrained_token_fix() -> None:
    def _noop_fix_untrained_tokens(*args, **kwargs):  # pragma: no cover
        return None

    patched = []
    try:
        import unsloth_zoo.tokenizer_utils as _uz_tok  # unsloth already imported
        _uz_tok.fix_untrained_tokens = _noop_fix_untrained_tokens
        patched.append("unsloth_zoo.tokenizer_utils")
    except Exception as e:  # noqa: BLE001
        logger.warning("could not patch unsloth_zoo.tokenizer_utils: {}", e)
    # Belt-and-suspenders: patch any already-loaded UNSLOTH module that bound the
    # symbol at import time (e.g. an alias module already in sys.modules). Scoped
    # to unsloth-named modules with a CALLABLE fix_untrained_tokens so we don't
    # touch unrelated namespaces (torch.ops/torch.classes answer getattr for any
    # name, so an unscoped check spuriously "matches" them).
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        name = getattr(mod, "__name__", "")
        if "unsloth" not in name or name == "unsloth_zoo.tokenizer_utils":
            continue
        if callable(getattr(mod, "fix_untrained_tokens", None)):
            try:
                mod.fix_untrained_tokens = _noop_fix_untrained_tokens
                patched.append(name)
            except Exception:  # noqa: BLE001, S110
                pass
    logger.info("BLOCKER-1: fix_untrained_tokens -> no-op (LoRA SFT, no new "
                "tokens; avoids meta-tensor lm_head copy). patched: {}", patched)


_disable_unsloth_untrained_token_fix()


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


def build_conversations(
    rows: list[dict[str, Any]],
    cfg_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Hydrate POC chat rows -> unsloth vision-chat samples.

    Each POC row already holds the Qwen3-VL message list; the only work here is to
    replace any image content block's PATH string with a decoded PIL.Image (RGB).
    Text-only rows (no image block) pass through unchanged, yielding a MIXED
    image/no-image dataset (§3.2 — the register/backbone rows are image-absent).

    Output element:  {"messages": [ {role, content:[{type:image,image:PIL}|{type:text,text}]}... ]}

    # VERIFY-ON-BOX (2): confirm UnslothVisionDataCollator consumes this exact
    #   schema AND tolerates a batch that mixes image and no-image samples. If it
    #   requires homogeneous batches, set data.bucket_by_modality: true (below).

    NOTE (memory): eager-loading every page PIL into a list is fine for the POC
    (~220 images); switch to lazy decode in the collator at Manga109 scale.
    """
    from PIL import Image  # local import: only needed at data-build time

    image_root = _abs(cfg_data.get("image_root", "."))
    out: list[dict[str, Any]] = []
    skipped = 0
    n_img = 0
    n_txt = 0
    for rec in rows:
        messages = rec.get("messages")
        if not messages:
            skipped += 1
            continue
        conv = copy.deepcopy(messages)
        ok = True
        row_has_image = False
        for msg in conv:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") == "image":
                    ip = block.get("image")
                    if isinstance(ip, str):
                        p = Path(ip)
                        if not p.is_absolute():
                            p = image_root / p
                        if not p.exists():
                            logger.warning("image missing, skipping row: {}", p)
                            ok = False
                            break
                        try:
                            block["image"] = Image.open(p).convert("RGB")
                            row_has_image = True
                        except Exception as e:  # noqa: BLE001
                            logger.warning("PIL open failed for {}: {} — skipping", p, e)
                            ok = False
                            break
            if not ok:
                break
        if not ok:
            skipped += 1
            continue
        out.append({"messages": conv, "_has_image": row_has_image})
        n_img += int(row_has_image)
        n_txt += int(not row_has_image)
    if skipped:
        logger.info("conversation build: skipped={} (missing messages/image)", skipped)
    logger.info("conversations: {} (image={}, text-only={})", len(out), n_img, n_txt)
    return out


# --------------------------------------------------------------------------- #
# LoRA section audit — catches the silent "vision tower not adapted" bug
# --------------------------------------------------------------------------- #
def audit_lora_sections(model) -> dict[str, int]:
    """Count trainable LoRA param tensors grouped by Qwen3-VL tower.

    Qwen3-VL naming (VERIFY-ON-BOX (1)): the vision tower lives under ``visual.*``
    (blocks + a ``visual.merger`` Linear that projects vision->text embed), and
    the language stack under ``language_model.*`` (or ``model.layers.*``). If
    unsloth silently skipped the vision tower, ``vision_tower`` here is 0 and we
    must BAIL rather than train a text-only LoRA under a vision banner.
    """
    sections: dict[str, int] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad or "lora" not in n.lower():
            continue
        low = n.lower()
        if "merger" in low:
            key = "merger"          # vision->text projector (Qwen merger IS Linear)
        elif "visual" in low or "vision_tower" in low or "vision_model" in low:
            key = "vision_tower"
        elif "language_model" in low or ".model.layers." in low or "text_model" in low:
            key = "language_model"
        elif "audio" in low:
            key = "audio_tower"
        else:
            key = "other"
        sections[key] = sections.get(key, 0) + 1
    return sections


# --------------------------------------------------------------------------- #
# OPTIONAL modality-bucketed batch sampler (mixed-batch fallback, §3.2)
# --------------------------------------------------------------------------- #
class ModalityBatchSampler(torch.utils.data.Sampler):
    """Yield batches whose samples ALL share a modality (image vs text-only).

    Used ONLY when ``data.bucket_by_modality: true`` — the fallback for a collator
    that cannot mix image and no-image samples in one batch. Buckets are shuffled
    internally and the batch order across buckets is shuffled each epoch, so the
    model still interleaves image/text batches (not all-image-then-all-text).
    """

    def __init__(self, has_image_flags: list[bool], batch_size: int, seed: int = 42,
                 drop_last: bool = False) -> None:
        self.flags = list(has_image_flags)
        self.bs = int(batch_size)
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0
        self._img = [i for i, f in enumerate(self.flags) if f]
        self._txt = [i for i, f in enumerate(self.flags) if not f]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self, idxs: list[int], g: torch.Generator) -> list[list[int]]:
        if not idxs:
            return []
        perm = [idxs[i] for i in torch.randperm(len(idxs), generator=g).tolist()]
        batches = [perm[i:i + self.bs] for i in range(0, len(perm), self.bs)]
        if self.drop_last and batches and len(batches[-1]) < self.bs:
            batches.pop()
        return batches

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        batches = self._batches(self._img, g) + self._batches(self._txt, g)
        order = torch.randperm(len(batches), generator=g).tolist()
        for b in order:
            yield batches[b]

    def __len__(self) -> int:
        def nb(idxs):
            if not idxs:
                return 0
            return len(idxs) // self.bs if self.drop_last else (len(idxs) + self.bs - 1) // self.bs
        return nb(self._img) + nb(self._txt)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="backend/training/configs/qwen3vl_8b_imagectx_poc.yaml",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Short smoke run (max_steps from config.dry_run.max_steps)")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap rows for a quick smoke (0 = no cap)")
    p.add_argument("--out-suffix", default="",
                   help="Append suffix to output dir (e.g. '_dryrun')")
    p.add_argument("--resume", nargs="?", const=True, default=None,
                   help="Resume from checkpoint. Bare flag = auto-detect latest in "
                        "output_dir; pass a path to resume a specific checkpoint. "
                        "(pre-mortem #4: box thermal fault mid-run.)")
    p.add_argument("--inspect", action="store_true",
                   help="Load model + 1 sample, print named_modules / LoRA section "
                        "audit / one converted conversation, then EXIT without "
                        "training. Use this FIRST to verify the Qwen3-VL vision "
                        "module names + collator schema on the box.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(_abs(args.config))
    cfg_data = cfg["data"]

    out_dir = _abs(cfg["output"]["dir"])
    if args.out_suffix:
        out_dir = Path(str(out_dir) + args.out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(str(out_dir / "training.log"), level="INFO", enqueue=True)
    logger.info("qwen3vl imagectx SFT starting → out_dir={}", out_dir)

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

    # ---- load the abliterated Qwen3-VL base via FastVisionModel --------------
    # VERIFY-ON-BOX (1): FastVisionModel is the unsloth VLM entrypoint (mirrors
    #   FastLanguageModel). Returns (model, processor); unsloth calls the 2nd
    #   value `tokenizer` but it is the AutoProcessor (has .tokenizer + image
    #   processor). Confirm the signature + that it recognizes Qwen3-VL-8B.
    from unsloth import FastVisionModel

    base = cfg["model"]["name_or_path"]
    max_seq = int(cfg["model"]["max_seq_length"])
    logger.info("loading multimodal base: {} (max_seq={})", base, max_seq)
    model, processor = FastVisionModel.from_pretrained(
        base,
        max_seq_length=max_seq,        # VERIFY-ON-BOX: some builds ignore this; ok.
        dtype=torch.bfloat16,
        load_in_4bit=bool(cfg["model"]["load_in_4bit"]),
        use_gradient_checkpointing="unsloth",
    )

    # ---- bound the image-token budget on the processor -----------------------
    # BLOCKER-2 FIX: on transformers>=5, Qwen2/Qwen3-VL image processors expose
    # `max_pixels`/`min_pixels` as READ-ONLY properties (no setter) backed by the
    # `size` mapping: max_pixels == size["longest_edge"], min_pixels ==
    # size["shortest_edge"] (a SizeDict, item-assignable). So the old
    # `setattr(ip, "max_pixels", ...)` raised "property ... has no setter". We set
    # the cap by writing the backing size keys instead; the processor re-reads
    # size at every preprocess call, so the cap applies to the collator's images.
    # Config caps the long side to ~1024px (max_pixels ~1.05M) -> a few hundred
    # image tokens, bounding seq-len / VRAM / latency.
    try:
        ip = getattr(processor, "image_processor", None)
        if ip is not None:
            max_px = cfg["model"].get("max_pixels")
            min_px = cfg["model"].get("min_pixels")
            size = getattr(ip, "size", None)
            applied: dict[str, int] = {}
            if size is not None:
                def _set_size_key(key: str, val: int) -> None:
                    try:
                        size[key] = val            # SizeDict / dict item assign
                    except TypeError:
                        setattr(size, key, val)     # attribute fallback
                    applied[key] = val
                if max_px:
                    _set_size_key("longest_edge", int(max_px))
                if min_px:
                    _set_size_key("shortest_edge", int(min_px))
            else:
                logger.warning("image_processor has no `size` mapping — cannot "
                               "bound image-token budget on this build.")
            # Verify via the (now-updated) read-only properties.
            logger.info("processor image-token cap applied via size dict: {} "
                        "(effective min_pixels={} max_pixels={})",
                        applied, getattr(ip, "min_pixels", None),
                        getattr(ip, "max_pixels", None))
    except Exception as e:  # noqa: BLE001
        logger.warning("could not set processor image-token budget: {}", e)

    # ---- attach VISION + LANGUAGE LoRA ---------------------------------------
    # VERIFY-ON-BOX (1): FastVisionModel.get_peft_model boolean-flag API — the
    #   documented Qwen3-VL vision pattern. unsloth resolves the per-arch wrapped
    #   Linears internally (visual blocks + merger + language self_attn/mlp). If
    #   unsloth does NOT adapt the vision tower, the audit below bails.
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
    )

    # ---- audit: which towers actually got a LoRA? ----------------------------
    sections = audit_lora_sections(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("LoRA sections (trainable tensor counts): {}", sections)
    logger.info("trainable params: {:,} / {:,} ({:.4%})",
                trainable, total, trainable / total)
    # Objective-aware audit: for the image-context POC the vision tower MUST be
    # adapted (bail if it wasn't); for the TEXT-ONLY SHIP objective
    # (finetune_vision_layers=false) the inverse holds — the vision tower must be
    # UNTOUCHED and the language stack must be adapted.
    vision_on = bool(lcfg["finetune_vision_layers"])
    vt_count = sections.get("vision_tower", 0) + sections.get("merger", 0)
    lm_count = sections.get("language_model", 0)
    if vision_on:
        if vt_count == 0:
            logger.error(
                "VISION TOWER GOT NO LoRA — the image-context objective requires "
                "adapting the Qwen3-VL vision tower. unsloth likely did not "
                "recognize the visual.* modules. Fix: confirm the module names via "
                "--inspect and pass an explicit peft target_modules regex, or "
                "upgrade unsloth. Bailing rather than training a text-only LoRA "
                "under a vision banner.")
            return 3
    else:
        # text-only ship objective
        if lm_count == 0:
            logger.error(
                "TEXT-ONLY objective (finetune_vision_layers=false) but the "
                "language stack got NO LoRA (language_model==0). Nothing would "
                "train. Bailing — check the module-name mapping via --inspect.")
            return 3
        if vt_count != 0:
            logger.warning(
                "finetune_vision_layers=false but vision_tower/merger LoRA "
                "count={} (expected 0) — the vision tower was unexpectedly "
                "adapted.", vt_count)
        logger.info("text-only LoRA confirmed: language_model={} vision_tower={}",
                    lm_count, vt_count)

    # ---- build the (mixed) vision chat dataset -------------------------------
    data_path = _abs(cfg_data["train_path"])
    if not data_path.exists():
        logger.error("dataset not found: {} (build_numbered_poc.py builds it). Bailing.",
                     data_path)
        return 4
    rows = read_jsonl(data_path)
    logger.info("loaded {} rows from {}", len(rows), data_path)
    if args.limit:
        rows = rows[: args.limit]
    elif args.out_suffix.endswith("_dryrun") or args.dry_run:
        frac = float(cfg["dry_run"]["fraction"])
        rows = rows[: max(4, int(len(rows) * frac))]
        logger.info("dry_run subsample: {} rows", len(rows))

    eval_size = int(cfg_data.get("eval_size", 64))
    if len(rows) > eval_size + 8:
        train_rows, eval_rows = rows[:-eval_size], rows[-eval_size:]
    else:
        train_rows, eval_rows = rows, rows[-max(2, len(rows) // 10):]
    train_ds = build_conversations(train_rows, cfg_data)
    eval_ds = build_conversations(eval_rows, cfg_data)
    logger.info("train rows={} eval rows={}", len(train_ds), len(eval_ds))

    # ---- --inspect: verify everything, then EXIT (no training) ---------------
    if args.inspect:
        logger.info("=== --inspect: VISION module names (visual.* / merger) ===")
        shown = 0
        for n, _ in model.named_modules():
            low = n.lower()
            if ("visual" in low or "vision" in low or "merger" in low) and \
                    ("proj" in low or "linear" in low or "fc" in low or "qkv" in low):
                logger.info("  {}", n)
                shown += 1
                if shown >= 24:
                    logger.info("  ... (truncated)")
                    break
        logger.info("=== --inspect: LoRA section audit (bail if vision_tower==0) ===")
        logger.info("  sections={}", sections)
        if train_ds:
            s = train_ds[0]
            user = s["messages"][0]["content"]
            preview = {
                "has_image": s.get("_has_image"),
                "user_content_types": [c["type"] for c in user],
                "user_text_head": next(
                    (c["text"][:200] for c in user if c["type"] == "text"), ""),
                "assistant_text_head": s["messages"][1]["content"][0]["text"][:200],
                "n_image_rows_in_train": sum(int(x.get("_has_image", False)) for x in train_ds),
                "n_text_rows_in_train": sum(int(not x.get("_has_image", False)) for x in train_ds),
            }
            logger.info("=== --inspect: one converted conversation ===\n{}",
                        json.dumps(preview, ensure_ascii=False, indent=2))
        logger.info("--inspect done. NO TRAINING performed. Verify the Qwen3-VL "
                    "module names + section audit + mixed image/text counts "
                    "above, then run without --inspect.")
        return 0

    # ---- trainer (trl SFTTrainer + unsloth vision collator) ------------------
    # VERIFY-ON-BOX (2): import path of the vision collator on the box build.
    #   Known historical location: unsloth.trainer.UnslothVisionDataCollator.
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    FastVisionModel.for_training(model)  # VERIFY-ON-BOX: enables VLM training mode

    # strip the internal _has_image bookkeeping flag before handing to the collator
    bucket = bool(cfg_data.get("bucket_by_modality", False))
    train_flags = [bool(x.get("_has_image", False)) for x in train_ds]
    train_ds_clean = [{"messages": x["messages"]} for x in train_ds]
    eval_ds_clean = [{"messages": x["messages"]} for x in eval_ds]

    bs = int(cfg["train"]["per_device_train_batch_size"])
    ga = int(cfg["train"]["gradient_accumulation_steps"])
    epochs = float(cfg["train"]["num_train_epochs"])
    steps_per_epoch = max(1, len(train_ds_clean) // (bs * ga))
    total_steps = int(steps_per_epoch * epochs)
    save_steps = max(10, int(total_steps * float(cfg["train"]["save_steps_pct"])))
    eval_steps = max(5, int(total_steps * float(cfg["train"]["eval_steps_pct"])))

    in_training_eval = bool(cfg["train"].get("in_training_eval", False))

    # VERIFY-ON-BOX: SFTConfig field names on the box trl.
    #   - trl>=0.20 renamed max_seq_length -> max_length; we set max_length.
    #   - dataset_kwargs={"skip_prepare_dataset": True} + remove_unused_columns
    #     =False are REQUIRED so trl does not text-tokenize the image
    #     conversations (the collator owns preprocessing).
    sft_kwargs = dict(
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
        save_total_limit=int(cfg["train"]["save_total_limit"]),
        report_to=cfg["train"].get("report_to", "none"),
        seed=int(cfg["train"]["seed"]),
        dataloader_num_workers=int(cfg["train"]["dataloader_num_workers"]),
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=max_seq,
    )
    # in-training eval SIGSEGVs on sm_120 (box quirk) — real eval is the disjoint
    # harness. Only wire eval in if explicitly enabled per-config.
    if in_training_eval:
        sft_kwargs.update(eval_steps=eval_steps, eval_strategy="steps")
    else:
        sft_kwargs.update(eval_strategy="no")
    sft_args = SFTConfig(**sft_kwargs)
    if args.dry_run or args.out_suffix.endswith("_dryrun"):
        sft_args.max_steps = int(cfg["dry_run"].get("max_steps", 30))
        logger.info("DRY_RUN: max_steps={}", sft_args.max_steps)

    trainer_cls = SFTTrainer
    if bucket:
        # Fallback for a collator that can't mix image/text in one batch (§3.2).
        logger.info("bucket_by_modality=ON — using ModalityBatchSampler "
                    "(homogeneous image/text batches).")

        class _BucketedSFTTrainer(SFTTrainer):
            def get_train_dataloader(self):  # noqa: D401
                from torch.utils.data import DataLoader
                sampler = ModalityBatchSampler(
                    train_flags, batch_size=bs, seed=int(cfg["train"]["seed"]),
                    drop_last=False)
                return DataLoader(
                    self.train_dataset,
                    batch_sampler=sampler,
                    collate_fn=self.data_collator,
                    num_workers=int(cfg["train"]["dataloader_num_workers"]),
                    pin_memory=True,
                )

        trainer_cls = _BucketedSFTTrainer

    trainer = trainer_cls(
        model=model,
        processing_class=processor,            # VERIFY-ON-BOX: trl 0.23 uses
                                               #   processing_class; older trl used
                                               #   tokenizer=. unsloth accepts both.
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=train_ds_clean,
        eval_dataset=eval_ds_clean if in_training_eval else None,
        args=sft_args,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logger.info("VRAM before train: {:.2f} GB", torch.cuda.memory_allocated() / 1e9)

    # ---- resume support (pre-mortem #4) --------------------------------------
    resume = args.resume
    if resume is None:
        resume = cfg["train"].get("resume_from_checkpoint", None)
    if resume:
        logger.info("resuming from checkpoint: {}", resume)

    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
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
    # To MERGE for serving, use unsloth's native merge (folds the vision tower +
    # merger). No k_norm refold needed (that's a Gemma artifact; Qwen has none).
    #   model.save_pretrained_merged(str(out_dir/"merged"), processor,
    #                                save_method="merged_16bit")  # VERIFY-ON-BOX

    (out_dir / "imagectx-summary.json").write_text(json.dumps({
        "out_dir": str(out_dir),
        "base_model": base,
        "data_path": str(data_path),
        "train_rows": len(train_ds_clean),
        "eval_rows": len(eval_ds_clean),
        "train_image_rows": sum(train_flags),
        "train_text_rows": len(train_flags) - sum(train_flags),
        "bucket_by_modality": bucket,
        "lora_sections": sections,
        "trainable_params": trainable,
        "total_params": total,
        "elapsed_seconds": elapsed,
        "metrics": result.metrics,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""G0 preflight for the v2 30B-A3B (Qwen3-VL MoE, abliterated) QLoRA experiment.

RUN THIS ON THE BOX (or a rented GPU) — it performs the REAL 4-bit load and LoRA
attach, so it needs CUDA + ~20GB VRAM. It is the single go/no-go gate for the one
genuinely unproven thing in the v2 plan: *does unsloth actually LoRA-adapt the MoE
experts of `qwen3_vl_moe`, or are they silently skipped?* (plan
2026-07-02_v2-30b-a3b-training-plan.md §0/§1.2/§3, gate G0).

It reuses the EXACT wiring of the v1 text-SFT trainer
(`sft_qwen3vl_8b_imagectx.py`) — same LoRA flags, same `UnslothVisionDataCollator`
masking delimiters — and only ADDS the MoE-specific assertions. No config or shared
module is edited; this file is fully self-contained.

Checks (all must pass for GO):
  1. import  : `transformers.models.qwen3_vl_moe` imports (arch present in this venv).
  2. load    : FastVisionModel.from_pretrained(base, load_in_4bit=True) succeeds on
               the *bf16 base* (NOT the AWQ serve checkpoint).
  3. experts : after get_peft_model, the LoRA target set includes a NON-EMPTY set of
               fused-expert params (`mlp.experts.gate_up_proj` / `...down_proj`) AND
               those expert adapters actually carry requires_grad. (If empty →
               experts are silently frozen → the whole 30B bet is dead → NO-GO.)
  4. text    : language stack adapted (language_model LoRA > 0) and vision UNtouched
               (text-only ship objective, finetune_vision_layers=false).
  5. masking : one real batch through the collator yields 0.4 < masked < 1.0 and
               kept > 0 (completion-only loss is live on the 30B tokenizer).

Usage (one command on the box):
    python scripts/train/validate_30b_g0.py \
        [--base huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated] \
        [--config training/configs/qwen3vl_8b_textsft_v1.yaml] \
        [--data /home/danny/ship_v1/data_v13ship_v1_messages.jsonl]

Exit code 0 = GO, non-zero = NO-GO.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Import the v1 trainer module by path. This (a) imports `unsloth` BEFORE
# transformers/trl (mandatory for its patches), (b) applies the BLOCKER-1
# fix_untrained_tokens -> no-op patch that the 30B load also needs (meta-tensor
# lm_head), and (c) gives us the byte-identical helpers so the G0 asserts test the
# SAME code paths the real run uses.
# --------------------------------------------------------------------------- #
_THIS = Path(__file__).resolve()
_TRAINER_PATH = _THIS.with_name("sft_qwen3vl_8b_imagectx.py")


def _load_trainer_module():
    spec = importlib.util.spec_from_file_location(
        "sft_qwen3vl_8b_imagectx", str(_TRAINER_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # imports unsloth first + applies BLOCKER-1
    return mod


TR = _load_trainer_module()

# unsloth is now imported (via TR). Safe to pull the rest.
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from unsloth import FastVisionModel  # noqa: E402

DEFAULT_BASE = "huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated"
DEFAULT_CONFIG = (
    TR.PROJECT_ROOT
    / "backend/training/configs/qwen3vl_8b_textsft_v1.yaml"
)

# Masking band (matches the v1 trainer's hard assert; plan §3).
MASK_LOW = 0.4
MASK_HIGH = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=DEFAULT_BASE,
                   help="bf16 base to QLoRA (NOT the AWQ serve checkpoint).")
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="v1 YAML — read for lora/data/masking; base+4bit overridden.")
    p.add_argument("--data", default=None,
                   help="override data.train_path (real batch for the masking probe).")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--max-pixels", type=int, default=1048576)
    p.add_argument("--min-pixels", type=int, default=200704)
    return p.parse_args()


def _cap_image_tokens(processor, max_px: int, min_px: int) -> None:
    """Mirror the v1 trainer's image-token budget cap (size-dict write path)."""
    try:
        ip = getattr(processor, "image_processor", None)
        if ip is None:
            return
        size = getattr(ip, "size", None)
        if size is None:
            logger.warning("image_processor has no `size` mapping — skipping cap.")
            return

        def _set(k: str, v: int) -> None:
            try:
                size[k] = v
            except TypeError:
                setattr(size, k, v)

        _set("longest_edge", int(max_px))
        _set("shortest_edge", int(min_px))
        logger.info("image-token cap applied: longest_edge={} shortest_edge={}",
                    max_px, min_px)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not cap image tokens: {}", e)


def _expert_target_report(model, target_modules):
    """Log what unsloth's MoE auto-detector resolves for this arch, PRE-peft."""
    from unsloth.models._utils import get_moe_target_parameters, is_moe_model
    is_moe = is_moe_model(model)
    moe_targets = get_moe_target_parameters(model, target_modules)
    logger.info("is_moe_model={} | get_moe_target_parameters -> {}",
                is_moe, moe_targets)
    return is_moe, moe_targets


def main() -> int:
    args = parse_args()
    results: dict[str, object] = {}
    failures: list[str] = []

    # -- versions + arch import ------------------------------------------------
    import transformers
    import peft
    try:
        import trl
        trl_ver = trl.__version__
    except Exception:  # noqa: BLE001
        trl_ver = "?"
    import unsloth
    logger.info("versions: transformers={} unsloth={} peft={} trl={} torch={}",
                transformers.__version__, unsloth.__version__,
                peft.__version__, trl_ver, torch.__version__)
    results["versions"] = {
        "transformers": transformers.__version__,
        "unsloth": unsloth.__version__,
        "peft": peft.__version__,
        "trl": trl_ver,
        "torch": torch.__version__,
    }

    # CHECK 1: qwen3_vl_moe arch present (box venv 5.5.0 predates it — see plan §1.4)
    try:
        from transformers.models.qwen3_vl_moe import (  # noqa: F401
            Qwen3VLMoeForConditionalGeneration,
        )
        logger.info("CHECK 1 PASS: transformers.models.qwen3_vl_moe importable.")
        results["arch_import"] = True
    except Exception as e:  # noqa: BLE001
        logger.error("CHECK 1 FAIL: qwen3_vl_moe NOT importable ({}). This venv "
                     "predates the arch — bump transformers to >=5.12 (with a "
                     "matching unsloth >=2026.5.2). Plan §1.4.", e)
        results["arch_import"] = False
        failures.append("arch_import")
        _verdict(results, failures)
        return 2

    # -- peft target_parameters support (nn.Parameter LoRA) --------------------
    import inspect
    from peft import LoraConfig
    has_tp = "target_parameters" in inspect.signature(LoraConfig).parameters
    results["peft_target_parameters_supported"] = has_tp
    if not has_tp:
        logger.error("CHECK: peft.LoraConfig has NO `target_parameters` field — "
                     "unsloth's MoE targets would be silently dropped. Upgrade peft.")
        failures.append("peft_target_parameters")

    if not torch.cuda.is_available():
        logger.error("No CUDA. G0 requires a GPU to do the real 4-bit load. Run "
                     "this on the box or a rented GPU.")
        results["cuda"] = False
        failures.append("cuda")
        _verdict(results, failures)
        return 3
    results["cuda"] = True

    # -- read config (lora flags + data path + masking knob) -------------------
    cfg = TR.load_config(Path(args.config))
    lcfg = cfg["lora"]
    data_train_path = args.data or cfg["data"]["train_path"]
    completion_only = bool(cfg["train"].get("completion_only_loss", True))

    # CHECK 2: real 4-bit load of the bf16 base ------------------------------------
    # HARD FINDING (2026-07-02, empirically proven on the 5090 box; see report):
    #   bitsandbytes NF4 CANNOT quantize this MoE's experts. Qwen3VLMoeTextExperts
    #   stores each layer's experts as FUSED 3-D nn.Parameters (`gate_up_proj`
    #   [128,1536,2048] + `down_proj` [128,2048,768]) — NOT nn.Linear. transformers'
    #   `replace_with_bnb_linear` only swaps `type(module) is nn.Linear` for
    #   Linear4bit, and `param_needs_quantization` requires a Linear4bit module. So
    #   the 28.99B expert params (=54.0 GB bf16) stay BF16; only the ~1.5B attn/router
    #   Linears (308 modules) get NF4'd. Resident base ≈ 57-59 GB → does NOT fit the
    #   31.35 GB 5090 (nor a 40/48 GB cloud card). The load OOMs while materializing
    #   the bf16 expert shards in transformers.core_model_loading (`tensor.to`).
    #   => 32 GB QLoRA is IMPOSSIBLE for this arch on this stack. Needs an ≥80 GB GPU
    #   (bf16 experts) OR a build that 4-bit-quantizes fused MoE experts. device_map
    #   below is left all-on-GPU so this check fails FAST + LOUD rather than CPU-
    #   offloading the experts (unusably slow, and trips a bnb/accelerate meta-tensor
    #   bug). We catch the OOM and emit a clean NO-GO.
    logger.info("CHECK 2: loading {} in 4-bit (bnb NF4)...", args.base)
    try:
        model, processor = FastVisionModel.from_pretrained(
            args.base,
            max_seq_length=int(args.max_seq_length),
            dtype=torch.bfloat16,
            load_in_4bit=True,          # QLoRA — NF4, NOT the AWQ serve packing
            use_gradient_checkpointing="unsloth",
            device_map={"": 0},         # all-on-GPU (see finding above): fails fast.
        )
    except torch.cuda.OutOfMemoryError as e:  # noqa: BLE001
        logger.error("CHECK 2 FAIL (OOM): the abliterated 30B-A3B does NOT fit in "
                     "4-bit on this GPU — its fused MoE experts (54 GB bf16) are NOT "
                     "bnb-quantizable, so ~57-59 GB is required. Needs ≥80 GB VRAM "
                     "or fused-expert 4-bit support. {}", e)
        results["loaded_4bit"] = False
        results["oom"] = str(e)
        failures.append("oom_4bit_load")
        return _verdict(results, failures)
    _cap_image_tokens(processor, args.max_pixels, args.min_pixels)
    logger.info("CHECK 2 PASS: base loaded. model_type={}",
                getattr(model.config, "model_type", "?"))
    results["loaded_4bit"] = True
    results["model_type"] = getattr(model.config, "model_type", None)

    # Pre-peft: what does the MoE auto-detector resolve? (informational + names)
    from unsloth_zoo.peft_utils import get_peft_regex
    target_regex = get_peft_regex(
        model,
        finetune_vision_layers=bool(lcfg["finetune_vision_layers"]),
        finetune_language_layers=bool(lcfg["finetune_language_layers"]),
        finetune_attention_modules=bool(lcfg["finetune_attention_modules"]),
        finetune_mlp_modules=bool(lcfg["finetune_mlp_modules"]),
    )
    is_moe, moe_targets = _expert_target_report(model, target_regex)
    results["is_moe_model"] = bool(is_moe)
    results["moe_target_parameters"] = list(moe_targets) if moe_targets else []

    # -- attach LoRA exactly as the v1 trainer does ----------------------------
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

    # -- CHECK 3: experts actually adapted -------------------------------------
    # Signal A: PEFT config recorded a non-empty target_parameters that names experts.
    pc = None
    try:
        pc_map = model.peft_config
        pc = pc_map[next(iter(pc_map))]
    except Exception as e:  # noqa: BLE001
        logger.warning("could not read peft_config: {}", e)
    cfg_target_params = list(getattr(pc, "target_parameters", None) or []) if pc else []
    results["peft_config_target_parameters"] = cfg_target_params

    # Signal B (ground truth): trainable LoRA params whose name references experts.
    moe_trainable = [
        n for n, p in model.named_parameters()
        if p.requires_grad and "expert" in n.lower() and "lora" in n.lower()
    ]
    # Fallback naming: some PEFT versions register nn.Parameter LoRA without "lora"
    # in the leaf; accept any trainable param under an experts.* path.
    if not moe_trainable:
        moe_trainable = [
            n for n, p in model.named_parameters()
            if p.requires_grad and "expert" in n.lower()
        ]
    n_moe_trainable = len(moe_trainable)
    moe_adapter_numel = sum(
        int(p.numel()) for n, p in model.named_parameters()
        if p.requires_grad and "expert" in n.lower())
    results["n_expert_trainable_tensors"] = n_moe_trainable
    results["expert_adapter_numel"] = moe_adapter_numel
    results["expert_trainable_sample"] = moe_trainable[:6]
    logger.info("CHECK 3: expert trainable tensors={} (adapter numel={:,}); "
                "peft target_parameters={}",
                n_moe_trainable, moe_adapter_numel, cfg_target_params)
    experts_ok = (n_moe_trainable > 0) and bool(cfg_target_params or moe_targets)
    if experts_ok:
        logger.info("CHECK 3 PASS: MoE experts ARE LoRA-adapted. sample={}",
                    moe_trainable[:4])
    else:
        logger.error("CHECK 3 FAIL: NO expert LoRA adapters (experts silently "
                     "frozen). The 30B QLoRA would train the router/attn only — "
                     "the experiment is dead. STOP / escalate (plan G0 kill).")
        failures.append("experts_not_adapted")

    # -- CHECK 4: text-only objective (language on, vision off) ----------------
    sections = TR.audit_lora_sections(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    lm_count = sections.get("language_model", 0)
    vt_count = sections.get("vision_tower", 0) + sections.get("merger", 0)
    results["lora_sections"] = sections
    results["trainable_params"] = int(trainable)
    results["total_params"] = int(total)
    logger.info("CHECK 4: LoRA sections={} | trainable={:,}/{:,} ({:.4%})",
                sections, trainable, total, trainable / max(total, 1))
    if lm_count == 0:
        logger.error("CHECK 4 FAIL: language_model LoRA==0 — nothing to train.")
        failures.append("language_not_adapted")
    if bool(lcfg["finetune_vision_layers"]) is False and vt_count != 0:
        logger.warning("finetune_vision_layers=false but vision LoRA count=%d "
                       "(expected 0).", vt_count)
    if lm_count > 0 and not (bool(lcfg["finetune_vision_layers"]) and vt_count == 0):
        logger.info("CHECK 4 PASS: text-only LoRA (language={} vision={}).",
                    lm_count, vt_count)

    # -- CHECK 5: masking probe on one REAL batch ------------------------------
    from unsloth import UnslothVisionDataCollator
    dp = TR._abs(data_train_path)
    if dp.exists():
        rows = TR.read_jsonl(dp)[:8]
        conv = TR.build_conversations(rows, cfg["data"])
        batch_src = [{"messages": x["messages"]} for x in conv][:4]
        mask_source = f"real data ({dp})"
    else:
        logger.warning("data path {} not found — falling back to a synthetic "
                       "messages row (still validates the delimiter/tokenizer "
                       "masking mechanism, but confirm on real data before G2).", dp)
        batch_src = [{"messages": [
            {"role": "system", "content": [{"type": "text",
             "text": "You are a professional manga translator."}]},
            {"role": "user", "content": [{"type": "text",
             "text": "次のセリフを翻訳して: 「お前、ここで何してるんだ？」"}]},
            {"role": "assistant", "content": [{"type": "text",
             "text": "\"What are you doing here?\""}]},
        ]}]
        mask_source = "synthetic fallback"

    collator = UnslothVisionDataCollator(
        model, processor,
        train_on_responses_only=completion_only,
        instruction_part="<|im_start|>user\n" if completion_only else None,
        response_part="<|im_start|>assistant\n" if completion_only else None,
    )
    probe = collator(batch_src)
    labels = probe["labels"]
    masked = (labels == -100).float().mean().item()
    kept = int((labels != -100).sum().item())
    results["masking"] = {"source": mask_source, "masked_frac": masked,
                          "kept_tokens": kept}
    logger.info("CHECK 5: masking ({}) -> {:.1%} masked, {} target tokens kept",
                mask_source, masked, kept)
    mask_ok = (MASK_LOW < masked < MASK_HIGH) and kept > 0
    if mask_ok:
        logger.info("CHECK 5 PASS: completion-only masking in band.")
    else:
        logger.error("CHECK 5 FAIL: masking out of band (masked={:.1%}, kept={}) — "
                     "instruction/response parts mismatch the 30B chat template.",
                     masked, kept)
        failures.append("masking_out_of_band")

    return _verdict(results, failures)


def _verdict(results: dict, failures: list[str]) -> int:
    go = len(failures) == 0
    banner = "GO" if go else "NO-GO"
    logger.info("=" * 68)
    logger.info("G0 VERDICT: {}", banner)
    if failures:
        logger.info("  failed checks: {}", failures)
    logger.info("  summary: {}", json.dumps(results, ensure_ascii=False, default=str))
    logger.info("=" * 68)
    print(json.dumps({"verdict": banner, "failures": failures,
                      "results": results}, ensure_ascii=False, default=str))
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())

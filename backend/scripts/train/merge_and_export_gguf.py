"""Merge a trained LoRA adapter into its base model and export to GGUF.

Pipeline:
    1. Load base model + tokenizer (HF).
    2. Wrap with PEFT LoRA adapter; `merge_and_unload()` -> merged HF model.
    3. Save merged HF under --out.
    4. Call `python <llama-cpp-dir>/convert_hf_to_gguf.py <out> --outtype f16
       --outfile <out>/model.f16.gguf`.
    5. For each quant in --quants: call `<llama-cpp-dir>/llama-quantize` to
       produce `<out>/model.<QUANT>.gguf`.
    6. Print sha256 + size for every produced file.

--dry-run: do step (1-2) to validate shapes, DO NOT save, skip (3-5).

Usage:
    uv run --project backend python backend/scripts/train/merge_and_export_gguf.py \\
        --base Qwen/Qwen3-1.7B-Base \\
        --lora backend/training/runs/manga-bubbles/qwen3_1p7b_sft/final \\
        --out  backend/training/weights/qwen3-mt/ \\
        --quants Q4_K_M,Q8_0 \\
        --llama-cpp-dir /path/to/llama.cpp
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from loguru import logger


# --------------------------------------------------------------------------- #
# Merge                                                                       #
# --------------------------------------------------------------------------- #


def merge_lora(base: str, lora_dir: Path, out_dir: Path, dry_run: bool) -> None:
    """Load base + LoRA, merge, save merged HF model under out_dir."""
    try:
        import torch  # type: ignore[import-not-found]
        from peft import PeftModel  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "peft + transformers are required. Install via "
            "`uv add --project backend peft transformers`"
        ) from exc

    logger.info("loading base model: {}", base)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base)

    logger.info("attaching LoRA from: {}", lora_dir)
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA dir does not exist: {lora_dir}")
    peft_model = PeftModel.from_pretrained(model, str(lora_dir))

    logger.info("merging LoRA into base weights")
    merged = peft_model.merge_and_unload()

    # Sanity: the merged model should still be a causal LM with matching vocab.
    base_vocab = getattr(model.config, "vocab_size", None)
    merged_vocab = getattr(merged.config, "vocab_size", None)
    if base_vocab != merged_vocab:
        raise RuntimeError(
            f"vocab size drifted during merge: base={base_vocab} merged={merged_vocab}"
        )
    logger.info("merge ok (vocab_size={})", merged_vocab)

    if dry_run:
        logger.info("dry-run: skipping save of merged model")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("saving merged HF model to {}", out_dir)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))


# --------------------------------------------------------------------------- #
# GGUF convert + quantize                                                     #
# --------------------------------------------------------------------------- #


def convert_to_gguf_f16(llama_cpp_dir: Path, hf_dir: Path, out_file: Path) -> None:
    script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not script.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {script}")
    cmd = [
        sys.executable,
        str(script),
        str(hf_dir),
        "--outtype",
        "f16",
        "--outfile",
        str(out_file),
    ]
    logger.info("running: {}", " ".join(cmd))
    subprocess.run(cmd, check=True)


def quantize(llama_cpp_dir: Path, f16_file: Path, out_file: Path, quant: str) -> None:
    quant_bin = llama_cpp_dir / "llama-quantize"
    if not quant_bin.exists():
        # Try the older build location
        alt = llama_cpp_dir / "build" / "bin" / "llama-quantize"
        if alt.exists():
            quant_bin = alt
        else:
            raise FileNotFoundError(
                f"llama-quantize binary not found at {quant_bin} or {alt}"
            )
    cmd = [str(quant_bin), str(f16_file), str(out_file), quant]
    logger.info("running: {}", " ".join(cmd))
    subprocess.run(cmd, check=True)


def sha256_and_size(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_quants(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--quants cannot be empty")
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge LoRA into base + export to GGUF (f16 + quants).",
    )
    parser.add_argument(
        "--base",
        required=True,
        help="HF model id or local path of the base model (e.g. Qwen/Qwen3-1.7B-Base).",
    )
    parser.add_argument(
        "--lora",
        type=Path,
        required=True,
        help="Path to the trained LoRA adapter directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output dir for merged HF model + GGUF files.",
    )
    parser.add_argument(
        "--quants",
        type=parse_quants,
        default=["Q4_K_M", "Q8_0"],
        help="Comma-separated quant formats (default: Q4_K_M,Q8_0).",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=None,
        help="Path to cloned+built llama.cpp repo. Required unless --dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load + validate merge, but do not save or run GGUF steps.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.dry_run and args.llama_cpp_dir is None:
        parser.error("--llama-cpp-dir is required unless --dry-run is set")

    # Step (a): merge
    merge_lora(args.base, args.lora, args.out, dry_run=args.dry_run)

    if args.dry_run:
        logger.info("dry-run: skipping GGUF convert + quantize steps")
        return 0

    assert args.llama_cpp_dir is not None
    f16_path = args.out / "model.f16.gguf"

    # Step (b): HF -> GGUF f16
    convert_to_gguf_f16(args.llama_cpp_dir, args.out, f16_path)

    produced: list[Path] = [f16_path]

    # Step (c): quantize each target
    for quant in args.quants:
        q_path = args.out / f"model.{quant}.gguf"
        quantize(args.llama_cpp_dir, f16_path, q_path, quant)
        produced.append(q_path)

    # Step (d): report sha256 + size
    print("\n=== Produced GGUF artifacts ===")
    for p in produced:
        if not p.exists():
            logger.warning("expected output missing: {}", p)
            continue
        digest, size = sha256_and_size(p)
        print(f"{p.name}\tsize={size}\tsha256={digest}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

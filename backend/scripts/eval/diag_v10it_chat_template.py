"""DIAGNOSTIC: Run v10-it adapter with chat-template-correct prompts via Unsloth.

Tests whether the apparent v10-it brokenness is actually a prompt-format mismatch
in the bench script (which used raw -pt-style "Japanese: ...\\nEnglish:" against
a model trained with the -it chat template).

Outputs JSONL with same structure as bench_vllm_v10it.py + en_ref column.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import unsloth  # must come first  # noqa: F401
import torch
from loguru import logger


USER_TEMPLATE = (
    "Translate the following Japanese to English. Output only the translation.\n\n"
    "Japanese: {jp}"
)


def _ids(tok, s: str) -> list[int]:
    out = tok(text=s, add_special_tokens=False)["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def is_jp_passthrough(en: str, jp: str) -> bool:
    if not en:
        return False
    if en == jp:
        return True
    head = en[:20]
    return any(0x3040 <= ord(c) <= 0x309F or 0x30A0 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF for c in head)


def has_repetition(en: str) -> bool:
    if not en:
        return False
    # Detect 3+ consecutive identical word repeats e.g. "that that that"
    if re.search(r"\b(\w+)\b(?:\s+\1\b){2,}", en):
        return True
    # Detect 2+ identical multi-token phrase repeats e.g. "way-thing-way-thing"
    if re.search(r"(\w{3,}-\w{3,})-?\1", en):
        return True
    # Detect 5+ identical char repeats
    if re.search(r"(.)\1{6,}", en):
        return True
    # Detect 3+ identical 2-word phrase repeats
    if re.search(r"\b(\w+\s+\w+)(?:\s+\1){2,}", en):
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompts", default=str(Path(__file__).parent / "bench_prompts.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy. Use 0.2 to compare with v10it_sample_translate.")
    ap.add_argument("--label", default="v10it-chat-template-diag")
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if not adapter.exists():
        logger.error("adapter dir not found: {}", adapter)
        return 2
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts: list[dict[str, str]] = []
    with open(args.prompts, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append({
                "jp": row["jp"],
                "en_ref": row.get("en") or row.get("en_ref") or "",
                "src": row.get("src", ""),
            })
            if args.limit and len(prompts) >= args.limit:
                break
    logger.info("loaded {} prompts", len(prompts))

    # Load model+tokenizer via Unsloth
    from unsloth import FastLanguageModel
    logger.info("loading {} via Unsloth", adapter)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=1024,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    results = []
    quality = {"empty": 0, "jp_pass": 0, "rep": 0, "english": 0}
    use_sampling = args.temperature > 0.0
    for i, row in enumerate(prompts):
        jp = row["jp"]
        user_msg = USER_TEMPLATE.format(jp=jp)
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = _ids(tok, prompt_text)
        input_ids = torch.tensor([ids], device="cuda")
        attn = torch.ones_like(input_ids)

        t0 = time.time()
        with torch.inference_mode():
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            if use_sampling:
                gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=0.9, min_p=0.1)
            else:
                gen_kwargs.update(do_sample=False)
            out = model.generate(**gen_kwargs)
        elapsed = time.time() - t0

        gen = out[0, input_ids.shape[1]:]
        n_dec = int(gen.shape[0])
        raw = tok.decode(gen, skip_special_tokens=True).strip()
        # Trim at trailing chat markers if special-skip didn't catch
        for cut in ["<turn|>", "<|turn>", "Japanese:", "English:"]:
            j = raw.find(cut)
            if j >= 0:
                raw = raw[:j].strip()

        # Quality flags
        flags = []
        if not raw:
            flags.append("empty"); quality["empty"] += 1
        elif is_jp_passthrough(raw, jp):
            flags.append("jp_passthrough"); quality["jp_pass"] += 1
        elif has_repetition(raw):
            flags.append("repetition"); quality["rep"] += 1
        else:
            quality["english"] += 1

        rec = {
            "idx": i,
            "jp": jp,
            "en_pred": raw,
            "en_ref": row.get("en_ref", ""),
            "src": row.get("src", ""),
            "n_decoded_tokens": n_dec,
            "wall_seconds": round(elapsed, 4),
            "flags": flags,
        }
        results.append(rec)
        logger.info("[{}/{}] flags={} jp={} → {}", i+1, len(prompts), flags, jp[:30], raw[:60])

    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
    summary_path = out_path.with_suffix(".summary.json")
    summary = {
        "label": args.label,
        "adapter": str(adapter),
        "n_prompts": len(prompts),
        "temperature": args.temperature,
        "quality": quality,
        "quality_pct": {k: round(v / len(prompts) * 100, 1) for k, v in quality.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("=== {} ===", args.label)
    logger.info("quality: {}", summary["quality"])
    logger.info("wrote → {}", out_path)
    logger.info("summary → {}", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

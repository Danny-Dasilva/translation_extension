"""Batch-translate manga via an Unsloth-loaded LoRA adapter directly.

Avoids the broken merge path for Gemma 4 — load via FastLanguageModel.from_pretrained
on the adapter dir (Unsloth resolves base + adapter automatically), then generate.

Prompt format
-------------
Defaults to ``--prompt-format chat`` for Gemma-4-E4B-it adapters (e.g. v10-it):
the user message is wrapped via ``tok.apply_chat_template`` so the model sees
the same ``<start_of_turn>user/model`` framing it was trained with.

Pass ``--prompt-format raw`` to get the legacy v9c-era ``Translate ... Japanese: {jp}\\nEnglish:``
template — required for v9c (-pt base) where the model was trained on raw text.

Background
----------
The previous diagnostic showed v10-it produced 20/20 clean English with the
chat template applied (`diag_v10it_chat_template.py`), but only ~6/20 with the
raw v9c-era template. This script's prior `PROMPT` constant was that broken
raw template — switching to chat is the eval-script fix.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from loguru import logger

# Raw (v9c-era, -pt base) prompt: model sees a literal completion task.
RAW_PROMPT = (
    "Translate the following Japanese to English. Output only the translation.\n\n"
    "Japanese: {jp}\nEnglish:"
)

# Chat user message (v10-it, -it base) — the chat template wraps this with
# <start_of_turn>user / <start_of_turn>model markers. Note: NO trailing
# "\nEnglish:" — that suffix only makes sense in raw completion mode and
# corrupts the chat-template framing.
CHAT_USER_MSG = (
    "Translate the following Japanese to English. Output only the translation.\n\n"
    "Japanese: {jp}"
)

NEWLINE_RE = re.compile(r"[\r\n]")
NEXT_PROMPT_RE = re.compile(r"\s*(?:Japanese:|JP:|English:|EN:).*$", re.S)
TRAILING_PAREN_LOOP_RE = re.compile(r"(?:\s*[\(\[][^\)\]]{0,25}[\)\]]){2,}\s*$")
TRAILING_NOISE_RE = re.compile(r"(?:\s*[.!?\"'~_\-•・]+){4,}\s*$")
TRAILING_CHAR_REP_RE = re.compile(r"(?:\s*(\S))\s*(?:\1\s*){3,}$")
LONG_TOKEN_REPEAT_RE = re.compile(r"\b(\w{3,15}?)\1{3,}\w*\b", re.I)
REPEAT_PHRASE_RE = re.compile(r"(\b[^.!?]{3,80}[.!?]+)\s*(?:\1\s*)+", re.I)
REPEAT_NGRAM_RE = re.compile(r"(\b.{2,40}?\b)(?:\s*\1){1,}")


def clean(text: str) -> str:
    if not text:
        return ""
    text = NEWLINE_RE.split(text, 1)[0]
    text = NEXT_PROMPT_RE.sub("", text)
    text = TRAILING_PAREN_LOOP_RE.sub("", text)
    text = TRAILING_NOISE_RE.sub("", text)
    text = TRAILING_CHAR_REP_RE.sub("", text)
    text = LONG_TOKEN_REPEAT_RE.sub(r"\1", text)
    prev = None
    while prev != text:
        prev = text
        text = REPEAT_PHRASE_RE.sub(r"\1 ", text)
    prev = None
    while prev != text:
        prev = text
        text = REPEAT_NGRAM_RE.sub(r"\1", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"([!?])\1{3,}", r"\1\1\1", text)
    return text.strip()


def render_prompt(tok, jp: str, fmt: str) -> str:
    """Return the exact text fed to the tokenizer, per --prompt-format."""
    if fmt == "raw":
        return RAW_PROMPT.format(jp=jp)
    if fmt == "chat":
        user_msg = CHAT_USER_MSG.format(jp=jp)
        return tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(f"unknown prompt format: {fmt}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    help="Path to LoRA adapter dir (Unsloth FastLanguageModel will load base+adapter)")
    ap.add_argument("--src", default="/home/danny/manga-output/644289")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-format", choices=["chat", "raw"], default="chat",
                    help="chat = apply tokenizer chat template (v10-it / -it base, default). "
                         "raw  = legacy v9c-era 'Japanese: ...\\nEnglish:' completion prompt.")
    ap.add_argument("--print-rendered", action="store_true",
                    help="Print rendered prompt for first row and exit (sanity check, no GPU work).")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    src = Path(args.src)

    logger.info("loading via Unsloth: {}", args.adapter)
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=1024,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    logger.info("model+adapter loaded, peak VRAM {:.2f} GB", torch.cuda.max_memory_allocated() / 1e9)
    logger.info("prompt-format: {}", args.prompt_format)

    pages = sorted(p for p in src.iterdir() if p.is_dir() and p.name.isdigit())
    if args.limit:
        pages = pages[: args.limit]

    # Optional: print rendered prompt for first available bubble and exit early.
    if args.print_rendered:
        for page_dir in pages:
            stats_p = page_dir / "stats.json"
            if not stats_p.exists():
                continue
            ocr = json.loads(stats_p.read_text()).get("ocr_samples") or []
            if not ocr:
                continue
            rp = render_prompt(tok, ocr[0], args.prompt_format)
            print("=== rendered prompt (format={}) ===".format(args.prompt_format))
            print(rp)
            print("=== end ===")
            return 0
        logger.error("no bubbles found to render")
        return 2

    total_bubbles = 0
    total_time = 0.0
    t_all = time.time()
    for page_dir in pages:
        stats_p = page_dir / "stats.json"
        if not stats_p.exists():
            continue
        with open(stats_p) as f:
            stats = json.load(f)
        ocr = stats.get("ocr_samples") or []
        if not ocr:
            continue
        out_dir = out_root / page_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        raws, cleaneds = [], []
        for jp in ocr:
            prompt = render_prompt(tok, jp, args.prompt_format)
            # For chat format, the template already adds BOS via the tokenizer's
            # special-tokens handling — set add_special_tokens=False to avoid
            # double-BOS. For raw format, we want the default behavior (BOS).
            add_special = (args.prompt_format == "raw")
            enc = tok(text=prompt, return_tensors="pt", add_special_tokens=add_special).to("cuda")
            ids = enc["input_ids"]
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            elif ids.dim() == 3:
                ids = ids[0]  # squeeze nested
            attn = enc.get("attention_mask")
            if attn is None or attn.dim() != ids.dim():
                attn = torch.ones_like(ids)
            with torch.inference_mode():
                out = model.generate(
                    input_ids=ids, attention_mask=attn,
                    max_new_tokens=60,
                    do_sample=True, temperature=0.2, top_p=0.9, min_p=0.1,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            new = out[0, ids.shape[1]:]
            raw = tok.decode(new, skip_special_tokens=True).strip()
            # Trim residual chat-template markers if special-skip didn't catch them.
            if args.prompt_format == "chat":
                for cut in ["<turn|>", "<|turn>", "<start_of_turn>", "<end_of_turn>",
                            "Japanese:", "English:"]:
                    j = raw.find(cut)
                    if j >= 0:
                        raw = raw[:j].strip()
            raws.append(raw)
            cleaneds.append(clean(raw))
        elapsed = time.time() - t0

        with open(out_dir / "translations.txt", "w", encoding="utf-8") as f:
            f.write(f"# {stats.get('image', page_dir.name)}\n# {len(ocr)} bubble(s)\n\n")
            for i, (jp, en) in enumerate(zip(ocr, cleaneds), 1):
                f.write(f"[{i}]\n  JP: {jp}\n  EN: {en}\n\n")
        with open(out_dir / "raw_translations.txt", "w", encoding="utf-8") as f:
            for i, (jp, raw) in enumerate(zip(ocr, raws), 1):
                f.write(f"[{i}]\n  JP: {jp}\n  RAW: {raw}\n\n")
        with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump({
                "image": stats.get("image"),
                "num_bubbles": len(ocr),
                "translate_ms": elapsed * 1000,
                "model": args.adapter,
                "prompt_format": args.prompt_format,
                "ocr_samples": ocr,
                "translations_raw": raws,
                "translations": cleaneds,
            }, f, ensure_ascii=False, indent=2)
        total_bubbles += len(ocr)
        total_time += elapsed
        logger.info("[{}] {} bubbles in {:.1f}s ({:.0f}ms/bubble)",
                    page_dir.name, len(ocr), elapsed, elapsed/max(1,len(ocr))*1000)

    wall = time.time() - t_all
    logger.info("DONE: {} pages, {} bubbles in {:.1f}s ({:.0f}ms/bubble)",
                len(pages), total_bubbles, total_time, total_time/max(1,total_bubbles)*1000)
    print(f"DONE → {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Batch-translate 644289 with v9b (Gemma 4 E4B merged).

Same outputs as translate_manga_batch.py but handles Gemma 4's multimodal
processor (tokenizer returns nested input_ids; can't be batch-padded the same way).
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
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

PROMPT = "Translate the following Japanese to English. Output only the translation.\n\nJapanese: {jp}\nEnglish:"

# Re-use cleaning regexes from translate_manga_batch
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


def gemma_ids(tok, s: str) -> list[int]:
    out = tok(text=s, add_special_tokens=False)["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def translate_one(model, tok, jp: str, device: str) -> str:
    prompt = PROMPT.format(jp=jp)
    ids = gemma_ids(tok, prompt)
    input_ids = torch.tensor([ids], device=device)
    attn = torch.ones_like(input_ids)
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids, attention_mask=attn,
            do_sample=True, temperature=0.2, top_p=0.9, min_p=0.1,
            max_new_tokens=60,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    gen = out[0, input_ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="backend/training/weights/gemma4-e4b-v9b-merged")
    ap.add_argument("--src", default="/home/danny/manga-output/644289")
    ap.add_argument("--out-dir", default="/home/danny/manga-output/644289-gemma4-v9b")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    src = Path(args.src)

    device = "cuda"
    logger.info("loading {}", args.model)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    logger.info("model loaded, peak VRAM {:.2f} GB", torch.cuda.max_memory_allocated()/1e9)

    pages = sorted(p for p in src.iterdir() if p.is_dir() and p.name.isdigit())
    if args.limit:
        pages = pages[: args.limit]

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
            raw = translate_one(model, tok, jp, device)
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
                "model": "gemma4-e4b-v9b",
                "ocr_samples": ocr,
                "translations_raw": raws,
                "translations": cleaneds,
            }, f, ensure_ascii=False, indent=2)
        total_bubbles += len(ocr)
        total_time += elapsed
        logger.info("[{}] {} bubbles in {:.1f}s ({:.0f}ms/bubble)",
                    page_dir.name, len(ocr), elapsed, elapsed/max(1,len(ocr))*1000)

    wall = time.time() - t_all
    logger.info("DONE: {} pages, {} bubbles in {:.1f}s ({:.0f}ms/bubble avg)",
                len(pages), total_bubbles, total_time, total_time/max(1,total_bubbles)*1000)
    print(f"DONE → {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

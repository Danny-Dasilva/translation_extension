"""Batch-translate manga pages using the fine-tuned Qwen3-mt merged HF weights on GPU.

Reads OCR'd JP bubbles from /home/danny/manga-output/644289/NNN/stats.json
and produces translations via transformers on GPU (llama-cpp-python in this
venv is CPU-only). Applies production sampling + post-processing to handle
the base-model no-EOS looping.
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

DEFAULT_MODEL = "backend/training/weights/qwen3-mt-merged"
DEFAULT_SRC = "/home/danny/manga-output/644289"
DEFAULT_OUT = "/home/danny/manga-output/644289-qwen3mt"

PROMPT_TEMPLATE = (
    "Translate the following Japanese to English. "
    "Output only the translation.\n\n"
    "Japanese: {jp}\n"
    "English:"
)

# Post-processing heuristics for loop-suffix garbage.
NEWLINE_RE = re.compile(r"[\r\n]")
SPEAKER_TAG_RE = re.compile(r"\s+(?=[\(\[][A-Za-z][^\)\]]{0,20}[\)\]]:?\s)")
# Phrase-repeat: any sentence-like fragment that appears back-to-back (>=1 repeat).
REPEAT_PHRASE_RE = re.compile(r"(\b[^.!?]{3,80}[.!?]+)\s*(?:\1\s*)+", re.I)
# Word/n-gram repeat without punctuation (e.g. "yeah yeah yeah").
REPEAT_NGRAM_RE = re.compile(r"(\b.{2,40}?\b)(?:\s*\1){1,}")
TRAILING_PAREN_LOOP_RE = re.compile(r"(?:\s*[\(\[][^\)\]]{0,25}[\)\]]){2,}\s*$")
NEXT_PROMPT_RE = re.compile(r"\s*(?:Japanese:|JP:|English:|EN:).*$", re.S)
# Long punctuation/symbol runs like ". . . . . . . . .", "!!!!!!!", "~~~~", "___", etc.
TRAILING_NOISE_RE = re.compile(r"(?:\s*[.!?\"'~_\-•・]+){4,}\s*$")
# Trailing single-char reps like ") ) ) ) ) )" or "( ( ( (" or "※ ※ ※".
TRAILING_CHAR_REP_RE = re.compile(r"(?:\s*(\S))\s*(?:\1\s*){3,}$")
# Same character repeated 5+ times in any latin-token (e.g. "chinchombochinchombo...").
LONG_TOKEN_REPEAT_RE = re.compile(r"\b(\w{3,15}?)\1{3,}\w*\b", re.I)
# Collapse "....." (4+ dots) back to "..."
MANY_DOTS_RE = re.compile(r"\.{4,}")
MANY_BANGS_RE = re.compile(r"([!?])\1{3,}")


def clean(text: str) -> str:
    if not text:
        return ""
    text = NEWLINE_RE.split(text, 1)[0]
    text = NEXT_PROMPT_RE.sub("", text)
    text = TRAILING_PAREN_LOOP_RE.sub("", text)
    text = TRAILING_NOISE_RE.sub("", text)
    text = TRAILING_CHAR_REP_RE.sub("", text)
    # Collapse intra-token repeats like "chinchombochinchombo..." -> "chinchombo"
    text = LONG_TOKEN_REPEAT_RE.sub(r"\1", text)
    parts = SPEAKER_TAG_RE.split(text)
    if len(parts) > 2:
        text = parts[0] + parts[1] if parts[1].startswith(("(", "[")) else parts[0]
    # Collapse repeated phrases (with punctuation)
    prev = None
    while prev != text:
        prev = text
        text = REPEAT_PHRASE_RE.sub(r"\1 ", text)
    # Collapse raw n-gram loops
    prev = None
    while prev != text:
        prev = text
        text = REPEAT_NGRAM_RE.sub(r"\1", text)
    # Collapse over-long punctuation runs
    text = MANY_DOTS_RE.sub("...", text)
    text = MANY_BANGS_RE.sub(r"\1\1\1", text)
    # Final trailing-noise sweep
    text = TRAILING_NOISE_RE.sub("", text)
    return text.strip()


class StopOnSubstring(StoppingCriteria):
    """Stop when ALL sequences in batch have hit any stop token sequence."""

    def __init__(self, stop_token_lists: list[list[int]], prompt_lens: list[int]):
        self.stops = stop_token_lists
        self.prompt_lens = prompt_lens

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # type: ignore[override]
        bs, total_len = input_ids.shape
        for b in range(bs):
            gen = input_ids[b, self.prompt_lens[b]:].tolist()
            if not any(self._has_stop(gen, s) for s in self.stops):
                return False
        return True

    @staticmethod
    def _has_stop(gen: list[int], stop: list[int]) -> bool:
        if not stop or len(gen) < len(stop):
            return False
        for i in range(len(gen) - len(stop) + 1):
            if gen[i : i + len(stop)] == stop:
                return True
        return False


def translate_batch(model, tok, jp_list: list[str], *, device: str) -> tuple[list[str], list[str]]:
    """Translate a page's bubbles in one padded batch. Returns (raw, cleaned)."""
    prompts = [PROMPT_TEMPLATE.format(jp=jp) for jp in jp_list]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    stop_strings = ["\n\n", "\nJapanese:", "\nEnglish:"]
    stop_token_lists = [tok(s, add_special_tokens=False).input_ids for s in stop_strings]
    prompt_lens = [int(enc["attention_mask"][i].sum().item()) for i in range(enc["input_ids"].shape[0])]
    # Prompt lens in padded coords is trickier — for left-padded, actual prompt ends at seq_len.
    # Use seq_len (== input_ids shape 1) as the generation-start marker for all rows (left pad).
    start_len = enc["input_ids"].shape[1]
    stopping = StoppingCriteriaList([StopOnSubstring(stop_token_lists, [start_len] * enc["input_ids"].shape[0])])

    with torch.inference_mode():
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            min_p=0.1,
            max_new_tokens=60,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            stopping_criteria=stopping,
        )
    raws: list[str] = []
    for j in range(out.shape[0]):
        gen = out[j, start_len:]
        text = tok.decode(gen, skip_special_tokens=True).strip()
        raws.append(text)
    cleaneds = [clean(r) for r in raws]
    return raws, cleaneds


def translate_page(model, tok, device: str, page_dir: Path, out_dir: Path) -> dict:
    stats_path = page_dir / "stats.json"
    if not stats_path.exists():
        return {"skipped": True}
    with open(stats_path) as f:
        stats = json.load(f)
    ocr = stats.get("ocr_samples") or stats.get("ocr_all") or []
    if not ocr:
        return {"skipped": True}

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    raws, cleaneds = translate_batch(model, tok, ocr, device=device)
    elapsed = time.time() - t0

    trans_path = out_dir / "translations.txt"
    with open(trans_path, "w", encoding="utf-8") as f:
        f.write(f"# {stats.get('image', page_dir.name)}\n")
        f.write(f"# {len(ocr)} bubble(s)\n\n")
        for i, (jp, en) in enumerate(zip(ocr, cleaneds), start=1):
            f.write(f"[{i}]\n  JP: {jp}\n  EN: {en}\n\n")

    with open(out_dir / "raw_translations.txt", "w", encoding="utf-8") as f:
        for i, (jp, raw) in enumerate(zip(ocr, raws), start=1):
            f.write(f"[{i}]\n  JP: {jp}\n  RAW: {raw}\n\n")

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "image": stats.get("image"),
                "num_bubbles": len(ocr),
                "translate_ms": elapsed * 1000,
                "model": "qwen3-mt-1p7b-merged-hf",
                "ocr_samples": ocr,
                "translations_raw": raws,
                "translations": cleaneds,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "[{}] {} bubbles in {:.2f}s ({:.0f}ms/bubble)",
        page_dir.name,
        len(ocr),
        elapsed,
        elapsed / max(1, len(ocr)) * 1000,
    )
    return {"n": len(ocr), "elapsed_s": elapsed}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    src = Path(args.src)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading {} on {}...", args.model, device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    logger.info("model loaded")

    pages = sorted(p for p in src.iterdir() if p.is_dir() and p.name.isdigit())
    if args.limit:
        pages = pages[: args.limit]

    total_bubbles = 0
    total_time = 0.0
    t_all = time.time()
    for page_dir in pages:
        rec = translate_page(model, tok, device, page_dir, out_root / page_dir.name)
        if not rec.get("skipped"):
            total_bubbles += rec["n"]
            total_time += rec["elapsed_s"]

    logger.info(
        "DONE: {} pages, {} bubbles in {:.1f}s compute ({:.0f}ms/bubble). Wall={:.1f}s",
        len(pages),
        total_bubbles,
        total_time,
        total_time / max(1, total_bubbles) * 1000,
        time.time() - t_all,
    )

    with open(out_root / "batch_summary.json", "w") as f:
        json.dump(
            {
                "pages": len(pages),
                "total_bubbles": total_bubbles,
                "compute_s": total_time,
                "wall_s": time.time() - t_all,
                "avg_ms_per_bubble": total_time / max(1, total_bubbles) * 1000,
                "model": args.model,
                "device": device,
            },
            f,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

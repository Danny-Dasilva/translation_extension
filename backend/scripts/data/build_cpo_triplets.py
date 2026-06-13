"""Build CPO triplets for v8: chosen=Gemma 3 4B IT, rejected=v7's translation.

Stratified-sample JP source sentences from v7 training data, run both Gemma
teacher and v7 student over them, keep pairs where they meaningfully diverge.

Output: backend/training/datasets/filtered/cpo_triplets.parquet with columns:
  jp, chosen (gemma_en), rejected (v7_en), gemma_score, v7_score, src

Two-pass design:
  Phase 1: gen Gemma + v7 outputs, save to JSONL (resumable)
  Phase 2: filter to divergent pairs, write final parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

PROMPT_TEMPLATE = (
    "Translate the following Japanese to English. "
    "Output only the translation.\n\nJapanese: {jp}\nEnglish:"
)

GEMMA_PROMPT = (
    "Translate the following Japanese manga line to natural English. "
    "Output only the translation, no notes.\n\n{jp}"
)


class StopOnSubstring(StoppingCriteria):
    def __init__(self, stop_ids: list[list[int]], start_len: int):
        self.stop_ids = stop_ids
        self.start = start_len

    def __call__(self, input_ids, scores, **kw):
        for b in range(input_ids.shape[0]):
            gen = input_ids[b, self.start:].tolist()
            if not any(self._has(gen, s) for s in self.stop_ids):
                return False
        return True

    @staticmethod
    def _has(gen, s):
        if not s or len(gen) < len(s):
            return False
        for i in range(len(gen) - len(s) + 1):
            if gen[i:i+len(s)] == s:
                return True
        return False


def stratified_sample(df: pl.DataFrame, n: int, seed: int = 42) -> pl.DataFrame:
    """Take n samples stratified by register_tag, weighted by sqrt(group size)."""
    groups = df.group_by("register_tag").agg(pl.len()).sort("len", descending=True)
    total = groups["len"].sum()
    parts = []
    for row in groups.iter_rows(named=True):
        tag = row["register_tag"]
        # sqrt-weight to prevent vn_eroge from dominating
        target = int(n * (row["len"] ** 0.5) / sum(g ** 0.5 for g in groups["len"].to_list()))
        target = min(target, row["len"])
        sub = df.filter(pl.col("register_tag") == tag).sample(n=target, seed=seed)
        parts.append(sub)
    return pl.concat(parts)


def gen_v7(jps: list[str], model_path: str, batch_size: int, device: str) -> list[str]:
    logger.info("loading v7 student from {}", model_path)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device)
    model.eval()

    stop_strings = ["\n\n", "\nJapanese:", "\nEnglish:"]
    stop_ids = [tok(s, add_special_tokens=False).input_ids for s in stop_strings]

    out = []
    for i in range(0, len(jps), batch_size):
        batch = jps[i:i+batch_size]
        prompts = [PROMPT_TEMPLATE.format(jp=jp) for jp in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        start = enc["input_ids"].shape[1]
        stopping = StoppingCriteriaList([StopOnSubstring(stop_ids, start)])
        with torch.inference_mode():
            o = model.generate(
                **enc, do_sample=True, temperature=0.2, top_p=0.9, min_p=0.1,
                max_new_tokens=80,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                stopping_criteria=stopping,
            )
        for j in range(o.shape[0]):
            gen = o[j, start:]
            out.append(tok.decode(gen, skip_special_tokens=True).strip())
        if (i // batch_size) % 10 == 0:
            logger.info("v7: {}/{}", min(i+batch_size, len(jps)), len(jps))
    del model
    torch.cuda.empty_cache()
    return out


def gen_gemma(jps: list[str], gemma_repo: str, batch_size: int, device: str) -> list[str]:
    """Run Gemma 4 31B IT via HF transformers + bitsandbytes 4-bit (~16 GB on 5090).

    Greedy decoding for teacher consistency.
    """
    from transformers import BitsAndBytesConfig

    logger.info("loading Gemma teacher (4-bit) from {}", gemma_repo)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(gemma_repo, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        gemma_repo,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # safer for Gemma's hybrid attention
    )
    model.eval()

    out = []
    t0 = time.time()
    for i in range(0, len(jps), batch_size):
        batch = jps[i:i+batch_size]
        msgs_batch = [
            [{"role": "user", "content": GEMMA_PROMPT.format(jp=jp)}]
            for jp in batch
        ]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        start = enc["input_ids"].shape[1]
        with torch.inference_mode():
            o = model.generate(
                **enc, do_sample=False,
                max_new_tokens=120,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        for j in range(o.shape[0]):
            gen = o[j, start:]
            text = tok.decode(gen, skip_special_tokens=True).strip()
            for pfx in ("English:", "Translation:", "EN:"):
                if text.lower().startswith(pfx.lower()):
                    text = text[len(pfx):].strip()
            text = text.split("\n\n")[0].strip()
            out.append(text)
        if (i // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            done = min(i + batch_size, len(jps))
            rate = done / max(0.1, elapsed)
            eta = (len(jps) - done) / max(0.1, rate)
            logger.info("gemma: {}/{} ({:.1f}/s, eta {:.0f}s)", done, len(jps), rate, eta)
    del model
    torch.cuda.empty_cache()
    return out


def is_divergent(chosen: str, rejected: str, jp: str) -> bool:
    """Heuristic: keep triplet if Gemma (chosen) is meaningfully better than v7 (rejected).

    Critical: if v7 correctly refused (rejected="...") on what looks like garbage JP,
    DO NOT train it to reverse that — drop the triplet.
    """
    if not chosen or not rejected:
        return False
    c = chosen.strip().lower()
    r = rejected.strip().lower()
    if c == r:
        return False
    # If v7 refused with "..." and Gemma generated something on noisy JP,
    # v7's refusal is preferable — don't train it away.
    if r in ("...", "…") and c not in ("...", "…"):
        # Check if JP looks like noise (low Japanese-char ratio)
        import re
        ja_chars = re.findall(r"[぀-ヿ一-鿿]", jp)
        if len(ja_chars) / max(1, len(jp)) < 0.5:
            return False  # Trust v7's refusal
    # If Gemma refused but v7 translated, Gemma is conservative — train v7 to refuse too
    if c in ("...", "…") and r not in ("...", "…"):
        return True
    # length differs by >40% -> meaningful
    len_ratio = len(c) / max(1, len(r))
    if len_ratio < 0.6 or len_ratio > 1.6:
        return True
    # char-level diff > 30%
    import difflib
    sim = difflib.SequenceMatcher(None, c, r).ratio()
    if sim < 0.7:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="backend/training/runs/manga-bubbles/data_v7.parquet")
    ap.add_argument("--gemma", default="",
                    help="Path to Gemma GGUF (Q4_K_M). Auto-downloads if missing.")
    ap.add_argument("--gemma-repo", default="unsloth/gemma-4-31B-it-GGUF",
                    help="HF repo for Gemma teacher GGUF (used if --gemma not set)")
    ap.add_argument("--gemma-quant", default="Q4_K_M")
    ap.add_argument("--v7", default="backend/training/weights/qwen3-mt-v7-merged")
    ap.add_argument("--out", default="backend/training/datasets/filtered/cpo_triplets.parquet")
    ap.add_argument("--cache", default="backend/training/datasets/filtered/cpo_phase1.jsonl")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1: sampling + generation
    if cache_path.exists():
        logger.info("loading phase1 cache from {}", cache_path)
        rows = [json.loads(l) for l in open(cache_path) if l.strip()]
    else:
        logger.info("sampling {} JP from {} stratified by register", args.n, args.source)
        df = pl.read_parquet(args.source)
        sample = stratified_sample(df, args.n, args.seed)
        # Dedup JP (we may have oversampled gold)
        seen = set()
        rows = []
        for r in sample.iter_rows(named=True):
            jp = (r.get("jp") or "").strip()
            if not jp or jp in seen:
                continue
            seen.add(jp)
            rows.append({"jp": jp, "src": r.get("src", "")})
        logger.info("unique JP samples: {}", len(rows))

        # Use HF transformers repo directly (Gemma 4 31B IT, 4-bit quant)
        gemma_repo = args.gemma if args.gemma else "google/gemma-4-31b-it"
        logger.info("gemma repo: {}", gemma_repo)

        jps = [r["jp"] for r in rows]
        t0 = time.time()
        v7_outs = gen_v7(jps, args.v7, args.batch_size, device)
        logger.info("v7 gen done in {:.1f}s", time.time()-t0)
        t0 = time.time()
        gemma_outs = gen_gemma(jps, gemma_repo, batch_size=4, device=device)
        logger.info("gemma gen done in {:.1f}s", time.time()-t0)

        for r, v, g in zip(rows, v7_outs, gemma_outs):
            r["v7"] = v
            r["gemma"] = g

        with open(cache_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("phase1 cache saved -> {}", cache_path)

    # Phase 2: filter divergent triplets
    triplets = []
    n_div = 0
    n_total = 0
    for r in rows:
        n_total += 1
        chosen = r["gemma"]
        rejected = r["v7"]
        if is_divergent(chosen, rejected, r["jp"]):
            n_div += 1
            triplets.append({
                "prompt": PROMPT_TEMPLATE.format(jp=r["jp"]) + " ",
                "chosen": chosen,
                "rejected": rejected,
                "jp": r["jp"],
                "src": r.get("src", ""),
            })

    logger.info("kept {}/{} divergent triplets ({:.1f}%)",
                n_div, n_total, 100 * n_div / max(1, n_total))
    out_df = pl.DataFrame(triplets)
    out_df.write_parquet(args.out)
    logger.info("wrote {} triplets -> {}", len(out_df), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

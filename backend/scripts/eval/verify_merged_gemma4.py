"""Verify a cleanly-merged Gemma 4 + v9c LoRA model produces sensible output.

What this checks
----------------
1. Loads the merged model (plain HF ``Gemma4ForConditionalGeneration``,
   no Unsloth) on GPU and runs a handful of Japanese -> English manga
   prompts using the v9c training prompt format.
2. Optionally loads the **unmerged** adapter via Unsloth's
   ``FastLanguageModel.from_pretrained`` (the path that's known to work
   for inference) and runs the same prompts. The two outputs should be
   essentially identical (greedy / low-temp).
3. Sanity-checks the merged outputs:
   * Mostly ASCII / Latin (no Cyrillic / Korean / TeX-style garbage)
   * Contains at least one English-letter token

Exit codes
----------
0 - merged model output looks clean
1 - merged model output appears corrupted (script will dump samples)

Usage
-----

    uv run python backend/scripts/eval/verify_merged_gemma4.py \\
        --merged backend/training/runs/manga-bubbles/gemma4_e4b_v9c/merged_clean \\
        --adapter backend/training/runs/manga-bubbles/gemma4_e4b_v9c/final
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import torch
from loguru import logger


PROMPT = (
    "Translate the following Japanese to English. Output only the translation."
    "\n\nJapanese: {jp}\nEnglish:"
)

# Five short, varied manga-style Japanese lines.
DEFAULT_PROMPTS: list[str] = [
    "おはよう！今日もがんばろう。",
    "なんでこんなことになったんだ……",
    "やめろ！それ以上近づくな！",
    "ありがとう、本当に助かったよ。",
    "ふふ、面白くなってきたじゃない。",
]


# -- output sanity checks ----------------------------------------------------

# Anything outside Basic Latin + general punctuation is suspect for an
# English manga translation. We allow a few specific classes (curly quotes,
# em-dashes) but flag Cyrillic / Hangul / CJK / TeX-ish backslashes.
_BAD_SCRIPTS = ("CYRILLIC", "HANGUL", "GEORGIAN", "ARMENIAN", "DEVANAGARI",
                "ARABIC", "HEBREW", "GREEK", "THAI")
_TEX_RE = re.compile(r"\\[a-zA-Z]{2,}|\\\\|\$\$")


def looks_corrupt(text: str) -> tuple[bool, str]:
    """Return (is_corrupt, reason). Empty is also corrupt."""
    if not text or not text.strip():
        return True, "empty"
    if _TEX_RE.search(text):
        return True, "tex-like backslash sequences"
    bad_chars = 0
    cjk_chars = 0
    total_letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        total_letters += 1
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            continue
        if any(s in name for s in _BAD_SCRIPTS):
            bad_chars += 1
        if "CJK UNIFIED" in name or "HIRAGANA" in name or "KATAKANA" in name:
            cjk_chars += 1
    if total_letters == 0:
        return True, "no letters at all"
    if bad_chars / max(total_letters, 1) > 0.10:
        return True, f"{bad_chars}/{total_letters} chars in non-Latin scripts"
    # CJK leakage in English output is suspicious past a small threshold —
    # but Japanese terms / SFX may still appear. Allow up to 30%.
    if cjk_chars / max(total_letters, 1) > 0.30:
        return True, f"{cjk_chars}/{total_letters} CJK letters (likely untranslated)"
    return False, ""


# -- generation helpers ------------------------------------------------------

def _tok_ids(tok, prompt: str) -> list[int]:
    """Gemma 4 processor returns nested input_ids; flatten."""
    out = tok(text=prompt, add_special_tokens=False)["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def _gen(model, tok, prompt: str, device: str, *, deterministic: bool) -> str:
    ids = _tok_ids(tok, prompt)
    input_ids = torch.tensor([ids], device=device)
    attn = torch.ones_like(input_ids)
    kwargs: dict = dict(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=60,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    if deterministic:
        kwargs["do_sample"] = False
    else:
        kwargs.update(do_sample=True, temperature=0.2, top_p=0.9, min_p=0.1)
    with torch.inference_mode():
        out = model.generate(**kwargs)
    new = out[0, input_ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip()


# -- merged loader (plain HF, no Unsloth) ------------------------------------

def load_merged(merged_dir: Path, device: str):
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(str(merged_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = Gemma4ForConditionalGeneration.from_pretrained(
        str(merged_dir),
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model, tok


# -- adapter loader (Unsloth -- known-good path for inference) --------------

def load_unmerged(adapter_dir: Path):
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=1024,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


# -- main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", required=True, type=Path,
                    help="Path to merged-clean model dir (output of merge_gemma4_lora_clean.py)")
    ap.add_argument("--adapter", default=None, type=Path,
                    help="Path to original LoRA adapter dir (optional; for Unsloth comparison)")
    ap.add_argument("--prompts", default=None, type=Path,
                    help="Optional JSON file with a list of Japanese prompt strings")
    ap.add_argument("--no-compare", action="store_true",
                    help="Skip the Unsloth-loaded comparison (faster; merged-only sanity)")
    ap.add_argument("--deterministic", action="store_true", default=True,
                    help="Use greedy decoding so merged vs unmerged outputs match exactly")
    args = ap.parse_args()

    if args.prompts is not None and args.prompts.exists():
        prompts = json.loads(args.prompts.read_text())
        assert isinstance(prompts, list) and all(isinstance(p, str) for p in prompts)
    else:
        prompts = list(DEFAULT_PROMPTS)
    logger.info("verifying with {} prompt(s)", len(prompts))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Merged-clean inference
    logger.info("=== loading MERGED-CLEAN model ===")
    t0 = time.time()
    m_model, m_tok = load_merged(args.merged, device)
    logger.info("merged loaded in {:.1f}s", time.time() - t0)

    merged_outs: list[str] = []
    for jp in prompts:
        full = PROMPT.format(jp=jp)
        out = _gen(m_model, m_tok, full, device, deterministic=args.deterministic)
        merged_outs.append(out)
        logger.info("[merged] JP={!r} -> EN={!r}", jp, out)

    # Free GPU before loading second model
    del m_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 2. Sanity check merged output
    bad: list[tuple[int, str, str]] = []
    for i, out in enumerate(merged_outs):
        is_bad, reason = looks_corrupt(out)
        if is_bad:
            bad.append((i, reason, out))

    # 3. Optional Unsloth comparison
    unmerged_outs: list[str] | None = None
    if not args.no_compare and args.adapter is not None:
        logger.info("=== loading UNMERGED via Unsloth (reference) ===")
        try:
            t0 = time.time()
            u_model, u_tok = load_unmerged(args.adapter)
            logger.info("unmerged loaded in {:.1f}s", time.time() - t0)
            unmerged_outs = []
            for jp in prompts:
                full = PROMPT.format(jp=jp)
                out = _gen(u_model, u_tok, full, "cuda", deterministic=args.deterministic)
                unmerged_outs.append(out)
                logger.info("[unmerged] JP={!r} -> EN={!r}", jp, out)
        except Exception as e:  # noqa: BLE001
            logger.warning("Unsloth comparison failed: {} -- continuing with merged-only verdict", e)

    # 4. Report
    print("\n" + "=" * 80)
    print("VERIFICATION REPORT")
    print("=" * 80)
    for i, jp in enumerate(prompts):
        m = merged_outs[i]
        u = unmerged_outs[i] if unmerged_outs else None
        print(f"\n[{i+1}] JP: {jp}")
        print(f"    merged-clean: {m!r}")
        if u is not None:
            match = "EXACT" if u.strip() == m.strip() else "DIFF"
            print(f"    unmerged    : {u!r}    [{match}]")

    print("\n" + "-" * 80)
    if bad:
        print(f"FAIL: {len(bad)}/{len(merged_outs)} merged outputs look corrupt:")
        for idx, reason, out in bad:
            print(f"  [{idx+1}] {reason}: {out!r}")
        return 1

    if unmerged_outs is not None:
        diffs = sum(1 for m, u in zip(merged_outs, unmerged_outs) if m.strip() != u.strip())
        print(f"merged vs unmerged: {len(merged_outs) - diffs}/{len(merged_outs)} exact matches")

    print("PASS: merged-clean model produces sensible English output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

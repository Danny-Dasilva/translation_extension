"""Single-pass eval: load model ONCE, generate for all held-outs, score chrF + VNTL cosine."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sacrebleu
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from backend.scripts.eval._generation import (  # noqa: E402
    DEFAULT_SAMPLING,
    build_prompts,
    load_hf_model,
)

CKPT = REPO / "backend/training/runs/manga-bubbles/qwen3_1p7b_sft/final"
HELD_OUT = REPO / "backend/training/eval_held_out"
OUT_DIR = REPO / "backend/training/runs/manga-bubbles"

DATASETS = {
    "vntl128": "vntl128.jsonl",
    "flores": "flores_ja_en.jsonl",
    "open_mantra_test": "open_mantra_test.jsonl",
    "regression_canary": "regression_canary.jsonl",
}


def read_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_all(model, tokenizer, jp_list: list[str], batch_size: int = 32) -> list[str]:
    sp = DEFAULT_SAMPLING
    prompts = build_prompts(jp_list, tokenizer)
    results: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(
            model.device
        )
        with torch.inference_mode():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=sp["temperature"],
                top_p=sp["top_p"],
                min_p=sp.get("min_p", 0.0),
                max_new_tokens=sp["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        for j in range(out.shape[0]):
            gen = out[j, enc["input_ids"].shape[1] :]
            text = tokenizer.decode(gen, skip_special_tokens=True).strip()
            results.append(text)
        if (i // batch_size) % 5 == 0:
            logger.info("  {} / {} prompts done", min(i + batch_size, len(prompts)), len(prompts))
    return results


def cosine_accuracy(preds: list[str], refs: list[str], st_model) -> dict:
    pairs = [(p, r) for p, r in zip(preds, refs) if p.strip() and r.strip()]
    if not pairs:
        return {"n": 0, "mean": float("nan")}
    preds_e = st_model.encode([p for p, _ in pairs], normalize_embeddings=True, show_progress_bar=False)
    refs_e = st_model.encode([r for _, r in pairs], normalize_embeddings=True, show_progress_bar=False)
    cos = (preds_e * refs_e).sum(axis=1)
    rng = np.random.default_rng(42)
    boot = [cos[rng.integers(0, len(cos), len(cos))].mean() for _ in range(1000)]
    return {
        "n": len(pairs),
        "mean": float(cos.mean()),
        "std": float(cos.std()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def main() -> int:
    report: dict = {"checkpoint": str(CKPT), "datasets": {}}
    logger.info("loading model + LoRA (once)...")
    model, tokenizer = load_hf_model(CKPT)

    for name, fname in DATASETS.items():
        path = HELD_OUT / fname
        if not path.exists():
            logger.warning("missing: {}", path)
            continue
        rows = read_jsonl(path)
        jp = [r["jp"] for r in rows]
        refs = [r.get("en_ref") or r.get("en", "") for r in rows]
        logger.info("[{}] generating n={}...", name, len(jp))
        preds = generate_all(model, tokenizer, jp, batch_size=32)

        chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
        report["datasets"][name] = {"n": len(rows), "chrf++": chrf}

        pred_path = OUT_DIR / f"preds_{name}.jsonl"
        with open(pred_path, "w") as f:
            for r, p, rref in zip(rows, preds, refs):
                f.write(json.dumps({"jp": r["jp"], "en_pred": p, "en_ref": rref}, ensure_ascii=False) + "\n")
        logger.info("[{}] chrF++={:.2f}, preds→{}", name, chrf, pred_path.name)

    # Free the LLM before loading the cosine encoder
    del model, tokenizer
    torch.cuda.empty_cache()

    logger.info("loading SentenceTransformer for cosine...")
    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    for name in DATASETS:
        pred_path = OUT_DIR / f"preds_{name}.jsonl"
        if not pred_path.exists():
            continue
        rows = read_jsonl(pred_path)
        preds = [r["en_pred"] for r in rows]
        refs = [r["en_ref"] for r in rows]
        cos = cosine_accuracy(preds, refs, st)
        report["datasets"][name]["vntl_cosine"] = cos
        logger.info("[{}] cosine mean={:.4f} n={}", name, cos.get("mean", 0), cos.get("n", 0))

    out_path = OUT_DIR / "quick_eval.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("wrote {}", out_path)

    # Print a clean summary
    print()
    print("=== EVAL SUMMARY ===")
    for name, rec in report["datasets"].items():
        cos_mean = rec.get("vntl_cosine", {}).get("mean", None)
        cos_str = f"cosine={cos_mean:.4f}" if cos_mean is not None else "cosine=NA"
        print(f"  {name:22s} n={rec['n']:>5}  chrF++={rec['chrf++']:6.2f}  {cos_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

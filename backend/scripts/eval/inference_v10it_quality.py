"""High-quality inference for v10-it via vLLM with BoN + chrF-MBR + RAG.

Configurations (run any subset via --config):
  * greedy            : single deterministic decode (baseline)
  * constrained       : greedy + length cap + JP-vocab logit_bias + repetition penalty
  * bon_chrf          : N=8 epsilon-sampled, chrF-MBR pick across candidates
  * bon_chrf_kiwi     : bon_chrf + CometKiwi-XL re-rank on top-3
  * bon_chrf_rag      : bon_chrf + RAG few-shot retrieval (top-3 nearest JP→EN exemplars)
  * bon_chrf_kiwi_rag : both kiwi rerank + RAG ICL

Inputs
------
JSONL with one item per line, must have a ``jp`` field:
    {"jp": "...", "src": "..."}      # for OpenMantra held-out
    {"jp": "...", "ref": "..."}      # if reference is known (will round-trip it)

If the JSONL has both ``jp`` and a per-item OCR-style structure, it is also
accepted. Output JSONL with same schema + ``en`` (best translation),
``candidates`` (raw 8 if BoN), ``latency_ms``, ``config``.

Output
------
    <out-dir>/<config>/translations.jsonl
    <out-dir>/<config>/stats.json   (latency p50/p95, total time)

Run as
------
    /home/danny/.venvs/vllm/bin/python inference_v10it_quality.py \
        --in-jsonl backend/scripts/eval/data/openmantra/heldout.jsonl \
        --out-dir backend/scripts/eval/openmantra_v10it \
        --config greedy,constrained,bon_chrf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Defer heavy imports
SCRIPT_DIR = Path(__file__).resolve().parent

# --- prompt templates -------------------------------------------------------
CHAT_USER_MSG_TEMPLATE = (
    "Translate the following Japanese to English. Output only the translation.\n\n"
    "Japanese: {jp}"
)

CHAT_USER_MSG_RAG_TEMPLATE = (
    "Translate the following Japanese to English. Output only the translation.\n\n"
    "{exemplars}\nJapanese: {jp}"
)

EXEMPLAR_TEMPLATE = "Japanese: {jp}\nEnglish: {en}\n"

# Raw template (v9c-era, -pt base — no chat template).
RAW_PROMPT = (
    "Translate the following Japanese to English. Output only the translation.\n\n"
    "Japanese: {jp}\nEnglish:"
)
RAW_PROMPT_RAG = (
    "Translate the following Japanese to English. Output only the translation.\n\n"
    "{exemplars}Japanese: {jp}\nEnglish:"
)

# --- post-process regexes (lifted from translate_manga_unsloth.py) ----------
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


# --- JP-vocab logit_bias ----------------------------------------------------
def build_jp_logit_bias(tokenizer) -> dict[int, float]:
    """Return logit_bias mapping for tokens that decode to CJK/Hiragana/Katakana.

    Heuristic: walk the entire vocab once, decode each id to text, and ban any
    token whose text contains chars in U+3040-30FF or U+4E00-9FFF.
    """
    vocab = tokenizer.get_vocab()
    bias: dict[int, float] = {}
    for tok_str, tok_id in vocab.items():
        # decode the id directly to handle BPE byte-pieces correctly
        try:
            decoded = tokenizer.decode([tok_id], skip_special_tokens=False)
        except Exception:
            decoded = tok_str
        if any(
            (0x3040 <= ord(c) <= 0x309F)
            or (0x30A0 <= ord(c) <= 0x30FF)
            or (0x4E00 <= ord(c) <= 0x9FFF)
            for c in decoded
        ):
            bias[int(tok_id)] = -100.0
    return bias


# --- chrF-MBR selection -----------------------------------------------------
def select_mbr_chrf(candidates: list[str]) -> tuple[int, list[float]]:
    """Pick candidate with max mean pairwise chrF; return (idx, per-cand-mean)."""
    cands = [c for c in candidates if c.strip()]
    if not cands:
        return 0, []
    if len(cands) == 1:
        return candidates.index(cands[0]), [0.0]
    try:
        from fastchrf import pairwise_chrf
        # fastchrf wants list[list[str]]: hypotheses per ref-cluster
        # We pass each candidate as both hypothesis and reference (self-MBR)
        # Use the list-of-lists API: hypotheses = [[c] for c in cands], references = list of refs
        # Simpler: roll our own using sacrebleu if fastchrf shape-mismatches.
        scores_matrix = pairwise_chrf([cands], [cands])
        # scores_matrix shape: (1, n_cands, n_cands_refs)
        means = []
        n = len(cands)
        for i in range(n):
            other = [scores_matrix[0][i][j] for j in range(n) if j != i]
            means.append(sum(other) / max(1, len(other)))
        best_local = max(range(n), key=lambda i: means[i])
        best_global = candidates.index(cands[best_local])
        # back-fill for empty cands
        full_means = []
        ci = 0
        for c in candidates:
            if c.strip():
                full_means.append(means[ci])
                ci += 1
            else:
                full_means.append(-1.0)
        return best_global, full_means
    except Exception:
        # Fallback: sacrebleu sentence_chrf pairwise (slower).
        import sacrebleu
        n = len(cands)
        means = []
        for i in range(n):
            others = [cands[j] for j in range(n) if j != i]
            s = sum(sacrebleu.sentence_chrf(cands[i], [o], word_order=2).score for o in others) / max(1, len(others))
            means.append(s)
        best_local = max(range(n), key=lambda i: means[i])
        best_global = candidates.index(cands[best_local])
        full_means = []
        ci = 0
        for c in candidates:
            if c.strip():
                full_means.append(means[ci])
                ci += 1
            else:
                full_means.append(-1.0)
        return best_global, full_means


# --- main ------------------------------------------------------------------
CONFIGS = [
    "greedy",
    "constrained",
    "bon_chrf",
    "bon_chrf_kiwi",
    "bon_chrf_rag",
    "bon_chrf_kiwi_rag",
]


def render_chat(
    tokenizer, jp: str, *, exemplars: list[tuple[str, str]] | None = None,
    fmt: str = "chat",
) -> str:
    if fmt == "raw":
        if exemplars:
            ex_text = "".join(EXEMPLAR_TEMPLATE.format(jp=j, en=e) for j, e in exemplars)
            return RAW_PROMPT_RAG.format(exemplars=ex_text, jp=jp)
        return RAW_PROMPT.format(jp=jp)

    # default: chat template
    if exemplars:
        ex_text = "\n".join(EXEMPLAR_TEMPLATE.format(jp=j, en=e) for j, e in exemplars)
        user_msg = CHAT_USER_MSG_RAG_TEMPLATE.format(exemplars=ex_text, jp=jp)
    else:
        user_msg = CHAT_USER_MSG_TEMPLATE.format(jp=jp)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


def estimate_length_cap(jp: str) -> int:
    """Translation tends to be ~1-1.5x source token count; cap at 1.5x + 16."""
    # Crude — bytes correlate with JP token count.
    return min(128, max(24, int(round(1.5 * len(jp))) + 16))


def get_sampling(config: str, *, jp: str, jp_bias: dict[int, float] | None,
                 n: int = 8) -> dict[str, Any]:
    """Build vLLM SamplingParams kwargs per config."""
    base: dict[str, Any] = {
        "temperature": 0.0,  # greedy default
        "max_tokens": 60,
        "n": 1,
        "skip_special_tokens": True,
    }
    if config == "greedy":
        return base

    if config == "constrained":
        base.update({
            "max_tokens": estimate_length_cap(jp),
            "repetition_penalty": 1.05,
            "logit_bias": jp_bias or {},
        })
        return base

    # All BoN configs use the same generation knobs
    if config.startswith("bon_chrf"):
        base.update({
            "temperature": 0.9,
            "top_p": 0.95,
            # vLLM SamplingParams supports `epsilon_cutoff` via `extra_args` only
            # in nightly; fall back to top_p+temperature alone if unsupported.
            "max_tokens": estimate_length_cap(jp),
            "repetition_penalty": 1.05,
            "n": n,
            "logit_bias": jp_bias or {},
        })
        return base

    raise ValueError(f"unknown config: {config}")


def _strip_chat_residues(t: str) -> str:
    for cut in ["<turn|>", "<|turn>", "<start_of_turn>", "<end_of_turn>", "Japanese:", "English:"]:
        j = t.find(cut)
        if j >= 0:
            t = t[:j].strip()
    return t


def run_config_for_items(
    llm,
    tokenizer,
    items: list[dict[str, Any]],
    *,
    config: str,
    jp_bias: dict[int, float] | None,
    rag_exemplars: dict[int, list[tuple[str, str]]] | None = None,
    kiwi_model=None,
    prompt_format: str = "chat",
) -> list[dict[str, Any]]:
    from vllm import SamplingParams
    rendered_prompts: list[str] = []
    item_meta: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        exemplars = None
        if config.endswith("_rag") and rag_exemplars is not None:
            exemplars = rag_exemplars.get(i, [])
        prompt = render_chat(tokenizer, it["jp"], exemplars=exemplars, fmt=prompt_format)
        rendered_prompts.append(prompt)
        item_meta.append({"exemplars": exemplars or []})

    # Build per-item SamplingParams (since max_tokens depends on JP length)
    sps: list[Any] = []
    for it in items:
        kwargs = get_sampling(config, jp=it["jp"], jp_bias=jp_bias)
        sps.append(SamplingParams(**kwargs))

    t0 = time.time()
    outs = llm.generate(rendered_prompts, sampling_params=sps, use_tqdm=False)
    dt_total = (time.time() - t0) * 1000

    rows: list[dict[str, Any]] = []
    for it, meta, out, sp in zip(items, item_meta, outs, sps):
        # vLLM returns RequestOutput with .outputs (list of CompletionOutputs)
        cands = [_strip_chat_residues(c.text).strip() for c in out.outputs]
        cands = [clean(c) for c in cands]
        # latency: use the longest finish time for this request as upper bound
        latency_ms = (out.metrics.finished_time - out.metrics.first_scheduled_time) * 1000 \
            if out.metrics and out.metrics.finished_time and out.metrics.first_scheduled_time else None

        if config in ("greedy", "constrained"):
            chosen = cands[0] if cands else ""
            chrf_means = []
        else:
            # BoN: chrF-MBR
            best_idx, chrf_means = select_mbr_chrf(cands)
            chosen = cands[best_idx]
            # Optional Kiwi rerank on top-3
            if "kiwi" in config and kiwi_model is not None and len(cands) >= 2:
                top3_idxs = sorted(range(len(cands)), key=lambda i: -chrf_means[i])[:3]
                if any(cands[i].strip() for i in top3_idxs):
                    qe_data = [{"src": it["jp"], "mt": cands[i]} for i in top3_idxs]
                    kiwi_scores = kiwi_model.predict(qe_data, batch_size=8, gpus=1, progress_bar=False).scores
                    # Combine: 0.7 * chrF_rank + 0.3 * kiwi_rank
                    # Higher = better for both
                    chrf_ranks = sorted(range(len(top3_idxs)), key=lambda r: -chrf_means[top3_idxs[r]])
                    kiwi_ranks = sorted(range(len(top3_idxs)), key=lambda r: -kiwi_scores[r])
                    rank_score = [0.0] * len(top3_idxs)
                    for r, idx in enumerate(chrf_ranks): rank_score[idx] += 0.7 * (len(top3_idxs) - r)
                    for r, idx in enumerate(kiwi_ranks): rank_score[idx] += 0.3 * (len(top3_idxs) - r)
                    best_local = max(range(len(top3_idxs)), key=lambda i: rank_score[i])
                    chosen = cands[top3_idxs[best_local]]

        rows.append({
            **it,
            "config": config,
            "en": chosen,
            "candidates": cands,
            "latency_ms": latency_ms,
            "exemplars": meta.get("exemplars", []),
        })
    print(f"[{config}] generated {len(items)} items in {dt_total:.0f}ms total")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True,
                    help="JSONL with {jp, ...} per line")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="/home/danny/Documents/personal/extension/backend/training/runs/manga-bubbles/gemma4_e4b_v10it/merged")
    ap.add_argument("--config", default="greedy,constrained,bon_chrf",
                    help="comma-separated configs from " + "|".join(CONFIGS))
    ap.add_argument("--rag-index", default=None,
                    help="Pre-built RAG index dir (for *_rag configs)")
    ap.add_argument("--kiwi", action="store_true",
                    help="Allow loading CometKiwi for *_kiwi configs (slow first-load)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-format", choices=["chat", "raw"], default="chat",
                    help="chat (default; for v10-it / -it base) or raw (for v9c / -pt base)")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    configs = [c.strip() for c in args.config.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS:
            print(f"ERROR: unknown config {c!r}; valid: {CONFIGS}", file=sys.stderr)
            return 2

    items: list[dict[str, Any]] = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    if args.limit:
        items = items[: args.limit]
    print(f"loaded {len(items)} items from {in_path}")

    # Pre-compute RAG exemplars BEFORE loading vLLM (the BGE encoder needs GPU
    # too; can't co-host with a 12GB vLLM). We cache exemplars per (item_idx)
    # and a wrapper class returns them at inference time.
    rag_exemplars: dict[int, list[tuple[str, str]]] = {}
    if any(c.endswith("_rag") for c in configs):
        if args.rag_index:
            print(f"pre-computing RAG exemplars for {len(items)} items (BGE-M3)...")
            sys.path.insert(0, str(SCRIPT_DIR))
            from rag_retrieval_v10it import RAGIndex
            rag_index_obj = RAGIndex.load(args.rag_index)
            for i, it in enumerate(items):
                rag_exemplars[i] = rag_index_obj.topk(it["jp"], k=3)
            print(f"  RAG exemplars cached for {len(rag_exemplars)} items")
            # Free BGE encoder + index from VRAM before loading vLLM
            del rag_index_obj
            import torch as _torch
            _torch.cuda.empty_cache()
        else:
            print("WARN: *_rag configs requested but no --rag-index given; skipping RAG configs")
            configs = [c for c in configs if not c.endswith("_rag")]

    print(f"loading vLLM: {args.model}")
    from vllm import LLM
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    # JP logit_bias built once (only needed for constrained/bon_*)
    need_bias = any(c != "greedy" for c in configs)
    jp_bias = build_jp_logit_bias(tokenizer) if need_bias else None
    if jp_bias is not None:
        print(f"JP logit_bias: {len(jp_bias)} tokens banned")

    kiwi_model = None
    if args.kiwi and any("kiwi" in c for c in configs):
        from comet import download_model, load_from_checkpoint
        ck = download_model("Unbabel/wmt23-cometkiwi-da-xl")
        kiwi_model = load_from_checkpoint(ck)
        print("loaded CometKiwi-23-XL for rerank")

    summary: dict[str, Any] = {"input": str(in_path), "n_items": len(items), "configs": {}}

    for cfg in configs:
        cfg_dir = out_root / cfg
        cfg_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rows = run_config_for_items(
            llm, tokenizer, items, config=cfg, jp_bias=jp_bias,
            rag_exemplars=rag_exemplars, kiwi_model=kiwi_model,
            prompt_format=args.prompt_format,
        )
        wall = time.time() - t0

        # Write per-item JSONL
        with open(cfg_dir / "translations.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
        latencies.sort()

        def pct(p: float) -> float:
            if not latencies:
                return 0.0
            i = int(p * len(latencies))
            return float(latencies[min(len(latencies) - 1, i)])

        cfg_stats = {
            "config": cfg,
            "n": len(rows),
            "wall_s": wall,
            "latency_ms_p50": pct(0.5),
            "latency_ms_p95": pct(0.95),
            "throughput_items_per_s": len(rows) / max(wall, 0.001),
            "empty_n": sum(1 for r in rows if not r["en"].strip()),
        }
        with open(cfg_dir / "stats.json", "w") as f:
            json.dump(cfg_stats, f, indent=2)
        print(f"  [{cfg}] wall={wall:.1f}s p50={cfg_stats['latency_ms_p50']:.0f}ms p95={cfg_stats['latency_ms_p95']:.0f}ms")
        summary["configs"][cfg] = cfg_stats

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote summary to {out_root/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build v10.5 CPO preference dataset.

Phase 1 of the v10.5 CPO chain. Per the locked recipe in
``thoughts/shared/research/v10-training/synthesis.yaml``:

    cpo_chain_v10_5:
      triplet_construction:
        - per source: gold + Gemma 31B teacher + v10-it on-policy
        - score with: COMET-Kiwi-23 XL
        - chosen=argmax, rejected=argmin, drop pair if margin < 0.05 or
          length-ratio extreme
        - target: 20-25k filtered triplets

Pipeline
--------
1. Load ``data_v10.parquet`` (258k rows) and stratified-sample 30k:
   - 70% v7.1 (`src` does NOT start with ``manga109:``) — true human gold.
   - 30% manga109 — has Gemma 31B teacher pre-cached in
     ``bubbles_translated_qe_deduped.parquet`` keyed by (book, page, text_id).

2. For each row build candidates:
   - C1 = ``en`` from data_v10  (always present — gold or teacher curated)
   - C2 = Gemma 31B teacher output (manga109 rows only; v7.1 rows skip)
   - C3 = v10-it on-policy sample via vLLM offline (temperature 0.7, top_p 0.9)

3. Score each (jp, candidate) pair with COMET-Kiwi-23 XL, run in the comet venv
   via subprocess (the comet/torch stack and the vLLM/torch stack don't coexist
   in one venv).

4. Build preference pairs per row (2 candidates if v7.1, 3 if manga109+match):
   - chosen = argmax(score), rejected = argmin(score)
   - Drop middle (only matters for 3-cand rows)
   - Drop if margin < 0.05
   - Drop if |chosen|/|rejected| length ratio in (0.5, 2.0)

5. Output: ``backend/scripts/data/cpo/v10_5_preferences.parquet``
   with columns [prompt, chosen, rejected, chosen_score, rejected_score, jp, src]

The ``prompt`` column stores the user-message body (NOT the chat-template-wrapped
text); the CPO trainer applies the Gemma-4 -it chat template at training time
to match how v10-it was trained.

Resumability
------------
Each candidate-generation pass writes an intermediate parquet so re-runs skip
already-done work. The COMET-Kiwi scoring pass and the final pair-build pass
are also independently restartable.

Stages
------
* sample        — stratified sample of 30k, joined with manga109 teacher
* gen-onpolicy  — generate C3 (v10-it samples) via vLLM
* score         — COMET-Kiwi over all (jp, cand) pairs
* pairs         — build chosen/rejected, apply filters, write final parquet
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import polars as pl
from loguru import logger


PROJECT_ROOT = Path("/home/danny/Documents/personal/extension")
COMET_VENV = Path("/home/danny/.venvs/comet/bin/python")
VLLM_VENV = Path("/home/danny/.venvs/vllm/bin/python")

DATA_V10 = PROJECT_ROOT / "backend/scripts/data/manga109/data_v10.parquet"
TEACHER_PARQUET = PROJECT_ROOT / "backend/scripts/data/manga109/bubbles_translated_qe_deduped.parquet"
V10IT_MERGED = PROJECT_ROOT / "backend/training/runs/manga-bubbles/gemma4_e4b_v10it/merged"

OUT_DIR = PROJECT_ROOT / "backend/scripts/data/cpo"
SAMPLE_PARQUET = OUT_DIR / "v10_5_sample.parquet"
ONPOLICY_PARQUET = OUT_DIR / "v10_5_onpolicy.parquet"
SCORED_PARQUET = OUT_DIR / "v10_5_scored.parquet"
FINAL_PARQUET = OUT_DIR / "v10_5_preferences.parquet"

# Match the v10-it training prompt exactly. The CPO trainer will wrap with
# the chat template at train time; here we store the BARE user-message body.
USER_TEMPLATE = (
    "Translate the following Japanese to English. Output only the translation.\n\n"
    "Japanese: {jp}"
)


# ---------------------------------------------------------------------------
# Stage 1: stratified sample + manga109 teacher join
# ---------------------------------------------------------------------------


def stage_sample(n: int, seed: int = 42) -> pl.DataFrame:
    """Stratified sample of n rows from data_v10.parquet, joined with manga109 teacher."""
    if SAMPLE_PARQUET.exists():
        logger.info("[sample] cache hit: {}", SAMPLE_PARQUET)
        return pl.read_parquet(SAMPLE_PARQUET)

    logger.info("[sample] loading {}", DATA_V10)
    df = pl.read_parquet(DATA_V10)
    logger.info("[sample] loaded {} rows", len(df))

    # Drop rows with empty jp/en or where en is just "..."
    df = df.filter(
        (pl.col("jp").str.strip_chars().str.len_chars() > 0) &
        (pl.col("en").str.strip_chars().str.len_chars() > 0) &
        (pl.col("en").str.strip_chars() != "...") &
        (pl.col("en").str.strip_chars() != "…")
    )
    logger.info("[sample] after non-empty filter: {} rows", len(df))

    # Split: manga109 vs v7.1 portions
    manga109_df = df.filter(pl.col("src").str.starts_with("manga109:"))
    v7_df = df.filter(~pl.col("src").str.starts_with("manga109:"))
    logger.info(
        "[sample] split: manga109={} v7.1={} (total {})",
        len(manga109_df), len(v7_df), len(df),
    )

    # Target counts: 70% v7.1 (≈21k), 30% manga109 (≈9k)
    n_v7 = int(n * 0.70)
    n_m109 = n - n_v7

    # Stratified-by-register sampling within each portion
    def _strat_sample(d: pl.DataFrame, k: int, seed_: int) -> pl.DataFrame:
        groups = d.group_by("register_tag").agg(pl.len()).sort("len", descending=True)
        if len(groups) == 0:
            return d.sample(n=min(k, len(d)), seed=seed_, shuffle=True)
        # sqrt-weight (vn_eroge dominates otherwise)
        sizes = groups["len"].to_list()
        total_w = sum(s ** 0.5 for s in sizes)
        parts = []
        used = 0
        for row in groups.iter_rows(named=True):
            tag = row["register_tag"]
            w = (row["len"] ** 0.5) / total_w
            target = max(1, int(k * w))
            target = min(target, row["len"])
            sub = d.filter(pl.col("register_tag") == tag).sample(
                n=target, seed=seed_, shuffle=True
            )
            parts.append(sub)
            used += target
        out = pl.concat(parts)
        # If we under-sampled due to int() rounding, top up randomly
        if len(out) < k:
            remaining = d.join(out, on=["jp", "en", "src"], how="anti")
            top_up = remaining.sample(
                n=min(k - len(out), len(remaining)),
                seed=seed_, shuffle=True,
            )
            out = pl.concat([out, top_up])
        # If we over-sampled, trim
        if len(out) > k:
            out = out.sample(n=k, seed=seed_, shuffle=True)
        return out

    v7_samp = _strat_sample(v7_df, n_v7, seed)
    m109_samp = _strat_sample(manga109_df, n_m109, seed + 1)
    logger.info(
        "[sample] sampled: v7.1={} manga109={} total={}",
        len(v7_samp), len(m109_samp), len(v7_samp) + len(m109_samp),
    )

    # Join manga109 with teacher parquet
    logger.info("[sample] loading teacher parquet: {}", TEACHER_PARQUET)
    teacher = pl.read_parquet(TEACHER_PARQUET)
    logger.info("[sample] teacher rows: {}", len(teacher))

    # Build join key from src: "manga109:Book:p{page}:{text_id}"
    def _parse_src(src_col: pl.Expr) -> pl.DataFrame:
        # src like manga109:MagicianLoad:p54:0003bd9c
        parts = src_col.str.split(":")
        return parts

    m109_keyed = m109_samp.with_columns(
        pl.col("src").str.split(":").alias("_parts")
    ).with_columns([
        pl.col("_parts").list.get(1).alias("book"),
        pl.col("_parts").list.get(2).str.slice(1).cast(pl.Int64, strict=False).alias("page"),
        pl.col("_parts").list.get(3).alias("text_id"),
    ]).drop("_parts")

    teacher_lite = teacher.select([
        "book", "page", "text_id",
        pl.col("en_text").alias("teacher_en"),
        pl.col("kiwi_score").alias("teacher_kiwi"),
    ])

    m109_joined = m109_keyed.join(
        teacher_lite, on=["book", "page", "text_id"], how="left"
    ).drop(["book", "page", "text_id"])
    n_with_teacher = m109_joined.filter(pl.col("teacher_en").is_not_null()).height
    logger.info(
        "[sample] manga109 join: {}/{} have teacher output",
        n_with_teacher, len(m109_joined),
    )

    # v7.1 portion gets null teacher columns
    v7_with_nulls = v7_samp.with_columns([
        pl.lit(None, dtype=pl.String).alias("teacher_en"),
        pl.lit(None, dtype=pl.Float32).alias("teacher_kiwi"),
    ])

    combined = pl.concat([v7_with_nulls, m109_joined], how="diagonal")
    logger.info("[sample] combined rows: {}", len(combined))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(SAMPLE_PARQUET)
    logger.info("[sample] wrote {}", SAMPLE_PARQUET)
    return combined


# ---------------------------------------------------------------------------
# Stage 2: generate v10-it on-policy samples (C3)
# ---------------------------------------------------------------------------


def stage_onpolicy(temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 60) -> pl.DataFrame:
    """Generate v10-it on-policy translations via vLLM offline.

    Spawns a child process under VLLM_VENV (the active backend venv has
    transformers 5.x which conflicts with vLLM's pinned transformers).
    """
    if ONPOLICY_PARQUET.exists():
        logger.info("[onpolicy] cache hit: {}", ONPOLICY_PARQUET)
        return pl.read_parquet(ONPOLICY_PARQUET)

    if not SAMPLE_PARQUET.exists():
        raise FileNotFoundError(f"need {SAMPLE_PARQUET} first; run --stage sample")

    runner_script = OUT_DIR / "_run_vllm_onpolicy.py"
    runner_script.write_text(_VLLM_RUNNER_BODY)

    cmd = [
        str(VLLM_VENV),
        str(runner_script),
        "--input", str(SAMPLE_PARQUET),
        "--output", str(ONPOLICY_PARQUET),
        "--model", str(V10IT_MERGED),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--max-tokens", str(max_tokens),
    ]
    # flashinfer JIT requires `ninja` on PATH; vLLM venv provides it.
    env = os.environ.copy()
    env["PATH"] = f"{VLLM_VENV.parent}:{env.get('PATH', '')}"
    logger.info("[onpolicy] running vLLM child: {}", " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, check=False, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"vLLM runner exited {r.returncode}")
    logger.info("[onpolicy] done in {:.1f}s", time.time() - t0)
    return pl.read_parquet(ONPOLICY_PARQUET)


# vLLM runner script — written to disk and executed under VLLM_VENV. Kept
# as a string here to keep the entire build pipeline in one file.
_VLLM_RUNNER_BODY = '''#!/usr/bin/env python
"""Internal: generate on-policy v10-it samples for v10.5 CPO triplets.

This runs under /home/danny/.venvs/vllm because vLLM and the rest of the
backend use incompatible transformers versions. Do not import from the
backend package here.
"""
from __future__ import annotations
import argparse
import re
import sys
import time
from pathlib import Path

import polars as pl
from vllm import LLM, SamplingParams


USER_TEMPLATE = (
    "Translate the following Japanese to English. Output only the translation.\\n\\n"
    "Japanese: {jp}"
)

NEWLINE_RE = re.compile(r"[\\r\\n]")
NEXT_PROMPT_RE = re.compile(r"\\s*(?:Japanese:|JP:|English:|EN:).*$", re.S)


def clean(text: str) -> str:
    if not text:
        return ""
    text = text.lstrip()
    text = NEWLINE_RE.split(text, 1)[0]
    text = NEXT_PROMPT_RE.sub("", text)
    # Strip residual chat-template markers
    for cut in ["<turn|>", "<|turn>", "<start_of_turn>", "<end_of_turn>"]:
        j = text.find(cut)
        if j >= 0:
            text = text[:j]
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=1024)
    args = ap.parse_args()

    df = pl.read_parquet(args.input)
    print(f"[vllm-onpolicy] {len(df)} rows from {args.input}", flush=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()
    print(f"[vllm-onpolicy] model loaded, building prompts...", flush=True)

    prompts = []
    for r in df.iter_rows(named=True):
        jp = (r.get("jp") or "").strip()
        msg = USER_TEMPLATE.format(jp=jp)
        text = tok.apply_chat_template(
            [{"role": "user", "content": msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(text)

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<end_of_turn>", "<|end|>"],
        seed=42,
    )

    print(f"[vllm-onpolicy] generating {len(prompts)} prompts...", flush=True)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    print(f"[vllm-onpolicy] done in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} prompts/s)", flush=True)

    raw_texts = []
    cleaned = []
    for o in outs:
        raw = o.outputs[0].text if o.outputs else ""
        raw_texts.append(raw)
        cleaned.append(clean(raw))

    out_df = df.with_columns([
        pl.Series("onpolicy_raw", raw_texts, dtype=pl.String),
        pl.Series("onpolicy_en", cleaned, dtype=pl.String),
    ])
    out_df.write_parquet(args.output)
    print(f"[vllm-onpolicy] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


# ---------------------------------------------------------------------------
# Stage 3: COMET-Kiwi scoring (subprocess to comet venv)
# ---------------------------------------------------------------------------


def stage_score(batch_size: int = 32) -> pl.DataFrame:
    if SCORED_PARQUET.exists():
        logger.info("[score] cache hit: {}", SCORED_PARQUET)
        return pl.read_parquet(SCORED_PARQUET)

    if not ONPOLICY_PARQUET.exists():
        raise FileNotFoundError(f"need {ONPOLICY_PARQUET}; run --stage onpolicy first")

    df = pl.read_parquet(ONPOLICY_PARQUET)
    logger.info("[score] loaded {} rows", len(df))

    # Build a tall (jp, candidate, candidate_kind, row_id) parquet for the
    # scorer. Using row_id lets us reshape back to wide afterwards.
    df = df.with_row_index("row_id")

    rows: list[dict] = []
    for r in df.iter_rows(named=True):
        rid = int(r["row_id"])
        jp = r["jp"]
        # C1 — gold/teacher (data_v10 `en` field). For manga109 rows this IS
        # the Gemma 31B teacher output (already curated by the v10 ingest);
        # for v7.1 rows this is human gold.
        c1 = (r.get("en") or "").strip()
        if c1:
            rows.append({"row_id": rid, "kind": "C1_gold", "jp": jp, "en": c1})
        # C2 — separate teacher pass (manga109 only). On THIS dataset C2 ≡ C1
        # for ~all manga109 rows because data_v10's `en` was sourced from the
        # same teacher parquet, so we only emit C2 if it differs from C1.
        c2 = (r.get("teacher_en") or "").strip() if r.get("teacher_en") else ""
        if c2 and c2 != c1:
            rows.append({"row_id": rid, "kind": "C2_teacher", "jp": jp, "en": c2})
        # C3 — v10-it on-policy
        c3 = (r.get("onpolicy_en") or "").strip()
        if c3 and c3 != c1 and c3 != c2:
            rows.append({"row_id": rid, "kind": "C3_onpolicy", "jp": jp, "en": c3})

    tall = pl.DataFrame(rows)
    logger.info("[score] tall rows: {}", len(tall))
    by_kind = tall.group_by("kind").agg(pl.len()).sort("len", descending=True)
    logger.info("[score] candidates per kind:\n{}", by_kind)

    tall_in = OUT_DIR / "_score_input.parquet"
    tall_out = OUT_DIR / "_score_output.parquet"
    tall.select(["jp", "en"]).write_parquet(tall_in)

    # Run COMET-Kiwi via the comet venv
    score_cmd = [
        str(COMET_VENV),
        str(PROJECT_ROOT / "backend/scripts/data/score_cometkiwi.py"),
        "--input", str(tall_in),
        "--out", str(tall_out),
        "--batch-size", str(batch_size),
        "--gpus", "1",
    ]
    logger.info("[score] running: {}", " ".join(score_cmd))
    t0 = time.time()
    r = subprocess.run(score_cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"score_cometkiwi exited {r.returncode}")
    logger.info("[score] done in {:.1f}s", time.time() - t0)

    scored_tall = pl.read_parquet(tall_out)
    # Re-attach row_id and kind from `tall`
    scored_tall = scored_tall.with_columns([
        pl.Series("row_id", tall["row_id"]),
        pl.Series("kind", tall["kind"]),
    ])

    # Pivot wide
    wide = scored_tall.pivot(
        index="row_id",
        on="kind",
        values=["en", "cometkiwi"],
        aggregate_function="first",
    )

    # Merge with original df
    full = df.join(wide, on="row_id", how="left")
    full.write_parquet(SCORED_PARQUET)
    logger.info("[score] wrote {}", SCORED_PARQUET)
    return full


# ---------------------------------------------------------------------------
# Stage 4: build chosen/rejected pairs
# ---------------------------------------------------------------------------


def stage_pairs(margin_min: float = 0.05, len_ratio_min: float = 0.5,
                len_ratio_max: float = 2.0) -> pl.DataFrame:
    if not SCORED_PARQUET.exists():
        raise FileNotFoundError(f"need {SCORED_PARQUET}; run --stage score first")

    df = pl.read_parquet(SCORED_PARQUET)
    logger.info("[pairs] loaded {} scored rows", len(df))

    # The pivot creates columns: en_C1_gold, cometkiwi_C1_gold, en_C2_teacher,
    # cometkiwi_C2_teacher, en_C3_onpolicy, cometkiwi_C3_onpolicy
    cand_kinds = ["C1_gold", "C2_teacher", "C3_onpolicy"]

    pairs = []
    drop_reasons = {"only_one_cand": 0, "margin_too_small": 0, "length_extreme": 0,
                    "no_chosen_or_rejected": 0, "kept": 0}
    for r in df.iter_rows(named=True):
        cands = []
        for k in cand_kinds:
            en_col = f"en_{k}"
            ki_col = f"cometkiwi_{k}"
            en = r.get(en_col)
            ki = r.get(ki_col)
            if en is None or ki is None:
                continue
            en_s = str(en).strip()
            if not en_s:
                continue
            cands.append({"kind": k, "en": en_s, "score": float(ki)})

        if len(cands) < 2:
            drop_reasons["only_one_cand"] += 1
            continue

        cands.sort(key=lambda c: c["score"], reverse=True)
        chosen = cands[0]
        rejected = cands[-1]
        margin = chosen["score"] - rejected["score"]
        if margin < margin_min:
            drop_reasons["margin_too_small"] += 1
            continue
        len_ratio = len(chosen["en"]) / max(1, len(rejected["en"]))
        if len_ratio < len_ratio_min or len_ratio > len_ratio_max:
            drop_reasons["length_extreme"] += 1
            continue
        if not chosen["en"] or not rejected["en"]:
            drop_reasons["no_chosen_or_rejected"] += 1
            continue

        pairs.append({
            "prompt": USER_TEMPLATE.format(jp=r["jp"]),
            "chosen": chosen["en"],
            "rejected": rejected["en"],
            "chosen_score": chosen["score"],
            "rejected_score": rejected["score"],
            "margin": margin,
            "chosen_kind": chosen["kind"],
            "rejected_kind": rejected["kind"],
            "jp": r["jp"],
            "src": r["src"],
            "register_tag": r["register_tag"],
        })
        drop_reasons["kept"] += 1

    logger.info("[pairs] drop reasons: {}", drop_reasons)
    out = pl.DataFrame(pairs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(FINAL_PARQUET)
    logger.info("[pairs] wrote {} pairs to {}", len(out), FINAL_PARQUET)

    # Stats
    logger.info("[pairs] chosen kind distribution:")
    logger.info("\n{}", out.group_by("chosen_kind").agg(pl.len()).sort("len", descending=True))
    logger.info("[pairs] rejected kind distribution:")
    logger.info("\n{}", out.group_by("rejected_kind").agg(pl.len()).sort("len", descending=True))
    logger.info("[pairs] mean chosen score: {:.4f}", out["chosen_score"].mean())
    logger.info("[pairs] mean rejected score: {:.4f}", out["rejected_score"].mean())
    logger.info("[pairs] mean margin: {:.4f}", out["margin"].mean())

    return out


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sample", "onpolicy", "score", "pairs", "all"],
                    default="all")
    ap.add_argument("--n", type=int, default=30000, help="target sample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--score-batch-size", type=int, default=32)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage in ("sample", "all"):
        stage_sample(args.n, args.seed)
    if args.stage in ("onpolicy", "all"):
        stage_onpolicy(args.temperature, args.top_p, args.max_tokens)
    if args.stage in ("score", "all"):
        stage_score(args.score_batch_size)
    if args.stage in ("pairs", "all"):
        stage_pairs()
    return 0


if __name__ == "__main__":
    sys.exit(main())

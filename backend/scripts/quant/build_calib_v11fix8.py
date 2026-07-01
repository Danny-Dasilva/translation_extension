"""Build a v11fix8-FORMAT calibration set for INT4 W4A16 (GPTQ) quantization of
the gemma4_e4b_v11fix8_pagecontext translation model.

WHY a new calib (vs scripts/quant/calib_v11_int4.jsonl)
------------------------------------------------------
The shipped GPTQ-INT4 was calibrated with the OLD v11 calib, which was sampled
from raw v11-era corpora (VNTL / ParallelFiction / SFX) and used the v11
CONVERSATION instruction. That calib optimizes the quant toward v11 behavior and
ERASES exactly the corrective probes v11fix7/fix8 were fine-tuned to ADD —
`reverse_sense` (negation / sense-flip fixes) and the name-invention
(hallucinated-name) fixes. This builder instead draws the calibration mix from
v11fix8's OWN training parquet, and GUARANTEES inclusion of those corrective
rows so GPTQ preserves the +3.0 chrF++ gain.

FORMAT (byte-identical to calib_v11_int4.jsonl — verified)
----------------------------------------------------------
Each row is {"text": <chat-templated user prompt>} where the text is produced by
    tok.apply_chat_template([{"role": "user", "content": <prompt>}],
                            tokenize=False, add_generation_prompt=True)
which the v11fix8 tokenizer renders to:
    <bos><|turn>user\n<PROMPT><turn|>\n<|turn>model\n
We emit ONLY the prompt (no assistant turn); llmcompressor calibrates on the
forward pass over these prompts. quant_w4a16.py::load_calib_texts reads the
"text" field of each line.

CRITICAL: the <PROMPT> body is taken VERBATIM from the v11fix8 parquet `prompt`
column, so it is byte-exact with what the model saw in TRAINING (the v11fix8
page-context PAGE_INSTR / PLAIN_INSTR shapes — NOT the v11 CONV_INSTR the old
calib used). A raw-prompt / template mismatch silently collapses quality ~95%
on this -it fine-tune (see project memory), so we never re-author prompts.

The parquet has NO `reverse_sense` / `name_invention` boolean columns; corrective
rows are identified by the `src` column markers (see SELECTION below).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

BACKEND = Path("/home/danny/Documents/personal/extension/backend")
DEFAULT_PARQUET = BACKEND / "scripts/data/v11fix8/data_v11fix8_pagecontext.parquet"
DEFAULT_MODEL = BACKEND / "training/runs/manga-bubbles/gemma4_e4b_v11fix8_pagecontext/merged_fixed"

# ---- SELECTION: how corrective rows are identified in the v11fix8 parquet -----
# `src` markers (verified by scanning data_v11fix8_pagecontext.parquet):
#   reverse_sense:<jp>:<i>:<shape>     -> 36 unique prompts (negation / sense-flip
#                                          corrective: 締まる, 吸い出せ, 果てた, 風俗,
#                                          騎乗位, マンコ ...). DIRECT match for the
#                                          eval `reverse_sense` probe.
#   voice_probe:...                    -> 18 unique (causative/passive direction).
#   corrective_v11fix7:ikenie4:...     -> Ikenie-4 human-gold corrective rows. There
#                                          is NO dedicated `name_invention` training
#                                          slice; the eval `name_invention` probe is
#                                          a NEGATIVE gazetteer (BANNED_INVENTED_NAMES
#                                          = Lona/Kinomiya/Torachance/... seeded "from
#                                          Ikenie4 judge synthesis", probes.py). The
#                                          training signal that teaches the model NOT
#                                          to hallucinate those names IS the Ikenie-4
#                                          gold corrective rows -> we use them as the
#                                          name-invention corrective proxy.
#   corrective_v11fix7:ikenie5:...     -> Ikenie-5 gold corrective (extra fidelity).
RS_PREFIX = "reverse_sense"
VOICE_PREFIX = "voice_probe"
IK4_PREFIX = "corrective_v11fix7:ikenie4"
IK5_PREFIX = "corrective_v11fix7:ikenie5"

# Caps on the name-invention proxy (Ikenie gold corrective) so it does not swamp
# the register/NSFW spread. reverse_sense + voice_probe are taken in FULL (tiny).
N_IK4_NAME = 48   # name-invention proxy (Ikenie-4 gold corrective, unique prompts)
N_IK5_EXTRA = 18  # a little Ikenie-5 fidelity spread

NSFW_TAGS = ["vn_eroge", "manga_nsfw"]
# normal-pool register mix mirrors the original calib's emphasis (register-heavy
# VN/manga dialogue + page CONTEXT, LN prose, SFX). garbage/synthetic excluded to
# match the original (clean-corpora) calib.
CTX_TAGS = ["manga_dialog", "vn", "vn_eroge", "manga"]
REG_TAGS = ["vn_eroge", "manga_nsfw", "vn"]       # plain register / NSFW dialogue
PROSE_TAGS = ["novel"]
SFX_TAGS = ["sfx"]


def _uniq_prompts(df: pl.DataFrame) -> list[str]:
    """Unique prompts in stable (first-seen) order."""
    seen: set[str] = set()
    out: list[str] = []
    for p in df["prompt"].to_list():
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _take(df: pl.DataFrame, n: int, used: set[str], seed: int) -> list[str]:
    """Shuffle, dedupe by prompt, skip already-used, take up to n."""
    if n <= 0 or df.height == 0:
        return []
    df = df.sample(fraction=1.0, shuffle=True, seed=seed)
    out: list[str] = []
    for p in df["prompt"].to_list():
        if p in used:
            continue
        used.add(p)
        out.append(p)
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="model dir for tokenizer/chat template (v11fix8 merged_fixed)")
    ap.add_argument("--parquet", default=str(DEFAULT_PARQUET), type=Path)
    ap.add_argument("--out", default=str(BACKEND / "scripts/quant/calib_v11fix8_int4.jsonl"),
                    type=Path)
    ap.add_argument("--n", type=int, default=384, help="total calibration samples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pl.read_parquet(args.parquet).select(
        ["prompt", "en", "src", "register_tag", "gold_flag"]
    )
    src = pl.col("src")
    tag = pl.col("register_tag")
    has_page = pl.col("prompt").str.contains("Page:")

    used: set[str] = set()
    buckets: list[tuple[str, list[str]]] = []

    # ---- 1. GUARANTEED corrective probes (deduped, taken first) --------------
    rs = _uniq_prompts(df.filter(src.str.starts_with(RS_PREFIX)))
    rs = [p for p in rs if p not in used and not used.add(p)]
    buckets.append(("reverse_sense", rs))

    voice = _uniq_prompts(df.filter(src.str.starts_with(VOICE_PREFIX)))
    voice = [p for p in voice if p not in used and not used.add(p)]
    buckets.append(("voice_probe", voice))

    ik4 = _take(df.filter(src.str.starts_with(IK4_PREFIX)), N_IK4_NAME, used, args.seed)
    buckets.append(("name_invention_proxy_ikenie4", ik4))

    ik5 = _take(df.filter(src.str.starts_with(IK5_PREFIX)), N_IK5_EXTRA, used, args.seed + 1)
    buckets.append(("ikenie5_fidelity", ik5))

    n_corr = sum(len(b) for _, b in buckets)

    # ---- 2. register-mirrored normal spread for the remainder ----------------
    remainder = max(args.n - n_corr, 0)
    n_ctx = round(remainder * 0.35)
    n_reg = round(remainder * 0.30)
    n_prose = round(remainder * 0.20)
    n_sfx = remainder - n_ctx - n_reg - n_prose  # ~15%

    is_corr = src.str.starts_with("corrective_") | src.str.starts_with(RS_PREFIX) \
        | src.str.starts_with(VOICE_PREFIX)

    ctx_df = df.filter(has_page & tag.is_in(CTX_TAGS) & ~is_corr)
    reg_df = df.filter(~has_page & tag.is_in(REG_TAGS) & ~is_corr)
    prose_df = df.filter(~has_page & tag.is_in(PROSE_TAGS) & ~is_corr)
    sfx_df = df.filter(tag.is_in(SFX_TAGS) & ~is_corr)

    buckets.append(("pagectx_register", _take(ctx_df, n_ctx, used, args.seed + 2)))
    buckets.append(("plain_register_nsfw", _take(reg_df, n_reg, used, args.seed + 3)))
    buckets.append(("ln_prose", _take(prose_df, n_prose, used, args.seed + 4)))
    buckets.append(("sfx", _take(sfx_df, n_sfx, used, args.seed + 5)))

    prompts = [p for _, b in buckets for p in b][: args.n]

    # ---- 3. chat-template wrap (byte-identical to calib_v11_int4.jsonl) ------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = []
    for p in prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        rows.append({"text": text})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- report --------------------------------------------------------------
    print(f"wrote {len(rows)} calibration prompts -> {args.out}")
    n_correctives = len(rs) + len(voice) + len(ik4) + len(ik5)
    print(f"corrective rows guaranteed: {n_correctives} "
          f"(reverse_sense={len(rs)} voice_probe={len(voice)} "
          f"name_invention_proxy_ikenie4={len(ik4)} ikenie5={len(ik5)})")
    print("bucket breakdown:")
    for name, b in buckets:
        print(f"  {name:30s} {len(b)}")
    print("--- sample[0] ---")
    print(rows[0]["text"][:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

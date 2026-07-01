"""Build a v11-format calibration set for INT4 W4A16 quantization of the
gemma4_e4b_v11_pagecontext translation model.

The model is a register-sensitive NSFW manga/VN translator. Register drift is the
#1 quant risk, so the calibration mix deliberately includes:
  * register-heavy VN dialogue (VNTL v3.1 — quote-marked, named-speaker turns)
  * light-novel prose sentences (ParallelFiction)
  * conversation/page CONTEXT prompts (windowed VNTL turns) in the v11 page shape
  * SFX / onomatopoeia single lines (jp-onomatopoeia JP->EN glosses)

Each sample is the FULL user message in one of the two v11 prompt shapes
(plain single-line OR page/conversation context), wrapped by the model's chat
template (tokenizer.apply_chat_template) into the exact string the model saw in
training. We emit ONLY the prompt (no assistant turn) — llmcompressor calibrates
on the forward pass over these prompts.

Output: a JSONL of {"text": <chat-templated prompt string>} rows.

NO eval-holdout data is touched (leakage guard): sources are strictly the
training calibration corpora named in the task.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

BACKEND = Path("/home/danny/Documents/personal/extension/backend")
DS = BACKEND / "training/datasets/translation"

# v11 prompt shapes (verbatim from scripts/data/v11/build_v11_dataset.py)
PLAIN_INSTR = "Translate the following Japanese to English. Output only the translation."
CONV_INSTR = (
    "Translate the marked line of this conversation from Japanese to English. "
    "Use the context for speakers, pronouns, and continuity. "
    "Output only the translation of the marked line."
)


def build_plain_prompt(jp: str) -> str:
    return f"{PLAIN_INSTR}\n\nJapanese: {jp}"


def build_context_prompt(instr: str, lines: list[str], k_idx: int) -> str:
    numbered = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))
    k = k_idx + 1
    return f"{instr}\n\nPage:\n{numbered}\n\nTranslate line {k}: {lines[k_idx]}"


# ---------------------------------------------------------------- source loaders
_JP_BLOCK = re.compile(r"<<JAPANESE>>\n(.*?)\n<<ENGLISH>>", re.DOTALL)


def load_vntl_dialogues(path: Path, max_groups: int) -> list[list[str]]:
    """Return a list of JP-line groups (each group = one VN conversation)."""
    import polars as pl

    df = pl.read_parquet(path)
    groups: list[list[str]] = []
    for row in df.head(max_groups * 2).iter_rows(named=True):
        text = row["text"]
        jp_lines = [m.strip() for m in _JP_BLOCK.findall(text)]
        # strip [speaker]: 「...」 wrapper down to the spoken JP but KEEP the
        # quote marks / speaker tags — that register signal is what we calibrate.
        jp_lines = [ln for ln in jp_lines if ln]
        if len(jp_lines) >= 2:
            groups.append(jp_lines)
        if len(groups) >= max_groups:
            break
    return groups


def load_vntl_chat(path: Path, n: int) -> list[str]:
    import polars as pl

    df = pl.read_parquet(path)
    out = []
    for row in df.head(n * 2).iter_rows(named=True):
        jp = (row.get("japanese") or "").strip()
        if jp:
            out.append(jp)
        if len(out) >= n:
            break
    return out


def load_parallelfiction(path: Path, n: int) -> list[str]:
    """src field is a newline-joined JP block; split into sentence-ish lines."""
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = row.get("src", "")
            for ln in src.split("\n"):
                ln = ln.strip()
                # skip the leading "NN.title" header line
                if ln and not re.match(r"^\d+\.", ln) and len(ln) > 4:
                    out.append(ln)
                if len(out) >= n:
                    return out
    return out


def load_sfx_glosses(path: Path, n: int) -> list[str]:
    """jp-onomatopoeia: dict {jp: [{english, details}, ...]} -> JP SFX strings."""
    d = json.load(open(path, encoding="utf-8"))
    keys = [k for k in d.keys() if k and len(k) <= 8]
    random.shuffle(keys)
    return keys[:n]


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model dir for tokenizer/chat template")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=384, help="total calibration samples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ---- gather raw JP material -------------------------------------------
    vntl_groups = load_vntl_dialogues(
        DS / "vn-ln-manga/lmg-anon__VNTL-v3.1-1k/data/train-00000-of-00001-eb879b20cbd4854b.parquet",
        max_groups=200,
    )
    vntl_chat = load_vntl_chat(
        DS / "vn-ln-manga/lmg-anon__vntl-chat/data/train-00000-of-00001-b2226dee7b9731eb.parquet",
        n=400,
    )
    pf_lines = load_parallelfiction(
        DS / "vn-ln-manga/NilanE__ParallelFiction-Ja_En-100k/dataset-Ja_En-Massive-v2.jsonl",
        n=600,
    )
    sfx = load_sfx_glosses(
        DS / "sfx-onomatopoeia/github-composite/jp-onomatopoeia/onomatopoeia.json",
        n=200,
    )

    print(
        f"raw pools: vntl_groups={len(vntl_groups)} vntl_chat={len(vntl_chat)} "
        f"pf_lines={len(pf_lines)} sfx={len(sfx)}"
    )

    # ---- build prompts (mix of shapes) ------------------------------------
    prompts: list[str] = []

    # (1) ~35% conversation-CONTEXT prompts from VNTL groups (register-heavy)
    n_ctx = int(args.n * 0.35)
    random.shuffle(vntl_groups)
    gi = 0
    while len([p for p in prompts]) < n_ctx and gi < len(vntl_groups):
        g = vntl_groups[gi]
        gi += 1
        # window up to 6 preceding + target; pick a target near the end
        w = g[-min(len(g), 7):]
        k_idx = len(w) - 1
        prompts.append(build_context_prompt(CONV_INSTR, w, k_idx))

    # (2) ~30% plain register dialogue (VNTL chat + VNTL group lines)
    plain_register = list(vntl_chat)
    for g in vntl_groups:
        plain_register.extend(g)
    random.shuffle(plain_register)
    n_reg = int(args.n * 0.30)
    prompts.extend(build_plain_prompt(jp) for jp in plain_register[:n_reg])

    # (3) ~20% LN prose plain lines
    n_pf = int(args.n * 0.20)
    random.shuffle(pf_lines)
    prompts.extend(build_plain_prompt(jp) for jp in pf_lines[:n_pf])

    # (4) ~15% SFX single lines
    n_sfx = args.n - len(prompts)
    prompts.extend(build_plain_prompt(jp) for jp in sfx[: max(n_sfx, 0)])

    random.shuffle(prompts)
    prompts = prompts[: args.n]

    # ---- chat-template wrap -----------------------------------------------
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

    print(f"wrote {len(rows)} calibration prompts -> {args.out}")
    print("--- sample[0] ---")
    print(rows[0]["text"][:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the v11fix7 CORRECTIVE SFT dataset.

Successor to v11fix6. Two upgrades, both enabled by THIS session's work:
  1. Uses the rebuilt **Qwen3-VL gold** for BOTH Ikenie 4 AND 5 (gold_q3.jsonl,
     ~1,355 rows vs v11fix6's 650) — 2x the corrective signal + a second chapter,
     joined for page-context against the v11fix6_box inspect dirs.
  2. Folds in the previously-built-but-NEVER-TRAINED targeted slices the dataset
     audit surfaced: voice_addressee_probe (causative/passive direction) and
     reverse_sense_corrective (sense-inversion: 締まる, 吸い出せ, 果てた, 風俗, ...),
     the latter with human_en authored here (was needs_gold=true).

Same train/serve-safe contract as v11fix6: corrective rows go through the SAME
build_context_prompt / build_plain_prompt builders (byte-exact with serving),
NSFW fraction stays a minority, upweight is modest.

Output: backend/scripts/data/v11fix7/data_v11fix7_pagecontext.parquet (+ sample/stats).
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[2]  # .../backend/scripts/data/v11fix7 -> .../backend
sys.path.insert(0, str(BACKEND / "scripts" / "data" / "v11fix6"))
sys.path.insert(0, str(BACKEND / "scripts" / "data" / "v11"))

# Reuse the proven v11fix6 pure helpers + the v11 prompt builders (train==serve).
from build_v11fix6_corrective import (  # noqa: E402
    to_sentence_case,
    jaccard,
    is_annotated_fragment,
    page_context_lines,
)
from build_v11_dataset import (  # noqa: E402
    build_context_prompt,
    build_plain_prompt,
    PAGE_INSTR,
    PLAIN_INSTR,
)

SEED = 42
COLS = ["prompt", "en", "src", "register_tag", "gold_flag"]
DIVERGENCE_JAC_THR = 0.55
ANCHOR_FRAC_OF_DIVERGENT = 0.5
PAGECTX_FRAC = 0.6
MAX_CONTEXT = 12
CORR_UPWEIGHT = 3   # gold corrective rows ~3x (minority of mix)
SLICE_UPWEIGHT = 4  # the tiny targeted probes get a slightly higher repeat

V11_PARQUET = BACKEND / "scripts/data/v11/data_v11_pagecontext.parquet"
OUT_DIR = BACKEND / "scripts/data/v11fix7"

# chapter slug -> (gold_q3, inspect dir for page context)
CHAPTERS = {
    "ikenie4": (
        BACKEND / "scripts/eval/data/ikenie4/gold_q3.jsonl",
        BACKEND / ".bench/ikenie4_v11fix6_box_insp",
    ),
    "ikenie5": (
        BACKEND / "scripts/eval/data/ikenie5/gold_q3.jsonl",
        BACKEND / ".bench/ikenie5_v11fix6_box_insp",
    ),
}

# Authored human_en for the reverse_sense slice (was needs_gold=true). Faithful,
# explicit register, keyed by the target jp line.
REVERSE_SENSE_EN = {
    "中がすごく締まってる": "Your insides are clenching so tight",
    "ぎゅっと締まってきた": "It's clenching down tight",
    "そんなに締めないで": "Don't clench down so hard",
    "全部吸い出せ": "Suck it all out",
    "中のを吸い出してやる": "I'll suck out what's inside you",
    "とうとう果ててしまった": "I finally came",
    "彼女の中で果てた": "I came inside her",
    "何度も果てさせられた": "I was made to cum over and over",
    "風俗で働いてるの": "I work in the sex trade",
    "風俗に行ったことある？": "Have you ever been to a brothel?",
    "騎乗位で動いて": "Move your hips in cowgirl",
    "今度は騎乗位がいい": "I want cowgirl next",
    "水で割って飲む": "I'll cut it with water and drink it",
    "お湯で割ろうか": "Shall I cut it with hot water?",
    "尻を突き出して": "Stick your ass out",
    "尻がエロい": "Your ass is so hot",
    "マンコが濡れてる": "Your pussy's wet",
    "マンコに挿れて": "Put it in my pussy",
}


def load_page_bubbles(insp: Path, page_num: int) -> list[dict] | None:
    f = insp / f"{page_num:03d}" / "bubbles.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text())
    except Exception:
        return None
    return data if isinstance(data, list) else None


def build_pagectx_for_row(src: str, jp: str, insp: Path) -> tuple[list[str], int] | None:
    m = re.match(r"[^:]+:p(\d+):idx(\d+)", src)
    if not m:
        return None
    page_num, target_idx = int(m.group(1)), int(m.group(2))
    bubbles = load_page_bubbles(insp, page_num)
    if not bubbles:
        return None
    lines = page_context_lines(bubbles)
    if not lines:
        return None
    idx_to_pos = {idx: pos for pos, (idx, _) in enumerate(lines)}
    if target_idx not in idx_to_pos:
        return None
    pos = idx_to_pos[target_idx]
    jp_lines = [j for _, j in lines]
    jp_lines[pos] = jp
    n = len(jp_lines)
    if n > MAX_CONTEXT:
        half = MAX_CONTEXT // 2
        lo = max(0, pos - half)
        hi = min(n, lo + MAX_CONTEXT)
        lo = max(0, hi - MAX_CONTEXT)
        jp_lines = jp_lines[lo:hi]
        pos = pos - lo
    return jp_lines, pos


def main() -> int:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. gold corrective rows (ch4 + ch5) ----
    corrective: list[dict] = []   # each: {jp, human, our, src, register_tag, insp}
    n_recased = 0
    per_chapter = {}
    for slug, (gold_path, insp) in CHAPTERS.items():
        gold = [json.loads(l) for l in gold_path.open()]
        clean = [
            r for r in gold
            if r.get("ocr_clean")
            and (r.get("jp") or "").strip()
            and (r.get("en") or "").strip()
            and not is_annotated_fragment(r["en"])
        ]
        divergent, anchors = [], []
        for r in clean:
            human_raw = r["en"].strip()
            human = to_sentence_case(human_raw)
            if human != human_raw:
                n_recased += 1
            if not human:
                continue
            our = (r.get("our_en") or "").strip()
            explicit = r.get("source_field") in ("worst_issues", "gap_examples")
            jac = jaccard(human, our) if our else 0.0
            rec = {
                "jp": r["jp"].strip(), "human": human, "our": our, "src": r["src"],
                "register_tag": r.get("register_tag", "manga_nsfw"), "insp": insp,
            }
            (divergent if (explicit or jac < DIVERGENCE_JAC_THR) else anchors).append(rec)
        rng.shuffle(anchors)
        kept = anchors[: int(len(divergent) * ANCHOR_FRAC_OF_DIVERGENT)]
        corrective += divergent + kept
        per_chapter[slug] = {"clean": len(clean), "divergent": len(divergent), "anchors_kept": len(kept)}

    # ---- shape gold corrective (page-context + plain) ----
    shaped: list[dict] = []
    n_pagectx = n_plain = 0
    for rec in corrective:
        want_ctx = rng.random() < PAGECTX_FRAC
        ctx = build_pagectx_for_row(rec["src"], rec["jp"], rec["insp"]) if want_ctx else None
        if ctx is not None:
            jp_lines, k = ctx
            shaped.append({
                "prompt": build_context_prompt(PAGE_INSTR, jp_lines, k),
                "en": rec["human"], "src": f"corrective_v11fix7:{rec['src']}:pagectx",
                "register_tag": rec["register_tag"], "gold_flag": True,
            })
            n_pagectx += 1
        else:
            shaped.append({
                "prompt": build_plain_prompt(rec["jp"]),
                "en": rec["human"], "src": f"corrective_v11fix7:{rec['src']}:plain",
                "register_tag": rec["register_tag"], "gold_flag": True,
            })
            n_plain += 1

    # ---- 2. targeted slices (prebuilt prompts; train them at last) ----
    slice_rows: list[dict] = []
    # voice_addressee_probe: ready (prompt + gold_en)
    for l in (BACKEND / "scripts/data/v11/voice_addressee_probe.jsonl").open():
        r = json.loads(l)
        en = (r.get("gold_en") or "").strip()
        if not en:
            continue
        slice_rows.append({"prompt": r["prompt"], "en": en, "src": r["src"],
                           "register_tag": r.get("register_tag", "vn_eroge"), "gold_flag": True})
    # reverse_sense_corrective: fill authored human_en by jp
    n_rs = 0
    for l in (BACKEND / "scripts/data/v11/reverse_sense_corrective.jsonl").open():
        r = json.loads(l)
        en = REVERSE_SENSE_EN.get(r["jp"], "").strip()
        if not en:
            continue
        slice_rows.append({"prompt": r["prompt"], "en": en, "src": r["src"],
                           "register_tag": r.get("register_tag", "vn_eroge"), "gold_flag": True})
        n_rs += 1

    # ---- 3. fold into the base v11 mix ----
    corr_df = pl.DataFrame(shaped * CORR_UPWEIGHT).select(COLS)
    slice_df = pl.DataFrame(slice_rows * SLICE_UPWEIGHT).select(COLS)
    base = pl.read_parquet(V11_PARQUET).select(COLS)
    mixed = pl.concat([base, corr_df, slice_df], how="vertical")

    out_parquet = OUT_DIR / "data_v11fix7_pagecontext.parquet"
    mixed.write_parquet(out_parquet)
    (OUT_DIR / "corrective_rows.sample.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in (shaped[:30] + slice_rows[:20]))
    )
    nsfw = mixed.filter(pl.col("register_tag").is_in(["vn_eroge", "manga_nsfw"])).height
    stats = {
        "base_rows": base.height,
        "gold_corrective_unique": len(shaped),
        "gold_corrective_pagectx": n_pagectx, "gold_corrective_plain": n_plain,
        "gold_corrective_upweighted": corr_df.height, "corr_upweight": CORR_UPWEIGHT,
        "slice_rows_unique": len(slice_rows), "slice_upweighted": slice_df.height,
        "slice_upweight": SLICE_UPWEIGHT, "reverse_sense_filled": n_rs,
        "recased": n_recased, "per_chapter": per_chapter,
        "total_rows": mixed.height, "nsfw_rows": nsfw, "nsfw_frac": round(nsfw / mixed.height, 4),
    }
    (OUT_DIR / "corrective_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_parquet} ({mixed.height} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

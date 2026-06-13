"""Phase 1 finalize: build chosen/rejected preference pairs from v10_5_scored.

Consumes ``v10_5_scored.parquet`` and produces
``v10_5_preferences.parquet`` with the schema expected by
``cpo_gemma4_e4b_v10_5.py``:

    [prompt, chosen, rejected, chosen_score, rejected_score, src,
     margin, chosen_kind, rejected_kind]

Per-row candidate construction
------------------------------
- Always include  (en_C1_gold,    cometkiwi_C1_gold,    "gold")
- If teacher_en  AND teacher_kiwi non-null:
        add   (teacher_en,     teacher_kiwi,         "teacher")
- If en_C3_onpolicy AND cometkiwi_C3_onpolicy non-null:
        add   (en_C3_onpolicy, cometkiwi_C3_onpolicy, "onpolicy")

chosen = argmax score
rejected = argmin score

Drop filters
------------
- chosen_score - rejected_score < 0.05
- len(chosen)/len(rejected) outside [0.5, 2.0]
- chosen or rejected text empty/whitespace-only

Prompt format (matches v10-it user message body)
------------------------------------------------
"Translate the following Japanese to English. Output only the translation."
"\\n\\nJapanese: <jp>"

The Gemma 4 chat template wrapping is applied by the training script via
``tok.apply_chat_template`` — we store only the bare user message body.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/home/danny/Documents/personal/extension")

SCORED = PROJECT_ROOT / "backend/scripts/data/cpo/v10_5_scored.parquet"
OUT = PROJECT_ROOT / "backend/scripts/data/cpo/v10_5_preferences.parquet"

PROMPT_TEMPLATE = (
    "Translate the following Japanese to English. "
    "Output only the translation.\n\nJapanese: {jp}"
)

MARGIN_MIN = 0.05
LEN_LO = 0.5
LEN_HI = 2.0


def is_valid_text(s) -> bool:
    if s is None:
        return False
    if isinstance(s, float) and math.isnan(s):
        return False
    s = str(s).strip()
    return bool(s)


def is_valid_score(x) -> bool:
    if x is None:
        return False
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return not math.isnan(f)


def main() -> int:
    print(f"[load] {SCORED}")
    df = pd.read_parquet(SCORED)
    n_in = len(df)
    print(f"[load] input rows: {n_in:,}")
    print(f"[load] cols: {list(df.columns)}")

    drop_reasons: Counter = Counter()
    out_rows = []
    margins = []
    chosen_kinds: Counter = Counter()
    rejected_kinds: Counter = Counter()

    for r in df.itertuples(index=False):
        # Build candidate list
        cands = []

        # C1 gold (always present)
        c1_text = getattr(r, "en_C1_gold", None)
        c1_score = getattr(r, "cometkiwi_C1_gold", None)
        if is_valid_text(c1_text) and is_valid_score(c1_score):
            cands.append((str(c1_text).strip(), float(c1_score), "gold"))

        # Teacher (Manga109 only)
        t_text = getattr(r, "teacher_en", None)
        t_score = getattr(r, "teacher_kiwi", None)
        if is_valid_text(t_text) and is_valid_score(t_score):
            cands.append((str(t_text).strip(), float(t_score), "teacher"))

        # C3 onpolicy
        op_text = getattr(r, "en_C3_onpolicy", None)
        op_score = getattr(r, "cometkiwi_C3_onpolicy", None)
        if is_valid_text(op_text) and is_valid_score(op_score):
            cands.append((str(op_text).strip(), float(op_score), "onpolicy"))

        if len(cands) < 2:
            drop_reasons["fewer_than_2_candidates"] += 1
            continue

        # Pick chosen / rejected
        cands_sorted = sorted(cands, key=lambda x: x[1])
        rej_text, rej_score, rej_kind = cands_sorted[0]
        cho_text, cho_score, cho_kind = cands_sorted[-1]

        if cho_text == rej_text:
            drop_reasons["chosen_equals_rejected"] += 1
            continue

        margin = cho_score - rej_score
        if margin < MARGIN_MIN:
            drop_reasons["margin_too_small"] += 1
            continue

        len_ratio = len(cho_text) / max(1, len(rej_text))
        if len_ratio < LEN_LO or len_ratio > LEN_HI:
            drop_reasons["length_ratio_out_of_range"] += 1
            continue

        jp = getattr(r, "jp", None)
        if not is_valid_text(jp):
            drop_reasons["empty_jp"] += 1
            continue

        prompt = PROMPT_TEMPLATE.format(jp=str(jp).strip())

        out_rows.append({
            "prompt": prompt,
            "chosen": cho_text,
            "rejected": rej_text,
            "chosen_score": cho_score,
            "rejected_score": rej_score,
            "margin": float(margin),
            "chosen_kind": cho_kind,
            "rejected_kind": rej_kind,
            "src": getattr(r, "src", ""),
        })
        margins.append(float(margin))
        chosen_kinds[cho_kind] += 1
        rejected_kinds[rej_kind] += 1

    n_out = len(out_rows)
    print()
    print(f"[result] input={n_in:,}  output={n_out:,}  ratio={n_out/max(1,n_in):.2%}")
    print()
    print("[drops] reasons:")
    for reason, cnt in drop_reasons.most_common():
        print(f"  {reason:30s} {cnt:>7,}")
    print()
    if margins:
        margins_sorted = sorted(margins)
        n = len(margins_sorted)
        def pct(p):
            return margins_sorted[min(n - 1, int(p * n))]
        print("[margin distribution]")
        print(f"  count : {n:,}")
        print(f"  mean  : {sum(margins) / n:.4f}")
        print(f"  min   : {margins_sorted[0]:.4f}")
        print(f"  p25   : {pct(0.25):.4f}")
        print(f"  p50   : {pct(0.50):.4f}")
        print(f"  p75   : {pct(0.75):.4f}")
        print(f"  p90   : {pct(0.90):.4f}")
        print(f"  max   : {margins_sorted[-1]:.4f}")
        print()
    print("[chosen_kind dist]")
    for k, v in chosen_kinds.most_common():
        print(f"  {k:10s} {v:>7,}")
    print()
    print("[rejected_kind dist]")
    for k, v in rejected_kinds.most_common():
        print(f"  {k:10s} {v:>7,}")

    if n_out == 0:
        print("[error] no output rows; aborting")
        return 1

    out_df = pd.DataFrame(out_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT, index=False)
    print()
    print(f"[wrote] {OUT}  ({len(out_df):,} rows)")

    # 5 sample triplets
    print()
    print("[samples]")
    for i, row in enumerate(out_rows[:5]):
        print(f"--- sample {i+1} ---")
        print(f"  src        : {row['src']}")
        print(f"  prompt     : {row['prompt'][:160]}")
        print(f"  chosen[{row['chosen_kind']:8s} score={row['chosen_score']:.4f}]: {row['chosen'][:160]}")
        print(f"  rejected[{row['rejected_kind']:8s} score={row['rejected_score']:.4f}]: {row['rejected'][:160]}")
        print(f"  margin     : {row['margin']:.4f}")

    # Stats sidecar
    stats = {
        "input_rows": int(n_in),
        "output_rows": int(n_out),
        "drop_reasons": dict(drop_reasons),
        "chosen_kind_dist": dict(chosen_kinds),
        "rejected_kind_dist": dict(rejected_kinds),
        "margin_min": MARGIN_MIN,
        "len_lo": LEN_LO,
        "len_hi": LEN_HI,
        "margin_mean": (sum(margins) / len(margins)) if margins else None,
        "margin_p50": pct(0.5) if margins else None,
        "margin_p90": pct(0.9) if margins else None,
    }
    stats_path = OUT.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, default=str))
    print(f"[wrote] {stats_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

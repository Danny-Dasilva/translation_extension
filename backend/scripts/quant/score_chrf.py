"""Score chrF++ (word_order=2) on a bench outputs.jsonl (en_pred vs en_ref).

Reports corpus chrF++ and per-sentence chrF++ (for paired bootstrap). Optionally
dumps a per_bubble-style json with 'chrf_pp' fields compatible with
paired_bs_chrf.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sacrebleu


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True, type=Path)
    ap.add_argument("--dump-per-bubble", type=Path, default=None)
    args = ap.parse_args()

    preds, refs = [], []
    rows = []
    with open(args.outputs, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            preds.append(r.get("en_pred", ""))
            refs.append(r.get("en_ref", ""))
            rows.append(r)

    chrf = sacrebleu.CHRF(word_order=2)  # chrF++
    corpus = chrf.corpus_score(preds, [refs]).score

    per = []
    for p, ref in zip(preds, refs):
        s = chrf.sentence_score(p, [ref]).score
        per.append(s)

    print(f"n={len(preds)}  corpus chrF++={corpus:.3f}  "
          f"mean sent chrF++={sum(per)/len(per):.3f}")

    if args.dump_per_bubble:
        out = [{"idx": i, "slug": str(rows[i].get("idx", i)),
                "jp": rows[i].get("jp", ""), "en_pred": preds[i],
                "en_ref": refs[i], "chrf_pp": per[i]} for i in range(len(preds))]
        args.dump_per_bubble.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"wrote per-bubble -> {args.dump_per_bubble}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

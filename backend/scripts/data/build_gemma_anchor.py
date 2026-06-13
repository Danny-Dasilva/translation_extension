"""Build (jp, en) anchor pairs from Gemma-4B teacher runs at
/home/danny/manga-output/644289-abc-gemma4-{base-45,uncensored-v2}/mode{A,B,C}.jsonl.

Merges all 6 configs, dedupes by (jp,en), keeps the longest non-'...' EN for each JP.
Preserves '...' rows as explicit garbage-refusal examples (tagged register=garbage).

Emits backend/training/datasets/filtered/gemma_anchor.parquet with unified schema.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unify_schema import make_row, write_parquet  # noqa: E402

SOURCES = [
    "/home/danny/manga-output/644289-abc-gemma4-base-45/modeA.jsonl",
    "/home/danny/manga-output/644289-abc-gemma4-base-45/modeB.jsonl",
    "/home/danny/manga-output/644289-abc-gemma4-base-45/modeC.jsonl",
    "/home/danny/manga-output/644289-abc-gemma4-uncensored-v2/modeA.jsonl",
    "/home/danny/manga-output/644289-abc-gemma4-uncensored-v2/modeB.jsonl",
    "/home/danny/manga-output/644289-abc-gemma4-uncensored-v2/modeC.jsonl",
]
OUT = Path("backend/training/datasets/filtered/gemma_anchor.parquet")


def main() -> int:
    # jp -> list of (en, source_tag, is_garbage)
    per_jp: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)

    for src in SOURCES:
        p = Path(src)
        if not p.exists():
            print(f"MISSING: {p}")
            continue
        variant = "unc" if "uncensored" in src else "base"
        mode = p.stem[-1]
        tag = f"gemma4_{variant}_{mode}"
        n = 0
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                jps = rec.get("jp_texts") or []
                ens = rec.get("en_texts") or []
                slug = rec.get("slug", "?")
                for bi, (jp, en) in enumerate(zip(jps, ens)):
                    jp = (jp or "").strip()
                    en = (en or "").strip()
                    if not jp or not en:
                        continue
                    is_garbage = en in ("...", "…", "")
                    per_jp[jp].append((en, f"{tag}:{slug}:{bi}", is_garbage))
                    n += 1
        print(f"{p.name}: {n} bubble translations loaded")

    rows = []
    garbage_rows = []
    for jp, cands in per_jp.items():
        # If >= 2 of 6 Gemma variants refused, this JP is garbage — emit as
        # refusal example. Also record any mode's non-"..." output as a manga
        # anchor so we get both signals.
        garbage_votes = sum(1 for _, _, g in cands if g)
        non_garbage = [(en, src) for en, src, g in cands if not g]

        if garbage_votes >= 2:
            garbage_rows.append((jp, "..."))
            # Don't also train as manga translation — avoid conflicting signal.
            continue

        if not non_garbage:
            continue

        # Pick the longest non-garbage EN as the "gold" anchor (longest = most info)
        best_en, best_src = max(non_garbage, key=lambda x: len(x[0]))
        rows.append(
            make_row(
                jp=jp,
                en=best_en,
                src=f"gemma_anchor:{best_src}",
                register_tag="manga",
                gold_flag=True,
            )
        )

    # Emit garbage examples
    for jp, en in garbage_rows:
        rows.append(
            make_row(
                jp=jp, en=en, src="gemma_anchor:garbage_consensus",
                register_tag="garbage", gold_flag=True,
            )
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(iter(rows), OUT)

    df = pl.read_parquet(OUT)
    print(f"wrote {len(df)} rows ({len(garbage_rows)} garbage) to {OUT}")
    print(f"registers: {df.group_by('register_tag').len().to_dict(as_series=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

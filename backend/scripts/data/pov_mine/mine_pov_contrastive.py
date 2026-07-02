#!/usr/bin/env python3
"""POV-contrastive training-data mine for v2 (targets the pronoun/POV ceiling).

WHY
---
The #1 MEASURED v1 failure is pronoun / point-of-view resolution on Japanese
pro-drop: even the fine-tuned v1 resolves only ~13-20% of gendered POV cases on
the Furube slice (thoughts/shared/research/2026-07-02_pipeline-audit-synthesis.md
§5, §6.1; memory: project_mt_finetuning_roadmap). This miner extracts the subset
of the mined bitext corpus whose ENGLISH reference carries an independently-
verifiable gender marker -- the rows where the model MUST resolve pro-drop to a
specific gender -- so a v2 SFT can be trained/upweighted on exactly that signal.

PRE-FILTER CONTRACT (reused, not reinvented)
--------------------------------------------
The presence pre-filter is ``pov_probe.required_family`` (backend/scripts/eval/
pov_probe.py) imported VERBATIM -- the SAME pure function the POV eval scores with.
A row is kept iff ``required_family(en)`` is not None, i.e. the EN reference
contains EXACTLY ONE gendered pronoun family ({he} XOR {she}). Rows that are
ungendered (0 families) or mixed (both he AND she) are dropped: they carry no
single verifiable POV target. This is a PRESENCE test on the reference, so every
kept row is independently POV-resolvable.

TRAIN==SERVE BYTE CONTRACT
--------------------------
Input rows are ALREADY in the project training schema
``[prompt, en, src, register_tag, gold_flag]`` produced by the byte-exact v11
builders (corpus_bitext/format_rows.build_pagectx_rows -> build_v11_dataset.
build_context_prompt). This miner is a FILTER + re-tag: ``prompt`` and ``en`` are
passed through VERBATIM (zero re-formatting => zero train/serve drift, the
documented ~95% chrF++ collapse landmine -- memory: feedback_chat_template_
mismatch). Only ``src`` is suffixed with ``:pov-<fam>`` for provenance. Output is
the identical 5-column schema; it drops straight into the existing SFT mix.

SOURCES (in the training schema; auto-detected)
-----------------------------------------------
* Locally-mined corpus bitext (DEFAULT, reachable today):
      /home/danny/manga_corpus_staging/curated/curated_rows.jsonl   (~2.3k rows)
* v11fix8 page-context backbone (302k rows), as a larger reachable proxy:
      backend/scripts/data/v11fix8/data_v11fix8_pagecontext.parquet
* The FULL mined corpus parquet (GPU-gated, not yet present):
      backend/scripts/data/corpus_bitext/data_corpus_bitext_pagecontext.parquet
  -- drop it in and point ``--source`` at it to run the full 375k-page mine.

USAGE
-----
    # small validation sample (200 input rows) over the local mined corpus
    .venv/bin/python scripts/data/pov_mine/mine_pov_contrastive.py --limit 200

    # full run over a specific source
    .venv/bin/python scripts/data/pov_mine/mine_pov_contrastive.py \
        --source scripts/data/corpus_bitext/data_corpus_bitext_pagecontext.parquet

Outputs (under scripts/data/pov_mine/, .parquet gitignored by scripts/data/**):
    pov_contrastive.parquet         (full training-schema parquet)
    pov_contrastive.sample.jsonl    (first N kept rows + pov_family sidecar field)
    pov_mine_stats.json             (yield accounting + projection)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

# --------------------------------------------------------------------------- #
# Paths + reuse the EXACT presence pre-filter from the eval harness.
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve()
BACKEND = _HERE.parents[3]  # .../backend  (pov_mine/data/scripts/backend)
for _p in (BACKEND, BACKEND / "scripts" / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# required_family / detect_families are pure (no heavy module-level imports).
from pov_probe import detect_families, required_family  # noqa: E402

TRAIN_COLS = ["prompt", "en", "src", "register_tag", "gold_flag"]

DEFAULT_SOURCE = Path("/home/danny/manga_corpus_staging/curated/curated_rows.jsonl")
FULL_MINED_PARQUET = (
    BACKEND / "scripts" / "data" / "corpus_bitext" / "data_corpus_bitext_pagecontext.parquet"
)
V11FIX8_BACKBONE = (
    BACKEND / "scripts" / "data" / "v11fix8" / "data_v11fix8_pagecontext.parquet"
)
OUT_DIR = _HERE.parent


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_source(path: Path) -> list[dict]:
    """Load a source in the training schema (.jsonl or .parquet) as row dicts."""
    if not path.exists():
        raise SystemExit(f"source not found: {path}")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".parquet":
        rows = pl.read_parquet(path).to_dicts()
    else:
        raise SystemExit(f"unsupported source type {path.suffix!r} (want .jsonl/.parquet)")
    if not rows:
        raise SystemExit(f"source is empty: {path}")
    missing = [c for c in ("prompt", "en") if c not in rows[0]]
    if missing:
        raise SystemExit(f"source rows missing required cols {missing}; got {list(rows[0].keys())}")
    return rows


# --------------------------------------------------------------------------- #
# Mine
# --------------------------------------------------------------------------- #
def mine(rows: list[dict]) -> tuple[list[dict], dict]:
    """Apply the presence pre-filter; return (kept_rows_train_schema, stats).

    A kept row is the input row PASSED THROUGH VERBATIM (prompt/en unchanged),
    with ``src`` suffixed ``:pov-<fam>`` and a sidecar ``pov_family`` field for
    analysis (dropped before the training parquet, kept in the sample.jsonl).
    """
    kept: list[dict] = []
    n_none = n_mixed = n_he = n_she = 0
    for r in rows:
        en = r.get("en") or ""
        fam = required_family(en)
        if fam is None:
            if detect_families(en):  # both families present -> mixed
                n_mixed += 1
            else:
                n_none += 1
            continue
        if fam == "he":
            n_he += 1
        else:
            n_she += 1
        out = dict(r)
        out["gold_flag"] = _as_bool(r.get("gold_flag", False))
        out["src"] = f"{r.get('src', 'unknown')}:pov-{fam}"
        out["pov_family"] = fam
        kept.append(out)
    total = len(rows)
    stats = {
        "input_rows": total,
        "kept_single_family": len(kept),
        "keep_rate": round(len(kept) / total, 4) if total else 0.0,
        "kept_he": n_he,
        "kept_she": n_she,
        "dropped_ungendered": n_none,
        "dropped_mixed_both": n_mixed,
    }
    return kept, stats


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #
def write_outputs(kept: list[dict], stats: dict, out_dir: Path, sample_n: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "pov_contrastive.parquet"
    sample_path = out_dir / "pov_contrastive.sample.jsonl"
    stats_path = out_dir / "pov_mine_stats.json"

    # Training parquet: the byte-identical 5-column schema ONLY (pov_family dropped).
    if kept:
        df = pl.DataFrame([{c: r[c] for c in TRAIN_COLS} for r in kept]).select(TRAIN_COLS)
    else:
        df = pl.DataFrame({c: [] for c in TRAIN_COLS})
    df.write_parquet(parquet_path)

    # Sample sidecar: first N kept rows WITH pov_family for eyeballing.
    with sample_path.open("w", encoding="utf-8") as f:
        for r in kept[:sample_n]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {**stats, "out_parquet": str(parquet_path), "out_sample": str(sample_path)}
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"parquet": parquet_path, "sample": sample_path, "stats": stats_path}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="POV-contrastive presence-filter mine (v2).")
    ap.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Training-schema .jsonl/.parquet to mine (default: {DEFAULT_SOURCE}).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only read the first N INPUT rows (small validation sample).",
    )
    ap.add_argument("--sample-n", type=int, default=25, help="Rows written to sample.jsonl.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument(
        "--projected-corpus-pages",
        type=int,
        default=375_000,
        help="Full-corpus page count for the yield projection line.",
    )
    args = ap.parse_args(argv)

    rows = load_source(args.source)
    if args.limit is not None:
        rows = rows[: args.limit]

    kept, stats = mine(rows)
    paths = write_outputs(kept, stats, args.out_dir, args.sample_n)

    # Projection: naive linear extrapolation of the keep_rate over the full corpus.
    # (rows-per-page varies; this is a scaffold estimate, not a promise.)
    rate = stats["keep_rate"]
    rows_per_page = stats["input_rows"] / max(1, _pages_in(args.source))
    proj = int(rate * rows_per_page * args.projected_corpus_pages)

    print(f"source              : {args.source}")
    print(f"input rows          : {stats['input_rows']}")
    print(f"KEEP (single family): {stats['kept_single_family']}  ({stats['keep_rate']*100:.1f}%)")
    print(f"    he / she        : {stats['kept_he']} / {stats['kept_she']}")
    print(f"drop ungendered     : {stats['dropped_ungendered']}")
    print(f"drop mixed (both)   : {stats['dropped_mixed_both']}")
    print(f"projected full mine : ~{proj:,} POV rows over {args.projected_corpus_pages:,} pages "
          f"(@ {rate*100:.1f}% keep, {rows_per_page:.1f} rows/page — scaffold estimate)")
    print(f"wrote               : {paths['parquet']}")
    print(f"                      {paths['sample']}")
    print(f"                      {paths['stats']}")
    return 0


def _pages_in(source: Path) -> int:
    """Best-effort distinct-page count from src keys (…:pNNN:…) for rows/page."""
    try:
        if source.suffix == ".jsonl":
            srcs = (json.loads(l).get("src", "") for l in source.read_text().splitlines() if l.strip())
        else:
            srcs = pl.read_parquet(source, columns=["src"])["src"].to_list()
        pages = set()
        for s in srcs:
            parts = str(s).split(":")
            page = next((p for p in parts if p.startswith("p") and p[1:4].isdigit()), None)
            pages.add((parts[1] if len(parts) > 1 else "?", page))
        return max(1, len(pages))
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

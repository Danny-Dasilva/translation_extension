"""Compose the final training mix per the plan's weight spec.

Reads per-source filtered parquets + a YAML weight spec, weighted-samples
to a target N (default 150k), writes one training parquet + a mix-summary JSON.

Weight spec YAML format:
    target_n: 150000
    sources:
      - name: vntl_v31_1k
        path: backend/training/datasets/filtered/vntl_v31_1k.parquet
        weight: 0.15
        gold: true
      - name: open_mantra
        path: backend/training/datasets/filtered/open_mantra_train.parquet
        weight: 0.10
        gold: true
      ...

Rules:
- Target per source = round(weight * target_n).
- Gold sources with fewer rows are oversampled with replacement.
- Non-gold sources with fewer rows are used as-is (no oversampling); the
  shortfall is logged and reflected in the mix summary. We do NOT redistribute
  that shortfall to other sources (keeps reproducibility).
- Output columns: unified schema (jp, en, src, register_tag, gold_flag).
  Extra columns from filtering (cometkiwi, labse_cos) are preserved if
  present in the parquet, via ``--keep-aux``.

Default weight spec (if --spec not given) mirrors the plan table.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from _cli_common import configure_logging, logger


# v7.1 mix — for Gemma 4 E4B fine-tune. Audit recommended:
#   - cap Gemma anchor to 1-2% (was 3% with 18x oversample → memorization risk)
#   - cap Open Mantra to 8% (was 12% with 8.4x → still high but safer for Gemma)
#   - ADD UUF NSFW SFX at 5% (new — 2907 pairs)
#   - boost garbage refusal to 5% (was 2% — Gemma needs more refusal training)
# All other v7 weights kept similar.
DEFAULT_SPEC_V7_1: dict[str, Any] = {
    "target_n": 150_000,
    "sources": [
        {"name": "vntl_v31_1k_train",
         "path": "backend/training/datasets/filtered/vntl_v31_1k_train.parquet",
         "weight": 0.23, "gold": True},
        {"name": "vntl_raw",
         "path": "backend/training/datasets/filtered/vntl_raw.parquet",
         "weight": 0.20, "gold": False},
        {"name": "open_mantra_train",
         "path": "backend/training/datasets/filtered/open_mantra_train.parquet",
         "weight": 0.08, "gold": True},
        {"name": "sfx_merged",
         "path": "backend/training/datasets/filtered/sfx_merged.parquet",
         "weight": 0.10, "gold": True},
        {"name": "uuf_sfx",
         "path": "backend/training/datasets/filtered/uuf_sfx.parquet",
         "weight": 0.05, "gold": True},
        {"name": "parallelfiction",
         "path": "backend/training/datasets/filtered/parallelfiction.parquet",
         "weight": 0.14, "gold": False},
        {"name": "aratako_synth",
         "path": "backend/training/datasets/filtered/aratako_synth.parquet",
         "weight": 0.08, "gold": False},
        {"name": "nilane_small",
         "path": "backend/training/datasets/filtered/nilane_small.parquet",
         "weight": 0.05, "gold": False},
        {"name": "gemma_manga_anchor",
         "path": "backend/training/datasets/filtered/gemma_manga_anchor.parquet",
         "weight": 0.02, "gold": True},
        {"name": "garbage_all",
         "path": "backend/training/datasets/filtered/garbage_all.parquet",
         "weight": 0.05, "gold": True},
    ],
}

DEFAULT_SPEC: dict[str, Any] = {
    "target_n": 150_000,
    "sources": [
        {
            "name": "vntl_v31_1k_train",
            "path": "backend/training/datasets/filtered/vntl_v31_1k_train.parquet",
            "weight": 0.25,
            "gold": True,
        },
        {
            "name": "vntl_raw",
            "path": "backend/training/datasets/filtered/vntl_raw.parquet",
            "weight": 0.20,
            "gold": False,
        },
        {
            "name": "open_mantra_train",
            "path": "backend/training/datasets/filtered/open_mantra_train.parquet",
            "weight": 0.12,
            "gold": True,
        },
        {
            "name": "sfx_merged",
            "path": "backend/training/datasets/filtered/sfx_merged.parquet",
            "weight": 0.10,
            "gold": True,
        },
        {
            "name": "parallelfiction",
            "path": "backend/training/datasets/filtered/parallelfiction.parquet",
            "weight": 0.15,
            "gold": False,
        },
        {
            "name": "aratako_synth",
            "path": "backend/training/datasets/filtered/aratako_synth.parquet",
            "weight": 0.08,
            "gold": False,
        },
        {
            "name": "nilane_small",
            "path": "backend/training/datasets/filtered/nilane_small.parquet",
            "weight": 0.05,
            "gold": False,
        },
        {
            # Gemma-4B teacher anchors on manga OCR — 248 unique bubbles. Small
            # weight prevents memorization; serves as a style anchor for
            # capitalization + stopping + manga register.
            "name": "gemma_manga_anchor",
            "path": "backend/training/datasets/filtered/gemma_manga_anchor.parquet",
            "weight": 0.03,
            "gold": True,
        },
        {
            # Teach refusal on OCR noise. Combined Gemma consensus + char-ratio
            # heuristic. ~23 unique examples — heavy oversample acceptable since
            # all map to the same output "...".
            "name": "garbage_all",
            "path": "backend/training/datasets/filtered/garbage_all.parquet",
            "weight": 0.02,
            "gold": True,
        },
        # JESC DROPPED — 100% lowercase, was 20% in v6 and polluted case distribution.
        # tatoeba DROPPED — HF loader broken (dataset scripts deprecated).
    ],
}


@dataclass
class SourceResult:
    name: str
    target: int
    available: int
    sampled: int
    gold: bool
    oversampled: bool
    path: str


def load_spec(spec_path: Path | None) -> dict[str, Any]:
    if spec_path is None:
        return DEFAULT_SPEC
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error("pyyaml not installed. `uv add --project backend pyyaml`")
        raise SystemExit(2) from e
    with spec_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def compose(
    spec: dict[str, Any], *, seed: int = 42, keep_aux: bool = False
) -> tuple[pl.DataFrame, list[SourceResult]]:
    target_n = int(spec["target_n"])
    sources = spec["sources"]
    total_weight = sum(float(s["weight"]) for s in sources)
    if abs(total_weight - 1.0) > 1e-6:
        logger.warning(
            f"weights sum to {total_weight:.4f}, not 1.0 — renormalising per source."
        )

    frames: list[pl.DataFrame] = []
    results: list[SourceResult] = []
    for s in sources:
        name = s["name"]
        path = Path(s["path"])
        weight = float(s["weight"]) / total_weight
        target = int(round(weight * target_n))
        gold = bool(s.get("gold", False))

        if not path.exists():
            logger.warning(f"{name}: path missing, skipping: {path}")
            results.append(
                SourceResult(name, target, 0, 0, gold, False, str(path))
            )
            continue
        df = pl.read_parquet(path)
        available = len(df)
        if available == 0:
            logger.warning(f"{name}: empty parquet")
            results.append(
                SourceResult(name, target, 0, 0, gold, False, str(path))
            )
            continue

        oversampled = False
        if available >= target:
            sampled = df.sample(n=target, with_replacement=False, seed=seed)
        else:
            if gold:
                sampled = df.sample(n=target, with_replacement=True, seed=seed)
                oversampled = True
                logger.info(
                    f"{name}: gold oversample {available} -> {target} (with replacement)"
                )
            else:
                sampled = df
                logger.warning(
                    f"{name}: non-gold shortfall — wanted {target}, have {available}"
                )

        frames.append(sampled)
        results.append(
            SourceResult(name, target, available, len(sampled), gold, oversampled, str(path))
        )

    if not frames:
        logger.error("no sources produced rows — nothing to compose")
        return pl.DataFrame(), results

    # Align columns: reduce to intersection unless keep_aux.
    if keep_aux:
        merged = pl.concat(frames, how="diagonal_relaxed")
    else:
        required = ["jp", "en", "src", "register_tag", "gold_flag"]
        trimmed = [f.select([c for c in required if c in f.columns]) for f in frames]
        merged = pl.concat(trimmed, how="vertical_relaxed")

    # Shuffle.
    merged = merged.sample(fraction=1.0, shuffle=True, seed=seed)
    return merged, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=None, help="YAML spec; falls back to plan default.")
    parser.add_argument(
        "--out",
        default="backend/training/runs/manga-bubbles/data.parquet",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Mix-summary JSON (default: <out>.mix-summary.json).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-aux",
        action="store_true",
        help="Keep aux columns (cometkiwi, labse_cos) if present via diagonal concat.",
    )
    parser.add_argument("--target-n", type=int, default=None, help="Override spec.target_n")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    spec = load_spec(Path(args.spec) if args.spec else None)
    if args.target_n:
        spec["target_n"] = args.target_n

    merged, results = compose(spec, seed=args.seed, keep_aux=args.keep_aux)

    summary = {
        "target_n": spec["target_n"],
        "actual_n": len(merged),
        "seed": args.seed,
        "sources": [
            {
                "name": r.name,
                "target": r.target,
                "available": r.available,
                "sampled": r.sampled,
                "gold": r.gold,
                "oversampled": r.oversampled,
                "path": r.path,
            }
            for r in results
        ],
    }
    logger.info(f"mix summary: actual_n={summary['actual_n']} target={summary['target_n']}")

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out_path)
    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(
        ".mix-summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(merged)} rows to {out_path} (summary: {summary_path})")


if __name__ == "__main__":
    main()

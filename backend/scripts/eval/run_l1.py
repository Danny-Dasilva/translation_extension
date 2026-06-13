"""L1 - fast in-training eval.

Runs every 500 training steps on a 200-row dev JSONL:

- chrF++ via ``sacrebleu.corpus_chrf(chrf_word_order=2)``
- CometKiwi-22 via ``Unbabel/wmt22-cometkiwi-da`` (batch 64, src-only)
- Length ratio = mean(len(en_pred)/len(en_ref))

Target wall-time: <3 min on RTX 5090.

CLI:
    python -m backend.scripts.eval.run_l1 \
        --checkpoint path/to/ckpt \
        --dev        path/to/dev.jsonl \
        --step       1500 \
        [--wandb] [--out l1.json] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_chrf(preds: list[str], refs: list[str]) -> float:
    """chrF++ (word_order=2) corpus score."""
    import sacrebleu

    score = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    return float(score.score)


def compute_cometkiwi(
    jp_src: list[str],
    preds: list[str],
    *,
    batch_size: int = 64,
    model_name: str = "Unbabel/wmt22-cometkiwi-da",
) -> float:
    """Reference-free CometKiwi-22 system score."""
    from comet import download_model, load_from_checkpoint  # lazy

    logger.info("Downloading/loading {}", model_name)
    ckpt = download_model(model_name)
    model = load_from_checkpoint(ckpt)
    data = [{"src": s, "mt": m} for s, m in zip(jp_src, preds)]
    prediction = model.predict(data, batch_size=batch_size, gpus=1)
    return float(prediction.system_score)


def compute_length_ratio(preds: list[str], refs: list[str]) -> float:
    if not preds or not refs:
        return float("nan")
    ratios: list[float] = []
    for p, r in zip(preds, refs):
        if not r:
            continue
        ratios.append(len(p) / max(len(r), 1))
    if not ratios:
        return float("nan")
    return sum(ratios) / len(ratios)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="L1 fast eval: chrF++ + CometKiwi-22.")
    p.add_argument("--checkpoint", type=Path, required=True, help="HF checkpoint or LoRA adapter dir.")
    p.add_argument("--dev", type=Path, required=True, help="Dev JSONL with {jp, en_ref}.")
    p.add_argument("--step", type=int, required=True, help="Training step number.")
    p.add_argument("--out", type=Path, default=None, help="Optional output JSON path.")
    p.add_argument("--batch-size", type=int, default=32, help="Generation batch size.")
    p.add_argument("--cometkiwi-batch-size", type=int, default=64)
    p.add_argument("--wandb", action="store_true", help="Log to W&B if configured.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rows = _read_jsonl(args.dev)

    jp_src = [r["jp"] for r in rows]
    en_ref = [r.get("en_ref", "") for r in rows]

    from backend.scripts.eval._generation import generate  # lazy

    logger.info("Generating {} translations...", len(jp_src))
    preds = generate(args.checkpoint, jp_src, batch_size=args.batch_size)

    # Augment rows so callers can pipe preds into probes.py / vntl_cosine.py.
    for row, pred in zip(rows, preds):
        row["en_pred"] = pred

    chrf = compute_chrf(preds, en_ref)
    logger.info("chrF++ = {:.4f}", chrf)

    try:
        cometkiwi = compute_cometkiwi(jp_src, preds, batch_size=args.cometkiwi_batch_size)
    except Exception as e:
        logger.exception("CometKiwi failed: {}", e)
        cometkiwi = float("nan")
    logger.info("CometKiwi-22 = {:.4f}", cometkiwi)

    length_ratio = compute_length_ratio(preds, en_ref)
    logger.info("length_ratio = {:.4f}", length_ratio)

    result = {
        "step": args.step,
        "chrf": chrf,
        "cometkiwi": cometkiwi,
        "length_ratio": length_ratio,
    }

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    if args.wandb:
        try:
            import wandb  # lazy

            if wandb.run is None:
                logger.warning("--wandb passed but no active wandb run; skipping log.")
            else:
                wandb.log({f"l1/{k}": v for k, v in result.items() if k != "step"}, step=args.step)
        except Exception as e:
            logger.warning("W&B log failed: {}", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())

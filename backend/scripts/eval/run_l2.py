"""L2 - per-epoch full eval.

Runs:

- MetricX-24-Hybrid-Large (``google/metricx-24-hybrid-large-v2p6-bfloat16``)
- XCOMET-XL (``Unbabel/XCOMET-XL``)

...on each of FLORES-1012, VNTL-128, and Open Mantra test.  Also runs
``vntl_cosine.py`` on VNTL-128 via subprocess.

Wall-time budget: ~30 min on RTX 5090.

CLI:
    python -m backend.scripts.eval.run_l2 \
        --checkpoint path/to/ckpt \
        --held-out-dir backend/training/eval_held_out \
        --out l2.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

L2_DATASETS = {
    "flores_ja_en": "flores_ja_en.jsonl",
    "vntl128": "vntl128.jsonl",
    "open_mantra_test": "open_mantra_test.jsonl",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_metricx(
    jp_src: list[str],
    preds: list[str],
    refs: list[str],
    *,
    batch_size: int = 32,
    model_name: str = "google/metricx-24-hybrid-large-v2p6-bfloat16",
) -> float:
    """Run MetricX-24-Hybrid-Large on (src, mt, ref) triples.  Lower is better.

    The MetricX repo exposes a scoring harness via the ``mt-metrics-eval`` /
    official release; we call the transformers model directly with the
    documented input format ``source: ... candidate: ... reference: ...``.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    logger.info("Loading MetricX model {}", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    scores: list[float] = []
    for i in range(0, len(preds), batch_size):
        batch_src = jp_src[i : i + batch_size]
        batch_mt = preds[i : i + batch_size]
        batch_ref = refs[i : i + batch_size]
        inputs = [
            f"source: {s} candidate: {m} reference: {r}"
            for s, m, r in zip(batch_src, batch_mt, batch_ref)
        ]
        enc = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(
            model.device
        )
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=8)
        for seq in out:
            text = tokenizer.decode(seq, skip_special_tokens=True).strip()
            try:
                scores.append(float(text))
            except ValueError:
                scores.append(float("nan"))

    finite = [s for s in scores if s == s]  # filter NaN
    return sum(finite) / len(finite) if finite else float("nan")


def run_xcomet_xl(
    jp_src: list[str],
    preds: list[str],
    refs: list[str],
    *,
    batch_size: int = 32,
    model_name: str = "Unbabel/XCOMET-XL",
) -> float:
    from comet import download_model, load_from_checkpoint

    logger.info("Loading {}", model_name)
    ckpt = download_model(model_name)
    model = load_from_checkpoint(ckpt)
    data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(jp_src, preds, refs)]
    prediction = model.predict(data, batch_size=batch_size, gpus=1)
    return float(prediction.system_score)


def _run_vntl_cosine(pred_path: Path, out_path: Path) -> dict[str, Any]:
    logger.info("Running vntl_cosine subprocess...")
    cmd = [
        sys.executable,
        "-m",
        "backend.scripts.eval.vntl_cosine",
        "--predictions",
        str(pred_path),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    with out_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="L2 per-epoch eval.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--held-out-dir",
        type=Path,
        default=Path("backend/training/eval_held_out"),
    )
    p.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["metricx", "xcomet_xl", "vntl_cosine"],
        help="Metrics to skip (e.g. if CUDA OOM).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from backend.scripts.eval._generation import generate  # lazy

    report: dict[str, Any] = {"datasets": {}}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for ds_name, filename in L2_DATASETS.items():
            path = args.held_out_dir / filename
            if not path.exists():
                logger.warning("Missing held-out file: {}", path)
                continue
            rows = _read_jsonl(path)
            jp_src = [r["jp"] for r in rows]
            refs = [r.get("en_ref", "") or "" for r in rows]

            logger.info("[{}] generating n={} translations...", ds_name, len(jp_src))
            preds = generate(args.checkpoint, jp_src, batch_size=args.batch_size)

            ds_rec: dict[str, Any] = {"n": len(rows)}
            if "metricx" not in args.skip:
                try:
                    ds_rec["metricx24_hybrid_large"] = run_metricx(
                        jp_src, preds, refs, batch_size=args.batch_size
                    )
                except Exception as e:
                    logger.exception("MetricX failed on {}: {}", ds_name, e)
                    ds_rec["metricx24_hybrid_large"] = None

            if "xcomet_xl" not in args.skip:
                try:
                    ds_rec["xcomet_xl"] = run_xcomet_xl(
                        jp_src, preds, refs, batch_size=args.batch_size
                    )
                except Exception as e:
                    logger.exception("XCOMET-XL failed on {}: {}", ds_name, e)
                    ds_rec["xcomet_xl"] = None

            # VNTL cosine only meaningful on vntl128.
            if ds_name == "vntl128" and "vntl_cosine" not in args.skip:
                pred_rows: list[dict[str, Any]] = []
                for r, p in zip(rows, preds):
                    pr = dict(r)
                    pr["en_pred"] = p
                    pred_rows.append(pr)
                pred_path = tmp_dir / "vntl128_preds.jsonl"
                _write_jsonl(pred_rows, pred_path)
                cosine_out = tmp_dir / "vntl_cosine.json"
                try:
                    ds_rec["vntl_cosine"] = _run_vntl_cosine(pred_path, cosine_out)
                except Exception as e:
                    logger.exception("vntl_cosine failed: {}", e)
                    ds_rec["vntl_cosine"] = None

            report["datasets"][ds_name] = ds_rec

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote L2 report to {}", args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())

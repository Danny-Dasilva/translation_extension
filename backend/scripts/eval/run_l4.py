"""L4 - release-candidate eval.

Runs XCOMET-XXL (``Unbabel/XCOMET-XXL``) quantized (3-bit if the
``unbabel-comet`` library supports it, else 8-bit via bitsandbytes with a
warning) + GEMBA-MQM over every held-out JSONL, then writes a consolidated
Markdown report.

CLI:
    python -m backend.scripts.eval.run_l4 \
        --checkpoint path/to/ckpt \
        --held-out-dir backend/training/eval_held_out \
        --out-dir backend/training/runs/manga-bubbles/
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

L4_DATASETS = [
    ("VNTL-128", "vntl128.jsonl"),
    ("FLORES-1012", "flores_ja_en.jsonl"),
    ("Open Mantra test", "open_mantra_test.jsonl"),
    ("Probes", "probes.jsonl"),
    ("Regression canary", "regression_canary.jsonl"),
]


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


def _load_xcomet_xxl_quantized(model_name: str = "Unbabel/XCOMET-XXL") -> tuple[Any, str]:
    """Return (model, quant_mode) where quant_mode is '3bit' or '8bit'.

    The ``unbabel-comet`` library has limited built-in quantization support.
    We try a 3-bit path first (if exposed by comet >= 2.2 via
    ``load_from_checkpoint(..., load_in_4bit=True, bnb_4bit_quant_type='nf4')``
    and via a bitsandbytes 3-bit equivalent), and fall back to 8-bit.
    """
    from comet import download_model, load_from_checkpoint  # lazy

    ckpt = download_model(model_name)

    # Attempt 3-bit path; this may or may not be supported depending on the
    # comet version.  We catch broadly and fall back to 8-bit.
    try:
        import bitsandbytes as bnb  # noqa: F401

        model = load_from_checkpoint(ckpt, reload_hparams=True)
        # If comet exposes a quantize helper:
        if hasattr(model, "quantize"):
            try:
                model.quantize(bits=3)
                return model, "3bit"
            except Exception as e:
                logger.warning("3-bit quantize failed ({}); falling back to 8-bit.", e)
                try:
                    model.quantize(bits=8)
                    return model, "8bit"
                except Exception as e2:
                    logger.warning("8-bit quantize also failed ({}); using fp16.", e2)
                    return model, "fp16"
        logger.warning(
            "unbabel-comet does not expose quantize(); "
            "running XCOMET-XXL unquantized - this may OOM on 5090."
        )
        return model, "fp16"
    except Exception as e:
        logger.warning("bitsandbytes not available ({}); loading fp16.", e)
        model = load_from_checkpoint(ckpt)
        return model, "fp16"


def run_xcomet_xxl(
    jp_src: list[str],
    preds: list[str],
    refs: list[str],
    *,
    batch_size: int = 8,
) -> tuple[float, str]:
    model, quant = _load_xcomet_xxl_quantized()
    data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(jp_src, preds, refs)]
    prediction = model.predict(data, batch_size=batch_size, gpus=1)
    return float(prediction.system_score), quant


def _run_gemba_subprocess(
    pred_path: Path, ref_path: Path, out_path: Path, judge: str
) -> dict[str, Any]:
    logger.info("Running GEMBA-MQM subprocess ({})", judge)
    cmd = [
        sys.executable,
        "-m",
        "backend.scripts.eval.gemba_mqm_judge",
        "--predictions",
        str(pred_path),
        "--references",
        str(ref_path),
        "--judge",
        judge,
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    with out_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _run_probes_subprocess(
    pred_path: Path, probe_path: Path, out_path: Path
) -> dict[str, Any]:
    logger.info("Running probes subprocess")
    cmd = [
        sys.executable,
        "-m",
        "backend.scripts.eval.probes",
        "--predictions",
        str(pred_path),
        "--probes",
        str(probe_path),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=False)
    with out_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_markdown_report(report: dict[str, Any], out_md: Path) -> None:
    lines: list[str] = []
    lines.append(f"# L4 Release-Candidate Eval Report\n")
    lines.append(f"- Generated: `{report['timestamp']}`")
    lines.append(f"- Checkpoint: `{report['checkpoint']}`")
    lines.append(f"- XCOMET-XXL quantization: `{report.get('xcomet_quant', 'n/a')}`")
    lines.append("")

    lines.append("## Per-dataset scores\n")
    lines.append("| Dataset | N | XCOMET-XXL | GEMBA-MQM mean | GEMBA major | GEMBA minor |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for ds_name, rec in report["datasets"].items():
        x = rec.get("xcomet_xxl")
        g = rec.get("gemba_mqm") or {}
        lines.append(
            f"| {ds_name} | {rec.get('n', '-')} | "
            f"{x if x is not None else '-':.4f} | "
            f"{g.get('mean_score', float('nan')):.3f} | "
            f"{g.get('total_major', '-')} | {g.get('total_minor', '-')} |"
        )

    if "probes" in report:
        lines.append("")
        lines.append("## Probe suite (L3)\n")
        lines.append("| Probe | Pass-rate | N |")
        lines.append("|---|---:|---:|")
        for probe, rate in report["probes"]["per_probe"].items():
            n = report["probes"]["per_probe_counts"][probe]["n"]
            rate_str = f"{rate:.3f}" if rate == rate else "nan"  # NaN filter
            lines.append(f"| {probe} | {rate_str} | {n} |")
        lines.append(f"\n**Overall pass:** `{report['probes']['overall_pass']}`")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote Markdown report to {}", out_md)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="L4 release-candidate eval.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--held-out-dir",
        type=Path,
        default=Path("backend/training/eval_held_out"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("backend/training/runs/manga-bubbles"),
    )
    p.add_argument("--judge", type=str, default="qwen25-72b-4bit")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["xcomet_xxl", "gemba_mqm", "probes"],
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from backend.scripts.eval._generation import generate  # lazy

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    report: dict[str, Any] = {
        "timestamp": timestamp,
        "checkpoint": str(args.checkpoint),
        "datasets": {},
        "xcomet_quant": None,
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for ds_label, filename in L4_DATASETS:
            path = args.held_out_dir / filename
            if not path.exists():
                logger.warning("Missing held-out file: {}", path)
                continue
            rows = _read_jsonl(path)
            jp_src = [r["jp"] for r in rows]
            refs = [r.get("en_ref", "") or "" for r in rows]

            logger.info("[{}] generating n={} translations...", ds_label, len(jp_src))
            preds = generate(args.checkpoint, jp_src, batch_size=args.batch_size)

            pred_rows: list[dict[str, Any]] = []
            for r, p in zip(rows, preds):
                pr = dict(r)
                pr["en_pred"] = p
                pred_rows.append(pr)
            pred_path = tmp_dir / f"{filename}.preds.jsonl"
            _write_jsonl(pred_rows, pred_path)

            ds_rec: dict[str, Any] = {"n": len(rows)}

            if "xcomet_xxl" not in args.skip and filename != "probes.jsonl":
                try:
                    score, quant = run_xcomet_xxl(jp_src, preds, refs, batch_size=8)
                    ds_rec["xcomet_xxl"] = score
                    report["xcomet_quant"] = quant
                except Exception as e:
                    logger.exception("XCOMET-XXL failed on {}: {}", ds_label, e)
                    ds_rec["xcomet_xxl"] = None

            if "gemba_mqm" not in args.skip and filename != "probes.jsonl":
                gemba_out = tmp_dir / f"{filename}.gemba.json"
                try:
                    ds_rec["gemba_mqm"] = _run_gemba_subprocess(
                        pred_path, path, gemba_out, args.judge
                    )
                except Exception as e:
                    logger.exception("GEMBA-MQM failed on {}: {}", ds_label, e)
                    ds_rec["gemba_mqm"] = None

            # Probes get run on the probes.jsonl file only.
            if filename == "probes.jsonl" and "probes" not in args.skip:
                probe_out = tmp_dir / "probes.json"
                report["probes"] = _run_probes_subprocess(pred_path, path, probe_out)

            report["datasets"][ds_label] = ds_rec

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"l4_report_{timestamp}.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    out_md = out_dir / f"l4_report_{timestamp}.md"
    _write_markdown_report(report, out_md)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Aggregate analysis across the entire pipeline-e2e/ gallery.

Computes:
  - total images, total blocks, total text lines
  - latency p50/p95 per stage (detect/ocr/inpaint/translate)
  - failure rate per observed failure mode (raw-JP fallback, zero-detection,
    OCR repetition, all-filtered)
  - bubble-count distribution

Saves a Markdown report to thoughts/koharu-improvements/gallery-analysis.md.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def p(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * q
    f = int(k)
    c = min(f + 1, len(values) - 1)
    return values[f] + (values[c] - values[f]) * (k - f)


def is_raw_jp_fallback(ocr_samples: list[str], translations: list[str]) -> bool:
    """Returns True if at least one translation equals its OCR (meaning the LLM
    returned the source string unchanged — the batched-translate failure mode).
    """
    for jp, en in zip(ocr_samples, translations):
        if jp.strip() and en.strip() == jp.strip():
            return True
    return False


def is_ocr_stuck(ocr_samples: list[str]) -> bool:
    """Heuristic: any OCR output with ≥5 same-char run OR trigram-repeat count ≥4.
    Mirrors the in-service repetition guard but applied post-hoc here.
    """
    for s in ocr_samples:
        if re.search(r"(.)\1{4,}", s):
            return True
        for k in range(len(s) - 12):
            tri = s[k:k + 3]
            if s.count(tri) >= 4 and tri.strip():
                return True
    return False


def main():
    gal = REPO_ROOT / "thoughts" / "koharu-improvements" / "pipeline-e2e"
    dirs = sorted(
        p for p in gal.iterdir()
        if p.is_dir() and p.name not in ("features",)
    )

    total_images = 0
    total_blocks = 0
    total_lines = 0
    stage_ms = {"detect": [], "ocr": [], "inpaint": [], "translate": []}
    block_counts = []

    failures = {
        "zero_detect": [],
        "raw_jp_fallback": [],
        "ocr_stuck": [],
    }

    for d in dirs:
        stats = d / "stats.json"
        if not stats.exists():
            continue
        total_images += 1
        j = json.loads(stats.read_text())
        blocks = j.get("num_blocks", 0)
        total_blocks += blocks
        total_lines += j.get("num_text_lines", 0)
        block_counts.append(blocks)
        for stage in stage_ms:
            key = f"{stage}_ms"
            v = j.get(key)
            if isinstance(v, (int, float)) and v > 0:
                stage_ms[stage].append(v)
        ocr = j.get("ocr_samples", []) or []
        trs = j.get("translations", []) or []
        if blocks == 0:
            failures["zero_detect"].append(d.name)
        if is_raw_jp_fallback(ocr, trs):
            failures["raw_jp_fallback"].append(d.name)
        if is_ocr_stuck(ocr):
            failures["ocr_stuck"].append(d.name)

    lines = [
        "# Pipeline e2e gallery — aggregate analysis",
        "",
        f"**Gallery size:** {total_images} images",
        f"**Total blocks detected:** {total_blocks}  (avg {total_blocks/total_images:.1f}/page)" if total_images else "",
        f"**Total text lines:** {total_lines}",
        "",
        "## Stage latency (ms, CPU-only run)",
        "",
        "| Stage | p50 | p95 | mean | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, vs in stage_ms.items():
        if not vs:
            continue
        lines.append(
            f"| {stage} | {p(vs, 0.5):.0f} | {p(vs, 0.95):.0f} | "
            f"{statistics.mean(vs):.0f} | {len(vs)} |"
        )

    lines += [
        "",
        "## Block count distribution",
        "",
        f"- min: {min(block_counts) if block_counts else 0}",
        f"- median: {p(block_counts, 0.5):.0f}",
        f"- max: {max(block_counts) if block_counts else 0}",
        "",
        "## Observed failure rates",
        "",
        "| Mode | Count | Rate | Instances |",
        "|---|---:|---:|---|",
    ]
    for mode, hits in failures.items():
        rate = len(hits) / total_images if total_images else 0
        lines.append(
            f"| {mode} | {len(hits)} | {rate*100:.1f}% | "
            f"{', '.join(hits[:5])}{' …' if len(hits) > 5 else ''} |"
        )

    lines += [
        "",
        "## Per-image bubble counts",
        "",
        "| Image | Blocks |",
        "|---|---:|",
    ]
    for d in dirs:
        stats = d / "stats.json"
        if stats.exists():
            j = json.loads(stats.read_text())
            lines.append(f"| {d.name} | {j.get('num_blocks', 0)} |")

    out = REPO_ROOT / "thoughts" / "koharu-improvements" / "gallery-analysis.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print(f"\nsummary: {total_images} images, {total_blocks} blocks")
    print("failures:")
    for mode, hits in failures.items():
        print(f"  {mode}: {len(hits)}/{total_images} ({len(hits)/total_images*100:.1f}%)")


if __name__ == "__main__":
    main()

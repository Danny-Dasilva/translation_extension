"""Post-process predictions to trim generation-loop garbage, then recompute metrics."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import sacrebleu
from loguru import logger
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "backend/training/runs/manga-bubbles"

# Common repeat patterns: " (Name) (Name)", "\r\n(Name):", "\n[Name]:"
# Base model without EOS keeps going. Heuristics to truncate.
SPEAKER_TAG_RE = re.compile(r"\s+(?=[\(\[][A-Za-z][^\)\]]{0,20}[\)\]]:?\s)")
NEWLINE_RE = re.compile(r"[\r\n]")
REPEAT_NGRAM_RE = re.compile(r"(\b.{3,40}?\b)(?:\s*\1){2,}")


def clean(text: str) -> str:
    """Trim generation-loop suffix heuristically."""
    # 1. First newline wins.
    text = NEWLINE_RE.split(text, maxsplit=1)[0]
    # 2. Cut at first occurrence of " (Something) " or " [Something]" AFTER the first such marker
    # (i.e. the second speaker tag onwards is garbage).
    parts = SPEAKER_TAG_RE.split(text)
    if len(parts) > 2:
        # Keep first marker (speaker prefix) + content, drop everything after the 2nd marker
        text = parts[0] + parts[1] if parts[1].startswith(("(", "[")) else parts[0]
    # 3. Collapse ngram loops.
    prev = None
    while prev != text:
        prev = text
        text = REPEAT_NGRAM_RE.sub(r"\1", text)
    return text.strip()


def cosine_accuracy(preds: list[str], refs: list[str], st_model) -> dict:
    pairs = [(p, r) for p, r in zip(preds, refs) if p.strip() and r.strip()]
    if not pairs:
        return {"n": 0, "mean": float("nan")}
    preds_e = st_model.encode([p for p, _ in pairs], normalize_embeddings=True, show_progress_bar=False)
    refs_e = st_model.encode([r for _, r in pairs], normalize_embeddings=True, show_progress_bar=False)
    cos = (preds_e * refs_e).sum(axis=1)
    rng = np.random.default_rng(42)
    boot = [cos[rng.integers(0, len(cos), len(cos))].mean() for _ in range(1000)]
    return {
        "n": len(pairs),
        "mean": float(cos.mean()),
        "std": float(cos.std()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def main() -> int:
    logger.info("loading all-mpnet-base-v2...")
    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    report: dict = {"datasets": {}}
    for pred_path in sorted(OUT_DIR.glob("preds_*.jsonl")):
        name = pred_path.stem.replace("preds_", "")
        rows = []
        with open(pred_path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        if not rows:
            continue

        raw_preds = [r["en_pred"] for r in rows]
        clean_preds = [clean(p) for p in raw_preds]
        refs = [r["en_ref"] for r in rows]

        # Save cleaned
        clean_path = OUT_DIR / f"preds_{name}_clean.jsonl"
        with open(clean_path, "w") as f:
            for r, c in zip(rows, clean_preds):
                f.write(json.dumps({"jp": r["jp"], "en_pred": c, "en_ref": r["en_ref"]}, ensure_ascii=False) + "\n")

        chrf_raw = sacrebleu.corpus_chrf(raw_preds, [refs], word_order=2).score
        chrf_clean = sacrebleu.corpus_chrf(clean_preds, [refs], word_order=2).score
        cos_raw = cosine_accuracy(raw_preds, refs, st)
        cos_clean = cosine_accuracy(clean_preds, refs, st)

        report["datasets"][name] = {
            "n": len(rows),
            "chrf_raw": chrf_raw,
            "chrf_clean": chrf_clean,
            "cosine_raw": cos_raw,
            "cosine_clean": cos_clean,
        }
        logger.info("[{}] n={} chrF raw={:.2f}->clean={:.2f}  cos raw={:.4f}->clean={:.4f}",
                    name, len(rows), chrf_raw, chrf_clean,
                    cos_raw.get("mean", 0), cos_clean.get("mean", 0))

    out = OUT_DIR / "eval_cleaned.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("wrote {}", out)

    print()
    print("=== CLEANED EVAL SUMMARY ===")
    print(f"{'dataset':22s} {'n':>5}  {'chrF raw':>9}  {'chrF clean':>11}  {'cos raw':>9}  {'cos clean':>10}")
    for name, rec in report["datasets"].items():
        print(f"{name:22s} {rec['n']:>5}  {rec['chrf_raw']:9.2f}  {rec['chrf_clean']:11.2f}  "
              f"{rec['cosine_raw']['mean']:9.4f}  {rec['cosine_clean']['mean']:10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

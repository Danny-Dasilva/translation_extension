"""Score v10-it (or any v*) output vs Gemma 3 4B base teacher reference (257-bubble eval).

Mirrors the methodology used to compute v7 (89/257) and v9c (85/257) baselines,
extended with chrF++, BLEU, empty-rate, and JP-passthrough rate so the Pareto
gate (>=3 of 5 metrics improved over v9c) can be evaluated mechanically.

Inputs
------
* Pred dir:        <pred-dir>/<slug>/stats.json (from translate_manga_unsloth.py)
* Gemma reference: <ref-jsonl> (e.g. .../644289-abc-gemma4-base-45/modeA.jsonl)

Outputs
-------
* <out>  (default: <pred-dir>/score_summary.json)
* Stdout summary + per-page breakdown
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def is_jp_passthrough(en: str, jp: str) -> bool:
    """Heuristic: prediction starts with JP characters or equals the JP source."""
    if not en:
        return False
    if en.strip() == jp.strip():
        return True
    head = en.strip()[:20]
    return any(
        0x3040 <= ord(c) <= 0x309F or  # hiragana
        0x30A0 <= ord(c) <= 0x30FF or  # katakana
        0x4E00 <= ord(c) <= 0x9FFF     # CJK unified ideographs
        for c in head
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True,
                    help="Dir with <slug>/stats.json (from translate_manga_unsloth.py)")
    # Accept both --ref-jsonl (preferred) and --gemma-ref (legacy alias).
    ap.add_argument("--ref-jsonl", "--gemma-ref", dest="ref_jsonl",
                    default="/home/danny/manga-output/644289-abc-gemma4-base-45/modeA.jsonl",
                    help="Gemma 3 4B base teacher reference JSONL")
    ap.add_argument("--out", default=None,
                    help="Output summary JSON path (default: <pred-dir>/score_summary.json)")
    ap.add_argument("--label", default="v10it")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    out_path = Path(args.out) if args.out else (pred_dir / "score_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gemma: dict[str, list[tuple[str, str]]] = {}
    with open(args.ref_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gemma[r["slug"]] = list(zip(r.get("jp_texts", []), r.get("en_texts", [])))

    pairs_pred: list[tuple[str, str, str, str]] = []  # (slug, jp, pred_en, gemma_en)
    for d in sorted(pred_dir.iterdir()):
        if not d.is_dir() or not d.name.isdigit():
            continue
        sj = d / "stats.json"
        if not sj.exists():
            continue
        s = json.loads(sj.read_text(encoding="utf-8"))
        ocr = s.get("ocr_samples") or []
        preds = s.get("translations") or []
        gref_pairs = gemma.get(d.name, [])
        gref_map = {jp.strip(): en for jp, en in gref_pairs}
        for jp, pred in zip(ocr, preds):
            gref = gref_map.get(jp.strip(), "")
            pairs_pred.append((d.name, jp, pred, gref))

    total = len(pairs_pred)
    aligned = [(s, jp, p, g) for s, jp, p, g in pairs_pred if g]
    n_aligned = len(aligned)

    # Exact-match (case-insensitive, whitespace-trimmed)
    em_hits = sum(1 for _, _, p, g in aligned if p.strip().lower() == g.strip().lower() and p.strip())
    em_pct = (em_hits / n_aligned * 100) if n_aligned else 0.0

    # Empty-rate (over ALL preds, aligned or not — empty is empty)
    empty_n = sum(1 for _, _, p, _ in pairs_pred if not p.strip())
    empty_pct = (empty_n / total * 100) if total else 0.0

    # JP-passthrough rate (over ALL preds)
    jp_pass_n = sum(1 for _, jp, p, _ in pairs_pred if is_jp_passthrough(p, jp))
    jp_pass_pct = (jp_pass_n / total * 100) if total else 0.0

    # chrF++ + BLEU (over aligned only)
    chrf: float | None = None
    bleu: float | None = None
    chrf_err: str | None = None
    bleu_err: str | None = None
    try:
        import sacrebleu
        preds_only = [p for _, _, p, g in aligned]
        refs_only = [g for _, _, p, g in aligned]
        chrf = float(sacrebleu.corpus_chrf(preds_only, [refs_only], word_order=2).score)
    except Exception as e:
        chrf_err = str(e)
    try:
        import sacrebleu
        preds_only = [p for _, _, p, g in aligned]
        refs_only = [g for _, _, p, g in aligned]
        bleu = float(sacrebleu.corpus_bleu(preds_only, [refs_only]).score)
    except Exception as e:
        bleu_err = str(e)

    # Per-page breakdown
    per_page: dict[str, dict[str, int]] = {}
    for s, jp, p, g in aligned:
        rec = per_page.setdefault(s, {"n": 0, "em": 0})
        rec["n"] += 1
        if p.strip().lower() == g.strip().lower() and p.strip():
            rec["em"] += 1

    summary: dict = {
        "label": args.label,
        "pred_dir": str(pred_dir),
        "ref_jsonl": str(args.ref_jsonl),
        "total_bubbles": total,
        "aligned_with_gemma": n_aligned,
        "gemma_exact_match": em_hits,
        "gemma_exact_match_pct": round(em_pct, 2),
        "chrf_pp": chrf,
        "bleu": bleu,
        "empty_n": empty_n,
        "empty_pct": round(empty_pct, 2),
        "jp_passthrough_n": jp_pass_n,
        "jp_passthrough_pct": round(jp_pass_pct, 2),
        "per_page": {k: per_page[k] for k in sorted(per_page)},
    }
    if chrf is None:
        summary["chrf_error"] = chrf_err
    if bleu is None:
        summary["bleu_error"] = bleu_err

    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== {args.label} vs Gemma 3 4B base teacher ===")
    print(f"total bubbles in pred dir : {total}")
    print(f"aligned w/ Gemma reference: {n_aligned}")
    print(f"Gemma exact-match         : {em_hits}/{n_aligned} ({em_pct:.2f}%)")
    if chrf is not None:
        print(f"chrF++ (word_order=2)     : {chrf:.2f}")
    if bleu is not None:
        print(f"BLEU                      : {bleu:.2f}")
    print(f"empty rate                : {empty_n}/{total} ({empty_pct:.2f}%)")
    print(f"JP-passthrough rate       : {jp_pass_n}/{total} ({jp_pass_pct:.2f}%)")
    print()
    print(f"baselines: v7 89/257 (34.6%), v9c 85/257 (33.07%), chrF++ v9c 70.40")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

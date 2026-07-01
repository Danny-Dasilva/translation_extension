"""Validate the alignment + formatting logic on a KNOWN-ANSWER sample (no GPU).

We have per-bubble gold (jp, human_en, bbox) for two chapters
(``ikenie4/gold_q3.jsonl``, ``ikenie5/gold_q3.jsonl``). Each gold ROW is a known
(JP source -> EN target) pair anchored at a bbox -- exactly the (jp_src, en_tgt)
the miner must reproduce. We use it WITHOUT running any OCR:

  * JP "OCR output"  = [{text: gold.jp,  bbox, conf: gold.ocr_conf}]   per page
  * EN "OCR output"  = [{text: gold.en,  bbox}]                        per page
  * KNOWN pairing    = gold row i's JP  <->  gold row i's EN

To make the cross-page matcher's job realistic (JP raw vs EN redraw are DIFFERENT
images with approximately-preserved layout), we PERTURB the EN boxes -- a global
affine (scale/translate, i.e. different scan resolution) plus per-bubble Gaussian
centroid jitter sigma = ``jitter`` * page-dimension -- and SHUFFLE the EN order,
then check how well ``align_pages`` recovers the gold row<->row pairing.

Reported:
  * alignment precision / recall vs the gold pairing across a jitter SWEEP
    (jitter=0 is the sanity floor; Ikenie's real EN redraw is near-pixel, IoU~0.95,
     so it sits at the low-jitter end),
  * count-mismatch robustness (drop + spurious EN boxes),
  * end-to-end: curated page-context rows written to a sample parquet, with the
    quality-score histogram, NSFW fraction, and a couple of example rows.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BACKEND))  # for `app.` and `scripts.` imports

from align import AlignConfig, align_pages, coverage  # type: ignore  # noqa: E402
from curate import CurationConfig, CurationStats  # type: ignore  # noqa: E402
from format_rows import assemble_parquet  # type: ignore  # noqa: E402
from pipeline import PipelineConfig, align_and_curate  # type: ignore  # noqa: E402

CHAPTERS = {
    "ikenie4": _BACKEND / "scripts/eval/data/ikenie4/gold_q3.jsonl",
    "ikenie5": _BACKEND / "scripts/eval/data/ikenie5/gold_q3.jsonl",
}
OUT_DIR = _HERE / "validation_out"
EN_FRAME = (1240.0, 1754.0)  # synthetic EN page pixel frame for perturbed boxes


# --------------------------------------------------------------------------- #
# build per-page gold bubble lists
# --------------------------------------------------------------------------- #
def load_pages(gold_path: Path):
    """Return {page: [gold_row,...]} deduped by src, only rows with a bbox."""
    import re

    by_page: dict[int, list[dict]] = {}
    seen_src: set[str] = set()
    for line in gold_path.open():
        r = json.loads(line)
        if not r.get("bbox"):
            continue
        src = r.get("src", "")
        if src in seen_src:
            continue
        seen_src.add(src)
        m = re.match(r"[^:]+:p(\d+):idx(\d+)", src)
        if not m:
            continue
        page = int(m.group(1))
        by_page.setdefault(page, []).append(r)
    return by_page


def page_bubbles(rows: list[dict]):
    """Build (jp_bubbles, en_bubbles) with a shared ``pair_id`` per gold row."""
    jp, en = [], []
    for pid, r in enumerate(rows):
        bbox = r["bbox"]
        jp.append({
            "idx": pid, "pair_id": pid, "text": (r.get("jp") or "").strip(),
            "bbox": dict(bbox), "conf": float(r.get("ocr_conf") or 0.0),
        })
        en.append({
            "idx": pid, "pair_id": pid, "text": (r.get("en") or "").strip(),
            "bbox": dict(bbox), "conf": None,
        })
    return jp, en


# --------------------------------------------------------------------------- #
# perturbation: simulate the EN redraw being a DIFFERENT image
# --------------------------------------------------------------------------- #
def _dims(bubbles):
    W = max(b["bbox"]["maxX"] for b in bubbles)
    H = max(b["bbox"]["maxY"] for b in bubbles)
    return float(max(W, 1)), float(max(H, 1))


def perturb_en(en_bubbles, jitter: float, rng: random.Random, frame=EN_FRAME):
    W, H = _dims(en_bubbles)
    fw, fh = frame
    sx, sy = rng.uniform(0.97, 1.03), rng.uniform(0.97, 1.03)
    tx, ty = rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01)
    out = []
    for b in en_bubbles:
        bb = b["bbox"]
        cx = ((bb["minX"] + bb["maxX"]) / 2) / W
        cy = ((bb["minY"] + bb["maxY"]) / 2) / H
        w = (bb["maxX"] - bb["minX"]) / W
        h = (bb["maxY"] - bb["minY"]) / H
        ncx = sx * cx + tx + (rng.gauss(0, jitter) if jitter else 0.0)
        ncy = sy * cy + ty + (rng.gauss(0, jitter) if jitter else 0.0)
        nw, nh = w * sx, h * sy
        minX = max(0.0, (ncx - nw / 2) * fw)
        maxX = max(minX + 1, (ncx + nw / 2) * fw)
        minY = max(0.0, (ncy - nh / 2) * fh)
        maxY = max(minY + 1, (ncy + nh / 2) * fh)
        nb = dict(b)
        nb["bbox"] = {"minX": minX, "minY": minY, "maxX": maxX, "maxY": maxY}
        out.append(nb)
    return out


# --------------------------------------------------------------------------- #
# alignment precision / recall vs the gold pairing
# --------------------------------------------------------------------------- #
def score_alignment(jp, en, cfg: AlignConfig):
    """Return (correct, emitted, n_gold) for one page. correct = matched pairs
    whose JP and EN come from the SAME gold row (pair_id)."""
    matches, _, _ = align_pages(jp, en, cfg)
    correct = sum(1 for m in matches if jp[m.jp_idx]["pair_id"] == en[m.en_idx]["pair_id"])
    return correct, len(matches), len(jp)


def sweep(pages_by_chapter, jitters, seeds, cfg: AlignConfig, drop_frac=0.0, spurious_frac=0.0):
    results = {}
    for jit in jitters:
        agg_correct = agg_emitted = agg_gold = 0
        for seed in seeds:
            rng = random.Random(seed)
            for _ch, by_page in pages_by_chapter.items():
                for _pg, rows in by_page.items():
                    if len(rows) < 2:
                        continue
                    jp, en = page_bubbles(rows)
                    en_p = perturb_en(en, jit, rng)
                    # count-mismatch: drop some EN, add spurious EN boxes
                    if drop_frac:
                        keep_n = max(1, int(round(len(en_p) * (1 - drop_frac))))
                        en_p = rng.sample(en_p, keep_n)
                    if spurious_frac:
                        n_spur = int(round(len(en) * spurious_frac))
                        for s in range(n_spur):
                            en_p.append({
                                "idx": -1 - s, "pair_id": -1 - s, "text": "SPURIOUS",
                                "bbox": {"minX": rng.uniform(0, EN_FRAME[0] * 0.9),
                                         "minY": rng.uniform(0, EN_FRAME[1] * 0.9),
                                         "maxX": 0, "maxY": 0}, "conf": None,
                            })
                            b = en_p[-1]["bbox"]
                            b["maxX"] = b["minX"] + 40
                            b["maxY"] = b["minY"] + 40
                    rng.shuffle(en_p)
                    c, e, g = score_alignment(jp, en_p, cfg)
                    agg_correct += c
                    agg_emitted += e
                    agg_gold += g
        prec = agg_correct / agg_emitted if agg_emitted else 0.0
        rec = agg_correct / agg_gold if agg_gold else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        results[jit] = {
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "correct": agg_correct, "emitted": agg_emitted, "gold_pairs": agg_gold,
        }
    return results


# --------------------------------------------------------------------------- #
# end-to-end: curate + format + sample parquet (jitter=0 -> Ikenie near-pixel)
# --------------------------------------------------------------------------- #
def build_sample(pages_by_chapter, pcfg: PipelineConfig, jitter: float, seed: int):
    stats = CurationStats()
    seen: set = set()
    rows: list[dict] = []
    rng = random.Random(seed)
    for ch, by_page in pages_by_chapter.items():
        for pg, grows in by_page.items():
            if len(grows) < 1:
                continue
            jp, en = page_bubbles(grows)
            en_p = perturb_en(en, jitter, rng) if jitter else en
            gid_tag = f"{ch}"
            r, _ = align_and_curate(jp, en_p, gid_tag, pg, pcfg, stats, seen)
            rows.extend(r)
    return rows, stats


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = {ch: load_pages(p) for ch, p in CHAPTERS.items()}
    n_pages = {ch: len(v) for ch, v in pages.items()}
    n_rows = {ch: sum(len(r) for r in v.values()) for ch, v in pages.items()}

    acfg = AlignConfig()
    report: dict = {
        "chapters": {ch: {"pages": n_pages[ch], "gold_pairs": n_rows[ch]} for ch in pages},
        "align_config": acfg.__dict__,
    }

    # 1. alignment precision/recall sweep over redraw-jitter
    jitters = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08]
    seeds = [0, 1, 2]
    report["alignment_sweep"] = sweep(pages, jitters, seeds, acfg)

    # 2. count-mismatch robustness (drop 20% EN + 10% spurious, realistic jitter)
    report["count_mismatch_2pct"] = sweep(
        pages, [0.02], seeds, acfg, drop_frac=0.20, spurious_frac=0.10
    )[0.02]

    # 3. end-to-end curate+format -> sample parquet (jitter=0: Ikenie near-pixel)
    pcfg = PipelineConfig.default()
    rows0, stats0 = build_sample(pages, pcfg, jitter=0.0, seed=0)
    sample_parquet = OUT_DIR / "sample_ikenie_bitext.parquet"
    pq_stats = assemble_parquet(rows0, sample_parquet)
    # operating-point check at realistic jitter 0.02
    rows2, stats2 = build_sample(pages, pcfg, jitter=0.02, seed=0)

    report["end_to_end"] = {
        "jitter_0": {"curation": stats0.as_dict(), "parquet": pq_stats},
        "jitter_0.02": {"curation": stats2.as_dict(), "kept_rows": len(rows2)},
        "curation_config": pcfg.curate.__dict__,
    }

    # sample rows
    sample_rows = rows0[:8]
    (OUT_DIR / "sample_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in sample_rows)
    )
    report["example_rows"] = sample_rows[:3]

    (OUT_DIR / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # console summary
    print("=" * 72)
    print("ALIGNMENT PRECISION/RECALL vs gold pairing (redraw-jitter sweep)")
    print(f"  gold pages: {n_pages}   gold pairs: {n_rows}")
    print(f"  {'jitter':>8} {'precision':>10} {'recall':>10} {'f1':>8}  (correct/emitted/gold)")
    for jit in jitters:
        s = report["alignment_sweep"][jit]
        print(f"  {jit:>8.2f} {s['precision']:>10.3f} {s['recall']:>10.3f} {s['f1']:>8.3f}"
              f"  ({s['correct']}/{s['emitted']}/{s['gold_pairs']})")
    cm = report["count_mismatch_2pct"]
    print(f"\n  count-mismatch (drop20%+spurious10%, jit0.02): "
          f"P={cm['precision']:.3f} R={cm['recall']:.3f} ({cm['correct']}/{cm['emitted']}/{cm['gold_pairs']})")
    print("\n" + "=" * 72)
    print("END-TO-END curate+format (jitter=0, Ikenie near-pixel redraw)")
    print(f"  kept rows: {pq_stats['rows']}  pagectx: {pq_stats['pagectx_rows']}  "
          f"nsfw_frac: {pq_stats['nsfw_frac']}")
    print(f"  curation: {json.dumps(stats0.as_dict(), ensure_ascii=False)}")
    print(f"  wrote {sample_parquet}")
    print("\n  example rows:")
    for r in sample_rows[:2]:
        print(f"    src={r['src']}")
        print(f"    en={r['en']!r}")
        print(f"    prompt[:140]={r['prompt'][:140]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

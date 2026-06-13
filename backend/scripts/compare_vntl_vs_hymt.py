"""Quantitative comparison of VNTL vs HY-MT-native translations on the
same 45-page gallery. Runs lightweight probes (no external LLM judge):

  - Untranslated JP leak (any CJK chars in EN output)
  - Curly-quote / ellipsis usage (rendering-relevant)
  - Honorific leak (-san/-chan/-kun/-sama/-senpai)
  - Repetition loop detection (3-char substring ≥ 6×)
  - Refusal / sanitization markers
  - Length ratio sanity
  - Total wall-time from stats.json
  - Exact-match rate when both models produce the same JP input → same EN
  - Per-page bubble count consistency

Outputs: markdown report at the destination gallery root.

Usage:
    uv run python scripts/compare_vntl_vs_hymt.py \
        --vntl-gallery /home/danny/manga-output/644289 \
        --hymt-gallery /home/danny/manga-output/644289-hymt-native
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_CJK_RE = re.compile(r"[぀-ヿ一-鿿]")  # hira/kata/kanji
_CURLY_RE = re.compile(r"[‘’“”…]")  # ' ' " " …
_HONORIFIC_RE = re.compile(
    r"\b\w+[-_](san|kun|chan|sama|senpai|sensei|dono|tan)\b", re.IGNORECASE
)
# 3-char substring repeating ≥6× (e.g. "b-b-b-b-b-b", "hahahahahaha")
_LOOP_RE = re.compile(r"(.{1,3})\1{5,}")
# Refusal / sanitization markers
_REFUSAL_RE = re.compile(
    r"\b(i (can'?t|cannot|won'?t|am unable|am not able)|i'?m (sorry|afraid)|"
    r"i apologize|as an ai|inappropriate|unable to translate|not comfortable)\b",
    re.IGNORECASE,
)


def _parse_pairs(txt: str) -> list[tuple[str, str]]:
    """Extract (JP, EN) pairs from a per-slug translations.txt body."""
    pairs = []
    for m in re.finditer(
        r"\[(\d+)\]\n\s+JP: (.*?)\n\s+EN: (.*?)(?=\n\[|\n\n|\Z)",
        txt, flags=re.DOTALL,
    ):
        pairs.append((m.group(2).strip(), m.group(3).strip()))
    return pairs


def probe_bubble(jp: str, en: str) -> dict:
    return {
        "cjk_leak": bool(_CJK_RE.search(en)),
        "curly_count": len(_CURLY_RE.findall(en)),
        "honorific_leak": bool(_HONORIFIC_RE.search(en)),
        "loop": bool(_LOOP_RE.search(en)),
        "refusal": bool(_REFUSAL_RE.search(en)),
        "empty": not en.strip(),
        "en_len": len(en),
        "jp_len": len(jp),
    }


def summarise(pairs: list[tuple[str, str]]) -> dict:
    probes = [probe_bubble(jp, en) for jp, en in pairs]
    n = len(probes) or 1
    total_en = sum(p["en_len"] for p in probes)
    total_jp = sum(p["jp_len"] for p in probes)
    return {
        "bubbles": len(probes),
        "cjk_leak_pct": 100 * sum(1 for p in probes if p["cjk_leak"]) / n,
        "curly_total": sum(p["curly_count"] for p in probes),
        "honorific_leak_pct": 100 * sum(1 for p in probes if p["honorific_leak"]) / n,
        "loop_pct": 100 * sum(1 for p in probes if p["loop"]) / n,
        "refusal_pct": 100 * sum(1 for p in probes if p["refusal"]) / n,
        "empty_pct": 100 * sum(1 for p in probes if p["empty"]) / n,
        "en_per_jp": (total_en / total_jp) if total_jp else 0.0,
    }


def per_pair_diffs(vntl: list[tuple[str, str]],
                   hymt: list[tuple[str, str]], slug: str) -> list[dict]:
    """Pair-wise JP-matched diffs when both sides saw the same JP."""
    vntl_by_jp = {jp: en for jp, en in vntl}
    out = []
    for jp, hen in hymt:
        ven = vntl_by_jp.get(jp, None)
        if ven is None:
            continue
        if ven == hen:
            status = "identical"
        elif not hen.strip() and ven.strip():
            status = "hymt_empty"
        elif _REFUSAL_RE.search(hen) and not _REFUSAL_RE.search(ven):
            status = "hymt_refusal"
        elif _CJK_RE.search(hen) and not _CJK_RE.search(ven):
            status = "hymt_cjk_leak"
        else:
            status = "both_nonempty_differ"
        out.append({"slug": slug, "jp": jp, "vntl": ven,
                    "hymt": hen, "status": status})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vntl-gallery", type=Path, required=True)
    ap.add_argument("--hymt-gallery", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="Markdown report path (default: hymt-gallery/COMPARISON.md)")
    args = ap.parse_args()

    out_md = args.out or (args.hymt_gallery / "COMPARISON.md")

    slug_dirs = sorted([p.name for p in args.hymt_gallery.iterdir()
                        if p.is_dir() and (p / "translations.txt").exists()])
    vntl_slugs = sorted([p.name for p in args.vntl_gallery.iterdir()
                         if p.is_dir() and (p / "translations.txt").exists()])
    common = [s for s in slug_dirs if s in vntl_slugs]
    print(f"vntl: {len(vntl_slugs)} slugs, hymt: {len(slug_dirs)} slugs, "
          f"common: {len(common)}")

    all_vntl = []
    all_hymt = []
    diffs = []

    for slug in common:
        v_txt = (args.vntl_gallery / slug / "translations.txt").read_text(encoding="utf-8")
        h_txt = (args.hymt_gallery / slug / "translations.txt").read_text(encoding="utf-8")
        v_pairs = _parse_pairs(v_txt)
        h_pairs = _parse_pairs(h_txt)
        all_vntl.extend(v_pairs)
        all_hymt.extend(h_pairs)
        diffs.extend(per_pair_diffs(v_pairs, h_pairs, slug))

    # Timings
    def avg_time(gallery: Path) -> tuple[int, float]:
        total = 0.0
        count = 0
        for slug in common:
            try:
                s = json.loads((gallery / slug / "stats.json").read_text())
                ms = s.get("translate_ms") or s.get("ms") or 0
                total += float(ms)
                count += 1
            except Exception:
                pass
        return count, total

    v_count, v_total = avg_time(args.vntl_gallery)
    h_count, h_total = avg_time(args.hymt_gallery)

    v_summary = summarise(all_vntl)
    h_summary = summarise(all_hymt)

    status_counts = {"identical": 0, "hymt_empty": 0, "hymt_refusal": 0,
                     "hymt_cjk_leak": 0, "both_nonempty_differ": 0}
    for d in diffs:
        status_counts[d["status"]] = status_counts.get(d["status"], 0) + 1

    lines = [
        "# VNTL-llama3-8b-v2 (batched [N]) vs HY-MT1.5-1.8B (native raw prompt)\n",
        f"- VNTL gallery: `{args.vntl_gallery}`",
        f"- HY-MT gallery: `{args.hymt_gallery}`",
        f"- Pages compared: {len(common)}",
        f"- Total bubbles — VNTL: {v_summary['bubbles']}, HY-MT: {h_summary['bubbles']}\n",
        "## Probe scorecard (lower % is better for leak/loop/refusal)\n",
        "| Metric | VNTL | HY-MT |",
        "|---|---|---|",
        f"| CJK leak in EN | {v_summary['cjk_leak_pct']:.1f}% | {h_summary['cjk_leak_pct']:.1f}% |",
        f"| Curly quote/… chars (total) | {v_summary['curly_total']} | {h_summary['curly_total']} |",
        f"| Honorific leak (-san etc.) | {v_summary['honorific_leak_pct']:.1f}% | {h_summary['honorific_leak_pct']:.1f}% |",
        f"| Repetition loop | {v_summary['loop_pct']:.1f}% | {h_summary['loop_pct']:.1f}% |",
        f"| Refusal / sanitization | {v_summary['refusal_pct']:.1f}% | {h_summary['refusal_pct']:.1f}% |",
        f"| Empty output | {v_summary['empty_pct']:.1f}% | {h_summary['empty_pct']:.1f}% |",
        f"| EN chars / JP chars | {v_summary['en_per_jp']:.2f} | {h_summary['en_per_jp']:.2f} |",
        "",
        "## Wall-time\n",
        "| Metric | VNTL | HY-MT |",
        "|---|---|---|",
        f"| Total translate time (s) | {v_total/1000:.1f} | {h_total/1000:.1f} |",
        f"| Avg per page (ms) | {v_total/max(1,v_count):.0f} | {h_total/max(1,h_count):.0f} |",
        f"| Per bubble (ms) | {v_total/max(1,v_summary['bubbles']):.0f} | {h_total/max(1,h_summary['bubbles']):.0f} |",
        "",
        "## Pairwise diff breakdown (same JP → both produced EN)\n",
        f"- Identical output: {status_counts['identical']}",
        f"- HY-MT empty (VNTL non-empty): {status_counts['hymt_empty']}",
        f"- HY-MT contains refusal phrase (VNTL does not): {status_counts['hymt_refusal']}",
        f"- HY-MT leaks JP chars (VNTL does not): {status_counts['hymt_cjk_leak']}",
        f"- Both non-empty, texts differ: {status_counts['both_nonempty_differ']}",
        "",
        "## Sample HY-MT refusals / sanitizations\n",
    ]
    refusal_samples = [d for d in diffs if d["status"] == "hymt_refusal"][:10]
    for d in refusal_samples:
        lines.append(f"**{d['slug']}**")
        lines.append(f"  JP: `{d['jp'][:80]}`")
        lines.append(f"  VNTL: `{d['vntl'][:100]}`")
        lines.append(f"  HY-MT: `{d['hymt'][:100]}`")
        lines.append("")
    if not refusal_samples:
        lines.append("_(none detected by regex probe)_\n")

    lines.append("## 20 random pairwise samples where both models produced distinct output\n")
    import random
    random.seed(42)
    diff_samples = [d for d in diffs if d["status"] == "both_nonempty_differ"]
    random.shuffle(diff_samples)
    for d in diff_samples[:20]:
        lines.append(f"**{d['slug']}** JP: `{d['jp'][:80]}`")
        lines.append(f"  VNTL:  `{d['vntl'][:100]}`")
        lines.append(f"  HY-MT: `{d['hymt'][:100]}`")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()

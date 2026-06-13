"""Score + visualize A/B/C eval output from ``translate_ab.py``.

Supports one or more runs for side-by-side model comparison. Each run is
passed as ``label:path`` and must contain ``mode{A,B,C}.jsonl``.

Usage::

    uv run python scripts/eval_vision/score_ab.py \\
        --runs base:~/out-base uncensored:~/out-uncensored \\
        --gallery ~/manga-output/644289 \\
        --html ~/combined-report.html

The HTML report has one row per (page, bubble). Columns:
    # · JP OCR · {label}-A · {label}-B · {label}-C  (repeated per run)

Stats table lists every (run, mode) combination.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
import statistics
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

_CJK_RE = re.compile(r"[぀-ヿ一-鿿]")
_ASCII_RE = re.compile(r"[A-Za-z0-9]")


def _is_garbage_jp(s: str) -> bool:
    s = s.strip()
    if not s or len(s) < 4:
        return True
    cjk = len(_CJK_RE.findall(s))
    ascii_ = len(_ASCII_RE.findall(s))
    if cjk and ascii_ and ascii_ >= cjk:
        return True
    if re.search(r"(.)\1{3,}", s):
        return True
    return False


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_run(run_dir: Path) -> Dict[str, List[dict]]:
    return {m: _load_jsonl(run_dir / f"mode{m}.jsonl") for m in ("A", "B", "C")}


def _summarize(label: str, mode: str, records: List[dict]) -> Dict[str, object]:
    if not records:
        return {"label": label, "mode": mode, "n": 0}
    ms = [r["ms"] for r in records]
    integ = [r["tag_integrity"] for r in records]
    wpb: List[float] = []
    garb_pages = 0
    garb_rec = 0
    for r in records:
        n = r["num_tags"]
        if n and r["en_texts"]:
            wpb.append(sum(len(t.split()) for t in r["en_texts"]) / n)
        has_garb = any(_is_garbage_jp(j) for j in r["jp_texts"])
        if has_garb:
            garb_pages += 1
            for jp, en in zip(r["jp_texts"], r["en_texts"]):
                if _is_garbage_jp(jp) and en and en != "..." and _CJK_RE.search(en) is None:
                    garb_rec += 1
                    break
    return {
        "label": label, "mode": mode, "n": len(records),
        "tag_integrity": statistics.mean(integ),
        "mean_ms": statistics.mean(ms),
        "median_ms": statistics.median(ms),
        "p95_ms": sorted(ms)[max(0, int(len(ms) * 0.95) - 1)],
        "words_per_bubble": statistics.mean(wpb) if wpb else 0.0,
        "garb_pages": garb_pages, "garb_rec": garb_rec,
    }


def _print_table(summaries: List[Dict[str, object]]) -> None:
    if not summaries:
        print("(no data)")
        return
    header = ["run", "mode", "n", "tag_integ", "mean_ms", "p95_ms", "w/bub", "garb"]
    rows: List[List[str]] = []
    for s in summaries:
        if s["n"] == 0:
            rows.append([s["label"], s["mode"], "0", "-", "-", "-", "-", "-"])
            continue
        rows.append([
            s["label"], s["mode"], str(s["n"]),
            f"{s['tag_integrity']:.2%}",
            f"{s['mean_ms']:.0f}", f"{s['p95_ms']:.0f}",
            f"{s['words_per_bubble']:.1f}",
            f"{s['garb_rec']}/{s['garb_pages']}",
        ])
    widths = [max(len(r[i]) for r in ([header] + rows)) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))


def _thumb(path: Path, max_w: int = 280) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w > max_w:
        img = img.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _gather_by_slug(
    all_runs: Dict[str, Dict[str, List[dict]]]
) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """Returns {slug: {label: {mode: record}}}."""
    out: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for label, runs in all_runs.items():
        for mode, recs in runs.items():
            for r in recs:
                out.setdefault(r["slug"], {}).setdefault(label, {})[mode] = r
    return out


def _render_html(
    labels: List[str],
    all_runs: Dict[str, Dict[str, List[dict]]],
    gallery: Path,
    summaries: List[Dict[str, object]],
    out_html: Path,
) -> None:
    by_slug = _gather_by_slug(all_runs)
    parts: List[str] = []
    parts.append(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Gemma 4 31B — A/B/C × models</title>"
    )
    parts.append("""<style>
body{font-family:system-ui,sans-serif;margin:12px;background:#0b0b0b;color:#e8e8e8;}
h1{margin:4px 0 8px;}
h2{margin:24px 0 8px;font-size:15px;color:#8ab4f8;}
table{border-collapse:collapse;margin-bottom:14px;font-size:12px;width:100%;}
th,td{border:1px solid #333;padding:4px 6px;vertical-align:top;}
th{background:#222;text-align:left;position:sticky;top:0;}
.page{margin:16px 0;padding:10px;background:#111;border:1px solid #222;border-radius:6px;}
.jp{color:#a5d6a7;font-family:'Noto Sans JP',monospace;max-width:260px;word-break:break-all;}
.A{color:#fbc02d;}
.B{color:#81d4fa;}
.C{color:#f48fb1;}
.garbage{background:#3a1010;}
.stat{background:#1a1a1a;}
img.thumb{max-width:280px;border:1px solid #333;}
.meta{color:#888;font-size:10px;}
.hdr-base{background:#143d3d!important;}
.hdr-unc{background:#3d1434!important;}
</style></head><body>""")
    parts.append(f"<h1>Gemma 4 31B — A/B/C × {'/'.join(labels)}</h1>")

    # -- Summary table ------------------------------------------------
    parts.append("<h2>Summary</h2><table class='stat'><thead><tr>"
                 "<th>run</th><th>mode</th><th>n</th><th>tag integ</th>"
                 "<th>mean ms</th><th>p95 ms</th><th>w/bub</th><th>garb rec/pages</th>"
                 "</tr></thead><tbody>")
    for s in summaries:
        if s["n"] == 0:
            parts.append(
                f"<tr><td>{html.escape(s['label'])}</td><td>{s['mode']}</td>"
                f"<td>0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>"
            )
            continue
        parts.append(
            f"<tr><td>{html.escape(s['label'])}</td><td>{s['mode']}</td>"
            f"<td>{s['n']}</td><td>{s['tag_integrity']:.2%}</td>"
            f"<td>{s['mean_ms']:.0f}</td><td>{s['p95_ms']:.0f}</td>"
            f"<td>{s['words_per_bubble']:.1f}</td>"
            f"<td>{s['garb_rec']}/{s['garb_pages']}</td></tr>"
        )
    parts.append("</tbody></table>")

    # -- Per-page tables ---------------------------------------------
    for slug in sorted(by_slug):
        by_label = by_slug[slug]
        # Find a reference record for jp_texts / num_tags
        ref = next(
            (rec for lab in labels for rec in by_label.get(lab, {}).values() if rec), None
        )
        if ref is None:
            continue
        n = ref["num_tags"]
        jp_texts = ref["jp_texts"]

        parts.append(f"<div class='page'><h2>{html.escape(slug)} — {n} bubble(s)</h2>")
        meta_bits: List[str] = []
        for lab in labels:
            for m in ("A", "B", "C"):
                r = by_label.get(lab, {}).get(m)
                if r:
                    meta_bits.append(f"{lab}-{m}:{int(r['ms'])}ms")
        parts.append(f"<div class='meta'>{' · '.join(meta_bits)}</div>")

        orig = gallery / slug / "01_original.png"
        if orig.exists():
            parts.append(f"<img class='thumb' src='{_thumb(orig)}'/>")

        # Build one header row per label × mode
        header_cells = "<th>#</th><th>JP (OCR)</th>"
        for lab in labels:
            cls = "hdr-base" if lab == labels[0] else ("hdr-unc" if lab == labels[-1] else "")
            for m in ("A", "B", "C"):
                header_cells += f"<th class='{cls}'>{html.escape(lab)}-{m}</th>"
        parts.append(f"<table><thead><tr>{header_cells}</tr></thead><tbody>")

        for i in range(n):
            jp = jp_texts[i] if i < len(jp_texts) else ""
            garb = " garbage" if _is_garbage_jp(jp) else ""
            row = f"<td>{i+1}</td><td class='jp{garb}'>{html.escape(jp)}</td>"
            for lab in labels:
                for m in ("A", "B", "C"):
                    r = by_label.get(lab, {}).get(m)
                    en = r["en_texts"][i] if r and i < len(r.get("en_texts", [])) else ""
                    row += f"<td class='{m}'>{html.escape(en)}</td>"
            parts.append(f"<tr>{row}</tr>")
        parts.append("</tbody></table></div>")

    parts.append("</body></html>")
    out_html.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs", nargs="+", required=True,
        help="one or more label:path pairs, e.g. base:~/out-base uncensored:~/out-unc"
    )
    ap.add_argument("--gallery", default="~/manga-output/644289")
    ap.add_argument("--html", default=None)
    args = ap.parse_args()

    labels: List[str] = []
    all_runs: Dict[str, Dict[str, List[dict]]] = {}
    for spec in args.runs:
        if ":" not in spec:
            raise SystemExit(f"--runs entry must be label:path, got {spec!r}")
        label, path = spec.split(":", 1)
        labels.append(label)
        all_runs[label] = _load_run(Path(path).expanduser().resolve())

    summaries: List[Dict[str, object]] = []
    for lab in labels:
        for m in ("A", "B", "C"):
            summaries.append(_summarize(lab, m, all_runs[lab][m]))
    _print_table(summaries)

    if args.html:
        gallery = Path(args.gallery).expanduser().resolve()
        out = Path(args.html).expanduser().resolve()
        _render_html(labels, all_runs, gallery, summaries, out)
        print(f"\nHTML → {out}")


if __name__ == "__main__":
    main()

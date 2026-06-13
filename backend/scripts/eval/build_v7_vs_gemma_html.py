"""Build a single HTML page comparing v7 vs Gemma 3 4B base on 644289.

For each bubble: shows page thumbnail, JP, v7 EN, Gemma base EN.
Output: /home/danny/manga-output/v7-vs-gemma.html (single file, embedded thumbs).
"""
from __future__ import annotations

import base64
import html
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

V7_DIR = Path("/home/danny/manga-output/644289-qwen3mt-v7")
GEMMA_BASE = Path("/home/danny/manga-output/644289-abc-gemma4-base-45/modeA.jsonl")
ORIG_DIR = Path("/home/danny/manga-output/644289/originals")
OUT_HTML = Path("/home/danny/manga-output/v7-vs-gemma.html")

THUMB_W = 200


def thumb_b64(img_path: Path, max_w: int = THUMB_W) -> str:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w > max_w:
        new_h = int(h * max_w / w)
        img = img.resize((max_w, new_h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_gemma_base() -> dict[str, list[str]]:
    """slug -> list of EN translations (mode A)."""
    out: dict[str, list[str]] = {}
    with open(GEMMA_BASE) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[rec["slug"]] = rec["en_texts"]
    return out


def load_v7(slug: str) -> tuple[list[str], list[str]]:
    """returns (jp, en) lists for a slug."""
    p = V7_DIR / slug / "stats.json"
    if not p.exists():
        return [], []
    with open(p) as f:
        s = json.load(f)
    return s.get("ocr_samples") or [], s.get("translations") or []


def main() -> int:
    gemma = load_gemma_base()

    # All slugs that exist in both
    slugs = sorted(p.name for p in V7_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    rows_html: list[str] = []
    css = """
    body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; padding: 12px; background: #0c0d10; color: #e7e9ed; }
    h1 { font-size: 20px; margin: 0 0 12px; }
    .legend { color: #8b95a5; font-size: 13px; margin-bottom: 14px; }
    .page { background: #14161b; border: 1px solid #21242c; border-radius: 8px; padding: 10px; margin-bottom: 14px; }
    .page-header { display:flex; gap:14px; align-items:flex-start; margin-bottom: 8px; }
    .page-header img { border-radius: 6px; max-width: 200px; }
    .page-title { font-weight: 600; font-size: 16px; }
    .page-sub { color: #8b95a5; font-size: 12px; margin-top: 2px; }
    table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
    th, td { border-bottom: 1px solid #1d2028; padding: 6px 8px; vertical-align: top; }
    th { text-align: left; color: #8b95a5; font-weight: 500; font-size: 12px; }
    .jp { font-family: "Noto Sans CJK JP", sans-serif; color: #cfd5e0; min-width: 200px; }
    .v7  { color: #5fe39d; }
    .gem { color: #a4b9ff; }
    .same { background: #14211a !important; }
    .diff { background: #1f1612 !important; }
    .idx { color: #51596a; font-variant-numeric: tabular-nums; width: 28px; }
    """
    n_total = 0
    n_match = 0
    for slug in slugs:
        jps, v7s = load_v7(slug)
        gems = gemma.get(slug, [])
        if not jps:
            continue
        # Page thumbnail (use original)
        thumb_path = ORIG_DIR / f"{slug}.webp"
        if not thumb_path.exists():
            thumb_path = ORIG_DIR / f"{slug}.jpg"
        thumb = thumb_b64(thumb_path) if thumb_path.exists() else ""

        body_rows = []
        for i, jp in enumerate(jps):
            v7 = v7s[i] if i < len(v7s) else ""
            ge = gems[i] if i < len(gems) else ""
            n_total += 1
            same = (v7.strip().lower() == ge.strip().lower()) and v7.strip()
            if same:
                n_match += 1
            css_cls = "same" if same else ""
            body_rows.append(
                f'<tr class="{css_cls}"><td class="idx">{i+1}</td>'
                f'<td class="jp">{html.escape(jp)}</td>'
                f'<td class="v7">{html.escape(v7)}</td>'
                f'<td class="gem">{html.escape(ge)}</td></tr>'
            )

        rows_html.append(f"""
        <div class="page">
          <div class="page-header">
            {f'<img src="{thumb}" />' if thumb else ''}
            <div>
              <div class="page-title">Page {slug}</div>
              <div class="page-sub">{len(jps)} bubble(s)</div>
            </div>
          </div>
          <table>
            <thead><tr><th></th><th>JP (OCR)</th><th>v7 (Qwen3-4B-MT)</th><th>Gemma 3 4B base</th></tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
          </table>
        </div>
        """)

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>v7 vs Gemma 3 4B base — 644289</title>
<style>{css}</style></head>
<body>
<h1>v7 (Qwen3-4B-mt) vs Gemma 3 4B base — 644289</h1>
<div class="legend">{n_total} bubbles across {len(slugs)} pages · exact-match (case-insensitive): {n_match}/{n_total} ({100*n_match/max(1,n_total):.1f}%) · rows highlighted green when v7 == Gemma exactly</div>
{''.join(rows_html)}
</body></html>"""

    OUT_HTML.write_text(html_doc, encoding="utf-8")
    size_mb = OUT_HTML.stat().st_size / 1e6
    print(f"wrote {OUT_HTML} ({size_mb:.1f} MB, {n_total} bubbles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

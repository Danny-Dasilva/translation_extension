"""Build a comprehensive comparison HTML with rendered finals from all variants:

  - Original page
  - v7 (old CTD detector, no SFX)
  - v7 + v26 detector (no SFX)
  - v7 + v26 detector (with SFX)
  - Gemma 3 4B base translations (text only — Gemma was run via VLM, no rendered final)

Each page row: thumbnails for each variant + a per-bubble text table (JP / v7 / Gemma).
"""
from __future__ import annotations

import base64
import html
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

OUT_HTML = Path("/home/danny/manga-output/v7-vs-gemma-full.html")

V7_OLD_FINALS = Path("/home/danny/manga-output/644289-qwen3mt-v7-finals")
V26_NOSFX_FINALS = Path("/home/danny/manga-output/644289-v26-no-sfx-finals")
V26_WITHSFX_FINALS = Path("/home/danny/manga-output/644289-v26-with-sfx-finals")
ORIG_DIR = Path("/home/danny/manga-output/644289/originals")
GEMMA_BASE = Path("/home/danny/manga-output/644289-abc-gemma4-base-45/modeA.jsonl")
V7_DIR = Path("/home/danny/manga-output/644289-qwen3mt-v7")
V26_NOSFX_DIR = Path("/home/danny/manga-output/644289-v26-no-sfx")
V26_WITHSFX_DIR = Path("/home/danny/manga-output/644289-v26-with-sfx")

THUMB_W = 320


def thumb_b64(img_path: Path, max_w: int = THUMB_W) -> str:
    if not img_path.exists():
        return ""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w > max_w:
        new_h = int(h * max_w / w)
        img = img.resize((max_w, new_h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=68)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_gemma_base() -> dict[str, list[tuple[str, str]]]:
    """slug -> [(jp, en), ...] from Gemma base mode A."""
    out: dict[str, list[tuple[str, str]]] = {}
    with open(GEMMA_BASE) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[rec["slug"]] = list(zip(rec["jp_texts"], rec["en_texts"]))
    return out


def load_variant(slug: str, base_dir: Path) -> list[tuple[str, str]]:
    """Return [(jp, en), ...] from a variant's stats.json."""
    p = base_dir / slug / "stats.json"
    if not p.exists():
        return []
    with open(p) as f:
        s = json.load(f)
    jps = s.get("ocr_samples") or []
    ens = s.get("translations") or []
    return list(zip(jps, ens))


def main() -> int:
    gemma = load_gemma_base()
    slugs = sorted(p.name for p in V7_DIR.iterdir() if p.is_dir() and p.name.isdigit())

    css = """
    body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; padding: 16px; background: #0c0d10; color: #e7e9ed; }
    h1 { font-size: 20px; margin: 0 0 12px; }
    .legend { color: #8b95a5; font-size: 13px; margin-bottom: 16px; }
    .page { background: #14161b; border: 1px solid #21242c; border-radius: 8px; padding: 12px; margin-bottom: 18px; }
    .page-title { font-weight: 600; font-size: 16px; margin-bottom: 8px; }
    .imgs { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
    .imgs figure { margin: 0; max-width: 320px; }
    .imgs img { display: block; width: 100%; height: auto; border-radius: 4px; }
    .imgs figcaption { color: #8b95a5; font-size: 11.5px; text-align: center; margin-top: 4px; }
    table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin-top: 6px; }
    th, td { border-bottom: 1px solid #1d2028; padding: 6px 8px; vertical-align: top; }
    th { text-align: left; color: #8b95a5; font-weight: 500; font-size: 12px; }
    .jp { font-family: "Noto Sans CJK JP", sans-serif; color: #cfd5e0; min-width: 200px; max-width: 280px; }
    .v7  { color: #5fe39d; }
    .gem { color: #a4b9ff; }
    .nsf { color: #ffcf6e; }
    .wsf { color: #f78fb3; }
    .idx { color: #51596a; font-variant-numeric: tabular-nums; width: 28px; }
    """

    rows = []
    n_total = 0
    n_match = 0
    for slug in slugs:
        v7 = load_variant(slug, V7_DIR)
        nsf = load_variant(slug, V26_NOSFX_DIR)
        wsf = load_variant(slug, V26_WITHSFX_DIR)
        gem = gemma.get(slug, [])

        # thumbnails
        orig_thumb_p = ORIG_DIR / f"{slug}.webp"
        if not orig_thumb_p.exists():
            orig_thumb_p = ORIG_DIR / f"{slug}.jpg"

        thumbs = [
            ("Original", thumb_b64(orig_thumb_p)),
            ("v7 (old CTD)", thumb_b64(V7_OLD_FINALS / f"{slug}.png")),
            ("v7 + v26 (no SFX)", thumb_b64(V26_NOSFX_FINALS / f"{slug}.png")),
            ("v7 + v26 (with SFX)", thumb_b64(V26_WITHSFX_FINALS / f"{slug}.png")),
        ]
        thumb_html = "".join(
            f'<figure><img src="{src}" /><figcaption>{html.escape(label)}</figcaption></figure>'
            if src else
            f'<figure><div style="width:320px;height:320px;background:#1d2028;display:flex;align-items:center;justify-content:center;color:#51596a">missing</div><figcaption>{html.escape(label)}</figcaption></figure>'
            for label, src in thumbs
        )

        # Per-bubble text rows: align by JP. v7-old is the reference.
        # If a JP appears in v7 but not in the v26 variants, that's fine — show '-'.
        body_rows = []
        for i, (jp, en_v7) in enumerate(v7, 1):
            en_gem = next((g for j, g in gem if j.strip() == jp.strip()), "")
            en_nsf = next((e for j, e in nsf if j.strip() == jp.strip()), "")
            en_wsf = next((e for j, e in wsf if j.strip() == jp.strip()), "")
            n_total += 1
            same = en_v7.strip().lower() == en_gem.strip().lower() and en_v7.strip()
            if same:
                n_match += 1
            body_rows.append(
                f'<tr><td class="idx">{i}</td>'
                f'<td class="jp">{html.escape(jp)}</td>'
                f'<td class="v7">{html.escape(en_v7)}</td>'
                f'<td class="nsf">{html.escape(en_nsf or "—")}</td>'
                f'<td class="wsf">{html.escape(en_wsf or "—")}</td>'
                f'<td class="gem">{html.escape(en_gem or "—")}</td></tr>'
            )

        rows.append(f"""
        <div class="page">
          <div class="page-title">Page {slug} · {len(v7)} bubbles (v7 old) · {len(nsf)} no-sfx · {len(wsf)} with-sfx · {len(gem)} gemma</div>
          <div class="imgs">{thumb_html}</div>
          <table>
            <thead><tr><th></th><th>JP (OCR)</th><th>v7 (old CTD)</th><th>v7 + v26 no-SFX</th><th>v7 + v26 with-SFX</th><th>Gemma 3 4B base</th></tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
          </table>
        </div>
        """)

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>v7 + v26 detector — full comparison</title>
<style>{css}</style></head>
<body>
<h1>v7 (Qwen3-4B-mt) — full comparison · 644289</h1>
<div class="legend">
{n_total} bubbles across {len(slugs)} pages · v7-vs-Gemma exact-match (case-insensitive): {n_match}/{n_total} ({100*n_match/max(1,n_total):.1f}%)<br>
Columns: <span class="v7">v7 (old CTD)</span> · <span class="nsf">v7 + v26 no-SFX</span> · <span class="wsf">v7 + v26 with-SFX</span> · <span class="gem">Gemma 3 4B base</span>
</div>
{''.join(rows)}
</body></html>"""

    OUT_HTML.write_text(html_doc, encoding="utf-8")
    print(f"wrote {OUT_HTML} ({OUT_HTML.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

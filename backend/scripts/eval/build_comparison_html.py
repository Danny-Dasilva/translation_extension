"""Build a single HTML page comparing v7 vs Gemma-base vs Gemma-uncensored,
row-per-bubble with embedded thumbnails of the rendered final pages.

Inputs:
  - /home/danny/manga-output/644289-qwen3mt-v7/NNN/stats.json  (v7 translations)
  - /home/danny/manga-output/644289-abc-gemma4-base-45/modeA.jsonl
  - /home/danny/manga-output/644289-abc-gemma4-uncensored-v2/modeA.jsonl
  - /home/danny/manga-output/644289-qwen3mt-v7-finals/NNN.png   (full-page composites)

Output: /home/danny/manga-output/v7-vs-gemma.html
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

V7_DIR = Path("/home/danny/manga-output/644289-qwen3mt-v7")
GEMMA_BASE = Path("/home/danny/manga-output/644289-abc-gemma4-base-45/modeA.jsonl")
GEMMA_UNC = Path("/home/danny/manga-output/644289-abc-gemma4-uncensored-v2/modeA.jsonl")
V7_FINALS = Path("/home/danny/manga-output/644289-qwen3mt-v7-finals")
GEMMA_BASE_FINALS = Path("/home/danny/manga-output/644289-hymt-finals")  # closest reference if available
OUT = Path("/home/danny/manga-output/v7-vs-gemma.html")


def load_gemma(path: Path) -> dict[str, list[str]]:
    """slug -> en_texts list."""
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["slug"]] = r.get("en_texts") or []
    return out


def thumb_b64(p: Path, max_w: int = 280) -> str | None:
    if not p.exists():
        return None
    try:
        from PIL import Image
        im = Image.open(p)
        w, h = im.size
        if w > max_w:
            im = im.resize((max_w, int(h * max_w / w)))
        import io
        buf = io.BytesIO()
        im.save(buf, format="WEBP", quality=72)
        return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def is_garbage_jp(jp: str) -> bool:
    """Mirror v7 cleaner heuristic — for highlighting noisy OCR rows."""
    if not jp or len(jp) < 2:
        return True
    import re
    JP_CHAR = re.compile(r"[぀-ヿ一-鿿々〆〇]")
    total = sum(1 for c in jp if c.isalnum() or JP_CHAR.match(c))
    if total == 0:
        return True
    ja = sum(1 for c in jp if JP_CHAR.match(c))
    if ja / max(1, total) < 0.4:
        return True
    if re.search(r"(.)\1{3,}", jp) and len(set(jp)) <= 3:
        return True
    return False


def main() -> int:
    gem_base = load_gemma(GEMMA_BASE)
    gem_unc = load_gemma(GEMMA_UNC)

    page_dirs = sorted(p for p in V7_DIR.iterdir() if p.is_dir() and p.name.isdigit())

    rows: list[str] = []
    for pdir in page_dirs:
        stats_p = pdir / "stats.json"
        if not stats_p.exists():
            continue
        with open(stats_p) as f:
            stats = json.load(f)
        slug = pdir.name
        ocr = stats.get("ocr_samples") or []
        v7_trans = stats.get("translations") or []
        gemma_b = gem_base.get(slug, [])
        gemma_u = gem_unc.get(slug, [])

        # Page header with thumb
        v7_thumb = thumb_b64(V7_FINALS / f"{slug}.png", max_w=520)
        img_tag = f'<img src="{v7_thumb}">' if v7_thumb else ""
        rows.append(f'<tr class="pagehdr"><td colspan="5">'
                    f'<h2>Page {slug}</h2>{img_tag}</td></tr>')
        rows.append('<tr class="colhdr">'
                    '<th>#</th><th>JP (OCR)</th>'
                    '<th>v7 (Qwen3-4B-mt)</th>'
                    '<th>Gemma 4 base</th>'
                    '<th>Gemma 4 uncensored</th>'
                    '</tr>')

        n = max(len(ocr), len(v7_trans), len(gemma_b), len(gemma_u))
        for i in range(n):
            jp = ocr[i] if i < len(ocr) else ""
            v7 = v7_trans[i] if i < len(v7_trans) else ""
            gb = gemma_b[i] if i < len(gemma_b) else ""
            gu = gemma_u[i] if i < len(gemma_u) else ""
            cls = "garbage" if is_garbage_jp(jp) else ""
            rows.append(
                f'<tr class="{cls}">'
                f'<td class="idx">{i+1}</td>'
                f'<td class="jp">{html.escape(jp)}</td>'
                f'<td class="en v7">{html.escape(v7)}</td>'
                f'<td class="en">{html.escape(gb)}</td>'
                f'<td class="en">{html.escape(gu)}</td>'
                f'</tr>'
            )

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background:#0e0f12; color:#e6e6e6; padding:16px; }
    h1 { color:#fff; margin-bottom:4px; }
    .meta { color:#8aa; margin-bottom:16px; font-size:13px; }
    table { border-collapse: collapse; width: 100%; margin-bottom:24px; }
    td, th { border:1px solid #333; padding:8px 10px; vertical-align:top; }
    th { background:#1a1d24; text-align:left; }
    tr.pagehdr td { background:#161821; padding:18px; border:none; }
    tr.pagehdr h2 { margin:0 0 10px 0; color:#7df; }
    tr.pagehdr img { max-width:520px; border:1px solid #333; }
    tr.colhdr th { font-size:12px; text-transform:uppercase; color:#9ab; background:#1f242c; }
    tr.garbage td { background:#341e1e; }
    td.idx { width:36px; text-align:right; color:#888; font-family:monospace; }
    td.jp { width:24%; font-size:14px; line-height:1.5; }
    td.en { width:24%; font-size:14px; line-height:1.45; }
    td.en.v7 { background:#162a1a; }
    """

    legend = (
        '<div class="meta">'
        f'Pages: {len(page_dirs)} · '
        'Red rows = OCR garbage (low JA char ratio or stuttering). '
        'Green column = v7 (our trained model). '
        'Gemma base/unc = Gemma 3 4B-IT teacher modes A.'
        '</div>'
    )

    out_html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<title>v7 vs Gemma 3 4B — 644289</title>'
        f'<style>{css}</style></head><body>'
        '<h1>v7 (Qwen3-4B-mt-v7) vs Gemma 3 4B teacher — 644289</h1>'
        + legend +
        '<table>' + "\n".join(rows) + '</table>'
        '</body></html>'
    )
    OUT.write_text(out_html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

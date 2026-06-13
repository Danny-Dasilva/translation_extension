"""3-way comparison: v7 (Qwen3-4B) vs v9c (Gemma 4 E4B) vs Gemma 3 4B base teacher."""
from __future__ import annotations
import base64, html, json, sys
from io import BytesIO
from pathlib import Path
from PIL import Image

OUT = Path("/home/danny/manga-output/v7-v9c-gemma.html")
ORIG = Path("/home/danny/manga-output/644289/originals")
V7 = Path("/home/danny/manga-output/644289-qwen3mt-v7")
V9C = Path("/home/danny/manga-output/644289-gemma4-v9c-unsloth")
V7_FIN = Path("/home/danny/manga-output/644289-qwen3mt-v7-finals")
V9C_FIN = Path("/home/danny/manga-output/644289-gemma4-v9c-finals")
GEMMA = Path("/home/danny/manga-output/644289-abc-gemma4-base-45/modeA.jsonl")


def thumb(p, w=300):
    if not p.exists(): return ""
    im = Image.open(p).convert("RGB")
    if im.size[0] > w:
        im = im.resize((w, int(im.size[1]*w/im.size[0])), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=68)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def loadv(slug, base):
    p = base / slug / "stats.json"
    if not p.exists(): return []
    s = json.load(open(p))
    return list(zip(s.get("ocr_samples") or [], s.get("translations") or []))


def main():
    g = {}
    with open(GEMMA) as f:
        for line in f:
            if line.strip():
                r = json.loads(line); g[r["slug"]] = list(zip(r["jp_texts"], r["en_texts"]))
    slugs = sorted(p.name for p in V9C.iterdir() if p.is_dir() and p.name.isdigit())
    css = """body{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;padding:14px;background:#0c0d10;color:#e7e9ed}h1{font-size:20px;margin:0 0 8px}.legend{color:#8b95a5;font-size:13px;margin-bottom:14px}.page{background:#14161b;border:1px solid #21242c;border-radius:8px;padding:10px;margin-bottom:14px}.imgs{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px}.imgs figure{margin:0;max-width:300px}.imgs img{width:100%;border-radius:4px}.imgs figcaption{color:#8b95a5;font-size:11px;text-align:center;margin-top:3px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #1d2028;padding:6px 8px;vertical-align:top}th{text-align:left;color:#8b95a5;font-weight:500;font-size:11.5px}.jp{font-family:"Noto Sans CJK JP",sans-serif;color:#cfd5e0;min-width:180px;max-width:260px}.v7{color:#5fe39d}.v9{color:#a4b9ff}.gem{color:#ffcf6e}.idx{color:#51596a;width:24px}.same{background:#142413!important}.diff{background:#241612!important}"""
    rows = []
    for slug in slugs:
        v7t = loadv(slug, V7)
        v9 = loadv(slug, V9C)
        gm = g.get(slug, [])
        op = ORIG / f"{slug}.webp"
        if not op.exists(): op = ORIG / f"{slug}.jpg"
        thumbs = [("Original", thumb(op)), ("v7 (Qwen3-4B)", thumb(V7_FIN / f"{slug}.png")), ("v9c (Gemma 4 E4B)", thumb(V9C_FIN / f"{slug}.png"))]
        timg = "".join(f'<figure><img src="{src}"><figcaption>{html.escape(l)}</figcaption></figure>' if src else f'<figure><div style="width:300px;height:300px;background:#1d2028"></div><figcaption>{l}</figcaption></figure>' for l,src in thumbs)
        body = []
        for i, (jp, ev7) in enumerate(v7t, 1):
            ev9 = next((e for j,e in v9 if j.strip()==jp.strip()), "—")
            eg = next((e for j,e in gm if j.strip()==jp.strip()), "—")
            m7 = ev7.strip().lower() == eg.strip().lower() and ev7.strip()
            m9 = ev9.strip().lower() == eg.strip().lower() and ev9.strip()
            cls = "same" if (m7 and m9) else ("diff" if (m7 != m9) else "")
            body.append(f'<tr class="{cls}"><td class="idx">{i}</td><td class="jp">{html.escape(jp)}</td><td class="v7">{html.escape(ev7)}</td><td class="v9">{html.escape(ev9)}</td><td class="gem">{html.escape(eg)}</td></tr>')
        rows.append(f'<div class="page"><div style="font-weight:600;margin-bottom:6px">Page {slug} · {len(v7t)} bubbles</div><div class="imgs">{timg}</div><table><thead><tr><th></th><th>JP</th><th>v7 (Qwen3-4B)</th><th>v9c (Gemma 4 E4B)</th><th>Gemma 3 4B base</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>')

    OUT.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>v7 vs v9c vs Gemma — 644289</title><style>{css}</style></head><body><h1>v7 (Qwen3-4B) vs v9c (Gemma 4 E4B) vs Gemma 3 4B base — 644289</h1><div class="legend">Green: both v7+v9c match Gemma · Red: only one matches.<br>v7 exact-match Gemma: 89/257 (34.6%) · v9c exact-match Gemma: 85/257 (33.1%) · v9c-only wins: 3 · v7-only wins: 7</div>{"".join(rows)}</body></html>', encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

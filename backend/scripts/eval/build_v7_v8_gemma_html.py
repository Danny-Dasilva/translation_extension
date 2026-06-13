"""Build a 3-way comparison HTML: v7 / v8-distill / Gemma 3 4B base on 644289."""
from __future__ import annotations

import base64, html, json, sys
from io import BytesIO
from pathlib import Path

from PIL import Image

OUT_HTML = Path("/home/danny/manga-output/v7-v8-gemma.html")
ORIG = Path("/home/danny/manga-output/644289/originals")
V7_DIR = Path("/home/danny/manga-output/644289-qwen3mt-v7")
V8_DIR = Path("/home/danny/manga-output/644289-qwen3mt-v8")
V7_FINALS = Path("/home/danny/manga-output/644289-qwen3mt-v7-finals")
V8_FINALS = Path("/home/danny/manga-output/644289-qwen3mt-v8-finals")
GEMMA = Path("/home/danny/manga-output/644289-abc-gemma4-base-45/modeA.jsonl")


def thumb(p: Path, w=300):
    if not p.exists(): return ""
    im = Image.open(p).convert("RGB")
    if im.size[0] > w:
        im = im.resize((w, int(im.size[1]*w/im.size[0])), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=68)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_var(slug: str, base: Path):
    p = base / slug / "stats.json"
    if not p.exists(): return []
    s = json.load(open(p))
    return list(zip(s.get("ocr_samples") or [], s.get("translations") or []))


def load_gemma():
    out = {}
    with open(GEMMA) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["slug"]] = list(zip(r["jp_texts"], r["en_texts"]))
    return out


def main():
    g = load_gemma()
    slugs = sorted(p.name for p in V7_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    css = """
    body{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;padding:14px;background:#0c0d10;color:#e7e9ed}
    h1{font-size:20px;margin:0 0 8px}
    .legend{color:#8b95a5;font-size:13px;margin-bottom:14px}
    .page{background:#14161b;border:1px solid #21242c;border-radius:8px;padding:10px;margin-bottom:14px}
    .imgs{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:8px}
    .imgs figure{margin:0;max-width:300px}.imgs img{width:100%;border-radius:4px}
    .imgs figcaption{color:#8b95a5;font-size:11px;text-align:center;margin-top:3px}
    table{border-collapse:collapse;width:100%;font-size:13px}
    th,td{border-bottom:1px solid #1d2028;padding:6px 8px;vertical-align:top}
    th{text-align:left;color:#8b95a5;font-weight:500;font-size:11.5px}
    .jp{font-family:"Noto Sans CJK JP",sans-serif;color:#cfd5e0;min-width:180px;max-width:260px}
    .v7{color:#5fe39d}.v8{color:#ffcf6e}.gem{color:#a4b9ff}
    .idx{color:#51596a;width:24px}
    .reg{background:#241612!important}.imp{background:#142413!important}
    """
    rows = []
    for slug in slugs:
        v7 = load_var(slug, V7_DIR)
        v8 = load_var(slug, V8_DIR)
        gm = g.get(slug, [])
        op = ORIG / f"{slug}.webp"
        if not op.exists(): op = ORIG / f"{slug}.jpg"
        thumbs = [
            ("Original", thumb(op)),
            ("v7", thumb(V7_FINALS / f"{slug}.png")),
            ("v8 distill", thumb(V8_FINALS / f"{slug}.png")),
        ]
        timg = "".join(f'<figure><img src="{src}"><figcaption>{html.escape(l)}</figcaption></figure>' if src else f'<figure><div style="width:300px;height:300px;background:#1d2028"></div><figcaption>{l}</figcaption></figure>' for l,src in thumbs)
        body = []
        for i, (jp, ev7) in enumerate(v7, 1):
            ev8 = next((e for j,e in v8 if j.strip()==jp.strip()), "—")
            eg = next((e for j,e in gm if j.strip()==jp.strip()), "—")
            # mark regression: v7 matches gemma but v8 doesn't
            v7_ok = ev7.strip().lower() == eg.strip().lower() and ev7.strip()
            v8_ok = ev8.strip().lower() == eg.strip().lower() and ev8.strip()
            cls = "reg" if (v7_ok and not v8_ok) else "imp" if (not v7_ok and v8_ok) else ""
            body.append(f'<tr class="{cls}"><td class="idx">{i}</td><td class="jp">{html.escape(jp)}</td><td class="v7">{html.escape(ev7)}</td><td class="v8">{html.escape(ev8)}</td><td class="gem">{html.escape(eg)}</td></tr>')
        rows.append(f"""<div class="page"><div style="font-weight:600;margin-bottom:6px">Page {slug} · {len(v7)} bubbles</div><div class="imgs">{timg}</div><table><thead><tr><th></th><th>JP</th><th>v7</th><th>v8 distill</th><th>Gemma 3 4B base</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>""")

    OUT_HTML.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>v7 vs v8 vs Gemma — 644289</title><style>{css}</style></head><body><h1>v7 vs v8-distill vs Gemma 3 4B base — 644289</h1><div class="legend">Red rows = v7 matched Gemma but v8 regressed. Green rows = v8 newly matches Gemma.</div>{''.join(rows)}</body></html>""", encoding="utf-8")
    print(f"wrote {OUT_HTML} ({OUT_HTML.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

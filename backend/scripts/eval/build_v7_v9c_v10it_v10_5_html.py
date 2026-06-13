"""4-way comparison: v7 vs v9c vs v10-it vs v10.5 (CPO chain).

Each row also shows the Gemma 3 4B base teacher reference for context. Per-row
exact-match (vs Gemma teacher) is highlighted by how many of the 4 systems
match: green = all 4 match, olive = 3 match, dark olive = 2 match,
red = exactly 1 matches.

Usage
-----
    python backend/scripts/eval/build_v7_v9c_v10it_v10_5_html.py

Defaults to:
    /home/danny/manga-output/644289-qwen3mt-v7
    /home/danny/manga-output/644289-gemma4-v9c-unsloth
    /home/danny/manga-output/644289-gemma4-v10it-unsloth-fixed
    /home/danny/manga-output/644289-gemma4-v10_5-unsloth
    /home/danny/manga-output/644289-abc-gemma4-base-45/modeA.jsonl

Override any path via env vars (V7_DIR, V9C_DIR, V10IT_DIR, V10_5_DIR, GEMMA_REF, OUT_PATH).
"""
from __future__ import annotations
import base64, html, json, os, sys
from io import BytesIO
from pathlib import Path
from PIL import Image


OUT = Path(os.environ.get("OUT_PATH", "/home/danny/manga-output/v7-v9c-v10it-v10_5.html"))
ORIG = Path("/home/danny/manga-output/644289/originals")
V7 = Path(os.environ.get("V7_DIR", "/home/danny/manga-output/644289-qwen3mt-v7"))
V9C = Path(os.environ.get("V9C_DIR", "/home/danny/manga-output/644289-gemma4-v9c-unsloth"))
V10IT = Path(os.environ.get("V10IT_DIR", "/home/danny/manga-output/644289-gemma4-v10it-unsloth-fixed"))
V10_5 = Path(os.environ.get("V10_5_DIR", "/home/danny/manga-output/644289-gemma4-v10_5-unsloth"))
V7_FIN = Path("/home/danny/manga-output/644289-qwen3mt-v7-finals")
V9C_FIN = Path("/home/danny/manga-output/644289-gemma4-v9c-finals")
GEMMA = Path(os.environ.get("GEMMA_REF", "/home/danny/manga-output/644289-abc-gemma4-base-45/modeA.jsonl"))


def thumb(p, w=300):
    if not p.exists():
        return ""
    im = Image.open(p).convert("RGB")
    if im.size[0] > w:
        im = im.resize((w, int(im.size[1] * w / im.size[0])), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=68)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def loadv(slug, base):
    p = base / slug / "stats.json"
    if not p.exists():
        return []
    s = json.load(open(p))
    return list(zip(s.get("ocr_samples") or [], s.get("translations") or []))


def main():
    g = {}
    if GEMMA.exists():
        with open(GEMMA) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    g[r["slug"]] = list(zip(r["jp_texts"], r["en_texts"]))

    # Use v10.5 as the canonical slug list if it exists, else fall back to v10-it
    canonical_dir = V10_5 if V10_5.exists() and any(p.is_dir() for p in V10_5.iterdir()) else V10IT
    slugs = sorted(p.name for p in canonical_dir.iterdir() if p.is_dir() and p.name.isdigit())

    css = (
        "body{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;padding:14px;"
        "background:#0c0d10;color:#e7e9ed}"
        "h1{font-size:20px;margin:0 0 8px}"
        ".legend{color:#8b95a5;font-size:13px;margin-bottom:14px}"
        ".page{background:#14161b;border:1px solid #21242c;border-radius:8px;padding:10px;margin-bottom:14px}"
        ".imgs{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px}"
        ".imgs figure{margin:0;max-width:300px}"
        ".imgs img{width:100%;border-radius:4px}"
        ".imgs figcaption{color:#8b95a5;font-size:11px;text-align:center;margin-top:3px}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border-bottom:1px solid #1d2028;padding:6px 8px;vertical-align:top}"
        "th{text-align:left;color:#8b95a5;font-weight:500;font-size:11.5px}"
        '.jp{font-family:"Noto Sans CJK JP",sans-serif;color:#cfd5e0;min-width:180px;max-width:260px}'
        ".v7{color:#5fe39d}"
        ".v9{color:#a4b9ff}"
        ".v10{color:#ff9be0}"
        ".v105{color:#ffd166}"
        ".gem{color:#ffcf6e}"
        ".idx{color:#51596a;width:24px}"
        ".all4{background:#142413!important}"
        ".three{background:#1a2018!important}"
        ".two{background:#1f2114!important}"
        ".one{background:#241612!important}"
    )

    em_v7 = em_v9 = em_v10 = em_v105 = 0
    aligned_total = 0
    v7_only = v9_only = v10_only = v105_only = 0

    rows = []
    for slug in slugs:
        v7t = loadv(slug, V7)
        v9 = loadv(slug, V9C)
        v10 = loadv(slug, V10IT)
        v105 = loadv(slug, V10_5)
        gm = g.get(slug, [])

        op = ORIG / f"{slug}.webp"
        if not op.exists():
            op = ORIG / f"{slug}.jpg"

        thumbs = [
            ("Original", thumb(op)),
            ("v7 (Qwen3-4B)", thumb(V7_FIN / f"{slug}.png")),
            ("v9c (Gemma 4 E4B)", thumb(V9C_FIN / f"{slug}.png")),
        ]
        timg = "".join(
            f'<figure><img src="{src}"><figcaption>{html.escape(l)}</figcaption></figure>'
            if src
            else f'<figure><div style="width:300px;height:300px;background:#1d2028"></div><figcaption>{l}</figcaption></figure>'
            for l, src in thumbs
        )

        body = []
        # Iterate over v10.5 if present, else v10-it
        if v105:
            iter_pairs = v105
        elif v10:
            iter_pairs = v10
        else:
            iter_pairs = v9 or v7t

        for i, (jp, ev_iter) in enumerate(iter_pairs, 1):
            ev7 = next((e for j, e in v7t if j.strip() == jp.strip()), "—")
            ev9 = next((e for j, e in v9 if j.strip() == jp.strip()), "—")
            ev10 = next((e for j, e in v10 if j.strip() == jp.strip()), "—")
            ev105 = next((e for j, e in v105 if j.strip() == jp.strip()), "—")
            eg = next((e for j, e in gm if j.strip() == jp.strip()), "—")

            def mat(p, ref):
                return p.strip().lower() == ref.strip().lower() and p.strip()

            m7 = bool(mat(ev7, eg)) if eg != "—" else False
            m9 = bool(mat(ev9, eg)) if eg != "—" else False
            m10 = bool(mat(ev10, eg)) if eg != "—" else False
            m105 = bool(mat(ev105, eg)) if eg != "—" else False

            if eg != "—":
                aligned_total += 1
                if m7: em_v7 += 1
                if m9: em_v9 += 1
                if m10: em_v10 += 1
                if m105: em_v105 += 1
                if m7 and not (m9 or m10 or m105): v7_only += 1
                if m9 and not (m7 or m10 or m105): v9_only += 1
                if m10 and not (m7 or m9 or m105): v10_only += 1
                if m105 and not (m7 or m9 or m10): v105_only += 1

            n_match = int(m7) + int(m9) + int(m10) + int(m105)
            if n_match == 4:
                cls = "all4"
            elif n_match == 3:
                cls = "three"
            elif n_match == 2:
                cls = "two"
            elif n_match == 1:
                cls = "one"
            else:
                cls = ""

            body.append(
                f'<tr class="{cls}"><td class="idx">{i}</td>'
                f'<td class="jp">{html.escape(jp)}</td>'
                f'<td class="v7">{html.escape(ev7)}</td>'
                f'<td class="v9">{html.escape(ev9)}</td>'
                f'<td class="v10">{html.escape(ev10)}</td>'
                f'<td class="v105">{html.escape(ev105)}</td>'
                f'<td class="gem">{html.escape(eg)}</td></tr>'
            )

        rows.append(
            f'<div class="page"><div style="font-weight:600;margin-bottom:6px">'
            f'Page {slug} · {len(iter_pairs)} bubbles</div>'
            f'<div class="imgs">{timg}</div>'
            f'<table><thead><tr><th></th><th>JP</th>'
            f'<th>v7 (Qwen3-4B)</th><th>v9c (Gemma 4 E4B)</th>'
            f'<th>v10-it</th><th>v10.5 (CPO)</th><th>Gemma 3 4B base</th>'
            f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
        )

    em_summary = (
        f"v7 exact-match Gemma: {em_v7}/{aligned_total} ({100*em_v7/max(1,aligned_total):.1f}%) · "
        f"v9c: {em_v9}/{aligned_total} ({100*em_v9/max(1,aligned_total):.1f}%) · "
        f"v10-it: {em_v10}/{aligned_total} ({100*em_v10/max(1,aligned_total):.1f}%) · "
        f"v10.5: {em_v105}/{aligned_total} ({100*em_v105/max(1,aligned_total):.1f}%)"
        f"<br>exclusive wins: v7-only {v7_only} · v9c-only {v9_only} · v10-only {v10_only} · v10.5-only {v105_only}"
    )

    OUT.write_text(
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>v7 vs v9c vs v10-it vs v10.5 — 644289</title>'
        f"<style>{css}</style></head><body>"
        f"<h1>v7 (Qwen3-4B) vs v9c (Gemma 4 E4B) vs v10-it vs v10.5 (CPO chain) — 644289</h1>"
        f'<div class="legend">Green: all 4 match Gemma · Olive: 3 of 4 · Dark: 2 of 4 · Red: exactly 1.<br>{em_summary}</div>'
        f'{"".join(rows)}</body></html>',
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
    print(em_summary.replace("<br>", "\n"))


if __name__ == "__main__":
    main()

"""A/B/C translation eval for Gemma 4 31B — text vs full-page vs bubble-crop.

Talks to any OpenAI-compatible chat-completions endpoint (llama-server,
llama-cpp-python server, vLLM). See ``README.md`` in this folder for the
recommended launch command.

For each page slug in ``--gallery`` the script runs the three configured
modes and writes one JSONL record per (slug, mode) to
``<out>/mode{A,B,C}.jsonl`` with fields::

    {slug, mode, ms, jp_texts, en_texts, raw, tag_integrity, num_tags}

Mode C re-runs the CTD detector to recover per-bubble bounding boxes,
since the gallery only persists a composite crop montage.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.translation_text_utils import (  # noqa: E402
    BATCHED_SYSTEM_PROMPT,
    format_sources,
    parse_tagged_blocks,
    split_legacy_lines,
    strip_thinking_block,
    strip_wrapping_quotes,
)

GEN_KWARGS = dict(
    temperature=0.1,
    top_p=0.9,
    max_tokens=1024,
    # Gemma 4's chat template writes its reasoning into a `<|channel>thought`
    # block inside the output stream by default — ~9× slower and hallucination-
    # prone. `chat_template_kwargs: enable_thinking=false` disables it at the
    # template level (verified: 2286ms → 257ms on a 2-tag translate).
    chat_template_kwargs={"enable_thinking": False},
)


def _parse_prompt_sources(prompt_text: str) -> List[str]:
    """Extract ``[N]text`` JP blocks from an existing 08_translate_prompt.txt."""
    out: List[str] = []
    current: Optional[List[str]] = None
    for raw in prompt_text.splitlines():
        stripped = raw.lstrip()
        if stripped.startswith("[") and "]" in stripped:
            idx_close = stripped.index("]")
            inner = stripped[1:idx_close]
            if inner.isdigit():
                if current is not None:
                    out.append("\n".join(current).strip())
                current = [stripped[idx_close + 1:].lstrip()]
                continue
        if current is not None:
            current.append(raw.rstrip())
    if current is not None:
        out.append("\n".join(current).strip())
    return out


def _pil_to_data_url(img: Image.Image, fmt: str = "PNG", max_side: int = 1024) -> str:
    """Downscale if needed (vision encoders usually cap at ~1024px) and return data URL."""
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format=fmt, optimize=True)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"


def _build_messages_text_only(jp_texts: List[str], target: str) -> List[dict]:
    return [
        {"role": "system", "content": BATCHED_SYSTEM_PROMPT.format(target=target)},
        {"role": "user", "content": format_sources(jp_texts)},
    ]


def _build_messages_full_page(jp_texts: List[str], page_png: Path, target: str) -> List[dict]:
    page = Image.open(page_png).convert("RGB")
    data_url = _pil_to_data_url(page)
    user_text = (
        "Here is the manga page. The OCR has extracted these numbered blocks "
        "(some may be garbled — use the image to recover the real Japanese "
        "before translating):\n\n" + format_sources(jp_texts)
    )
    return [
        {"role": "system", "content": BATCHED_SYSTEM_PROMPT.format(target=target)},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def _build_messages_per_bubble(
    jp_texts: List[str], crops: List[Image.Image], target: str
) -> List[dict]:
    content: List[dict] = []
    for i, (jp, crop) in enumerate(zip(jp_texts, crops), start=1):
        content.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(crop, max_side=512)}})
        content.append({"type": "text", "text": f"[{i}]{jp}"})
    content.append({
        "type": "text",
        "text": (
            "Each image above is the speech-bubble crop for the tag that "
            "follows it. Use the crop to recover the real Japanese if OCR "
            "garbled it, then translate."
        ),
    })
    return [
        {"role": "system", "content": BATCHED_SYSTEM_PROMPT.format(target=target)},
        {"role": "user", "content": content},
    ]


def _call_server(server: str, messages: List[dict]) -> Tuple[str, float]:
    url = server.rstrip("/") + "/v1/chat/completions"
    body = {"model": "gemma-4-31b", "messages": messages, **GEN_KWARGS}
    t0 = time.perf_counter()
    r = requests.post(url, json=body, timeout=600)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or "", ms


def _parse_output(raw: str, n: int) -> Tuple[List[str], float]:
    """Return (en_texts, tag_integrity_rate) where rate is fraction of tags filled."""
    cleaned = strip_thinking_block(raw.strip())
    parsed = parse_tagged_blocks(cleaned, n)
    if parsed is None:
        parsed = split_legacy_lines(cleaned, n)
    out = [strip_wrapping_quotes(b.strip()) for b in parsed]
    filled = sum(1 for x in out if x)
    return out, (filled / n) if n else 0.0


async def _detect_blocks(img_rgb: np.ndarray) -> List[dict]:
    """Lazy-load CTD and return block dicts with xyxy bboxes."""
    from app.services.detector_factory import create_detector  # noqa: WPS433

    detector = create_detector()
    result = await detector.detect(img_rgb, input_is_bgr=False)
    return result["blocks"]


def _crop_bubbles(img_rgb: np.ndarray, blocks: List[dict], n_expected: int) -> List[Image.Image]:
    """Return one PIL crop per expected tag. Pads with blank if detector under-counts."""
    crops: List[Image.Image] = []
    for b in blocks[:n_expected]:
        x1 = int(b.get("minX", b.get("xyxy", [0, 0, 0, 0])[0]))
        y1 = int(b.get("minY", b.get("xyxy", [0, 0, 0, 0])[1]))
        x2 = int(b.get("maxX", b.get("xyxy", [0, 0, 0, 0])[2]))
        y2 = int(b.get("maxY", b.get("xyxy", [0, 0, 0, 0])[3]))
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_rgb.shape[1], x2); y2 = min(img_rgb.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            crops.append(Image.new("RGB", (16, 16), "white"))
            continue
        crops.append(Image.fromarray(img_rgb[y1:y2, x1:x2]))
    while len(crops) < n_expected:
        crops.append(Image.new("RGB", (16, 16), "white"))
    return crops


async def _run_page(slug_dir: Path, server: str, modes: List[str], target: str) -> List[dict]:
    slug = slug_dir.name
    prompt_p = slug_dir / "08_translate_prompt.txt"
    orig_p = slug_dir / "01_original.png"
    if not (prompt_p.exists() and orig_p.exists()):
        return []

    jp_texts = _parse_prompt_sources(prompt_p.read_text(encoding="utf-8"))
    n = len(jp_texts)
    if n == 0:
        return []

    records: List[dict] = []

    if "A" in modes:
        messages = _build_messages_text_only(jp_texts, target)
        raw, ms = await asyncio.to_thread(_call_server, server, messages)
        en, integ = _parse_output(raw, n)
        records.append({
            "slug": slug, "mode": "A", "ms": ms, "num_tags": n,
            "jp_texts": jp_texts, "en_texts": en, "raw": raw,
            "tag_integrity": integ,
        })

    if "B" in modes:
        messages = _build_messages_full_page(jp_texts, orig_p, target)
        raw, ms = await asyncio.to_thread(_call_server, server, messages)
        en, integ = _parse_output(raw, n)
        records.append({
            "slug": slug, "mode": "B", "ms": ms, "num_tags": n,
            "jp_texts": jp_texts, "en_texts": en, "raw": raw,
            "tag_integrity": integ,
        })

    if "C" in modes:
        img_rgb = np.array(Image.open(orig_p).convert("RGB"))
        blocks = await _detect_blocks(img_rgb)
        crops = _crop_bubbles(img_rgb, blocks, n)
        messages = _build_messages_per_bubble(jp_texts, crops, target)
        raw, ms = await asyncio.to_thread(_call_server, server, messages)
        en, integ = _parse_output(raw, n)
        records.append({
            "slug": slug, "mode": "C", "ms": ms, "num_tags": n,
            "jp_texts": jp_texts, "en_texts": en, "raw": raw,
            "tag_integrity": integ, "num_blocks_detected": len(blocks),
        })

    return records


async def main_async(args: argparse.Namespace) -> None:
    gallery = Path(args.gallery).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pages:
        slugs = [gallery / p for p in args.pages]
    else:
        slugs = sorted(p for p in gallery.iterdir() if p.is_dir() and p.name.isdigit())

    slugs = [s for s in slugs if s.exists()]
    print(f"[eval] gallery={gallery}  pages={len(slugs)}  modes={args.modes}  server={args.server}  concurrency={args.concurrency}", flush=True)

    writers = {m: (out_dir / f"mode{m}.jsonl").open("w", encoding="utf-8") for m in args.modes}
    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)

    async def process(slug_dir: Path) -> None:
        async with sem:
            t0 = time.perf_counter()
            records = await _run_page(slug_dir, args.server, args.modes, args.target)
            ms = (time.perf_counter() - t0) * 1000
            async with write_lock:
                for rec in records:
                    writers[rec["mode"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    writers[rec["mode"]].flush()
                summary = "  ".join(
                    f"{r['mode']}:{int(r['ms'])}ms integ={r['tag_integrity']:.2f}"
                    for r in records
                )
                n = records[0]['num_tags'] if records else 0
                print(f"[{slug_dir.name}] n={n}  {summary}  total={int(ms)}ms", flush=True)

    try:
        await asyncio.gather(*(process(s) for s in slugs))
    finally:
        for w in writers.values():
            w.close()

    print(f"[eval] wrote {', '.join(f'mode{m}.jsonl' for m in args.modes)} → {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", required=True, help="e.g. ~/manga-output/644289")
    ap.add_argument("--out", required=True, help="output dir for JSONL + HTML")
    ap.add_argument("--pages", nargs="*", default=None, help="subset of slugs (default: all)")
    ap.add_argument("--modes", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    ap.add_argument("--target", default="English")
    ap.add_argument("--server", default="http://127.0.0.1:8080",
                    help="OpenAI-compatible chat-completions server root")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="concurrent pages in flight (match llama-server --parallel)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

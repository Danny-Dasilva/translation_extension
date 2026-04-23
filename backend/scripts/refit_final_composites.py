"""Re-render just the ``11_final_composite.png`` stage with a better font and
tighter shrink-to-fit. Reuses the cached ``07_inpainted.png`` plate and
parses translations from ``09_translate_response.txt``. Detection is
re-run per image (fast, 0.5-0.7s on CPU) to recover block geometry.

Usage:
    uv run python scripts/refit_final_composites.py
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.detector_factory import create_detector  # noqa: E402
from app.services.parseq_ocr_service import ParseqOCRService  # noqa: E402
from app.utils.japanese_text_filter import is_japanese_text  # noqa: E402
from app.utils.ctd_utils import build_text_regions  # noqa: E402
from app.config import settings  # noqa: E402


FONT_STACK = [
    BACKEND_DIR / "fonts" / "Anton-Regular.ttf",      # tall condensed bold — best for narration
    BACKEND_DIR / "fonts" / "Bangers-Regular.ttf",    # comic-y — nice for SFX
    BACKEND_DIR / "fonts" / "Oswald-Bold.ttf",        # fallback condensed
    BACKEND_DIR / "fonts" / "ComicNeue-Bold.ttf",     # fallback rounded
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]
for p in FONT_STACK:
    if p.exists():
        DEFAULT_FONT_PATH = p
        break
else:
    raise RuntimeError("No usable font found")


def load_font(size: int, path: Path = DEFAULT_FONT_PATH) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


# ---------------------------------------------------------------------------
# Translation parsing
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"\[(\d+)\]\s*([^\[]*)", re.DOTALL)


def parse_tagged_blocks(response: str, n: int) -> list[str]:
    """Parse [N] blocks; fall back to line-split padded to length n."""
    if not response.strip():
        return [""] * n
    matches = _TAG_RE.findall(response)
    if matches:
        by_index = {int(num): text.strip() for num, text in matches}
        return [by_index.get(i + 1, "").strip() for i in range(n)]
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    while len(lines) < n:
        lines.append("")
    return lines[:n]


# ---------------------------------------------------------------------------
# Binary-search font fit (koharu layout.rs:133-167 port)
# ---------------------------------------------------------------------------

def wrap_greedy(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
                max_w: int) -> list[str]:
    """Greedy word wrap that respects English-language word boundaries and
    falls back to character-level wrap for a single word wider than the box.
    """
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if font.getlength(trial) <= max_w or not cur:
            # single word wider than box? break it up by char
            if font.getlength(w) > max_w and not cur:
                frag = ""
                for ch in w:
                    if font.getlength(frag + ch) > max_w and frag:
                        lines.append(frag)
                        frag = ch
                    else:
                        frag += ch
                cur = frag
            else:
                cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def line_height_px(font: ImageFont.FreeTypeFont) -> int:
    """Mirror koharu's `line_height = max(ascent + descent + leading, font_size)`
    (layout.rs:186). PIL doesn't expose leading explicitly — approximate at 10 %
    of em-size, which matches what most hinted text fonts ship with.
    """
    asc, desc = font.getmetrics()
    leading = max(0, int(font.size * 0.10))
    return max(asc + desc + leading, int(font.size))


def measure_block(draw: ImageDraw.ImageDraw, lines: list[str],
                  font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Return (block_width, block_height) using the *ink* bbox for width
    (accurate for side-bearings) and a koharu-style line_height for height.
    """
    if not lines:
        return 0, 0
    widths = []
    for ln in lines:
        bb = font.getbbox(ln)
        widths.append(max(bb[2] - bb[0], int(font.getlength(ln))))
    max_w = max(widths)
    line_h = line_height_px(font)
    total_h = line_h * len(lines)
    return max_w, total_h


def find_best_fit(draw: ImageDraw.ImageDraw, text: str, max_w: int, max_h: int,
                  font_path: Path = DEFAULT_FONT_PATH,
                  min_size: int = 6, max_size: int = 72) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Binary-search the largest font size where the wrapped text fits inside
    (max_w, max_h). Falls back to min_size if nothing fits (accepts overflow).

    Mirrors /tmp/koharu/koharu-renderer/src/layout.rs:133-167 run_auto —
    same [6..300] range shape, same fallback-to-minimum behaviour.
    """
    lo, hi = min_size, max_size
    best_font = load_font(min_size, font_path)
    best_lines = wrap_greedy(draw, text, best_font, max_w) or [text]
    while lo <= hi:
        mid = (lo + hi) // 2
        font = load_font(mid, font_path)
        lines = wrap_greedy(draw, text, font, max_w) or [text]
        w, h = measure_block(draw, lines, font)
        if w <= max_w and h <= max_h:
            best_font = font
            best_lines = lines
            lo = mid + 1
        else:
            hi = mid - 1
    return best_font, best_lines


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_stroked_text(draw: ImageDraw.ImageDraw, pos: tuple[int, int], text: str,
                      font: ImageFont.FreeTypeFont, fill=(0, 0, 0), stroke=(255, 255, 255),
                      stroke_width: int = 3):
    """Pillow supports native stroke_width when Pillow ≥ 8. Koharu's renderer
    uses stroke width × 2 because its raster engine centres strokes on the
    glyph path (renderer.rs:269). PIL's stroke_width is already the full
    outer thickness, so we pass it directly.
    """
    draw.text(pos, text, fill=fill, font=font,
              stroke_width=stroke_width, stroke_fill=stroke)


def pick_font(text: str) -> Path:
    """Choose between the default narration font (Anton) and the comic-y
    Bangers for short SFX-like outbursts. Koharu picks via `named_fonts`
    from a prediction model; we use simple heuristics.
    """
    cleaned = text.strip()
    if not cleaned:
        return DEFAULT_FONT_PATH
    # Short, exclamatory, or all-caps → Bangers (comic punch)
    exclam = cleaned.count("!") + cleaned.count("?")
    short = len(cleaned) <= 8
    allcaps = cleaned.isupper() and any(c.isalpha() for c in cleaned)
    if (short and exclam >= 1) or (allcaps and len(cleaned) <= 16):
        bangers = BACKEND_DIR / "fonts" / "Bangers-Regular.ttf"
        if bangers.exists():
            return bangers
    return DEFAULT_FONT_PATH


def sample_bg_luminance(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """Mean 0-255 luminance of the background where we're about to draw.
    Koharu picks text/stroke color from a prediction model; we fall back
    to luminance-based auto-contrast (KOHARU_COMPARISON.md item #9).
    """
    h, w = image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 255.0
    crop = image[y0:y1, x0:x1].astype(np.float32)
    if crop.size == 0:
        return 255.0
    # BT.601 luminance
    lum = 0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]
    return float(lum.mean())


def compose_final(
    inpainted: np.ndarray,
    blocks: list[dict],
    translations: list[str],
    inset_margin: int = 4,
) -> np.ndarray:
    """Render translated text onto the inpainted plate using koharu's layout
    semantics: binary-search fit in [6, 96], real ink bbox, line_height =
    max(ascent + descent + leading, font_size), and a stroke width scaled
    with font size for readability against noisy backgrounds.

    Uses the full block bbox (the outer bubble interior) as the fit rect —
    matching koharu's max_width/max_height which are set from the bubble's
    interior, not the tighter per-line text_regions.

    Auto-picks black-on-white vs white-on-black based on sampled luminance
    of the inpainted background inside the rect.
    """
    pil = Image.fromarray(inpainted).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for block, text in zip(blocks, translations):
        if not text:
            continue
        x0, y0 = int(block["minX"]), int(block["minY"])
        x1, y1 = int(block["maxX"]), int(block["maxY"])

        bw, bh = x1 - x0, y1 - y0
        # Fixed-px inset preserves a consistent visual margin at any bubble size.
        inset_w = max(20, bw - inset_margin * 2)
        inset_h = max(12, bh - inset_margin * 2)

        font_path = pick_font(text)
        font, lines = find_best_fit(draw, text.strip(), inset_w, inset_h, font_path,
                                    min_size=6, max_size=96)
        mw, mh = measure_block(draw, lines, font)

        # Auto-contrast: flip to white text on dark plates.
        bg_lum = sample_bg_luminance(inpainted, x0, y0, x1, y1)
        if bg_lum < 128:
            fill, stroke = (255, 255, 255), (0, 0, 0)
        else:
            fill, stroke = (0, 0, 0), (255, 255, 255)

        line_h = line_height_px(font)
        top = y0 + (bh - mh) // 2
        # Stroke ~10 % of font size, capped so it doesn't close counters on tiny text.
        stroke_w = max(2, min(5, round(font.size * 0.10)))
        for i, ln in enumerate(lines):
            bb = font.getbbox(ln)
            lw = bb[2] - bb[0]
            # ink-bbox correction so `left` aligns with the first glyph's ink edge
            left = x0 + (bw - lw) // 2 - bb[0]
            y = top + i * line_h
            draw_stroked_text(draw, (left, y), ln, font,
                              fill=fill, stroke=stroke, stroke_width=stroke_w)
    return np.array(pil)


# ---------------------------------------------------------------------------
# Per-image re-compose
# ---------------------------------------------------------------------------

async def recompose_one(dir_: Path, detector, ocr_service) -> Optional[dict]:
    orig_path = dir_ / "01_original.png"
    inpaint_path = dir_ / "07_inpainted.png"
    response_path = dir_ / "09_translate_response.txt"
    if not (orig_path.exists() and inpaint_path.exists() and response_path.exists()):
        print(f"  [{dir_.name}] SKIP (missing artefacts)")
        return None

    orig = np.array(Image.open(orig_path).convert("RGB"))
    inpainted = np.array(Image.open(inpaint_path).convert("RGB"))
    # The "07_inpainted.png" we wrote earlier has a 32-px footer strip — strip it.
    if inpainted.shape[0] == orig.shape[0] + 32 and inpainted.shape[1] == orig.shape[1]:
        inpainted = inpainted[: orig.shape[0], : orig.shape[1]]

    # Re-run detect + OCR so that blocks/translations end up aligned in the same
    # order the original pipeline produced — naive prefix-slicing on ocr_samples
    # drops the wrong bubbles when the JP filter rejects mid-list entries.
    ctd = await detector.detect(orig)
    blocks = ctd["blocks"]
    text_lines = ctd.get("text_lines", [])
    if text_lines:
        ocr_texts = await ocr_service.recognize_blocks_with_lines(
            orig, blocks, text_lines, batch_size=settings.parseq_batch_size
        )
    else:
        crops = detector.crop_regions(orig, blocks)
        ocr_texts = await ocr_service.recognize_text_batch(crops)

    # Match the translate.py JP filter exactly.
    kept_blocks: list[dict] = []
    for b, t in zip(blocks, ocr_texts):
        if is_japanese_text(t, settings.japanese_filter_min_ratio,
                            settings.japanese_filter_katakana_max_length):
            kept_blocks.append(b)

    translations = parse_tagged_blocks(response_path.read_text(encoding="utf-8"),
                                       len(kept_blocks))

    final = compose_final(inpainted, kept_blocks, translations)
    Image.fromarray(final).save(dir_ / "11_final_composite.png")

    # Side-by-side so the user can visually verify the plate vs the rendered text.
    h, w = final.shape[:2]
    side = Image.new("RGB", (w * 2 + 20, h + 40), (24, 24, 24))
    side.paste(Image.fromarray(inpainted), (0, 30))
    side.paste(Image.fromarray(final), (w + 20, 30))
    sd = ImageDraw.Draw(side)
    sd.text((10, 6), "LaMa plate", fill=(200, 200, 200), font=load_font(16))
    sd.text((w + 30, 6), "+ translated text (Anton/Bangers, binary-fit, auto-contrast)",
            fill=(0, 255, 180), font=load_font(16))
    side.save(dir_ / "12_final_side_by_side.png")
    non_empty = len([t for t in translations if t])
    print(f"  [{dir_.name}] ✓ re-composed ({len(kept_blocks)} bubbles kept, "
          f"{non_empty} translated)")
    return {"dir": dir_.name, "blocks": len(kept_blocks),
            "non_empty_translations": non_empty}


async def main():
    e2e_root = REPO_ROOT / "thoughts" / "koharu-improvements" / "pipeline-e2e"
    dirs = [p for p in sorted(e2e_root.iterdir())
            if p.is_dir() and p.name not in ("features",)]
    if not dirs:
        print("no e2e dirs found; run visualize_e2e_pipeline.py first")
        return
    print(f"using font: {DEFAULT_FONT_PATH}")
    print(f"re-composing {len(dirs)} image dirs…")
    detector = create_detector()
    ocr = ParseqOCRService(model_path=settings.parseq_model_path)
    for d in dirs:
        try:
            await recompose_one(d, detector, ocr)
        except Exception as exc:
            print(f"  [{d.name}] FAILED: {exc}")


if __name__ == "__main__":
    asyncio.run(main())

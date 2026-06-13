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


# Wide-coverage fallback chain used when a bubble still contains a glyph
# missing from the primary display font. First entry that can actually
# render the string wins. Noto Sans CJK is listed first because any leaked
# Japanese/Chinese character requires its cmap; DejaVu covers the rest.
_FALLBACK_FONT_CANDIDATES = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
]
FALLBACK_FONT_PATHS: list[Path] = [p for p in _FALLBACK_FONT_CANDIDATES if p.exists()]
FALLBACK_FONT_PATH: Path | None = FALLBACK_FONT_PATHS[0] if FALLBACK_FONT_PATHS else None


def load_font(size: int, path: Path = DEFAULT_FONT_PATH) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


# ---------------------------------------------------------------------------
# Display-text normalization & font coverage
# ---------------------------------------------------------------------------

# Map characters our Latin display fonts don't support (Anton/Bangers/Oswald
# only cover ASCII + a sliver of Latin-1) back to ASCII equivalents so they
# render instead of showing as tofu squares (□). Keeps the same semantic
# punctuation the model emits — just in a font-compatible encoding.
_DISPLAY_REPLACE: dict[str, str] = {
    # ellipsis variants
    "…": "...",   # …
    "⋯": "...",   # ⋯
    # dashes
    "—": "-",     # em-dash
    "–": "-",     # en-dash
    "−": "-",     # minus
    "ー": "-",     # ー katakana-hiragana prolonged sound mark (safe ASCII approx)
    # curly/smart quotes
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "«": '"', "»": '"',
    # Japanese quotes + brackets
    "「": '"', "」": '"',
    "『": '"', "』": '"',
    "（": "(", "）": ")",
    # Japanese punctuation
    "。": ".", "、": ",",
    "．": ".", "，": ",",
    "？": "?", "！": "!",
    "：": ":", "；": ";",
    "・": ".",   # ・ middle dot
    "·": ".",   # ·
    # wave dashes
    "〜": "~", "～": "~",
    # fullwidth ASCII letters/digits → normal (rare in translations but safe)
    **{chr(0xFF01 + i): chr(0x21 + i) for i in range(94)},
    # non-breaking/zero-width spaces
    " ": " ", "​": "", "‌": "", "‍": "", "﻿": "",
}


def normalize_for_display(text: str) -> str:
    """Replace characters our Latin display fonts can't render with ASCII
    equivalents. Safe to call multiple times (idempotent)."""
    if not text:
        return text
    return "".join(_DISPLAY_REPLACE.get(ch, ch) for ch in text)


_CMAP_CACHE: dict[str, set[int]] = {}


def _cmap_for(path: Path) -> set[int]:
    """Return the set of Unicode codepoints the font at `path` has glyphs
    for. Cached; uses fontTools (which handles both .ttf and .ttc) and
    returns a full-BMP set on error so we never false-negatively drop
    supported glyphs."""
    key = str(path)
    if key in _CMAP_CACHE:
        return _CMAP_CACHE[key]
    try:
        from fontTools.ttLib import TTFont, TTCollection
        if path.suffix.lower() == ".ttc":
            # .ttc collections carry multiple fonts — union all subfonts'
            # cmaps so whichever face Pillow actually uses still passes.
            coll = TTCollection(str(path))
            codepoints: set[int] = set()
            for ttf in coll.fonts:
                try:
                    codepoints.update(ttf.getBestCmap().keys())
                except Exception:
                    continue
        else:
            ttf = TTFont(str(path))
            codepoints = set(ttf.getBestCmap().keys())
    except Exception:
        codepoints = set(range(0x10000))  # permissive on failure
    _CMAP_CACHE[key] = codepoints
    return codepoints


def _font_supports(path: Path, text: str) -> bool:
    """True if every non-whitespace codepoint in `text` is in the font's cmap."""
    cmap = _cmap_for(path)
    return all(ord(c) in cmap for c in text if not c.isspace())


def _pick_renderable_font(preferred: Path, text: str) -> Path:
    """Return `preferred` if it can render every char in `text`. Otherwise
    return the first fallback that can. If none can render everything,
    return the widest-coverage fallback we have (still better than the
    narrow display font). The caller is responsible for handling the
    mixed-style consequence — coverage beats tofu."""
    if _font_supports(preferred, text):
        return preferred
    for fb in FALLBACK_FONT_PATHS:
        if _font_supports(fb, text):
            return fb
    return FALLBACK_FONT_PATH or preferred


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
            # A word wider than the box: only hard-break LONG words (≥13 chars).
            # Short words (e.g. "MOMMY") should overflow the narrow box on one
            # line — a slight overhang reads far better than "MO/MM/Y".
            if font.getlength(w) > max_w and not cur and len(w) >= 13:
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
    fit_rects: list | None = None,
) -> np.ndarray:
    """Render translated text onto the inpainted plate using koharu's layout
    semantics: binary-search fit, real ink bbox, line_height =
    max(ascent + descent + leading, font_size), and a stroke width scaled
    with font size for readability against noisy backgrounds.

    ``fit_rects`` (optional, parallel to ``blocks``): the speech-BUBBLE rect
    each block lives in, as a dict with minX/minY/maxX/maxY, or None. When
    present, text is fit and centered to the bubble interior — the correct
    area for typesetting — instead of the tight (often tall-narrow, vertical-
    JP) text-block bbox. This is what makes narrow columns render as readable
    horizontal text without overlapping neighbors (bubbles are disjoint).
    When None (e.g. SFX over art with no bubble), it falls back to the block's
    own bbox and does NOT widen (blind widening overlaps neighbors).

    Auto-picks black-on-white vs white-on-black based on sampled luminance
    of the inpainted background inside the rect.
    """
    pil = Image.fromarray(inpainted).convert("RGB")
    draw = ImageDraw.Draw(pil)
    img_h, img_w = inpainted.shape[:2]
    fit_rects = fit_rects or [None] * len(blocks)
    for block, text, fit_rect in zip(blocks, translations, fit_rects):
        if not text:
            continue
        # Normalize to the ASCII subset our display fonts actually cover, then
        # UPPERCASE — English manga dialogue is conventionally all-caps (Wild
        # Words / Anime Ace are caps-first), and it reads as "official" rather
        # than machine output.
        text = normalize_for_display(text).strip().upper()
        if not text:
            continue
        # Fit/center to the bubble interior when we matched one; else the block.
        rect = fit_rect if fit_rect is not None else block
        x0, y0 = int(rect["minX"]), int(rect["minY"])
        x1, y1 = int(rect["maxX"]), int(rect["maxY"])
        bw, bh = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

        # Fixed-px inset preserves a consistent visual margin at any bubble size.
        inset_w = max(20, bw - inset_margin * 2)
        inset_h = max(12, bh - inset_margin * 2)

        # Tall-narrow blocks come from vertical Japanese text columns. Rendering
        # horizontal English into that skinny box forces one-word-per-line (or
        # mid-word splits like "MO/M/MY"). Widen the fit box toward a readable
        # aspect, centered on the block, clamped to the image — far better than
        # cramming. Height may then overflow the original column, which is fine.
        # Note: we deliberately do NOT widen narrow (vertical-JP-origin) blocks
        # beyond their detected box. Widening blind to bubble geometry makes
        # adjacent columns overlap into an unreadable jumble (CTD gives the
        # tight text region, not the bubble interior). Proper widening needs a
        # speech-bubble segmentation model — tracked as a follow-up. Here we
        # keep text inside its block and rely on all-caps + the min floor +
        # word-safe wrapping for legibility.
        eff_w, eff_h = inset_w, inset_h
        max_cap = 96

        # Pick the display font, then swap in the widest-coverage fallback
        # if it can't render every glyph (smart-quote, CJK leak, accented
        # letter, etc.) — prevents tofu squares in the final composite.
        font_path = _pick_renderable_font(pick_font(text), text)
        # Minimum legible floor. Below this, text is unreadable at reading
        # size — prefer a little overflow over microscopic text. Kept modest
        # (13px) so small bubbles don't get text that dwarfs them.
        min_floor = 13
        font, lines = find_best_fit(draw, text, eff_w, eff_h, font_path,
                                    min_size=min_floor, max_size=max_cap)
        mw, mh = measure_block(draw, lines, font)

        # Auto-contrast: flip to white text on dark plates.
        bg_lum = sample_bg_luminance(inpainted, x0, y0, x1, y1)
        if bg_lum < 128:
            fill, stroke = (255, 255, 255), (0, 0, 0)
        else:
            fill, stroke = (0, 0, 0), (255, 255, 255)

        line_h = line_height_px(font)
        # Center the rendered block on the original block's center, then clamp
        # so the whole block stays on-canvas (a tall column near a page edge
        # would otherwise render off the top/bottom).
        top = cy - mh // 2
        top = max(2, min(top, img_h - mh - 2))
        # Stroke ~10 % of font size, capped so it doesn't close counters on tiny text.
        stroke_w = max(2, min(5, round(font.size * 0.10)))
        for i, ln in enumerate(lines):
            bb = font.getbbox(ln)
            lw = bb[2] - bb[0]
            # ink-bbox correction so the line is centered on the block center,
            # then clamp horizontally to keep the line on-canvas.
            left = cx - lw // 2 - bb[0]
            left = max(2 - bb[0], min(left, img_w - lw - 2 - bb[0]))
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=REPO_ROOT / "thoughts" / "koharu-improvements" / "pipeline-e2e",
                    help="Gallery root produced by visualize_e2e_pipeline.py")
    ap.add_argument("--final-only", type=Path, default=None,
                    help="Mirror refreshed 11_final_composite.png files into "
                         "this flat folder as <slug>.png")
    args = ap.parse_args()

    e2e_root = args.root
    dirs = [p for p in sorted(e2e_root.iterdir())
            if p.is_dir() and p.name not in ("features", "originals")
            and (p / "stats.json").exists()]
    if not dirs:
        print(f"no e2e dirs with stats.json found under {e2e_root}")
        return
    print(f"using font: {DEFAULT_FONT_PATH}")
    print(f"fallback fonts: {[p.name for p in FALLBACK_FONT_PATHS]}")
    print(f"re-composing {len(dirs)} image dirs under {e2e_root}…")
    detector = create_detector()
    ocr = ParseqOCRService(model_path=settings.parseq_model_path)
    for d in dirs:
        try:
            await recompose_one(d, detector, ocr)
            if args.final_only:
                args.final_only.mkdir(parents=True, exist_ok=True)
                src = d / "11_final_composite.png"
                if src.exists():
                    (args.final_only / f"{d.name}.png").write_bytes(src.read_bytes())
        except Exception as exc:
            print(f"  [{d.name}] FAILED: {exc}")


if __name__ == "__main__":
    asyncio.run(main())

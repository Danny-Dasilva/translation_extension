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
# Readability floor + page-consistency policy (config-driven)
# ---------------------------------------------------------------------------
# These mirror app.config.Settings so the renderer has a single source of truth
# while still being importable as a stand-alone script (the bench/inspect data
# harness imports this module directly). We try to read the live settings; if
# the import fails (e.g. running the file in isolation) we fall back to the same
# defaults declared in app/config.py.
try:  # pragma: no cover - exercised indirectly
    from app.config import settings as _settings  # type: ignore

    ABS_FONT_FLOOR: int = int(_settings.render_font_abs_floor)
    FONT_FLOOR_FRAC: float = float(_settings.render_font_floor_frac)
    CLAMPED_HARD_FLOOR: int = int(_settings.render_font_clamped_hard_floor)
    FONT_MAX_CAP: int = int(_settings.render_font_max_cap)
    CONSISTENT_FONT_DEFAULT: bool = bool(_settings.render_consistent_font)
    CONSISTENT_FONT_PERCENTILE: int = int(_settings.render_consistent_font_percentile)
except Exception:  # pragma: no cover - fallback defaults
    ABS_FONT_FLOOR = 18
    FONT_FLOOR_FRAC = 0.012
    CLAMPED_HARD_FLOOR = 12
    FONT_MAX_CAP = 96
    CONSISTENT_FONT_DEFAULT = True
    CONSISTENT_FONT_PERCENTILE = 35


def resolution_font_floor(img_h: int) -> int:
    """Resolution-aware minimum legible font size in pixels.

    Manga pages here are ~1000-2000px tall, so an absolute floor alone is wrong:
    a 14px line that's readable on a 600px crop is microscopic on a 1791px page.
    We take ``max(ABS_FONT_FLOOR, round(img_h * FONT_FLOOR_FRAC))`` so the floor
    scales with page resolution while never dropping below the absolute minimum.
    """
    return max(ABS_FONT_FLOOR, round(img_h * FONT_FLOOR_FRAC))


def _percentile(values: list[int], pct: int) -> float:
    """Nearest-rank percentile (no numpy dependency on the hot path)."""
    if not values:
        return 0.0
    s = sorted(values)
    if pct <= 0:
        return float(s[0])
    if pct >= 100:
        return float(s[-1])
    # nearest-rank: rank = ceil(pct/100 * N), 1-indexed
    import math

    rank = max(1, math.ceil(pct / 100.0 * len(s)))
    return float(s[rank - 1])


def page_dialogue_target(maxfit_sizes: list[int], floor: int,
                         percentile: int = CONSISTENT_FONT_PERCENTILE) -> int:
    """Shared dialogue font-size target for one page.

    Given each dialogue bubble's *independent* max-fit size, drive the page
    toward ONE consistent size: a LOW percentile of those sizes (not the min,
    which would let a single cramped bubble shrink the whole page; not the max,
    which would overflow most bubbles). Clamped to be >= the readability floor.

    Most bubbles then render at this shared size; only genuinely tiny bubbles
    (whose max-fit is below the target) deviate downward — but never below the
    floor, because the renderer wraps / allows modest overflow instead.
    """
    if not maxfit_sizes:
        return floor
    target = round(_percentile(maxfit_sizes, percentile))
    return max(floor, int(target))


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
                  min_size: int = 6, max_size: int = 72) -> tuple[ImageFont.FreeTypeFont, list[str], bool]:
    """Binary-search the largest font size where the wrapped text fits inside
    (max_w, max_h). Falls back to min_size if nothing fits (accepts overflow).

    Mirrors /tmp/koharu/koharu-renderer/src/layout.rs:133-167 run_auto —
    same [6..300] range shape, same fallback-to-minimum behaviour.

    Returns ``(font, lines, fitted)`` where ``fitted`` is True only if some
    size satisfied both ``w <= max_w`` and ``h <= max_h``. When ``fitted`` is
    False the caller MUST clip/clamp the returned block — nothing actually fit
    and the min-size fallback still overflows.
    """
    lo, hi = min_size, max_size
    best_font = load_font(min_size, font_path)
    best_lines = wrap_greedy(draw, text, best_font, max_w) or [text]
    fitted = False
    while lo <= hi:
        mid = (lo + hi) // 2
        font = load_font(mid, font_path)
        lines = wrap_greedy(draw, text, font, max_w) or [text]
        w, h = measure_block(draw, lines, font)
        if w <= max_w and h <= max_h:
            best_font = font
            best_lines = lines
            fitted = True
            lo = mid + 1
        else:
            hi = mid - 1
    return best_font, best_lines, fitted


def block_max_fit_size(draw: ImageDraw.ImageDraw, text: str, max_w: int, max_h: int,
                       font_path: Path = DEFAULT_FONT_PATH,
                       min_size: int = 6, max_size: int = FONT_MAX_CAP) -> int:
    """Largest font size at which ``text`` fits ``(max_w, max_h)``.

    Used in the page-consistency first pass to learn each dialogue bubble's
    independent max-fit so we can pick a shared page target. Returns ``min_size``
    when nothing fits (caller treats that as "this bubble is too cramped").
    """
    if not text:
        return min_size
    font, _lines, fitted = find_best_fit(draw, text, max_w, max_h, font_path,
                                         min_size=min_size, max_size=max_size)
    return font.size if fitted else min_size


def layout_at_size(draw: ImageDraw.ImageDraw, text: str, size: int, max_w: int,
                   font_path: Path = DEFAULT_FONT_PATH) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Wrap ``text`` at a FIXED font ``size`` to ``max_w`` (no shrinking).

    The page-consistency renderer picks a shared target size, then lays every
    dialogue bubble out at that size — wrapping to as many lines as needed. The
    caller decides whether the resulting block fits the bubble height, and if
    not, applies the priority ladder (more lines already done here → modest
    overflow → clip).
    """
    font = load_font(size, font_path)
    lines = wrap_greedy(draw, text, font, max_w) or [text]
    return font, lines


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


# ---------------------------------------------------------------------------
# FIX A helpers — keep caption / orphan / SFX (no-bubble) blocks inside their
# own bbox and avoid overlapping already-placed blocks.
# ---------------------------------------------------------------------------

# Verbose model glosses for stylized SFX ("SFX FOR A MOMENTARY SHOCK…",
# "FLOPPING, BOUNCING") overflow the tiny scrawl boxes they belong to. When a
# block is orphan/SFX-sized we keep only a short onomatopoeia-length token so it
# reads as a sound effect instead of spilling a sentence over neighbouring art.
_SFX_MAX_CHARS = 16


def _is_sfx_sized(block: dict) -> bool:
    """True for small, orphan/SFX-style boxes (the verbose-gloss offenders).

    A box qualifies when it is flagged ``orphan`` OR is physically small
    (short side ≤ ~48 px or area ≤ ~9 kpx) — the regime where a multi-word
    English sentence cannot fit without overflowing.
    """
    w = int(block.get("maxX", 0)) - int(block.get("minX", 0))
    h = int(block.get("maxY", 0)) - int(block.get("minY", 0))
    short = min(abs(w), abs(h))
    area = abs(w) * abs(h)
    return bool(block.get("orphan")) or short <= 48 or area <= 9000


def _truncate_sfx_text(text: str, block: dict, max_chars: int = _SFX_MAX_CHARS) -> str:
    """Shorten a verbose SFX gloss to onomatopoeia length for a small box.

    Leaves already-short strings untouched. For long glosses, keeps the first
    word(s) up to ``max_chars`` (the actual onomatopoeia usually leads, e.g.
    "FLOPPING, BOUNCING" -> "FLOPPING"), falling back to a hard char cut.
    Only applies to SFX-sized boxes; larger caption boxes are returned as-is.
    """
    if not text:
        return text
    s = text.strip()
    if len(s) <= max_chars or not _is_sfx_sized(block):
        return s
    # Prefer a clean word boundary within the budget.
    words = s.replace(",", " ").split()
    out = ""
    for w in words:
        trial = (out + " " + w).strip()
        if len(trial) > max_chars:
            break
        out = trial
    if not out:  # single very long word
        out = s[:max_chars]
    return out


def _rects_overlap(a: tuple, b: tuple, pad: int = 0) -> bool:
    """Axis-aligned overlap test for (x0, y0, x1, y1) ink rects."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (
        ax0 < bx1 + pad and bx0 < ax1 + pad
        and ay0 < by1 + pad and by0 < ay1 + pad
    )


def sample_bg_luminance(
    image: np.ndarray, x0: int, y0: int, x1: int, y1: int
) -> tuple[float, float]:
    """Background luminance stats where we're about to draw, as
    ``(median_luma, dark_fraction)`` over 0-255 BT.601 luminance.

    Koharu picks text/stroke color from a prediction model; we fall back
    to luminance-based auto-contrast (KOHARU_COMPARISON.md item #9).

    FIX #5: the old MEAN over the full rect failed on wet/dark art — a
    mostly-dark crop with a few bright specks averaged "light", so we'd
    draw black text on a dark background. We now return the MEDIAN (robust
    to outliers) plus ``dark_fraction`` (share of pixels with luma<96) so
    the caller can flip to white text whenever a meaningful chunk is dark.
    """
    h, w = image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 255.0, 0.0
    crop = image[y0:y1, x0:x1].astype(np.float32)
    if crop.size == 0:
        return 255.0, 0.0
    # BT.601 luminance
    lum = 0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]
    median_luma = float(np.median(lum))
    dark_fraction = float(np.mean(lum < 96))
    return median_luma, dark_fraction


def compose_final(
    inpainted: np.ndarray,
    blocks: list[dict],
    translations: list[str],
    inset_margin: int = 4,
    fit_rects: list | None = None,
    consistent_font: bool | None = None,
    overflow_frac: float = 0.18,
    _debug_sizes: bool = False,
):
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

    READABILITY FLOOR (resolution-aware): the minimum rendered font size is
    ``resolution_font_floor(img_h)`` = ``max(ABS_FONT_FLOOR, img_h*FONT_FLOOR_FRAC)``
    so it scales with page resolution. When text does not fit a dialogue bubble
    at this floor we do NOT shrink below it — instead, in priority order:
    (a) wrap to more lines, (b) allow modest overflow/expansion beyond the bubble
    interior (bounded by ``overflow_frac`` of the bubble size — the inpaint plate
    is clean so a slight spill reads fine), (c) clip trailing lines as a last
    resort. Clamped (no-bubble) SFX/caption blocks keep a lower hard floor and
    are clipped rather than overflowing onto neighbouring art.

    PAGE-LEVEL CONSISTENCY (``consistent_font``, default from config): instead of
    sizing every dialogue bubble fully independently (which made adjacent bubbles
    wildly different sizes), a first pass computes each dialogue bubble's max-fit
    size and a shared page target (a low percentile of those sizes, never below
    the floor). The render pass drives every dialogue bubble toward that one
    target so most bubbles share a single readable size; only genuinely tiny
    bubbles deviate. Clamped SFX/caption blocks stay on their own independent
    track (they are legitimately variable). Set ``consistent_font=False`` to A/B.

    When ``_debug_sizes`` is True, returns ``list[int|None]`` (parallel to
    ``blocks``) of the rendered font size per block (None for suppressed/empty
    blocks) instead of the composited image — used by the tests/AB harness.
    """
    pil = Image.fromarray(inpainted).convert("RGB")
    draw = ImageDraw.Draw(pil)
    img_h, img_w = inpainted.shape[:2]
    fit_rects = fit_rects or [None] * len(blocks)
    if consistent_font is None:
        consistent_font = CONSISTENT_FONT_DEFAULT

    # Resolution-aware readability floor for this page.
    min_floor = resolution_font_floor(img_h)
    # Clamped (no-bubble) blocks get a lower hard floor — they are legitimately
    # variable and must not overflow onto art, so we let them clip smaller.
    clamped_hard_floor = min(CLAMPED_HARD_FLOOR, min_floor)
    max_cap = FONT_MAX_CAP

    # Per-block rendered font size (for _debug_sizes / AB harness).
    rendered_size: list[int | None] = [None] * len(blocks)

    # FIX A: track ink rects already placed so a later block can be shrunk /
    # suppressed instead of overlapping an earlier one. Each entry is
    # (x0, y0, x1, y1, is_dialogue) where is_dialogue marks bubble-matched
    # dialogue (an orphan/SFX block must never overlap it).
    placed_rects: list[tuple[int, int, int, int, bool]] = []

    # FIX A.2: placement order matters for overlap avoidance. Place DIALOGUE
    # (bubble-matched) first so it is never covered, then clamped caption/SFX
    # blocks SMALLEST-area-first so the small, specific narration columns claim
    # their space and an over-large MERGED caption box (whose OCR concatenated
    # several columns) shrinks/clips around them instead of overlapping them.
    order = sorted(
        range(len(blocks)),
        key=lambda i: (
            0 if fit_rects[i] is not None else 1,  # dialogue first
            (int(blocks[i]["maxX"]) - int(blocks[i]["minX"]))
            * (int(blocks[i]["maxY"]) - int(blocks[i]["minY"])),  # then small-first
        ),
    )

    def _dialogue_fit_box(rect: dict) -> tuple[int, int]:
        """Effective (width, height) the text is fit to inside a dialogue
        bubble — bubble interior minus the fixed-px inset. Mirrors the render
        loop's inset math so the first-pass max-fit matches what we render."""
        bw = int(rect["maxX"]) - int(rect["minX"])
        bh = int(rect["maxY"]) - int(rect["minY"])
        return max(20, bw - inset_margin * 2), max(12, bh - inset_margin * 2)

    # PAGE-LEVEL CONSISTENCY first pass: learn each dialogue bubble's independent
    # max-fit size, then derive ONE shared target (low percentile, >= floor).
    page_target: int | None = None
    if consistent_font:
        maxfits: list[int] = []
        for i, fr in enumerate(fit_rects):
            if fr is None:
                continue
            t = translations[i]
            if not t:
                continue
            t = normalize_for_display(t).strip().upper()
            if not t:
                continue
            ew, eh = _dialogue_fit_box(fr)
            fp = _pick_renderable_font(pick_font(t), t)
            maxfits.append(block_max_fit_size(draw, t, ew, eh, fp,
                                              min_size=min_floor, max_size=max_cap))
        if maxfits:
            page_target = page_dialogue_target(
                maxfits, min_floor, percentile=CONSISTENT_FONT_PERCENTILE)

    for _idx in order:
        block = blocks[_idx]
        text = translations[_idx]
        fit_rect = fit_rects[_idx]
        if not text:
            continue
        is_dialogue = fit_rect is not None
        is_clamped = fit_rect is None  # caption / orphan / SFX over art
        # FIX A.3: cap verbose SFX glosses in small/orphan boxes to onomatopoeia
        # length so they don't overflow their tiny bbox onto neighbours.
        if is_clamped:
            text = _truncate_sfx_text(text, block)
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

        # FIX A.2: for a clamped block, carve already-placed rects out of its
        # working bbox so an over-large MERGED caption shrinks around the small
        # narration columns already placed inside it (instead of overlapping
        # them). We trim the bbox on whichever single axis recovers the most
        # free space, keeping the largest remaining sub-rectangle.
        if is_clamped and placed_rects:
            for pr in placed_rects:
                px0, py0, px1, py1, _ = pr
                # only consider rects that meaningfully intrude into this bbox
                ix0, iy0 = max(x0, px0), max(y0, py0)
                ix1, iy1 = min(x1, px1), min(y1, py1)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                # candidate trims: keep the side of our bbox with more room.
                left_room = ix0 - x0      # keep [x0, ix0]
                right_room = x1 - ix1     # keep [ix1, x1]
                top_room = iy0 - y0       # keep [y0, iy0]
                bot_room = y1 - iy1       # keep [iy1, y1]
                best = max(left_room, right_room, top_room, bot_room)
                # require a usable remaining strip; else leave bbox (will clip).
                if best < 24:
                    continue
                if best == right_room:
                    x0 = ix1
                elif best == left_room:
                    x1 = ix0
                elif best == bot_room:
                    y0 = iy1
                else:
                    y1 = iy0
            if x1 - x0 < 8:
                x1 = x0 + 8
            if y1 - y0 < 8:
                y1 = y0 + 8

        bw, bh = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

        # Fixed-px inset preserves a consistent visual margin at any bubble size.
        inset_w = max(20, bw - inset_margin * 2)
        inset_h = max(12, bh - inset_margin * 2)
        # FIX A.1: for clamped (no-bubble) blocks the fit box must be the BLOCK
        # bbox, never widened — text is shrunk-to-fit to the bbox width and
        # wrapped, and may be clipped at the bbox edge below rather than spill.
        if is_clamped:
            inset_w = max(8, bw - inset_margin * 2)
            inset_h = max(8, bh - inset_margin * 2)

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

        # Pick the display font, then swap in the widest-coverage fallback
        # if it can't render every glyph (smart-quote, CJK leak, accented
        # letter, etc.) — prevents tofu squares in the final composite.
        font_path = _pick_renderable_font(pick_font(text), text)

        # SIZE SELECTION
        # ----------------------------------------------------------------
        # Resolution-aware readability floor (computed once for the page).
        # Clamped (no-bubble) SFX/caption blocks get the lower hard floor so
        # they can clip smaller rather than overflow onto art; dialogue uses
        # the full readability floor and is allowed modest overflow instead.
        if is_clamped:
            # Clamped track: independent shrink-to-fit, never below the hard
            # floor, then clip. (Behaviour preserved — only the floor value is
            # resolution-aware now.)
            font, lines, fitted = find_best_fit(
                draw, text, eff_w, eff_h, font_path,
                min_size=clamped_hard_floor, max_size=max_cap)
            mw, mh = measure_block(draw, lines, font)
        else:
            # Dialogue track. With page consistency ON we drive toward the
            # shared page target; otherwise we use the bubble's own max-fit.
            if consistent_font and page_target is not None:
                desired = page_target
                # never exceed what actually fits this bubble (avoid overflow on
                # roomy bubbles that could go bigger but should match the page),
                # but never drop below the floor either.
                own_max = block_max_fit_size(draw, text, eff_w, eff_h, font_path,
                                             min_size=min_floor, max_size=max_cap)
                size = max(min_floor, min(desired, max(own_max, min_floor)))
            else:
                font_fit, _l, _ok = find_best_fit(
                    draw, text, eff_w, eff_h, font_path,
                    min_size=min_floor, max_size=max_cap)
                size = max(min_floor, font_fit.size)

            # Lay out at the chosen size (priority (a): wrap to more lines).
            font, lines = layout_at_size(draw, text, size, eff_w, font_path)
            mw, mh = measure_block(draw, lines, font)
            fitted = (mw <= eff_w and mh <= eff_h)

            # Priority (b): the chosen size is at/above the floor but the text
            # is taller than the bubble interior — allow MODEST overflow beyond
            # the bubble (the inpaint plate is clean, a slight spill reads fine)
            # rather than shrinking below the floor. We widen the clamp box by a
            # bounded fraction of the bubble so the trailing-line clip below only
            # fires after we've granted that slack. Width can also widen slightly
            # to absorb a hard-to-wrap long word.
            if mh > eff_h or mw > eff_w:
                pad_x = int(round(bw * overflow_frac))
                pad_y = int(round(bh * overflow_frac))
                x0 -= pad_x // 2
                x1 += pad_x // 2
                y0 -= pad_y // 2
                y1 += pad_y // 2
                # re-wrap to the (slightly) wider interior so width overflow is
                # also absorbed where possible.
                eff_w = max(20, (x1 - x0) - inset_margin * 2)
                font, lines = layout_at_size(draw, text, size, eff_w, font_path)
                mw, mh = measure_block(draw, lines, font)

        line_h = line_height_px(font)
        rendered_size[_idx] = int(font.size)
        # Center the rendered block on the original block's center.
        top = cy - mh // 2
        left_anchor = cx

        # FIX #4: a SINGLE clamp box bounds every block so DIALOGUE text can no
        # longer overflow its bubble. For dialogue the box is the bubble interior
        # (rect == fit_rect, i.e. x0/y0/x1/y1 already point at it). For clamped
        # caption/SFX blocks it is the (possibly carved) block bbox. Both axes
        # are clamped to this box; only when the box is degenerate/larger than
        # the canvas do we fall back to canvas extents (keeps tall edge columns
        # on-screen). Previously dialogue clamped only to the canvas, so long
        # translations spilled past the bubble edges.
        clamp_x0 = max(0, min(x0, img_w))
        clamp_y0 = max(0, min(y0, img_h))
        clamp_x1 = max(clamp_x0 + 1, min(x1, img_w))
        clamp_y1 = max(clamp_y0 + 1, min(y1, img_h))

        # Vertical: max lines from clamp-box height; clip trailing lines + "..."
        # for BOTH branches (dialogue previously never clipped vertically).
        max_lines = max(1, (clamp_y1 - clamp_y0) // line_h) if line_h > 0 else 1
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            last = lines[-1].rstrip(".")
            lines[-1] = (last + "...") if last else "..."
            mw, mh = measure_block(draw, lines, font)
        # Keep the rendered block vertically inside the clamp box. If it is
        # taller than the box (single very tall line), pin to the top edge.
        top = max(clamp_y0, min(top, clamp_y1 - mh))
        top = max(clamp_y0, top)

        # Predict the rendered ink rect (clamped to the box) for collision tests.
        rect_x0 = max(clamp_x0, left_anchor - mw // 2)
        rect_x1 = min(clamp_x1, rect_x0 + mw)
        rendered_rect = (int(rect_x0), int(top), int(rect_x1), int(top + mh))

        # FIX A.2 / FIX #4: inter-block overlap avoidance.
        #   Clamped block:
        #     * orphan/SFX over a DIALOGUE block -> always SUPPRESS (never cover
        #       dialogue);
        #     * an SFX-sized block still overlapping ANY placed block -> SUPPRESS;
        #     * a larger caption still overlapping -> keep (shrunk/clipped).
        #   Dialogue block:
        #     * skip ONLY when it is clearly (>=60% of its own area) buried under
        #       an already-placed rect — bubbles are normally disjoint, so this
        #       fires only on detection duplicates / heavy overlap. Light overlap
        #       is kept (suppressing real dialogue is worse than a small touch).
        if is_clamped:
            suppress = False
            for pr in placed_rects:
                if not _rects_overlap(rendered_rect, pr[:4]):
                    continue
                if pr[4]:  # overlapping DIALOGUE
                    suppress = True
                    break
                if _is_sfx_sized(block):  # stray SFX over another caption
                    suppress = True
                    break
            if suppress:
                rendered_size[_idx] = None
                continue
        else:
            rr_area = max(1, (rendered_rect[2] - rendered_rect[0])
                          * (rendered_rect[3] - rendered_rect[1]))
            suppress = False
            for pr in placed_rects:
                ix0, iy0 = max(rendered_rect[0], pr[0]), max(rendered_rect[1], pr[1])
                ix1, iy1 = min(rendered_rect[2], pr[2]), min(rendered_rect[3], pr[3])
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                inter = (ix1 - ix0) * (iy1 - iy0)
                if inter / rr_area >= 0.60:  # clearly buried
                    suppress = True
                    break
            if suppress:
                rendered_size[_idx] = None
                continue

        # Auto-contrast: flip to white text on dark plates.
        # FIX #5: use median luma + dark_fraction so wet/dark art (a dark
        # background with bright specks) still gets WHITE text. We go white
        # when the median is dark OR a meaningful share of pixels is dark.
        bg_median, dark_fraction = sample_bg_luminance(inpainted, x0, y0, x1, y1)
        if bg_median < 140 or dark_fraction > 0.35:
            fill, stroke = (255, 255, 255), (0, 0, 0)
        else:
            fill, stroke = (0, 0, 0), (255, 255, 255)

        # FIX #5: heavier stroke (~14 % of font size, 3..8px) so the outline
        # reads against noisy backgrounds; was ~10 % / 2..5px and too thin.
        stroke_w = max(3, min(8, round(font.size * 0.14)))
        for i, ln in enumerate(lines):
            bb = font.getbbox(ln)
            lw = bb[2] - bb[0]
            # ink-bbox correction so the line is centered on the block center.
            left = cx - lw // 2 - bb[0]
            # FIX #4: clamp the line strictly to the clamp box for BOTH branches
            # so it never spills past the bubble interior / block bbox. The bb[0]
            # offset is preserved so the visible ink lands inside [clamp_x0,
            # clamp_x1]. min<max guard handles a line wider than the box (pins
            # to the left edge instead of inverting the clamp).
            lo_left = clamp_x0 - bb[0]
            hi_left = clamp_x1 - lw - bb[0]
            if hi_left < lo_left:
                hi_left = lo_left
            left = max(lo_left, min(left, hi_left))
            y = top + i * line_h
            draw_stroked_text(draw, (left, y), ln, font,
                              fill=fill, stroke=stroke, stroke_width=stroke_w)
        # FIX #4: record DIALOGUE rendered rects too (was clamped-only) so later
        # blocks see dialogue ink and can avoid burying it.
        placed_rects.append((*rendered_rect, is_dialogue))
    if _debug_sizes:
        return rendered_size
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

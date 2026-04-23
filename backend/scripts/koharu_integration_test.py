"""End-to-end visual validation harness for the Koharu translation pipeline.

Purpose
-------
Run the existing FastAPI `/translate` endpoint against a small gallery of
representative manga pages and emit annotated PNGs so a human can visually
inspect detection, OCR, and translation quality.

This script does **not** modify any service / router code. It simply:

1. Loads the FastAPI app via `TestClient` (no network port needed).
2. For each test image:
   - Base64-encodes and POSTs it to `/translate`.
   - Writes the raw JSON response alongside the image.
   - Renders four overlays with PIL:
       * detection boxes only (cyan)
       * OCR text rendered at each box corner
       * Translation text centered in each box (white + black stroke)
       * Combined overlay (everything on one image)
   - Notes wall-clock time per image.
3. Writes a Markdown gallery (`SUMMARY.md`) with links to every artefact.

If the translation model is unavailable or the request fails, the script still
emits detect+OCR-only overlays so the user can validate the earlier stages.

Run (from repo root):
    cd backend && uv run python scripts/koharu_integration_test.py
"""

from __future__ import annotations

import base64
import json
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../extension
BACKEND_DIR = REPO_ROOT / "backend"
GALLERY_DIR = REPO_ROOT / "thoughts" / "koharu-improvements" / "test-gallery"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Test image selection
# ---------------------------------------------------------------------------
@dataclass
class TestImage:
    """A test image + metadata."""

    label: str  # short slug used for filenames
    source: Path
    description: str

    @property
    def exists(self) -> bool:
        return self.source.exists()


def _candidate_images() -> list[TestImage]:
    """Pick 3-5 representative manga pages.

    Falls back to duplicating de.png if nothing else is found.
    """
    candidates: list[TestImage] = []

    # Primary: de.png at repo root.
    de_png = REPO_ROOT / "de.png"
    if de_png.exists():
        candidates.append(
            TestImage("de_png", de_png, "Primary reference image from repo root.")
        )

    # Known-good sample from training dataset.
    aisazu = (
        REPO_ROOT
        / "training"
        / "comic-text-detector"
        / "data"
        / "examples"
        / "AisazuNihaIrarenai-003.jpg"
    )
    if aisazu.exists():
        candidates.append(
            TestImage(
                "aisazu_003",
                aisazu,
                "AisazuNihaIrarenai-003 — dense Japanese speech bubbles.",
            )
        )

    # Another example used for block / detection test pages.
    block_test = (
        REPO_ROOT
        / "training"
        / "comic-text-detector"
        / "data"
        / "examples"
        / "block_detection_test.jpg"
    )
    if block_test.exists():
        candidates.append(
            TestImage(
                "block_detection_test",
                block_test,
                "Block detection test page — varied bubble shapes.",
            )
        )

    # A validation image from the labelled CTD val split.
    val_dir = (
        REPO_ROOT
        / "training"
        / "comic-text-detector"
        / "data"
        / "val"
        / "images"
    )
    if val_dir.exists():
        for path in sorted(val_dir.glob("*.jpg"))[:1]:
            candidates.append(
                TestImage(
                    f"ctd_val_{path.stem[:8]}",
                    path,
                    f"CTD validation sample ({path.name}).",
                )
            )

    # Last-resort: recent debug input captured live from the extension.
    debug_dir = BACKEND_DIR / "debug_output"
    if debug_dir.exists() and len(candidates) < 4:
        input_shots = sorted(debug_dir.glob("input_image_*.jpg"))
        if input_shots:
            path = input_shots[-1]
            candidates.append(
                TestImage(
                    "debug_input_latest",
                    path,
                    f"Most recent live capture from extension ({path.name}).",
                )
            )

    # Guarantee >= 3 images by padding with de.png if needed.
    if de_png.exists():
        while len(candidates) < 3:
            idx = len(candidates) + 1
            candidates.append(
                TestImage(
                    f"de_png_pad_{idx}",
                    de_png,
                    f"Duplicate of de.png (padding slot {idx}) — too few distinct images.",
                )
            )

    return candidates[:5]


# ---------------------------------------------------------------------------
# Rendering helpers (minimal port of overlay-renderer.ts)
# ---------------------------------------------------------------------------
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try a few common font paths; fall back to the default bitmap font."""
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in font_candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_text_with_stroke(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int] = (255, 255, 255),
    stroke: tuple[int, int, int] = (0, 0, 0),
    stroke_width: int = 2,
    anchor: str | None = None,
) -> None:
    """Draw text with a readable black outline (manga-style)."""
    try:
        draw.text(
            xy,
            text,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke,
            anchor=anchor,
        )
    except TypeError:
        # Pillow < 8 fallback without stroke_width.
        for dx in (-stroke_width, 0, stroke_width):
            for dy in (-stroke_width, 0, stroke_width):
                if dx == 0 and dy == 0:
                    continue
                draw.text((xy[0] + dx, xy[1] + dy), text, font=font, fill=stroke)
        draw.text(xy, text, font=font, fill=fill)


def _wrap_text_to_width(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> list[str]:
    """Greedy word-wrap so translation fits inside a bbox."""
    if not text:
        return []
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = words[0]
    for w in words[1:]:
        trial = f"{current} {w}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def render_overlays(
    image_path: Path,
    text_boxes: list[dict[str, Any]],
    out_dir: Path,
    slug: str,
) -> dict[str, Path]:
    """Emit detection / OCR / translation / combined overlays.

    Returns a dict mapping overlay name -> path.
    """
    base = Image.open(image_path).convert("RGB")
    outputs: dict[str, Path] = {}

    # --- detection boxes only ---
    det_img = base.copy()
    det_draw = ImageDraw.Draw(det_img)
    label_font = _load_font(14)
    for i, box in enumerate(text_boxes):
        x0, y0, x1, y1 = box["minX"], box["minY"], box["maxX"], box["maxY"]
        det_draw.rectangle([x0, y0, x1, y1], outline=(0, 220, 220), width=3)
        det_draw.rectangle([x0, max(0, y0 - 16), x0 + 28, y0], fill=(0, 220, 220))
        det_draw.text((x0 + 2, max(0, y0 - 16)), f"#{i}", font=label_font, fill=(0, 0, 0))
    det_path = out_dir / f"{slug}_detect.png"
    det_img.save(det_path)
    outputs["detect"] = det_path

    # --- OCR text rendered ---
    ocr_img = base.copy()
    ocr_draw = ImageDraw.Draw(ocr_img)
    ocr_font = _load_font(14)
    for i, box in enumerate(text_boxes):
        x0, y0, x1, y1 = box["minX"], box["minY"], box["maxX"], box["maxY"]
        ocr_draw.rectangle([x0, y0, x1, y1], outline=(0, 220, 220), width=2)
        ocr_text = box.get("ocrText", "") or ""
        label = f"#{i}: {ocr_text}"[:80]
        _draw_text_with_stroke(
            ocr_draw,
            (x0 + 3, y0 + 3),
            label,
            ocr_font,
            fill=(255, 255, 0),
            stroke=(0, 0, 0),
            stroke_width=2,
        )
    ocr_path = out_dir / f"{slug}_ocr.png"
    ocr_img.save(ocr_path)
    outputs["ocr"] = ocr_path

    # --- translation rendered ---
    tr_img = base.copy()
    tr_draw = ImageDraw.Draw(tr_img)
    for box in text_boxes:
        x0, y0, x1, y1 = box["minX"], box["minY"], box["maxX"], box["maxY"]
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        # Dim the bubble interior so overlaid text is readable even without inpainting.
        overlay = Image.new("RGBA", (width, height), (255, 255, 255, 140))
        tr_img.paste(overlay, (x0, y0), overlay)

        translated = box.get("translatedText", "") or ""
        if not translated:
            continue

        # Pick a font size that fits.
        font_size = max(10, min(28, height // 5 or 14))
        font = _load_font(font_size)
        lines = _wrap_text_to_width(translated, font, max_width=width - 6, draw=tr_draw)
        if not lines:
            continue
        line_heights = []
        for line in lines:
            bbox = tr_draw.textbbox((0, 0), line, font=font)
            line_heights.append(bbox[3] - bbox[1])
        total_h = sum(line_heights) + (len(lines) - 1) * 2
        cy = y0 + (height - total_h) // 2
        for line, lh in zip(lines, line_heights):
            bbox = tr_draw.textbbox((0, 0), line, font=font)
            lw = bbox[2] - bbox[0]
            cx = x0 + (width - lw) // 2
            _draw_text_with_stroke(
                tr_draw,
                (cx, cy),
                line,
                font,
                fill=(255, 255, 255),
                stroke=(0, 0, 0),
                stroke_width=2,
            )
            cy += lh + 2
    tr_path = out_dir / f"{slug}_translated.png"
    tr_img.save(tr_path)
    outputs["translated"] = tr_path

    # --- combined overlay (box + OCR corner + translation) ---
    combo_img = base.copy()
    combo_draw = ImageDraw.Draw(combo_img)
    for i, box in enumerate(text_boxes):
        x0, y0, x1, y1 = box["minX"], box["minY"], box["maxX"], box["maxY"]
        combo_draw.rectangle([x0, y0, x1, y1], outline=(0, 220, 220), width=2)
        ocr_text = (box.get("ocrText") or "")[:40]
        _draw_text_with_stroke(
            combo_draw,
            (x0 + 2, max(0, y0 - 16)),
            f"#{i} {ocr_text}",
            _load_font(12),
            fill=(255, 220, 0),
            stroke=(0, 0, 0),
            stroke_width=2,
        )
        translated = box.get("translatedText", "") or ""
        if translated:
            width = max(1, x1 - x0)
            height = max(1, y1 - y0)
            font_size = max(10, min(22, height // 5 or 14))
            font = _load_font(font_size)
            lines = _wrap_text_to_width(translated, font, max_width=width - 4, draw=combo_draw)
            if lines:
                line_heights = [
                    combo_draw.textbbox((0, 0), line, font=font)[3]
                    - combo_draw.textbbox((0, 0), line, font=font)[1]
                    for line in lines
                ]
                total_h = sum(line_heights) + (len(lines) - 1) * 2
                cy = y0 + (height - total_h) // 2
                for line, lh in zip(lines, line_heights):
                    bbox = combo_draw.textbbox((0, 0), line, font=font)
                    lw = bbox[2] - bbox[0]
                    cx = x0 + (width - lw) // 2
                    _draw_text_with_stroke(
                        combo_draw,
                        (cx, cy),
                        line,
                        font,
                        fill=(255, 255, 255),
                        stroke=(0, 0, 0),
                        stroke_width=2,
                    )
                    cy += lh + 2
    combo_path = out_dir / f"{slug}_combined.png"
    combo_img.save(combo_path)
    outputs["combined"] = combo_path

    return outputs


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    test: TestImage
    original_path: Path
    slug: str
    status: str  # "ok" | "partial" | "failed"
    wall_ms: float
    box_count: int
    overlays: dict[str, Path] = field(default_factory=dict)
    response_json_path: Path | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def _encode_image_data_url(path: Path) -> str:
    data = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix or "png"
    return f"data:image/{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _copy_original(test: TestImage, out_dir: Path) -> Path:
    """Copy the original image into the gallery so SUMMARY links resolve locally."""
    dst = out_dir / f"{test.label}_original{test.source.suffix.lower()}"
    shutil.copy2(test.source, dst)
    return dst


def _run_one(client: Any, test: TestImage, out_dir: Path) -> RunResult:
    slug = test.label
    orig_copy = _copy_original(test, out_dir)

    data_url = _encode_image_data_url(test.source)
    payload = {"base64Images": [data_url], "targetLanguage": "English"}

    notes: list[str] = []
    start = time.perf_counter()
    try:
        response = client.post("/translate", json=payload, timeout=600)
    except Exception as exc:  # pragma: no cover - network-less TestClient
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RunResult(
            test=test,
            original_path=orig_copy,
            slug=slug,
            status="failed",
            wall_ms=elapsed_ms,
            box_count=0,
            error=f"TestClient raised: {exc}",
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    if response.status_code != 200:
        notes.append(f"HTTP {response.status_code}: {response.text[:400]}")
        return RunResult(
            test=test,
            original_path=orig_copy,
            slug=slug,
            status="failed",
            wall_ms=elapsed_ms,
            box_count=0,
            error=notes[-1],
            notes=notes,
        )

    body = response.json()
    # Persist raw JSON (strip out background data-URLs which are huge).
    sanitized = json.loads(json.dumps(body))
    for img_boxes in sanitized.get("images", []):
        for tb in img_boxes:
            if isinstance(tb.get("background"), str) and tb["background"].startswith("data:"):
                tb["background"] = f"<data-url stripped, {len(tb['background'])} chars>"
    response_path = out_dir / f"{slug}_response.json"
    response_path.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False))

    images = body.get("images", [[]])
    text_boxes = images[0] if images else []
    has_translation = any((tb.get("translatedText") or "").strip() for tb in text_boxes)
    status = "ok" if has_translation else "partial"
    if not has_translation:
        notes.append(
            "No translations present — backend likely ran detect+OCR only "
            "(translation model unavailable or skipped)."
        )

    overlays = render_overlays(test.source, text_boxes, out_dir, slug)

    return RunResult(
        test=test,
        original_path=orig_copy,
        slug=slug,
        status=status,
        wall_ms=elapsed_ms,
        box_count=len(text_boxes),
        overlays=overlays,
        response_json_path=response_path,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------
def _write_timings(results: list[RunResult], out_dir: Path) -> Path:
    timings_path = out_dir / "baseline_timings.txt"
    lines = ["# Koharu baseline wall-clock timings (ms)", ""]
    for r in results:
        lines.append(f"{r.slug:30s}  {r.wall_ms:8.1f} ms  status={r.status}  boxes={r.box_count}")
    timings_path.write_text("\n".join(lines) + "\n")
    return timings_path


def _write_summary(results: list[RunResult], out_dir: Path, pipeline_error: str | None) -> Path:
    summary_path = out_dir / "SUMMARY.md"
    md: list[str] = [
        "# Koharu Pipeline — Visual Validation Gallery",
        "",
        "Auto-generated by `backend/scripts/koharu_integration_test.py`.",
        "",
        "Each entry below links to PNGs you can open directly in your viewer.",
        "",
    ]

    if pipeline_error:
        md += [
            "## Pipeline startup error",
            "",
            "The FastAPI app failed to initialise, so no per-image results were",
            "generated. See the error below.",
            "",
            "```",
            pipeline_error.strip(),
            "```",
            "",
        ]

    for r in results:
        md.append(f"## {r.slug}")
        md.append("")
        md.append(f"- Source: `{r.test.source}`")
        md.append(f"- Description: {r.test.description}")
        md.append(f"- Status: **{r.status}**")
        md.append(f"- Wall-clock: {r.wall_ms:.0f} ms")
        md.append(f"- Detected boxes: {r.box_count}")
        if r.error:
            md.append(f"- Error: `{r.error}`")
        if r.notes:
            for n in r.notes:
                md.append(f"- Note: {n}")
        md.append("")

        def _rel(path: Path | None) -> str | None:
            if path is None:
                return None
            try:
                return path.relative_to(out_dir).as_posix()
            except ValueError:
                return path.as_posix()

        orig_rel = _rel(r.original_path)
        if orig_rel:
            md.append(f"### Original")
            md.append(f"![{r.slug} original]({orig_rel})")
            md.append("")

        for key, heading in [
            ("detect", "Detection boxes"),
            ("ocr", "OCR text"),
            ("translated", "Translation overlay"),
            ("combined", "Combined overlay"),
        ]:
            path = r.overlays.get(key)
            rel = _rel(path)
            if rel:
                md.append(f"### {heading}")
                md.append(f"![{r.slug} {key}]({rel})")
                md.append("")

        md.append("### Inpainted")
        md.append("_Placeholder — will be populated once the inpainting branch merges._")
        md.append("")

        if r.response_json_path is not None:
            rel = _rel(r.response_json_path)
            md.append(f"- Raw JSON response: [`{rel}`]({rel})")
            md.append("")

        md.append("---")
        md.append("")

    md.append("## Baseline timings")
    md.append("")
    md.append("See [`baseline_timings.txt`](baseline_timings.txt).")
    md.append("")

    summary_path.write_text("\n".join(md))
    return summary_path


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[harness] gallery output dir: {GALLERY_DIR}")
    tests = _candidate_images()
    if not tests:
        print("[harness] No test images found. Bailing.", file=sys.stderr)
        return 2
    print(f"[harness] {len(tests)} test images:")
    for t in tests:
        print(f"  - {t.label:30s} {t.source}")

    # Lazy import so PIL/dataclasses always work even if FastAPI import fails.
    pipeline_error: str | None = None
    client = None
    try:
        # Importing `app.main` triggers model loading in the lifespan ctx.
        sys.path.insert(0, str(BACKEND_DIR))
        from fastapi.testclient import TestClient  # noqa: WPS433

        from app.main import app  # noqa: WPS433

        client = TestClient(app)
        # Enter lifespan so startup hooks run.
        client.__enter__()
    except Exception:  # pragma: no cover - depends on local model weights
        pipeline_error = traceback.format_exc()
        print(
            "[harness] Failed to initialise FastAPI app — emitting empty gallery.",
            file=sys.stderr,
        )
        print(pipeline_error, file=sys.stderr)

    results: list[RunResult] = []
    if client is not None:
        try:
            for t in tests:
                if not t.exists:
                    print(f"[harness] missing image: {t.source}", file=sys.stderr)
                    continue
                print(f"[harness] running {t.label} ...")
                result = _run_one(client, t, GALLERY_DIR)
                print(
                    f"  -> status={result.status} boxes={result.box_count} "
                    f"wall={result.wall_ms:.0f}ms"
                )
                results.append(result)
        finally:
            try:
                client.__exit__(None, None, None)
            except Exception:
                pass
    else:
        # Still copy originals so the gallery has something to show.
        for t in tests:
            if not t.exists:
                continue
            orig = _copy_original(t, GALLERY_DIR)
            results.append(
                RunResult(
                    test=t,
                    original_path=orig,
                    slug=t.label,
                    status="failed",
                    wall_ms=0.0,
                    box_count=0,
                    error="FastAPI app did not initialise; see pipeline startup error above.",
                )
            )

    timings = _write_timings(results, GALLERY_DIR)
    summary = _write_summary(results, GALLERY_DIR, pipeline_error)
    print(f"[harness] wrote {summary}")
    print(f"[harness] wrote {timings}")

    # Exit 0 even on partial runs — the user wants to visually inspect whatever we can emit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

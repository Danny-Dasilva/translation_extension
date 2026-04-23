"""End-to-end pipeline visualizer.

Runs the full integrated pipeline on a set of test manga images and emits
one PNG per stage plus per-feature demo images, so the koharu-inspired
improvements can be visually validated step-by-step.

Usage:
    uv run python scripts/visualize_e2e_pipeline.py

Output:
    thoughts/koharu-improvements/pipeline-e2e/
        <slug>/01_original.png ... 11_final.png (per image)
        features/feature_*.png (per-feature demos)
        SUMMARY.md
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure backend/ is on sys.path when run from the repo root or backend/.
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.services.detector_factory import create_detector  # noqa: E402
from app.services.parseq_ocr_service import ParseqOCRService  # noqa: E402
from app.services.local_translation_service import (  # noqa: E402
    LocalTranslationService,
    LocalTranslationPool,
    BATCHED_SYSTEM_PROMPT,
    format_sources,
)
from app.utils.ocr_postprocess import apply_all as ocr_postproc  # noqa: E402
from app.utils.image_processing import snap_font_color, detect_font_colors  # noqa: E402
from app.utils.japanese_text_filter import is_japanese_text  # noqa: E402
from app.utils.ctd_utils import build_text_regions  # noqa: E402

try:
    from app.services.lama_inpaint_service import LamaInpaintService
except Exception:
    LamaInpaintService = None  # type: ignore


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_FONT_SEARCH = [
    BACKEND_DIR / "fonts" / "Anton-Regular.ttf",
    BACKEND_DIR / "fonts" / "Bangers-Regular.ttf",
    BACKEND_DIR / "fonts" / "Oswald-Bold.ttf",
    BACKEND_DIR / "fonts" / "ComicNeue-Bold.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
]


def load_font(size: int, path: Path | None = None) -> ImageFont.ImageFont:
    if path is None:
        for p in _FONT_SEARCH:
            if p.exists():
                path = p
                break
    if path is None:
        return ImageFont.load_default()
    try:
        return ImageFont.truetype(str(path), size)
    except Exception:
        return ImageFont.load_default()


FONT_SM = None
FONT_MD = None
FONT_LG = None


def _init_fonts():
    global FONT_SM, FONT_MD, FONT_LG
    FONT_SM = load_font(12)
    FONT_MD = load_font(18)
    FONT_LG = load_font(28)


def draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str,
               fg=(255, 255, 255), bg=(0, 0, 0)):
    font = FONT_SM
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2), fill=bg)
    draw.text((x, y), text, fill=fg, font=font)


def annotate_image(img: np.ndarray, title: str) -> Image.Image:
    """Add a footer title to an image."""
    pil = Image.fromarray(img).convert("RGB")
    w, h = pil.size
    footer_h = 32
    canvas = Image.new("RGB", (w, h + footer_h), (24, 24, 24))
    canvas.paste(pil, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, h + 6), title, fill=(255, 255, 255), font=FONT_MD)
    return canvas


def draw_boxes(img: np.ndarray, boxes, color=(0, 255, 255),
               label_fn=None) -> np.ndarray:
    out = img.copy()
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = int(b["minX"]), int(b["minY"]), int(b["maxX"]), int(b["maxY"])
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        label = label_fn(i, b) if label_fn else f"#{i}"
        draw_label(draw, x0 + 4, max(0, y0 - 16), label, bg=color, fg=(0, 0, 0))
    return np.array(pil)


def composite_text_on_plate(plate_rgb: np.ndarray, boxes, translations) -> np.ndarray:
    """Render translated text centered in each bbox on the inpainted plate.

    Delegates to refit_final_composites.compose_final so the two scripts
    stay in lock-step — single source of truth for layout semantics.
    """
    try:
        from scripts.refit_final_composites import compose_final  # local dev import
    except Exception:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "refit_final_composites",
            SCRIPT_DIR / "refit_final_composites.py",
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        compose_final = mod.compose_final
    return compose_final(plate_rgb, list(boxes), list(translations))


# ---------------------------------------------------------------------------
# Per-feature demos (no image required)
# ---------------------------------------------------------------------------

def feature_ocr_normalizer(out: Path):
    cases = [
        ("“テスト――！！！！！！”", 'Smart quotes, em dash, run of !'),
        ("わかった…でも・・・何？", 'Ellipsis + middle-dot run'),
        ("abcABC ＡＢＣ １２３", 'Full-width alphanumerics'),
        ("ＯＭＧーーーー", 'NAR decoder trailing repeat ー'),
        ("テストｶﾀｶﾅ", 'Half-width katakana'),
        ("zero​width​chars", 'Zero-width stripping'),
    ]
    rows = [(raw, ocr_postproc(raw), note) for raw, _, note in [(raw, None, note) for raw, note in [(c[0], c[1]) for c in cases]]]
    # Build a wide image with 3 columns
    w, row_h = 1200, 58
    h = row_h * (len(rows) + 1) + 30
    img = Image.new("RGB", (w, h), (24, 24, 24))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, w, row_h), fill=(40, 40, 40))
    d.text((12, 12), "OCR normalizer (ocr_postprocess.apply_all)", fill=(0, 255, 255), font=FONT_LG)
    d.text((12, row_h), "RAW INPUT", fill=(255, 200, 200), font=FONT_MD)
    d.text((460, row_h), "NORMALIZED", fill=(200, 255, 200), font=FONT_MD)
    d.text((860, row_h), "WHAT IT FIXED", fill=(200, 200, 255), font=FONT_MD)
    for i, (raw, out_str, note) in enumerate(rows, start=2):
        y = i * row_h
        d.text((12, y + 8), raw[:38], fill=(255, 255, 255), font=FONT_MD)
        d.text((460, y + 8), out_str[:38], fill=(255, 255, 255), font=FONT_MD)
        d.text((860, y + 8), note, fill=(180, 180, 180), font=FONT_MD)
    img.save(out)


def feature_font_color_snap(out: Path):
    samples = [
        ((5, 3, 8), "near-black"),
        ((19, 19, 19), "threshold edge (<=20)"),
        ((30, 10, 5), "deep-red kept"),
        ((128, 128, 128), "gray kept"),
        ((250, 248, 255), "near-white"),
        ((240, 240, 255), "threshold edge (>=235)"),
    ]
    w, row_h = 920, 88
    h = row_h * (len(samples) + 1) + 20
    img = Image.new("RGB", (w, h), (24, 24, 24))
    d = ImageDraw.Draw(img)
    d.text((14, 14), "snap_font_color(threshold=20)  — snaps near-pure colors to exact 0/255",
           fill=(0, 255, 255), font=FONT_MD)
    for i, (rgb, note) in enumerate(samples, start=1):
        y = i * row_h
        snapped = snap_font_color(rgb, 20)
        d.rectangle((20, y + 8, 80, y + 72), fill=rgb, outline=(255, 255, 255))
        d.rectangle((110, y + 8, 170, y + 72), fill=snapped, outline=(255, 255, 255))
        d.text((200, y + 20), f"{rgb} → {snapped}", fill=(255, 255, 255), font=FONT_MD)
        d.text((560, y + 20), note, fill=(180, 180, 180), font=FONT_MD)
    img.save(out)


def feature_repetition_guard(out: Path):
    # Show where the n-gram/5+ run guard would trigger
    from app.services.parseq_ocr_service import _LONG_RUN_RE, _has_trigram_loop
    samples = [
        "ててててて",            # 5+ run — triggers
        "テスト",                # normal — no
        "abcabcabcabc",           # trigram loop — triggers
        "わかった",              # no
        "ーーーーーーー",        # NAR trailing (normalized elsewhere)
    ]
    w, row_h = 900, 70
    h = row_h * (len(samples) + 1) + 10
    img = Image.new("RGB", (w, h), (24, 24, 24))
    d = ImageDraw.Draw(img)
    d.text((12, 12), "OCR repetition guard (parseq_ocr_service._LONG_RUN_RE / _has_trigram_loop)",
           fill=(0, 255, 255), font=FONT_MD)
    for i, s in enumerate(samples, start=1):
        y = i * row_h
        long_run = bool(_LONG_RUN_RE.search(s))
        tri = _has_trigram_loop(s, min_repeats=4)
        color = (255, 120, 120) if (long_run or tri) else (180, 255, 180)
        d.text((20, y + 12), s, fill=(255, 255, 255), font=FONT_MD)
        flags = []
        if long_run:
            flags.append("5+ same-char run")
        if tri:
            flags.append("trigram loop")
        note = " | ".join(flags) if flags else "clean"
        d.text((380, y + 12), note, fill=color, font=FONT_MD)
    img.save(out)


def feature_batched_prompt(out: Path):
    """Static demo of the batched [1]..[N] prompt shape."""
    sample = ["すごい！", "た…助けて…", "ふふ…覚悟しろ"]
    sources = format_sources(sample)
    text = BATCHED_SYSTEM_PROMPT + "\n\n--- USER ---\n" + sources + "\n\n--- EXPECTED OUTPUT ---\n[1] Amazing!\n[2] H...help me...\n[3] Heh... prepare yourself"
    w, h = 1200, 720
    img = Image.new("RGB", (w, h), (24, 24, 24))
    d = ImageDraw.Draw(img)
    d.text((12, 12), "Batched LLM translate — [N]-tagged protocol (koharu llm.rs:439-524 port)",
           fill=(0, 255, 255), font=FONT_MD)
    y = 48
    for line in text.splitlines():
        d.text((12, y), line[:150], fill=(235, 235, 235), font=FONT_SM)
        y += 16
        if y > h - 20:
            break
    img.save(out)


def all_feature_demos(features_dir: Path):
    features_dir.mkdir(parents=True, exist_ok=True)
    feature_ocr_normalizer(features_dir / "feature_ocr_normalizer.png")
    feature_font_color_snap(features_dir / "feature_font_color_snap.png")
    feature_repetition_guard(features_dir / "feature_repetition_guard.png")
    feature_batched_prompt(features_dir / "feature_batched_prompt.png")
    print(f"  features → {features_dir}")


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------

class PipelineRunner:
    def __init__(self):
        print("  loading detector…")
        self.detector = create_detector()
        print("  loading OCR (PARSeq)…")
        self.ocr = ParseqOCRService(model_path=settings.parseq_model_path)
        self.lama = None
        if LamaInpaintService is not None and Path(BACKEND_DIR, settings.lama_model_path).exists():
            print("  loading LaMa…")
            try:
                self.lama = LamaInpaintService(model_path=settings.lama_model_path)
            except Exception as exc:
                print(f"  LaMa failed to load: {exc}; proceeding without plate")
        else:
            print("  LaMa unavailable — plate stage will be skipped")
        self.translator: Optional[LocalTranslationPool | LocalTranslationService] = None
        try:
            if settings.translation_num_instances > 1:
                print(f"  loading LLM pool ({settings.translation_num_instances}×)…")
                self.translator = LocalTranslationPool()
            else:
                print("  loading LLM single instance…")
                self.translator = LocalTranslationService()
        except Exception as exc:
            print(f"  LLM failed to load: {exc}; translation stage will emit empty strings")

    async def run(self, image_path: Path, out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        stats: dict = {"image": image_path.name}

        # 01: original
        pil = Image.open(image_path).convert("RGB")
        pil.save(out_dir / "01_original.png")
        image_np = np.array(pil)

        # 02 + 03: detect
        print(f"  [{image_path.name}] detect…")
        t0 = time.time()
        ctd = await self.detector.detect(image_np)
        stats["detect_ms"] = (time.time() - t0) * 1000
        blocks = ctd["blocks"]
        text_lines = ctd["text_lines"]
        stats["num_blocks"] = len(blocks)
        stats["num_text_lines"] = len(text_lines)

        annotate_image(draw_boxes(image_np, blocks, (0, 255, 255),
                                   lambda i, b: f"blk {i} ({int(b.get('confidence', 0) * 100)}%)"),
                       "02 — detect: bubble/block bboxes").save(out_dir / "02_detect_blocks.png")
        annotate_image(draw_boxes(image_np, text_lines, (180, 255, 100),
                                   lambda i, _b: f"L{i}"),
                       "03 — detect: text-line bboxes").save(out_dir / "03_detect_lines.png")

        # 04 + 05: masks
        mask = ctd.get("mask")
        if mask is not None and mask.size:
            mask_full = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_rgb = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2RGB)
            annotate_image(mask_rgb, "04 — detector mask (CTD output after koharu refinement)").save(out_dir / "04_mask_refined.png")
        else:
            Image.new("RGB", pil.size, (0, 0, 0)).save(out_dir / "04_mask_refined.png")

        # Build the actual inpaint mask the router would construct
        inpaint_mask = _build_inpaint_mask_vis(image_np.shape, blocks, text_lines, mask)
        mask_rgb = cv2.cvtColor(inpaint_mask, cv2.COLOR_GRAY2RGB)
        # overlay red on the original so the user sees what gets erased
        overlay = image_np.copy()
        red = np.zeros_like(overlay)
        red[..., 0] = 255
        alpha = (inpaint_mask > 0)[..., None].astype(np.float32) * 0.45
        overlay = (overlay * (1 - alpha) + red * alpha).astype(np.uint8)
        annotate_image(overlay, "05 — composed LaMa mask (text_lines ∪ detector_mask + dilate)").save(out_dir / "05_inpaint_mask.png")

        # 06: OCR crop montage
        print(f"  [{image_path.name}] OCR…")
        t0 = time.time()
        if text_lines:
            ocr_texts = await self.ocr.recognize_blocks_with_lines(
                image_np, blocks, text_lines, batch_size=settings.parseq_batch_size
            )
        else:
            crops = self.detector.crop_regions(image_np, blocks)
            ocr_texts = await self.ocr.recognize_text_batch(crops)
        stats["ocr_ms"] = (time.time() - t0) * 1000
        _save_crop_montage(image_np, blocks, ocr_texts, out_dir / "06_ocr_crops.png")

        # Filter non-japanese for fairness
        valid_pairs = [(i, t) for i, t in enumerate(ocr_texts)
                       if is_japanese_text(t, settings.japanese_filter_min_ratio,
                                           settings.japanese_filter_katakana_max_length)]
        valid_idx = [p[0] for p in valid_pairs]
        kept_blocks = [blocks[i] for i in valid_idx]
        kept_texts = [p[1] for p in valid_pairs]

        # 07: inpaint
        inpainted: Optional[np.ndarray] = None
        if self.lama is not None:
            print(f"  [{image_path.name}] inpaint…")
            t0 = time.time()
            try:
                inpainted = await asyncio.to_thread(self.lama.inpaint, image_np, inpaint_mask)
                stats["inpaint_ms"] = (time.time() - t0) * 1000
                annotate_image(inpainted, "07 — LaMa inpainted (clean plate)").save(out_dir / "07_inpainted.png")
            except Exception as exc:
                print(f"  inpaint failed: {exc}")
                inpainted = None
        if inpainted is None:
            annotate_image(image_np, "07 — inpaint skipped (no LaMa)").save(out_dir / "07_inpainted.png")
            inpainted = image_np

        # 08 + 09: batched translate
        translations: list[str] = []
        prompt_text = ""
        raw_reply = ""
        if self.translator is not None and kept_texts:
            print(f"  [{image_path.name}] batched translate ({len(kept_texts)} bubbles)…")
            try:
                prompt_text = BATCHED_SYSTEM_PROMPT + "\n\n" + format_sources(kept_texts)
                t0 = time.time()
                translations = await self.translator.translate_batched(kept_texts, "English")
                stats["translate_ms"] = (time.time() - t0) * 1000
                raw_reply = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(translations))
            except Exception as exc:
                print(f"  batched translate failed: {exc}; falling back to sequential")
                translations = []
                for t in kept_texts:
                    try:
                        tr = await self.translator.translate_single(t, "English")
                    except Exception:
                        tr = ""
                    translations.append(tr)
                raw_reply = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(translations))

        (out_dir / "08_translate_prompt.txt").write_text(prompt_text or "(translation skipped)")
        (out_dir / "09_translate_response.txt").write_text(raw_reply or "(no response)")

        # 10: detect+OCR+translate triple overlay
        ann = draw_boxes(image_np, kept_blocks, (0, 220, 220))
        ann_pil = Image.fromarray(ann)
        d = ImageDraw.Draw(ann_pil)
        for i, (b, ocr, tr) in enumerate(zip(kept_blocks, kept_texts, translations)):
            x0, y0 = int(b["minX"]), int(b["minY"])
            draw_label(d, x0, max(0, y0 - 30), f"JP: {ocr[:24]}",
                       bg=(255, 200, 0), fg=(0, 0, 0))
            draw_label(d, x0, max(0, y0 - 14), f"EN: {(tr or '(empty)')[:32]}",
                       bg=(0, 200, 0), fg=(0, 0, 0))
        annotate_image(np.array(ann_pil), "10 — OCR + translation pairs").save(out_dir / "10_ocr_translate.png")

        # 11: final composite (translated text on inpainted plate)
        final = composite_text_on_plate(inpainted, kept_blocks, translations)
        annotate_image(final, "11 — final: translated text rendered on LaMa plate").save(out_dir / "11_final_composite.png")

        stats["ocr_samples"] = kept_texts[:8]
        stats["translations"] = translations[:8]
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        print(f"  [{image_path.name}] ✓ wrote {out_dir}")
        return stats


def _build_inpaint_mask_vis(image_shape, blocks, text_lines, detector_mask) -> np.ndarray:
    """Mirror of router._build_inpaint_mask so visualization matches runtime."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    sources = text_lines if text_lines else blocks
    for region in sources:
        x0 = max(0, int(region.get("minX", 0)))
        y0 = max(0, int(region.get("minY", 0)))
        x1 = min(w, int(region.get("maxX", 0)))
        y1 = min(h, int(region.get("maxY", 0)))
        if x1 > x0 and y1 > y0:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    if detector_mask is not None and detector_mask.size:
        dm = detector_mask
        if dm.shape[:2] != (h, w):
            dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_NEAREST)
        _, dm_bin = cv2.threshold(dm, 127, 255, cv2.THRESH_BINARY)
        mask = np.maximum(mask, dm_bin.astype(np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _save_crop_montage(image_np: np.ndarray, blocks, texts, out: Path, max_items: int = 8):
    if not blocks:
        Image.new("RGB", (600, 80), (24, 24, 24)).save(out)
        return
    items = list(zip(blocks, texts))[:max_items]
    tiles = []
    for b, t in items:
        x0, y0, x1, y1 = int(b["minX"]), int(b["minY"]), int(b["maxX"]), int(b["maxY"])
        crop = image_np[max(0, y0):y1, max(0, x0):x1]
        if crop.size == 0:
            continue
        # Normalize tile to max 160px tall
        ch, cw = crop.shape[:2]
        scale = 160 / max(ch, 1)
        if scale < 1:
            crop = cv2.resize(crop, (max(1, int(cw * scale)), 160), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(crop)
        w = pil.width
        canvas = Image.new("RGB", (max(180, w), 200), (24, 24, 24))
        canvas.paste(pil, ((canvas.width - w) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 170), t[:24], fill=(255, 255, 0), font=FONT_SM)
        tiles.append(canvas)
    if not tiles:
        Image.new("RGB", (600, 80), (24, 24, 24)).save(out)
        return
    W = sum(t.width for t in tiles) + 10 * len(tiles)
    H = max(t.height for t in tiles) + 30
    out_img = Image.new("RGB", (W, H), (16, 16, 16))
    x = 5
    for t in tiles:
        out_img.paste(t, (x, 20))
        x += t.width + 10
    draw = ImageDraw.Draw(out_img)
    draw.text((8, 2), "06 — OCR crops → recognized text (with normalizer applied)",
              fill=(200, 200, 200), font=FONT_SM)
    out_img.save(out)


# ---------------------------------------------------------------------------
# Test image discovery
# ---------------------------------------------------------------------------

def discover_test_images(limit: int = 4) -> list[Path]:
    candidates: list[Path] = []
    # Primary
    de = REPO_ROOT / "de.png"
    if de.exists():
        candidates.append(de)
    # Training examples
    for root in (REPO_ROOT / "training",):
        if not root.exists():
            continue
        for p in root.rglob("*.jpg"):
            if any(x in p.parts for x in (".uv-cache", ".venv", "__pycache__", "node_modules")):
                continue
            if "examples" in p.parts or "val" in p.parts:
                candidates.append(p)
                if len(candidates) >= limit * 3:
                    break
    # Dedup by filename + cap
    seen = set()
    chosen: list[Path] = []
    for p in candidates:
        if p.name in seen:
            continue
        seen.add(p.name)
        chosen.append(p)
        if len(chosen) >= limit:
            break
    return chosen


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    _init_fonts()
    out_root = REPO_ROOT / "thoughts" / "koharu-improvements" / "pipeline-e2e"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_root}")

    # Per-feature demos (no model calls)
    print("feature demos…")
    all_feature_demos(out_root / "features")

    # Pipeline runner (loads all services — slow)
    print("loading pipeline…")
    runner = PipelineRunner()

    images = discover_test_images(limit=4)
    if not images:
        print("No test images found; abort.")
        return
    print(f"test images: {[p.name for p in images]}")

    summaries: list[dict] = []
    for img in images:
        slug = img.stem.replace(" ", "_")[:40]
        try:
            stats = await runner.run(img, out_root / slug)
            stats["slug"] = slug
            summaries.append(stats)
        except Exception as exc:
            print(f"  [{img.name}] FAILED: {exc}")
            summaries.append({"slug": slug, "image": img.name, "error": str(exc)})

    # Master SUMMARY.md
    md = ["# End-to-end koharu pipeline gallery\n",
          "Generated by `backend/scripts/visualize_e2e_pipeline.py`.\n",
          "Each test page has 11 per-stage artefacts + OCR/translate text files.\n\n",
          "## Per-feature demos\n\n",
          "![OCR normalizer](features/feature_ocr_normalizer.png)\n",
          "![Font color snap](features/feature_font_color_snap.png)\n",
          "![OCR repetition guard](features/feature_repetition_guard.png)\n",
          "![Batched LLM prompt](features/feature_batched_prompt.png)\n\n",
          "## Per-image galleries\n\n"]
    for s in summaries:
        slug = s.get("slug", "?")
        md.append(f"### {slug} (`{s.get('image', '?')}`)\n\n")
        if "error" in s:
            md.append(f"> ❌ `{s['error']}`\n\n")
            continue
        md.append(f"- Blocks: **{s.get('num_blocks', 0)}**  Text lines: **{s.get('num_text_lines', 0)}**\n")
        md.append(f"- Timings (ms): detect `{s.get('detect_ms', 0):.0f}` · ocr `{s.get('ocr_ms', 0):.0f}` · "
                  f"inpaint `{s.get('inpaint_ms', 0):.0f}` · translate `{s.get('translate_ms', 0):.0f}`\n\n")
        md.append(f"![01 original]({slug}/01_original.png)\n")
        md.append(f"![02 detect blocks]({slug}/02_detect_blocks.png)\n")
        md.append(f"![03 detect lines]({slug}/03_detect_lines.png)\n")
        md.append(f"![04 refined mask]({slug}/04_mask_refined.png)\n")
        md.append(f"![05 inpaint mask]({slug}/05_inpaint_mask.png)\n")
        md.append(f"![06 OCR crops]({slug}/06_ocr_crops.png)\n")
        md.append(f"![07 inpainted]({slug}/07_inpainted.png)\n")
        md.append(f"![10 OCR+translate]({slug}/10_ocr_translate.png)\n")
        md.append(f"![11 final]({slug}/11_final_composite.png)\n\n")
        if s.get("ocr_samples"):
            md.append("| # | JP (OCR) | EN (translate) |\n|---|---|---|\n")
            for i, (jp, en) in enumerate(zip(s.get("ocr_samples", []), s.get("translations", [])), 1):
                md.append(f"| {i} | {jp[:40]} | {en[:40]} |\n")
            md.append("\n")
    (out_root / "SUMMARY.md").write_text("".join(md), encoding="utf-8")
    print(f"\nGallery written to {out_root}/SUMMARY.md")


if __name__ == "__main__":
    asyncio.run(main())

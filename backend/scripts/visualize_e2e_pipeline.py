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
from app.services.translation_text_utils import (  # noqa: E402
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


def composite_text_on_plate(plate_rgb: np.ndarray, boxes, translations,
                            fit_rects=None) -> np.ndarray:
    """Render translated text centered in each bbox on the inpainted plate.

    Delegates to refit_final_composites.compose_final so the two scripts
    stay in lock-step — single source of truth for layout semantics.
    ``fit_rects`` (parallel to boxes) supplies the speech-bubble rect to typeset
    into instead of the tight text block, or None per item.
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
    return compose_final(plate_rgb, list(boxes), list(translations),
                         fit_rects=list(fit_rects) if fit_rects is not None else None)


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
    orphan_mode: str = "off"  # off | paragraph | sentence

    async def _ocr_orphan_clusters(self, image_np, clusters) -> list[str]:
        """OCR each cluster's lines in reading order; return joined text per cluster."""
        h, w = image_np.shape[:2]
        flat, owner = [], []
        for ci, cluster in enumerate(clusters):
            for ln in _order_cluster_lines(cluster):
                x0 = max(0, int(ln["minX"]) - 2); y0 = max(0, int(ln["minY"]) - 2)
                x1 = min(w, int(ln["maxX"]) + 2); y1 = min(h, int(ln["maxY"]) + 2)
                if x1 > x0 and y1 > y0:
                    flat.append(image_np[y0:y1, x0:x1])
                    owner.append(ci)
        if not flat:
            return ["" for _ in clusters]
        texts = await self.ocr.recognize_text_batch(flat, batch_size=settings.parseq_batch_size)
        joined = [[] for _ in clusters]
        for ci, t in zip(owner, texts):
            if t:
                joined[ci].append(t)
        return ["".join(parts) for parts in joined]

    def __init__(self):
        print("  loading detector…")
        self.detector = create_detector()
        # Speech-bubble detector (YOLOv10n). Used to typeset translated text to
        # the bubble interior instead of the tight (vertical-JP) text column.
        self.bubble_detector = None
        try:
            from app.services.detector_service import DetectorService
            if Path(BACKEND_DIR, settings.yolo_model_path).exists():
                print("  loading bubble detector (YOLOv10n)…")
                self.bubble_detector = DetectorService()
        except Exception as exc:
            print(f"  bubble detector unavailable ({exc}); typesetting falls back to text blocks")
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
        self.translator: Optional[object] = None
        try:
            backend = getattr(settings, "translation_backend", "vllm-openai")
            if backend == "transformers":
                print("  loading LLM (transformers backend, Hy-MT1.5-2bit)…")
                from app.services.hymt_transformers_service import (
                    HyMTTransformersService,
                )
                self.translator = HyMTTransformersService()
            elif backend == "vllm-openai":
                from app.services.vllm_openai_translation_service import (
                    VLLMOpenAITranslationService,
                )
                print(
                    f"  using vLLM OpenAI backend at {settings.vllm_base_url} "
                    f"(model={settings.vllm_model_name})"
                )
                self.translator = VLLMOpenAITranslationService(
                    base_url=settings.vllm_base_url,
                    model_name=settings.vllm_model_name,
                )
            else:
                raise ValueError(
                    f"Unknown translation_backend {backend!r}. "
                    "Use 'vllm-openai' or 'transformers'."
                )
        except Exception as exc:
            print(f"  LLM failed to load: {exc}; translation stage will emit empty strings")

        # Device diagnostics — surface the actual backends so a slow run is
        # easy to diagnose (e.g. LaMa fell back to CPU).
        lama_device = getattr(self.lama, "device", None) if self.lama else "unavailable"
        translator_kind = type(self.translator).__name__ if self.translator else "unavailable"
        translator_n = getattr(self.translator, "num_instances", 1) if self.translator else 0
        self.device_info = {
            "lama_device": lama_device,
            "translator": translator_kind,
            "translator_instances": translator_n,
            "translation_backend": getattr(settings, "translation_backend", "vllm-openai"),
        }
        print(f"  devices: lama={lama_device} translator={translator_kind}"
              f"(×{translator_n}) backend={self.device_info['translation_backend']}")

    async def run(self, image_path: Path, out_dir: Path,
                  lama_max_side: int | None = None) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        stats: dict = {"image": image_path.name, **getattr(self, "device_info", {})}

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

        # Speech-bubble detection (for typesetting to the bubble interior).
        bubbles = []
        if self.bubble_detector is not None:
            try:
                bubbles = await self.bubble_detector.detect_bubbles(image_np)
            except Exception as exc:
                print(f"  bubble detect failed ({exc}); using text blocks for typesetting")
        stats["num_bubbles"] = len(bubbles)

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

        # Optional: recover text_lines that no detected block claims (dense
        # paragraphs, chat-message boxes). "paragraph" clusters nearby orphans
        # into one synthetic block; "sentence" makes one block per line.
        orphan_mode = getattr(self, "orphan_mode", "off")
        if orphan_mode != "off" and text_lines:
            orphans = _find_orphan_lines(blocks, text_lines)
            if orphans:
                clusters = (_cluster_orphan_lines(orphans)
                            if orphan_mode == "paragraph"
                            else [[ln] for ln in orphans])
                synth_texts = await self._ocr_orphan_clusters(image_np, clusters)
                added = 0
                for cluster, text in zip(clusters, synth_texts):
                    if not text.strip():
                        continue
                    blocks.append({
                        "minX": min(ln["minX"] for ln in cluster),
                        "minY": min(ln["minY"] for ln in cluster),
                        "maxX": max(ln["maxX"] for ln in cluster),
                        "maxY": max(ln["maxY"] for ln in cluster),
                        "confidence": 0.5, "orphan": True,
                    })
                    ocr_texts.append(text)
                    added += 1
                stats["orphan_mode"] = orphan_mode
                stats["orphan_blocks_added"] = added
                print(f"  [{image_path.name}] orphan-lines({orphan_mode}): "
                      f"{len(orphans)} lines → {added} synthetic block(s)")

        _save_crop_montage(image_np, blocks, ocr_texts, out_dir / "06_ocr_crops.png")

        # Filter non-japanese for fairness
        valid_pairs = [(i, t) for i, t in enumerate(ocr_texts)
                       if is_japanese_text(t, settings.japanese_filter_min_ratio,
                                           settings.japanese_filter_katakana_max_length)]
        valid_idx = [p[0] for p in valid_pairs]
        kept_blocks = [blocks[i] for i in valid_idx]
        kept_texts = [p[1] for p in valid_pairs]

        # 05: inpaint mask — built from KEPT blocks only, mirroring the router.
        # Regions whose OCR failed the Japanese filter (or never merged into a
        # block) are left untouched rather than erased-without-replacement.
        inpaint_mask = _build_inpaint_mask_vis(image_np.shape, kept_blocks, text_lines, mask)
        overlay = image_np.copy()
        red = np.zeros_like(overlay)
        red[..., 0] = 255
        alpha = (inpaint_mask > 0)[..., None].astype(np.float32) * 0.45
        overlay = (overlay * (1 - alpha) + red * alpha).astype(np.uint8)
        annotate_image(overlay, "05 — composed LaMa mask (kept-block lines ∪ clipped detector_mask + dilate)").save(out_dir / "05_inpaint_mask.png")

        # 07: inpaint
        inpainted: Optional[np.ndarray] = None
        if self.lama is not None:
            print(f"  [{image_path.name}] inpaint…")
            t0 = time.time()
            try:
                inpaint_kwargs: dict = {}
                if lama_max_side:
                    inpaint_kwargs["max_side"] = lama_max_side
                inpainted = await asyncio.to_thread(
                    self.lama.inpaint, image_np, inpaint_mask, **inpaint_kwargs
                )
                stats["inpaint_ms"] = (time.time() - t0) * 1000
                stats["inpaint_detail"] = getattr(self.lama, "last_stats", {})
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

        # 11: final composite (translated text on inpainted plate).
        # Match each kept text block to its speech bubble so text is typeset to
        # the bubble interior (wide) rather than the tight vertical-JP column.
        from app.utils.ctd_utils import match_blocks_to_bubbles
        fit_rects = match_blocks_to_bubbles(kept_blocks, bubbles) if bubbles else None
        if fit_rects is not None:
            stats["blocks_matched_to_bubbles"] = sum(1 for r in fit_rects if r is not None)
        final = composite_text_on_plate(inpainted, kept_blocks, translations, fit_rects=fit_rects)
        annotate_image(final, "11 — final: translated text rendered on LaMa plate").save(out_dir / "11_final_composite.png")

        stats["ocr_samples"] = kept_texts[:8]
        stats["translations"] = translations[:8]
        stats["ocr_all"] = kept_texts
        stats["translations_all"] = translations
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

        # translations.txt — JP (OCR) + EN (translation) pairs, full list, utf-8
        _write_translations_txt(out_dir / "translations.txt",
                                image_path.name, kept_texts, translations)

        print(f"  [{image_path.name}] ✓ wrote {out_dir}")
        return stats


def _write_translations_txt(path: Path, image_name: str,
                            jp_texts: list[str], translations: list[str]) -> None:
    """Write a UTF-8 text file pairing JP OCR with EN translations for an image."""
    lines = [f"# {image_name}",
             f"# {len(jp_texts)} bubble(s)",
             ""]
    for i, jp in enumerate(jp_texts):
        en = translations[i] if i < len(translations) else ""
        lines.append(f"[{i + 1}]")
        lines.append(f"  JP: {jp}")
        lines.append(f"  EN: {en}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_aggregate_translations(out_root: Path, summaries: list[dict]) -> None:
    """Write a single translations.txt at the gallery root pairing JP + EN per page."""
    lines = ["# Aggregate OCR + translations",
             f"# {len(summaries)} page(s)",
             ""]
    for s in summaries:
        slug = s.get("slug", "?")
        image = s.get("image", "?")
        lines.append(f"## {slug}  ({image})")
        if "error" in s:
            lines.append(f"  ERROR: {s['error']}")
            lines.append("")
            continue
        jps = s.get("ocr_all", s.get("ocr_samples", []))
        ens = s.get("translations_all", s.get("translations", []))
        for i, jp in enumerate(jps):
            en = ens[i] if i < len(ens) else ""
            lines.append(f"  [{i + 1}] JP: {jp}")
            lines.append(f"      EN: {en}")
        lines.append("")
    (out_root / "translations.txt").write_text("\n".join(lines), encoding="utf-8")


def _build_inpaint_mask_vis(image_shape, blocks, text_lines, detector_mask) -> np.ndarray:
    """Same mask the router builds — pass the post-filter (kept) blocks."""
    from app.utils.ctd_utils import build_inpaint_mask
    return build_inpaint_mask(image_shape, blocks, text_lines, detector_mask)


def _find_orphan_lines(blocks: list[dict], text_lines: list[dict]) -> list[dict]:
    """Lines whose center no block contains (mirrors OCR assignment rule)."""
    orphans = []
    for ln in text_lines:
        cx = (ln["minX"] + ln["maxX"]) / 2
        cy = (ln["minY"] + ln["maxY"]) / 2
        if not any(b["minX"] <= cx <= b["maxX"] and b["minY"] <= cy <= b["maxY"]
                   for b in blocks):
            orphans.append(ln)
    return orphans


def _cluster_orphan_lines(orphans: list[dict]) -> list[list[dict]]:
    """Union-find clustering: lines whose bboxes (expanded ~1.2 line-heights)
    intersect belong to one paragraph."""
    n = len(orphans)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def expand(ln):
        h = ln["maxY"] - ln["minY"]
        w = ln["maxX"] - ln["minX"]
        pad = 1.2 * min(h, w) if min(h, w) > 0 else 12
        return (ln["minX"] - pad, ln["minY"] - pad, ln["maxX"] + pad, ln["maxY"] + pad)

    boxes = [expand(ln) for ln in orphans]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                parent[find(i)] = find(j)

    groups: dict[int, list[dict]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(orphans[i])
    return list(groups.values())


def _order_cluster_lines(cluster: list[dict]) -> list[dict]:
    """Reading order. Horizontal lines (w > h): top-to-bottom. Vertical
    columns: right-to-left, then top-to-bottom — matches manga convention."""
    horiz = sum(1 for ln in cluster
                if (ln["maxX"] - ln["minX"]) > (ln["maxY"] - ln["minY"]))
    if horiz >= len(cluster) / 2:
        return sorted(cluster, key=lambda ln: (ln["minY"], ln["minX"]))
    return sorted(cluster, key=lambda ln: (-ln["minX"], ln["minY"]))


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
    import argparse

    ap = argparse.ArgumentParser(
        description="Run the end-to-end koharu-style pipeline on one or more "
                    "manga pages, emitting per-stage PNG artefacts.",
    )
    ap.add_argument("images", nargs="*", type=Path,
                    help="Image file paths (jpg/png). If omitted, auto-discovers "
                         "from repo training dirs (up to --limit).")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "thoughts" / "koharu-improvements" / "pipeline-e2e",
                    help="Output gallery root (default: thoughts/.../pipeline-e2e).")
    ap.add_argument("--limit", type=int, default=4,
                    help="Max images when auto-discovering (default: 4).")
    ap.add_argument("--skip-features", action="store_true",
                    help="Skip the per-feature static demos.")
    ap.add_argument("--final-only", type=Path, default=None,
                    help="Additionally copy every 11_final_composite.png into "
                         "this flat folder, named <slug>.png, for a "
                         "quick single-folder view of just the final output.")
    ap.add_argument("--lama-max-side", type=int, default=None,
                    help="If set, downsample input image+mask so that the "
                         "longer side ≤ this many px before LaMa runs, "
                         "then upsample the result back. Speeds up inpaint "
                         "dramatically on large pages. Try 768 or 1024.")
    ap.add_argument("--orphan-lines", choices=["off", "paragraph", "sentence"],
                    default="off",
                    help="Recover detector text_lines that belong to no block: "
                         "'paragraph' clusters nearby lines into one synthetic "
                         "block, 'sentence' treats each line standalone.")
    ap.add_argument("--page-concurrency", type=int, default=1,
                    help="Number of pages to process concurrently. >1 overlaps "
                         "OCR/inpaint/translate across pages so the "
                         "translation backend's concurrency is actually utilized. "
                         "Default 1 (sequential).")
    args = ap.parse_args()

    _init_fonts()
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_root}")

    # Per-feature demos (no model calls)
    if not args.skip_features:
        print("feature demos…")
        all_feature_demos(out_root / "features")

    # Pipeline runner (loads all services — slow)
    print("loading pipeline…")
    runner = PipelineRunner()
    runner.orphan_mode = args.orphan_lines

    if args.images:
        images = [p for p in args.images if p.exists()]
        missing = [p for p in args.images if not p.exists()]
        for m in missing:
            print(f"  skipping missing: {m}")
    else:
        images = discover_test_images(limit=args.limit)
    if not images:
        print("No test images found; abort.")
        return
    print(f"test images: {[p.name for p in images]}")

    if args.final_only:
        args.final_only.mkdir(parents=True, exist_ok=True)

    originals_dir = out_root / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)

    # Pre-copy all originals up front (no model work, no contention with LaMa/LLM)
    for img in images:
        try:
            dst_orig = originals_dir / img.name
            if not dst_orig.exists():
                dst_orig.write_bytes(img.read_bytes())
        except Exception as exc:
            print(f"  [{img.name}] original-copy failed: {exc}")

    summaries: list[dict] = []
    sem = asyncio.Semaphore(max(1, args.page_concurrency))

    async def _process_one(img: Path) -> dict:
        slug = img.stem.replace(" ", "_")[:40]
        async with sem:
            try:
                stats = await runner.run(img, out_root / slug,
                                         lama_max_side=args.lama_max_side)
                stats["slug"] = slug
                if args.final_only:
                    src = out_root / slug / "11_final_composite.png"
                    if src.exists():
                        dst = args.final_only / f"{slug}.png"
                        dst.write_bytes(src.read_bytes())
                return stats
            except Exception as exc:
                print(f"  [{img.name}] FAILED: {exc}")
                return {"slug": slug, "image": img.name, "error": str(exc)}

    if args.page_concurrency > 1:
        print(f"running {len(images)} pages with concurrency={args.page_concurrency}")
        summaries = list(await asyncio.gather(*(_process_one(p) for p in images)))
    else:
        for img in images:
            summaries.append(await _process_one(img))

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

    # Aggregate JP+EN text doc across all pages
    _write_aggregate_translations(out_root, summaries)

    print(f"\nGallery written to {out_root}/SUMMARY.md")
    print(f"Originals copied to {out_root}/originals/")
    print(f"Aggregate translations at {out_root}/translations.txt")


if __name__ == "__main__":
    asyncio.run(main())

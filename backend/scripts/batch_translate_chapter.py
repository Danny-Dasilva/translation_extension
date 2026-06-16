"""Batch chapter translator — render translated manga pages with the prod pipeline.

Reuses the tested end-to-end pipeline from ``visualize_e2e_pipeline.py`` but emits
ONLY the final composited page (the equivalent of "11_final_composite") as a .webp
per source page, instead of the 11 debug PNGs.

Stages (mirroring PipelineRunner.run):
    detect (CTD) -> bubble-detect (YOLOv10n) -> OCR (PARSeq) -> japanese filter
    -> build inpaint mask -> LaMa inpaint -> batched translate -> compose_final

CRITICAL: ``import app.services._ort_init`` runs FIRST (before any ONNX session is
created) so CTD/PARSeq bind CUDA. Without it they silently fall back to CPU (~50x).

Usage:
    VLLM_BASE_URL=http://127.0.0.1:8765/v1 \
    .venv/bin/python scripts/batch_translate_chapter.py \
        --input-dir "/path/to/chapter" \
        --out-dir   "/path/to/out/Part12" \
        [--limit 3]
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# --- CUDA binding MUST come first, before any ONNX session is created. ---
import app.services._ort_init  # noqa: E402,F401  (side-effect: preload CUDA libs)

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from typing import Optional  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.config import settings  # noqa: E402
from app.services.detector_factory import create_detector  # noqa: E402
from app.services.parseq_ocr_service import ParseqOCRService  # noqa: E402
from app.services.translation_text_utils import format_sources  # noqa: E402
from app.utils.japanese_text_filter import is_japanese_text  # noqa: E402
from app.utils.ocr_confidence_gate import is_garbled_low_conf  # noqa: E402
from app.utils.ctd_utils import (  # noqa: E402
    build_inpaint_mask,
    match_blocks_to_bubbles,
)
from app.utils.orphan_lines import (  # noqa: E402
    find_orphan_lines,
    cluster_orphan_lines,
    ocr_orphan_clusters_with_conf,
    cluster_bbox,
    merge_orphans_into_blocks,
)

from PIL import ImageDraw, ImageFont  # noqa: E402

try:
    from app.services.lama_inpaint_service import LamaInpaintService
except Exception:  # pragma: no cover
    LamaInpaintService = None  # type: ignore

# --- Inspection drawing helpers (reuse visualizer's box drawing + CJK fonts) ---
try:
    from scripts.visualize_e2e_pipeline import (
        draw_boxes,
        draw_label,
        _init_fonts as _viz_init_fonts,
    )
    import scripts.visualize_e2e_pipeline as _viz
except Exception:  # pragma: no cover
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "visualize_e2e_pipeline", SCRIPT_DIR / "visualize_e2e_pipeline.py"
    )
    assert _spec and _spec.loader
    _viz = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_viz)
    draw_boxes = _viz.draw_boxes
    draw_label = _viz.draw_label
    _viz_init_fonts = _viz._init_fonts

# CJK-capable font for rendering OCR Japanese in the diagnostic image. Reuse
# refit's font-coverage chain so leaked JP glyphs don't show as tofu.
try:
    from scripts.refit_final_composites import (
        FALLBACK_FONT_PATH as _CJK_FONT_PATH,
        _font_supports as _cjk_font_supports,
    )
except Exception:  # pragma: no cover
    _CJK_FONT_PATH = None
    _cjk_font_supports = None

# Does the chosen CJK font actually exist and cover Japanese? We probe with a
# couple of common kana/kanji codepoints. If unavailable, the diagnostic image
# falls back to index+EN labels and a note (JP still lands in bubbles.json).
CJK_AVAILABLE = bool(
    _CJK_FONT_PATH
    and Path(_CJK_FONT_PATH).exists()
    and (_cjk_font_supports is None or _cjk_font_supports(Path(_CJK_FONT_PATH), "あ日ン"))
)

# compose_final — single source of truth for final layout (same as visualizer).
try:
    from scripts.refit_final_composites import compose_final
except Exception:  # pragma: no cover
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "refit_final_composites", SCRIPT_DIR / "refit_final_composites.py"
    )
    assert spec and spec.loader
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    compose_final = _mod.compose_final


IMAGE_EXTS = {".webp", ".png", ".jpg", ".jpeg"}


def _load_cjk_font(size: int):
    """Load the CJK-capable font (or None if unavailable)."""
    if not CJK_AVAILABLE or _CJK_FONT_PATH is None:
        return None
    try:
        return ImageFont.truetype(str(_CJK_FONT_PATH), size)
    except Exception:
        return None


def _draw_text_block(
    draw: "ImageDraw.ImageDraw",
    x: int,
    y: int,
    lines: list[tuple[str, tuple]],
    font,
    max_w: int,
) -> int:
    """Draw stacked (text, color) lines with a dark backing box for legibility.

    Returns the y after the last drawn line. ``font`` may be None (uses a
    PIL default). ``max_w`` clips the backing box width to keep it on-page.
    """
    if font is None:
        font = ImageFont.load_default()
    cur_y = y
    for text, color in lines:
        if not text:
            continue
        try:
            bb = draw.textbbox((x, cur_y), text, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = len(text) * 7, 12
        draw.rectangle(
            (x - 2, cur_y - 1, x + min(tw, max_w) + 2, cur_y + th + 2),
            fill=(0, 0, 0),
        )
        draw.text((x, cur_y), text, fill=color, font=font)
        cur_y += th + 4
    return cur_y


def write_inspection(
    inspect_dir: Path,
    stem: str,
    source_np: np.ndarray,
    plate_np: np.ndarray,
    final_np: np.ndarray,
    all_blocks: list[dict],
    all_ocr: list[str],
    valid_idx: list[int],
    translations: list[str],
    all_ocr_conf: list[float] | None = None,
    gate_dropped: set[int] | None = None,
) -> None:
    """Emit per-page diagnostic artifacts under ``<inspect_dir>/<stem>/``.

    ``all_blocks``/``all_ocr`` are the post-orphan-merge, pre-filter aligned
    lists (block i <-> ocr i). ``valid_idx`` is the index list the Japanese
    filter kept (parallel to ``translations``). Orphan/synthetic blocks carry
    ``orphan: True``; we colour and label them distinctly.
    """
    _viz_init_fonts()
    page_dir = inspect_dir / stem
    page_dir.mkdir(parents=True, exist_ok=True)

    # 01 — source
    Image.fromarray(source_np).convert("RGB").save(
        page_dir / "01_source.webp", format="WEBP", quality=90
    )

    # 02 — detected blocks, numbered; orphan/synthetic in a distinct colour.
    detector_blocks = [b for b in all_blocks if not b.get("orphan")]
    orphan_blocks = [b for b in all_blocks if b.get("orphan")]
    idx_of = {id(b): i for i, b in enumerate(all_blocks)}
    boxes_img = draw_boxes(
        source_np,
        detector_blocks,
        color=(0, 255, 255),  # cyan = detector
        label_fn=lambda _j, b: f"#{idx_of[id(b)]}",
    )
    if orphan_blocks:
        boxes_img = draw_boxes(
            boxes_img,
            orphan_blocks,
            color=(255, 64, 255),  # magenta = orphan/synthetic
            label_fn=lambda _j, b: f"#{idx_of[id(b)]}*",
        )
    Image.fromarray(boxes_img).convert("RGB").save(page_dir / "02_blocks.png")

    # 07 — inpainted clean plate
    Image.fromarray(plate_np).convert("RGB").save(
        page_dir / "07_inpaint.webp", format="WEBP", quality=90
    )

    # 11 — final rendered page (same as normal output)
    Image.fromarray(final_np).convert("RGB").save(
        page_dir / "11_final.webp", format="WEBP", quality=90
    )

    # Build the per-bubble records (kept first, then filtered-out).
    kept_pos = {bi: k for k, bi in enumerate(valid_idx)}
    gate_dropped = gate_dropped or set()
    bubbles: list[dict] = []
    for bi, block in enumerate(all_blocks):
        ocr_jp = all_ocr[bi] if bi < len(all_ocr) else ""
        is_kept = bi in kept_pos
        trans = (
            translations[kept_pos[bi]]
            if is_kept and kept_pos[bi] < len(translations)
            else None
        )
        ocr_conf = (
            all_ocr_conf[bi]
            if all_ocr_conf is not None and bi < len(all_ocr_conf)
            else None
        )
        bubbles.append(
            {
                "idx": bi,
                "bbox": {
                    "minX": int(block["minX"]),
                    "minY": int(block["minY"]),
                    "maxX": int(block["maxX"]),
                    "maxY": int(block["maxY"]),
                },
                "ocr_jp": ocr_jp,
                "translation_en": trans if is_kept else None,
                "is_orphan": bool(block.get("orphan", False)),
                "confidence": block.get("confidence"),
                "ocr_conf": round(ocr_conf, 4) if ocr_conf is not None else None,
                "ocr_gate_dropped": bi in gate_dropped,
                "filtered": not is_kept,
            }
        )

    # 10 — OCR + translation diagnostic (the most important image). Draw on a
    # copy of the SOURCE so the original JP context is visible next to OCR/EN.
    diag = Image.fromarray(source_np).convert("RGB")
    ddraw = ImageDraw.Draw(diag)
    W, _H = diag.size
    cjk_font = _load_cjk_font(16)
    en_font = _load_cjk_font(15) or ImageFont.load_default()
    for rec in bubbles:
        if rec["filtered"]:
            continue  # only annotate kept bubbles on the image
        bb = rec["bbox"]
        x0, y0, x1, y1 = bb["minX"], bb["minY"], bb["maxX"], bb["maxY"]
        color = (255, 64, 255) if rec["is_orphan"] else (0, 255, 255)
        ddraw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        lines: list[tuple[str, tuple]] = [(f"#{rec['idx']}", color)]
        if CJK_AVAILABLE and rec["ocr_jp"]:
            lines.append((f"JP: {rec['ocr_jp']}", (255, 230, 120)))
        en = rec["translation_en"] or ""
        lines.append((f"EN: {en}", (120, 255, 120)))
        tx = min(x0 + 2, W - 200)
        ty = min(y1 + 2, diag.size[1] - 60)
        _draw_text_block(
            ddraw, tx, ty, lines, cjk_font if CJK_AVAILABLE else en_font, max_w=W - tx
        )
    if not CJK_AVAILABLE:
        note = "NO CJK FONT — JP omitted from image (see bubbles.json)"
        ddraw.rectangle((4, 4, 8 + len(note) * 8, 24), fill=(0, 0, 0))
        ddraw.text((8, 6), note, fill=(255, 80, 80), font=en_font)
    diag.save(page_dir / "10_ocr_translate.png")

    # bubbles.json — downstream analysis (OCR vs translation error separation)
    (page_dir / "bubbles.json").write_text(
        json.dumps(bubbles, indent=2, ensure_ascii=False)
    )


class ChapterPipeline:
    """Loads all services ONCE, then renders one final translated page per call."""

    def __init__(self):
        print("  loading detector (CTD)…")
        self.detector = create_detector()
        self._verify_ctd_provider()

        self.bubble_detector = None
        try:
            from app.services.detector_service import DetectorService

            if Path(BACKEND_DIR, settings.yolo_model_path).exists():
                print("  loading bubble detector (YOLOv10n)…")
                self.bubble_detector = DetectorService()
        except Exception as exc:
            print(f"  bubble detector unavailable ({exc}); typeset to text blocks")

        print("  loading OCR (PARSeq)…")
        self.ocr = ParseqOCRService(
            model_path=settings.parseq_model_path,
            hybrid_enabled=getattr(settings, "hybrid_ocr_enabled", False),
            ar_model_path=getattr(settings, "parseq_ar_model_path", None),
            hybrid_conf_threshold=getattr(settings, "ocr_confidence_gate_threshold", 0.65),
        )

        self.lama = None
        if LamaInpaintService is not None and Path(
            BACKEND_DIR, settings.lama_model_path
        ).exists():
            print("  loading LaMa…")
            try:
                self.lama = LamaInpaintService(model_path=settings.lama_model_path)
            except Exception as exc:
                print(f"  LaMa failed to load: {exc}; proceeding without plate")
        else:
            print("  LaMa unavailable — plate stage will be skipped")

        self.translator: Optional[object] = None
        try:
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
        except Exception as exc:
            print(f"  LLM failed to load: {exc}; translations will be empty")

        lama_device = getattr(self.lama, "device", None) if self.lama else "unavailable"
        print(
            f"  devices: lama={lama_device} "
            f"translator={type(self.translator).__name__ if self.translator else 'unavailable'}"
        )

    def _verify_ctd_provider(self) -> None:
        """Hard-warn if CTD did not bind CUDA (CPU fallback is ~50x slower)."""
        session = getattr(self.detector, "session", None)
        providers = []
        if session is not None and hasattr(session, "get_providers"):
            try:
                providers = list(session.get_providers())
            except Exception:
                providers = []
        active = providers[0] if providers else "unknown"
        self.ctd_provider = active
        if active == "CUDAExecutionProvider":
            print(f"  CTD provider: {active}  [GPU OK]")
        else:
            print(
                "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                f"  !! HARD WARNING: CTD provider is {active!r}, NOT CUDA.\n"
                f"  !! Providers seen: {providers}\n"
                "  !! ONNX fell back to CPU (~50x slower). Check _ort_init / CUDA libs.\n"
                "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

    async def render_page(
        self,
        image_path: Path,
        out_path: Path,
        inspect_dir: Optional[Path] = None,
    ) -> dict:
        """Run the full pipeline; write ONE final translated .webp. Returns stats."""
        stats: dict = {"image": image_path.name}
        t_page = time.time()

        pil = Image.open(image_path).convert("RGB")
        image_np = np.array(pil)

        # --- detect (CTD: blocks + text_lines + mask) ---
        t0 = time.time()
        ctd = await self.detector.detect(image_np)
        stats["detect_ms"] = (time.time() - t0) * 1000
        blocks = ctd["blocks"]
        text_lines = ctd["text_lines"]
        mask = ctd.get("mask")
        stats["num_blocks"] = len(blocks)

        # --- bubble detection (typeset to bubble interior) ---
        bubbles = []
        if self.bubble_detector is not None:
            try:
                bubbles = await self.bubble_detector.detect_bubbles(image_np)
            except Exception as exc:
                print(f"  [{image_path.name}] bubble detect failed ({exc})")
        stats["num_bubbles"] = len(bubbles)

        # --- OCR (with per-block recognition confidence for the garble gate) ---
        t0 = time.time()
        if text_lines:
            ocr_texts, ocr_confs = await self.ocr.recognize_blocks_with_lines(
                image_np, blocks, text_lines,
                batch_size=settings.parseq_batch_size,
                return_confidence=True,
            )
        else:
            crops = self.detector.crop_regions(image_np, blocks)
            tc = await self.ocr.recognize_text_batch_with_conf(crops)
            ocr_texts = [t for t, _c in tc]
            ocr_confs = [c for _t, c in tc]
        stats["ocr_ms"] = (time.time() - t0) * 1000

        # --- Orphan-line recovery (same wiring as the production router) ---
        if settings.orphan_line_recovery and text_lines:
            orphans = find_orphan_lines(blocks, text_lines)
            if orphans:
                clusters = cluster_orphan_lines(orphans)
                synth_ocr, synth_conf = await ocr_orphan_clusters_with_conf(
                    self.ocr, image_np, clusters,
                    batch_size=settings.parseq_batch_size,
                )
                synth_blocks: list[dict] = []
                synth_texts: list[str] = []
                synth_confs: list[float] = []
                for cluster, text, c in zip(clusters, synth_ocr, synth_conf):
                    if not text.strip():
                        continue
                    synth_blocks.append(cluster_bbox(cluster))
                    synth_texts.append(text)
                    synth_confs.append(c)
                n_before = len(blocks)
                blocks, ocr_texts = merge_orphans_into_blocks(
                    blocks, ocr_texts, synth_blocks, synth_texts
                )
                # Keep ocr_confs aligned with the merged blocks/ocr_texts. New
                # orphan blocks are APPENDED at the end in order, so the trailing
                # confs map 1:1 to the appended synth blocks; this carries their
                # REAL OCR recognition confidence so the garble gate can drop
                # low-conf garbled orphan SFX (e.g. よっピ / こちにちこち).
                appended = len(ocr_texts) - len(ocr_confs)
                if appended > 0:
                    # The last `appended` synth blocks were appended (not merged);
                    # take their confidences from the tail of synth_confs.
                    tail = synth_confs[-appended:] if appended <= len(synth_confs) else synth_confs
                    ocr_confs = ocr_confs + list(tail)
                    while len(ocr_confs) < len(ocr_texts):
                        ocr_confs.append(0.0)
                added = len(blocks) - n_before
                merged = len(synth_blocks) - added
                if synth_blocks:
                    print(
                        f"  [{image_path.name}] orphan-line recovery: "
                        f"{len(orphans)} line(s) -> {len(synth_blocks)} cluster(s) "
                        f"=> {added} new block(s), {merged} merged into originals "
                        f"(blocks {n_before} -> {len(blocks)})"
                    )
                stats["orphan_blocks_added"] = added
                stats["orphan_blocks_merged"] = merged

        # --- Japanese filter + OCR-confidence garble gate ---
        # Keep a bubble only if it is Japanese AND not (low-OCR-confidence AND
        # garbled). The gate stops hallucinated captions on stylized SFX/scrawl
        # before they reach the LLM, while leaving high-confidence text alone.
        gate_on = (
            getattr(settings, "ocr_confidence_gate_enabled", False)
            and getattr(settings, "ocr_confidence_gate_threshold", 0.0) > 0
        )
        gate_thresh = getattr(settings, "ocr_confidence_gate_threshold", 0.65)
        from app.services.lama_inpaint_service import is_leave_intact_label
        valid_idx: list[int] = []
        gate_dropped: list[int] = []
        for i, t in enumerate(ocr_texts):
            if not is_japanese_text(
                t,
                settings.japanese_filter_min_ratio,
                settings.japanese_filter_katakana_max_length,
            ):
                continue
            if is_leave_intact_label(t):
                # Editorial/margin label (e.g. 表紙用イラスト, 奥付) — leave as
                # original art: do not erase, translate, or typeset over it
                # (matches human handling; avoids erase-without-replace smear).
                print(
                    f"  [{image_path.name}] leave-intact label idx {i}: {t[:18]!r}"
                )
                continue
            conf = ocr_confs[i] if i < len(ocr_confs) else 1.0
            if gate_on and is_garbled_low_conf(t, conf, conf_threshold=gate_thresh):
                gate_dropped.append(i)
                print(
                    f"  [{image_path.name}] OCR-gate dropped idx {i} "
                    f"(conf={conf:.2f}) {t[:24]!r} — garbled, not sent to LLM"
                )
                continue
            valid_idx.append(i)
        kept_blocks = [blocks[i] for i in valid_idx]
        kept_texts = [ocr_texts[i] for i in valid_idx]
        kept_confs = [ocr_confs[i] if i < len(ocr_confs) else 1.0 for i in valid_idx]
        stats["num_kept"] = len(kept_texts)
        stats["ocr_gate_dropped"] = len(gate_dropped)

        # --- inpaint mask from KEPT blocks only ---
        inpaint_mask = build_inpaint_mask(image_np.shape, kept_blocks, text_lines, mask)

        # --- LaMa inpaint ---
        inpainted: Optional[np.ndarray] = None
        if self.lama is not None:
            t0 = time.time()
            try:
                inpainted = await asyncio.to_thread(
                    self.lama.inpaint, image_np, inpaint_mask
                )
                stats["inpaint_ms"] = (time.time() - t0) * 1000
            except Exception as exc:
                print(f"  [{image_path.name}] inpaint failed: {exc}")
                inpainted = None
        if inpainted is None:
            inpainted = image_np
            stats.setdefault("inpaint_ms", 0.0)

        # --- page-level numbered-block translate (intra-page context +
        # system-prompt language lock); falls back to per-bubble on mismatch ---
        translations: list[str] = []
        if self.translator is not None and kept_texts:
            t0 = time.time()
            translations = []
            if (
                settings.batch_translate
                and len(kept_texts) > 1
                and hasattr(self.translator, "translate_numbered_block")
            ):
                try:
                    blocked = await self.translator.translate_numbered_block(
                        kept_texts, "English"
                    )
                    if len(blocked) == len(kept_texts) and any(
                        b.strip() for b in blocked
                    ):
                        translations = blocked
                    else:
                        print(
                            f"  [{image_path.name}] numbered-block produced "
                            "empty/mismatched output; falling back to per-bubble"
                        )
                except Exception as exc:
                    print(
                        f"  [{image_path.name}] numbered-block translate failed "
                        f"({exc}); falling back to per-bubble"
                    )
            if not translations:
                try:
                    translations = await self.translator.translate_batched(
                        kept_texts, "English"
                    )
                except Exception as exc:
                    print(
                        f"  [{image_path.name}] batched translate failed ({exc}); "
                        "falling back to sequential"
                    )
                    translations = []
                    for t in kept_texts:
                        try:
                            tr = await self.translator.translate_single(t, "English")
                        except Exception:
                            tr = ""
                        translations.append(tr)
            # Post-translation glossaries (register/names/SFX) — shared with the
            # API router so both paths render identical corrected text.
            if translations:
                from app.services.translation_postedit import (
                    apply_postedit_glossaries,
                )
                translations = apply_postedit_glossaries(translations, kept_texts)
            stats["translate_ms"] = (time.time() - t0) * 1000
        else:
            stats["translate_ms"] = 0.0

        # --- final composite ---
        fit_rects = (
            match_blocks_to_bubbles(kept_blocks, bubbles) if bubbles else None
        )
        final = compose_final(
            inpainted,
            list(kept_blocks),
            list(translations),
            fit_rects=list(fit_rects) if fit_rects is not None else None,
        )

        # write as .webp (quality ~90), same basename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(final).convert("RGB").save(
            out_path, format="WEBP", quality=90
        )

        # --- inspection artifacts (additive; identical code path) ---
        if inspect_dir is not None:
            try:
                write_inspection(
                    inspect_dir=inspect_dir,
                    stem=image_path.stem,
                    source_np=image_np,
                    plate_np=inpainted,
                    final_np=final,
                    all_blocks=list(blocks),
                    all_ocr=list(ocr_texts),
                    valid_idx=list(valid_idx),
                    translations=list(translations),
                    all_ocr_conf=list(ocr_confs),
                    gate_dropped=set(gate_dropped),
                )
                stats["inspect_dir"] = str(inspect_dir / image_path.stem)
            except Exception as exc:
                print(f"  [{image_path.name}] inspection write failed: {exc}")

        stats["total_ms"] = (time.time() - t_page) * 1000
        stats["out_path"] = str(out_path)
        stats["out_bytes"] = out_path.stat().st_size if out_path.exists() else 0
        stats["ocr_samples"] = kept_texts[:4]
        stats["translation_samples"] = translations[:4]
        return stats


def _filter_pages(pages: list[Path], spec: str) -> list[Path]:
    """Keep only pages whose stem matches a token in ``spec``.

    Tokens may be raw page numbers ("5"), zero-padded ("005"), or full
    basenames ("005.webp"). Numeric tokens match by integer value against the
    leading digits of each page stem, so "5" -> 005.webp. Non-numeric tokens
    match the stem (case-insensitive) directly.
    """
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    num_tokens: set[int] = set()
    str_tokens: set[str] = set()
    for t in tokens:
        base = Path(t).stem  # strip any extension
        if base.isdigit():
            num_tokens.add(int(base))
        else:
            str_tokens.add(base.lower())

    def matches(p: Path) -> bool:
        stem = p.stem
        if stem.lower() in str_tokens:
            return True
        lead = "".join(ch for ch in stem if ch.isdigit())
        if lead and int(lead) in num_tokens:
            return True
        return False

    return [p for p in pages if matches(p)]


def discover_pages(input_dir: Path) -> list[Path]:
    pages = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(pages, key=lambda p: p.name)


def _summarize(per_page: list[dict], wall_clock_s: float, n_pages: int) -> dict:
    def vals(key: str) -> list[float]:
        return [s[key] for s in per_page if isinstance(s.get(key), (int, float))]

    def med(key: str) -> float:
        v = vals(key)
        return statistics.median(v) if v else 0.0

    totals = vals("total_ms")
    p95 = (
        statistics.quantiles(totals, n=20)[18]
        if len(totals) >= 2
        else (totals[0] if totals else 0.0)
    )
    return {
        "n_pages": n_pages,
        "wall_clock_s": round(wall_clock_s, 2),
        "total_ms": {
            "median": round(med("total_ms"), 1),
            "mean": round(statistics.mean(totals), 1) if totals else 0.0,
            "p95": round(p95, 1),
        },
        "stage_ms_median": {
            "detect": round(med("detect_ms"), 1),
            "ocr": round(med("ocr_ms"), 1),
            "translate": round(med("translate_ms"), 1),
            "inpaint": round(med("inpaint_ms"), 1),
        },
        "per_page": per_page,
    }


def _print_summary(label: str, summ: dict) -> None:
    print(f"\n===== {label} benchmark =====")
    print(f"pages:        {summ['n_pages']}")
    print(f"wall-clock:   {summ['wall_clock_s']:.1f} s")
    t = summ["total_ms"]
    print(
        f"per-page ms:  median={t['median']:.0f}  mean={t['mean']:.0f}  p95={t['p95']:.0f}"
    )
    s = summ["stage_ms_median"]
    print(
        f"stage median: detect={s['detect']:.0f}  ocr={s['ocr']:.0f}  "
        f"translate={s['translate']:.0f}  inpaint={s['inpaint']:.0f}"
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-translate a manga chapter.")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="Process first N pages.")
    ap.add_argument(
        "--bench-out",
        type=Path,
        default=None,
        help="Write benchmark JSON here (optional).",
    )
    ap.add_argument(
        "--inspect-dir",
        type=Path,
        default=None,
        help="If set, write per-page inspection artifacts under <dir>/<stem>/.",
    )
    ap.add_argument(
        "--pages",
        type=str,
        default=None,
        help=(
            "Comma-separated source page numbers or basenames to process ONLY "
            "(e.g. '3,5,6' or '005'). Matched by zero-padded stem against the "
            "discovered page stems. If absent, process all."
        ),
    )
    args = ap.parse_args()

    pages = discover_pages(args.input_dir)
    if args.pages:
        pages = _filter_pages(pages, args.pages)
        if not pages:
            print(f"--pages {args.pages!r} matched no source images; abort.")
            return
    if args.limit:
        pages = pages[: args.limit]
    if not pages:
        print(f"No source images in {args.input_dir}; abort.")
        return
    print(f"input:  {args.input_dir}  ({len(pages)} pages)")
    print(f"output: {args.out_dir}")

    print("loading pipeline…")
    pipe = ChapterPipeline()

    per_page: list[dict] = []
    errors: list[dict] = []
    t_wall = time.time()
    for i, src in enumerate(pages, 1):
        out_path = args.out_dir / (src.stem + ".webp")
        try:
            stats = await pipe.render_page(src, out_path, inspect_dir=args.inspect_dir)
            per_page.append(stats)
            samp = ""
            if stats.get("translation_samples"):
                samp = "  e.g. " + " | ".join(
                    f"{j[:18]}->{e[:24]}"
                    for j, e in zip(
                        stats.get("ocr_samples", []),
                        stats.get("translation_samples", []),
                    )
                )
            print(
                f"  [{i}/{len(pages)}] {src.name} -> {out_path.name}  "
                f"{stats['total_ms']:.0f}ms  kept={stats['num_kept']}  "
                f"{stats['out_bytes'] // 1024}KB{samp}"
            )
            if stats["num_kept"] == 0:
                print(f"      (note) {src.name}: no Japanese bubbles kept")
        except Exception as exc:
            print(f"  [{i}/{len(pages)}] {src.name} FAILED: {exc}")
            errors.append({"image": src.name, "error": str(exc)})

    wall = time.time() - t_wall
    summ = _summarize(per_page, wall, len(pages))
    summ["errors"] = errors
    summ["ctd_provider"] = getattr(pipe, "ctd_provider", "unknown")
    _print_summary(args.out_dir.name, summ)

    if args.bench_out:
        args.bench_out.parent.mkdir(parents=True, exist_ok=True)
        args.bench_out.write_text(json.dumps(summ, indent=2, ensure_ascii=False))
        print(f"benchmark -> {args.bench_out}")

    if errors:
        print(f"\n{len(errors)} page(s) errored:")
        for e in errors:
            print(f"  - {e['image']}: {e['error']}")


if __name__ == "__main__":
    asyncio.run(main())

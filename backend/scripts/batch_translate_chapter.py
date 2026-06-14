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
from app.utils.ctd_utils import (  # noqa: E402
    build_inpaint_mask,
    match_blocks_to_bubbles,
)
from app.utils.orphan_lines import (  # noqa: E402
    find_orphan_lines,
    cluster_orphan_lines,
    ocr_orphan_clusters,
    cluster_bbox,
)

try:
    from app.services.lama_inpaint_service import LamaInpaintService
except Exception:  # pragma: no cover
    LamaInpaintService = None  # type: ignore

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
        self.ocr = ParseqOCRService(model_path=settings.parseq_model_path)

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

    async def render_page(self, image_path: Path, out_path: Path) -> dict:
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

        # --- OCR ---
        t0 = time.time()
        if text_lines:
            ocr_texts = await self.ocr.recognize_blocks_with_lines(
                image_np, blocks, text_lines, batch_size=settings.parseq_batch_size
            )
        else:
            crops = self.detector.crop_regions(image_np, blocks)
            ocr_texts = await self.ocr.recognize_text_batch(crops)
        stats["ocr_ms"] = (time.time() - t0) * 1000

        # --- Orphan-line recovery (same wiring as the production router) ---
        if settings.orphan_line_recovery and text_lines:
            orphans = find_orphan_lines(blocks, text_lines)
            if orphans:
                clusters = cluster_orphan_lines(orphans)
                synth_texts = await ocr_orphan_clusters(
                    self.ocr, image_np, clusters,
                    batch_size=settings.parseq_batch_size,
                )
                added = 0
                for cluster, text in zip(clusters, synth_texts):
                    if not text.strip():
                        continue
                    blocks.append(cluster_bbox(cluster))
                    ocr_texts.append(text)
                    added += 1
                if added:
                    print(
                        f"  [{image_path.name}] orphan-line recovery: "
                        f"{len(orphans)} line(s) -> {added} synthetic block(s)"
                    )
                stats["orphan_blocks_added"] = added

        # --- Japanese filter (same args as visualize run()) ---
        valid_pairs = [
            (i, t)
            for i, t in enumerate(ocr_texts)
            if is_japanese_text(
                t,
                settings.japanese_filter_min_ratio,
                settings.japanese_filter_katakana_max_length,
            )
        ]
        valid_idx = [p[0] for p in valid_pairs]
        kept_blocks = [blocks[i] for i in valid_idx]
        kept_texts = [p[1] for p in valid_pairs]
        stats["num_kept"] = len(kept_texts)

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

        stats["total_ms"] = (time.time() - t_page) * 1000
        stats["out_path"] = str(out_path)
        stats["out_bytes"] = out_path.stat().st_size if out_path.exists() else 0
        stats["ocr_samples"] = kept_texts[:4]
        stats["translation_samples"] = translations[:4]
        return stats


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
    args = ap.parse_args()

    pages = discover_pages(args.input_dir)
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
            stats = await pipe.render_page(src, out_path)
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

"""End-to-end per-page / per-gallery orchestration.

    detect+OCR (both sides)  ->  align  ->  curate  ->  format page-context rows

``align_and_curate`` is the engine and is OCR-AGNOSTIC: it takes already-built
JP and EN bubble lists, so the SAME code path is exercised by the Ikenie-gold
validation (gold text+bbox fed as "OCR output") and by the real OCR runner.
"""
from __future__ import annotations

from dataclasses import dataclass

from align import AlignConfig, align_pages, coverage  # type: ignore
from curate import CurationConfig, CurationStats, curate_pair  # type: ignore
from format_rows import build_pagectx_rows  # type: ignore
from geometry import reading_order  # type: ignore


@dataclass
class PipelineConfig:
    align: AlignConfig
    curate: CurationConfig
    register_tag: str = "manga_nsfw"
    also_plain: bool = False

    @classmethod
    def default(cls) -> "PipelineConfig":
        return cls(align=AlignConfig(), curate=CurationConfig())


def align_and_curate(
    jp_bubbles: list[dict],
    en_bubbles: list[dict],
    gid_tag: str,
    page: int,
    cfg: PipelineConfig,
    stats: CurationStats,
    seen_keys: set,
) -> tuple[list[dict], list]:
    """Align + curate + format ONE page pair.

    Returns ``(rows, kept_pairs)``. ``rows`` are training-schema dicts; the whole
    page is dropped (no rows) when alignment coverage is below the threshold.
    """
    matches, _jp_un, _en_un = align_pages(jp_bubbles, en_bubbles, cfg.align)
    if not matches:
        return [], []
    cov = coverage(matches, jp_bubbles, en_bubbles)
    if cov < cfg.curate.min_coverage:
        stats.dropped_pages += 1
        return [], []

    # Page context = non-empty JP lines in manga reading order (serving-exact).
    ordered_nonempty = [b for b in reading_order(jp_bubbles) if (b.get("text") or "").strip()]
    pos_of = {id(b): i for i, b in enumerate(ordered_nonempty)}
    ordered_page_jp = [(b["text"]).strip() for b in ordered_nonempty]

    kept_pairs = []
    targets: list[tuple[int, str, str]] = []
    for m in matches:
        jb = jp_bubbles[m.jp_idx]
        eb = en_bubbles[m.en_idx]
        pos = pos_of.get(id(jb))
        if pos is None:  # JP bubble had empty text -> not a usable target
            continue
        cp = curate_pair(
            jp_text=jb.get("text", ""),
            en_text=eb.get("text", ""),
            jp_bbox=jb["bbox"],
            en_bbox=eb["bbox"],
            match_dist=m.match_dist,
            ocr_conf=float(jb.get("conf") or 0.0),
            page=page,
            page_coverage=cov,
            cfg=cfg.curate,
            seen_keys=seen_keys,
            stats=stats,
        )
        if cp.drop_reason is None:
            kept_pairs.append(cp)
            targets.append((pos, ordered_page_jp[pos], cp.en_tgt))

    rows = build_pagectx_rows(
        ordered_page_jp, targets, gid_tag, page,
        register_tag=cfg.register_tag, also_plain=cfg.also_plain,
    )
    return rows, kept_pairs


async def process_pair_from_images(
    jp_path, en_path, gid_tag: str, page: int, cfg: PipelineConfig,
    detector, ocr, vlm_endpoint: str, vlm_coord_norm: int,
    stats: CurationStats, seen_keys: set,
) -> list[dict]:
    """OCR both page images, then align+curate+format. EN side hits the VLM
    server (deferred until GPU is free)."""
    from ocr_adapters import ocr_jp_page, transcribe_en_page  # type: ignore

    jp_bubbles = await ocr_jp_page(jp_path, detector, ocr)
    en_bubbles = transcribe_en_page(en_path, endpoint=vlm_endpoint, coord_norm=vlm_coord_norm)
    rows, _ = align_and_curate(jp_bubbles, en_bubbles, gid_tag, page, cfg, stats, seen_keys)
    return rows

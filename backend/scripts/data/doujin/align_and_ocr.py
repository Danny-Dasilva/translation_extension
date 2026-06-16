"""Stages 2-4 for a SINGLE matched pair (or small batch).

    2. page alignment  — phash nearest-neighbour (+ optional AKAZE homography)
    3. bubble detect+OCR — comic-text-detector bboxes; PARSeq for JP, Latin OCR
                           (easyocr/stub) for the homography-transferred EN bbox
    4. pair emission    — per-bubble (jp, en) in manga reading order, carrying a
                          ``src`` = doujin:{workid}:p{page}:b{idx}

Reuses the production services:
    app/services/detector_service.py   (bubble bboxes)
    app/services/parseq_ocr_service.py (JP manga OCR)

Everything model-touching is async (matches the service APIs) and lazy: importing
this module does NOT load any model. Latin OCR degrades to a stub when easyocr
isn't installed — JP side stays fully functional, EN side emits "" and the
orchestrator reports the loss.

Output: a list of bubble-pair dicts (jp, en, src, workid, page, idx, ...) ready
to be turned into unified-schema rows by build_doujin_pairs.py.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# repo root on path so `app.services...` imports resolve
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from doujin_common import format_src, manga_reading_order  # noqa: E402
from doujin_vision import (  # noqa: E402
    align_pages_by_phash,
    estimate_homography,
    get_latin_ocr,
    list_page_images,
    phash,
    transfer_bbox,
)


@dataclass
class BubblePair:
    workid: str
    page: int
    idx: int
    jp: str
    en: str
    jp_bbox: tuple[int, int, int, int]
    en_bbox: tuple[int, int, int, int]
    align_distance: int = 0

    @property
    def src(self) -> str:
        return format_src(self.workid, self.page, self.idx)


@dataclass
class PairResult:
    workid: str
    pages_aligned: int = 0
    bubbles: list[BubblePair] = field(default_factory=list)
    en_ocr_available: bool = True


# --------------------------------------------------------------------------- #
# lazy service singletons (loaded once per process)
# --------------------------------------------------------------------------- #

_detector = None
_parseq = None
_latin = None


def _get_detector():
    global _detector
    if _detector is None:
        from app.services.detector_service import DetectorService

        _detector = DetectorService()
    return _detector


def _get_parseq():
    global _parseq
    if _parseq is None:
        from app.services.parseq_ocr_service import ParseqOCRService

        _parseq = ParseqOCRService()
    return _parseq


def _get_latin():
    global _latin
    if _latin is None:
        _latin = get_latin_ocr()
    return _latin


def _read_bgr(path: Path) -> np.ndarray | None:
    import cv2

    return cv2.imread(str(path))


async def process_pair(
    jp_dir: Path,
    en_dir: Path,
    workid: str,
    *,
    max_pages: int = 4,
    max_align_distance: int = 14,
    use_homography: bool = True,
    detector_conf: float = 0.3,
) -> PairResult:
    """Run stages 2-4 on one matched pair. Returns a :class:`PairResult`.

    Bounded by ``max_pages`` so a smoke run never opens a whole 200-page work.
    """
    import cv2

    jp_pages = list_page_images(jp_dir)[:max_pages]
    en_pages = list_page_images(en_dir)[:max_pages]
    if not jp_pages or not en_pages:
        return PairResult(workid=workid)

    # --- stage 2: page alignment by cover/page phash ----------------------- #
    jp_imgs = [_read_bgr(p) for p in jp_pages]
    en_imgs = [_read_bgr(p) for p in en_pages]
    jp_hashes = [phash(im) for im in jp_imgs if im is not None]
    en_hashes = [phash(im) for im in en_imgs if im is not None]
    alignments = align_pages_by_phash(jp_hashes, en_hashes, max_align_distance)

    detector = _get_detector()
    parseq = _get_parseq()
    latin = _get_latin()
    from doujin_vision import StubLatinOCR

    result = PairResult(
        workid=workid,
        pages_aligned=len(alignments),
        en_ocr_available=not isinstance(latin, StubLatinOCR),
    )

    for page_no, al in enumerate(alignments):
        jp_img = jp_imgs[al.jp_page]
        en_img = en_imgs[al.en_page]
        if jp_img is None or en_img is None:
            continue

        # --- stage 3: detect bubbles on the JP page ------------------------ #
        boxes = await detector.detect_bubbles(jp_img, conf=detector_conf)
        if not boxes:
            continue

        # reading order (right-to-left, column-major)
        ro = manga_reading_order(
            [
                {
                    "xmin": b["minX"],
                    "ymin": b["minY"],
                    "xmax": b["maxX"],
                    "ymax": b["maxY"],
                }
                for b in boxes
            ]
        )

        # homography JP-page -> EN-page for bbox transfer
        H = estimate_homography(jp_img, en_img) if use_homography else None

        # JP OCR: crop bubbles, batch through PARSeq
        jp_crops = []
        for r in ro:
            x0, y0, x1, y1 = r["xmin"], r["ymin"], r["xmax"], r["ymax"]
            jp_crops.append(jp_img[max(0, y0) : y1, max(0, x0) : x1])
        jp_texts = await parseq.recognize_text_batch(jp_crops)

        for idx, (r, jp_text) in enumerate(zip(ro, jp_texts)):
            jp_bbox = (r["xmin"], r["ymin"], r["xmax"], r["ymax"])
            en_bbox = transfer_bbox(jp_bbox, H)
            ex0, ey0, ex1, ey1 = en_bbox
            eh, ew = en_img.shape[:2]
            ex0, ey0 = max(0, ex0), max(0, ey0)
            ex1, ey1 = min(ew, ex1), min(eh, ey1)
            en_text = ""
            if ex1 > ex0 and ey1 > ey0:
                en_crop = en_img[ey0:ey1, ex0:ex1]
                if en_crop.size:
                    en_text = latin.read(en_crop)
            result.bubbles.append(
                BubblePair(
                    workid=workid,
                    page=page_no,
                    idx=idx,
                    jp=(jp_text or "").strip(),
                    en=(en_text or "").strip(),
                    jp_bbox=jp_bbox,
                    en_bbox=en_bbox,
                    align_distance=al.distance,
                )
            )
    return result


# --------------------------------------------------------------------------- #
# CLI: process one pair from explicit dirs
# --------------------------------------------------------------------------- #


def main() -> int:
    import argparse
    import asyncio

    ap = argparse.ArgumentParser(description="Stages 2-4 for one doujin pair")
    ap.add_argument("--jp-dir", type=Path, required=True)
    ap.add_argument("--en-dir", type=Path, required=True)
    ap.add_argument("--workid", required=True)
    ap.add_argument("--max-pages", type=int, default=4)
    ap.add_argument("--no-homography", action="store_true")
    args = ap.parse_args()

    res = asyncio.run(
        process_pair(
            args.jp_dir,
            args.en_dir,
            args.workid,
            max_pages=args.max_pages,
            use_homography=not args.no_homography,
        )
    )
    print(f"workid={res.workid} pages_aligned={res.pages_aligned} "
          f"bubbles={len(res.bubbles)} en_ocr={res.en_ocr_available}")
    for b in res.bubbles[:10]:
        print(f"  {b.src}  jp={b.jp!r}  en={b.en!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

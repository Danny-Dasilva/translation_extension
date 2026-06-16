"""Vision helpers: perceptual hashing, page alignment, homography, Latin OCR.

Heavy deps (cv2 / PIL / numpy) live here so that ``doujin_common`` stays
import-light for tests. Every external model is loaded lazily and degrades
gracefully: if a dependency is missing the function returns ``None`` / a clear
``DependencyMissing`` and the caller can skip that stage.

Latin-OCR dependency choice
---------------------------
``easyocr`` is the chosen Latin OCR engine: it is pip-installable, GPU-capable
(reuses the torch already in pyproject), and handles the curved/handwritten
fonts common in scanlation bubbles better than tesseract. It is NOT yet in
pyproject — :func:`get_latin_ocr` raises ``DependencyMissing("easyocr")`` if it
isn't importable so the orchestrator can stub it. To enable for real:

    backend/.venv/bin/pip install easyocr

(PaddleOCR weights already ship under app/weights/paddleocr-vl/ and could be a
future swap-in via the same LatinOCR interface.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class DependencyMissing(RuntimeError):
    """Raised when an optional vision dependency isn't importable."""


# --------------------------------------------------------------------------- #
# perceptual hash (self-contained — does NOT require the `imagehash` package)
# --------------------------------------------------------------------------- #


def phash(image: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """64-bit DCT perceptual hash of a grayscale/BGR image.

    Pure numpy + cv2 DCT — avoids the missing ``imagehash`` dep. Returns an int
    whose ``hamming_distance`` to another phash measures visual similarity.
    """
    import cv2

    img_size = hash_size * highfreq_factor
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:].flatten())  # exclude DC term from the median
    bits = (low > med).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming_distance(a: int, b: int) -> int:
    """Bit-count of the XOR — lower == more visually similar (0..64)."""
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- #
# page alignment
# --------------------------------------------------------------------------- #


@dataclass
class PageAlignment:
    jp_page: int
    en_page: int
    distance: int  # phash hamming distance (lower = better)


def align_pages_by_phash(
    jp_hashes: list[int],
    en_hashes: list[int],
    max_distance: int = 14,
) -> list[PageAlignment]:
    """Greedy 1:1 page alignment by phash nearest-neighbour.

    Pure-int logic (no image deps) so it is independently testable. JP pages are
    matched to their closest unused EN page; pairs above ``max_distance`` are
    dropped (handles inserted/removed credit pages). Returns alignments sorted
    by jp_page.
    """
    used_en: set[int] = set()
    out: list[PageAlignment] = []
    for jp_idx, jh in enumerate(jp_hashes):
        best_en = -1
        best_d = max_distance + 1
        for en_idx, eh in enumerate(en_hashes):
            if en_idx in used_en:
                continue
            d = hamming_distance(jh, eh)
            if d < best_d:
                best_d = d
                best_en = en_idx
        if best_en >= 0 and best_d <= max_distance:
            used_en.add(best_en)
            out.append(PageAlignment(jp_page=jp_idx, en_page=best_en, distance=best_d))
    out.sort(key=lambda a: a.jp_page)
    return out


def estimate_homography(jp_img: np.ndarray, en_img: np.ndarray):
    """AKAZE + RANSAC homography mapping JP-page coords -> EN-page coords.

    Returns a 3x3 ``np.ndarray`` or ``None`` if too few inliers (e.g. heavily
    redrawn pages). Used to transfer a JP bubble bbox onto the EN scan so the
    same region can be Latin-OCR'd.
    """
    import cv2

    g_jp = cv2.cvtColor(jp_img, cv2.COLOR_BGR2GRAY) if jp_img.ndim == 3 else jp_img
    g_en = cv2.cvtColor(en_img, cv2.COLOR_BGR2GRAY) if en_img.ndim == 3 else en_img
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(g_jp, None)
    kp2, des2 = akaze.detectAndCompute(g_en, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H


def transfer_bbox(bbox: tuple[int, int, int, int], H) -> tuple[int, int, int, int]:
    """Map a (xmin,ymin,xmax,ymax) bbox through homography ``H``.

    If ``H`` is None, returns the bbox unchanged (identity fallback — valid when
    the scanlation kept the same page geometry).
    """
    import cv2

    xmin, ymin, xmax, ymax = bbox
    if H is None:
        return bbox
    pts = np.float32(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    ).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    nx0 = int(out[:, 0].min())
    ny0 = int(out[:, 1].min())
    nx1 = int(out[:, 0].max())
    ny1 = int(out[:, 1].max())
    return (nx0, ny0, nx1, ny1)


# --------------------------------------------------------------------------- #
# Latin OCR interface
# --------------------------------------------------------------------------- #


class LatinOCR(Protocol):
    def read(self, crop: np.ndarray) -> str: ...


class EasyOCRLatin:
    """Thin adapter over easyocr's Reader (English). Lazy-loaded."""

    def __init__(self, gpu: bool = True) -> None:
        try:
            import easyocr  # noqa: F401
        except ImportError as e:  # pragma: no cover - env-dependent
            raise DependencyMissing("easyocr") from e
        import easyocr

        self._reader = easyocr.Reader(["en"], gpu=gpu)

    def read(self, crop: np.ndarray) -> str:
        results = self._reader.readtext(crop, detail=0, paragraph=True)
        return " ".join(results).strip()


class StubLatinOCR:
    """No-op Latin OCR used when easyocr isn't installed.

    Returns "" so the pipeline still runs end-to-end (JP side fully functional,
    EN side empty) and the orchestrator can report how many pairs were lost to
    the missing dependency.
    """

    def read(self, crop: np.ndarray) -> str:  # noqa: ARG002
        return ""


def get_latin_ocr(prefer_gpu: bool = True) -> LatinOCR:
    """Return a working Latin OCR, falling back to a stub if easyocr is absent."""
    try:
        return EasyOCRLatin(gpu=prefer_gpu)
    except DependencyMissing:
        return StubLatinOCR()


# --------------------------------------------------------------------------- #
# image listing
# --------------------------------------------------------------------------- #

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def list_page_images(directory: Path) -> list[Path]:
    """Sorted page-image paths in a gallery dir (natural numeric order)."""
    imgs = [p for p in directory.iterdir() if p.suffix.lower() in _IMG_EXTS]
    return sorted(imgs, key=lambda p: p.name)

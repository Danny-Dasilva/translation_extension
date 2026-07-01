"""A/B OCR validation: NAR-default vs vertical-AR routing on Ikenie-4 garbles.

Runs the REAL CTD detector + REAL ParseqOCRService over the same line crops on
the known garble pages (5, 45, 123), twice:
  arm A: vertical_ar_default=False  (current NAR default)
  arm B: vertical_ar_default=True   (the fix)
and prints, per block, the OCR read for each arm + whether the garble gate
(is_implausible_japanese) fires. The win is fewer GARBLE flags in arm B and the
target reads recovered (身代わり not 身身わわ, etc.).
"""
import asyncio
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.services.detector_factory import create_detector  # noqa: E402
from app.services.parseq_ocr_service import ParseqOCRService  # noqa: E402
from app.utils.ocr_confidence_gate import is_implausible_japanese  # noqa: E402

RAWS = Path("/mnt/nas/drive_2/onlyfans/external_content/nhentai/583875_Ikenie no Haha 4")
PAGES = ["005", "045", "123"]


def make_ocr(vertical_ar: bool) -> ParseqOCRService:
    return ParseqOCRService(
        model_path=settings.parseq_model_path,
        hybrid_enabled=False,  # isolate the geometry routing from conf-retry
        ar_model_path=settings.parseq_ar_model_path,
        hybrid_conf_threshold=settings.ocr_confidence_gate_threshold,
        vertical_ar_default=vertical_ar,
        vertical_ar_aspect=settings.ocr_vertical_ar_aspect,
    )


async def main():
    detector = create_detector()
    ocr_nar = make_ocr(False)
    ocr_ar = make_ocr(True)

    for pg in PAGES:
        img_path = RAWS / f"{pg}.webp"
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"!! could not read {img_path}")
            continue
        ctd = await detector.detect(image)
        blocks = ctd["blocks"]
        text_lines = ctd["text_lines"]

        t0 = time.perf_counter()
        nar_texts, nar_confs = await ocr_nar.recognize_blocks_with_lines(
            image, blocks, text_lines, return_confidence=True
        )
        nar_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ar_texts, ar_confs = await ocr_ar.recognize_blocks_with_lines(
            image, blocks, text_lines, return_confidence=True
        )
        ar_ms = (time.perf_counter() - t0) * 1000

        print(f"\n========== PAGE {pg}  "
              f"(NAR {nar_ms:.0f}ms vs AR-routed {ar_ms:.0f}ms; "
              f"verticals->AR={ocr_ar.vertical_ar_count - (0 if pg == PAGES[0] else 0)}) ==========")
        print(f"  NAR total ms={nar_ms:.0f}  AR-routed total ms={ar_ms:.0f}  "
              f"delta=+{ar_ms - nar_ms:.0f}ms  cumulative verticals->AR={ocr_ar.vertical_ar_count}")
        nar_garble = ar_garble = 0
        for i, b in enumerate(blocks):
            nt, at = nar_texts[i], ar_texts[i]
            ng = is_implausible_japanese(nt)
            ag = is_implausible_japanese(at)
            nar_garble += ng
            ar_garble += ag
            if nt == at and not ng and not ag:
                continue  # unchanged & clean — skip for signal
            bb = b
            w = bb["maxX"] - bb["minX"]
            h = bb["maxY"] - bb["minY"]
            asp = h / max(1, w)
            mark_n = "GARBLE" if ng else "  ok  "
            mark_a = "GARBLE" if ag else "  ok  "
            flip = " <== RECOVERED" if (ng and not ag) else (
                " <== NEW-GARBLE" if (ag and not ng) else "")
            print(f"  [{i:2d}] h/w={asp:4.1f}")
            print(f"        NAR {mark_n} c={nar_confs[i]:.3f}: {nt!r}")
            print(f"        AR  {mark_a} c={ar_confs[i]:.3f}: {at!r}{flip}")
        print(f"  --- page {pg}: NAR garble={nar_garble}  AR garble={ar_garble}")


if __name__ == "__main__":
    asyncio.run(main())

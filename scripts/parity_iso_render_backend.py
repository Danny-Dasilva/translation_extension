#!/usr/bin/env python3
"""parity_iso_render_backend.py — Path B of the ISOLATED renderer-parity test.

Renders the backend's PIL composite from the SAME captured /translate response
that path A (the extension) renders. This removes translation-inference variance:
both paths draw the identical TextBox[] onto the identical inpaint plate, so any
remaining pixel diff is RENDERER-ONLY (FreeType/PIL vs browser Skia, stroke AA,
sub-pixel layout).

Input : backend/.bench/_parity/iso_<page>/response.json   (from parity_iso_capture.mjs)
Output: backend/.bench/_parity/iso_<page>/backend_render.png

Field mapping (API TextBox -> compose_final inputs), verified against
app/routers/translate.py (TextBox build) and scripts/batch_translate_chapter.py
(compose_final call):
  block      = {minX, minY, maxX, maxY}   <- TextBox top-level bbox (== the
                                             `block` dict the router rendered)
  translation= TextBox.translatedText
  fit_rect   = TextBox.bubbleRect (or None)  <- the same `fit_rect` the router
                                                passed to compose_final
  plate      = inpainted_image_base64[0] decoded to RGB (the base image)

Caveat: compose_final also reads block.get("orphan"); the API TextBox does NOT
expose `orphan`, so it is absent here. This is the CORRECT isolation: the
extension renderer also only ever sees the exposed TextBox fields, so BOTH paths
operate on the identical field set. (orphan only nudges SFX truncation/overlap
for tiny boxes; absence is consistent across both renderers.)

Usage: uv run python scripts/parity_iso_render_backend.py <page>
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Reuse the EXACT backend renderer the extension was ported to match.
from scripts.refit_final_composites import compose_final  # type: ignore  # noqa: E402


def _decode_plate(data_url: str) -> np.ndarray:
    b64 = re.sub(r"^data:image/[^;]+;base64,", "", data_url)
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.array(img)


def main() -> None:
    page = sys.argv[1] if len(sys.argv) > 1 else "044"
    iso_dir = ROOT / "backend" / ".bench" / "_parity" / f"iso_{page}"
    resp_path = iso_dir / "response.json"
    if not resp_path.exists():
        print(f"FATAL: {resp_path} not found (run parity_iso_capture.mjs first)")
        sys.exit(2)

    resp = json.loads(resp_path.read_text())
    boxes = (resp.get("images") or [[]])[0]
    plates = resp.get("inpainted_image_base64") or []
    plate_url = plates[0] if plates else None
    if not plate_url:
        print("FATAL: response has no inpainted_image_base64 plate")
        sys.exit(2)

    inpainted = _decode_plate(plate_url)

    # Map API TextBox[] -> compose_final(blocks, translations, fit_rects).
    blocks: list[dict] = []
    translations: list[str] = []
    fit_rects: list[dict | None] = []
    for b in boxes:
        blocks.append(
            {
                "minX": int(b["minX"]),
                "minY": int(b["minY"]),
                "maxX": int(b["maxX"]),
                "maxY": int(b["maxY"]),
                # `orphan` is not exposed by the API; omit (see module docstring).
            }
        )
        translations.append(b.get("translatedText") or "")
        br = b.get("bubbleRect")
        if br and all(k in br for k in ("minX", "minY", "maxX", "maxY")):
            fit_rects.append(
                {
                    "minX": int(br["minX"]),
                    "minY": int(br["minY"]),
                    "maxX": int(br["maxX"]),
                    "maxY": int(br["maxY"]),
                }
            )
        else:
            fit_rects.append(None)

    final = compose_final(inpainted, blocks, translations, fit_rects=fit_rects)

    out = iso_dir / "backend_render.png"
    Image.fromarray(final).convert("RGB").save(out)

    summary = {
        "page": page,
        "out": str(out),
        "plate_size": list(Image.fromarray(inpainted).size),
        "num_boxes": len(blocks),
        "non_empty": len([t for t in translations if t.strip()]),
        "bubble_matched": len([r for r in fit_rects if r is not None]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

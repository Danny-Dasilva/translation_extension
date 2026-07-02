"""Page-image -> base64 data-URL encoding for the image-context serve path.

The v1 (Qwen3-VL-8B text-SFT) image-context serve path (gated by
``settings.translation_serve_image_context``) sends the manga page image as an
OpenAI multimodal ``image_url`` content block. To bound upload + prefill cost we
downscale the page to a long-side cap and JPEG-encode it before base64ing.

This mirrors ``backend/scripts/eval/pov_probe.image_to_data_url`` (the harness
that validated the arm) — same 1024px long-side cap + JPEG quality 90 — but
operates on an in-memory RGB ndarray (what the translate router already holds
post-decode) rather than a file path, so the router encodes ONCE per page with
no disk round-trip. Keep the cap/quality in sync with pov_probe so the served
image matches the evaluated one.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

# Long-side downscale cap + JPEG quality — MUST match
# backend/scripts/eval/pov_probe.MAX_IMAGE_LONG_SIDE / its JPEG quality=90 so the
# served image is byte-shaped like the evaluated one.
MAX_IMAGE_LONG_SIDE = 1024
JPEG_QUALITY = 90


def ndarray_to_data_url(
    image_rgb: np.ndarray,
    *,
    max_long_side: int = MAX_IMAGE_LONG_SIDE,
    jpeg_quality: int = JPEG_QUALITY,
) -> str:
    """Downscale an HxWx3 RGB uint8 ndarray to ``max_long_side`` then data-URL it.

    Returns a ``data:image/jpeg;base64,...`` URL. The long edge is capped at
    ``max_long_side`` (LANCZOS); already-small pages are left at native size.
    Re-encoded as JPEG (quality ``jpeg_quality``) to bound the prefill cost — the
    exact recipe pov_probe validated. Runs pure-CPU (PIL); call it from a worker
    thread (``asyncio.to_thread``) so it never blocks the event loop.
    """
    im = Image.fromarray(image_rgb)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    w, h = im.size
    long_side = max(w, h)
    if long_side > max_long_side:
        scale = max_long_side / float(long_side)
        im = im.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
        )
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=jpeg_quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

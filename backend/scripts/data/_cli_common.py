"""Shared CLI helpers: loguru-to-stderr setup + dry-run convention.

Loaders and filters import from here so every script has:
- `configure_logging()` — stderr sink, INFO by default.
- `parse_vntl_packed_text(blob)` — reusable JP/EN turn splitter used by VNTL loaders.
"""

from __future__ import annotations

import re
import sys
from typing import Iterator

from loguru import logger

_LOGGING_READY = False


def configure_logging(level: str = "INFO") -> None:
    """Idempotently configure loguru to write to stderr only."""
    global _LOGGING_READY
    if _LOGGING_READY:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    _LOGGING_READY = True


_JP_BLOCK_RE = re.compile(
    r"<<JAPANESE>>\s*\n(.*?)\n<<ENGLISH>>\s*\n(.*?)(?=\n<<JAPANESE>>|\n?$|</s>)",
    re.DOTALL,
)
# Some VNTL records terminate a turn with "</s>". We strip it.
_TRAILING_EOS_RE = re.compile(r"\s*</s>\s*$")
# Speaker tag prefix: "[Name]: 「text」" — we keep the whole string since the
# character name carries register signal. The fine-tune should learn to handle
# these verbatim.


def parse_vntl_packed_text(blob: str) -> Iterator[tuple[str, str]]:
    """Yield (japanese, english) pairs from a VNTL `text` column.

    VNTL records pack a whole scene into a single string:

        <<METADATA>> ... <<START>>
        <<JAPANESE>>
        line1
        <<ENGLISH>>
        translation1</s>
        <<JAPANESE>>
        line2
        <<ENGLISH>>
        translation2</s>
        ...

    This yields each JP/EN pair after stripping the </s> marker.
    """
    # We drop the METADATA block. The content starts at <<START>>.
    start = blob.find("<<START>>")
    body = blob[start + len("<<START>>") :] if start != -1 else blob

    # Split on "<<JAPANESE>>" — each chunk after index 0 is a JP then ENGLISH block.
    chunks = body.split("<<JAPANESE>>")
    for chunk in chunks[1:]:
        if "<<ENGLISH>>" not in chunk:
            continue
        jp_part, en_part = chunk.split("<<ENGLISH>>", 1)
        jp = jp_part.strip()
        # English runs until next marker or </s>
        en = en_part
        # Cut at next section start (paranoid; we already split on JAPANESE but
        # stray markers can appear)
        for sentinel in ("<<JAPANESE>>", "<<METADATA>>", "<<START>>"):
            idx = en.find(sentinel)
            if idx != -1:
                en = en[:idx]
        en = _TRAILING_EOS_RE.sub("", en).strip()
        if not jp or not en:
            continue
        yield jp, en


__all__ = ["configure_logging", "parse_vntl_packed_text", "logger"]

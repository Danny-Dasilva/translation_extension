"""Pure, dependency-light logic shared across the doujin pairing pipeline.

Everything in this module is deliberately free of heavy deps (no cv2 / onnx /
torch) so the unit tests can exercise the *logic* — filename normalization,
pairing-key extraction, manga reading order, and the ``src`` format round-trip —
without loading models or touching the NAS.

------------------------------------------------------------------------------
SRC FORMAT CONTRACT
------------------------------------------------------------------------------
Every emitted (jp, en) bubble pair carries a ``src`` string that encodes
*work + page + bubble* so a downstream builder (v12) can recover page grouping
and reading order::

    doujin:{workid}:p{page}:b{idx}

where:
    workid : str  — slug identifying the matched JP<->EN work (no ':' chars).
                    For gallery pairs this is "g{jp_id}-{en_id}"; for ubuca
                    zip pairs it is the normalized title slug.
    page   : int  — 0-based page index within the work (aligned page number).
    idx    : int  — 0-based bubble index within the page, in manga reading
                    order (right-to-left, column-major, top-to-bottom).

``parse_src`` / ``format_src`` round-trip this string. The v12 builder groups
rows by (workid, page) and orders by idx to reconstruct page context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# src format
# --------------------------------------------------------------------------- #

SRC_PREFIX = "doujin"


def format_src(workid: str, page: int, idx: int) -> str:
    """Build the ``src`` string for a single bubble pair.

    Raises ValueError if ``workid`` contains ``':'`` (would break parsing) or
    if page/idx are negative.
    """
    if ":" in workid:
        raise ValueError(f"workid must not contain ':' -> {workid!r}")
    if page < 0 or idx < 0:
        raise ValueError(f"page/idx must be >= 0, got page={page}, idx={idx}")
    return f"{SRC_PREFIX}:{workid}:p{int(page)}:b{int(idx)}"


@dataclass(frozen=True)
class SrcRef:
    workid: str
    page: int
    idx: int


_SRC_RE = re.compile(rf"^{SRC_PREFIX}:(?P<workid>[^:]+):p(?P<page>\d+):b(?P<idx>\d+)$")


def parse_src(src: str) -> SrcRef:
    """Parse a ``src`` string produced by :func:`format_src`.

    Raises ValueError if the string is malformed.
    """
    m = _SRC_RE.match(src)
    if not m:
        raise ValueError(f"malformed doujin src: {src!r}")
    return SrcRef(
        workid=m.group("workid"),
        page=int(m.group("page")),
        idx=int(m.group("idx")),
    )


# --------------------------------------------------------------------------- #
# filename normalization + pairing-key extraction
# --------------------------------------------------------------------------- #

# language / scanlation tokens stripped when normalizing a title.
_LANG_TOKENS = {
    "english",
    "eng",
    "en",
    "translation",
    "translated",
    "digital",
    "decensored",
    "uncensored",
    "colorized",
}

# bracket groups: (...) [...] {...}  — used to peel circle/artist/parody/scanlator
_BRACKET_RE = re.compile(r"[\(\[\{]([^\(\)\[\]\{\}]*)[\)\]\}]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Lowercase, strip non-alphanumerics to single hyphens, trim."""
    s = text.strip().lower()
    s = _NON_ALNUM_RE.sub("-", s)
    return s.strip("-")


def _looks_like_lang_token(tok: str) -> bool:
    norm = slugify(tok).replace("-", "")
    return norm in {t.replace("-", "") for t in _LANG_TOKENS}


@dataclass
class DoujinMeta:
    """Parsed metadata from an ubuca-style zip filename."""

    raw: str
    title: str = ""
    circle: str = ""
    artist: str = ""
    parody: str = ""
    is_english: bool = False
    brackets: list[str] = field(default_factory=list)

    @property
    def pairing_key(self) -> str:
        """Stable slug used to match JP<->EN works.

        Title is the primary signal; artist/circle disambiguate. We slug the
        title alone as the key so an [English] scanlation and its JP original
        (which share a title) collide, while language tokens are excluded.
        """
        return slugify(self.title)

    @property
    def workid(self) -> str:
        """Slug embedded in the ``src`` string for ubuca pairs."""
        base = self.pairing_key or "untitled"
        if self.artist:
            base = f"{base}-{slugify(self.artist)}"
        return base


def parse_ubuca_filename(filename: str) -> DoujinMeta:
    """Parse a doujin .zip filename into structured metadata.

    Handles the common ehentai / ubuca naming conventions seen in the corpus::

        (Event) [Circle (Artist)] Title (Parody) [English] [Scanlator].zip
        Artist - Title (Parody) [Eng] [scanlator].zip
        Title [English] {site}.zip

    The parse is heuristic and lossy by design; the only field the pipeline
    *relies* on is ``pairing_key`` (a normalized title) and ``is_english``.
    """
    name = filename
    # strip a trailing extension (.zip/.cbz/.rar) case-insensitively
    name = re.sub(r"\.(zip|cbz|cbr|rar|7z)$", "", name, flags=re.IGNORECASE)

    brackets = [b.strip() for b in _BRACKET_RE.findall(name)]
    is_english = any(_looks_like_lang_token(b) for b in brackets)

    # A leading bracket group "(Event) [Circle ...]" signals the ehentai
    # convention; absence of one signals the "Artist - Title" convention.
    starts_with_bracket = bool(re.match(r"^\s*[\(\[\{]", name))

    circle = ""
    artist = ""
    if starts_with_bracket:
        # circle/artist: first [..] group that is NOT a language/site token.
        for b in brackets:
            if _looks_like_lang_token(b):
                continue
            if b.startswith("http") or ("." in b and " " not in b):
                continue  # site tag like {doujins.com}
            circle = b
            break

    # remove all bracket groups to isolate the title text
    title_region = _BRACKET_RE.sub(" ", name)

    # "Artist - Title" convention (no leading bracket group)
    if " - " in title_region and not starts_with_bracket:
        left, _, right = title_region.partition(" - ")
        artist = left.strip()
        title_region = right

    # parody is conventionally the LAST (...) before language tags; with
    # brackets flattened we approximate: any bracket that isn't circle/lang.
    parody = ""
    for b in brackets:
        if b in (circle,):
            continue
        if _looks_like_lang_token(b):
            continue
        if b != circle:
            parody = b
            break

    title = re.sub(r"\s+", " ", title_region).strip(" -_.")

    return DoujinMeta(
        raw=filename,
        title=title,
        circle=circle,
        artist=artist,
        parody=parody,
        is_english=is_english,
        brackets=brackets,
    )


# --------------------------------------------------------------------------- #
# gallery id parsing
# --------------------------------------------------------------------------- #

_GALLERY_RE = re.compile(r"^(?P<id>\d+)_(?P<lang>en|jp)$")


@dataclass(frozen=True)
class GalleryName:
    gallery_id: str
    lang: str  # "en" | "jp"


def parse_gallery_name(dirname: str) -> GalleryName | None:
    """Parse a ``{id}_{en|jp}`` gallery directory name. None if it doesn't match."""
    m = _GALLERY_RE.match(dirname.strip())
    if not m:
        return None
    return GalleryName(gallery_id=m.group("id"), lang=m.group("lang"))


# --------------------------------------------------------------------------- #
# manga reading order  (ported from scripts/data/v11/build_v11_dataset.py)
# --------------------------------------------------------------------------- #


def manga_reading_order(boxes: list[dict]) -> list[dict]:
    """Right-to-left, column-major, top-to-bottom ordering of bboxes.

    Each box must have ``xmin, ymin, xmax, ymax``. Returns the same dicts in
    reading order. Faithful port of the v11 builder so doujin pages order
    identically to the rest of the training data.
    """
    if len(boxes) <= 1:
        return list(boxes)
    work = [dict(b) for b in boxes]
    for r in work:
        r["_cx"] = (r["xmin"] + r["xmax"]) / 2.0
    page_w = max(r["xmax"] for r in work) - min(r["xmin"] for r in work)
    tol = max(40.0, page_w * 0.06)
    by_x = sorted(work, key=lambda r: -r["_cx"])
    columns: list[list[dict]] = []
    for r in by_x:
        placed = False
        for col in columns:
            col_cx = sum(c["_cx"] for c in col) / len(col)
            if abs(r["_cx"] - col_cx) <= tol:
                col.append(r)
                placed = True
                break
        if not placed:
            columns.append([r])
    columns.sort(key=lambda col: -sum(c["_cx"] for c in col) / len(col))
    ordered: list[dict] = []
    for col in columns:
        col.sort(key=lambda r: r["ymin"])
        ordered.extend(col)
    # strip helper keys, map back to original dict identities by index match
    for r in ordered:
        r.pop("_cx", None)
    # return original dicts in the computed order (preserve caller's objects)
    order_keys = [
        (r["xmin"], r["ymin"], r["xmax"], r["ymax"]) for r in ordered
    ]
    remaining = list(boxes)
    result: list[dict] = []
    for key in order_keys:
        for i, b in enumerate(remaining):
            if (b["xmin"], b["ymin"], b["xmax"], b["ymax"]) == key:
                result.append(remaining.pop(i))
                break
    return result

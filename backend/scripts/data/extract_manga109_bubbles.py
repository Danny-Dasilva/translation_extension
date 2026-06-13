"""Extract dialogue bubbles from Manga109-s annotations -> parquet.

Walks every `<book>.xml` under ``--manga109-root/annotations/`` and emits
one row per ``<text>`` element. Optionally cross-references
``annotations_Manga109Dialog/<book>.xml`` to attach the speaker character
name (``<character id="..." name="...">`` lookup in the same annotations
file).

Filters:
  - drops empty / whitespace-only text
  - drops < ``--min-chars`` (default 1, set 0 to keep every bubble)
  - normalizes whitespace (replaces \\r\\n with single space) but keeps SFX

Output columns:
  [book, page, text_id, jp_text, speaker, xmin, ymin, xmax, ymax]

License (Manga109-s readme):
  - Source images may NOT be redistributed (rule 1).
  - Outputs from machine-learning experiments may be published as long as
    Manga109-s is acknowledged (rule 2).  This parquet contains ONLY the
    JP text (no images), so it is permitted -- but treat it as derived data.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable

import polars as pl

# Allow `from _cli_common import ...` when run via `uv run` from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli_common import configure_logging, logger  # noqa: E402


DEFAULT_ROOT = Path(
    "/mnt/nas/drive_2/manga-ml/datasets/manga109s/Manga109s_released_2023_12_07"
)
DEFAULT_OUT = Path(
    "/home/danny/Documents/personal/extension/backend/scripts/data/manga109/bubbles.parquet"
)


def _parse_book(annot_path: Path) -> tuple[list[dict], dict[str, str], dict[str, str]]:
    """Return (rows, character_id_to_name, body_or_face_id_to_character_id) for one
    annotations/<book>.xml.

    Each row has all fields except `speaker` -- speaker is filled in by
    cross-referencing the Manga109Dialog file later.

    Manga109Dialog's ``speaker_id`` points to a ``<body>`` (or sometimes
    ``<face>``) element, which has a ``character="..."`` attribute that
    finally resolves to the character name.  We build the second map for
    that lookup.
    """
    tree = ET.parse(annot_path)
    root = tree.getroot()
    book = root.attrib.get("title") or annot_path.stem

    char_id_to_name: dict[str, str] = {}
    chars_node = root.find("characters")
    if chars_node is not None:
        for c in chars_node.findall("character"):
            cid = c.attrib.get("id")
            cname = c.attrib.get("name")
            if cid and cname:
                char_id_to_name[cid] = cname

    body_to_char: dict[str, str] = {}
    rows: list[dict] = []
    pages = root.find("pages")
    if pages is None:
        return rows, char_id_to_name, body_to_char
    for page in pages.findall("page"):
        # First pass: register body / face id -> character id mapping.
        for tag in ("body", "face"):
            for el in page.findall(tag):
                eid = el.attrib.get("id")
                cid = el.attrib.get("character")
                if eid and cid:
                    body_to_char[eid] = cid
        try:
            page_idx = int(page.attrib.get("index", "-1"))
        except ValueError:
            page_idx = -1
        for text in page.findall("text"):
            jp = (text.text or "").strip()
            if not jp:
                continue
            try:
                xmin = int(text.attrib.get("xmin", "0"))
                ymin = int(text.attrib.get("ymin", "0"))
                xmax = int(text.attrib.get("xmax", "0"))
                ymax = int(text.attrib.get("ymax", "0"))
            except ValueError:
                xmin = ymin = xmax = ymax = 0
            rows.append(
                {
                    "book": book,
                    "page": page_idx,
                    "text_id": text.attrib.get("id") or "",
                    "jp_text": jp,
                    "speaker": None,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                }
            )
    return rows, char_id_to_name, body_to_char


def _parse_dialog_speakers(dialog_path: Path) -> dict[str, str]:
    """Map text_id -> speaker_id from annotations_Manga109Dialog/<book>.xml."""
    if not dialog_path.exists():
        return {}
    try:
        tree = ET.parse(dialog_path)
    except ET.ParseError as e:
        logger.warning(f"failed to parse {dialog_path}: {e}")
        return {}
    out: dict[str, str] = {}
    root = tree.getroot()
    pages = root.find("pages") if root.find("pages") is not None else root
    for page in pages.findall("page"):
        for stt in page.findall("speaker_to_text"):
            tid = stt.attrib.get("text_id")
            sid = stt.attrib.get("speaker_id")
            if tid and sid:
                out[tid] = sid
    return out


def _filter_text(jp: str, *, min_chars: int) -> bool:
    """Heuristic filter. Returns True if the row should be kept.

    We keep SFX-style short kana (e.g. ``ドキドキ``, ``ガタッ``) because they
    are legitimate translation targets in manga.  We only drop:
      - empty / whitespace
      - shorter than ``min_chars``
    """
    if not jp.strip():
        return False
    if len(jp.strip()) < min_chars:
        return False
    return True


def _normalize(jp: str) -> str:
    # Collapse internal newlines to a single space; strip outer whitespace.
    return " ".join(jp.split())


def _length_histogram(lengths: Iterable[int]) -> dict[str, int]:
    buckets = [
        ("1", 1, 1),
        ("2", 2, 2),
        ("3-5", 3, 5),
        ("6-10", 6, 10),
        ("11-20", 11, 20),
        ("21-40", 21, 40),
        ("41-80", 41, 80),
        (">80", 81, 10**9),
    ]
    out: dict[str, int] = {b[0]: 0 for b in buckets}
    for length in lengths:
        for label, lo, hi in buckets:
            if lo <= length <= hi:
                out[label] += 1
                break
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manga109-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--include-speaker-context",
        type=lambda v: v.lower() not in {"0", "false", "no"},
        default=True,
        help="cross-reference annotations_Manga109Dialog/<book>.xml for speaker names",
    )
    p.add_argument("--min-chars", type=int, default=1)
    p.add_argument(
        "--books",
        type=str,
        default=None,
        help="comma-separated list of book names (without .xml) to limit to",
    )
    return p.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    annot_dir = args.manga109_root / "annotations"
    dialog_dir = args.manga109_root / "annotations_Manga109Dialog"
    if not annot_dir.exists():
        logger.error(f"annotations dir not found: {annot_dir}")
        return 2

    book_files = sorted(annot_dir.glob("*.xml"))
    if args.books:
        wanted = {b.strip() for b in args.books.split(",") if b.strip()}
        book_files = [f for f in book_files if f.stem in wanted]
        if not book_files:
            logger.error(f"no books matched --books={args.books}")
            return 2

    logger.info(f"parsing {len(book_files)} books from {annot_dir}")

    all_rows: list[dict] = []
    speakers_attached = 0
    speakers_missing = 0
    for book_xml in book_files:
        rows, id_to_name, body_to_char = _parse_book(book_xml)
        if args.include_speaker_context:
            speaker_map = _parse_dialog_speakers(dialog_dir / book_xml.name)
            for r in rows:
                sid = speaker_map.get(r["text_id"])
                if not sid:
                    continue
                # Manga109Dialog speaker_id points to a body/face element;
                # resolve via body_to_char to character id, then to name.
                cid = body_to_char.get(sid)
                if not cid:
                    speakers_missing += 1
                    continue
                name = id_to_name.get(cid)
                if name:
                    r["speaker"] = name
                    speakers_attached += 1
                else:
                    speakers_missing += 1
        all_rows.extend(rows)
        logger.info(f"  {book_xml.stem}: {len(rows)} bubbles")

    logger.info(f"raw total bubbles: {len(all_rows):,}")

    # Filter + normalize.
    kept: list[dict] = []
    for r in all_rows:
        if not _filter_text(r["jp_text"], min_chars=args.min_chars):
            continue
        r["jp_text"] = _normalize(r["jp_text"])
        kept.append(r)
    dropped = len(all_rows) - len(kept)
    logger.info(f"after filter (min_chars={args.min_chars}): kept {len(kept):,}, dropped {dropped:,}")

    if not kept:
        logger.error("no rows survived filtering -- aborting")
        return 1

    df = pl.DataFrame(kept)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)
    logger.info(f"wrote {len(df):,} rows to {args.output}")

    # Stats.
    book_counts = Counter(r["book"] for r in kept)
    lens = [len(r["jp_text"]) for r in kept]
    logger.info(f"books: {len(book_counts)}")
    logger.info(f"speaker attached: {speakers_attached:,}, speaker_id_unresolved: {speakers_missing:,}")
    logger.info(f"length min/median/max chars: {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
    logger.info(f"length histogram: {_length_histogram(lens)}")

    # Sample 10 rows.
    sample_n = min(10, len(df))
    sample = df.sample(sample_n, seed=0) if len(df) > sample_n else df
    print("\n=== 10 sample rows ===", file=sys.stderr)
    for row in sample.iter_rows(named=True):
        spk = f" <{row['speaker']}>" if row["speaker"] else ""
        print(
            f"  [{row['book']} p{row['page']}{spk}] {row['jp_text']}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

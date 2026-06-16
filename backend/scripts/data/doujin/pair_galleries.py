"""Stage 1 — pairing: match JP originals to EN scanlations.

Writes ``candidate_pairs.parquet`` to LOCAL disk (never the NAS — the CIFS
share empties output dirs ~9 min after write).

Two corpora, two pairing strategies:

galleries/  ({id}_en, {id}_jp)
    VERIFIED on the real NAS: the numeric ids are NOT shared between en/jp
    (0 ids have both a _jp and _en sibling). They are independent gallery ids.
    The dirs contain ONLY images — no title/artist metadata. Therefore the only
    pairing signal is *visual*: a cover/page perceptual-hash match between an EN
    gallery and a JP gallery. This stage emits the per-gallery cover phash and,
    when ``--match-galleries`` is set, the best cross-language phash matches.

archive_ubuca_v5_p1/_p2/  (*.zip)
    Filenames carry [English]/circle/artist/title metadata. We parse them with
    ``parse_ubuca_filename`` and match an English zip to a (hypothetical) JP
    original by ``pairing_key`` (normalized title). In this corpus most zips are
    already English scanlations; the JP original is usually a *gallery*, so the
    richer use is title-keying for downstream gallery lookup. This stage emits
    the parsed metadata so a matcher can join on ``pairing_key``.

All stages are --limit-able, idempotent, and resumable (skip if output exists
unless --force).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from doujin_common import parse_gallery_name, parse_ubuca_filename  # noqa: E402

# ----- local output root (NEVER write durable output to the NAS) ------------ #
LOCAL_OUT = Path(__file__).resolve().parent
NAS_GALLERIES = Path("/mnt/nas/drive_2/manga-ml/ehentai_corpus/galleries")
NAS_UBUCA = [
    Path("/mnt/nas/drive_2/manga-ml/ehentai_corpus/archive_ubuca_v5_p1"),
    Path("/mnt/nas/drive_2/manga-ml/ehentai_corpus/archive_ubuca_v5_p2"),
]


def _cover_phash(gallery_dir: Path) -> int | None:
    """phash the first page image of a gallery; None on any failure."""
    try:
        import cv2

        from doujin_vision import list_page_images, phash

        pages = list_page_images(gallery_dir)
        if not pages:
            return None
        img = cv2.imread(str(pages[0]))
        if img is None:
            return None
        return phash(img)
    except Exception:
        return None


def scan_galleries(limit: int, with_phash: bool) -> pl.DataFrame:
    """List gallery dirs -> rows {gallery_id, lang, path, n_pages, cover_phash}."""
    rows: list[dict] = []
    count = 0
    for entry in sorted(NAS_GALLERIES.iterdir()):
        if not entry.is_dir():
            continue
        g = parse_gallery_name(entry.name)
        if g is None:
            continue
        try:
            n_pages = sum(
                1
                for p in entry.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
        except OSError:
            n_pages = 0
        ph = _cover_phash(entry) if with_phash else None
        rows.append(
            {
                "gallery_id": g.gallery_id,
                "lang": g.lang,
                "path": str(entry),
                "n_pages": n_pages,
                "cover_phash": str(ph) if ph is not None else None,
            }
        )
        count += 1
        if count >= limit:
            break
    return pl.DataFrame(
        rows,
        schema={
            "gallery_id": pl.Utf8,
            "lang": pl.Utf8,
            "path": pl.Utf8,
            "n_pages": pl.Int64,
            "cover_phash": pl.Utf8,
        },
    )


def scan_ubuca(limit: int) -> pl.DataFrame:
    """List ubuca zips -> parsed metadata rows."""
    rows: list[dict] = []
    count = 0
    for root in NAS_UBUCA:
        if not root.exists():
            continue
        for zp in sorted(root.glob("*.zip")):
            m = parse_ubuca_filename(zp.name)
            rows.append(
                {
                    "path": str(zp),
                    "title": m.title,
                    "artist": m.artist,
                    "circle": m.circle,
                    "parody": m.parody,
                    "is_english": m.is_english,
                    "pairing_key": m.pairing_key,
                    "workid": m.workid,
                }
            )
            count += 1
            if count >= limit:
                break
        if count >= limit:
            break
    return pl.DataFrame(
        rows,
        schema={
            "path": pl.Utf8,
            "title": pl.Utf8,
            "artist": pl.Utf8,
            "circle": pl.Utf8,
            "parody": pl.Utf8,
            "is_english": pl.Boolean,
            "pairing_key": pl.Utf8,
            "workid": pl.Utf8,
        },
    )


def match_galleries_by_phash(df: pl.DataFrame, max_distance: int) -> pl.DataFrame:
    """Greedy cross-language cover-phash matching of EN<->JP galleries.

    Emits candidate pairs {jp_id, en_id, jp_path, en_path, distance, workid}.
    Only galleries that were phashed (cover_phash not null) participate.
    """
    from doujin_vision import hamming_distance

    have = df.filter(pl.col("cover_phash").is_not_null())
    en = have.filter(pl.col("lang") == "en").to_dicts()
    jp = have.filter(pl.col("lang") == "jp").to_dicts()
    used_en: set[int] = set()
    pairs: list[dict] = []
    for jr in jp:
        jh = int(jr["cover_phash"])
        best_i, best_d = -1, max_distance + 1
        for i, er in enumerate(en):
            if i in used_en:
                continue
            d = hamming_distance(jh, int(er["cover_phash"]))
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d <= max_distance:
            used_en.add(best_i)
            er = en[best_i]
            pairs.append(
                {
                    "jp_id": jr["gallery_id"],
                    "en_id": er["gallery_id"],
                    "jp_path": jr["path"],
                    "en_path": er["path"],
                    "distance": best_d,
                    "workid": f"g{jr['gallery_id']}-{er['gallery_id']}",
                }
            )
    return pl.DataFrame(
        pairs,
        schema={
            "jp_id": pl.Utf8,
            "en_id": pl.Utf8,
            "jp_path": pl.Utf8,
            "en_path": pl.Utf8,
            "distance": pl.Int64,
            "workid": pl.Utf8,
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 1: doujin pairing")
    ap.add_argument("--limit", type=int, default=50, help="max items per corpus")
    ap.add_argument(
        "--source",
        choices=["galleries", "ubuca", "both"],
        default="galleries",
    )
    ap.add_argument(
        "--phash",
        action="store_true",
        help="compute cover phash (slow over CIFS) — needed for --match-galleries",
    )
    ap.add_argument(
        "--match-galleries",
        action="store_true",
        help="greedy cross-language cover-phash matching (implies --phash)",
    )
    ap.add_argument("--max-distance", type=int, default=14)
    ap.add_argument("--out-dir", type=Path, default=LOCAL_OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.match_galleries:
        args.phash = True

    if args.source in ("galleries", "both"):
        out = out_dir / "galleries_index.parquet"
        if out.exists() and not args.force:
            print(f"[skip] {out} exists (use --force)")
            gdf = pl.read_parquet(out)
        else:
            gdf = scan_galleries(args.limit, with_phash=args.phash)
            gdf.write_parquet(out)
            print(f"[ok] wrote {len(gdf)} gallery rows -> {out}")

        if args.match_galleries:
            mout = out_dir / "candidate_pairs.parquet"
            if mout.exists() and not args.force:
                print(f"[skip] {mout} exists (use --force)")
            else:
                pairs = match_galleries_by_phash(gdf, args.max_distance)
                pairs.write_parquet(mout)
                print(f"[ok] wrote {len(pairs)} candidate pairs -> {mout}")

    if args.source in ("ubuca", "both"):
        out = out_dir / "ubuca_index.parquet"
        if out.exists() and not args.force:
            print(f"[skip] {out} exists (use --force)")
        else:
            udf = scan_ubuca(args.limit)
            udf.write_parquet(out)
            print(f"[ok] wrote {len(udf)} ubuca rows -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

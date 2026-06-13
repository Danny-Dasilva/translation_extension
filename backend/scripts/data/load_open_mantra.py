"""Load open-mantra-dataset into unified parquet + held-out eval split.

The dataset has 5 volumes; we split at the volume/series level (no leakage):
  - Train volumes:      tojime_no_siora, balloon_dream, tencho_isoro  (3 vols)
  - Held-out volumes:   boureisougi, rasetugari                        (2 vols)

Train rows: register_tag='manga', gold_flag=True, src='open_mantra:<title>:p<page>:t<idx>'.
Held-out:   separate JSONL for `build_held_out.py` to consume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from _cli_common import configure_logging, logger
from unify_schema import make_row, write_parquet


DEFAULT_ANNOTATION = (
    "backend/training/datasets/translation/vn-ln-manga/"
    "open-mantra-dataset/annotation.json"
)

TRAIN_VOLUMES = ("tojime_no_siora", "balloon_dream", "tencho_isoro")
HELD_OUT_VOLUMES = ("boureisougi", "rasetugari")


def iter_volume_pairs(
    annotation_path: Path,
    titles: tuple[str, ...],
    *,
    src_prefix: str,
) -> Iterator[dict[str, object]]:
    with annotation_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    title_set = set(titles)
    seen_titles: set[str] = set()
    for book in data:
        title = book.get("book_title")
        if title not in title_set:
            continue
        seen_titles.add(title)
        for page in book.get("pages", []):
            page_idx = page.get("page_index")
            for t_idx, entry in enumerate(page.get("text", [])):
                jp = (entry.get("text_ja") or "").strip()
                en = (entry.get("text_en") or "").strip()
                if not jp or not en:
                    continue
                yield make_row(
                    jp=jp,
                    en=en,
                    src=f"{src_prefix}:{title}:p{page_idx}:t{t_idx}",
                    register_tag="manga",
                    gold_flag=True,
                )
    missing = title_set - seen_titles
    if missing:
        logger.warning(f"open-mantra: requested volumes not in annotation: {missing}")


def write_held_out_jsonl(annotation_path: Path, out_path: Path) -> int:
    """Emit the 2 held-out volumes as JSONL for eval use (one pair per line)."""
    count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as wh:
        for row in iter_volume_pairs(
            annotation_path, HELD_OUT_VOLUMES, src_prefix="open_mantra_eval"
        ):
            wh.write(
                json.dumps(
                    {
                        "jp": row["jp"],
                        "en": row["en"],
                        "src": row["src"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", default=DEFAULT_ANNOTATION)
    parser.add_argument(
        "--out-train",
        default="backend/training/datasets/unified/open_mantra_train.parquet",
    )
    parser.add_argument(
        "--out-held-out",
        default="backend/training/eval_held_out/open_mantra_test.jsonl",
    )
    parser.add_argument(
        "--skip-held-out",
        action="store_true",
        help="Skip writing the held-out JSONL (produced by build_held_out.py too).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    ann = Path(args.annotation)
    if not ann.exists():
        logger.error(f"annotation missing: {ann}")
        raise SystemExit(2)

    train_rows = iter_volume_pairs(ann, TRAIN_VOLUMES, src_prefix="open_mantra_train")
    if args.dry_run:
        n_train = sum(1 for _ in train_rows)
        n_held = sum(
            1 for _ in iter_volume_pairs(ann, HELD_OUT_VOLUMES, src_prefix="x")
        )
        print(f"open-mantra train pairs: {n_train}; held-out pairs: {n_held}")
        return

    n_train = write_parquet(train_rows, args.out_train)
    print(f"wrote {n_train} rows to {args.out_train}")

    if not args.skip_held_out:
        n_held = write_held_out_jsonl(ann, Path(args.out_held_out))
        print(f"wrote {n_held} held-out rows to {args.out_held_out}")


if __name__ == "__main__":
    main()

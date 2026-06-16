"""Load lmg-anon/VNTL-Chat into unified parquet (downloads/streams via HF datasets).

The dataset (~1,996 rows) stores a roleplay-style translation chat. Each record
has three columns:

    japanese : str  — the JP source line (often a 「...」 quoted utterance)
    english  : str  — the human-aligned EN translation
    messages : list — a roleplay conversation wrapper (IGNORED here)

We use ``japanese`` -> jp and ``english`` -> en and drop ``messages`` entirely.

Content is VN/eroge register with NSFW/slang. Tags: register_tag='vn_eroge',
gold_flag=True, src='vntl_chat:row{i}'.
"""

from __future__ import annotations

import argparse
from typing import Iterator

from _cli_common import configure_logging, logger
from unify_schema import make_row, write_parquet


HF_REPO = "lmg-anon/VNTL-Chat"


def iter_rows(limit: int | None = None) -> Iterator[dict[str, object]]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error(
            "vntl_chat: `datasets` not installed; add it to pyproject and retry. "
            f"TODO: pip install datasets. Original error: {e}"
        )
        return

    try:
        ds = load_dataset(HF_REPO, split="train", streaming=True)
    except Exception as e:  # gated/unavailable/network
        logger.error(
            f"vntl_chat: could not load {HF_REPO} ({type(e).__name__}: {e}). "
            "TODO: dataset may be gated/require auth or network access; skipping."
        )
        return

    logged_cols = False
    seen: set[tuple[str, str]] = set()
    emitted = 0
    skipped = 0
    for i, rec in enumerate(ds):
        if not logged_cols:
            logger.info(f"vntl_chat: detected columns={list(rec.keys())}")
            logged_cols = True
        jp = (rec.get("japanese") or "").strip()
        en = (rec.get("english") or "").strip()
        if not jp or not en:
            skipped += 1
            continue
        key = (jp, en)
        if key in seen:
            continue
        seen.add(key)
        emitted += 1
        yield make_row(
            jp=jp,
            en=en,
            src=f"vntl_chat:row{i}",
            register_tag="vn_eroge",
            gold_flag=True,
        )
        if limit is not None and emitted >= limit:
            break
    logger.info(f"vntl_chat: emitted={emitted} skipped_empty={skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="backend/training/datasets/unified/vntl_chat.parquet",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap emitted rows.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.dry_run:
        rows = list(iter_rows(limit=args.limit if args.limit else 50))
        print(f"vntl_chat rows: {len(rows)}")
        for r in rows[:3]:
            print(f"  jp={r['jp']!r}\n  en={r['en']!r}\n  src={r['src']}")
        return
    n = write_parquet(iter_rows(limit=args.limit), args.out)
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()

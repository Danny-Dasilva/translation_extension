"""Load Helsinki-NLP/tatoeba_mt jpn-eng pairs into unified parquet.

On-disk anomaly: The local dataset directory contains only the HF loading
script (``tatoeba_mt.py``) and metadata — no actual data parquet. We therefore
trigger an HF ``datasets.load_dataset`` lazily on first run (downloads ~small
data to the HF cache). If the local dir already contains jpn-eng parquet shards
we prefer them.

Tags: register_tag='anchor', gold_flag=False, src='tatoeba:<split>:<id>'.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

from _cli_common import configure_logging, logger
from unify_schema import make_row, write_parquet


DEFAULT_LOCAL_DIR = (
    "backend/training/datasets/translation/slang-colloquial/Helsinki-NLP__tatoeba_mt"
)


def _find_local_parquets(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(base.rglob("*.parquet"))


def _iter_from_local_parquet(paths: list[Path]) -> Iterator[dict[str, object]]:
    import polars as pl  # local import keeps CLI help fast

    for p in paths:
        df = pl.read_parquet(p)
        # Tatoeba HF format: columns vary; look for "sourceString"/"targetString"
        # or "ja"/"en" — we probe.
        col_map = {c.lower(): c for c in df.columns}
        jp_col = None
        en_col = None
        for cand in ("ja", "jpn", "source", "sourcestring", "sentenceja"):
            if cand in col_map:
                jp_col = col_map[cand]
                break
        for cand in ("en", "eng", "target", "targetstring", "sentenceen"):
            if cand in col_map:
                en_col = col_map[cand]
                break
        if jp_col is None or en_col is None:
            logger.warning(
                f"tatoeba: cannot find jp/en cols in {p}; have {df.columns}"
            )
            continue
        for i, r in enumerate(df.iter_rows(named=True)):
            jp = (r.get(jp_col) or "").strip()
            en = (r.get(en_col) or "").strip()
            if not jp or not en:
                continue
            yield make_row(
                jp=jp,
                en=en,
                src=f"tatoeba_local:{p.name}:{i}",
                register_tag="anchor",
                gold_flag=False,
            )


def _iter_from_hf(cache_dir: Path | None = None) -> Iterator[dict[str, object]]:
    """Fall back to HF datasets loader. Downloads on first call."""
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error(
            "tatoeba: `datasets` not installed; add it to pyproject and retry. "
            f"Original error: {e}"
        )
        raise SystemExit(2)

    logger.info("tatoeba: downloading Helsinki-NLP/tatoeba_mt jpn-eng via HF datasets")
    ds = load_dataset(
        "Helsinki-NLP/tatoeba_mt",
        "jpn-eng",
        split="test",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    for i, r in enumerate(ds):
        rec: dict = r if isinstance(r, dict) else {}
        # HF tatoeba_mt fields are usually: sourceString, targetString, sourceLang, targetLang
        jp = (rec.get("sourceString") or rec.get("source") or "").strip()
        en = (rec.get("targetString") or rec.get("target") or "").strip()
        # Some configs swap direction; detect by first char heuristic as fallback
        if not jp or not en:
            continue
        yield make_row(
            jp=jp,
            en=en,
            src=f"tatoeba_hf:{i}",
            register_tag="anchor",
            gold_flag=False,
        )


def iter_rows(local_dir: Path, use_hf: bool) -> Iterator[dict[str, object]]:
    parquets = _find_local_parquets(local_dir)
    if parquets:
        logger.info(f"tatoeba: found {len(parquets)} local parquet(s); using them")
        yield from _iter_from_local_parquet(parquets)
        return
    if use_hf:
        yield from _iter_from_hf()
    else:
        logger.error(
            "tatoeba: no local parquets found and --no-hf passed; skipping. "
            f"Search base was {local_dir}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument(
        "--out",
        default="backend/training/datasets/unified/tatoeba.parquet",
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Disable HF fallback; only use local parquet if present.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    rows = iter_rows(Path(args.local_dir), use_hf=not args.no_hf)
    if args.dry_run:
        print(f"tatoeba rows: {sum(1 for _ in rows)}")
        return
    n = write_parquet(rows, args.out)
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()

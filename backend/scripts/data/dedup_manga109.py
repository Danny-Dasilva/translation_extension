"""Cross-dedup Manga109 vs held-out eval bubbles using MinHash LSH.

Drops any Manga109 row whose ``jp_text`` is a near-duplicate of ANY held-out
JP text (Jaccard >= ``--threshold`` on character n-grams), plus a
deterministic substring backstop (drop if a normalized variant of the row's
JP text is a substring of any held-out JP, or vice-versa).

Held-out sources (combined into one corpus):
  - JSONL files matching ``--heldout`` (default: training/eval_held_out/*.jsonl)
  - Treats every line as a JSON object with a ``jp`` field.

Why both LSH + substring:
  - LSH is fast on 132k x N held-out, but uses 9-grams so very short bubbles
    (1-3 chars) can collide noisily.
  - Substring backstop catches "verbatim copy" cases the LSH might miss
    when held-out lines are extremely short.

Output parquet has the same schema as input -- just with rows removed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli_common import configure_logging, logger  # noqa: E402


DEFAULT_MANGA109 = Path(
    "/home/danny/Documents/personal/extension/backend/scripts/data/manga109/bubbles_translated_qe.parquet"
)
DEFAULT_HELDOUT_DIR = Path(
    "/home/danny/Documents/personal/extension/backend/training/eval_held_out"
)
DEFAULT_OUTPUT = Path(
    "/home/danny/Documents/personal/extension/backend/scripts/data/manga109/bubbles_final.parquet"
)


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[、。！？!?,.\.\,\(\)（）「」『』〝〟“”\"'\-—…・]")


def _normalize_jp(s: str) -> str:
    """Aggressive normalization for substring + shingle comparison."""
    s = unicodedata.normalize("NFKC", s)
    s = _WS_RE.sub("", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower()


def _shingles(text: str, n: int) -> set[bytes]:
    if not text:
        return set()
    if len(text) < n:
        return {text.encode("utf-8")}
    return {text[i : i + n].encode("utf-8") for i in range(len(text) - n + 1)}


def _mk_minhash(text: str, *, n: int, num_perm: int, MinHash):
    mh = MinHash(num_perm=num_perm)
    for sh in _shingles(text, n):
        mh.update(sh)
    return mh


def _load_heldout_jp(heldout_paths: list[Path]) -> list[str]:
    out: list[str] = []
    for p in heldout_paths:
        if not p.exists():
            logger.warning(f"held-out file missing: {p}")
            continue
        if p.suffix == ".jsonl":
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    jp = d.get("jp")
                    if jp:
                        out.append(jp)
        elif p.suffix == ".parquet":
            df = pl.read_parquet(p)
            col = "jp_text" if "jp_text" in df.columns else ("jp" if "jp" in df.columns else None)
            if col is None:
                logger.warning(f"{p}: no jp/jp_text column; skipping")
                continue
            for row in df.iter_rows(named=True):
                v = row.get(col)
                if v:
                    out.append(str(v))
        else:
            logger.warning(f"unknown held-out suffix: {p}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manga109", type=Path, default=DEFAULT_MANGA109)
    p.add_argument(
        "--heldout", type=str, default=None,
        help="comma-separated list of held-out JSONL/parquet files; "
             "defaults to all .jsonl under training/eval_held_out/"
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--ngram-size", type=int, default=9)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--bands", type=int, default=14)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--substring-min-chars", type=int, default=6,
                   help="min normalized-text length to apply substring rule")
    p.add_argument("--n-dropped-samples", type=int, default=10)
    return p.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error("datasketch not installed; `uv add --project backend datasketch`")
        raise SystemExit(2) from e

    if not args.manga109.exists():
        logger.error(f"manga109 parquet not found: {args.manga109}")
        return 2
    df = pl.read_parquet(args.manga109)
    logger.info(f"loaded manga109: {len(df):,} rows from {args.manga109}")

    if args.heldout:
        heldout_paths = [Path(s.strip()) for s in args.heldout.split(",") if s.strip()]
    else:
        heldout_paths = sorted(DEFAULT_HELDOUT_DIR.glob("*.jsonl"))
    logger.info(f"held-out files: {[str(p.name) for p in heldout_paths]}")

    heldout_jp = _load_heldout_jp(heldout_paths)
    logger.info(f"held-out JP lines: {len(heldout_jp):,}")
    if not heldout_jp:
        logger.error("no held-out JP loaded; refusing to run (would be a no-op)")
        return 2

    # Build LSH on normalized held-out JP.
    num_perm = args.bands * args.rows
    lsh = MinHashLSH(
        threshold=args.threshold,
        num_perm=num_perm,
        params=(args.bands, args.rows),
    )
    heldout_norm: list[str] = []
    logger.info(f"indexing held-out (n-gram={args.ngram_size}, num_perm={num_perm})")
    for i, jp in enumerate(heldout_jp):
        norm = _normalize_jp(jp)
        heldout_norm.append(norm)
        if not norm:
            continue
        mh = _mk_minhash(norm, n=args.ngram_size, num_perm=num_perm, MinHash=MinHash)
        lsh.insert(f"ho{i}", mh)

    # Filter manga109.
    keep_mask: list[bool] = []
    drop_reasons: list[str] = []
    drop_lsh = 0
    drop_substr = 0
    logger.info(f"probing {len(df):,} manga109 rows")
    for row in df.iter_rows(named=True):
        jp = row["jp_text"] or ""
        norm = _normalize_jp(jp)
        if not norm:
            keep_mask.append(False)
            drop_reasons.append("empty_after_norm")
            continue
        # 1) LSH match.
        mh = _mk_minhash(norm, n=args.ngram_size, num_perm=num_perm, MinHash=MinHash)
        if lsh.query(mh):
            keep_mask.append(False)
            drop_reasons.append("lsh")
            drop_lsh += 1
            continue
        # 2) Substring backstop.  Both the held-out and the row's text must
        # be long enough after normalization, otherwise short particles like
        # `あ` (which appear thousands of times in manga held-out) match
        # every long bubble and we drop everything.  We only flag verbatim
        # copies: held-out fully contained in row, OR row fully contained
        # in held-out, AND BOTH sides >= substring_min_chars.
        if len(norm) >= args.substring_min_chars:
            hit = False
            for ho in heldout_norm:
                if len(ho) < args.substring_min_chars:
                    continue
                if norm in ho or ho in norm:
                    hit = True
                    break
            if hit:
                keep_mask.append(False)
                drop_reasons.append("substring")
                drop_substr += 1
                continue
        keep_mask.append(True)
        drop_reasons.append("")

    kept = df.filter(pl.Series(keep_mask))
    dropped_idx = [i for i, k in enumerate(keep_mask) if not k]
    logger.info(
        f"kept {len(kept):,} / {len(df):,} ({len(kept)/len(df):.1%}); "
        f"dropped lsh={drop_lsh:,} substring={drop_substr:,} "
        f"empty={len(dropped_idx) - drop_lsh - drop_substr}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept.write_parquet(args.output)
    logger.info(f"wrote {len(kept):,} rows to {args.output}")

    # Sample dropped rows.
    if dropped_idx:
        sample = dropped_idx[: args.n_dropped_samples]
        print("\n=== sample dropped rows ===", file=sys.stderr)
        for i in sample:
            row = df.row(i, named=True)
            print(
                f"  [{row['book']} p{row['page']}] reason={drop_reasons[i]} "
                f"JP: {row['jp_text']}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Load lmg-anon/VNTL-v2.5-1.6k-dpo-pairs into unified parquet (SFT use only).

The dataset (~8,988 rows) is a DPO preference set with three columns:

    prompt   : str — a windowed multi-turn <<JAPANESE>>/<<ENGLISH>> block with
                     speaker + fidelity metadata. The FINAL <<JAPANESE>> block
                     in the prompt is the line to translate; its trailing
                     ``<<ENGLISH>> (fidelity = ...)`` header is left empty.
    chosen   : str — the human-aligned EN translation of that final JP line.
    rejected : str — the dispreferred EN (IGNORED — this loader is for SFT).

Extraction strategy: we only keep the (jp, en) pair where ``jp`` is the source
text of the *last* <<JAPANESE>> block in ``prompt`` and ``en`` is ``chosen``.
We deliberately do NOT use ``parse_vntl_packed_text`` (which pairs already-filled
JP/EN turns) because the target turn's EN lives in ``chosen``, not the prompt.

The earlier filled turns in the prompt are context windows that repeat across
many rows, so harvesting them would create massive duplication; we skip them.

Tags: register_tag='vn_eroge', gold_flag=True, src='vntl_dpo:row{i}'.
"""

from __future__ import annotations

import argparse
import re
from typing import Iterator

from _cli_common import configure_logging, logger
from unify_schema import make_row, write_parquet


HF_REPO = "lmg-anon/VNTL-v2.5-1.6k-dpo-pairs"

# Match every "<<JAPANESE>>\n<jp text>\n<<ENGLISH>> (fidelity = ...)" block.
# The JP source we want is the FINAL such block in the prompt (the line whose
# translation lives in ``chosen``); we take the last match. Using a tempered
# pattern so JP text can't swallow the next marker.
_JP_BLOCK_RE = re.compile(
    r"<<JAPANESE>>\s*\n(?P<jp>(?:(?!<<JAPANESE>>|<<ENGLISH>>).)*?)\s*\n"
    r"<<ENGLISH>>(?:\s*\(fidelity[^)]*\))?",
    re.DOTALL,
)
_TRAILING_EOS_RE = re.compile(r"\s*</s>\s*$")


def _extract_target(prompt: str, chosen: str) -> tuple[str, str] | None:
    """Return (jp, en) for the final JP block paired with ``chosen``.

    The final <<JAPANESE>> block's EN slot is normally empty (the answer is in
    ``chosen``). If it happens to carry trailing text we ignore it and trust
    ``chosen`` as the canonical human-aligned target.
    """
    matches = list(_JP_BLOCK_RE.finditer(prompt))
    if not matches:
        return None
    jp = matches[-1].group("jp").strip()
    en = _TRAILING_EOS_RE.sub("", chosen or "").strip()
    if not jp or not en:
        return None
    return jp, en


def iter_rows(limit: int | None = None) -> Iterator[dict[str, object]]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error(
            "vntl_dpo: `datasets` not installed; add it to pyproject and retry. "
            f"TODO: pip install datasets. Original error: {e}"
        )
        return

    try:
        ds = load_dataset(HF_REPO, split="train", streaming=True)
    except Exception as e:
        logger.error(
            f"vntl_dpo: could not load {HF_REPO} ({type(e).__name__}: {e}). "
            "TODO: dataset may be gated/require auth or network access; skipping."
        )
        return

    logged_cols = False
    seen: set[tuple[str, str]] = set()
    emitted = 0
    skipped = 0
    for i, rec in enumerate(ds):
        if not logged_cols:
            logger.info(f"vntl_dpo: detected columns={list(rec.keys())}")
            logged_cols = True
        pair = _extract_target(rec.get("prompt") or "", rec.get("chosen") or "")
        if pair is None:
            skipped += 1
            continue
        jp, en = pair
        if (jp, en) in seen:
            continue
        seen.add((jp, en))
        emitted += 1
        yield make_row(
            jp=jp,
            en=en,
            src=f"vntl_dpo:row{i}",
            register_tag="vn_eroge",
            gold_flag=True,
        )
        if limit is not None and emitted >= limit:
            break
    logger.info(f"vntl_dpo: emitted={emitted} skipped={skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="backend/training/datasets/unified/vntl_dpo.parquet",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap emitted rows.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.dry_run:
        rows = list(iter_rows(limit=args.limit if args.limit else 50))
        print(f"vntl_dpo rows: {len(rows)}")
        for r in rows[:3]:
            print(f"  jp={r['jp']!r}\n  en={r['en']!r}\n  src={r['src']}")
        return
    n = write_parquet(iter_rows(limit=args.limit), args.out)
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()

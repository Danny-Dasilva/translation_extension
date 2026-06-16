"""Orchestrator — run stages 1-5 over a small sample and emit unified rows.

Writes ``backend/scripts/data/doujin/doujin_pairs.parquet`` in the unified
schema (jp, en, src, register_tag, gold_flag) consumed by the v12 builder.

SRC FORMAT CONTRACT (the v12 builder relies on this)
----------------------------------------------------
    src = "doujin:{workid}:p{page}:b{idx}"

    workid : slug for the matched JP<->EN work, no ':' chars
             gallery pair -> "g{jp_id}-{en_id}";  ubuca -> title(+artist) slug
    page   : 0-based aligned page index within the work
    idx    : 0-based bubble index within the page, in manga reading order
             (right-to-left, column-major, top-to-bottom)

Group rows by (workid, page) and order by idx to reconstruct page context.
register_tag = "nsfw_doujin", gold_flag = False (silver / OCR-mined).

STAGE 5 — QE filter
-------------------
LaBSE cosine adequacy gate (>= --labse-threshold, default 0.6) is the hard gate.
COMET is a SOFT score only (slang/NSFW register tanks COMET; never hard-gate on
it). Both degrade gracefully: if the model can't load, the gate is bypassed and
every pair is kept with qe_score=None and a warning — the interface is wired so
enabling it later is a one-line change.

NAS WARNING: all durable output goes to LOCAL disk under this directory. The
/mnt/nas/drive_2 CIFS share empties output dirs ~9 min after write.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from align_and_ocr import BubblePair, process_pair  # noqa: E402

LOCAL_OUT = Path(__file__).resolve().parent

# unified register tag for this source. unify_schema.make_row validates against
# its own set, so we build rows directly (nsfw_doujin is a doujin-specific tag).
REGISTER_TAG = "nsfw_doujin"


# --------------------------------------------------------------------------- #
# Stage 5: QE adequacy gate (LaBSE cosine) — graceful stub
# --------------------------------------------------------------------------- #


class LabseGate:
    """LaBSE cosine adequacy gate. Bypasses (keeps all) if model unavailable."""

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold
        self._model = None
        self._available = False
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("sentence-transformers/LaBSE")
            self._available = True
        except Exception as e:  # noqa: BLE001 - any load failure -> bypass
            print(f"[qe] LaBSE unavailable ({type(e).__name__}); gate bypassed")

    @property
    def available(self) -> bool:
        return self._available

    def score(self, jp: str, en: str) -> float | None:
        if not self._available or not jp or not en:
            return None
        import numpy as np

        emb = self._model.encode([jp, en], normalize_embeddings=True)
        return float(np.dot(emb[0], emb[1]))

    def keep(self, jp: str, en: str) -> tuple[bool, float | None]:
        s = self.score(jp, en)
        if s is None:
            return True, None  # bypass when no score
        return s >= self.threshold, s


# --------------------------------------------------------------------------- #
# pairing source loaders
# --------------------------------------------------------------------------- #


def load_candidate_pairs(out_dir: Path) -> list[dict]:
    """Load candidate_pairs.parquet (gallery phash matches) if present."""
    p = out_dir / "candidate_pairs.parquet"
    if not p.exists():
        return []
    return pl.read_parquet(p).to_dicts()


def bubble_to_row(b: BubblePair, qe_score: float | None) -> dict:
    """Map a BubblePair to a unified-schema row (+ qe_score sidecar column)."""
    return {
        "jp": b.jp,
        "en": b.en,
        "src": b.src,
        "register_tag": REGISTER_TAG,
        "gold_flag": False,
        "qe_score": qe_score,
    }


async def build(
    out_dir: Path,
    limit: int,
    max_pages: int,
    labse_threshold: float,
    keep_empty_en: bool,
) -> Path:
    pairs = load_candidate_pairs(out_dir)
    if not pairs:
        print(
            "[build] no candidate_pairs.parquet found. Run:\n"
            "  pair_galleries.py --source galleries --match-galleries "
            "--limit N --phash"
        )
    pairs = pairs[:limit]
    gate = LabseGate(labse_threshold)

    rows: list[dict] = []
    stats = {"pairs": 0, "bubbles": 0, "kept": 0, "empty_en": 0, "gated": 0}
    for cp in pairs:
        res = await process_pair(
            Path(cp["jp_path"]),
            Path(cp["en_path"]),
            cp["workid"],
            max_pages=max_pages,
        )
        stats["pairs"] += 1
        for b in res.bubbles:
            stats["bubbles"] += 1
            if not b.jp:
                continue
            if not b.en:
                stats["empty_en"] += 1
                if not keep_empty_en:
                    continue
            keep, score = gate.keep(b.jp, b.en)
            if not keep:
                stats["gated"] += 1
                continue
            rows.append(bubble_to_row(b, score))
            stats["kept"] += 1

    out_path = out_dir / "doujin_pairs.parquet"
    schema = {
        "jp": pl.Utf8,
        "en": pl.Utf8,
        "src": pl.Utf8,
        "register_tag": pl.Utf8,
        "gold_flag": pl.Boolean,
        "qe_score": pl.Float64,
    }
    df = pl.DataFrame(rows, schema=schema)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(
        f"[ok] wrote {len(df)} rows -> {out_path}\n"
        f"     stats: {stats}  (qe_gate={'on' if gate.available else 'BYPASSED'})"
    )
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Orchestrate doujin pair mining")
    ap.add_argument("--limit", type=int, default=2, help="max matched pairs")
    ap.add_argument("--max-pages", type=int, default=4)
    ap.add_argument("--labse-threshold", type=float, default=0.6)
    ap.add_argument(
        "--keep-empty-en",
        action="store_true",
        help="keep pairs with empty EN (e.g. when Latin OCR is stubbed)",
    )
    ap.add_argument("--out-dir", type=Path, default=LOCAL_OUT)
    args = ap.parse_args()

    asyncio.run(
        build(
            Path(args.out_dir),
            args.limit,
            args.max_pages,
            args.labse_threshold,
            args.keep_empty_en,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

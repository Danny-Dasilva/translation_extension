"""Build the REFUSAL PROBE set (gate signal #2).

Samples JP manga pages across a random subset of the local staging corpus, OCRs
the JP bubbles (CTD detect + PARSeq recognize, CPU-only), and emits a probe set
of page-context rows. ``refusal_eval.py`` then runs each row through the model
under test using the REAL production marked-line prompt and checks that the model
never refuses (hard gate: refusal_rate == 0).

The corpus is uniformly adult, so plain random sampling of galleries/pages is the
correct construction -- no explicitness filter exists or is needed; the point is
to stress the model on the same NSFW distribution it serves.

Each emitted row (one JSON object per line)::

    {
      "id":        "<en_gid>_<jp_gid>:<page_stem>",
      "gallery":   "<en_gid>_<jp_gid>",
      "page_img":  "/abs/path/to/jp/page.jpg",
      "lines":     ["<jp bubble 1>", ...],   # reading order, non-empty only
      "target_idx": <0-based index into lines>
    }

Reading order uses ``manga_reading_order`` from the v11 dataset builder (RTL,
column-major, top-to-bottom) when bboxes are available.

Run (CPU OCR -- fine to run fully)::

    cd backend && .venv/bin/python scripts/eval/build_refusal_probe.py \
        --galleries 50 --pages-per-gallery 6 --target-rows 250

Deterministic given ``--seed``. Not resumable, but re-running with a larger
``--galleries`` / ``--pages-per-gallery`` / ``--target-rows`` / ``--max-minutes``
extends coverage (the same seed reproduces the same sample prefix).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parents[1]
for p in (
    str(_BACKEND),
    str(_BACKEND / "scripts" / "data" / "corpus_bitext"),
    str(_BACKEND / "scripts" / "data" / "v11"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_v11_dataset import manga_reading_order  # type: ignore  # noqa: E402

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _has_jp(text: str) -> bool:
    """True if the string contains hiragana / katakana / CJK ideographs."""
    for ch in text:
        o = ord(ch)
        if (
            0x3040 <= o <= 0x309F  # hiragana
            or 0x30A0 <= o <= 0x30FF  # katakana
            or 0x4E00 <= o <= 0x9FFF  # CJK unified
            or 0x3400 <= o <= 0x4DBF  # CJK ext A
        ):
            return True
    return False


def _jp_len(text: str) -> int:
    return sum(1 for ch in text if _has_jp(ch))


def _order_bubbles(bubbles: list[dict]) -> list[dict]:
    """Return non-empty bubbles in manga reading order.

    ``ocr_jp_page`` emits bbox as {minX,minY,maxX,maxY}; ``manga_reading_order``
    consumes {xmin,ymin,xmax,ymax}. We adapt, order, then read ``text`` back.
    """
    rows = []
    for b in bubbles:
        txt = (b.get("text") or "").strip()
        if not txt:
            continue
        bb = b["bbox"]
        rows.append(
            {
                "text": txt,
                "xmin": bb["minX"],
                "ymin": bb["minY"],
                "xmax": bb["maxX"],
                "ymax": bb["maxY"],
            }
        )
    if not rows:
        return []
    return manga_reading_order(rows)


def _pick_target(lines: list[str]) -> int:
    """Pick the most-substantive line (most JP chars) as the marked line.

    The longest JP line is the most likely to be real dialogue (and, on this
    corpus, the most likely to carry the explicit content that would provoke a
    refusal), so it is the strongest probe. Deterministic tie-break by index.
    """
    best_i, best_len = 0, -1
    for i, ln in enumerate(lines):
        n = _jp_len(ln)
        if n > best_len:
            best_i, best_len = i, n
    return best_i


def _window(lines: list[str], target_idx: int, cap: int) -> tuple[list[str], int]:
    """Cap the page to ``cap`` lines, keeping a contiguous window around target.

    Production serves a bounded page context (MAX_BUBBLES_CONTEXT=12); we keep a
    slightly larger window so the probe stays realistic without blowing the
    trained max_seq. Returns (windowed_lines, new_target_idx).
    """
    if len(lines) <= cap:
        return lines, target_idx
    half = cap // 2
    start = max(0, target_idx - half)
    end = start + cap
    if end > len(lines):
        end = len(lines)
        start = end - cap
    return lines[start:end], target_idx - start


def _list_pages(gallery_dir: Path) -> list[Path]:
    if not gallery_dir.exists():
        return []
    return sorted(
        p for p in gallery_dir.iterdir() if p.suffix.lower() in _IMG_EXTS
    )


async def build(args: argparse.Namespace) -> int:
    from ocr_adapters import build_jp_engines, ocr_jp_page  # type: ignore

    rng = random.Random(args.seed)
    pairs = [json.loads(l) for l in args.pairs.open() if l.strip()]
    if not pairs:
        print(f"no pairs in {args.pairs}", file=sys.stderr)
        return 1

    n_gal = min(args.galleries, len(pairs))
    sampled = rng.sample(pairs, n_gal)

    # Build a round-robin task list of (gallery_tag, page_path) so an early
    # time cutoff still yields diverse gallery coverage.
    per_gallery_tasks: list[list[tuple[str, Path]]] = []
    for pr in sampled:
        tag = f"{pr['en_gid']}_{pr['jp_gid']}"
        pages = _list_pages(Path(pr["jp_dir"]))
        if not pages:
            continue
        k = min(args.pages_per_gallery, len(pages))
        chosen = rng.sample(pages, k)
        per_gallery_tasks.append([(tag, pg) for pg in chosen])

    tasks: list[tuple[str, Path]] = []
    for i in range(args.pages_per_gallery):
        for g in per_gallery_tasks:
            if i < len(g):
                tasks.append(g[i])

    print(
        f"sampled {len(per_gallery_tasks)} galleries, {len(tasks)} candidate pages; "
        f"target_rows={args.target_rows} min_bubbles={args.min_bubbles}",
        file=sys.stderr,
    )

    detector, ocr = build_jp_engines(cpu_only=True)

    rows: list[dict] = []
    seen_pages: set[str] = set()
    galleries_hit: set[str] = set()
    n_ocr = 0
    n_skip_few = 0
    t0 = time.time()
    deadline = t0 + args.max_minutes * 60

    args.out.parent.mkdir(parents=True, exist_ok=True)

    for tag, page_path in tasks:
        if len(rows) >= args.target_rows:
            break
        if time.time() > deadline:
            print(
                f"[time] hit --max-minutes={args.max_minutes}; stopping with "
                f"{len(rows)} rows",
                file=sys.stderr,
            )
            break
        pid = f"{tag}:{page_path.stem}"
        if pid in seen_pages:
            continue
        seen_pages.add(pid)
        try:
            bubbles = await ocr_jp_page(page_path, detector, ocr)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] JP OCR failed {page_path}: {e}", file=sys.stderr)
            continue
        n_ocr += 1
        ordered = _order_bubbles(bubbles)
        # Require some JP content on the lines we keep.
        lines = [b["text"] for b in ordered if _jp_len(b["text"]) >= 1]
        if len(lines) < args.min_bubbles:
            n_skip_few += 1
            continue
        target_idx = _pick_target(lines)
        lines, target_idx = _window(lines, target_idx, args.max_lines)
        rows.append(
            {
                "id": pid,
                "gallery": tag,
                "page_img": str(page_path),
                "lines": lines,
                "target_idx": target_idx,
            }
        )
        galleries_hit.add(tag)
        if n_ocr % 20 == 0:
            print(
                f"  ocr={n_ocr} rows={len(rows)} skip_few={n_skip_few} "
                f"elapsed={time.time() - t0:.0f}s",
                file=sys.stderr,
            )

    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    bubble_counts = [len(r["lines"]) for r in rows]
    elapsed = time.time() - t0
    print(
        json.dumps(
            {
                "out": str(args.out),
                "rows": len(rows),
                "galleries": len(galleries_hit),
                "pages_ocred": n_ocr,
                "pages_skipped_few_bubbles": n_skip_few,
                "bubbles_min": min(bubble_counts) if bubble_counts else 0,
                "bubbles_max": max(bubble_counts) if bubble_counts else 0,
                "bubbles_mean": (
                    round(sum(bubble_counts) / len(bubble_counts), 2)
                    if bubble_counts
                    else 0
                ),
                "elapsed_sec": round(elapsed, 1),
                "seed": args.seed,
            },
            indent=2,
        )
    )
    if len(rows) < args.target_rows:
        print(
            "\nTo extend: re-run with a larger --galleries / --pages-per-gallery / "
            "--target-rows / --max-minutes (same --seed reproduces this prefix).",
            file=sys.stderr,
        )
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pairs",
        type=Path,
        default=Path("/home/danny/manga_corpus_staging/available_pairs.jsonl"),
    )
    ap.add_argument("--out", type=Path, default=_HERE / "refusal_probe.jsonl")
    ap.add_argument("--galleries", type=int, default=50)
    ap.add_argument("--pages-per-gallery", type=int, default=6)
    ap.add_argument("--target-rows", type=int, default=250)
    ap.add_argument("--min-bubbles", type=int, default=3)
    ap.add_argument(
        "--max-lines",
        type=int,
        default=16,
        help="cap page context to a window of this many lines around the target",
    )
    ap.add_argument("--max-minutes", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1337)
    return ap


if __name__ == "__main__":
    raise SystemExit(asyncio.run(build(build_argparser().parse_args())))

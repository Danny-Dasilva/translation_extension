"""Synthesize 'JP OCR noise -> en="..."' garbage-refusal training examples from
our own OCR pipeline output at /home/danny/manga-output/*/stats.json.

Signal: bubbles where the JP has low Japanese-character ratio (<0.5) or strong
repetition patterns typical of PARSeq failure modes.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unify_schema import make_row, write_parquet  # noqa: E402

SEARCH_ROOTS = [
    "/home/danny/manga-output/644289",
]
OUT = Path("backend/training/datasets/filtered/ocr_garbage.parquet")

JP_CHAR = re.compile(r"[぀-ヿ一-鿿々〆〇]")


def is_garbage_jp(jp: str) -> bool:
    """Heuristics for 'OCR noise likely' — mirror what Gemma refuses on."""
    if not jp or len(jp) < 2:
        return True
    # JA char ratio < 0.4 → garbage (numbers, punctuation, latin leak)
    total_alnum = sum(1 for c in jp if c.isalnum() or JP_CHAR.match(c))
    if total_alnum == 0:
        return True
    ja_count = sum(1 for c in jp if JP_CHAR.match(c))
    if ja_count / max(1, total_alnum) < 0.4:
        return True
    # Single char repeated 4+ times (PARSeq stuttering)
    if re.search(r"(.)\1{3,}", jp) and len(set(jp)) <= 3:
        return True
    # Same 2-3 char block repeated ("ちんぼちんぼ")
    if re.search(r"(.{2,3})\1{2,}", jp) and len(set(jp)) <= 4:
        return True
    return False


def main() -> int:
    rows = []
    seen = set()
    for root_str in SEARCH_ROOTS:
        root = Path(root_str)
        if not root.exists():
            continue
        for stats_p in sorted(root.rglob("stats.json")):
            try:
                with open(stats_p) as f:
                    stats = json.load(f)
            except Exception:
                continue
            for bi, jp in enumerate(stats.get("ocr_samples") or []):
                jp = (jp or "").strip()
                if not jp or jp in seen:
                    continue
                if not is_garbage_jp(jp):
                    continue
                seen.add(jp)
                rows.append(
                    make_row(
                        jp=jp,
                        en="...",
                        src=f"ocr_garbage:{stats_p.parent.name}:{bi}",
                        register_tag="garbage",
                        gold_flag=True,
                    )
                )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(iter(rows), OUT)
    print(f"wrote {len(rows)} OCR-garbage examples -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

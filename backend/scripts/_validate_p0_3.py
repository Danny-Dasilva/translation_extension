#!/usr/bin/env python3
"""Validate P0-3 character-name canonicalizer against the full-pipeline bench.

The v11 model is deterministic at temp 0, so post-editing the *stored* output
is equivalent to regenerating it. This loads representative bench pages, applies
``canonicalize_names(translation_en, ocr_jp)`` to every bubble, and prints
``jp | before | after`` for every bubble that CHANGES.

Read-only: only the bench JSON outside the worktree is touched.

Run:
    /home/danny/Documents/personal/extension/backend/.venv/bin/python \
        scripts/_validate_p0_3.py
"""

from __future__ import annotations

import json
import os
import sys

# Allow `from app...` when run from backend/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.name_glossary import canonicalize_names  # noqa: E402

BENCH = (
    "/home/danny/Documents/personal/extension/backend/.bench/"
    "full_pipeline/588828_mesu2_insp"
)

# Pages called out in the task as carrying the known corruptions.
PAGES = ["044", "041", "069", "039"]


def main() -> int:
    any_change = False
    print(f"{'pg':>3} | {'jp':<34} | before -> after")
    print("-" * 100)
    for page in PAGES:
        path = os.path.join(BENCH, page, "bubbles.json")
        with open(path, encoding="utf-8") as fh:
            bubbles = json.load(fh)
        for b in bubbles:
            jp = b.get("ocr_jp") or ""
            before = b.get("translation_en")
            if not before:
                continue
            after = canonicalize_names(before, jp)
            if after != before:
                any_change = True
                jp_disp = (jp[:31] + "...") if len(jp) > 34 else jp
                print(f"{page:>3} | {jp_disp:<34} | {before!r}")
                print(f"{'':>3} | {'':<34} |   -> {after!r}")
                # Idempotency check.
                assert canonicalize_names(after, jp) == after, "NOT idempotent!"
    print("-" * 100)
    if not any_change:
        print("NO CHANGES — something is wrong, expected the known corruptions.")
        return 1
    print("All changes above are name corrections; idempotency verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

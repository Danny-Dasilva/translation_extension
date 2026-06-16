"""Validation harness for Fix P2-2 (SFX onomatopoeia post-processing).

NO GPU. Loads stored deterministic model output (bubbles.json) from the
benchmark run, applies ``clean_sfx_output(translation_en, ocr_jp)``, and prints
a ``jp | before | after`` table for every CHANGED bubble.

Stored model output == regenerated result (the v11 model is deterministic), so
post-editing the stored ``translation_en`` is equivalent to post-editing a live
result.

Source data is READ-ONLY:
    backend/.bench/full_pipeline/588828_mesu2_insp/NNN/bubbles.json

Run:
    cd backend && /path/to/.venv/bin/python scripts/_validate_p2_2.py
"""

from __future__ import annotations

import json
import os
import sys

# Make `app` importable when run from backend/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.sfx_glossary import clean_sfx_output, is_sfx_meta_description  # noqa: E402

BENCH = (
    "/home/danny/Documents/personal/extension/backend/.bench/"
    "full_pipeline/588828_mesu2_insp"
)

REPRESENTATIVE = ["030", "037", "060", "061"]


def _load(page: str):
    path = os.path.join(BENCH, page, "bubbles.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    pages = sys.argv[1:] or REPRESENTATIVE
    total_changed = 0
    leaks_remaining = 0

    for page in pages:
        try:
            bubbles = _load(page)
        except FileNotFoundError:
            print(f"[skip] page {page}: no bubbles.json")
            continue

        header_printed = False
        for b in bubbles:
            jp = b.get("ocr_jp")
            before = b.get("translation_en")
            after = clean_sfx_output(before, jp)
            if after != before:
                if not header_printed:
                    print(f"\n=== PAGE {page} ===")
                    print(f"{'jp':<14} | {'before':<60} | after")
                    print("-" * 100)
                    header_printed = True
                total_changed += 1
                print(f"{(jp or ''):<14} | {(str(before)):<60} | {after!r}")
                # any residual meta-description after cleaning is a failure
                if after and is_sfx_meta_description(after):
                    leaks_remaining += 1

    # Post-condition: no "SFX for ..." leak survives ANY page.
    print("\n=== GLOBAL LEAK SCAN (all pages) ===")
    surviving = 0
    for page in sorted(os.listdir(BENCH)):
        p = os.path.join(BENCH, page, "bubbles.json")
        if not os.path.isfile(p):
            continue
        for b in json.load(open(p, encoding="utf-8")):
            after = clean_sfx_output(b.get("translation_en"), b.get("ocr_jp"))
            if after and is_sfx_meta_description(after):
                surviving += 1
                print(f"  LEAK SURVIVED page {page}: {after!r}")
    if surviving == 0:
        print("  OK — no 'SFX for ...' meta-descriptions survive on any page.")

    print(f"\nChanged bubbles (representative pages): {total_changed}")
    print(f"Residual meta leaks (representative):    {leaks_remaining}")
    print(f"Residual meta leaks (all pages):         {surviving}")
    return 1 if (leaks_remaining or surviving) else 0


if __name__ == "__main__":
    raise SystemExit(main())

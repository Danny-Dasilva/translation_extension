#!/usr/bin/env python
"""P1-2 validation: explicit-register post-edit (potential glossary).

NO GPU. Deterministic. Reads the stored full-pipeline bench output (the
post-edit applied to a stored translation == what we'd get on a regenerated
run, because restore_register is a pure function of (en, jp)).

For the representative pages it prints:
    jp | before(model EN) | after(restore_register) | human-ref line

Representative pages:
    031 — 潮まみれ -> model said "seawater"; human keeps explicit register.
    064 — チンコ comparison the model DROPPED (unrecoverable by substitution).

Usage:
    .venv/bin/python scripts/_validate_p1_2.py
"""

import json
import sys
from pathlib import Path

# Make `app` importable when run from backend/.
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.services.register_glossary import restore_register  # noqa: E402

# READ-ONLY bench data lives in the MAIN checkout (shared, not the worktree).
BENCH = Path(
    "/home/danny/Documents/personal/extension/backend/.bench/full_pipeline/588828_mesu2_insp"
)

# Human reference lines transcribed from the .webp images (read-only).
HUMAN_REF = {
    ("031", "リビングが潮まみれじゃん"): "FLOOD YOUR FUCKING LIVING ROOM, LOL!",
    ("064", "こっちのこっちのチンコのがセックスイイわよっ!!のが...!"):
        "SEX WITH YOU IS THE BEST! YOUR DICK IS SO MUCH BETTER THAN HIS!",
}


def load(page: str):
    f = BENCH / page / "bubbles.json"
    return json.loads(f.read_text())


def main() -> int:
    changed = 0
    print(f"{'PAGE':<5} {'CHG':<4} JP | BEFORE | AFTER | HUMAN-REF")
    print("-" * 100)
    for page in ("031", "064"):
        for it in load(page):
            jp = it.get("ocr_jp") or ""
            before = it.get("translation_en")
            after = restore_register(before, jp)
            chg = "*" if after != before else ""
            if chg:
                changed += 1
            ref = HUMAN_REF.get((page, jp), "")
            # Only print lines that changed OR have a human ref (the targets),
            # to keep the table focused.
            if chg or ref:
                print(
                    f"{page:<5} {chg:<4} {jp!r}\n"
                    f"      before: {before!r}\n"
                    f"      after : {after!r}\n"
                    f"      human : {ref!r}\n"
                )

    print("-" * 100)
    print(f"Lines changed by restore_register: {changed}")

    # Assert the headline win + clean-line safety.
    p031 = load("031")
    shio = next(b for b in p031 if "潮" in (b.get("ocr_jp") or ""))
    after_shio = restore_register(shio["translation_en"], shio["ocr_jp"])
    clean = next(b for b in p031 if b.get("ocr_jp") == "あーあーすげぇな")
    after_clean = restore_register(clean["translation_en"], clean["ocr_jp"])

    ok_shio = "squirt" in (after_shio or "").lower() and "seawater" not in (after_shio or "").lower()
    ok_clean = after_clean == clean["translation_en"]

    print(f"\n[031 潮 line]  squirt-based & no seawater : {'PASS' if ok_shio else 'FAIL'}  -> {after_shio!r}")
    print(f"[031 clean]    unchanged                  : {'PASS' if ok_clean else 'FAIL'}  -> {after_clean!r}")
    return 0 if (ok_shio and ok_clean) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Replay-validation for FIX P1-1 (JP-plausibility OCR gate).

NO GPU. Loads stored per-page bubbles.json from the full-pipeline bench output,
re-runs the OLD gate (plausibility OFF) vs the NEW gate (plausibility ON,
default) on each (ocr_jp, ocr_conf), and prints every DECISION FLIP.

Success criterion:
  * garbled title/credit lines (070, 071, 074) flip KEPT -> DROPPED
  * real dialogue (005/021/050 ...) stays KEPT (no KEPT -> DROPPED flips there)

Run:
  cd backend && ../.venv/bin/python scripts/_validate_p1_1.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.ocr_confidence_gate import is_garbled_low_conf  # noqa: E402

BENCH = (
    "/home/danny/Documents/personal/extension/backend/"
    ".bench/full_pipeline/588828_mesu2_insp"
)

# Pages of interest: known garbles + clean-dialogue controls.
FOCUS = {"070", "071", "074", "005", "021", "050"}


def decide(text: str, conf: float, *, plausibility: bool) -> bool:
    """Return True if the gate DROPS this bubble."""
    return is_garbled_low_conf(text, conf, check_plausibility=plausibility)


def main() -> int:
    pages = sorted(glob.glob(os.path.join(BENCH, "*", "bubbles.json")))
    if not pages:
        print(f"NO DATA at {BENCH}", file=sys.stderr)
        return 2

    flips = []  # (page, conf, old_drop, new_drop, jp, en)
    total = 0
    new_drops = 0
    for path in pages:
        page = os.path.basename(os.path.dirname(path))
        for b in json.load(open(path)):
            jp = b.get("ocr_jp", "") or ""
            conf = float(b.get("ocr_conf", 1.0))
            total += 1
            old = decide(jp, conf, plausibility=False)
            new = decide(jp, conf, plausibility=True)
            if new:
                new_drops += 1
            if old != new:
                flips.append(
                    (page, conf, old, new, jp, (b.get("translation_en") or "")[:48])
                )

    print(f"corpus: {total} bubbles across {len(pages)} pages")
    print(f"OLD gate drops: {sum(1 for p in pages for b in json.load(open(p)) if decide(b.get('ocr_jp','') or '', float(b.get('ocr_conf',1.0)), plausibility=False))}")
    print(f"NEW gate drops: {new_drops}")
    print(f"DECISION FLIPS: {len(flips)}\n")

    def show(rows, title):
        print(title)
        if not rows:
            print("   (none)")
            return
        for page, conf, old, new, jp, en in rows:
            old_s = "DROP" if old else "KEEP"
            new_s = "DROP" if new else "KEEP"
            star = "  <-- FOCUS" if page in FOCUS else ""
            print(f"   p{page} conf={conf:.3f} {old_s}->{new_s}{star}")
            print(f"        jp = {jp!r}")
            print(f"        en = {en!r}")

    k2d = [f for f in flips if not f[2] and f[3]]  # KEEP -> DROP (newly caught)
    d2k = [f for f in flips if f[2] and not f[3]]  # DROP -> KEEP (regressions)
    show(k2d, "KEPT -> DROPPED (newly caught garble):")
    print()
    show(d2k, "DROPPED -> KEPT (regressions — must be empty):")

    # Verdict
    print("\n--- verdict ---")
    focus_caught = {f[0] for f in k2d if f[0] in {"070", "071", "074"}}
    control_harmed = [f for f in k2d if f[0] in {"005", "021", "050"}]
    ok = bool(focus_caught) and not control_harmed and not d2k
    print(f"garble pages newly caught: {sorted(focus_caught)}")
    print(f"control dialogue harmed:   {[f[0] for f in control_harmed] or 'none'}")
    print(f"regressions (DROP->KEEP):  {len(d2k)}")
    print("RESULT:", "PASS" if ok else "REVIEW")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

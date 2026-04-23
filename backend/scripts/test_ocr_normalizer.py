#!/usr/bin/env python3
"""Pure-string unit tests for the PARSeq OCR postprocess normalizer.

No ONNX model needed. Exercises:
- Zero-width char stripping
- Smart/curly quote mapping
- Dash -> prolonged sound mark (ー)
- Fullwidth punctuation -> halfwidth
- Fullwidth alnum -> halfwidth
- Halfwidth katakana -> fullwidth
- Ligature punctuation (‼ ⁇ ⁈ ⁉)
- Fullwidth tilde -> wave dash
- Ideographic space -> ASCII space
- Horizontal ellipsis -> "..."
- Middle-dot runs (・・・+) -> "..."
- Trailing-repeat capping
- Mid-text repeat capping

Run:
    cd backend && uv run python scripts/test_ocr_normalizer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure "app" is importable when running from backend/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.ocr_postprocess import apply_all, normalize_text, strip_trailing_repeats  # noqa: E402


CASES: list[tuple[str, str, str]] = [
    # (description, input, expected)
    (
        "smart double quotes",
        "“こんにちは”",
        '"こんにちは"',
    ),
    (
        "smart single quotes",
        "‘test’",
        "'test'",
    ),
    (
        "em dash -> prolonged sound mark",
        "ああ—",
        "ああー",
    ),
    (
        "en dash -> prolonged sound mark",
        "ああ–",
        "ああー",
    ),
    (
        "horizontal ellipsis (trailing dots capped by strip_trailing_repeats)",
        "そう…",
        # … -> "..." then trailing-repeat cap (max_trailing=2) trims to "..".
        "そう..",
    ),
    (
        "fullwidth exclamation and question",
        "やった！？",
        "やった!?",
    ),
    (
        "ligature double exclaim / interrobang",
        "‼⁇⁈⁉",
        "!!???!!?",
    ),
    (
        "fullwidth tilde -> wave dash",
        "もう～",
        "もう〜",
    ),
    (
        "ideographic space -> ASCII space",
        "a　b",
        "a b",
    ),
    (
        "zero-width chars stripped",
        "ab​c‌d﻿e",
        "abcde",
    ),
    (
        "fullwidth alnum -> halfwidth",
        "ＡＢｃ０９",
        "ABc09",
    ),
    (
        "halfwidth katakana -> fullwidth",
        "ｶﾀﾅ",  # ｶﾀﾅ
        "カタナ",
    ),
    (
        "middle-dot run -> ...",
        "あ・・・・・い",
        "あ...い",
    ),
    (
        "trailing repeat cap (ー many)",
        "そうだーーーーーーー",
        # mid-cap=3 collapses the 7 ーs down to 3, trailing-cap=2 then trims to 2.
        "そうだーー",
    ),
    (
        "trailing ! repeat cap",
        "やった！！！！！！",
        "やった!!",
    ),
    (
        "mid-text repeat cap (5 あ then suffix)",
        "あああああい",
        "あああい",
    ),
    (
        "combined: smart quote + em dash + interior !",
        # Closing ” means the run of ！ is mid-text, so mid-cap=3 applies.
        "“そう—！！！！”",
        '"そうー!!!"',
    ),
]


def main() -> int:
    failures: list[str] = []
    print("=" * 72)
    print("PARSeq OCR postprocess normalizer test")
    print("=" * 72)
    for desc, inp, expected in CASES:
        got = apply_all(inp)
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {desc}")
        print(f"       in:       {inp!r}")
        print(f"       expected: {expected!r}")
        print(f"       got:      {got!r}")
        if not ok:
            failures.append(desc)
        print()

    # Smoke-test the individual primitives too.
    assert normalize_text("") == ""
    assert strip_trailing_repeats("a") == "a"
    assert strip_trailing_repeats("") == ""
    assert strip_trailing_repeats("aaaaaa", max_trailing=2, max_mid=3) == "aa"

    print("-" * 72)
    if failures:
        print(f"{len(failures)} / {len(CASES)} FAILED:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print(f"All {len(CASES)} cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

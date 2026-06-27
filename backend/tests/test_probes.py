"""Unit tests for backend.scripts.eval.probes.

Covers each of the 8 probes with at least one passing and one failing row.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the backend package importable when pytest is run from repo root.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR.parent))

from backend.scripts.eval.probes import (  # noqa: E402
    detect_high_repetition,
    run_probes,
)


def _row(probe_type: str, jp: str, en_pred: str) -> dict:
    return {"probe_type": probe_type, "jp": jp, "en_pred": en_pred}


# ---------------------------------------------------------------------------
# detect_high_repetition()
# ---------------------------------------------------------------------------


def test_detect_high_repetition_triggers_on_long_consecutive_substring() -> None:
    assert detect_high_repetition("abcde" * 30) is True


def test_detect_high_repetition_ignores_normal_text() -> None:
    assert detect_high_repetition("The quick brown fox jumps over the lazy dog.") is False


def test_detect_high_repetition_triggers_on_token_loop() -> None:
    assert detect_high_repetition(("hello " * 30).strip()) is True


# ---------------------------------------------------------------------------
# run_probes() - one pass row + one fail row per probe
# ---------------------------------------------------------------------------


def test_run_probes_full_matrix() -> None:
    rows = [
        # 1. name
        _row("name", "佐藤さんが来た。", "Sato came."),                # pass
        _row("name", "佐藤さんが来た。", "Satoshi came."),            # fail: substring 'sato' IS in 'satoshi' -> pass! use different surname
        _row("name", "田中さんが来た。", "He came."),                  # fail: no Tanaka
        # 2. honorific
        _row("honorific", "鈴木さん", "Mr. Suzuki"),                   # pass
        _row("honorific", "鈴木さん", "Hi Suzuki-san, how are you?"),  # fail
        # 3. curly
        _row("curly", "こんにちは", 'He said "hi".'),                  # pass (straight quotes)
        _row("curly", "こんにちは", "He said “hi”."),                  # fail (curly)
        # 4. repeat
        _row("repeat", "こんにちは", "Hello there friend."),           # pass
        _row("repeat", "こんにちは", "hello " * 40),                   # fail
        # 5. refusal
        _row("refusal", "アダルトな内容", "She looked at him coldly."),  # pass
        _row("refusal", "アダルトな内容", "I can't help with that."),    # fail
        # 6. length
        _row("length", "あいうえお", "Hello there!"),                  # pass: ratio 12/5 = 2.4
        _row("length", "あ", "x" * 100),                                # fail: ratio 100
        # 7. sfx
        _row("sfx", "ドキドキする", "My heart went thump."),           # pass
        _row("sfx", "ドキドキする", "My heart was beating fast."),     # fail (no 'thump')
        # 8. idiom
        _row("idiom", "一石二鳥だ", "It was two birds with one stone."),  # pass
        _row("idiom", "一石二鳥だ", "It was convenient."),                # fail
    ]
    # Note: row #2 above actually passes because 'sato' is a substring of
    # 'satoshi'. Replace it to be an unambiguous failure.
    rows[1] = _row("name", "鈴木さんが来た。", "He came.")  # Suzuki missing -> fail

    result = run_probes(rows)

    counts = result.per_probe_counts
    # Every probe above has 2 rows, exactly one pass and one fail...
    # except 'name' which has 3 rows (2 fail, 1 pass).
    assert counts["name"]["n"] == 3
    assert counts["name"]["pass"] == 1
    assert counts["name"]["fail"] == 2

    for probe in ("honorific", "curly", "repeat", "refusal", "length", "sfx", "idiom"):
        assert counts[probe]["n"] == 2, f"{probe} count"
        assert counts[probe]["pass"] == 1, f"{probe} pass"
        assert counts[probe]["fail"] == 1, f"{probe} fail"

    # overall_pass must be False because every category has at least one fail
    # and therefore cannot meet the (strict) target.
    assert result.overall_pass is False


def test_run_probes_all_pass_meets_targets() -> None:
    rows = [
        _row("name", "田中さんが来た。", "Tanaka came."),
        _row("honorific", "先輩", "Upperclassman arrived."),
        _row("curly", "こんにちは", "Hi."),
        _row("repeat", "こんにちは", "Hello there."),
        _row("refusal", "あいさつ", "Hello."),
        _row("length", "あいうえおかきくけこ", "Hello everyone!"),
        _row("sfx", "ドキドキ", "thump"),
        _row("idiom", "一石二鳥", "two birds with one stone"),
    ]
    result = run_probes(rows)
    # Only assert on probes that actually had rows (others are nan with n=0).
    for probe, rate in result.per_probe.items():
        if result.per_probe_counts[probe]["n"] == 0:
            continue
        assert rate == 1.0, f"{probe} rate={rate}"
    assert result.overall_pass is True


def test_run_probes_regressions_vs_baseline() -> None:
    rows = [
        # name: 0/2 pass = 0.0 vs baseline 1.0 = -100pp regression
        _row("name", "田中さんが来た。", "He came."),
        _row("name", "鈴木さんが来た。", "They arrived."),
    ]
    baseline = {"name": 1.0}
    result = run_probes(rows, baseline=baseline)
    assert "name" in result.regressions_vs_baseline
    assert result.regressions_vs_baseline["name"] < -5.0


def test_run_probes_empty_rows_skip_unknown_probe_types() -> None:
    rows = [{"probe_type": "unknown", "jp": "x", "en_pred": "y"}]
    result = run_probes(rows)
    assert result.per_probe_counts["name"]["n"] == 0

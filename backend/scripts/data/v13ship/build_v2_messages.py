#!/usr/bin/env python3
"""Convert the v2 5-col mix parquet -> trainer messages JSONL.

The 30B trainer (sft_qwen3vl_8b_imagectx.py::build_conversations) consumes a
JSONL of ``{"messages": [...]}`` records, NOT the 5-col parquet that
compose_training_mix.py emits. This reuses the EXACT production record builder
(``row_to_message`` / ``_write_messages_jsonl``) from the v13ship text-SFT
builder so the message shape is byte-identical to v1's train set — the
train/serve format landmine is avoided by construction, not by re-implementation.

Usage:
  python scripts/data/v13ship/build_v2_messages.py \
    --parquet scripts/data/v13ship/data_v2_30b_mix.parquet \
    --out     scripts/data/v13ship/data_v2_30b_messages.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from scripts.data.v13ship.build_textsft_refusalstripped import (
    row_to_message,
    _write_messages_jsonl,
    messages_format_check,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="scripts/data/v13ship/data_v2_30b_mix.parquet")
    ap.add_argument("--out", default="scripts/data/v13ship/data_v2_30b_messages.jsonl")
    args = ap.parse_args()

    rows = pl.read_parquet(args.parquet).to_dicts()

    # _write_messages_jsonl applies row_to_message internally -> pass raw 5-col rows.
    out = Path(args.out)
    n = _write_messages_jsonl(rows, out)

    # Validate the schema + byte-exact round-trip (user text == prompt, asst == en).
    msgs = [row_to_message(r) for r in rows]
    check = messages_format_check(msgs, rows)
    print(f"wrote {n} rows -> {out}")
    print(f"format check: {check}")
    if not (check["valid"] and check["roundtrip_ok"] and check["has_image_all_false"]):
        raise SystemExit(f"FAIL: messages format check did not fully pass: {check}")
    print("OK — schema valid, byte-exact round-trip, text-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

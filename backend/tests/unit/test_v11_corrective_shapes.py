"""ITEMS 1-3: data-shape changes for the dominant model buckets.

These tests exercise the DATA BUILDERS only (no GPU, no parquet read). They use
tiny in-memory inputs so they run via the main venv in milliseconds.

ITEM 1 (fix6 shape fix): build_v11_dataset.corrective_rows() must be able to
emit a FRACTION of corrective rows in PAGE-CONTEXT shape (build_context_prompt)
with real surrounding JP lines — because the gender/speaker-inversion failures
ONLY manifest in page-context shape. Plain-only corrective rows cannot move that
bucket. Shape fraction is configurable.

ITEM 2 (reverse-sense corrective data): a builder that, for each reverse-sense
lexeme (締まる->'closing' etc.), emits 2-3 VARIED JP carriers in BOTH plain and
page-context shape, each with a contrastive-margin field and human_en left as a
TODO (never fabricated). NSFW fraction stays FLAT (no oversampling).

ITEM 3 (voice/addressee probe): a structured JSONL of causative-passive
(させられる) and 2nd<->1st-person command-inversion patterns WITH gold targets,
so a future SFT/eval can target grammatical voice.
"""

from __future__ import annotations

import importlib

import pytest

build_v11 = importlib.import_module("v11.build_v11_dataset")
reverse_sense = importlib.import_module("v11.build_reverse_sense_corrective")
voice = importlib.import_module("v11.build_voice_addressee_probe")


# ===========================================================================
# ITEM 1: fix6 page-context corrective shape
# ===========================================================================

def _corr_records():
    """A few corrective seed dicts mimicking the parquet row schema."""
    return [
        {
            "jp": "締まってきた",
            "en": "It's getting tighter",
            "src": "corrective_v11:cleanocr:050_0",
            "register_tag": "vn_eroge",
            "gold_flag": True,
            # page-context support fields (NEW; optional for plain rows):
            "context_jp": ["どう？", "締まってきた", "もっと"],
            "context_k": 1,
        },
        {
            "jp": "おかえり",
            "en": "Welcome back",
            "src": "corrective_v11:cleanocr:051_3",
            "register_tag": "manga_dialog",
            "gold_flag": True,
            "context_jp": ["ただいま", "おかえり"],
            "context_k": 1,
        },
    ]


def test_corrective_rows_plain_only_by_default_unchanged_shape():
    # Backward-compatible default: pagectx fraction 0 => all plain (old behavior).
    rows = build_v11.corrective_rows(_corr_records(), pagectx_frac=0.0)
    assert all(r["prompt"].startswith(build_v11.PLAIN_INSTR) for r in rows)
    assert all("Page:" not in r["prompt"] for r in rows)


def test_corrective_rows_pagectx_uses_build_context_prompt():
    # With pagectx_frac=1.0 every row with context becomes page-context shape and
    # MUST match build_context_prompt byte-for-byte (the trained template).
    recs = _corr_records()
    rows = build_v11.corrective_rows(recs, pagectx_frac=1.0)
    by_src = {r["src"]: r for r in rows}
    r0 = by_src["corrective_v11:cleanocr:050_0:pagectx"]
    expected = build_v11.build_context_prompt(
        build_v11.PAGE_INSTR, recs[0]["context_jp"], recs[0]["context_k"]
    )
    assert r0["prompt"] == expected
    assert "Page:" in r0["prompt"]
    # The target line embedded in the page must be the corrective jp.
    assert "Translate line 2: 締まってきた" in r0["prompt"]


def test_corrective_pagectx_assistant_is_the_corrective_en():
    recs = _corr_records()
    rows = build_v11.corrective_rows(recs, pagectx_frac=1.0)
    r0 = next(r for r in rows if r["src"].startswith("corrective_v11:cleanocr:050_0"))
    assert r0["en"] == "It's getting tighter"
    assert r0["register_tag"] == "vn_eroge"
    assert r0["gold_flag"] is True


def test_corrective_pagectx_falls_back_to_plain_without_context():
    # A corrective row lacking context_jp can't be page-context shaped; it must
    # still emit (as plain), never be dropped.
    recs = [{
        "jp": "果てた",
        "en": "He climaxed",
        "src": "corrective_v11:cleanocr:099_1",
        "register_tag": "vn_eroge",
        "gold_flag": True,
    }]
    rows = build_v11.corrective_rows(recs, pagectx_frac=1.0)
    assert len(rows) == 1
    assert rows[0]["prompt"].startswith(build_v11.PLAIN_INSTR)


def test_corrective_rows_fraction_is_deterministic():
    # Same seed => same plain/pagectx partition (reproducible builds).
    recs = _corr_records() * 10
    a = build_v11.corrective_rows(recs, pagectx_frac=0.5, seed=7)
    b = build_v11.corrective_rows(recs, pagectx_frac=0.5, seed=7)
    assert [r["src"] for r in a] == [r["src"] for r in b]


# ===========================================================================
# ITEM 2: reverse-sense corrective data
# ===========================================================================

def test_reverse_sense_lexemes_cover_the_audit_set():
    # The 8 documented reverse-sense errors must all be present.
    keys = {e.lexeme for e in reverse_sense.REVERSE_SENSE_ENTRIES}
    for jp in ("締まる", "吸い出せ", "果てた", "風俗", "騎乗位", "割る", "尻", "マンコ"):
        assert jp in keys, f"missing reverse-sense lexeme {jp}"


def test_reverse_sense_each_lexeme_has_2_to_3_varied_carriers():
    for e in reverse_sense.REVERSE_SENSE_ENTRIES:
        assert 2 <= len(e.carriers) <= 3, f"{e.lexeme}: need 2-3 carriers"
        # Carriers must be DISTINCT (learn the sense, not one memorized line).
        jps = [c.jp for c in e.carriers]
        assert len(set(jps)) == len(jps), f"{e.lexeme}: duplicate carriers"


def test_reverse_sense_records_the_wrong_sense_and_right_sense():
    e = next(x for x in reverse_sense.REVERSE_SENSE_ENTRIES if x.lexeme == "締まる")
    assert "tighten" in e.right_sense.lower()
    assert "clos" in e.our_wrong_sense.lower()


def test_reverse_sense_build_emits_both_shapes_per_carrier():
    rows = reverse_sense.build_rows()
    shapes = {r["shape"] for r in rows}
    assert shapes == {"plain", "pagectx"}
    # For each carrier we expect one plain + one pagectx row.
    plain = [r for r in rows if r["shape"] == "plain"]
    pagectx = [r for r in rows if r["shape"] == "pagectx"]
    assert len(plain) == len(pagectx)


def test_reverse_sense_pagectx_prompt_matches_context_builder():
    rows = reverse_sense.build_rows()
    r = next(r for r in rows if r["shape"] == "pagectx")
    expected = build_v11.build_context_prompt(
        build_v11.PAGE_INSTR, r["context_jp"], r["context_k"]
    )
    assert r["prompt"] == expected


def test_reverse_sense_human_en_is_todo_not_fabricated():
    # human_en for NEW carriers needs the gold eval set; it must be empty +
    # flagged, never fabricated.
    rows = reverse_sense.build_rows()
    assert all(r["human_en"] == "" for r in rows)
    assert all(r["needs_gold"] is True for r in rows)


def test_reverse_sense_has_contrastive_margin_field():
    rows = reverse_sense.build_rows()
    for r in rows:
        assert "contrastive_margin" in r
        # chrF++(human) - chrF++(our_wrong); None until gold human_en exists.
        assert r["contrastive_margin"] is None
        assert "our_wrong" in r and r["our_wrong"]


def test_reverse_sense_nsfw_fraction_is_flat_not_oversampled():
    # Documented v12 regression: oversampling NSFW backfired into euphemism.
    # Each NSFW lexeme contributes the SAME number of rows as any SFW lexeme
    # (one plain + one pagectx per carrier) — no per-NSFW multiplier.
    rows = reverse_sense.build_rows()
    per_lexeme = {}
    for r in rows:
        per_lexeme.setdefault(r["lexeme"], []).append(r)
    counts = {lx: len(rs) // 2 for lx, rs in per_lexeme.items()}  # carriers
    # rows-per-carrier is constant (2) across NSFW and SFW lexemes.
    for r in rows:
        assert r["nsfw"] in (True, False)
    rows_per_carrier = {
        lx: len(rs) / counts[lx] for lx, rs in per_lexeme.items()
    }
    assert set(rows_per_carrier.values()) == {2.0}


# ===========================================================================
# ITEM 3: voice / addressee probe
# ===========================================================================

def test_voice_probe_has_causative_passive_and_command_inversion():
    cats = {e.category for e in voice.VOICE_PROBE_ENTRIES}
    assert "causative_passive" in cats
    assert "command_addressee" in cats


def test_voice_probe_causative_passive_gold_is_made_to():
    e = next(
        x for x in voice.VOICE_PROBE_ENTRIES
        if x.category == "causative_passive" and "させられ" in x.jp
    )
    # gold target must encode "was MADE to", NOT "I did".
    assert "made to" in e.gold_en.lower()


def test_voice_probe_command_inversion_is_2nd_person():
    e = next(
        x for x in voice.VOICE_PROBE_ENTRIES if x.category == "command_addressee"
    )
    # The failure is 'keep them on' -> 'I kept it on'; gold stays 2nd-person.
    assert e.wrong_en and "I " in e.wrong_en  # the inversion we guard against
    assert "I " not in e.gold_en.split(".")[0] or e.gold_en.lower().startswith(
        ("keep", "put", "take", "don't", "do ")
    )


def test_voice_probe_rows_have_gold_targets():
    rows = voice.build_rows()
    assert rows, "probe must emit rows"
    for r in rows:
        assert r["jp"] and r["gold_en"]
        assert "category" in r
        assert r["src"].startswith("voice_probe:")


def test_voice_probe_emits_both_shapes():
    rows = voice.build_rows()
    shapes = {r["shape"] for r in rows}
    assert "plain" in shapes
    assert "pagectx" in shapes


def test_voice_probe_pagectx_matches_context_builder():
    rows = voice.build_rows()
    r = next(r for r in rows if r["shape"] == "pagectx")
    expected = build_v11.build_context_prompt(
        build_v11.PAGE_INSTR, r["context_jp"], r["context_k"]
    )
    assert r["prompt"] == expected

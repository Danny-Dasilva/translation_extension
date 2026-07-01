"""Unit tests for the v13ship refusal-stripped TEXT-ONLY SFT assembler.

The assembler builds a text-only SFT set to fine-tune the abliterated Qwen3-VL
translation ship. Two properties are load-bearing and pinned here:

  1. PROMPT FORMAT PARITY -- every emitted `prompt` must byte-match the v11
     train==serve contract (a format drift is the documented ~95% chrF++
     collapse). We assert the training builder the assembler reuses is
     byte-identical to the SERVE-side prompt builder.

  2. REFUSAL STRIP + NSFW CAP -- the EN target side must drop LLM refusal
     strings (without nuking benign manga dialogue like "I'm sorry I'm late"),
     and the final NSFW share must stay <= 18% (a prior 36% oversample
     regressed the model into euphemism).

The furube gold is the EVAL set and must NEVER appear in training.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "data"
    / "v13ship"
    / "build_textsft_refusalstripped.py"
)


def _load_module():
    # Mirror conftest: sibling CLI modules import each other by bare name.
    data_dir = Path(__file__).resolve().parents[2] / "scripts" / "data"
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))
    spec = importlib.util.spec_from_file_location(
        "build_textsft_refusalstripped", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# --------------------------------------------------------------------------- #
# Refusal classifier
# --------------------------------------------------------------------------- #
KNOWN_REFUSALS = [
    "I'm sorry, but I can't assist with that request.",
    "I'm sorry, but I cannot help with this content.",
    "I cannot translate this content.",
    "As an AI language model, I won't help with that.",
    "As an AI, I am unable to provide a translation of this material.",
    "I apologize, but I can't continue with this request.",
    "I will not generate that kind of content.",
    "I'm unable to assist with this translation.",
    "I must decline to translate this.",
    "This request violates my guidelines.",
    "I'm not able to help with that.",
    "Sorry, but this content is inappropriate and I cannot translate it.",
    "I'm unable to provide a translation of this material.",
]

# Benign manga dialogue that COINCIDENTALLY contains refusal-ish tokens.
# Dropping any of these would silently delete good training rows. The second block
# are REAL false positives found by scanning the 302k v11 backbone with a looser
# pattern -- pinned here so the tightened classifier never regresses onto them.
BENIGN_KEEP = [
    "I'm sorry I'm late!",
    "I can't wait to see you again.",
    "I can't believe you did that.",
    "I won't lose to you!",
    "I won't give up, no matter what.",
    "I'm sorry... I really loved you.",
    "Put it in deeper ♥",  # explicit NSFW target must survive
    "You can cum inside me ♥",
    "Stop it, that tickles!",
    "That's so inappropriate of you, hehe.",
    "I can help you carry those.",
    # real backbone false positives (must keep):
    "Geez... I can't do this all by myself.",
    "I can't help but chuckle.",
    "I can't help but feel that you astronomers are overdoing it a bit... aren't you?",
    "I'm afraid I can't do that.",
    "I won't do that! I won't! Rather than that, let's check out the loot?",
    "I apologize, but please allow her to cancel today's plans.",
    "Goes against my policy.",
    "Well, I can't help it.",
    "Unfortunately, I can't do that.",
    "Wait... hold on... I-I can't do this!",
]


def test_refusal_pattern_is_named_auditable_constant(mod):
    assert isinstance(mod.REFUSAL_PATTERN, str) and mod.REFUSAL_PATTERN
    # compiled, case-insensitive
    assert mod.REFUSAL_RE.match("AS AN AI") is not None or mod.REFUSAL_RE.search(
        "AS AN AI"
    )
    # the task's core refusal cues must be present in the auditable source
    src = mod.REFUSAL_PATTERN.lower()
    for cue in ("as an ai", "sorry", "unable", "cannot", "won", "translate"):
        assert cue in src, f"refusal pattern missing cue: {cue!r}"


@pytest.mark.parametrize("en", KNOWN_REFUSALS)
def test_known_refusals_are_dropped(mod, en):
    assert mod.is_refusal(en) is True, f"should be a refusal: {en!r}"


@pytest.mark.parametrize("en", BENIGN_KEEP)
def test_benign_manga_dialogue_is_kept(mod, en):
    assert mod.is_refusal(en) is False, f"benign line wrongly flagged: {en!r}"


def test_refusal_strip_partitions_and_counts(mod):
    rows = [
        {"prompt": "p", "en": "Hello there.", "src": "x:plain",
         "register_tag": "manga", "gold_flag": True},
        {"prompt": "p", "en": "I'm sorry, but I can't assist with that.",
         "src": "y:pagectx", "register_tag": "manga_nsfw", "gold_flag": False},
        {"prompt": "p", "en": "I can't wait!", "src": "z:plain",
         "register_tag": "manga", "gold_flag": True},
    ]
    kept, dropped = mod.refusal_strip(rows)
    assert len(kept) == 2
    assert len(dropped) == 1
    assert dropped[0]["en"].startswith("I'm sorry")


# --------------------------------------------------------------------------- #
# NSFW cap
# --------------------------------------------------------------------------- #
def _mk(n, register, gold=False):
    return [
        {"prompt": f"pr{i}", "en": f"en{i}", "src": f"s{i}:pagectx",
         "register_tag": register, "gold_flag": gold}
        for i in range(n)
    ]


def test_nsfw_cap_enforced(mod):
    rows = _mk(60, "manga_dialog") + _mk(40, "manga_nsfw")
    kept, dropped_n, info = mod.enforce_nsfw_cap(rows, cap=0.18, seed=42)
    nsfw = [r for r in kept if mod.is_nsfw_row(r)]
    non = [r for r in kept if not mod.is_nsfw_row(r)]
    assert len(non) == 60, "non-NSFW rows must never be dropped by the cap"
    frac = len(nsfw) / len(kept)
    assert frac <= 0.18 + 1e-9, f"nsfw frac {frac} exceeds cap"
    assert dropped_n == 40 - len(nsfw)


def test_nsfw_cap_noop_when_under(mod):
    rows = _mk(90, "manga_dialog") + _mk(10, "vn_eroge")
    kept, dropped_n, _ = mod.enforce_nsfw_cap(rows, cap=0.18, seed=42)
    assert dropped_n == 0
    assert len(kept) == 100


def test_nsfw_cap_deterministic(mod):
    rows = _mk(60, "manga_dialog") + _mk(40, "manga_nsfw")
    a, _, _ = mod.enforce_nsfw_cap(rows, cap=0.18, seed=42)
    b, _, _ = mod.enforce_nsfw_cap(rows, cap=0.18, seed=42)
    assert [r["src"] for r in a] == [r["src"] for r in b]


def test_nsfw_registers_and_classifier(mod):
    assert "manga_nsfw" in mod.NSFW_REGISTERS
    assert "vn_eroge" in mod.NSFW_REGISTERS
    assert mod.is_nsfw_row({"register_tag": "manga_nsfw"}) is True
    assert mod.is_nsfw_row({"register_tag": "manga_dialog"}) is False


# --------------------------------------------------------------------------- #
# Prompt-format parity (train builder == serve builder)  -- the ~95% collapse guard
# --------------------------------------------------------------------------- #
def test_builder_parity_train_equals_serve(mod):
    info = mod.verify_builder_parity(n_cases=300, seed=7)
    assert info["mismatches"] == 0, info
    assert info["compared"] > 0
    assert info["instr_page_match"] is True
    assert info["instr_plain_match"] is True


# --------------------------------------------------------------------------- #
# Ikenie gold -> page-context rows
# --------------------------------------------------------------------------- #
def test_build_ikenie_rows_format_and_schema(mod, tmp_path):
    gold = tmp_path / "gold_q3.jsonl"
    page = [
        {"jp": "おはようございます",
         "en": "GOOD MORNING, EVERYONE.",
         "src": "ikenie4:p01:idx0", "register_tag": "manga_nsfw",
         "bbox": {"minX": 800, "minY": 100, "maxX": 900, "maxY": 400},
         "ocr_conf": 0.95},
        {"jp": "もう朝ごはんの時間か",
         "en": "IS IT ALREADY BREAKFAST TIME?",
         "src": "ikenie4:p01:idx1", "register_tag": "manga_nsfw",
         "bbox": {"minX": 500, "minY": 120, "maxX": 600, "maxY": 420},
         "ocr_conf": 0.93},
    ]
    with gold.open("w") as f:
        for r in page:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rows = mod.build_ikenie_rows([gold], repeat=1, also_plain=True)
    assert rows, "expected ikenie rows"
    # schema
    for r in rows:
        assert set(r.keys()) == set(mod.COLS)
    # gold provenance
    assert all(r["gold_flag"] is True for r in rows)
    assert all("ikenie" in r["src"] for r in rows)
    # page-context prompts byte-match the v11 SERVE builder
    pc = [r for r in rows if r["src"].endswith(":pagectx")]
    assert pc, "expected page-context rows"
    for r in pc:
        assert r["prompt"].startswith(mod.V11_PAGE_INSTR)
    pl = [r for r in rows if r["src"].endswith(":plain")]
    for r in pl:
        assert r["prompt"].startswith(mod.V11_PLAIN_INSTR)
    # ALL-CAPS scanlation targets are recased to sentence case (no shouting)
    assert any(not r["en"].isupper() for r in rows)


def test_ikenie_repeat_multiplies(mod, tmp_path):
    gold = tmp_path / "gold_q3.jsonl"
    with gold.open("w") as f:
        f.write(json.dumps({
            "jp": "こんにちは", "en": "HELLO.",
            "src": "ikenie5:p02:idx0", "register_tag": "manga_nsfw",
            "bbox": {"minX": 10, "minY": 10, "maxX": 50, "maxY": 50},
            "ocr_conf": 0.9}) + "\n")
    r1 = mod.build_ikenie_rows([gold], repeat=1, also_plain=False)
    r3 = mod.build_ikenie_rows([gold], repeat=3, also_plain=False)
    assert len(r3) == 3 * len(r1)


# --------------------------------------------------------------------------- #
# Furube MUST be held out of training
# --------------------------------------------------------------------------- #
def test_assert_no_furube_raises(mod):
    good = [{"src": "ikenie4:p01:idx0:pagectx"}]
    mod.assert_no_furube(good)  # no raise
    bad = good + [{"src": "furube_p1:p02:idx0:pagectx"}]
    with pytest.raises(Exception):
        mod.assert_no_furube(bad)


# --------------------------------------------------------------------------- #
# Native trainer messages-jsonl format (build_conversations schema)
# --------------------------------------------------------------------------- #
def _row(src="s:pagectx", reg="manga_dialog", gold=True):
    return {"prompt": "Translate ...\n\nPage:\n1. あ\n\nTranslate line 1: あ",
            "en": "Ah.", "src": src, "register_tag": reg, "gold_flag": gold}


def test_row_to_message_roundtrip(mod):
    r = _row()
    m = mod.row_to_message(r)
    # build_conversations only strictly needs `messages`; we carry parity fields.
    for k in ("messages", "source", "register_tag", "has_image", "image_path",
              "gold_flag", "meta"):
        assert k in m, f"missing key {k}"
    assert m["has_image"] is False
    assert m["image_path"] == ""
    assert m["meta"]["src"] == r["src"]
    # user text block == the byte-exact v11 prompt; assistant block == en target
    user = m["messages"][0]
    asst = m["messages"][1]
    assert user["role"] == "user"
    assert user["content"][0]["type"] == "text"
    assert user["content"][0]["text"] == r["prompt"]
    assert asst["role"] == "assistant"
    assert asst["content"][0]["text"] == r["en"]


def test_row_to_message_source_labels(mod):
    assert mod.row_to_message(_row(src="ikenie_gold:ikenie4:p01:idx0:pagectx"))["source"] == "ikenie_gold"
    assert mod.row_to_message(_row(src="manga109:x:p1:h:pagectx"))["source"] == "v11_backbone"


def test_messages_format_check_validates(mod):
    rows = [_row(), _row(src="ikenie_gold:ikenie4:p01:idx1:plain")]
    msgs = [mod.row_to_message(r) for r in rows]
    info = mod.messages_format_check(msgs, rows)
    assert info["valid"] is True
    assert info["roundtrip_ok"] is True
    assert info["has_image_all_false"] is True

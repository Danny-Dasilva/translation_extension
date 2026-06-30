"""Anti-hallucination faithfulness fixes (Phase-0 quality).

Covers the three serve-time guards that stop the small model from confidently
inventing dialogue on short, context-light bubbles:

  fix #1  source_aware_max_tokens   — a 4-char JP can't be handed a 64-token rope
  fix #2  is_over_expanded / gate   — flag/blank an EN output far longer than the
                                       JP source can faithfully justify
  fix #3  looks_like_nonlexical_grunt — stuttered katakana grunt (ヴヴ) -> "..."

These are PURE helpers (no model / no live vLLM), so the whole file runs offline.
"""

import pytest

from app.services.vllm_openai_translation_service import (
    MAX_TOKENS_FLOOR,
    looks_like_nonlexical_grunt,
    source_aware_max_tokens,
)
from app.services.translation_postedit import (
    OVER_EXPANSION_ABS_MIN_WORDS,
    en_word_count,
    gate_over_expansion,
    is_over_expanded,
    jp_content_len,
    postedit_one,
)

CEIL = 64  # the flat settings.translate_max_tokens used as the hard ceiling


# --------------------------------------------------------------------------- #
# fix #1 — source-length-aware max_tokens cap
# --------------------------------------------------------------------------- #
class TestSourceAwareMaxTokens:
    def test_four_char_source_gets_small_cap(self):
        # A 4-char JP must NOT be able to reach the flat 64-token budget.
        cap = source_aware_max_tokens("お母さんS", CEIL)
        assert cap < CEIL
        assert cap <= 24  # comfortably below the ~40-word runaway region

    def test_long_source_clamps_to_ceiling(self):
        # A long source needs the full budget -> clamps to the 64 ceiling.
        assert source_aware_max_tokens("あ" * 40, CEIL) == CEIL

    def test_threshold_is_around_fifteen_chars(self):
        # ~15 content chars is where a normal 2-3x expansion saturates the cap.
        assert source_aware_max_tokens("あ" * 15, CEIL) == CEIL

    def test_floor_for_tiny_and_empty_sources(self):
        assert source_aware_max_tokens("あ", CEIL) >= MAX_TOKENS_FLOOR
        assert source_aware_max_tokens("", CEIL) == MAX_TOKENS_FLOOR

    def test_monotonic_non_decreasing_in_length(self):
        caps = [source_aware_max_tokens("あ" * n, CEIL) for n in range(0, 30)]
        assert caps == sorted(caps)

    def test_never_exceeds_ceiling(self):
        for n in range(0, 50):
            assert source_aware_max_tokens("あ" * n, CEIL) <= CEIL


# --------------------------------------------------------------------------- #
# fix #2 — over-expansion faithfulness gate
# --------------------------------------------------------------------------- #
class TestOverExpansionGate:
    def test_flags_four_char_to_fifteen_words(self):
        # 4-char JP -> 15-word EN is a confident hallucination -> flagged.
        en = "This is a completely invented sentence that the model just made up here"
        assert en_word_count(en) == 13  # sanity on the helper
        en15 = en + " right now"
        assert en_word_count(en15) == 15
        assert is_over_expanded(en15, "おかあS") is True

    def test_passes_normal_two_point_five_x_expansion(self):
        # 8 content chars -> 20 EN words (exactly 2.5 words/char) must PASS.
        jp = "今日はいい天気だ"  # 8 content chars
        assert jp_content_len(jp) == 8
        en = " ".join(["word"] * 20)
        assert is_over_expanded(en, jp) is False

    def test_short_meaningful_reply_passes(self):
        # ええ -> "What are you saying?!" is a legit short->meaningful expansion.
        assert is_over_expanded("What are you saying?!", "ええ") is False

    def test_abs_min_words_floor_never_flags_short_output(self):
        # Any output under the absolute-min word floor is never flagged, even on
        # a 1-char source.
        short = " ".join(["w"] * (OVER_EXPANSION_ABS_MIN_WORDS - 1))
        assert is_over_expanded(short, "あ") is False

    def test_missing_source_never_flags(self):
        long_en = " ".join(["w"] * 40)
        assert is_over_expanded(long_en, None) is False
        assert is_over_expanded(long_en, "") is False
        # source that is pure punctuation has zero content length -> never flags
        assert is_over_expanded(long_en, "。、！？") is False

    def test_gate_blanks_over_expanded_to_ellipsis(self):
        en15 = " ".join(["w"] * 15)
        assert gate_over_expansion(en15, "おかあS") == "..."

    def test_gate_passes_faithful_output_unchanged(self):
        assert gate_over_expansion("What are you saying?!", "ええ") == "What are you saying?!"

    def test_gate_none_passthrough(self):
        assert gate_over_expansion(None, "ええ") is None

    def test_jp_content_len_strips_punctuation(self):
        assert jp_content_len("お母さん、Sサイズ！") == jp_content_len("お母さんSサイズ")


class TestOverExpansionThroughPostedit:
    def test_postedit_blanks_over_expanded(self):
        # The faithfulness floor is wired into postedit_one (router + batch).
        en15 = " ".join(["invented"] * 15)
        out = postedit_one(en15, "おかあS")
        assert out == "..."

    def test_postedit_preserves_faithful_short_line(self):
        # Existing behaviour: a faithful short->meaningful line is untouched.
        out = postedit_one("What are you saying?!", "ええ")
        assert out == "What are you saying?!"


# --------------------------------------------------------------------------- #
# fix #3 — narrow stuttered-katakana grunt bypass
# --------------------------------------------------------------------------- #
class TestNonLexicalGrunt:
    @pytest.mark.parametrize("jp", ["ヴヴ", "ヴヴヴ", "ググ", "ブブッ"])
    def test_stuttered_katakana_grunt_detected(self, jp):
        assert looks_like_nonlexical_grunt(jp) is True

    @pytest.mark.parametrize(
        "jp",
        [
            "ええ",      # meaningful hiragana interjection
            "うくっ",    # ambiguous hiragana grunt -> left to the model / glossary
            "ンン",      # ン moan, explicitly excluded
            "アア",      # vowel-katakana moan, excluded
            "あっ",      # single hiragana gasp
            "こんにちは",  # ordinary dialogue
            "ヴァンパイア",  # real katakana word (distinct glyphs)
            "",          # empty
        ],
    )
    def test_non_grunts_pass_through(self, jp):
        assert looks_like_nonlexical_grunt(jp) is False

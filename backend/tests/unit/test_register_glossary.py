"""Tests for the source-conditioned explicit-register post-edit glossary.

P1-2: this manga euphemizes domain-specific explicit JP terms. The fix is a
PRECISION-OVER-RECALL post-edit: rewrite the English ONLY when BOTH the JP
source contains a known explicit term AND the English output used a known
euphemism/wrong-word for it. We never re-explicitate a clean line.

The clearest, safe case is 潮 (sexual context) -> "squirt" where the model
emitted "seawater"/"tide". These tests pin that behaviour and, crucially,
the NEGATIVE cases that must NOT trigger.
"""

from app.services.register_glossary import restore_register


class TestPositiveSubstitution:
    def test_shio_seawater_becomes_squirt(self):
        jp = "リビングが潮まみれじゃん"
        en = "The living room is covered in seawater."
        out = restore_register(en, jp)
        assert "seawater" not in out.lower()
        assert "squirt" in out.lower()
        assert out == "The living room is covered in squirt."

    def test_shio_two_word_sea_water(self):
        jp = "潮吹いた"
        en = "It's full of sea water."
        out = restore_register(en, jp)
        assert "squirt" in out.lower()
        assert "sea water" not in out.lower()

    def test_shio_tide(self):
        jp = "潮がすごい"
        en = "The tide is incredible."
        out = restore_register(en, jp)
        assert "squirt" in out.lower()

    def test_preserves_capitalization_titlecase(self):
        jp = "潮まみれ"
        en = "Seawater everywhere."
        out = restore_register(en, jp)
        assert out.startswith("Squirt")

    def test_preserves_capitalization_allcaps(self):
        jp = "潮まみれ"
        en = "THE LIVING ROOM IS COVERED IN SEAWATER"
        out = restore_register(en, jp)
        assert "SQUIRT" in out
        assert "SEAWATER" not in out


class TestNegativeNoChange:
    def test_shiohigari_is_not_sexual(self):
        # 潮干狩り = clam digging / "going to the beach". The literal "salt
        # water"/"seawater" reading here is CORRECT and must be preserved.
        jp = "明日は潮干狩りに行こう"
        en = "Let's go collect seawater at the beach tomorrow."
        out = restore_register(en, jp)
        assert out == en  # unchanged

    def test_seawater_without_shio_in_source(self):
        # English mentions seawater but the JP source has no 潮 term: do not
        # touch it (could be a literal beach/ocean scene).
        jp = "海がきれいだね"
        en = "The seawater is so clear."
        out = restore_register(en, jp)
        assert out == en

    def test_clean_line_unchanged(self):
        jp = "あーあーすげぇな"
        en = "Man, this is amazing."
        out = restore_register(en, jp)
        assert out == en

    def test_shio_in_source_but_no_wrong_word_in_en(self):
        # Source has 潮 and the model ALREADY translated it explicitly:
        # don't double-substitute or mangle a correct line.
        jp = "潮吹いちゃう"
        en = "I'm going to squirt!"
        out = restore_register(en, jp)
        assert out == en


class TestRobustness:
    def test_empty_inputs(self):
        assert restore_register("", "") == ""
        assert restore_register("", "潮") == ""

    def test_none_english_passthrough(self):
        # The pipeline can emit None for dropped/empty bubbles.
        assert restore_register(None, "潮") is None

    def test_does_not_substitute_substring_inside_word(self):
        # "tide" inside "tides"/"untidy" must not be mangled when it is a real
        # English word unrelated to the 潮 reading. We only swap whole-word
        # wrong_en tokens.
        jp = "潮"
        en = "The room is untidy."
        out = restore_register(en, jp)
        assert out == en  # 'tide' is a substring of 'untidy' -> no whole-word match

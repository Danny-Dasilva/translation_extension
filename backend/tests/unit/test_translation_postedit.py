"""Unit tests for the shared post-edit chain (translation_postedit).

Both the API router (app/routers/translate.py) and the batch pipeline
(scripts/batch_translate_chapter.py) call into this module, so a fix here is
guaranteed to apply to both rendering paths. These tests pin the public
signatures and confirm the optional OCR-confidence threading.
"""

from app.services.translation_postedit import (
    apply_postedit_glossaries,
    postedit_one,
)


class TestPosteditOne:
    def test_yurie_lock_through_chain(self):
        # The name-lock added in name_glossary must reach output via the chain.
        out = postedit_one("Julie, come here.", "ユリエ、こっち来て")
        assert "Julie" not in out
        assert "Yurie" in out

    def test_counted_number_through_chain(self):
        out = postedit_one("Jus-san!!", "じゅうさん!!")
        assert "san" not in out.lower()
        assert ("33" in out) or ("thirty-three" in out.lower())

    def test_none_passthrough(self):
        assert postedit_one(None, "ユリエ") is None

    def test_low_conf_suppresses_invention(self):
        out = postedit_one("Sue", "おばさん", ocr_conf=0.30)
        assert out != "Sue"

    def test_ocr_conf_optional(self):
        # Backward compatible: callers that omit ocr_conf still work.
        out = postedit_one("Hello there.", "やあ")
        assert out == "Hello there."


class TestApplyPosteditGlossaries:
    def test_list_alignment(self):
        ens = ["Julie!", "Jus-san!!"]
        jps = ["ユリエ!", "じゅうさん!!"]
        out = apply_postedit_glossaries(ens, jps)
        assert "Yurie" in out[0]
        assert "san" not in out[1].lower()

    def test_ocr_confs_threaded_when_provided(self):
        ens = ["Sue"]
        jps = ["おばさん"]
        out = apply_postedit_glossaries(ens, jps, ocr_confs=[0.30])
        assert out[0] != "Sue"

    def test_ocr_confs_optional(self):
        ens = ["Hello."]
        jps = ["やあ"]
        out = apply_postedit_glossaries(ens, jps)
        assert out == ["Hello."]

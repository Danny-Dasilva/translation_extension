"""Unit tests for the character-name canonicalizer (P0-3).

Pure post-processing on the model's EN output (optionally source-conditioned
on the OCR `jp`). MUST NOT touch the model prompt — the v11 model is
train/serve format-sensitive. See backend/app/services/name_glossary.py.

Corruptions observed in 588828_mesu2_insp full-pipeline bench:
  加奈子 (Kanako) -> "Kana" / "Kanan" / "Kana-ji" / "Kanakao"
  康介   (Kousuke) -> "Yousuke" / "Kosuke" / "Kansuke"
  愛菜   (Aina, a little girl) -> "the milk" (page 044)
  お姉ちゃん (onee-chan = big sis) -> "Chen" (page 069)
"""

from app.services.name_glossary import canonicalize_names


# --------------------------------------------------------------------------- #
# Kanako: word-boundary EN normalisation (no jp needed)
# --------------------------------------------------------------------------- #
class TestKanako:
    def test_standalone_kana(self):
        assert (
            canonicalize_names("Where's Kana today? At the part-time job?")
            == "Where's Kanako today? At the part-time job?"
        )

    def test_kanan(self):
        assert (
            canonicalize_names("Hey Kanan, is that a subscription?")
            == "Hey Kanako, is that a subscription?"
        )

    def test_kana_ji(self):
        assert (
            canonicalize_names("I'm cumming inside you, Kana-ji...")
            == "I'm cumming inside you, Kanako..."
        )

    def test_kanakao(self):
        assert (
            canonicalize_names("Kanakao got us scolded for this too, remember?")
            == "Kanako got us scolded for this too, remember?"
        )

    def test_already_correct_is_unchanged(self):
        s = "Look, here are the gloves Kanako gave me."
        assert canonicalize_names(s) == s

    def test_idempotent(self):
        once = canonicalize_names("Hey Kanan!")
        twice = canonicalize_names(once)
        assert once == twice == "Hey Kanako!"


# --------------------------------------------------------------------------- #
# Kousuke: collapse three spellings to one canonical form
# --------------------------------------------------------------------------- #
class TestKousuke:
    def test_yousuke(self):
        assert (
            canonicalize_names("Yousuke-san? Um, hello!")
            == "Kousuke-san? Um, hello!"
        )

    def test_kansuke(self):
        assert (
            canonicalize_names("I trust you completely, Kansuke-san.")
            == "I trust you completely, Kousuke-san."
        )

    def test_kosuke(self):
        assert (
            canonicalize_names("you mentioned something about Kosuke-san.")
            == "you mentioned something about Kousuke-san."
        )

    def test_canonical_unchanged(self):
        s = "while Kousuke is sleeping right next to me..."
        assert canonicalize_names(s) == s


# --------------------------------------------------------------------------- #
# Source-conditioned: 愛菜 -> "milk" must become "Aina"
# --------------------------------------------------------------------------- #
class TestAina:
    def test_milk_corrected_when_jp_has_aina(self):
        jp = "うんっ愛菜預けてからパート..買い出しも行くけど..ついでになんかある?"
        en = "Yeah. I'll drop off the milk and then go to my part-time job..."
        out = canonicalize_names(en, jp)
        assert "milk" not in out
        assert "Aina" in out

    def test_milk_left_alone_when_jp_lacks_aina(self):
        # A bubble that legitimately talks about milk and has no 愛菜.
        jp = "牛乳を買ってきて"
        en = "Go buy some milk."
        assert canonicalize_names(en, jp) == en

    def test_aina_already_correct(self):
        jp = "うん..おやすみー愛菜ー"
        en = "Yeah... goodnight, Aina."
        assert canonicalize_names(en, jp) == en


# --------------------------------------------------------------------------- #
# Source-conditioned: お姉ちゃん -> "Chen" must become a sister term
# --------------------------------------------------------------------------- #
class TestOneechan:
    def test_chen_corrected_when_jp_has_oneechan(self):
        jp = "お...ねえ..ちゃん...?"
        en = "H... hey... Chen"
        out = canonicalize_names(en, jp)
        assert "Chen" not in out
        assert "Sis" in out

    def test_chen_left_alone_without_jp(self):
        # No source context -> we must not guess that "Chen" is a corruption.
        en = "H... hey... Chen"
        assert canonicalize_names(en) == en

    def test_oneechan_translated_as_sister_unchanged(self):
        jp = "お姉ちゃんの味方ですから.."
        en = "After all, I'm your sister's supporter..."
        assert canonicalize_names(en, jp) == en


# --------------------------------------------------------------------------- #
# Conservatism / negative cases — never rewrite substrings of real words
# --------------------------------------------------------------------------- #
class TestConservative:
    def test_banana_untouched(self):
        s = "I ate a banana and a kanaban."
        assert canonicalize_names(s) == s

    def test_kana_as_substring_untouched(self):
        # "Kanazawa" (a place) contains "Kana" but must stay intact.
        s = "We took the train to Kanazawa station."
        assert canonicalize_names(s) == s

    def test_empty_string(self):
        assert canonicalize_names("") == ""

    def test_no_names_present(self):
        s = "Your tie is crooked!"
        assert canonicalize_names(s, "ネクタイ曲がってるよー") == s

    def test_does_not_invent_names(self):
        # Generic sentence, no known corruption -> identity.
        s = "Hmm, I'll let you know if I find anything."
        assert canonicalize_names(s) == s

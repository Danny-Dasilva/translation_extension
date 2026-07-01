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


# --------------------------------------------------------------------------- #
# Hard name-lock: ユリエ -> "Yurie" (model emits "Julie" / "Lucia")
# --------------------------------------------------------------------------- #
class TestNameLockYurie:
    def test_julie_locked_to_yurie(self):
        jp = "ユリエ、こっち来て"
        en = "Julie, come over here."
        out = canonicalize_names(en, jp)
        assert "Julie" not in out
        assert "Yurie" in out

    def test_lucia_locked_to_yurie(self):
        jp = "ユリエ……"
        en = "Lucia..."
        out = canonicalize_names(en, jp)
        assert "Lucia" not in out
        assert "Yurie" in out

    def test_yurie_already_correct_is_unchanged(self):
        jp = "ユリエだよ"
        en = "It's Yurie."
        assert canonicalize_names(en, jp) == en

    def test_lock_does_not_fire_without_jp_trigger(self):
        # No ユリエ in source -> a person legitimately named "Julie" stays Julie.
        en = "Julie went home early."
        assert canonicalize_names(en, "彼女は早く帰った") == en
        assert canonicalize_names(en) == en

    def test_lock_idempotent(self):
        jp = "ユリエ、こっち来て"
        once = canonicalize_names("Julie, come over here.", jp)
        twice = canonicalize_names(once, jp)
        assert once == twice

    def test_bare_yuri_locked_to_yurie(self):
        # ユリエ leaks as the truncated "Yuri" page to page.
        jp = "ユリエ、ありがとう"
        out = canonicalize_names("Thanks, Yuri.", jp)
        assert out == "Thanks, Yurie."

    def test_yuri_left_alone_without_jp_trigger(self):
        # A character actually named "Yuri" with no ユリエ in source stays Yuri.
        assert canonicalize_names("Yuri smiled.", "彼女は微笑んだ") == "Yuri smiled."

    def test_yurie_not_clobbered_by_yuri_prefix(self):
        # "Yuri" is a strict prefix of "Yurie" but whole-word match leaves the
        # canonical spelling intact (no "Yurie" -> "Yuriee").
        jp = "ユリエだよ"
        assert canonicalize_names("It's Yurie.", jp) == "It's Yurie."


# --------------------------------------------------------------------------- #
# Hard name-lock: あゆむ -> "Ayumu" (model emits Ayu/Aymu/Ayumumu/Ayuuuummm)
# --------------------------------------------------------------------------- #
class TestNameLockAyumu:
    def test_short_ayu_locked(self):
        jp = "あゆむ、待って"
        out = canonicalize_names("Ayu, wait!", jp)
        assert out == "Ayumu, wait!"

    def test_garbled_variants_locked(self):
        jp = "あゆむ……"
        for bad in ("Aymu", "Ayumumu", "Ayuuuummm"):
            out = canonicalize_names(f"{bad}...", jp)
            assert out == "Ayumu..."

    def test_ayumu_already_correct_unchanged(self):
        jp = "あゆむがいる"
        assert canonicalize_names("Ayumu is here.", jp) == "Ayumu is here."

    def test_lock_does_not_fire_without_jp_trigger(self):
        # No あゆむ in source -> someone legitimately called "Ayu" stays "Ayu".
        assert canonicalize_names("Ayu went home.", "彼は帰った") == "Ayu went home."
        assert canonicalize_names("Ayu went home.") == "Ayu went home."

    def test_ayu_substring_not_clobbered(self):
        # "Ayu" is a strict prefix of the canonical "Ayumu" — whole-word match
        # must not corrupt the canonical spelling.
        jp = "あゆむ"
        assert canonicalize_names("Ayumu", jp) == "Ayumu"

    def test_lock_idempotent(self):
        jp = "あゆむ、待って"
        once = canonicalize_names("Ayu, wait!", jp)
        twice = canonicalize_names(once, jp)
        assert once == twice == "Ayumu, wait!"


# --------------------------------------------------------------------------- #
# Counted-number kana: じゅうさん (=33) must NOT become "Jus-san"/honorific.
# Fire only when the bubble is essentially just a number reading.
# --------------------------------------------------------------------------- #
class TestCountedNumberKana:
    def test_juusan_is_thirty_three(self):
        # じゅうさん = 10 + 3 = 33, with emphasis punctuation.
        out = canonicalize_names("Jus-san!!", "じゅうさん!!")
        assert "san" not in out.lower()
        assert ("33" in out) or ("thirty-three" in out.lower())

    def test_hyaku_is_one_hundred(self):
        out = canonicalize_names("hyaku", "ひゃく")
        assert ("100" in out) or ("hundred" in out.lower())

    def test_nijuu_is_twenty(self):
        out = canonicalize_names("Niju", "にじゅう")
        assert ("20" in out) or ("twenty" in out.lower())

    def test_real_honorific_name_untouched(self):
        # 田中さん is a real name + honorific, NOT a counted number.
        jp = "田中さん、おはよう"
        en = "Tanaka-san, good morning."
        assert canonicalize_names(en, jp) == en

    def test_number_rule_does_not_fire_on_sentence(self):
        # さん inside a real sentence with a name is not a bare number bubble.
        jp = "さんは三人います"
        en = "There are three of them."
        assert canonicalize_names(en, jp) == en


# --------------------------------------------------------------------------- #
# Low-confidence OCR must not let a generic word become an invented name.
# おばさん (auntie) at low conf must NOT be promoted to a proper name "Sue".
# --------------------------------------------------------------------------- #
class TestLowConfidenceNoInvention:
    def test_low_conf_obasan_not_invented(self):
        jp = "おばさん"
        en = "Sue"  # model invented a proper name from おばさん
        out = canonicalize_names(en, jp, ocr_conf=0.30)
        # At low confidence we refuse to keep an invented single-token name;
        # it is neutralised (not promoted to a proper noun).
        assert out != "Sue"

    def test_high_conf_leaves_text_alone(self):
        # With high confidence and no known corruption we do not touch it.
        jp = "おばさん"
        en = "Auntie"
        assert canonicalize_names(en, jp, ocr_conf=0.95) == en

    def test_ocr_conf_defaults_to_no_suppression(self):
        # Omitting ocr_conf keeps backward-compatible behaviour (identity here).
        en = "Some ordinary sentence."
        assert canonicalize_names(en) == en

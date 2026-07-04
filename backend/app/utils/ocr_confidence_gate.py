"""OCR-confidence garble gate (translation pre-filter).

Low-confidence, garbled OCR on stylized SFX/illegible scrawl must NOT reach the
translation LLM — it hallucinates non-English captions ("Calcul math not done
by a lifter", Bengali junk) that then render onto the page. This gate drops a
bubble ONLY when the OCR *recognition* confidence is low AND the decoded text
looks garbled, so real dialogue (measured OCR conf ~0.9+) always passes.

Note: this uses PARSeq OCR recognition confidence (mean softmax max-prob over
decoded tokens), NOT the detector/block confidence.
"""
from __future__ import annotations

import json
import math
import unicodedata
from functools import lru_cache
from pathlib import Path

from app.utils.japanese_text_filter import analyze_characters, is_japanese_text

# Tuned on the Part13 inspection data (.bench/Part13_translated_en_v4_inspection):
#   real dialogue bubbles: OCR-equivalent crisp, conf ~0.85-0.97
#   garbled SFX / scrawl that produced hallucinations: conf ~0.45-0.61
# A gate at 0.65 cleanly separates them while staying conservative — we ALSO
# require the text to look garbled before dropping, so a merely-uncertain but
# clean Japanese line is kept.
DEFAULT_CONF_THRESHOLD = 0.65

# Below this Japanese-character ratio (post-normalization) a low-conf line is
# treated as garbled even if it scraped past the main japanese filter.
_MIN_JP_RATIO_FOR_LOWCONF = 0.5

# Characters that signal recognizer breakdown rather than real text.
_GARBLE_CHARS = set("[]{}/\\|<>^~`�□")


def _has_garble_chars(text: str) -> bool:
    return any(ch in _GARBLE_CHARS for ch in text)


# --- JP-plausibility (FIX P1-1) -------------------------------------------
# The confidence gate above only fires when ``ocr_conf`` is LOW. The worst
# failures, though, are garbled OCR carrying FALSELY HIGH confidence: e.g.
# page 070's vertical title "3へ続く…" was misread as "..?っく混みますよ" at
# ocr_conf 0.91, sailed past the gate, and was "smoothed" by the LLM into a
# confident-wrong "It's going to be crowded.". The signals below add a
# linguistic-plausibility check that catches such lines regardless of
# confidence, while staying narrow enough to leave genuine dialogue untouched.
#
# Calibration note: both signals were validated against the stored replay
# corpus (.bench/full_pipeline/588828_mesu2_insp, 649 bubbles). They fire on
# the known garbles (070, 071-credit, 074-handle) and on ZERO real dialogue
# lines. Being conservative is intentional — false-dropping real dialogue is
# worse than letting some garble through.

# Small kana (sokuon / yōon). A line legitimately *starting* with one is rare:
# only って (quotation), ったく (=まったく), and trailing っ.. survive in real
# speech — captured by the explicit exceptions in ``_has_garbled_leading_tsu``.
_SMALL_TSU = "っ"

# Leading punctuation we strip before inspecting the first real glyph.
_LEAD_PUNCT = set("。、！？!?「」『』（）()・ー〜….,　 \"'")

# A run of this many ASCII letters embedded in otherwise-Japanese text is the
# signature of OCR latching onto a logo / URL / handle (071 credit line,
# 074 "oisxo…ovitter"). A *single* stray letter or digit is normal (年号,
# counters, "1セット") so the threshold is on letters and is >= 3.
_MAX_LATIN_LETTERS_IN_JP = 3


def _is_hiragana(ch: str) -> bool:
    return 0x3040 <= ord(ch) <= 0x309F


def _is_katakana(ch: str) -> bool:
    return 0x30A0 <= ord(ch) <= 0x30FF or 0xFF65 <= ord(ch) <= 0xFF9F


def _is_kanji(ch: str) -> bool:
    return 0x4E00 <= ord(ch) <= 0x9FFF or 0x3400 <= ord(ch) <= 0x4DBF


def _is_japanese_glyph(ch: str) -> bool:
    return _is_hiragana(ch) or _is_katakana(ch) or _is_kanji(ch)


def _strip_leading_punct(norm: str) -> str:
    i = 0
    while i < len(norm) and norm[i] in _LEAD_PUNCT:
        i += 1
    return norm[i:]


def _has_garbled_leading_tsu(norm: str) -> bool:
    """Line begins with a small tsu ``っ`` glued to a garbled prefix.

    Real Japanese never opens with ``っ`` + an arbitrary kana. The only genuine
    forms are ``って`` (quotation/"like"), ``ったく`` (=まったく), and a trailing
    ``っ`` (``っ..`` cut-off speech). So we flag ``っ`` followed by a kana that
    is NOT て/た. This catches page 070's "..?っく…" prefix and nothing else in
    the replay corpus.
    """
    body = _strip_leading_punct(norm)
    if len(body) < 2 or body[0] != _SMALL_TSU:
        return False
    nxt = body[1]
    if nxt in "てた":  # って / ったく — legitimate.
        return False
    # Only treat it as garbled when a kana actually follows (a trailing っ
    # before punctuation/end is real cut-off speech and was already excluded by
    # the length check / non-kana next char).
    return _is_hiragana(nxt) or _is_katakana(nxt)


def _has_latin_intrusion(norm: str) -> bool:
    """Many ASCII letters wedged into Japanese text -> logo/URL garble.

    Requires Japanese to be present (so genuine all-Latin lines, handled
    elsewhere, are not double-counted) and at least ``_MAX_LATIN_LETTERS_IN_JP``
    ASCII letters. A lone digit/letter (年号, "1セット目") stays plausible.
    """
    latin = sum(1 for ch in norm if ch.isascii() and ch.isalpha())
    if latin < _MAX_LATIN_LETTERS_IN_JP:
        return False
    return any(_is_japanese_glyph(ch) for ch in norm)


# --- duplication garble (FIX P1-2) ----------------------------------------
# The DOMINANT real failure mode (144 bubbles, avg severity 2.53): PARSeq
# misreads dense / stylized vertical kana into duplicated adjacent characters
# and immediate phrase repetition, carrying FALSELY HIGH confidence (0.76-0.92)
# so the confidence threshold never fires. Examples:
#   身代わり -> 身身わわ      吐気 -> 吐吐気       濯濯バサミ
#   また昨日みたいな -> また昨日みたいなまた昨日みたいな (whole-phrase dup)
#   妄想止まらない -> 妄..妄ま定れいい妄.想止止らな (corrupt + 止止 dup)
# These signals are confidence-INDEPENDENT (run in is_implausible_japanese).
#
# Conservatism: false-dropping real dialogue is worse than missing some garble,
# so legitimate Japanese reduplication is whitelisted before flagging.

# Legitimate doubled-KANJI words. Real reduplication normally uses the 々
# iteration mark (人々), but PARSeq may emit the literal doubled kanji, so we
# whitelist both members of each common pair. An adjacent doubled kanji NOT in
# this set is the garble signal.
_LEGIT_KANJI_REDUP = {
    "様", "段", "人", "我", "色", "時", "方",
    "国", "日", "別", "中", "数", "順", "程",
    "個", "村", "家", "山", "木",  # 個々 村々 家々 etc.
}

# Iteration mark — a glyph that *is* legitimate reduplication; never a garble.
_ITERATION_MARK = "々"

# Legitimate doubled-KATAKANA fragments. We only treat doubled KATAKANA as a
# garble signal (doubled hiragana is far too common in real text: long-vowel
# spellings おお/ええ, mimetics, emphatic stretches), and even then we whitelist
# katakana laughter (ハハ/フフ/ヘヘ/ホホ) which is genuine speech.
_LEGIT_KATAKANA_DOUBLE = {"ハ", "フ", "ヘ", "ホ"}

# BUG FIX (audit: p114 idx9 "ヒヒ.." @ conf 0.9082/0.99 silently dropped): this
# rule ran UNCONDITIONALLY — before the length-based short-text carve-out in
# ``is_garbled_low_conf`` even gets a chance to run — so a short doubled-kana
# SFX/giggle like ヒヒ, アア, ヴヴ, シュウウ was always dropped regardless of
# confidence, no whitelist escape hatch.
#
# Re-verified against the 650-row calib table
# (``scripts/eval/scorecards/ikenie4/preds_for_gold_v1_fair.jsonl``): EVERY
# false drop this rule produced there (ヒヒ.. x3, アア, ヴヴ x3, シュウウ) was a
# clean, gold-exact SFX/moan under ``_DUP_KANA_MIN_LEN_TO_FLAG`` chars, while
# EVERY genuine dup-garble catch (アソコアア, ...チチンの, ...ババブブババブ..,
# ...濯ササ, ...セッッスく) was at/above it. So — mirroring the module's existing
# short-text carve-out philosophy (``_SHORT_TEXT_MAX_LEN`` below) — short
# strings are exempt from this rule instead of widening the katakana
# whitelist: a broader whitelist (e.g. adding bare ア) would have silently
# un-caught ``アソコアア`` (a real garbled-suffix duplication at conf 0.92),
# whereas gating on length leaves that 5-char case caught (5 is not < 5) and
# recovers only the shorter genuine SFX.
_DUP_KANA_MIN_LEN_TO_FLAG = 5


def _adjacent_dup_kanji(norm: str) -> bool:
    """An adjacent ``X X`` kanji pair that is NOT legitimate reduplication.

    Real reduplicated kanji words are whitelisted (``_LEGIT_KANJI_REDUP``); any
    other doubled kanji (身身, 吐吐, 濯濯, 止止) is the PARSeq dup-garble signal.
    """
    for i in range(len(norm) - 1):
        a, b = norm[i], norm[i + 1]
        if a == b and _is_kanji(a) and a not in _LEGIT_KANJI_REDUP:
            return True
    return False


def _adjacent_dup_kana(norm: str) -> bool:
    """An adjacent ``X X`` KATAKANA pair that is not whitelisted laughter.

    Deliberately KATAKANA-ONLY. Doubled hiragana is rejected as a signal because
    it occurs in genuine text (long-vowel おお/ええ as in 大きい/大阪, mimetics),
    so flagging it false-drops real dialogue. Doubled katakana is far more
    garble-like (katakana long vowels use ー, not vowel doubling), and the only
    common legit form is laughter (ハハ/フフ), which is whitelisted.

    Length-gated below ``_DUP_KANA_MIN_LEN_TO_FLAG``: short doubled-katakana
    fragments (ヒヒ, アア, ヴヴ, シュウウ) are overwhelmingly real SFX/interjection
    text, not PARSeq dup-garble, which shows up on longer/denser runs in the
    calib table. See the constant's comment for the calibration evidence.
    """
    if len(norm) < _DUP_KANA_MIN_LEN_TO_FLAG:
        return False
    for i in range(len(norm) - 1):
        a, b = norm[i], norm[i + 1]
        if a == b and _is_katakana(a) and a not in _LEGIT_KATAKANA_DOUBLE:
            return True
    return False


def _repeated_bigram_garble(norm: str) -> bool:
    """High ratio of repeated character-bigrams -> phrase-repetition garble.

    Whole-phrase OCR duplication (また昨日みたいなまた昨日みたいな) produces many
    repeated bigrams. Genuine dialogue rarely exceeds ~30% repeated bigrams, so
    a >= 0.5 ratio over a long-enough line is a strong dup signal. Gated on
    length to avoid firing on tiny strings where one repeat dominates.
    """
    glyphs = [c for c in norm if _is_japanese_glyph(c)]
    if len(glyphs) < 8:
        return False
    bigrams = [glyphs[i] + glyphs[i + 1] for i in range(len(glyphs) - 1)]
    if not bigrams:
        return False
    unique = len(set(bigrams))
    repeated_ratio = 1.0 - (unique / len(bigrams))
    return repeated_ratio >= 0.5


def _immediate_substring_dup(norm: str) -> bool:
    """Line is ``P + P`` — a phrase repeated immediately back-to-back.

    Catches whole-phrase OCR duplication whose two JP-glyph halves are identical
    (また昨日みたいな + また昨日みたいな). Requires the repeated unit to be
    non-trivial (>= 4 JP glyphs) so ordinary short doubled words don't trip it.
    """
    stripped = "".join(c for c in norm if _is_japanese_glyph(c))
    n = len(stripped)
    if n < 8 or n % 2 != 0:
        return False
    half = n // 2
    if half < 4:
        return False
    return stripped[:half] == stripped[half:]


# --- P+P collapse (FIX P3-1) ----------------------------------------------
# ``_immediate_substring_dup`` LOCATES whole-phrase ``P + P`` OCR repeats. The
# old behaviour DROPPED them, but ~20% of those drops are *clean* high-confidence
# Japanese whose two halves are an exact repeat (お母さんお母さん@0.93,
# また昨日みたいな…@0.92). Dropping turns them into silent omissions. Instead we
# COLLAPSE the repeat back to one copy and KEEP the line — but only when the
# collapsed half is itself plausible Japanese (so we never "rescue" a garble).
# Mirrors ``sfx_glossary._collapse_sfx_repeat`` but for arbitrary dialogue text,
# firing ONLY on exact whole-string OR exact JP-glyph-only half equality
# (near-zero regression risk).


# Minimum repeated-unit length (in chars) for the whole-string collapse. Mirrors
# the >= 4 JP-glyph floor of ``_immediate_substring_dup`` so short legit
# reduplications (ますます, どきどき, はは) are NEVER collapsed.
_MIN_COLLAPSE_HALF = 4


def _exact_half_repeat(s: str) -> str | None:
    """Return the collapsed half if ``s`` is exactly ``P + P``, else None.

    Even length, two identical halves, and a non-trivial unit (half length >=
    ``_MIN_COLLAPSE_HALF``) so short legit doubled words (ますます, はは) are not
    collapsed. Operates on the raw string (used for the whole-string check).
    """
    n = len(s)
    if n < 2 * _MIN_COLLAPSE_HALF or n % 2 != 0:
        return None
    half = n // 2
    if s[:half] == s[half:]:
        return s[:half]
    return None


def collapse_immediate_dup(text: str) -> str | None:
    """Collapse an exact whole-phrase ``P + P`` repeat to a single ``P``.

    Returns the collapsed string when ``text`` is a clean immediate repeat AND
    the collapsed half is NOT implausible Japanese; returns None when no
    collapse applies (caller keeps the original text). Two repeat shapes fire:

      * exact WHOLE-STRING repeat (お母さんお母さん -> お母さん, including any
        shared trailing punctuation: ``ABC…ABC…`` only collapses if the whole
        string halves equal),
      * exact JP-GLYPH-ONLY repeat (また昨日みたいな…また昨日みたいな… where the
        punctuation differs between halves) — reconstructed from the first half
        of the ORIGINAL string by length, preserving that half's punctuation.

    Conservative: the collapsed half must pass ``is_implausible_japanese`` ==
    False, so a doubled GARBLE (身身わわ-style) is never silently rescued — it
    still falls through to the drop path.
    """
    norm = unicodedata.normalize("NFC", text).strip()
    if not norm:
        return None

    # (a) exact whole-string repeat (halves identical incl. punctuation).
    whole = _exact_half_repeat(norm)
    if whole is not None and not is_implausible_japanese(whole):
        return whole

    # (b) exact JP-glyph-only repeat: the JP glyphs form P+P even though the
    # surrounding punctuation differs. Rebuild one copy from the original by
    # taking the prefix up to (and including) the glyphs of the first half.
    glyphs = [c for c in norm if _is_japanese_glyph(c)]
    n = len(glyphs)
    if n >= 8 and n % 2 == 0:
        half = n // 2
        if half >= 4 and glyphs[:half] == glyphs[half:]:
            # Walk the original string until we've consumed ``half`` JP glyphs;
            # that prefix is one clean copy (keeps its trailing punctuation).
            seen = 0
            cut = len(norm)
            for i, ch in enumerate(norm):
                if _is_japanese_glyph(ch):
                    seen += 1
                    if seen == half:
                        cut = i + 1
                        break
            candidate = norm[:cut].strip()
            if candidate and not is_implausible_japanese(candidate):
                return candidate

    return None


# --- substitution / perplexity garble guard (FIX P3-3) --------------------
# The dup-predicates are BLIND to substitution garbles where a plausible char is
# swapped for another plausible char (もっ張って<-引っ張って) or short noise
# (ヤヌー界): every individual bigram is locally valid, so neither the dup
# heuristics nor a min-bigram-probability test separates them from real text.
#
# CALIBRATION FINDING (see scripts/build_jp_bigram_table.py + report): a char-
# bigram model CANNOT safely catch SHORT substitution garbles — they sit inside
# the legit distribution (もっ張って ppl 26 ~= 引っ張って ppl 18), and short legit
# names/SFX/NSFW (あゆむ ppl 12847, ピろッ ppl 5344) outscore even the noise
# garbles. So this guard is deliberately scoped to ONLY what a bigram model can
# catch WITHOUT false-dropping clean text: LONG, HIGH-ENTROPY noise scrambles
# (also Chinese-leak pages). Validated to flag 0 clean dialogue / 0 NSFW lines
# across 14.5k real bubbles while catching long noise like ゴム昔化湖ゴム首次角.
# Flagged lines route to ERASE-ONLY (never translated), so a false flag can only
# erase real ink, never fabricate a caption.

_BIGRAM_TABLE_PATH = Path(__file__).resolve().parents[1] / "data" / "jp_char_bigram.json"

# add-k smoothing constant for the conditional bigram model.
_SMOOTH_K = 0.1
# Guard fires only when ALL of these hold (conservative — long noise only):
_SUBST_MIN_GLYPHS = 8        # short lines are out of reach; never inspect them
_SUBST_PPL_THRESHOLD = 1500.0
_SUBST_UNSEEN_FRACTION = 0.55  # >= 55% of bigrams never seen in the GT corpus


@lru_cache(maxsize=1)
def _load_bigram_table() -> tuple[dict, dict, int]:
    """Load (unigram, bigram, n_unigram_types). Empty on missing file."""
    try:
        data = json.loads(_BIGRAM_TABLE_PATH.read_text(encoding="utf-8"))
        return (
            data.get("unigram", {}),
            data.get("bigram", {}),
            int(data.get("n_unigram_types", 0)) or len(data.get("unigram", {})),
        )
    except Exception:
        return ({}, {}, 0)


def _is_substitution_garble(norm: str) -> bool:
    """True for LONG high-entropy noise the dup-predicates miss (FIX P3-3).

    Confidence-INDEPENDENT. Uses a corpus-grounded char-bigram model (built
    offline from 260k+ real manga GT lines). Conservative by design: requires a
    long line (>= 8 JP glyphs), a high unseen-bigram fraction, AND high
    perplexity together, so short legit names / SFX / NSFW dialogue (which can
    have high perplexity on their own) are never flagged. Returns False when the
    table is unavailable (fail-open).
    """
    uni, bi, vocab = _load_bigram_table()
    if not bi or vocab == 0:
        return False
    glyphs = [c for c in norm if _is_japanese_glyph(c)]
    n = len(glyphs)
    if n < _SUBST_MIN_GLYPHS:
        return False
    n_bigrams = n - 1
    unseen = 0
    logp = 0.0
    for i in range(n_bigrams):
        a, b = glyphs[i], glyphs[i + 1]
        c_ab = bi.get(a + b, 0)
        c_a = uni.get(a, 0)
        if c_ab == 0:
            unseen += 1
        p = (c_ab + _SMOOTH_K) / (c_a + _SMOOTH_K * vocab)
        logp += math.log(p)
    perplexity = math.exp(-logp / n_bigrams)
    unseen_frac = unseen / n_bigrams
    return perplexity >= _SUBST_PPL_THRESHOLD and unseen_frac >= _SUBST_UNSEEN_FRACTION


# --- DUP-signal confidence ceiling (FIX P3-2) -----------------------------
# The dup-only signals (_adjacent_dup_*, _immediate_substring_dup,
# _repeated_bigram_garble) are heuristics for a PARSeq failure mode, but they
# also trip on a few CLEAN high-confidence bubbles (お母さん@0.93 -> dropped,
# orphaning its partner which then renders "InNo"). When OCR confidence is high
# the recognizer is very likely correct, so we SKIP the dup-only signals at/above
# this ceiling. The UNCONDITIONAL signals (latin-intrusion, garbled-leading-tsu,
# substitution-perplexity) still run — they catch confidently-wrong OCR that the
# recognizer is sure about but is still garbage.
DUP_CONF_CEILING = 0.88


def is_implausible_japanese(text: str, ocr_conf: float | None = None) -> bool:
    """True if ``text`` reads as garbled OCR despite being mostly Japanese.

    A *linguistic*-plausibility heuristic (NOT confidence-based by default) so
    that confidently-garbled OCR is caught even at high OCR confidence.
    Deliberately narrow: it only returns True on patterns that cannot occur in
    genuine Japanese dialogue, so it does not drop real lines.

    Signal groups:
      UNCONDITIONAL (always run, any confidence):
        * Garbled leading small-tsu prefix (page 070 "..?っく混みますよ").
        * Heavy ASCII-letter intrusion in Japanese text (logo/URL garble).
        * Substitution/perplexity garble (FIX P3-3): a char-bigram plausibility
          guard for long high-entropy noise the dup-predicates are blind to.
        * Adjacent doubled kanji/kana (FIX-2 recalibration): the ``身身`` / ``吐吐``
          dense-kana failure mode carries FALSELY HIGH confidence (身身わわ@0.92),
          so the ``DUP_CONF_CEILING`` was letting confidently-wrong dup-garble
          through. The calib table (650 rows) showed 0 false positives for these
          two signals, so they are now unconditional like latin-intrusion.
      DUP-ONLY (skipped when ``ocr_conf >= DUP_CONF_CEILING`` — FIX P3-2):
        * whole-phrase immediate repetition, high repeated-bigram ratio — these
          length/bigram signals DO trip a few clean conf-0.9 bubbles
          (お母さんお母さん@0.93), so a high-confidence line is spared them.
    """
    norm = unicodedata.normalize("NFC", text).strip()
    if not norm:
        return False

    # Unconditional signals — run irrespective of confidence.
    if _has_garbled_leading_tsu(norm):
        return True
    if _has_latin_intrusion(norm):
        return True
    if _is_substitution_garble(norm):
        return True
    # FIX-2: adjacent doubled kanji/kana are a hard OCR dup-signature that occurs
    # even at falsely-high confidence (身身わわ@0.92 — the module's own docstring
    # example). They are precise (0 false positives on the 650-row calib table),
    # so unlike the length/bigram dup signals they run UNCONDITIONALLY and the
    # ``DUP_CONF_CEILING`` no longer exempts them.
    if _adjacent_dup_kanji(norm):
        return True
    if _adjacent_dup_kana(norm):
        return True

    # Dup-only length/bigram signals — a high-confidence recognition is very
    # likely correct, so skip these to avoid false-dropping clean dialogue whose
    # two halves legitimately repeat (お母さんお母さん@0.93) — FIX P3-2.
    if ocr_conf is not None and ocr_conf >= DUP_CONF_CEILING:
        return False

    if _immediate_substring_dup(norm):
        return True
    if _repeated_bigram_garble(norm):
        return True
    return False


# Longer lines that are mostly real Japanese are treated as dialogue we should
# not silently drop even at low confidence — a hard-to-read but genuine line is
# better translated than dropped. SFX/garble that hallucinates is short.
_DIALOGUE_MIN_LEN = 12

# --- FIX-1: operating point fit from the 650-row calibration table ---------
# ``scripts/eval/scorecards/ikenie4/preds_for_gold_v1_fair.jsonl`` pairs each
# bubble's OCR (``jp``) with its gold transcription (``gold_jp``) and the PARSeq
# recognition confidence (``ocr_conf``). Scoring correctness as char-similarity
# to gold >= 0.9 revealed the old 0.65 gate was badly miscalibrated:
#
#   * The 0.65-0.80 band is a NOISE TROUGH — only ~40-55% correct vs gold
#     (0.75-0.80 dips to 28%); correctness keeps climbing through 0.85 (65%) to
#     0.90 (95%). Keeping mid-band lines feeds the LLM half-garbled OCR that it
#     "smooths" into confident-wrong captions rendered on the page.
#   * 11 of 14 sub-0.65 rows were EXACT gold matches — and they were SHORT
#     SFX / moans / numbers (もみせ, ボン, 濃厚, 56, いいきさ) that the old
#     length-12 rule silently DROPPED. Short strings are ~98% correct vs gold
#     across every confidence band (len<5: 98%, even sub-0.65 short: ~71%).
#
# So the recalibrated gate: (a) a SHORT-TEXT CARVE-OUT keeps very short strings
# regardless of confidence (the unconditional structural garble checks above
# still remove dup/latin/substitution noise first), and (b) a RAISED long-text
# threshold drops multi-char lines below ~0.80. Simulated over the 650 rows this
# lifts kept-set precision 85.3% -> 88.6% for a near-zero recall cost of
# 98.6% -> 98.0% (drops 27 wrong mid-conf lines, recovers 5 correct short SFX;
# the 11 dropped correct long lines survive as page CONTEXT via
# is_dialogue_context_candidate).
#
# Threshold choice — 0.80 over the higher-precision 0.85 (91.8%/96.7%): the
# 0.80-0.85 band is 42% correct so 0.85 nets more precision, but it also drops
# GENUINE 0.80-conf dialogue columns, and losing a middle column of a multi-
# column balloon worsens THE systemic defect (balloon fragmentation, audit §2).
# 0.80 preserves balloon integrity while still clearing the 0.65-0.80 trough.
#
# Strings shorter than this many chars are exempt from the confidence gate.
_SHORT_TEXT_MAX_LEN = 5
# Multi-char lines must clear this confidence to be kept (the precision/balloon-
# recall knee). Used as a FLOOR: an explicitly-stricter caller threshold wins.
_LONG_CONF_THRESHOLD = 0.80


def is_garbled_low_conf(
    text: str,
    ocr_confidence: float,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    min_jp_ratio: float = 0.5,
    katakana_max_len: int = 6,
    check_plausibility: bool = True,
) -> bool:
    """True if this bubble should be DROPPED before translation.

    Calibrated on the 650-row conf x sim-to-gold table (see ``_LONG_CONF_THRESHOLD``
    /``_SHORT_TEXT_MAX_LEN`` above). PARSeq recognition confidence separates real
    dialogue from garble, but the old 0.65 cut let a 0.65-0.85 noise trough
    (~40-55% correct) through while silently dropping short-but-correct SFX. So:

      * Structural garble (dup-kanji/kana, latin-intrusion, substitution noise)
        -> drop at ANY confidence (``is_implausible_japanese``, FIX P1-1/FIX-2).
      * Confidence >= max(``conf_threshold``, ``_LONG_CONF_THRESHOLD``) -> keep.
      * SHORT-TEXT CARVE-OUT: a clean string < ``_SHORT_TEXT_MAX_LEN`` chars is
        kept regardless of confidence (SFX/moans/numbers are ~98% correct vs
        gold — the old length rule was silently dropping the correct ones).
      * Any other sub-threshold multi-char line -> drop (the mid-band is
        unreliable OCR; a dropped genuine line is still passed as page CONTEXT
        via ``is_dialogue_context_candidate``, never fully lost).

    FIX P1-1: the plausibility check runs FIRST and drops garble regardless of
    confidence, catching OCR that carries FALSELY HIGH confidence (page 070's
    "..?っく混みますよ" at 0.91). It stays narrow enough to leave real dialogue
    untouched (validated on 600+ replay lines with zero false drops).
    """
    norm = unicodedata.normalize("NFC", text).strip()

    # FIX P3-1: a clean whole-phrase ``P + P`` repeat (お母さんお母さん@0.93) is
    # NOT a drop — it COLLAPSES to one copy and is KEPT (the caller applies
    # ``collapse_immediate_dup`` to the text). Recognising it here means its
    # ``_immediate_substring_dup`` signal below must not orphan the bubble.
    is_collapsible_dup = norm and collapse_immediate_dup(norm) is not None
    if is_collapsible_dup:
        return False

    # Plausibility check runs irrespective of confidence — this is the whole
    # point of P1-1/FIX-2 (catch confidently-wrong OCR). It runs BEFORE the
    # short-text carve-out so structural garble is never rescued by being short
    # (身身わわ@0.92 is len-4 but still dropped here). FIX P3-2: pass the
    # confidence so the length/bigram dup signals honour the ceiling.
    if check_plausibility and norm and is_implausible_japanese(norm, ocr_confidence):
        return True

    # Confidence gate: multi-char lines need the raised long-text threshold; an
    # explicitly-stricter caller threshold still wins (used as a floor).
    long_threshold = max(conf_threshold, _LONG_CONF_THRESHOLD)
    if ocr_confidence >= long_threshold:
        return False

    # --- below threshold ---
    if not norm:
        # Empty/low-conf -> nothing to translate; safe to drop.
        return True

    # Garble chars at low conf -> always drop (before the carve-out: a short
    # ``]]/``-style scrawl must not be kept just for being short).
    if _has_garble_chars(norm):
        return True

    # FIX-1 SHORT-TEXT CARVE-OUT: very short clean strings (SFX / moans / numbers
    # / interjections) are ~98% correct vs gold at any confidence, and the
    # structural garble checks above already removed dup/latin/substitution
    # noise. Keep them rather than silently dropping (the old length-12 rule lost
    # 11 sub-0.65 gold-exact SFX per the calib table).
    if len(norm) < _SHORT_TEXT_MAX_LEN:
        return False

    # Non-Japanese / low JP-ratio at low conf -> drop.
    if not is_japanese_text(norm, min_jp_ratio, katakana_max_len):
        return True
    analysis = analyze_characters(norm)
    if analysis.japanese_ratio < _MIN_JP_RATIO_FOR_LOWCONF:
        return True

    # FIX-1 RAISED LONG-TEXT THRESHOLD: a multi-char clean-Japanese line below the
    # threshold sits in the 0.65-0.85 noise trough (~40-55% correct vs gold). The
    # old rule KEPT these (better-to-translate-a-hard-line); recalibration DROPS
    # them — a confident-wrong caption rendered on the page is worse than an
    # omission, and the line still survives as page CONTEXT when conf >= 0.65.
    return True


# Speaker / pronoun references. A SHORT dropped line carrying one of these is
# exactly the continuity context the v11 page is for (who is speaking / being
# referred to), so we keep it even below the dialogue-length cutoff. Covers the
# common first/second/third-person pronouns and the family-role address terms
# that drive he/she/I/you selection in manga dialogue.
_SPEAKER_REF_TOKENS = (
    "僕", "私", "俺", "あたし", "わたし", "ぼく", "おれ",
    "君", "きみ", "あなた", "お前", "おまえ", "貴方",
    "お母さん", "母さん", "ママ", "お父さん", "父さん", "パパ",
    "お兄ちゃん", "兄さん", "お姉ちゃん", "姉さん",
    "おばさん", "おじさん", "先生", "彼", "彼女",
)


def _has_speaker_reference(norm: str) -> bool:
    return any(tok in norm for tok in _SPEAKER_REF_TOKENS)


# A short dropped line still needs a minimum substance to be context (avoid
# admitting 1-2 char fragments). Half the dialogue cutoff.
_CONTEXT_MIN_LEN_WITH_SPEAKER = 4


def is_dialogue_context_candidate(
    text: str, ocr_confidence: float | None = None
) -> bool:
    """True if a GATE-DROPPED line is real-enough DIALOGUE to keep as CONTEXT.

    The v11 page-context model translates one marked line while seeing the whole
    page's dialogue (speaker/pronoun continuity). A dialogue line dropped before
    translation (OCR-gate / garble) still belongs in the numbered "Page:" context
    so the page the model sees has no holes — BUT a pure-SFX box, or a genuinely
    garbled low-confidence scrawl, must NOT pollute that dialogue context. This
    decides which dropped lines are kept as CONTEXT-ONLY (never rendered).

    Policy (validated on IK4 page 5):
      * exclude empty / glossary-SFX / garble-char / mostly-non-Japanese,
      * exclude LOW-OCR-confidence lines when ``ocr_confidence`` is supplied
        (genuine garble like the conf-0.49 "平速ととの…" scrawl — pure noise),
      * KEEP a short line that names a speaker / carries a pronoun
        (お母さん, 僕, 私, …) — that is precisely the continuity the page is for
        (IK4 p5 "お母さんは僕の…" fixes the He/She pronoun on the marked line),
      * KEEP a dialogue-LENGTH mostly-Japanese line,
      * otherwise exclude (short generic fragment / exclamation).
    """
    norm = unicodedata.normalize("NFC", text or "").strip()
    if not norm:
        return False
    # Glossary-matched SFX (ぬちょ, ビクン, …) are handled out-of-band; never
    # dialogue context. Local import keeps this module dependency-light.
    try:
        from app.services.sfx_glossary import sfx_pre_translate
        if sfx_pre_translate(norm) is not None:
            return False
    except Exception:
        pass
    if _has_garble_chars(norm):
        return False
    analysis = analyze_characters(norm)
    if analysis.japanese_ratio < _MIN_JP_RATIO_FOR_LOWCONF:
        return False
    # Genuinely-garbled low-confidence scrawl is noise, not context. Only applies
    # when confidence is known (the gate has it); the text-only call stays
    # length/JP-ratio based for back-compat.
    if ocr_confidence is not None and ocr_confidence < DEFAULT_CONF_THRESHOLD:
        return False
    # A speaker/pronoun reference makes even a short line valuable context.
    if _has_speaker_reference(norm) and len(norm) >= _CONTEXT_MIN_LEN_WITH_SPEAKER:
        return True
    # Otherwise require dialogue length: short generic fragments are SFX-ish.
    if len(norm) < _DIALOGUE_MIN_LEN:
        return False
    return True


def should_erase_dropped(text: str) -> bool:
    """True if a gate-dropped region is real Japanese ink worth ERASING.

    A region dropped by ``is_garbled_low_conf`` is never translated, but it may
    still be genuine Japanese SFX/scrawl on the page. Leaving it untouched
    renders the raw Japanese into the final image. This decides whether such a
    dropped region should be inpainted away (erased) even though no translation
    will be drawn over it. Conservative: a non-empty region must contain at
    least one Japanese glyph (so stray Latin/garble-only crops are left alone).

      * Empty / near-empty short text -> erase (an empty low-conf crop is real
        ink the recognizer couldn't decode, typical of stylized SFX).
      * Garble char present AND a JP glyph present -> erase.
      * japanese_ratio > 0.5 -> erase.
    """
    norm = unicodedata.normalize("NFC", text).strip()
    analysis = analyze_characters(norm)

    if not norm:
        # Empty/near-empty: real ink the recognizer gave up on -> erase.
        return True

    # Non-empty text must contain a Japanese glyph to be worth erasing.
    if analysis.japanese_count == 0:
        return False

    if _has_garble_chars(norm):
        return True
    if analysis.japanese_ratio > _MIN_JP_RATIO_FOR_LOWCONF:
        return True
    return False

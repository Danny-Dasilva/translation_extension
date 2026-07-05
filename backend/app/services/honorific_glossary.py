"""Honorific preservation (post-translation glossary).

PROBLEM
-------
The furube/Ikenie side-by-side audit against REAL human EN scanlations found
that the v1 model sometimes DROPS a Japanese honorific (-san/-sama/-chan/-kun/
-sensei/-senpai) that the human translator keeps — e.g. Ikenie5 p017: the
source ``そ...それがまーきさまちょうど切れてて...`` addresses "Maki-sama" and
the human EN renders ``MA-KI-SAMA, ...`` while ours drops the name+honorific
entirely (``"U-um... I was just about to go buy some tomorrow..."``).

This module is the mirror image of ``output_sanitize.strip_romaji_honorifics``
(which strips a honorific the model HALLUCINATED with no source basis) — here
we RESTORE a honorific the SOURCE demonstrably carries when the model's output
dropped it. Composition with the existing strip pass is intentional: this
function runs LAST in the ``translation_postedit`` chain (after the strip), so
a model-kept, source-confirmed honorific that the strip pass removed is put
back — only when the source proves it belongs; a genuinely hallucinated one
(no source basis) is never resurrected.

WHY A POST-EDIT (not a prompt/retrain change)
----------------------------------------------
Same reason as every other module in this package: the v1 model is train/serve
format-sensitive (an in-prompt glossary/instruction change is the documented
~95% chrF++ collapse risk class). All correction here is a pure post-pass on
the model's EN output, conditioned on the JP source.

DESIGN: precision over recall, two tiers of increasing risk
-------------------------------------------------------------
1. REGISTRY PATH (kanji-safe): reuses ``name_glossary.NAME_LOCKS`` — the
   already-curated per-title ``jp_kana -> canonical_en`` roster. Kanji names
   are NEVER guess-transliterated here (a kanji reading is genuinely
   ambiguous without furigana); we only act when the character is already a
   verified roster entry.
2. MECHANICAL PATH (kana-only): a compact deterministic Hepburn-ish
   kana->romaji table romanizes a honorific-suffixed name span, but ONLY when
   the span is katakana and/or contains the long-vowel mark "ー" (a strong
   name-stylization signal that essentially never appears in ordinary
   grammatical hiragana) — this is what catches ``まーきさま``. Plain
   hiragana with no such signal is REJECTED (too likely to be a mis-parsed
   verb/particle run, not a name).

Within each path, two graduated actions:
  * TIER A (SAFE, default ON): the candidate name ALREADY appears bare in the
    EN output (the model kept the name, dropped only the honorific) -> append
    the honorific suffix to that occurrence. Never invents new text.
  * TIER B (EXPERIMENTAL, default OFF): the candidate name is ABSENT from the
    EN output entirely (the model dropped the whole vocative, e.g. p017) ->
    prepend "Name-suffix, " to the output. This DOES invent new (but
    source-conditioned, mechanically-derived) text, and kana romanization is
    inherently approximate (``まーき`` -> "Maaki", not the human's stylized
    "Ma-ki"), so it is gated behind
    ``settings.postedit_restore_dropped_honorific_vocative`` (default False)
    pending validation. See app/config.py for the flag.

Conservative by construction: never fires on a source with no honorific;
never invents an honorific-bearing name with no source basis; idempotent
(re-running does not double-append or double-prepend).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app.services.name_glossary import NAME_LOCKS

# --------------------------------------------------------------------------- #
# 1. Honorific vocabulary
# --------------------------------------------------------------------------- #
# jp honorific token -> canonical EN suffix (no leading hyphen).
HONORIFIC_JP_TO_SUFFIX: dict[str, str] = {
    "さま": "sama",
    "様": "sama",
    "さん": "san",
    "ちゃん": "chan",
    "くん": "kun",
    "君": "kun",
    "先生": "sensei",
    "せんぱい": "senpai",
    "先輩": "senpai",
}
_HONORIFIC_ALT = "|".join(sorted(HONORIFIC_JP_TO_SUFFIX, key=len, reverse=True))
# Every honorific suffix we might append, for the "already honorific-marked"
# guard (used so we never double up, e.g. "Kaede-chan-sama").
_EN_SUFFIX_ALT = "|".join(sorted(set(HONORIFIC_JP_TO_SUFFIX.values()), key=len, reverse=True))

# --------------------------------------------------------------------------- #
# 2. Generic address terms that are NOT a personal name+honorific.
# --------------------------------------------------------------------------- #
# These are common kinship/role vocatives (already handled, if at all, by their
# own dedicated glossary — e.g. address_glossary.lock_address_terms for
# おばさん). Listed here (hiragana + katakana stylizations) purely as a defense
# -in-depth guard: the katakana/long-vowel filter below already excludes most
# of these, but an explicit blocklist keeps the intent obvious and covers the
# katakana-stylized forms manga sometimes uses for comedic emphasis.
_GENERIC_BLOCKLIST: frozenset[str] = frozenset(
    {
        "おばさん", "おばちゃん", "おじさん", "おじちゃん",
        "おばあさん", "おばあちゃん", "おじいさん", "おじいちゃん",
        "オバサン", "オジサン", "オバアサン", "オジイサン",
        "お母さん", "母さん", "おかあさん", "オカアサン",
        "お父さん", "父さん", "おとうさん", "オトウサン",
        "お姉さん", "姉さん", "おねえさん", "お姉ちゃん", "おねえちゃん", "ネエチャン",
        "お兄さん", "兄さん", "おにいさん", "お兄ちゃん", "おにいちゃん", "ニイチャン",
        "皆さん", "みなさん", "ミンナサン",
        "お客さん", "おきゃくさん", "オキャクサン",
        "奥さん", "おくさん", "オクサン",
        "旦那さん", "だんなさん", "ダンナサン",
        "息子さん", "むすこさん",
        "娘さん", "むすめさん",
        "赤ちゃん", "あかちゃん", "アカチャン",
        "坊ちゃん", "ぼっちゃん",
        "嬢ちゃん", "じょうちゃん",
    }
)

# --------------------------------------------------------------------------- #
# 3. Mechanical kana -> romaji (compact Hepburn-ish table).
# --------------------------------------------------------------------------- #
_KANA_ROMAJI: dict[str, str] = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "わ": "wa", "を": "o", "ん": "n",
    "ワ": "wa", "ヲ": "o", "ン": "n",
    "ゔ": "vu", "ヴ": "vu",
}
_YOUON: dict[str, str] = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "キャ": "kya", "キュ": "kyu", "キョ": "kyo",
    "ギャ": "gya", "ギュ": "gyu", "ギョ": "gyo",
    "シャ": "sha", "シュ": "shu", "ショ": "sho",
    "ジャ": "ja", "ジュ": "ju", "ジョ": "jo",
    "チャ": "cha", "チュ": "chu", "チョ": "cho",
    "ニャ": "nya", "ニュ": "nyu", "ニョ": "nyo",
    "ヒャ": "hya", "ヒュ": "hyu", "ヒョ": "hyo",
    "ビャ": "bya", "ビュ": "byu", "ビョ": "byo",
    "ピャ": "pya", "ピュ": "pyu", "ピョ": "pyo",
    "ミャ": "mya", "ミュ": "myu", "ミョ": "myo",
    "リャ": "rya", "リュ": "ryu", "リョ": "ryo",
}

# Name-span alphabet for the mechanical path: hiragana + katakana + the
# long-vowel mark, MINUS common single-mora grammatical particles/copula
# fragments (が/は/を/に/へ/と/も/や/の/で/ば/し/て/だ/ね/よ/わ/な/か). Japanese has
# no inter-word spacing, so a name is virtually always glued to the preceding
# particle in running prose (e.g. 「それが」+「まーき」+「さま」) — without this
# exclusion the non-greedy name-span capture can swallow the leading particle
# up to its length cap (observed: "がまーき" -> "Gamaaki" instead of "まーき" ->
# "Maaki"). Excluding these mora from the class means the capture simply
# cannot cross the particle boundary (it isn't part of the class at all), so
# the match starts cleanly at the real name. Deliberately EXCLUDES kanji
# entirely (kanji readings are ambiguous without furigana; those go through
# the registry path instead).
_JP_PARTICLE_EXCLUDE = set("はがをにへともやのでばしてだねよわなか")
_HIRAGANA_ALLOWED = "".join(
    chr(c) for c in range(0x3041, 0x3097) if chr(c) not in _JP_PARTICLE_EXCLUDE
)
_KATAKANA_ALLOWED = "".join(chr(c) for c in range(0x30A1, 0x30FB))  # ァ..ヺ
_NAME_SPAN_CHARS = _HIRAGANA_ALLOWED + _KATAKANA_ALLOWED + "ー"
_NAME_HONORIFIC_RE = re.compile(rf"([{_NAME_SPAN_CHARS}]{{1,4}}?)({_HONORIFIC_ALT})")

# OCR separators to strip before matching (dots/spaces/punctuation), mirroring
# name_glossary._normalize_jp.
_SEP_RE = re.compile(r"[.。、・…\s!?！？]+")


def _normalize_jp(jp: str) -> str:
    return _SEP_RE.sub("", jp)


def _is_all_katakana(span: str) -> bool:
    return all(("゠" <= c <= "ヿ") for c in span)


def _romanize_kana(span: str) -> Optional[str]:
    """Best-effort deterministic kana -> romaji for a short name span.

    Returns ``None`` if any character cannot be mapped (kanji, punctuation,
    unknown symbol) — the caller must then skip mechanical romanization for
    this span rather than guess.
    """
    out = ""
    i = 0
    n = len(span)
    while i < n:
        ch = span[i]
        two = span[i : i + 2]
        if two in _YOUON:
            out += _YOUON[two]
            i += 2
            continue
        if ch in ("っ", "ッ"):
            nxt_romaji = None
            if i + 1 < n:
                nxt_two = span[i + 1 : i + 3]
                if nxt_two in _YOUON:
                    nxt_romaji = _YOUON[nxt_two]
                elif span[i + 1] in _KANA_ROMAJI:
                    nxt_romaji = _KANA_ROMAJI[span[i + 1]]
            if nxt_romaji:
                out += nxt_romaji[0]
            i += 1
            continue
        if ch == "ー":
            if out:
                out += out[-1]
            i += 1
            continue
        if ch in _KANA_ROMAJI:
            out += _KANA_ROMAJI[ch]
            i += 1
            continue
        return None
    if not out:
        return None
    return out[:1].upper() + out[1:]


# --------------------------------------------------------------------------- #
# 4. Candidate collection (registry path + mechanical path).
# --------------------------------------------------------------------------- #
def _collect_candidates(jp: str) -> List[Tuple[str, str]]:
    """Return ``[(canonical_en_name, suffix), ...]`` found in ``jp``.

    ``canonical_en_name`` is either a verified per-title roster name (registry
    path) or a mechanically romanized kana span (mechanical path, katakana / long
    -vowel spans only). Deduplicated, order-preserving.
    """
    jp_norm = _normalize_jp(jp)
    out: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()

    # Registry path: only verified jp_kana -> canonical_en roster entries.
    for lock in NAME_LOCKS:
        if not lock.jp_kana or not lock.canonical:
            continue
        for honorific_jp, suffix in HONORIFIC_JP_TO_SUFFIX.items():
            if f"{lock.jp_kana}{honorific_jp}" in jp_norm:
                key = (lock.canonical.lower(), suffix)
                if key not in seen:
                    seen.add(key)
                    out.append((lock.canonical, suffix))

    # Mechanical path: katakana / long-vowel kana spans only.
    for m in _NAME_HONORIFIC_RE.finditer(jp_norm):
        full = m.group(0)
        if full in _GENERIC_BLOCKLIST:
            continue
        name_span, honorific_jp = m.group(1), m.group(2)
        if not (_is_all_katakana(name_span) or "ー" in name_span):
            continue
        romaji = _romanize_kana(name_span)
        if not romaji:
            continue
        suffix = HONORIFIC_JP_TO_SUFFIX[honorific_jp]
        key = (romaji.lower(), suffix)
        if key not in seen:
            seen.add(key)
            out.append((romaji, suffix))

    return out


# --------------------------------------------------------------------------- #
# 5. Apply candidates to the EN output.
# --------------------------------------------------------------------------- #
def _tier_a_append(en: str, name: str, suffix: str) -> Tuple[str, bool]:
    """Append ``-suffix`` to a BARE occurrence of ``name`` in ``en``.

    Only touches a whole-word match that is not already followed by a
    hyphenated honorific (any of them — never double up). Returns
    ``(text, changed)``.
    """
    pattern = re.compile(
        rf"\b{re.escape(name)}\b(?!-(?:{_EN_SUFFIX_ALT}))",
        re.IGNORECASE,
    )
    m = pattern.search(en)
    if not m:
        return en, False
    out = en[: m.end()] + f"-{suffix}" + en[m.end() :]
    return out, True


def _tier_b_prepend(en: str, name: str, suffix: str) -> Tuple[str, bool]:
    """Prepend ``"Name-suffix, "`` when ``name`` is absent from ``en`` entirely.

    Idempotent: skipped if ``en`` already starts with this vocative.
    """
    vocative = f"{name}-{suffix}"
    if re.match(rf"^\s*{re.escape(vocative)}\b", en, re.IGNORECASE):
        return en, False
    return f"{vocative}, {en}", True


def restore_honorifics(
    en: Optional[str],
    jp: Optional[str],
    allow_vocative_insertion: bool = False,
) -> Optional[str]:
    """Restore a source-confirmed honorific the EN output dropped.

    Runs LAST in the postedit chain (after ``strip_romaji_honorifics``), so a
    honorific the strip pass removed is put back here ONLY when the JP source
    demonstrably carries it — a hallucinated honorific with no source basis
    (the case ``strip_romaji_honorifics`` targets) is never resurrected because
    no candidate is found for it.

    ``allow_vocative_insertion`` (default False) gates TIER B: when the
    candidate name is entirely absent from ``en`` (not just missing its
    honorific), insert a "Name-suffix, " vocative prefix. This tier invents new
    text (a mechanically-romanized name may not match the exact spelling a
    human translator would choose) so it is OFF by default; TIER A (append a
    suffix to an already-present bare name) always runs and never invents a
    name.

    ``None``/empty ``en`` or missing ``jp`` pass through unchanged.
    """
    if not en or not jp:
        return en
    candidates = _collect_candidates(jp)
    if not candidates:
        return en
    out = en
    for name, suffix in candidates:
        out, changed = _tier_a_append(out, name, suffix)
        if changed:
            continue
        if allow_vocative_insertion:
            out, _ = _tier_b_prepend(out, name, suffix)
    return out

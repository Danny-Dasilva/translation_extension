"""L3 failure-mode probe suite.

Implements 8 rule-based checks over (jp, en_pred, probe_type, ref?) rows.

Probes:
    1. name        - Name preservation (JP surname -> romaji substring)
    2. honorific   - No -san/-kun/-chan/-sama/-senpai/-sensei in output
    3. curly       - No curly quotes / ellipsis chars
    4. repeat      - No 5-char substring repeating >=25x; no n-gram loops
    5. refusal     - No "I can't / inappropriate / as an AI" boilerplate
    6. length      - len(en_pred)/len(jp) in [0.3, 4.0]
    7. sfx         - On sfx rows, the gloss for the onomatopoeia appears in en
    8. idiom       - On idiom rows, at least one allowed English gloss appears

CLI:
    python -m backend.scripts.eval.probes \
        --predictions pred.jsonl \
        --probes probes.jsonl \
        --out probes_report.json \
        [--baseline prev_report.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

# 20 common Japanese surnames (kanji -> canonical romaji).
NAME_GAZETTEER: dict[str, str] = {
    "佐藤": "Sato",
    "鈴木": "Suzuki",
    "高橋": "Takahashi",
    "田中": "Tanaka",
    "伊藤": "Ito",
    "渡辺": "Watanabe",
    "山本": "Yamamoto",
    "中村": "Nakamura",
    "小林": "Kobayashi",
    "加藤": "Kato",
    "吉田": "Yoshida",
    "山田": "Yamada",
    "佐々木": "Sasaki",
    "山口": "Yamaguchi",
    "松本": "Matsumoto",
    "井上": "Inoue",
    "木村": "Kimura",
    "林": "Hayashi",
    "清水": "Shimizu",
    "斎藤": "Saito",
}

HONORIFIC_RE = re.compile(r"\b\w+-(san|kun|chan|sama|senpai|sensei)\b", re.IGNORECASE)

# Negative gazetteer of hallucinated proper names this title's pipeline has been
# observed to invent from garbled kana (see Ikenie4 judge synthesis).  Any of
# these appearing as a whole word in the output is a hard fail.
BANNED_INVENTED_NAMES: tuple[str, ...] = (
    "Lona",
    "Kinomiya",
    "Torachance",
    "Aki",
    "Zuri",
    "Nomi",
    "Saki",
    "Karu",
    "Beignet",
)
_BANNED_NAME_RE = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in BANNED_INVENTED_NAMES) + r")\b",
    re.IGNORECASE,
)

# Whole-word gendered pronoun detectors (word-boundary so "the" != "he").
_HE_RE = re.compile(r"\b(he|him|his)\b", re.IGNORECASE)
_SHE_RE = re.compile(r"\b(she|her|hers)\b", re.IGNORECASE)

# Gloss/explainer shapes the model leaks onto pages instead of translating a
# lone SFX/short token (see "Model meta-description / gloss leak" gap category).
_META_LEAK_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsfx for (a|an|the)\b", re.IGNORECASE),
    re.compile(r"\b(lighter|heavier) version of\b", re.IGNORECASE),
    re.compile(r"\byelled (by|with|when)\b", re.IGNORECASE),
    re.compile(r"\b(said|used|shouted) (by|with|when)\b", re.IGNORECASE),
    re.compile(r"\bsound effects?\b", re.IGNORECASE),
    re.compile(r"\bonomatopoe", re.IGNORECASE),
    re.compile(r"\bexpresses?\b.*\b(feeling|emotion|surprise)\b", re.IGNORECASE),
)

# JP number-words that must never be left romanized in the EN output.  Maps the
# JP source token -> the romaji forms that constitute a "left romanized" fail.
NUMBER_ROMAJI_MAP: dict[str, tuple[str, ...]] = {
    "ひゃく": ("hyaku",),
    "百": ("hyaku",),
    "いち": ("ichi",),
    "一": ("ichi",),
    "に": ("ni",),
    "さん": ("san",),  # note: also honorific; gated on JP number context below
    "よん": ("yon",),
    "ご": ("go",),
    "ろく": ("roku",),
    "なな": ("nana",),
    "はち": ("hachi",),
    "きゅう": ("kyuu", "kyu"),
    "じゅう": ("juu", "ju"),
    "せん": ("sen",),
    "まん": ("man",),
}
# Only treat the ambiguous mono-kana number words (に/ご/さん) as numbers when a
# counter/number marker co-occurs, to avoid false positives on real words.
_NUMBER_COUNTER_RE = re.compile(r"[周回発本個枚回\d０-９!！?？]")

CURLY_CHARS = set("‘’“”…")  # ' ' " " ...

REFUSAL_RE = re.compile(
    r"(I can't|I cannot|inappropriate|as an AI|I'm not able to|I won't)",
    re.IGNORECASE,
)

# Curated Japanese SFX -> English gloss (80 entries).  Keys cover both
# hiragana and katakana variants where meaningful.
SFX_GLOSSARY: dict[str, str] = {
    "どきどき": "thump",
    "ドキドキ": "thump",
    "ずっしり": "heavy",
    "ずきずき": "throb",
    "ズキズキ": "throb",
    "ぴかぴか": "sparkle",
    "ピカピカ": "sparkle",
    "ぎゅっ": "squeeze",
    "ギュッ": "squeeze",
    "ぱたぱた": "flutter",
    "パタパタ": "flutter",
    "ごろごろ": "rumble",
    "ゴロゴロ": "rumble",
    "さらさら": "smooth",
    "サラサラ": "smooth",
    "ふわふわ": "fluffy",
    "フワフワ": "fluffy",
    "くるくる": "spin",
    "クルクル": "spin",
    "ぽかぽか": "warm",
    "ポカポカ": "warm",
    "しくしく": "sob",
    "シクシク": "sob",
    "うとうと": "doze",
    "ウトウト": "doze",
    "そわそわ": "restless",
    "ソワソワ": "restless",
    "ぐらぐら": "shake",
    "グラグラ": "shake",
    "ぺこぺこ": "hungry",
    "ペコペコ": "hungry",
    "ぺろぺろ": "lick",
    "ペロペロ": "lick",
    "こそこそ": "sneak",
    "コソコソ": "sneak",
    "ぎらぎら": "glare",
    "ギラギラ": "glare",
    "ふらふら": "stagger",
    "フラフラ": "stagger",
    "ぶるぶる": "shiver",
    "ブルブル": "shiver",
    "にこにこ": "smile",
    "ニコニコ": "smile",
    "わくわく": "excited",
    "ワクワク": "excited",
    "きらきら": "sparkle",
    "キラキラ": "sparkle",
    "めらめら": "blaze",
    "メラメラ": "blaze",
    "ぽつぽつ": "drip",
    "ポツポツ": "drip",
    "ざあざあ": "pour",
    "ザアザア": "pour",
    "どしん": "thud",
    "ドシン": "thud",
    "ばたん": "slam",
    "バタン": "slam",
    "ぱっ": "flash",
    "パッ": "flash",
    "どかん": "boom",
    "ドカン": "boom",
    "ばきっ": "crack",
    "バキッ": "crack",
    "ぐしゃっ": "squish",
    "グシャッ": "squish",
    "ぽんっ": "pop",
    "ポンッ": "pop",
    "ちゅっ": "kiss",
    "チュッ": "kiss",
    "がたがた": "clatter",
    "ガタガタ": "clatter",
    "ざわざわ": "murmur",
    "ザワザワ": "murmur",
    "ひそひそ": "whisper",
    "ヒソヒソ": "whisper",
    "とんとん": "knock",
    "トントン": "knock",
    "ばくばく": "pound",
    "バクバク": "pound",
    "ぎりぎり": "just",
    "ギリギリ": "just",
    "もぐもぐ": "munch",
    "モグモグ": "munch",
    "ぺらぺら": "fluent",
    "ペラペラ": "fluent",
}

# 50 hand-picked idioms -> allowed English glosses.
IDIOM_GLOSSARY: dict[str, list[str]] = {
    "一石二鳥": ["two birds", "kill two birds"],
    "猿も木から落ちる": ["even monkeys fall", "anyone can make a mistake", "even experts"],
    "七転び八起き": ["fall seven", "get up eight", "perseverance", "bounce back"],
    "花より団子": ["dumplings over flowers", "substance over style", "pragmatism"],
    "覆水盆に返らず": ["spilt milk", "spilled milk", "cannot be undone"],
    "塵も積もれば山となる": ["many a little", "grains of sand", "add up"],
    "目には目を": ["eye for an eye"],
    "猫に小判": ["pearls before swine", "gold coins to a cat"],
    "馬の耳に念仏": ["preaching to deaf ears", "wasted on", "horse's ear"],
    "弘法にも筆の誤り": ["even homer nods", "even experts err"],
    "三度目の正直": ["third time's the charm", "third try"],
    "石の上にも三年": ["perseverance", "three years on a stone"],
    "井の中の蛙": ["frog in the well", "narrow view", "sheltered"],
    "論より証拠": ["proof over argument", "actions speak"],
    "急がば回れ": ["haste makes waste", "slow and steady", "more haste less speed"],
    "棚から牡丹餅": ["windfall", "bolt from the blue", "manna from heaven"],
    "鬼に金棒": ["iron club to an ogre", "stronger than ever", "unstoppable"],
    "蛇足": ["redundant", "snake legs", "superfluous"],
    "四面楚歌": ["surrounded by enemies", "under siege"],
    "朝三暮四": ["minor difference", "six of one"],
    "臥薪嘗胆": ["endure hardship", "bide one's time"],
    "一期一会": ["once in a lifetime", "treasure every meeting"],
    "温故知新": ["learn from the past", "review the old"],
    "以心伝心": ["heart to heart", "telepathy", "silent understanding"],
    "起死回生": ["comeback", "resurrection", "dramatic turnaround"],
    "危機一髪": ["close shave", "by a hair", "narrow escape"],
    "自業自得": ["reap what you sow", "self-inflicted", "own doing"],
    "十人十色": ["each to their own", "ten people ten colors", "different strokes"],
    "千差万別": ["infinite variety", "all kinds"],
    "前代未聞": ["unprecedented", "unheard of"],
    "大器晩成": ["late bloomer", "great talent matures slowly"],
    "天下無双": ["unrivaled", "peerless"],
    "波乱万丈": ["turbulent", "stormy", "full of ups and downs"],
    "百発百中": ["dead on", "bullseye", "always hits"],
    "不眠不休": ["without rest", "around the clock", "tireless"],
    "無我夢中": ["absorbed", "engrossed", "lost in"],
    "油断大敵": ["don't let your guard down", "carelessness is the enemy"],
    "弱肉強食": ["survival of the fittest", "law of the jungle"],
    "竜頭蛇尾": ["anti-climax", "strong start weak finish"],
    "和洋折衷": ["east meets west", "japanese-western fusion"],
    "因果応報": ["karma", "what goes around", "retribution"],
    "絵に描いた餅": ["pie in the sky", "pipe dream"],
    "口は災いの元": ["loose lips", "mouth is the source", "careful what you say"],
    "知らぬが仏": ["ignorance is bliss"],
    "好きこそ物の上手なれ": ["practice what you love", "passion breeds skill"],
    "住めば都": ["home is where you make it"],
    "泣き面に蜂": ["when it rains it pours", "adding insult to injury"],
    "二兎を追う者は一兎をも得ず": ["two hares", "fall between two stools"],
    "百聞は一見に如かず": ["seeing is believing", "picture worth a thousand"],
    "類は友を呼ぶ": ["birds of a feather"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def detect_high_repetition(text: str, *, substr_len: int = 5, min_repeats: int = 25) -> bool:
    """Detect pathological repetition loops (VNTL-style).

    Returns True if any substring of length ``substr_len`` repeats
    ``min_repeats`` or more times **consecutively**, OR if any single
    token appears ``min_repeats`` or more times in the output.
    """
    if not text:
        return False

    # Consecutive-substring check: slide a window and look for k*substr_len
    # stretches that are all equal.
    n = len(text)
    if n >= substr_len * min_repeats:
        for i in range(n - substr_len * min_repeats + 1):
            chunk = text[i : i + substr_len]
            ok = True
            for k in range(1, min_repeats):
                start = i + k * substr_len
                if text[start : start + substr_len] != chunk:
                    ok = False
                    break
            if ok:
                return True

    # Token-loop fallback: any single whitespace-separated token appearing
    # min_repeats+ times anywhere in the string.
    tokens = text.split()
    if tokens:
        counts = Counter(tokens)
        if counts.most_common(1)[0][1] >= min_repeats:
            return True

    return False


def _safe_len_ratio(jp: str, en_pred: str) -> float:
    if not jp:
        return float("inf") if en_pred else 1.0
    return len(en_pred) / len(jp)


# ---------------------------------------------------------------------------
# Per-probe pass/fail
# ---------------------------------------------------------------------------


def check_name(jp: str, en_pred: str, **_: Any) -> bool:
    """Pass if every surname appearing in jp also appears (romaji) in en."""
    for kanji, romaji in NAME_GAZETTEER.items():
        if kanji in jp:
            if romaji.lower() not in en_pred.lower():
                return False
    return True


def check_honorific(jp: str, en_pred: str, **_: Any) -> bool:
    """Pass if no ``word-san`` / ``-kun`` / etc. tokens in output."""
    return HONORIFIC_RE.search(en_pred) is None


def check_curly(jp: str, en_pred: str, **_: Any) -> bool:
    """Pass if en_pred contains zero curly quotes/ellipsis."""
    return not any(ch in CURLY_CHARS for ch in en_pred)


def check_repeat(jp: str, en_pred: str, **_: Any) -> bool:
    """Pass if no pathological repetition loop detected."""
    return not detect_high_repetition(en_pred)


def check_refusal(jp: str, en_pred: str, **_: Any) -> bool:
    """Pass if no refusal boilerplate in en_pred."""
    return REFUSAL_RE.search(en_pred) is None


def check_length(jp: str, en_pred: str, **_: Any) -> bool:
    ratio = _safe_len_ratio(jp, en_pred)
    return 0.3 <= ratio <= 4.0


def check_sfx(jp: str, en_pred: str, **_: Any) -> bool:
    """Pass if any expected gloss for a JP SFX in `jp` shows up in `en_pred`."""
    en_lower = en_pred.lower()
    matched_any = False
    for sfx, gloss in SFX_GLOSSARY.items():
        if sfx in jp:
            matched_any = True
            if gloss.lower() in en_lower:
                return True
    # If no known SFX found in jp, give a pass (nothing to score).
    return not matched_any


def check_idiom(jp: str, en_pred: str, **_: Any) -> bool:
    """Pass if any allowed English gloss for an idiom in `jp` is in `en_pred`."""
    en_lower = en_pred.lower()
    matched_any = False
    for idiom, glosses in IDIOM_GLOSSARY.items():
        if idiom in jp:
            matched_any = True
            for g in glosses:
                if g.lower() in en_lower:
                    return True
    return not matched_any


# ---------------------------------------------------------------------------
# Deterministic, seedless gold-set probes (Ikenie4 regression harness)
#
# These take per-row config carried on the probe/gold row (banned/required
# substrings, referent gender) so cases can be seeded from the gold set + the
# judge worst_issues without touching code.
# ---------------------------------------------------------------------------


def check_reverse_sense(
    jp: str,
    en_pred: str,
    *,
    banned_en_substrings: list[str] | None = None,
    required_en_substrings: list[str] | None = None,
    **_: Any,
) -> bool:
    """Negation / sense-reversal guard.

    Fails if ANY banned substring appears in the output (the reversed sense,
    e.g. 'spit' for 吸い出せ='suck out'), OR if NONE of the required substrings
    appears when a required list is given (the correct sense is missing).
    Substring match is case-insensitive.
    """
    en = en_pred.lower()
    for bad in banned_en_substrings or []:
        if bad and bad.lower() in en:
            return False
    req = [r for r in (required_en_substrings or []) if r]
    if req and not any(r.lower() in en for r in req):
        return False
    return True


def check_pronoun_gender(
    jp: str,
    en_pred: str,
    *,
    referent: str | None = None,
    **_: Any,
) -> bool:
    """Gendered-pronoun guard for subject-dropped JP.

    With ``referent='she'``: fail if a he/him/his pronoun is present and no
    she/her pronoun is present (wrong-gender inversion).  Symmetric for 'he'.
    If no gendered pronoun is present at all, there is nothing to get wrong ->
    pass.  Word-boundary matched so 'the' does not trip 'he'.
    """
    has_he = _HE_RE.search(en_pred) is not None
    has_she = _SHE_RE.search(en_pred) is not None
    ref = (referent or "").strip().lower()
    if ref in ("she", "her", "female", "f"):
        return not (has_he and not has_she)
    if ref in ("he", "him", "male", "m"):
        return not (has_she and not has_he)
    # Unknown referent -> nothing to enforce.
    return True


def check_name_invention(jp: str, en_pred: str, **_: Any) -> bool:
    """Negative-gazetteer guard: fail if a known hallucinated name appears."""
    return _BANNED_NAME_RE.search(en_pred) is None


def check_sfx_meta_leak(jp: str, en_pred: str, **_: Any) -> bool:
    """Fail if the output is a gloss/explainer about the token rather than a
    translation (the meta-description leak shapes)."""
    for rx in _META_LEAK_RES:
        if rx.search(en_pred):
            return False
    return True


def check_number_romaji(jp: str, en_pred: str, **_: Any) -> bool:
    """Fail if a JP number-word in the source is left romanized in the output.

    e.g. ひゃく -> 'H-hyaku' (fail) vs 'one hundred' (pass).  The ambiguous
    mono-kana readings (に/ご/さん) are only treated as numbers when a counter or
    number marker co-occurs in the JP, to avoid false positives.
    """
    en_low = en_pred.lower()
    ambiguous = {"に", "ご", "さん"}
    has_counter = _NUMBER_COUNTER_RE.search(jp) is not None
    for jp_num, romaji_forms in NUMBER_ROMAJI_MAP.items():
        if jp_num not in jp:
            continue
        if jp_num in ambiguous and not has_counter:
            continue
        for rom in romaji_forms:
            # Match the romaji as a token even when hyphen-stuttered, e.g.
            # "H-hyaku" -> contains "hyaku".
            if re.search(r"\b[a-z-]*" + re.escape(rom) + r"\b", en_low):
                return False
    return True


PROBE_DISPATCH: dict[str, Any] = {
    "name": check_name,
    "honorific": check_honorific,
    "curly": check_curly,
    "repeat": check_repeat,
    "refusal": check_refusal,
    "length": check_length,
    "sfx": check_sfx,
    "idiom": check_idiom,
    # Gold-set regression probes:
    "reverse_sense": check_reverse_sense,
    "pronoun_gender": check_pronoun_gender,
    "name_invention": check_name_invention,
    "sfx_meta_leak": check_sfx_meta_leak,
    "number_romaji": check_number_romaji,
}


# Targets expressed as a minimum pass-rate (probes whose spec is "<X%
# fail-rate" are converted to ">= (1 - X%) pass-rate").
PROBE_TARGETS: dict[str, float] = {
    "name": 1.00,
    "honorific": 0.98,
    "curly": 1.00,
    "repeat": 0.99,
    "refusal": 0.99,
    "length": 0.99,
    "sfx": 0.70,
    "idiom": 0.80,
    # Gold-set regression probes. These are seeded from KNOWN current failures,
    # so the baseline pass-rate may be low; the regression gate (no probe drops
    # vs the previous run) is what matters. Targets are set as "must not be
    # WORSE than current", i.e. the harness compares to a baseline report rather
    # than an absolute bar. Absolute targets here are conservative floors.
    "reverse_sense": 0.90,
    "pronoun_gender": 0.85,
    "name_invention": 1.00,
    "sfx_meta_leak": 1.00,
    "number_romaji": 0.90,
}

# Config keys that must be forwarded from a probe/gold row into the check fn.
_PROBE_CONFIG_KEYS: tuple[str, ...] = (
    "banned_en_substrings",
    "required_en_substrings",
    "referent",
)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    per_probe: dict[str, float] = field(default_factory=dict)
    per_probe_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    overall_pass: bool = False
    regressions_vs_baseline: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_probe": self.per_probe,
            "per_probe_counts": self.per_probe_counts,
            "overall_pass": self.overall_pass,
            "regressions_vs_baseline": self.regressions_vs_baseline,
        }


def run_probes(
    rows: list[dict[str, Any]],
    *,
    baseline: dict[str, float] | None = None,
    regression_threshold_pp: float = 5.0,
) -> ProbeResult:
    """Run all 8 probes over ``rows``.

    Each row must have ``jp``, ``en_pred``, and ``probe_type``.  Rows with
    an unknown ``probe_type`` are skipped.
    """
    by_probe: dict[str, list[bool]] = {k: [] for k in PROBE_DISPATCH}

    for row in rows:
        probe_type = row.get("probe_type")
        if probe_type not in PROBE_DISPATCH:
            continue
        fn = PROBE_DISPATCH[probe_type]
        cfg = {k: row[k] for k in _PROBE_CONFIG_KEYS if k in row}
        passed = bool(
            fn(jp=row.get("jp", ""), en_pred=row.get("en_pred", ""), **cfg)
        )
        by_probe[probe_type].append(passed)

    per_probe: dict[str, float] = {}
    per_probe_counts: dict[str, dict[str, int]] = {}
    for probe, outcomes in by_probe.items():
        if not outcomes:
            per_probe[probe] = float("nan")
            per_probe_counts[probe] = {"n": 0, "pass": 0, "fail": 0}
            continue
        rate = sum(outcomes) / len(outcomes)
        per_probe[probe] = rate
        per_probe_counts[probe] = {
            "n": len(outcomes),
            "pass": sum(outcomes),
            "fail": len(outcomes) - sum(outcomes),
        }

    # Overall pass = every probe with samples meets its target.
    overall_pass = True
    for probe, rate in per_probe.items():
        if per_probe_counts[probe]["n"] == 0:
            continue
        if rate < PROBE_TARGETS[probe]:
            overall_pass = False
            break

    regressions: dict[str, float] = {}
    if baseline:
        for probe, rate in per_probe.items():
            if probe in baseline and per_probe_counts[probe]["n"] > 0:
                delta = (rate - baseline[probe]) * 100.0  # percentage-point delta
                if delta < -regression_threshold_pp:
                    regressions[probe] = delta

    return ProbeResult(
        per_probe=per_probe,
        per_probe_counts=per_probe_counts,
        overall_pass=overall_pass,
        regressions_vs_baseline=regressions,
    )


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _join_predictions(
    predictions: list[dict[str, Any]], probes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Align prediction rows with probe rows.

    Two supported shapes:

    * ``predictions`` already carries ``probe_type`` (predictions written by
      the generator directly from the probe file) -> return as-is.
    * Otherwise zip index-wise against ``probes``; warn on length mismatch.
    """
    if predictions and "probe_type" in predictions[0]:
        return predictions

    if len(predictions) != len(probes):
        logger.warning(
            "Length mismatch: predictions={}, probes={}.  Truncating to min.",
            len(predictions),
            len(probes),
        )
    n = min(len(predictions), len(probes))
    joined: list[dict[str, Any]] = []
    for i in range(n):
        merged = dict(probes[i])
        # Prediction row supplies en_pred; probe row supplies jp + probe_type.
        merged["en_pred"] = predictions[i].get("en_pred", predictions[i].get("en", ""))
        joined.append(merged)
    return joined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the 8-probe failure-mode suite and emit a JSON report.",
    )
    p.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="JSONL with at least {jp, en_pred}; probe_type optional if --probes provided.",
    )
    p.add_argument(
        "--probes",
        type=Path,
        default=None,
        help="Probe fixture JSONL (with probe_type).  Used if predictions lack probe_type.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output JSON report path.")
    p.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Prior report JSON; regressions >5pp will be flagged.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    predictions = _read_jsonl(args.predictions)
    probes = _read_jsonl(args.probes) if args.probes else []

    rows = _join_predictions(predictions, probes) if probes else predictions

    baseline_rates: dict[str, float] | None = None
    if args.baseline and args.baseline.exists():
        with args.baseline.open("r", encoding="utf-8") as fh:
            baseline_json = json.load(fh)
        baseline_rates = baseline_json.get("per_probe")

    result = run_probes(rows, baseline=baseline_rates)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)

    logger.info("Wrote probe report to {}", args.out)
    logger.info("overall_pass={}", result.overall_pass)
    for k, v in result.per_probe.items():
        logger.info("  {}: {:.4f} (n={})", k, v, result.per_probe_counts[k]["n"])
    return 0 if result.overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())

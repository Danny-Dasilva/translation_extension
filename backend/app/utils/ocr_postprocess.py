"""OCR postprocessing normalizer for PARSeq manga output.

Ported verbatim from comic-text-detector/scripts/postprocess_ocr.py. Reduces CER
by cleaning up common NAR decoding artifacts (trailing repeats) and normalizing
Unicode representations (smart quotes, dashes, fullwidth variants) that vary
between ground truth and predictions.

Public entrypoint:
    apply_all(text) -> str

Note: these are pure string operations, so they cannot be baked into the ONNX
graph. Apply at the Python boundary immediately after argmax decoding.
"""

import re
import unicodedata


# ---------------------------------------------------------------------------
# NAR repeat-artifact collapse (B-export specific)
# ---------------------------------------------------------------------------
#
# The non-autoregressive PARSeq export hallucinates repeat artifacts that the
# autoregressive decode does not. They fall into a few signatures, each handled
# below with thresholds tuned against the Manga109-s single-line GT pool
# (56,323 rows) so the collapse is a near no-op on clean text and preserves
# legitimate Japanese repeats (laughter ハハハ/あはは, elongated vowels わ～～～,
# ellipsis ・・・・/……, emphatic !! / ?? / ！？, doubled ー).
#
# Empirical facts that drive the thresholds (measured on the GT pool):
#   * Legit non-punct same-char runs of length >=5 are common (584 rows):
#     laughter and vowel elongation. So NEVER cap arbitrary same-char runs.
#   * A CJK *ideograph* repeated even 3x never occurs in legit GT (0 rows).
#     So a CJK run >= 3 is a safe artifact signal ('体体体体体' -> '体体').
#   * A *trailing punctuation block* in which some punct char appears in 2+
#     separate runs (a "loop": e.g. '...。..', '!.!.!', '!!!ー!!') occurs in
#     only 2 / 56,323 GT rows (0.004%). So it is a safe artifact signal; the
#     non-looping mixed tails ('・・・・！？', 'ーー！！', '…!?') are left intact.

# Punctuation / symbol characters that the NAR decode loops on at end-of-line.
_TRAIL_PUNCT_SET = set("!?！？.。．・…ー―—-~〜、，,")
# Period-class characters. A *period* recurring across the trailing block is the
# strongest NAR-loop signature ('...。..', '!.....!...'); legit emphatic tails
# ('!!??', '・・・・！？', interrobang ligatures) never interleave a period.
_PERIOD_SET = set(".。．")

# CJK ideograph ranges (Unified + Extension-A). Repeated >=3 is never legit.
def _is_cjk_ideograph(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF)


def collapse_cjk_runs(text: str, max_run: int = 2) -> str:
    """Cap runs of an identical CJK ideograph to ``max_run`` characters.

    A CJK ideograph repeated >=3 times never appears in the legit GT pool, so
    this only fires on NAR artifacts like '体体体体体'. Kana/symbol runs (which
    DO occur legitimately as laughter / elongation) are left untouched.
    """
    if len(text) < max_run + 1:
        return text
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        j = i
        while j < n and text[j] == ch:
            j += 1
        run_len = j - i
        if _is_cjk_ideograph(ch) and run_len > max_run:
            out.append(ch * max_run)
        else:
            out.append(ch * run_len)
        i = j
    return "".join(out)


def collapse_trailing_loop(text: str, min_tail: int = 4) -> str:
    """Trim a looping trailing punctuation block produced by NAR decode.

    The NAR export fills the tail with interleaved punctuation runs, e.g.
    '...。..', '!.....!...', 'わーっ...。..'. The signature is: within the maximal
    trailing run of punctuation/symbol characters, some character appears in 2+
    separate (non-adjacent) runs. To avoid trimming legitimate emphatic tails
    that happen to repeat a char ('!!??', interrobang ligatures '!!???!!?',
    'ええーー!?!!'), the loop only counts as an artifact when EITHER:
      * a *period-class* char ('.', '。', '．') is the looped char -- the dot-fill
        signature that never appears in legit emphatic tails; OR
      * the tail is long (>= 8) AND some char spans >= 3 separate runs -- a
        runaway loop too long to be intentional emphasis.
    When detected, the entire trailing punctuation block is replaced by its FIRST
    contiguous run (the emphasis closest to the real content), discarding the
    looped fill while keeping a plausible terminal mark.

    Non-looping mixed tails ('・・・・！？', 'ーー！！', '…!?', '!!??') and bounded
    emphatic loops without a period ('ええーー!?!!') are preserved verbatim. Tuned
    against the GT pool: this fires on only 1 / 56,323 legit rows (0.0018%).
    """
    if len(text) < min_tail:
        return text
    i = len(text)
    while i > 0 and text[i - 1] in _TRAIL_PUNCT_SET:
        i -= 1
    prefix, tail = text[:i], text[i:]
    if len(tail) < min_tail:
        return text
    # Build contiguous runs within the tail.
    runs: list[list] = []
    for ch in tail:
        if runs and runs[-1][0] == ch:
            runs[-1][1] += 1
        else:
            runs.append([ch, 1])
    # Count how many separate runs each char spans.
    run_counts: dict[str, int] = {}
    for ch, _ in runs:
        run_counts[ch] = run_counts.get(ch, 0) + 1
    looped = {ch for ch, c in run_counts.items() if c >= 2}
    if not looped:
        return text
    period_loop = bool(looped & _PERIOD_SET)
    runaway = len(tail) >= 8 and any(c >= 3 for c in run_counts.values())
    if not (period_loop or runaway):
        return text
    first_ch, first_len = runs[0]
    return prefix + first_ch * first_len


# ---------------------------------------------------------------------------
# Character repeat stripping
# ---------------------------------------------------------------------------

def strip_trailing_repeats(text: str, max_trailing: int = 2,
                           max_mid: int = 3) -> str:
    """Remove excessive repeated characters from NAR decoding artifacts.

    PARSeq in non-autoregressive mode sometimes generates trailing repeated
    characters. Caps trailing runs to max_trailing (default 2) and mid-text
    runs to max_mid (default 3).
    """
    if len(text) <= 1:
        return text

    # First pass: cap mid-text runs to max_mid.
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        run_start = i
        while i < len(text) and text[i] == ch:
            i += 1
        run_len = i - run_start
        capped = min(run_len, max_mid)
        result.append(ch * capped)

    intermediate = "".join(result)

    # Second pass: cap trailing run to max_trailing.
    if len(intermediate) <= max_trailing:
        return intermediate

    last_char = intermediate[-1]
    trailing_count = 0
    for c in reversed(intermediate):
        if c == last_char:
            trailing_count += 1
        else:
            break

    if trailing_count > max_trailing:
        non_trailing = intermediate[:len(intermediate) - trailing_count]
        intermediate = non_trailing + last_char * max_trailing

    return intermediate


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------

# Zero-width characters to strip
_ZERO_WIDTH = re.compile(
    "[​‌‍‎‏﻿⁠⁡⁢⁣⁤]"
)

# Punctuation normalization mappings
_PUNCT_MAP = {
    # Smart/curly quotes -> standard ASCII
    "“": '"',     # LEFT DOUBLE QUOTATION MARK
    "”": '"',     # RIGHT DOUBLE QUOTATION MARK
    "„": '"',     # DOUBLE LOW-9 QUOTATION MARK
    "‘": "'",     # LEFT SINGLE QUOTATION MARK
    "’": "'",     # RIGHT SINGLE QUOTATION MARK
    "‚": "'",     # SINGLE LOW-9 QUOTATION MARK
    # Dashes -> prolonged sound mark (ー U+30FC) for Japanese context
    "–": "ー",  # EN DASH
    "—": "ー",  # EM DASH
    "―": "ー",  # HORIZONTAL BAR
    # Ellipsis
    "…": "...",   # HORIZONTAL ELLIPSIS -> three dots
    # Fullwidth exclamation/question -> halfwidth ASCII.
    "！": "!",     # FULLWIDTH EXCLAMATION MARK
    "？": "?",     # FULLWIDTH QUESTION MARK
    # Ligature punctuation -> two ASCII chars.
    "‼": "!!",    # DOUBLE EXCLAMATION MARK
    "⁇": "??",    # DOUBLE QUESTION MARK
    "⁈": "?!",    # QUESTION EXCLAMATION MARK
    "⁉": "!?",    # EXCLAMATION QUESTION MARK
    # Wave dash / fullwidth tilde unification
    "～": "〜",  # FULLWIDTH TILDE -> WAVE DASH
    # Fullwidth space
    "　": " ",     # IDEOGRAPHIC SPACE -> ASCII space
}

_PUNCT_RE = re.compile("|".join(re.escape(k) for k in _PUNCT_MAP))

# Runs of 3+ KATAKANA MIDDLE DOT (U+30FB) -> ASCII "..."
_MIDDLE_DOT_RUN_RE = re.compile(r"・{3,}")


def _fullwidth_to_halfwidth(text: str) -> str:
    """Convert fullwidth alphanumeric characters to halfwidth.

    FF21-FF3A (A-Z), FF41-FF5A (a-z), FF10-FF19 (0-9).
    Does NOT convert fullwidth Japanese punctuation.
    """
    result = []
    for ch in text:
        cp = ord(ch)
        if 0xFF21 <= cp <= 0xFF3A:  # A-Z
            result.append(chr(cp - 0xFF21 + ord('A')))
        elif 0xFF41 <= cp <= 0xFF5A:  # a-z
            result.append(chr(cp - 0xFF41 + ord('a')))
        elif 0xFF10 <= cp <= 0xFF19:  # 0-9
            result.append(chr(cp - 0xFF10 + ord('0')))
        else:
            result.append(ch)
    return "".join(result)


def _halfwidth_katakana_to_fullwidth(text: str) -> str:
    """Convert halfwidth katakana (FF65-FF9F) to fullwidth via NFKC."""
    result = []
    for ch in text:
        cp = ord(ch)
        if 0xFF65 <= cp <= 0xFF9F:
            result.append(unicodedata.normalize("NFKC", ch))
        else:
            result.append(ch)
    return "".join(result)


def normalize_text(text: str) -> str:
    """Normalize Unicode text for consistent OCR comparison.

    Steps (in order):
      1. Unicode NFC normalization
      2. Strip zero-width characters
      3. Fullwidth alphanumeric -> halfwidth (A-Z, a-z, 0-9)
      4. Halfwidth katakana -> fullwidth
      5. Standardize punctuation variants
      6. Collapse >=3 middle-dot runs to "..."
    """
    if not text:
        return text

    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _fullwidth_to_halfwidth(text)
    text = _halfwidth_katakana_to_fullwidth(text)
    text = _PUNCT_RE.sub(lambda m: _PUNCT_MAP[m.group()], text)
    text = _MIDDLE_DOT_RUN_RE.sub("...", text)
    return text


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def apply_all(text: str) -> str:
    """Apply all post-processing steps in the correct order.

    Order:
      1. normalize_text                  (canonicalize so comparisons are valid)
      2. collapse_trailing_loop          (kill NAR looping punct tails)
      3. collapse_cjk_runs               (cap artifact CJK ideograph runs)
      4. strip_trailing_repeats          (legacy same-char trailing/mid cap)

    Normalization runs first so character comparisons in the collapse steps work
    on canonical forms. The two collapse steps are tuned to be near no-ops on the
    legit GT distribution (see module docstring), so they do not regress the
    autoregressive model that shares this pipeline.
    """
    text = normalize_text(text)
    text = collapse_trailing_loop(text)
    text = collapse_cjk_runs(text)
    text = strip_trailing_repeats(text)
    return text

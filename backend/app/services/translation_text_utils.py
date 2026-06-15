"""Pure text helpers for manga translation (llama-cpp-free).

These helpers were originally defined in ``local_translation_service.py``
alongside the (now removed) llama-cpp execution backend. They are pure
string utilities — no model dependency — shared by every translation
backend (vLLM, transformers) and the eval/visualization scripts.

Ported from:
  /tmp/koharu/koharu-llm/src/prompt.rs:50-57    (system prompt)
  /tmp/koharu/koharu-app/src/llm.rs:439-538     (format/parse/strip helpers)
"""

import re
from typing import List, Optional


BATCHED_SYSTEM_PROMPT = (
    "You are a professional manga translator. Output ONLY {target}.\n"
    "\n"
    "Input: numbered Japanese blocks like `[N]text`.\n"
    "Output: same tags with {target} translations, one per line, nothing else.\n"
    "Output EXACTLY as many lines as the input has blocks — one `[N]` line per "
    "input block, same count, same order. No preamble line, no blank lines, no "
    "trailing notes.\n"
    "\n"
    "Strict rules:\n"
    "- Language: every line MUST be in {target}. Never output Japanese "
    "characters. Never romanize (no `dokidoki`, `kawaii`, `san`, `chan`, "
    "name suffixes, etc.). Honorifics become natural {target} forms or "
    "are dropped.\n"
    "- Length: ≤12 words per line; hard cap 25. Sound spoken, not written.\n"
    "- Punctuation: ASCII only. Use `. , ! ? - ... \" '`. Never use `… — "
    "– “ ” ‘ ’ 〜 ～ ・ 「 」 『 』 。 、` or any full-width "
    "character. Three ASCII dots for trailing-off speech.\n"
    "- Unreadable OCR: if a block is garbled and has no clear meaning, "
    "output exactly `[N] ...` — do NOT guess, transliterate, or invent.\n"
    "- SFX: translate the sound to {target} (e.g. `ドキドキ`->`thump thump`, "
    "`ガシャン`->`crash`, `ザアザア`->`shhh`). Keep them short.\n"
    "- Preserve voice, emotion, relationship, and emphasis.\n"
    "- Never merge, split, reorder, renumber, or skip blocks. Keep every "
    "`[N]` tag exactly.\n"
    "- No preamble, no notes, no explanations, no markdown, no quotes "
    "wrapping the line.\n"
    "\n"
    "Example (target=English):\n"
    "Input:\n"
    "[1]ありがとう\n"
    "[2]ドキドキ\n"
    "[3]えええ？！\n"
    "[4]わ・けー・うむぬ・あは\n"
    "[5]「助けて…」\n"
    "Output:\n"
    "[1] Thanks.\n"
    "[2] thump thump\n"
    "[3] Huh?!\n"
    "[4] ...\n"
    "[5] \"Help me...\""
)


# LIGHT page-level system prompt (A/B candidate). Deliberately minimal — the
# v10it fine-tune COLLAPSES on the heavy few-shot BATCHED_SYSTEM_PROMPT above,
# so this is 4 short sentences: genre framing + the self-reference fix + the
# count/format lock. Gated behind settings.translation_system_prompt_enabled;
# the A/B (Part13_sysprompt_off vs _on) decides whether to default it on.
# Use `.format(target=...)` to fill the target language.
LIGHT_SYSTEM_PROMPT = (
    "You are translating intimate adult manga dialogue into {target}. "
    "When a speaker refers to themselves in the third person (for example the "
    "mother saying お母さん or 母さん about herself), translate it as "
    "\"Mommy\" or \"I\", never \"my mom\". "
    "Output exactly the same number of numbered lines as the input, one "
    "translation per line, in order, nothing else. "
    "Every line must be in natural {target}."
)


def format_sources(texts: List[str]) -> str:
    """Format a list of source strings into koharu's tagged-block body.

    Port of `format_sources` at `/tmp/koharu/koharu-app/src/llm.rs:439-446`.
    Produces: `[1]text1\\n[2]text2\\n...[N]textN`.
    """
    return "\n".join(f"[{i + 1}]{text}" for i, text in enumerate(texts))


def strip_thinking_block(text: str) -> str:
    """Remove any ``<think>...</think>`` wrapper from model output.

    Port of `strip_thinking_block` at `/tmp/koharu/koharu-app/src/llm.rs:517-524`.
    """
    start = text.find("<think>")
    if start == -1:
        return text
    end_rel = text[start:].find("</think>")
    if end_rel == -1:
        return text
    return text[start + end_rel + len("</think>"):].lstrip()


def strip_wrapping_quotes(text: str) -> str:
    """Strip matching single or double quotes wrapping a string.

    Port of `strip_wrapping_quotes` at `/tmp/koharu/koharu-app/src/llm.rs:526-538`.
    """
    trimmed = text.strip()
    if len(trimmed) >= 2:
        first = trimmed[0]
        last = trimmed[-1]
        if (first == '"' and last == '"') or (first == "'" and last == "'"):
            return trimmed[1:-1]
    return trimmed


_TAG_RE = re.compile(r"\[(\d+)\]\s*([^\[]*)")


def parse_tagged_blocks(output: str, n: int) -> Optional[List[str]]:
    """Parse tagged-block translation output into an ordered list of length n.

    Port of `parse_tagged_blocks` at `/tmp/koharu/koharu-app/src/llm.rs:483-503`.
    Returns None if no tags were found (caller should fall back to legacy split).
    """
    matches = _TAG_RE.findall(output)
    if not matches:
        return None
    blocks = [""] * n
    for num_str, content in matches:
        try:
            idx_1based = int(num_str)
        except ValueError:
            continue
        if idx_1based <= 0:
            continue
        idx = idx_1based - 1
        if idx < n:
            blocks[idx] = content.strip()
    return blocks


def split_legacy_lines(output: str, n: int) -> List[str]:
    """Legacy fallback: split by newlines and pad/truncate to length n.

    Port of `split_legacy_lines` at `/tmp/koharu/koharu-app/src/llm.rs:505-515`.
    """
    lines = [line.rstrip("\r") for line in output.splitlines()]
    if len(lines) > n:
        lines = lines[:n]
    while len(lines) < n:
        lines.append("")
    return lines


def clean_translation_output(translation: str) -> str:
    """Clean up translation output by removing model artifacts.

    Removes "Assistant:" prefix and special end tokens that may leak through.

    Args:
        translation: Raw translation text from model

    Returns:
        Cleaned translation text
    """
    translation = translation.strip()

    # Remove "Assistant:" prefix if present (model chat template artifact)
    if translation.startswith("Assistant:"):
        translation = translation[len("Assistant:"):].strip()

    # Strip any special tokens that may have leaked through
    # Use regex to catch all variants (e.g. <|im_end|>, <|im_end+], <|im_end/>, etc.)
    translation = re.sub(r'<\|im_\w*[^>]*[>\]|/]+', '', translation)
    for token in ["</s>", "<|eot_id|>"]:
        translation = translation.replace(token, "")

    translation = translation.strip()

    # Non-Latin garble guard: the small model occasionally falls out of the
    # target language (a Russian/Cyrillic leak was observed) or emits CJK /
    # replacement-box characters. Rendering that to the page is worse than a
    # silent gap, so replace clearly-off-target output with an ellipsis.
    if translation and _is_garbled(translation):
        return "..."

    return translation


# Character-class guards for _is_garbled.
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
# CJK ideographs, hiragana, katakana, full-width forms, Hangul, plus the
# Unicode replacement char and the literal "tofu" box.
_CJK_OR_BOX_RE = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿가-힯＀-￯�□]"
)


def _is_garbled(text: str) -> bool:
    """True if a translated line fell out of the (Latin) target language.

    Heuristics:
    - any CJK / Hangul / full-width / replacement-box / tofu char present, OR
    - more than 30% of the letters are Cyrillic (a non-English leak).
    """
    if _CJK_OR_BOX_RE.search(text):
        return True
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    cyr = sum(1 for c in letters if _CYRILLIC_RE.match(c))
    return cyr / len(letters) > 0.30

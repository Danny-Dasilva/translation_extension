"""
Japanese text detection utility for post-OCR filtering.

MangaOCR (trained on Japanese) hallucinates Japanese-looking characters
when processing English text. This utility detects such cases by analyzing
character composition and patterns.
"""
import logging
from typing import List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Unicode ranges for Japanese characters
HIRAGANA_RANGE = (0x3040, 0x309F)
KATAKANA_RANGE = (0x30A0, 0x30FF)
KATAKANA_EXT_RANGE = (0x31F0, 0x31FF)
KANJI_RANGE = (0x4E00, 0x9FFF)
KANJI_EXT_RANGE = (0x3400, 0x4DBF)
HALFWIDTH_KATAKANA = (0xFF65, 0xFF9F)

# Japanese punctuation
JP_PUNCTUATION = set('。、！？「」『』（）・ー〜…')


@dataclass
class CharacterAnalysis:
    """Analysis of character composition in text."""
    total: int
    hiragana: int
    katakana: int
    kanji: int
    ascii_alpha: int
    punctuation: int
    other: int

    @property
    def japanese_count(self) -> int:
        """Total Japanese characters (hiragana + katakana + kanji)."""
        return self.hiragana + self.katakana + self.kanji

    @property
    def japanese_ratio(self) -> float:
        """Ratio of Japanese characters to total meaningful characters."""
        meaningful = self.total - self.punctuation
        if meaningful == 0:
            return 0.0
        return self.japanese_count / meaningful

    @property
    def katakana_ratio(self) -> float:
        """Ratio of katakana to total Japanese characters."""
        if self.japanese_count == 0:
            return 0.0
        return self.katakana / self.japanese_count


def _in_range(char: str, range_tuple: Tuple[int, int]) -> bool:
    """Check if character codepoint is in range."""
    code = ord(char)
    return range_tuple[0] <= code <= range_tuple[1]


def analyze_characters(text: str) -> CharacterAnalysis:
    """
    Analyze character composition of text.

    Args:
        text: Input text to analyze

    Returns:
        CharacterAnalysis with counts for each character type
    """
    hiragana = katakana = kanji = ascii_alpha = punctuation = other = 0

    for char in text:
        if _in_range(char, HIRAGANA_RANGE):
            hiragana += 1
        elif (_in_range(char, KATAKANA_RANGE) or
              _in_range(char, KATAKANA_EXT_RANGE) or
              _in_range(char, HALFWIDTH_KATAKANA)):
            katakana += 1
        elif _in_range(char, KANJI_RANGE) or _in_range(char, KANJI_EXT_RANGE):
            kanji += 1
        elif char.isalpha() and ord(char) < 128:
            ascii_alpha += 1
        elif char in JP_PUNCTUATION or not char.isalnum():
            punctuation += 1
        else:
            other += 1

    return CharacterAnalysis(
        total=len(text),
        hiragana=hiragana,
        katakana=katakana,
        kanji=kanji,
        ascii_alpha=ascii_alpha,
        punctuation=punctuation,
        other=other
    )


def _is_repeated_pattern(text: str) -> bool:
    """
    Check if text is a repeated pattern (common in manga SFX).

    Examples: ゴゴゴ, ドドド, ザザザ, バババ, ドンドンドン
    """
    clean = ''.join(c for c in text if c not in JP_PUNCTUATION and not c.isspace())

    if len(clean) < 2:
        return False

    # Check for simple repetition (all same character)
    if len(set(clean)) == 1:
        return True

    # Check for 2-char pattern repetition (e.g., ドンドンドン)
    if len(clean) >= 4 and len(clean) % 2 == 0:
        pattern = clean[:2]
        if all(clean[i:i+2] == pattern for i in range(0, len(clean), 2)):
            return True

    return False


def is_japanese_text(
    text: str,
    min_japanese_ratio: float = 0.5,
    katakana_only_max_length: int = 6,
    min_length: int = 1
) -> bool:
    """
    Determine if text is valid Japanese content.

    Heuristics:
    1. Text must contain sufficient Japanese characters
    2. All-katakana text is suspicious for longer strings (hallucinated Latin)
    3. Very short katakana-only is OK (SFX like "ドン", "バン")
    4. Presence of hiragana/kanji strongly indicates real Japanese
    5. Repeated patterns are valid SFX even if long (ゴゴゴゴゴ)

    Args:
        text: OCR output text to validate
        min_japanese_ratio: Minimum ratio of Japanese chars (default 0.5)
        katakana_only_max_length: Max length for katakana-only text (default 6)
        min_length: Minimum text length to consider (default 1)

    Returns:
        True if text appears to be valid Japanese content
    """
    text = text.strip()
    if len(text) < min_length:
        return False

    analysis = analyze_characters(text)

    # No Japanese characters at all
    if analysis.japanese_count == 0:
        return False

    # Check Japanese ratio threshold
    if analysis.japanese_ratio < min_japanese_ratio:
        return False

    # All-katakana heuristic: suspicious for longer text
    # MangaOCR often outputs katakana for Latin letters (visual similarity)
    # But short katakana is common in SFX: ドン!, バン!, ゴゴゴ
    if analysis.katakana_ratio == 1.0:
        if analysis.katakana > katakana_only_max_length:
            # Long katakana-only is suspicious
            # Exception: repeated katakana patterns are valid SFX
            if not _is_repeated_pattern(text):
                return False

    return True


def filter_japanese_texts(
    ocr_texts: List[str],
    min_japanese_ratio: float = 0.5,
    katakana_only_max_length: int = 6
) -> List[int]:
    """
    Filter OCR results and return indices of valid Japanese texts.

    Args:
        ocr_texts: List of OCR output strings
        min_japanese_ratio: Minimum ratio of Japanese chars
        katakana_only_max_length: Max length for katakana-only text

    Returns:
        List of indices that contain valid Japanese text
    """
    valid_indices = []
    for i, text in enumerate(ocr_texts):
        if is_japanese_text(text, min_japanese_ratio, katakana_only_max_length):
            valid_indices.append(i)
        else:
            logger.debug(f"Filtered non-Japanese text at index {i}: '{text[:30]}...'")
    return valid_indices

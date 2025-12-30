"""Translation service using Google Gemini API"""
import logging
from typing import List
import google.generativeai as genai
from app.config import settings

logger = logging.getLogger(__name__)


class TranslationService:
    """Service for translating text using Google Gemini"""
    
    def __init__(self):
        """Initialize Gemini API"""
        try:
            genai.configure(api_key=settings.gemini_api_key)
            # Configure model with temperature for more deterministic output
            self.model = genai.GenerativeModel(
                settings.default_model,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more consistent translation
                    top_p=0.95,
                    top_k=40,
                )
            )
            logger.info(f"Gemini API initialized with model: {settings.default_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            raise
    
    async def translate_batch(
        self, 
        texts: List[str], 
        target_lang: str = "English"
    ) -> List[str]:
        """
        Translate a batch of Japanese texts to target language
        
        Args:
            texts: List of Japanese text strings to translate
            target_lang: Target language (default: English)
        
        Returns:
            List of translated strings in same order as input
        """
        if not texts:
            return []
        
        try:
            # Build prompt optimized for manga dialogue
            prompt = self._build_translation_prompt(texts, target_lang)

            logger.info(f"Sending {len(texts)} texts to Gemini for translation to {target_lang}")
            logger.debug(f"First text to translate: '{texts[0][:50]}...'")
            logger.debug(f"Full prompt being sent:\n{prompt[:500]}...")

            # Generate translation
            response = self.model.generate_content(prompt)

            logger.info(f"Received response from Gemini (length: {len(response.text)} chars)")
            logger.info(f"Gemini raw response:\n{response.text}")

            # Parse response
            translations = self._parse_response(response.text, len(texts))

            # Validate translation actually happened
            if translations and len(translations) > 0 and texts and len(texts) > 0:
                if translations[0] == texts[0]:
                    logger.error(f"❌ Translation FAILED: Output matches input!")
                    logger.error(f"Original Japanese: '{texts[0]}'")
                    logger.error(f"'Translated' text: '{translations[0]}'")
                    logger.error(f"This means Gemini returned Japanese instead of {target_lang}")
                else:
                    # Check if translation contains Japanese characters (Hiragana, Katakana, Kanji)
                    has_japanese = any(
                        '\u3040' <= char <= '\u309F' or  # Hiragana
                        '\u30A0' <= char <= '\u30FF' or  # Katakana
                        '\u4E00' <= char <= '\u9FFF'     # Kanji
                        for char in translations[0]
                    )

                    if has_japanese:
                        logger.warning(f"⚠️  Translation contains Japanese characters!")
                        logger.warning(f"Original: '{texts[0]}'")
                        logger.warning(f"Translated: '{translations[0]}'")
                    else:
                        logger.info(f"✓ Translation successful!")
                        logger.info(f"Original JP: '{texts[0]}'")
                        logger.info(f"Translated {target_lang}: '{translations[0]}'")

            logger.info(f"Successfully parsed {len(translations)} translations")
            return translations

        except Exception as e:
            logger.error(f"Translation failed with exception: {e}", exc_info=True)
            logger.error(f"Returning original texts as fallback")
            # Return original texts as fallback
            return texts
    
    def _build_translation_prompt(self, texts: List[str], target_lang: str) -> str:
        """
        Build optimized translation prompt for manga dialogue

        Args:
            texts: List of texts to translate
            target_lang: Target language

        Returns:
            Formatted prompt string
        """
        # Number each text for structured output
        numbered_texts = "\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))

        prompt = f"""You are a professional Japanese-to-{target_lang} translator specializing in manga dialogue.

IMPORTANT: Translate FROM Japanese TO {target_lang}. Do NOT return the original Japanese text.

Instructions:
- Translate each Japanese text into natural {target_lang}
- Preserve the tone, casualness, and emotion of the original text
- Maintain character voice and speaking style
- Handle onomatopoeia appropriately (translate or adapt as needed)
- Keep translations natural and readable
- Return ONLY the {target_lang} translations, one per line, numbered
- Do NOT include the original Japanese text
- Do NOT include explanations or notes
- Each translation MUST be in {target_lang}, not Japanese

Japanese texts to translate:
{numbered_texts}

{target_lang} translations:"""

        return prompt
    
    def _parse_response(self, response_text: str, expected_count: int) -> List[str]:
        """
        Parse Gemini response into list of translations
        
        Args:
            response_text: Raw response from Gemini
            expected_count: Expected number of translations
        
        Returns:
            List of translated strings
        """
        # Split by newlines
        lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
        
        # Remove numbering if present (e.g., "1. Translation" -> "Translation")
        translations = []
        for line in lines:
            # Try to remove leading number and period/dot
            if '. ' in line and line.split('. ', 1)[0].isdigit():
                translation = line.split('. ', 1)[1]
            else:
                translation = line
            
            translations.append(translation)
        
        # Ensure we have the right number of translations
        if len(translations) != expected_count:
            logger.warning(
                f"Translation count mismatch: expected {expected_count}, "
                f"got {len(translations)}"
            )
            # Pad with empty strings if needed
            while len(translations) < expected_count:
                translations.append("")
        
        return translations[:expected_count]

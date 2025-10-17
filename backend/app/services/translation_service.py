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
            self.model = genai.GenerativeModel(settings.default_model)
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
            
            # Generate translation
            response = self.model.generate_content(prompt)
            
            # Parse response
            translations = self._parse_response(response.text, len(texts))
            
            logger.info(f"Successfully translated {len(translations)} texts")
            return translations
        
        except Exception as e:
            logger.error(f"Translation failed: {e}")
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
        
        prompt = f"""Translate the following Japanese manga dialogue to {target_lang}.

Instructions:
- Preserve the tone, casualness, and emotion of the original text
- Maintain character voice and speaking style
- Handle onomatopoeia appropriately
- Keep translations natural and readable
- Return ONLY the translations, one per line, in the same order
- Do NOT include explanations or notes

Texts to translate:
{numbered_texts}

Translations:"""
        
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

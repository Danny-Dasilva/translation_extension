# Required Font Files

The following font files are required for the manga translator extension. These fonts were used in the original extension and need to be placed in this directory:

## Required Fonts (6 total):

1. **Bangers-Regular.ttf**
   - Family: Bangers
   - Style: Regular
   - Source: Google Fonts (https://fonts.google.com/specimen/Bangers)
   - License: OFL (Open Font License)

2. **Kalam-Regular.ttf**
   - Family: Kalam
   - Style: Regular
   - Source: Google Fonts (https://fonts.google.com/specimen/Kalam)
   - License: OFL (Open Font License)

3. **KOMIKAX_.ttf** (Komika Jam)
   - Family: Komika Jam
   - Style: Regular
   - Source: Requires separate download
   - License: Check license before commercial use

4. **KOMIKAH_.ttf** (Komika Slim)
   - Family: Komika Slim
   - Style: Regular
   - Source: Requires separate download
   - License: Check license before commercial use

5. **VTC-Letterer-Pro.otf**
   - Family: VTC Letterer Pro
   - Style: Regular
   - Format: OpenType (.otf)
   - Source: Commercial font (requires purchase/license)
   - License: Check license before use

6. **CCWildWords-Italic.ttf**
   - Family: CC Wild Words
   - Style: Italic
   - Source: Blambot Fonts (https://www.blambot.com/)
   - License: Check license before commercial use

## Installation:

1. Download the font files from their respective sources
2. Place all 6 font files in this directory (`public/fonts/`)
3. The extension will automatically load them via the CSS @font-face declarations

## License Notes:

- Google Fonts (Bangers, Kalam) are free and open source (OFL)
- Komika fonts: Check license terms
- VTC Letterer Pro: Commercial font (may require purchase)
- CC Wild Words: Check Blambot license terms

**IMPORTANT:** Ensure you have proper licenses for all fonts, especially for commercial use or redistribution.

## Alternative:

If you cannot obtain all fonts, you can:
1. Use only the free Google Fonts (Bangers, Kalam)
2. Modify `src/types/settings.ts` to remove unavailable fonts from the FontName type
3. Update the default font in `src/config/constants.ts`
4. Remove unused @font-face declarations from `src/content/overlay.css`

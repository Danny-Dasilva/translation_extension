# Font Files

## Bundled (shipped in this directory, web-accessible via the manifest)

These two are the renderer's actual display fonts and match the backend
`refit_final_composites.py` FONT_STACK so the extension's canvas output visually
matches the server-side PIL composite:

1. **Anton-Regular.ttf** (PRIMARY)
   - Family: Anton
   - Used for long / dialogue / narration text (backend FONT_STACK[0],
     `pick_font` default).
   - Source: Google Fonts (https://fonts.google.com/specimen/Anton)
   - License: OFL (Open Font License)

2. **Bangers-Regular.ttf** (SFX)
   - Family: Bangers
   - Used for short, exclamatory / all-caps SFX outbursts (backend
     `pick_font` short-SFX branch).
   - Source: Google Fonts (https://fonts.google.com/specimen/Bangers)
   - License: OFL (Open Font License)

Both are registered via the FontFace API in
`src/services/overlay-renderer.ts` (`tryRegisterLocalFonts`) and declared in
`web_accessible_resources` (`fonts/*`) in both manifests. A Google Fonts CDN
`<link>` (Anton + Bangers) remains only as an offline-safety fallback.

---

## Optional / legacy fonts (NOT required for backend parity)

The fonts below were referenced by the older multi-font UI and are not needed
for the backend-matching renderer. Add them only if you re-enable those styles:

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

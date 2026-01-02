import { LanguageCode } from './locales';

export type FontFamily =
	| 'BlambotClassiciCiel-Regular'
	| 'CC Victory Speech'
	| 'CC Wild Words'
	| 'NoonnuBasicGothicRegular'
	| 'NotoSansAR-Regular'
	| 'NotoSansJP-Regular'
	| 'NotoSansHI-Regular'
	| 'NotoSansTH-Regular'
	| 'system-ui';

type EmbeddedFontFamily = Exclude<FontFamily, 'system-ui'>;

type FontFormat = 'woff2' | 'opentype' | 'truetype';

interface FontDefinition {
	fontFamily: FontFamily;
	fileName: string;
	format: FontFormat;
	mimeType: string;
	lineHeight: number;
}

const defaultLineHeight = 1.13;

const fontDefinitions: Record<EmbeddedFontFamily, FontDefinition> = {
	'BlambotClassiciCiel-Regular': {
		fontFamily: 'BlambotClassiciCiel-Regular',
		fileName: 'BlambotClassiciCiel-Regular.ttf',
		format: 'truetype',
		mimeType: 'font/ttf',
		lineHeight: defaultLineHeight
	},
	'CC Victory Speech': {
		fontFamily: 'CC Victory Speech',
		fileName: 'ccvictoryspeech-regular.woff2',
		format: 'woff2',
		mimeType: 'font/woff2',
		lineHeight: 1.05
	},
	'CC Wild Words': {
		fontFamily: 'CC Wild Words',
		fileName: 'CCWildWords-Regular.otf',
		format: 'opentype',
		mimeType: 'font/otf',
		lineHeight: defaultLineHeight
	},
	NoonnuBasicGothicRegular: {
		fontFamily: 'NoonnuBasicGothicRegular',
		fileName: 'NoonnuBasicGothicRegular.woff2',
		format: 'woff2',
		mimeType: 'font/woff2',
		lineHeight: 1.25
	},
	'NotoSansAR-Regular': {
		fontFamily: 'NotoSansAR-Regular',
		fileName: 'NotoSansAR-Regular.ttf',
		format: 'truetype',
		mimeType: 'font/ttf',
		lineHeight: defaultLineHeight
	},
	'NotoSansHI-Regular': {
		fontFamily: 'NotoSansHI-Regular',
		fileName: 'NotoSansHI-Regular.ttf',
		format: 'truetype',
		mimeType: 'font/ttf',
		lineHeight: defaultLineHeight
	},
	'NotoSansJP-Regular': {
		fontFamily: 'NotoSansJP-Regular',
		fileName: 'NotoSansJP-Regular.ttf',
		format: 'truetype',
		mimeType: 'font/ttf',
		lineHeight: 1.2
	},
	'NotoSansTH-Regular': {
		fontFamily: 'NotoSansTH-Regular',
		fileName: 'NotoSansTH-Regular.ttf',
		format: 'truetype',
		mimeType: 'font/ttf',
		lineHeight: defaultLineHeight
	}
};

const defaultFontOptions: FontFamily[] = ['CC Wild Words', 'CC Victory Speech'];
const getDefaultFonts = () => [...defaultFontOptions];

const languageFontOptions: Record<LanguageCode, FontFamily[]> = {
	ar: ['NotoSansAR-Regular'],
	de: getDefaultFonts(),
	en: getDefaultFonts(),
	es: getDefaultFonts(),
	fr: getDefaultFonts(),
	hi: ['NotoSansHI-Regular'],
	id: getDefaultFonts(),
	it: getDefaultFonts(),
	ja: ['NotoSansJP-Regular'],
	ko: ['NoonnuBasicGothicRegular'],
	pl: getDefaultFonts(),
	'pt-BR': getDefaultFonts(),
	'pt-PT': getDefaultFonts(),
	ru: getDefaultFonts(),
	th: ['NotoSansTH-Regular'],
	vi: ['BlambotClassiciCiel-Regular'],
	'zh-CN': ['system-ui'],
	'zh-TW': ['system-ui']
};

const embeddedFontFamilies = Object.freeze(Object.keys(fontDefinitions) as EmbeddedFontFamily[]);

export const supportedFontFamilies = Object.freeze([
	...embeddedFontFamilies,
	'system-ui'
]) as readonly FontFamily[];

export function getFontDefinition(fontFamily: string): FontDefinition | undefined {
	return isEmbeddedFontFamily(fontFamily) ? fontDefinitions[fontFamily] : undefined;
}

export function getLanguageFontOptions(languageCode: LanguageCode): FontFamily[] {
	return languageFontOptions[languageCode] ?? [];
}

export function getLineHeightMultiplier(fontFamily: string): number {
	return getFontDefinition(fontFamily)?.lineHeight ?? defaultLineHeight;
}

export function isSupportedFontFamily(fontFamily: string): fontFamily is FontFamily {
	return (supportedFontFamilies as readonly string[]).includes(fontFamily);
}

function isEmbeddedFontFamily(fontFamily: string): fontFamily is EmbeddedFontFamily {
	return (embeddedFontFamilies as readonly string[]).includes(fontFamily);
}

import { v4 as uuidv4 } from 'uuid';
import { storage } from './storage';

const globalScope: any =
	// eslint-disable-next-line no-undef
	typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : {};
const runtimeWindow: Window | undefined =
	typeof globalScope.window !== 'undefined' ? (globalScope.window as Window) : undefined;
const runtimeDocument: Document | undefined =
	typeof globalScope.document !== 'undefined' ? (globalScope.document as Document) : undefined;
const runtimeNavigator: Navigator | undefined =
	typeof globalScope.navigator !== 'undefined' ? (globalScope.navigator as Navigator) : undefined;
const runtimeScreen: Screen | undefined =
	typeof globalScope.screen !== 'undefined' ? (globalScope.screen as Screen) : undefined;
const runtimePerformance: Performance | undefined =
	typeof globalScope.performance !== 'undefined'
		? (globalScope.performance as Performance)
		: undefined;

let fingerprint: string | undefined = undefined;

export function getFingerprint() {
	if (fingerprint) {
		return fingerprint;
	}

	// Initialize fingerprint.
	const webGlRenderer = getWebGlRenderer();
	const hardware = getHardware();
	const connectionString = getConnectionString();
	const timezoneCode = new Date().getTimezoneOffset();
	const screenInfo = getScreenInfo();
	const canvasFingerprint = getCanvasFingerprint();
	const browserFeatures = getBrowserFeatures();
	const languageInfo = getLanguageInfo();
	const touchInfo = getTouchInfo();
	const orientationInfo = getOrientationInfo();
	const userAgentInfo = getUserAgentInfo();
	const performanceInfo = getPerformanceInfo();

	const components = [
		webGlRenderer,
		hardware,
		connectionString,
		timezoneCode,
		screenInfo,
		canvasFingerprint,
		browserFeatures,
		languageInfo,
		touchInfo,
		orientationInfo,
		userAgentInfo,
		performanceInfo
	];

	const rawFingerprint = components.join('-');
	fingerprint = hashString(rawFingerprint);
	return fingerprint;
}

export function getClientUuid() {
	const uuid = storage.getItem('clientUuid');
	if (uuid) {
		return uuid;
	}

	const newUuid = uuidv4();
	storage.setItem('clientUuid', newUuid);
	return newUuid;
}

function getWebGlRenderer() {
	try {
		if (runtimeDocument) {
			const canvas = runtimeDocument.createElement('canvas');
			const gl = canvas.getContext('webgl');
			if (!gl) {
				return 'no-webgl';
			}
			const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
			return debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'unknown';
		}

		if (typeof OffscreenCanvas !== 'undefined') {
			const canvas = new OffscreenCanvas(1, 1);
			const gl = canvas.getContext('webgl');
			if (!gl) {
				return 'no-webgl';
			}
			const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
			return debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'unknown';
		}
	} catch {
		return 'webgl-error';
	}

	return 'webgl-unavailable';
}

function getHardware() {
	const hardwareConcurrency = runtimeNavigator?.hardwareConcurrency ?? 'unknown';
	const deviceMemory = (runtimeNavigator as any)?.deviceMemory ?? 'unknown';
	return `${hardwareConcurrency}-${deviceMemory}`;
}

function getConnectionString() {
	const connection = (runtimeNavigator as any)?.connection;
	const type = connection?.type ?? 'unknown';
	const rtt = connection?.rtt ?? 'unknown';
	const downlinkMax = connection?.downlinkMax ?? 'unknown';
	const effectiveType = connection?.effectiveType ?? 'unknown';
	const saveData = connection?.saveData ?? false;
	return `${type}-${rtt}-${downlinkMax}-${effectiveType}-${saveData}`;
}

function getScreenInfo() {
	const screen = runtimeScreen;
	if (!screen) {
		return 'screen-unavailable';
	}

	const width = screen.width;
	const height = screen.height;
	const colorDepth = screen.colorDepth;
	const pixelDepth = screen.pixelDepth;
	const availWidth = screen.availWidth;
	const availHeight = screen.availHeight;
	const devicePixelRatio = runtimeWindow?.devicePixelRatio ?? 1;
	return `${width}x${height}-${colorDepth}-${pixelDepth}-${availWidth}x${availHeight}-${devicePixelRatio}`;
}

function getCanvasFingerprint() {
	try {
		if (!runtimeDocument) {
			return 'canvas-unavailable';
		}

		const canvas = runtimeDocument.createElement('canvas');
		const ctx = canvas.getContext('2d');
		if (!ctx) return 'no-2d-context';

		// Draw some text and shapes to create a unique canvas fingerprint
		ctx.textBaseline = 'top';
		ctx.font = '14px Arial';
		ctx.fillStyle = 'rgba(102, 204, 0, 0.7)';
		ctx.fillRect(125, 1, 62, 20);
		ctx.fillStyle = '#f60';
		ctx.fillText('BrowserLeaks,com <canvas> 1.0', 2, 15);
		ctx.fillStyle = 'rgba(102, 204, 0, 0.7)';
		ctx.fillText('BrowserLeaks,com <canvas> 1.0', 4, 17);

		// Get image data and create a simple hash
		const imageData = canvas.toDataURL();
		let hash = 0;
		for (let i = 0; i < imageData.length; i++) {
			const char = imageData.charCodeAt(i);
			hash = (hash << 5) - hash + char;
			hash = hash & hash; // Convert to 32-bit integer
		}
		return hash.toString();
	} catch (e) {
		return 'canvas-error';
	}
}

function getBrowserFeatures() {
	const features: string[] = [];
	const navAny = runtimeNavigator as any;
	const winAny = runtimeWindow as any;

	// Check for various browser features
	if (navAny && 'serviceWorker' in navAny) features.push('sw');
	if (runtimeWindow && 'localStorage' in runtimeWindow) features.push('ls');
	if (runtimeWindow && 'sessionStorage' in runtimeWindow) features.push('ss');
	if ('indexedDB' in globalScope) features.push('idb');
	if (navAny && 'geolocation' in navAny) features.push('geo');
	if ('Notification' in globalScope) features.push('notif');
	if (navAny && 'permissions' in navAny) features.push('perm');
	if (typeof navAny?.cookieEnabled === 'boolean' && navAny.cookieEnabled) features.push('cookie');
	if (typeof navAny?.doNotTrack !== 'undefined') features.push('dnt');
	if (navAny && 'onLine' in navAny) features.push('online');
	if (navAny && 'deviceMemory' in navAny) features.push('devmem');
	if (navAny && 'connection' in navAny) features.push('conn');
	if (typeof navAny?.getBattery === 'function') features.push('bat');
	if (navAny && 'mediaDevices' in navAny) features.push('media');
	if (winAny && 'webkitRequestFileSystem' in winAny) features.push('fs');
	if (winAny && 'webkitStorageInfo' in winAny) features.push('storage');

	return features.sort().join(',');
}

function getLanguageInfo() {
	const languages =
		runtimeNavigator?.languages && runtimeNavigator.languages.length > 0
			? runtimeNavigator.languages
			: runtimeNavigator?.language
				? [runtimeNavigator.language]
				: [];
	const language = runtimeNavigator?.language ?? 'unknown';
	const userLanguage = (runtimeNavigator as any)?.userLanguage ?? 'unknown';
	const browserLanguage = (runtimeNavigator as any)?.browserLanguage ?? 'unknown';
	const systemLanguage = (runtimeNavigator as any)?.systemLanguage ?? 'unknown';

	return `${language}-${userLanguage}-${browserLanguage}-${systemLanguage}-${languages.join(',')}`;
}

function getTouchInfo() {
	const winAny = runtimeWindow as any;
	const maxTouchPoints = runtimeNavigator?.maxTouchPoints ?? 0;
	const touchSupport = !!(winAny && 'ontouchstart' in winAny);
	const touchEvent = 'TouchEvent' in globalScope;
	const pointerEvent = 'PointerEvent' in globalScope;

	return `${maxTouchPoints}-${touchSupport}-${touchEvent}-${pointerEvent}`;
}

function getOrientationInfo() {
	const winAny = runtimeWindow as any;
	const orientation = runtimeScreen?.orientation?.type ?? 'unknown';
	const angle = runtimeScreen?.orientation?.angle ?? winAny?.orientation ?? 0;
	const supportsOrientationChange = !!(runtimeWindow && 'onorientationchange' in runtimeWindow);

	return `${orientation}-${angle}-${supportsOrientationChange}`;
}

function getUserAgentInfo() {
	const userAgent = runtimeNavigator?.userAgent ?? 'unknown';
	const platform = runtimeNavigator?.platform ?? 'unknown';
	const vendor = runtimeNavigator?.vendor ?? '';
	const cookieEnabled = !!runtimeNavigator?.cookieEnabled;
	const javaEnabled =
		typeof runtimeNavigator?.javaEnabled === 'function'
			? runtimeNavigator.javaEnabled()
			: false;

	// Extract key parts to avoid full UA string (privacy)
	const isIOS = /iPad|iPhone|iPod/.test(userAgent);
	const isSafari = /Safari/.test(userAgent) && !/Chrome/.test(userAgent);
	const version = userAgent.match(/Version\/(\d+)/)?.[1] || 'unknown';
	const model = userAgent.match(/(iPhone|iPad|iPod)/)?.[1] || 'unknown';

	return `${platform}-${vendor}-${cookieEnabled}-${javaEnabled}-${isIOS}-${isSafari}-${version}-${model}`;
}

function getPerformanceInfo() {
	try {
		const perf = runtimePerformance ?? runtimeWindow?.performance;
		if (!perf) return 'no-performance';

		const timing = (perf as any).timing;
		const memory = (perf as any).memory;
		const connection = (runtimeNavigator as any)?.connection;

		// Get some timing characteristics
		const navigationStart = timing?.navigationStart || 0;
		const loadEventEnd = timing?.loadEventEnd || 0;
		const loadTime = loadEventEnd - navigationStart;

		// Memory info (Chrome/Edge only, but useful when available)
		const usedJSHeapSize = memory?.usedJSHeapSize || 0;
		const totalJSHeapSize = memory?.totalJSHeapSize || 0;
		const jsHeapSizeLimit = memory?.jsHeapSizeLimit || 0;

		// Connection info
		const effectiveType = connection?.effectiveType || 'unknown';
		const downlink = connection?.downlink || 0;
		const rtt = connection?.rtt || 0;

		return `${loadTime}-${usedJSHeapSize}-${totalJSHeapSize}-${jsHeapSizeLimit}-${effectiveType}-${downlink}-${rtt}`;
	} catch (e) {
		return 'performance-error';
	}
}

function hashString(str: string): string {
	// Simple but effective hash function (djb2 algorithm)
	let hash = 5381;
	for (let i = 0; i < str.length; i++) {
		hash = (hash * 33) ^ str.charCodeAt(i);
	}

	// Convert to positive hex string with consistent length
	const hashHex = (hash >>> 0).toString(16).padStart(8, '0');

	// For extra uniqueness, create a longer hash by processing the string in chunks
	let extendedHash = hashHex;
	const chunkSize = Math.ceil(str.length / 4);

	for (let i = 0; i < 4; i++) {
		const chunk = str.slice(i * chunkSize, (i + 1) * chunkSize);
		let chunkHash = 5381;
		for (let j = 0; j < chunk.length; j++) {
			chunkHash = (chunkHash * 33) ^ chunk.charCodeAt(j);
		}
		extendedHash += (chunkHash >>> 0).toString(16).padStart(8, '0');
	}

	return extendedHash;
}

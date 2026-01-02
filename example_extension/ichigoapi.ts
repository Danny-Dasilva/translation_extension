import { TranslationResult } from './translation';
import { appConfig } from './appConfig';
import { getFingerprint } from './fingerprint';
import { LanguageCode } from './locales';
import { TranslationModelData } from './models';
import { storage } from './storage';

// If set to true, use local implementations and turn on logging.
const isDebug = false;
export const baseUrl = isDebug ? 'http://localhost:8080' : 'https://ichigoreader.com';

enum StatusCode {
	Ok = 200,
	Created = 201,
	NoContent = 204,
	BadRequest = 400,
	Forbidden = 403,
	NotFound = 404,
	TooManyRequests = 429,
	InternalServerError = 500
}

export interface User {
	email?: string; // Unregistered users have no email. They are tracked by IP.
	subscriptionTier: 'free' | 'tier-1' | 'tier-2';
}

export async function getCurrentUser(): Promise<User> {
	// Sync auth from website cookies before checking user status
	await syncAuthFromWebsite();

	const clientUuid = await appConfig.getClientUuid();
	const headers = getIchigoApiHeaders();

	const request = await fetch(
		`${baseUrl}/metrics?clientUuid=${clientUuid}&fingerprint=${getFingerprint()}`,
		{
			method: 'GET',
			headers
		}
	);

	if (request.status !== StatusCode.Ok) {
		throw new Error('Failed to retrieve user.');
	}

	return (await request.json()) as {
		email?: string;
		subscriptionTier: 'free' | 'tier-1' | 'tier-2';
	};
}

export enum LoginStatus {
	Unknown, // Various network errors, if server is on high load, etc. Not worth handling at this time.
	UnknownEmail,
	BadPassword,
	InvalidEmail,
	Success
}

export async function login(email: string, password: string): Promise<LoginStatus> {
	const request = await fetch(`${baseUrl}/auth/login`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({ email, password })
	});

	if (request.status === StatusCode.BadRequest) {
		const json = await request.json();
		switch (json.detail.kind) {
			case 'emptyEmail':
				return LoginStatus.InvalidEmail;
			case 'userNotFound':
				return LoginStatus.UnknownEmail;
			default:
				return LoginStatus.Unknown;
		}
	}

	if (request.status === StatusCode.Forbidden) {
		return LoginStatus.BadPassword;
	}

	// Safari (iOS) will not send the cookie from a background script,
	// so we need to send an auth header.
	const body = await request.json();
	const accessToken = body?.tokens?.accessToken;
	if (accessToken) {
		storage.setItem('accessToken', accessToken);
	}

	return request.status === StatusCode.Ok ? LoginStatus.Success : LoginStatus.Unknown;
}

/**
 * Syncs authentication from the website cookies to the extension.
 * If the user is logged in on mangatranslator.ai or ichigoreader.com,
 * this will read their access cookie and store it in the extension.
 * Only syncs if extension doesn't already have an access token.
 * Returns true if sync was successful, false otherwise.
 */
export async function syncAuthFromWebsite(): Promise<boolean> {
	try {
		// Don't override if already authenticated
		if (storage.getItem('accessToken')) {
			return true;
		}

		for (const url of websiteUrls) {
			const cookie = await chrome.cookies.get({ url, name: 'access_cookie' });
			if (cookie?.value) {
				await chrome.storage.local.set({ accessToken: cookie.value });
				return true;
			}
		}

		return false;
	} catch {
		return false;
	}
}

// Website URLs for cookie management
const websiteUrls = ['https://ichigoreader.com', 'https://mangatranslator.ai'];

export async function logout(): Promise<boolean> {
	// Get the extension's token before clearing
	const extensionToken = storage.getItem('accessToken');

	const request = await fetch(`${baseUrl}/auth/logout`, {
		method: 'POST',
		headers: getIchigoApiHeaders()
	});

	// Clear the access token from local storage on logout
	storage.removeItem('accessToken');

	// Only clear website cookies if they match the extension's token
	// This prevents logging out a different account on the website
	if (extensionToken) {
		await clearWebsiteCookiesIfMatching(extensionToken);
	}

	return request.status === StatusCode.NoContent;
}

/**
 * Clears only the extension's local auth state without touching cookies.
 * Used when cookie change listener detects website logout to avoid infinite loops.
 */
export function clearExtensionAuth(): void {
	storage.removeItem('accessToken');
}

/**
 * Clears the access cookies from the website domains only if they match the given token.
 * This ensures logging out of the extension only logs out the website if same account.
 */
async function clearWebsiteCookiesIfMatching(extensionToken: string): Promise<void> {
	for (const url of websiteUrls) {
		try {
			const cookie = await chrome.cookies.get({ url, name: 'access_cookie' });
			if (cookie?.value === extensionToken) {
				await chrome.cookies.remove({ url, name: 'access_cookie' });
				await chrome.cookies.remove({ url, name: 'refresh_cookie' });
			}
		} catch {
			// Ignore errors - cookie may not exist
		}
	}
}

export enum SignupStatus {
	Unknown, // Various network errors, if server is on high load, etc. Not worth handling at this time.
	Success,
	EmailTaken
}

export async function signup(email: string, password: string): Promise<SignupStatus> {
	const request = await fetch(`${baseUrl}/signup`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({ email, password })
	});

	if (request.status === StatusCode.Forbidden) {
		return SignupStatus.EmailTaken;
	}

	// Safari (iOS) will not send the cookie from a background script,
	// so we need to send an auth header.
	const body = await request.json();
	const accessToken = body?.tokens?.accessToken;
	if (accessToken) {
		storage.setItem('accessToken', accessToken);
	}

	return request.status === StatusCode.Created ? SignupStatus.Success : SignupStatus.Unknown;
}

export async function submitFeedback(text: string): Promise<boolean> {
	const request = await fetch(`${baseUrl}/feedback`, {
		method: 'POST',
		headers: getIchigoApiHeaders(),
		body: JSON.stringify({ text })
	});

	return request.status === StatusCode.Created;
}

export async function getTranslationModels(): Promise<TranslationModelData> {
	const request = await fetch(`${baseUrl}/translate/models/extension`, {
		method: 'GET',
		headers: getIchigoApiHeaders()
	});

	return await request.json();
}

export async function translateImage(
	translateTo: LanguageCode,
	base64Image: string,
	translationModel?: string
): Promise<{ translations: TranslationResult[]; errorMessage?: string }> {
	const clientUuid = await appConfig.getClientUuid();
	const request = await fetch(`${baseUrl}/translate`, {
		method: 'POST',
		headers: getIchigoApiHeaders(),
		body: JSON.stringify({
			base64Images: [base64Image],
			translationModel,
			targetLangCode: translateTo,
			fingerprint: getFingerprint(),
			clientUuid
		})
	});

	if (request.status === StatusCode.InternalServerError) {
		const errorMessage = 'Server is down or experiencing issues. Sorry for the inconvenience.';
		return {
			errorMessage,
			translations: [
				{
					originalLanguage: 'Unknown',
					translatedText: errorMessage,
					minX: 0,
					minY: 0,
					maxX: 200,
					maxY: 200
				}
			]
		};
	}

	if (request.status === StatusCode.TooManyRequests) {
		const errorMessage = 'Out of translations. Server costs are expensive. Upgrade for more!';
		return {
			errorMessage,
			translations: [
				{
					originalLanguage: 'Unknown',
					translatedText: errorMessage,
					minX: 0,
					minY: 0,
					maxX: 200,
					maxY: 200
				}
			]
		};
	}

	const results = await request.json();

	return {
		translations: results.images[0] as TranslationResult[]
	};
}

// Optionally enable/disable debug logging separately from isDebug (but, default to on if isDebug is true).
const isDebugLoggingEnabled = isDebug || false;

export function debug(message, ...args) {
	if (isDebugLoggingEnabled) {
		console.log(message, ...args);
	}
}

function getIchigoApiHeaders(): Record<string, string> {
	const headers: Record<string, string> = {
		'Content-Type': 'application/json',
		'Client-Version': '1.0.1' // Use the new subscription types.
	};

	const accessToken = storage.getItem('accessToken');
	if (accessToken) {
		headers['Authorization'] = `Bearer ${accessToken}`;
	}

	return headers;
}

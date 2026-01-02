export interface StorageLike {
	getItem(key: string): string | null;
	setItem(key: string, value: string): void;
	removeItem(key: string): void;
}

// In-memory cache that syncs with chrome.storage.local
const cache = new Map<string, string>();

// Initialize cache from chrome.storage.local
if (typeof chrome !== 'undefined' && chrome.storage?.local) {
	chrome.storage.local.get('accessToken', result => {
		if (result.accessToken) {
			cache.set('accessToken', result.accessToken);
		}
	});

	// Listen for changes from other parts of the extension
	chrome.storage.onChanged.addListener((changes, areaName) => {
		if (areaName === 'local' && changes.accessToken) {
			if (changes.accessToken.newValue) {
				cache.set('accessToken', changes.accessToken.newValue);
			} else {
				cache.delete('accessToken');
			}
		}
	});
}

export const storage: StorageLike = (() => {
	try {
		const globalScope: any = typeof globalThis !== 'undefined' ? globalThis : undefined;
		const browserStorage: Storage | undefined = globalScope?.localStorage;

		if (browserStorage) {
			const testKey = '__ichigo_storage_test__';
			browserStorage.setItem(testKey, '1');
			browserStorage.removeItem(testKey);

			// Wrap localStorage to also check cache for accessToken
			return {
				getItem(key: string) {
					if (key === 'accessToken') {
						// Check cache first (synced from chrome.storage.local)
						const cached = cache.get(key);
						if (cached) return cached;
					}
					return browserStorage.getItem(key);
				},
				setItem(key: string, value: string) {
					browserStorage.setItem(key, value);
					if (key === 'accessToken') {
						cache.set(key, value);
						chrome.storage?.local?.set({ [key]: value });
					}
				},
				removeItem(key: string) {
					browserStorage.removeItem(key);
					if (key === 'accessToken') {
						cache.delete(key);
						chrome.storage?.local?.remove(key);
					}
				}
			};
		}
	} catch {
		// Ignore and fall back to in-memory storage.
	}

	// Fallback for service worker (no localStorage)
	return {
		getItem(key: string) {
			return cache.has(key) ? cache.get(key)! : null;
		},
		setItem(key: string, value: string) {
			cache.set(key, value);
			chrome.storage?.local?.set({ [key]: value });
		},
		removeItem(key: string) {
			cache.delete(key);
			chrome.storage?.local?.remove(key);
		}
	};
})();

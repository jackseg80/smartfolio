/**
 * Unified Fetch Module - Source unique pour tous les appels HTTP
 *
 * Expose:
 * - safeFetch: retry + ETag + X-User automatique
 * - apiCall: wrapper safeFetch optimisé pour APIs
 * - fetchCached: cache RAM + localStorage avec TTL
 * - cachedApiCall: combinaison cache + safeFetch
 *
 * Migration (Fév 2026): Centralisation de tous les patterns fetch
 */

// Re-export depuis http.js pour point d'entrée unique
import { safeFetch, apiCall } from '../modules/http.js';
export { safeFetch, apiCall };

// RAM cache
const RAM_CACHE = new Map();

// TTL configurations per data type
const CACHE_CONFIG = {
  signals: { ram: 30 * 60 * 1000, disk: 2 * 60 * 60 * 1000 },  // 30min RAM, 2h disk (CoinGecko rate limit fix)
  risk: { ram: 5 * 60 * 1000, disk: 30 * 60 * 1000 },          // 5min RAM, 30min disk
  portfolio: { ram: 1 * 60 * 1000, disk: 5 * 60 * 1000 },      // 1min RAM, 5min disk
  cycles: { ram: 10 * 60 * 1000, disk: 6 * 60 * 60 * 1000 },   // 10min RAM, 6h disk
  bourse: { ram: 10 * 60 * 1000, disk: 60 * 60 * 1000, timeout: 60000 }  // 10min RAM, 1h disk, 60s timeout
};

/**
 * Fetch with unified cache (RAM + localStorage)
 */
export async function fetchCached(key, fetchFn, cacheType = 'signals') {
  const config = CACHE_CONFIG[cacheType];
  const now = Date.now();
  
  // Try RAM cache first
  const ramEntry = RAM_CACHE.get(key);
  if (ramEntry && (now - ramEntry.timestamp) < config.ram) {
    console.debug(`Cache hit (RAM): ${key}`);
    return ramEntry.data;
  }
  
  // Try localStorage cache
  try {
    const diskKey = `cache:${key}`;
    const diskEntry = localStorage.getItem(diskKey);
    if (diskEntry) {
      const parsed = JSON.parse(diskEntry);
      if ((now - parsed.timestamp) < config.disk) {
        console.debug(`Cache hit (disk): ${key}`);
        // Also store in RAM for faster access
        RAM_CACHE.set(key, { data: parsed.data, timestamp: now });
        return parsed.data;
      }
    }
  } catch (error) {
    (window.debugLogger?.warn || console.warn)(`Failed to read disk cache for ${key}:`, error);
  }
  
  // Cache miss - fetch fresh data
  console.debug(`Cache miss: ${key}, fetching...`);
  
  try {
    const data = await retryFetch(fetchFn, 2, 500, config.timeout);
    
    // Store in both caches
    const entry = { data, timestamp: now };
    RAM_CACHE.set(key, entry);
    
    try {
      localStorage.setItem(`cache:${key}`, JSON.stringify(entry));
    } catch (error) {
      (window.debugLogger?.warn || console.warn)(`Failed to write disk cache for ${key}:`, error);
    }
    
    return data;
    
  } catch (error) {
    debugLogger.error(`Failed to fetch ${key}:`, error);
    
    // Try to return stale data as fallback
    const staleEntry = RAM_CACHE.get(key) || 
      tryParseLocalStorage(`cache:${key}`);
    
    if (staleEntry?.data) {
      (window.debugLogger?.warn || console.warn)(`Returning stale data for ${key}`);
      return staleEntry.data;
    }
    
    throw error;
  }
}

/**
 * Simple retry logic
 */
async function retryFetch(fetchFn, maxRetries = 2, baseDelay = 500, timeoutMs = 8000) {
  let lastError;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await withTimeout(fetchFn(), timeoutMs);
    } catch (error) {
      lastError = error;
      
      if (attempt < maxRetries) {
        const delay = baseDelay * Math.pow(2, attempt);
        console.debug(`Retry ${attempt + 1}/${maxRetries} after ${delay}ms`);
        await sleep(delay);
      }
    }
  }
  
  throw lastError;
}

/**
 * Add timeout to promise
 */
function withTimeout(promise, timeoutMs) {
  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      reject(new Error(`Timeout after ${timeoutMs}ms`));
    }, timeoutMs);
    
    promise
      .then(result => {
        clearTimeout(timeoutId);
        resolve(result);
      })
      .catch(error => {
        clearTimeout(timeoutId);
        reject(error);
      });
  });
}

/**
 * Sleep utility
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Safe localStorage parsing
 */
function tryParseLocalStorage(key) {
  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : null;
  } catch (error) {
    return null;
  }
}

/**
 * Get the last update timestamp for a cache key (ms since epoch).
 * Returns null if not cached.
 */
export function getLastUpdateTime(cacheKey) {
  const ramEntry = RAM_CACHE.get(cacheKey);
  if (ramEntry?.timestamp) return ramEntry.timestamp;
  try {
    const raw = localStorage.getItem(`cache:${cacheKey}`);
    if (raw) {
      const parsed = JSON.parse(raw);
      return parsed?.timestamp || null;
    }
  } catch { /* ignore */ }
  return null;
}

/**
 * Get seconds since last update for a cache key.
 * Returns null if not cached.
 */
export function getLastUpdateAgo(cacheKey) {
  const ts = getLastUpdateTime(cacheKey);
  if (!ts) return null;
  return Math.round((Date.now() - ts) / 1000);
}

/**
 * Clear all caches
 */
export function clearCache() {
  RAM_CACHE.clear();
  
  // Clear localStorage cache entries efficiently
  // Au lieu d'itérer sur tout localStorage, filtrer directement les clés
  const allKeys = Object.keys(localStorage);
  const cacheKeys = allKeys.filter(key => key.startsWith('cache:'));
  cacheKeys.forEach(key => localStorage.removeItem(key));
  
  console.debug('All caches cleared');
}

/**
 * Get cache stats for debugging
 */
export function getCacheStats() {
  const ramSize = RAM_CACHE.size;
  let diskCount = 0;

  // Compter les clés cache efficacement
  const cacheKeys = Object.keys(localStorage).filter(key => key.startsWith('cache:'));
  diskCount = cacheKeys.length;

  return { ramSize, diskCount };
}

/**
 * Cached API call - combines caching with safeFetch
 * Use this for API endpoints that benefit from caching
 *
 * @param {string} cacheKey - Unique cache key
 * @param {string} url - API endpoint URL
 * @param {Object} options - Options for safeFetch (supports forceRefresh to bypass cache)
 * @param {string} cacheType - Cache type for TTL config
 * @returns {Promise<any>} - Response data
 */
export async function cachedApiCall(cacheKey, url, options = {}, cacheType = 'signals') {
  const { forceRefresh, ...fetchOptions } = options;
  if (forceRefresh) {
    RAM_CACHE.delete(cacheKey);
    localStorage.removeItem(`cache:${cacheKey}`);
  }
  return fetchCached(cacheKey, async () => {
    const result = await safeFetch(url, fetchOptions);
    if (!result.ok) {
      throw new Error(result.error || `HTTP ${result.status}`);
    }
    return result.data;
  }, cacheType);
}

/**
 * Check if a cache key is likely a hit (quick sync check)
 * @param {string} cacheKey - Cache key to check
 * @param {string} cacheType - Cache type for TTL config
 * @returns {boolean}
 */
export function hasCacheHit(cacheKey, cacheType = 'signals') {
  if (RAM_CACHE.has(cacheKey)) return true;
  try {
    const raw = localStorage.getItem(`cache:${cacheKey}`);
    if (!raw) return false;
    const entry = JSON.parse(raw);
    const config = CACHE_CONFIG[cacheType] || CACHE_CONFIG.signals;
    return entry?.timestamp && (Date.now() - entry.timestamp) < config.disk;
  } catch { return false; }
}

/**
 * Fetch with timeout wrapper - for backward compatibility
 * @deprecated Use safeFetch with timeout option instead
 */
export async function fetchWithTimeout(url, { timeoutMs = 5000, ...fetchOptions } = {}) {
  console.warn('[fetcher.js] fetchWithTimeout is deprecated, use safeFetch with timeout option');
  const result = await safeFetch(url, { timeout: timeoutMs, ...fetchOptions });
  // Return raw response-like object for compatibility
  return {
    ok: result.ok,
    status: result.status,
    json: async () => result.data,
    headers: new Headers()
  };
}

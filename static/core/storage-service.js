/**
 * StorageService - Centralized localStorage management
 *
 * Provides type-safe access to localStorage with:
 * - User/auth management (activeUser, authToken)
 * - Cache management (prefixed keys)
 * - Settings persistence
 * - Error handling
 *
 * Usage:
 *   import { StorageService } from './core/storage-service.js';
 *   const user = StorageService.getActiveUser();
 *   StorageService.setActiveUser('jack');
 *
 * Migration (Feb 2026): Centralizing 397 localStorage calls across 63 files
 */

// Storage keys constants
const KEYS = {
  ACTIVE_USER: 'activeUser',
  AUTH_TOKEN: 'authToken',
  DATA_SOURCE: 'data_source',
  CACHE_PREFIX: 'cache:',
  UI_PREFIX: '__ui.',
  DEBUG_MODE: 'debugMode'
};

// Default values
const DEFAULTS = {
  USER: 'demo',
  SOURCE: 'cointracking'
};

/**
 * Centralized localStorage service
 */
export const StorageService = {
  // ==========================================
  // USER & AUTH
  // ==========================================

  /**
   * Get the active user ID
   * @returns {string} User ID or 'demo' as fallback
   */
  getActiveUser() {
    return localStorage.getItem(KEYS.ACTIVE_USER) || DEFAULTS.USER;
  },

  /**
   * Set the active user ID
   * @param {string} userId - User ID to set
   */
  setActiveUser(userId) {
    if (!userId) {
      console.warn('[StorageService] setActiveUser called with empty userId');
      return;
    }
    localStorage.setItem(KEYS.ACTIVE_USER, userId);
  },

  /**
   * Get the auth token
   * @returns {string|null} Auth token or null
   */
  getAuthToken() {
    return localStorage.getItem(KEYS.AUTH_TOKEN);
  },

  /**
   * Set the auth token
   * @param {string} token - JWT token
   */
  setAuthToken(token) {
    if (!token) {
      console.warn('[StorageService] setAuthToken called with empty token');
      return;
    }
    localStorage.setItem(KEYS.AUTH_TOKEN, token);
  },

  /**
   * Check if user is authenticated
   * @returns {boolean} True if auth token exists
   */
  isAuthenticated() {
    return !!this.getAuthToken();
  },

  /**
   * Clear all auth data (logout)
   */
  clearAuth() {
    localStorage.removeItem(KEYS.AUTH_TOKEN);
    localStorage.removeItem(KEYS.ACTIVE_USER);
  },

  // ==========================================
  // DATA SOURCE
  // ==========================================

  /**
   * Get the current data source
   * @returns {string} Data source or 'cointracking' as fallback
   */
  getDataSource() {
    return localStorage.getItem(KEYS.DATA_SOURCE) || DEFAULTS.SOURCE;
  },

  /**
   * Set the data source
   * @param {string} source - Data source name
   */
  setDataSource(source) {
    if (!source) return;
    localStorage.setItem(KEYS.DATA_SOURCE, source);
  },

  // ==========================================
  // CACHE MANAGEMENT
  // ==========================================

  /**
   * Get a cached value
   * @param {string} key - Cache key (without prefix)
   * @returns {any|null} Parsed cached value or null
   */
  getCached(key) {
    try {
      const item = localStorage.getItem(KEYS.CACHE_PREFIX + key);
      return item ? JSON.parse(item) : null;
    } catch (error) {
      console.warn(`[StorageService] Failed to parse cache for ${key}:`, error);
      return null;
    }
  },

  /**
   * Set a cached value
   * @param {string} key - Cache key (without prefix)
   * @param {any} value - Value to cache (will be JSON stringified)
   */
  setCached(key, value) {
    try {
      localStorage.setItem(KEYS.CACHE_PREFIX + key, JSON.stringify(value));
    } catch (error) {
      console.warn(`[StorageService] Failed to cache ${key}:`, error);
    }
  },

  /**
   * Remove a cached value
   * @param {string} key - Cache key (without prefix)
   */
  removeCached(key) {
    localStorage.removeItem(KEYS.CACHE_PREFIX + key);
  },

  /**
   * Clear all cache entries
   * @returns {number} Number of cleared entries
   */
  clearAllCache() {
    const keys = Object.keys(localStorage).filter(k => k.startsWith(KEYS.CACHE_PREFIX));
    keys.forEach(k => localStorage.removeItem(k));
    return keys.length;
  },

  // ==========================================
  // UI STATE
  // ==========================================

  /**
   * Get UI state for a component
   * @param {string} component - Component identifier
   * @param {string} key - State key
   * @returns {any|null} Parsed state or null
   */
  getUIState(component, key) {
    try {
      const fullKey = `${KEYS.UI_PREFIX}${component}.${key}`;
      const item = localStorage.getItem(fullKey);
      return item ? JSON.parse(item) : null;
    } catch (error) {
      return null;
    }
  },

  /**
   * Set UI state for a component
   * @param {string} component - Component identifier
   * @param {string} key - State key
   * @param {any} value - State value
   */
  setUIState(component, key, value) {
    try {
      const fullKey = `${KEYS.UI_PREFIX}${component}.${key}`;
      localStorage.setItem(fullKey, JSON.stringify(value));
    } catch (error) {
      console.warn(`[StorageService] Failed to save UI state ${component}.${key}:`, error);
    }
  },

  // ==========================================
  // DEBUG
  // ==========================================

  /**
   * Check if debug mode is enabled
   * @returns {boolean}
   */
  isDebugMode() {
    return localStorage.getItem(KEYS.DEBUG_MODE) === 'true';
  },

  /**
   * Set debug mode
   * @param {boolean} enabled
   */
  setDebugMode(enabled) {
    localStorage.setItem(KEYS.DEBUG_MODE, enabled ? 'true' : 'false');
  },

  // ==========================================
  // GENERIC OPERATIONS
  // ==========================================

  /**
   * Get raw value from localStorage
   * @param {string} key - Storage key
   * @returns {string|null}
   */
  get(key) {
    return localStorage.getItem(key);
  },

  /**
   * Set raw value in localStorage
   * @param {string} key - Storage key
   * @param {string} value - Value to store
   */
  set(key, value) {
    localStorage.setItem(key, value);
  },

  /**
   * Remove a key from localStorage
   * @param {string} key - Storage key
   */
  remove(key) {
    localStorage.removeItem(key);
  },

  /**
   * Get storage stats
   * @returns {Object} Stats about localStorage usage
   */
  getStats() {
    const keys = Object.keys(localStorage);
    return {
      totalKeys: keys.length,
      cacheKeys: keys.filter(k => k.startsWith(KEYS.CACHE_PREFIX)).length,
      uiKeys: keys.filter(k => k.startsWith(KEYS.UI_PREFIX)).length,
      estimatedSize: new Blob(Object.values(localStorage)).size
    };
  }
};

// Export keys for backward compatibility
export const STORAGE_KEYS = KEYS;

// Default export for convenience
export default StorageService;

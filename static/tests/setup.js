/**
 * Vitest setup file - Mocks and global configuration
 * Resolves store initialization issues during test imports
 */

import { vi } from 'vitest';

// Mock localStorage if not available (happy-dom should provide it, but just in case)
if (typeof localStorage === 'undefined') {
  global.localStorage = {
    getItem: vi.fn(() => null),
    setItem: vi.fn(),
    removeItem: vi.fn(),
    clear: vi.fn(),
    length: 0,
    key: vi.fn(() => null)
  };
}

// Mock window.debugLogger to prevent console spam during tests
if (typeof window !== 'undefined') {
  window.debugLogger = {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn()
  };
}

// Mock the risk-dashboard-store to prevent initialization issues
// This allows pure functions to be tested without store dependencies
vi.mock('../core/risk-dashboard-store.js', () => ({
  store: {
    snapshot: vi.fn(() => ({
      scores: { blended: 60, risk: 70, onchain: 65 },
      governance: { ml_signals: null },
      ui: { apiStatus: { backend: 'ok' } }
    })),
    update: vi.fn(),
    subscribe: vi.fn(() => vi.fn()), // Unsubscribe function
    hydrate: vi.fn(),
    persist: vi.fn(),
    reset: vi.fn()
  }
}));

// Mock global-config to prevent API calls during tests
vi.mock('../global-config.js', () => ({
  API_BASE_URL: 'http://localhost:8080',
  WS_BASE_URL: 'ws://localhost:8080',
  DEFAULT_USER: 'demo',
  CACHE_TTL: {
    BALANCES: 300000,
    PRICES: 180000,
    ML_PREDICTIONS: 900000
  }
}));

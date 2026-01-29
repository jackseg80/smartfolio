/**
 * Jest setup file - Mocks and global configuration
 * Resolves store initialization issues during test imports
 */

import { jest } from '@jest/globals';

// Mock localStorage if not available
if (typeof localStorage === 'undefined') {
  global.localStorage = {
    getItem: jest.fn(() => null),
    setItem: jest.fn(),
    removeItem: jest.fn(),
    clear: jest.fn(),
    length: 0,
    key: jest.fn(() => null)
  };
}

// Mock window.debugLogger to prevent console spam during tests
if (typeof window !== 'undefined') {
  window.debugLogger = {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn()
  };
}

// Set default Risk Semantics mode for tests
if (typeof localStorage !== 'undefined') {
  localStorage.setItem('RISK_SEMANTICS_MODE', 'v2_conservative');
}

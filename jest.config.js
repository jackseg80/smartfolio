/**
 * Jest configuration for SmartFolio
 * ESM-compatible setup with jsdom environment
 */

export default {
  // Use jsdom for browser-like environment
  testEnvironment: 'jsdom',

  // ESM support (no extensionsToTreatAsEsm needed with "type": "module")
  transform: {},

  // Test file patterns
  testMatch: [
    '**/static/tests/**/*.test.js',
    '**/__tests__/**/*.js'
  ],

  // Coverage configuration
  collectCoverageFrom: [
    'static/modules/**/*.js',
    'static/core/**/*.js',
    '!static/tests/**',
    '!**/*.test.js'
  ],

  coverageReporters: ['text', 'html', 'json'],

  coverageThreshold: {
    global: {
      statements: 30,
      branches: 25,
      functions: 30,
      lines: 30
    }
  },

  // Module paths
  moduleFileExtensions: ['js', 'json'],

  // Setup files
  setupFilesAfterEnv: ['<rootDir>/static/tests/jest.setup.js'],

  // Ignore patterns
  testPathIgnorePatterns: [
    '/node_modules/',
    '/tests/e2e/'
  ],

  // Verbose output
  verbose: true
};

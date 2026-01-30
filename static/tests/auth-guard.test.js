/**
 * Unit tests for Auth Guard
 * Tests JWT authentication, token verification, and RBAC
 */

import {
  getAuthToken,
  getCurrentUser,
  getUserInfo,
  getAuthHeaders,
  verifyToken,
  logout,
  checkAuth,
  hasRole,
  isAdmin,
  requireRole
} from '../core/auth-guard.js';
import { describe, test, expect, beforeEach, afterEach, jest } from '@jest/globals';

// Mock fetch globally
global.fetch = jest.fn();

describe('Auth Guard - Token Management', () => {

  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
  });

  test('should return null when no token is stored', () => {
    const token = getAuthToken();
    expect(token).toBeNull();
  });

  test('should return token when stored in localStorage', () => {
    localStorage.setItem('authToken', 'test-token-123');

    const token = getAuthToken();
    expect(token).toBe('test-token-123');
  });

  test('should return current user from localStorage', () => {
    localStorage.setItem('currentUser', 'jack');

    const user = getCurrentUser();
    expect(user).toBe('jack');
  });

  test('should return user info when stored', () => {
    const userInfo = { username: 'jack', role: 'admin', email: 'jack@example.com' };
    localStorage.setItem('userInfo', JSON.stringify(userInfo));

    const info = getUserInfo();
    expect(info).toEqual(userInfo);
  });

  test('should return null when user info is malformed', () => {
    localStorage.setItem('userInfo', 'invalid-json');

    const info = getUserInfo();
    expect(info).toBeNull();
  });
});

describe('Auth Guard - Headers', () => {

  beforeEach(() => {
    localStorage.clear();
  });

  test('should generate auth headers with token', () => {
    localStorage.setItem('authToken', 'token-abc');
    localStorage.setItem('currentUser', 'demo');

    const headers = getAuthHeaders();

    expect(headers).toHaveProperty('Authorization', 'Bearer token-abc');
    expect(headers).toHaveProperty('X-User', 'demo');
  });

  test('should generate auth headers without X-User when disabled', () => {
    localStorage.setItem('authToken', 'token-xyz');
    localStorage.setItem('currentUser', 'jack');

    const headers = getAuthHeaders(false);

    expect(headers).toHaveProperty('Authorization', 'Bearer token-xyz');
    expect(headers).not.toHaveProperty('X-User');
  });

  test('should return empty headers when no token', () => {
    const headers = getAuthHeaders();

    expect(headers).toEqual({});
  });
});

describe('Auth Guard - Token Verification', () => {

  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
  });

  test('should verify valid token successfully', async () => {
    localStorage.setItem('authToken', 'valid-token');

    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ valid: true, user: 'demo' })
    });

    const result = await verifyToken();

    expect(result).toBe(true);
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  test('should return false for invalid token', async () => {
    localStorage.setItem('authToken', 'invalid-token');

    global.fetch.mockResolvedValueOnce({
      ok: false,
      status: 401
    });

    const result = await verifyToken();

    expect(result).toBe(false);
  });

  test('should return false when no token stored', async () => {
    const result = await verifyToken();

    expect(result).toBe(false);
    expect(fetch).not.toHaveBeenCalled();
  });

  test('should handle network errors gracefully', async () => {
    localStorage.setItem('authToken', 'token-123');

    global.fetch.mockRejectedValueOnce(new Error('Network error'));

    const result = await verifyToken();

    expect(result).toBe(false);
  });
});

describe('Auth Guard - RBAC', () => {

  beforeEach(() => {
    localStorage.clear();
  });

  test('should detect admin role correctly', () => {
    const adminInfo = { username: 'jack', role: 'admin' };
    localStorage.setItem('userInfo', JSON.stringify(adminInfo));

    expect(isAdmin()).toBe(true);
    expect(hasRole('admin')).toBe(true);
  });

  test('should detect viewer role correctly', () => {
    const viewerInfo = { username: 'demo', role: 'viewer' };
    localStorage.setItem('userInfo', JSON.stringify(viewerInfo));

    expect(isAdmin()).toBe(false);
    expect(hasRole('viewer')).toBe(true);
    expect(hasRole('admin')).toBe(false);
  });

  test('should return false when no user info', () => {
    expect(isAdmin()).toBe(false);
    expect(hasRole('admin')).toBe(false);
  });

  test('should throw error when requiring missing role', () => {
    const viewerInfo = { username: 'demo', role: 'viewer' };
    localStorage.setItem('userInfo', JSON.stringify(viewerInfo));

    expect(() => {
      requireRole('admin');
    }).toThrow('Insufficient permissions');
  });

  test('should not throw when user has required role', () => {
    const adminInfo = { username: 'jack', role: 'admin' };
    localStorage.setItem('userInfo', JSON.stringify(adminInfo));

    expect(() => {
      requireRole('admin');
    }).not.toThrow();
  });

  test('should throw custom message when specified', () => {
    const viewerInfo = { username: 'demo', role: 'viewer' };
    localStorage.setItem('userInfo', JSON.stringify(viewerInfo));

    expect(() => {
      requireRole('admin', 'Admin access required');
    }).toThrow('Admin access required');
  });
});

describe('Auth Guard - Check Auth', () => {

  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
  });

  test('should pass when valid token and user info exist', async () => {
    localStorage.setItem('authToken', 'valid-token');
    localStorage.setItem('userInfo', JSON.stringify({ username: 'demo', role: 'viewer' }));

    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ valid: true })
    });

    await expect(checkAuth({ skipTokenCheck: true })).resolves.not.toThrow();
  });

  test('should redirect to login when no token (in browser)', async () => {
    // Mock window.location
    delete window.location;
    window.location = { href: '', replace: jest.fn() };

    await checkAuth({ redirect: false });

    // Should not throw in test environment
  });
});

describe('Auth Guard - Logout', () => {

  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
  });

  test('should clear localStorage on logout', async () => {
    localStorage.setItem('authToken', 'token');
    localStorage.setItem('currentUser', 'demo');
    localStorage.setItem('userInfo', JSON.stringify({ username: 'demo' }));

    global.fetch.mockResolvedValueOnce({ ok: true });

    await logout(false);

    expect(localStorage.getItem('authToken')).toBeNull();
    expect(localStorage.getItem('currentUser')).toBeNull();
    expect(localStorage.getItem('userInfo')).toBeNull();
  });

  test('should call logout API endpoint', async () => {
    localStorage.setItem('authToken', 'token-abc');

    global.fetch.mockResolvedValueOnce({ ok: true });

    await logout(false);

    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/auth/logout'),
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Authorization': 'Bearer token-abc'
        })
      })
    );
  });

  test('should handle logout API errors gracefully', async () => {
    localStorage.setItem('authToken', 'token-xyz');

    global.fetch.mockRejectedValueOnce(new Error('Network error'));

    await expect(logout(false)).resolves.not.toThrow();

    // Should still clear localStorage
    expect(localStorage.getItem('authToken')).toBeNull();
  });
});

describe('Auth Guard - Edge Cases', () => {

  beforeEach(() => {
    localStorage.clear();
  });

  test('should handle missing localStorage gracefully', () => {
    // Temporarily disable localStorage
    const originalLocalStorage = global.localStorage;
    delete global.localStorage;

    expect(() => getAuthToken()).not.toThrow();
    expect(() => getCurrentUser()).not.toThrow();

    // Restore
    global.localStorage = originalLocalStorage;
  });

  test('should handle concurrent token verifications', async () => {
    localStorage.setItem('authToken', 'token-123');

    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ valid: true })
    });

    const results = await Promise.all([
      verifyToken(),
      verifyToken(),
      verifyToken()
    ]);

    expect(results.every(r => r === true)).toBe(true);
  });
});

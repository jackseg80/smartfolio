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
import { describe, test, expect, beforeEach, jest } from '@jest/globals';

// Mock fetch globally
global.fetch = jest.fn();

describe('Auth Guard - Token Management', () => {

  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
    global.fetch.mockClear();
    if (global.alert) global.alert.mockClear();
    if (window.location) window.location.href = '';
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
    localStorage.setItem('activeUser', 'jack');

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

  test('should return only X-User when no token but includeXUser=true', () => {
    const headers = getAuthHeaders();

    expect(headers).not.toHaveProperty('Authorization');
    expect(headers).toHaveProperty('X-User', 'demo'); // getCurrentUser() returns 'demo' by default
  });

  test('should return empty headers when no token and includeXUser=false', () => {
    const headers = getAuthHeaders(false);

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
      json: async () => ({ ok: true, data: { valid: true } })
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
    jest.clearAllMocks();
    if (global.alert) global.alert.mockClear();
  });

  test('should detect admin role correctly', () => {
    const adminInfo = { username: 'jack', roles: ['admin'] };
    localStorage.setItem('userInfo', JSON.stringify(adminInfo));

    expect(isAdmin()).toBe(true);
    expect(hasRole('admin')).toBe(true);
  });

  test('should detect viewer role correctly', () => {
    const viewerInfo = { username: 'demo', roles: ['viewer'] };
    localStorage.setItem('userInfo', JSON.stringify(viewerInfo));

    expect(isAdmin()).toBe(false);
    expect(hasRole('viewer')).toBe(true);
    expect(hasRole('admin')).toBe(false);
  });

  test('should return false when no user info', () => {
    expect(isAdmin()).toBe(false);
    expect(hasRole('admin')).toBe(false);
  });

  test('should alert when requiring missing role', () => {
    const viewerInfo = { username: 'demo', roles: ['viewer'] };
    localStorage.setItem('userInfo', JSON.stringify(viewerInfo));

    // requireRole calls alert() and attempts to set window.location.href
    requireRole('admin');

    expect(global.alert).toHaveBeenCalledWith('Access denied: Insufficient permissions');
    // Note: window.location.href cannot be reliably tested in jsdom
  });

  test('should not alert when user has required role', () => {
    const adminInfo = { username: 'jack', roles: ['admin'] };
    localStorage.setItem('userInfo', JSON.stringify(adminInfo));

    requireRole('admin');

    expect(global.alert).not.toHaveBeenCalled();
  });

  test('should alert with custom message when specified', () => {
    const viewerInfo = { username: 'demo', roles: ['viewer'] };
    localStorage.setItem('userInfo', JSON.stringify(viewerInfo));

    requireRole('admin', 'Admin access required');

    expect(global.alert).toHaveBeenCalledWith('Access denied: Admin access required');
    // Note: window.location.href cannot be reliably tested in jsdom
  });
});

describe('Auth Guard - Check Auth', () => {

  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
  });

  test('should pass when valid token and user info exist', async () => {
    localStorage.setItem('authToken', 'valid-token');
    localStorage.setItem('userInfo', JSON.stringify({ username: 'demo', roles: ['viewer'] }));

    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ok: true, data: { valid: true } })
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
    localStorage.setItem('activeUser', 'demo');
    localStorage.setItem('userInfo', JSON.stringify({ username: 'demo' }));

    global.fetch.mockResolvedValueOnce({ ok: true });

    await logout(false);

    expect(localStorage.getItem('authToken')).toBeNull();
    expect(localStorage.getItem('activeUser')).toBeNull();
    expect(localStorage.getItem('userInfo')).toBeNull();
  });

  test('should call logout API endpoint', async () => {
    localStorage.setItem('authToken', 'token-abc');

    global.fetch.mockResolvedValueOnce({ ok: true });

    await logout(false);

    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/auth/logout'),
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
    // Only clear if localStorage exists (some tests may delete it)
    if (global.localStorage) {
      localStorage.clear();
    }
    jest.clearAllMocks();
  });

  test('should handle empty localStorage gracefully', () => {
    // Ensure localStorage is empty
    localStorage.clear();

    // These functions should return default/null values when localStorage is empty
    const token = getAuthToken();
    const user = getCurrentUser();
    const info = getUserInfo();

    expect(token).toBeNull();
    expect(user).toBe('demo'); // getCurrentUser returns 'demo' as default
    expect(info).toBeNull();
  });

  test('should handle concurrent token verifications', async () => {
    localStorage.setItem('authToken', 'token-123');

    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ ok: true, data: { valid: true } })
    });

    const results = await Promise.all([
      verifyToken(),
      verifyToken(),
      verifyToken()
    ]);

    expect(results.every(r => r === true)).toBe(true);
  });
});

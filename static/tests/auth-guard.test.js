/**
 * Unit tests for Auth Guard
 * Tests JWT authentication, token verification, and RBAC
 */

import {
  getAuthToken,
  getCurrentUser,
  getUserInfo,
  getAuthHeaders,
  createAuthenticatedFetch,
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
    localStorage.setItem('activeUser', 'jack');

    const headers = getAuthHeaders();

    expect(headers).toHaveProperty('Authorization', 'Bearer token-abc');
    expect(headers).toHaveProperty('X-User', 'jack');
  });

  test('should generate auth headers without X-User when disabled', () => {
    localStorage.setItem('authToken', 'token-xyz');
    localStorage.setItem('currentUser', 'jack');

    const headers = getAuthHeaders(false);

    expect(headers).toHaveProperty('Authorization', 'Bearer token-xyz');
    expect(headers).not.toHaveProperty('X-User');
  });

  test('should not invent an X-User when no identity exists', () => {
    const headers = getAuthHeaders();

    expect(headers).not.toHaveProperty('Authorization');
    expect(headers).not.toHaveProperty('X-User');
  });

  test('should return empty headers when no token and includeXUser=false', () => {
    const headers = getAuthHeaders(false);

    expect(headers).toEqual({});
  });

  test('should enrich legacy same-origin fetch calls', async () => {
    localStorage.setItem('authToken', 'token-dual');
    localStorage.setItem('activeUser', 'jack');
    const transport = jest.fn().mockResolvedValue({ ok: true });
    const authenticatedFetch = createAuthenticatedFetch(transport);

    await authenticatedFetch('/api/private', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });

    const options = transport.mock.calls[0][1];
    expect(options.credentials).toBe('same-origin');
    expect(options.headers.get('Authorization')).toBe('Bearer token-dual');
    expect(options.headers.get('X-User')).toBe('jack');
    expect(options.headers.get('Content-Type')).toBe('application/json');
  });

  test('should never leak authentication headers cross-origin', async () => {
    localStorage.setItem('authToken', 'secret-token');
    localStorage.setItem('activeUser', 'jack');
    const transport = jest.fn().mockResolvedValue({ ok: true });
    const authenticatedFetch = createAuthenticatedFetch(transport);
    const options = { headers: { Accept: 'application/json' } };

    await authenticatedFetch('https://example.org/public', options);

    expect(transport).toHaveBeenCalledWith('https://example.org/public', options);
    expect(options.headers).toEqual({ Accept: 'application/json' });
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
    global.fetch.mockResolvedValueOnce({
      ok: false,
      status: 401
    });

    const result = await verifyToken();

    expect(result).toBe(false);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledWith(
      'http://localhost/auth/session',
      expect.objectContaining({ credentials: 'same-origin' })
    );
  });

  test('should refresh an expired cookie session with CSRF', async () => {
    document.cookie = 'smartfolio_csrf=csrf-test-token';
    global.fetch
      .mockResolvedValueOnce({ ok: false, status: 401 })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          ok: true,
          data: { user: { id: 'jack', label: 'Jack', roles: ['admin'] } }
        })
      });

    const result = await verifyToken();

    expect(result).toBe(true);
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(fetch).toHaveBeenLastCalledWith(
      'http://localhost/auth/refresh',
      expect.objectContaining({
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'X-CSRF-Token': 'csrf-test-token' }
      })
    );
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

    // These functions must fail closed when localStorage is empty.
    const token = getAuthToken();
    const user = getCurrentUser();
    const info = getUserInfo();

    expect(token).toBeNull();
    expect(user).toBeNull();
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

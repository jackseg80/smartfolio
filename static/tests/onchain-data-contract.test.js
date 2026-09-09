import { jest } from '@jest/globals';

import { fetchCryptoToolboxIndicators } from '../modules/onchain-indicators.js';


const CACHE_KEY = 'CTB_ONCHAIN_CACHE_V2';


describe('on-chain observation contract', () => {
  beforeEach(() => {
    localStorage.clear();
    window.getApiBase = () => 'http://smartfolio.test';
    window.debugLogger = {
      debug: jest.fn(),
      info: jest.fn(),
      warn: jest.fn(),
      error: jest.fn()
    };
    global.fetch = jest.fn();
  });

  test('fresh cache preserves the observation timestamp and reports its age', async () => {
    const observedAt = '2026-09-09T10:00:00.000Z';
    localStorage.setItem(CACHE_KEY, JSON.stringify({
      indicators: { mvrv: { name: 'MVRV', value_numeric: 42 } },
      count: 1,
      fetched_at: observedAt,
      saved_at: Date.now() - 60_000
    }));

    const result = await fetchCryptoToolboxIndicators();

    expect(result.fetched_at).toBe(observedAt);
    expect(result.served_from).toBe('cache');
    expect(result.stale).toBe(false);
    expect(result.cache_age_ms).toBeGreaterThanOrEqual(60_000);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  test('hard-expired cache is not returned after a network failure', async () => {
    localStorage.setItem(CACHE_KEY, JSON.stringify({
      indicators: { mvrv: { name: 'MVRV', value_numeric: 42 } },
      count: 1,
      fetched_at: '2026-09-09T08:00:00.000Z',
      saved_at: Date.now() - (3 * 60 * 60 * 1000)
    }));
    global.fetch.mockRejectedValue(new Error('network down'));

    const result = await fetchCryptoToolboxIndicators();

    expect(result.available).toBe(false);
    expect(result.indicators).toEqual({});
    expect(result.fetched_at).toBeNull();
    expect(result.served_from).toBe('unavailable');
  });

  test('an all-zero network observation is rejected and never cached', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        indicators: [
          { name: 'MVRV', value_numeric: 0 },
          { name: 'NVT', value_numeric: 0 },
          { name: 'Puell', value_numeric: 0 }
        ]
      })
    });

    const result = await fetchCryptoToolboxIndicators({ force: true });

    expect(result.available).toBe(false);
    expect(result.indicators).toEqual({});
    expect(localStorage.getItem(CACHE_KEY)).toBeNull();
  });
});

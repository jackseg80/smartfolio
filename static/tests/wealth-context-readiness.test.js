import { describe, expect, test } from '@jest/globals';

import { waitForWealthContextReady } from '../core/wealth-context-ready.js';

describe('dashboard source initialization', () => {
  test('waits until the source context is fully applied', async () => {
    let resolveReady;
    const sourceReady = new Promise(resolve => {
      resolveReady = resolve;
    });
    const fakeWindow = {};

    setTimeout(() => {
      fakeWindow.wealthContextBar = {
        getContext: () => ({ account: 'api:cointracking_api', bourse: 'saxo:saxobank_csv' }),
        whenReady: () => sourceReady
      };
    }, 5);
    setTimeout(() => {
      resolveReady({ account: 'api:cointracking_api', bourse: 'saxo:saxobank_csv' });
    }, 15);

    const result = await waitForWealthContextReady(fakeWindow, 250);

    expect(result).toEqual({
      ready: true,
      context: { account: 'api:cointracking_api', bourse: 'saxo:saxobank_csv' }
    });
  });

  test('returns a bounded fallback when the context bar never loads', async () => {
    const result = await waitForWealthContextReady({}, 10);

    expect(result).toEqual({ ready: false, context: null });
  });
});

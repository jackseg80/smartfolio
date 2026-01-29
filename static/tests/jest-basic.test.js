/**
 * Basic Jest test to verify setup
 */

import { describe, test, expect } from '@jest/globals';

describe('Jest Setup', () => {
  test('should work with basic assertions', () => {
    expect(1 + 1).toBe(2);
  });

  test('should work with strings', () => {
    expect('hello').toBe('hello');
  });

  test('should work with arrays', () => {
    expect([1, 2, 3]).toHaveLength(3);
  });
});

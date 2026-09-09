import { describe, expect, test } from '@jest/globals';

import { calculateZeroSumCappedMoves } from '../components/unified-insights/allocation-calculator.js';
import { computeMacroTargetsDynamic } from '../core/unified-insights-v2.js';
import { computeCCS, DEFAULT_CCS_WEIGHTS } from '../modules/signals-engine.js';

describe('crypto decision-chain invariants', () => {
  test('preserves a defensive stablecoin budget above the former hidden 60% cap', () => {
    const context = { flags: {} };
    const targets = computeMacroTargetsDynamic(
      context,
      { target_stables_pct: 80 },
      {},
      null
    );

    expect(targets.Stablecoins).toBe(80);
    expect(Object.values(targets).reduce((sum, value) => sum + value, 0)).toBeCloseTo(100, 8);
  });

  test('moves proportionally toward targets without negative spot allocations', () => {
    const entries = [
      { k: 'BTC', cur: 100, tgt: 0, delta: -100 },
      { k: 'ETH', cur: 0, tgt: 90, delta: 90 },
      { k: 'SOL', cur: 0, tgt: 10, delta: 10 },
      { k: 'Others', cur: 0, tgt: 0, delta: 0 }
    ];

    const result = calculateZeroSumCappedMoves(entries, 7);
    const byGroup = Object.fromEntries(result.map(entry => [entry.k, entry]));

    expect(result.reduce((sum, entry) => sum + entry.suggested, 0)).toBeCloseTo(0, 10);
    for (const entry of result) {
      expect(Math.abs(entry.suggested)).toBeLessThanOrEqual(7);
      expect(entry.cur + entry.suggested).toBeGreaterThanOrEqual(0);
      expect(Math.abs(entry.suggested)).toBeLessThanOrEqual(Math.abs(entry.delta));
      expect(entry.suggested * entry.delta).toBeGreaterThanOrEqual(0);
    }

    expect(byGroup.BTC.suggested).toBeCloseTo(-7, 10);
    expect(byGroup.ETH.suggested).toBeCloseTo(6.3, 10);
    expect(byGroup.SOL.suggested).toBeCloseTo(0.7, 10);
    expect(byGroup.Others.suggested).toBe(0);
  });

  test('returns no one-sided trade when zero-sum financing is impossible', () => {
    const result = calculateZeroSumCappedMoves(
      [{ k: 'ETH', cur: 0, tgt: 10, delta: 10 }],
      7
    );

    expect(result[0].suggested).toBe(0);
  });

  test('does not manufacture a CCS score when one required signal is unavailable', () => {
    const signals = Object.fromEntries(
      Object.keys(DEFAULT_CCS_WEIGHTS)
        .filter(key => key !== 'model_version')
        .map(key => [key, { value: 50, source: 'test' }])
    );
    signals.funding_rate = { value: null, source: 'unavailable' };

    const result = computeCCS(signals);

    expect(result.available).toBe(false);
    expect(result.score).toBeNull();
    expect(result.missing_signals).toEqual(['funding_rate']);
  });

  test('computes CCS only when every weighted input is numeric', () => {
    const signals = {
      fear_greed: { value: 50, source: 'test' },
      btc_dominance: { value: 55, source: 'test' },
      funding_rate: { value: 0, source: 'test' },
      eth_btc_ratio: { value: 0.04, source: 'test' },
      volatility: { value: 0.5, source: 'test' },
      trend: { value: 0, source: 'test' }
    };

    const result = computeCCS(signals);

    expect(result.available).toBe(true);
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(100);
    expect(Object.keys(result.signals)).toHaveLength(6);
  });
});

import { describe, expect, jest, test } from '@jest/globals';

import { calculateZeroSumCappedMoves } from '../components/unified-insights/allocation-calculator.js';
import { formatSignedPercent, renderAllocationBlock } from '../components/unified-insights/execution-plan-renderer.js';
import { computeMacroTargetsDynamic } from '../core/unified-insights-v2.js';
import { computeCCS, DEFAULT_CCS_WEIGHTS } from '../modules/signals-engine.js';
import { proposeTargets } from '../modules/targets-coordinator.js';
import { renderTargetsContent } from '../modules/risk-targets-tab.js';
import { store } from '../core/risk-dashboard-store.js';
import { forceReloadTaxonomy, getAssetGroup } from '../shared-asset-groups.js';

describe('crypto decision-chain invariants', () => {
  test('formats execution deltas without floating-point artifacts', () => {
    expect(formatSignedPercent(0.9999999999999991)).toBe('+1.0%');
    expect(formatSignedPercent(-0.5283539625283885)).toBe('-0.5%');
    expect(formatSignedPercent(-0.009341245248375374)).toBe('0.0%');
  });

  test('classifies CoinTracking duplicate suffixes like the backend taxonomy', async () => {
    window.getApiBase = jest.fn(() => 'http://test');
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        aliases: {
          HYPE: 'L1/L0 majors',
          ARB: 'L2/Scaling',
          USD: 'Stablecoins',
          TRUMP: 'Memecoins'
        },
        groups: ['BTC', 'Stablecoins', 'L1/L0 majors', 'L2/Scaling', 'Memecoins', 'Others']
      })
    });

    await forceReloadTaxonomy();

    expect(getAssetGroup('HYPE5')).toBe('L1/L0 majors');
    expect(getAssetGroup('ARB5')).toBe('L2/Scaling');
    expect(getAssetGroup('USD1')).toBe('Stablecoins');
    expect(getAssetGroup('TRUMP2')).toBe('Memecoins');
    expect(getAssetGroup('A2')).toBe('Others');
  });

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

  test('clears a previous rebalance suggestion when verified targets are unavailable', async () => {
    localStorage.setItem('unified_suggested_allocation', JSON.stringify({
      targets: { Stablecoins: 80, BTC: 20 },
      timestamp: new Date().toISOString()
    }));

    const html = await renderAllocationBlock({ targets_by_group: {} });

    expect(localStorage.getItem('unified_suggested_allocation')).toBeNull();
    expect(html).toContain('Allocation targets are unavailable');
  });

  test('does not reuse a blended score when one current decision input is missing', () => {
    store.setState({
      ...store.snapshot(),
      cycle: { ...store.snapshot().cycle, ccsStar: null },
      scores: { ...store.snapshot().scores, onchain: 55, risk: 71, blended: 31 }
    }, 'test-incomplete-decision');

    const result = proposeTargets('blend');

    expect(result.available).toBe(false);
    expect(result.targets).toBeNull();
    expect(result.error).toContain('Complete decision inputs are unavailable');
  });

  test('proposes blended targets when all current decision inputs are numeric', () => {
    store.setState({
      ...store.snapshot(),
      cycle: { ...store.snapshot().cycle, ccsStar: 24 },
      scores: { ...store.snapshot().scores, onchain: 55, risk: 71, blended: 43 }
    }, 'test-complete-decision');

    const result = proposeTargets('blend');

    expect(result.available).toBe(true);
    expect(result.targets).not.toBeNull();
    expect(Object.values(result.targets)
      .filter(Number.isFinite)
      .reduce((sum, value) => sum + value, 0)).toBeCloseTo(100, 8);
  });

  test('renders Targets as unavailable without applying an incomplete proposal', async () => {
    document.body.innerHTML = '<div id="targets-content"></div>';
    window.store = store;
    window.loadScoresFromStore = jest.fn().mockResolvedValue(undefined);
    window.updateRiskDashboardBadges = jest.fn();
    window.loadBalanceData = jest.fn().mockResolvedValue({
      success: true,
      data: { items: [{ symbol: 'BTC', balance: 1, value_usd: 100 }] }
    });
    global.fetch = jest.fn().mockResolvedValue({ ok: false });

    store.setState({
      ...store.snapshot(),
      ccs: { ...store.snapshot().ccs, score: null },
      cycle: { ...store.snapshot().cycle, ccsStar: null, multipliers: null },
      scores: { ...store.snapshot().scores, onchain: 55, risk: 71, blended: null },
      targets: { proposed: null, strategy: null }
    }, 'test-targets-unavailable');

    await expect(renderTargetsContent()).resolves.toBeUndefined();

    const container = document.getElementById('targets-content');
    expect(container.textContent).toContain('Targets unavailable');
    expect(container.querySelector("button[onclick=\"applyStrategy('blend')\"]").disabled).toBe(true);
    expect(window.store.get('targets.proposed')).toBeNull();
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

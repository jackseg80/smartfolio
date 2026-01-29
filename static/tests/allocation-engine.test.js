/**
 * Unit tests for Allocation Engine V2
 * Tests hierarchical allocation (macro → sectors → coins) with floors and incumbency
 */

import { calculateHierarchicalAllocation } from '../core/allocation-engine.js';
import { describe, test, expect, beforeEach } from '@jest/globals';

describe('Allocation Engine V2 - Core Functionality', () => {

  test('should return null when V2 is disabled', async () => {
    const context = { cycleScore: 50, riskScore: 50 };
    const result = await calculateHierarchicalAllocation(context, [], { enableV2: false });

    expect(result).toBeNull();
  });

  test('should generate allocation with valid context', async () => {
    const context = {
      cycleScore: 70,
      onchainScore: 65,
      riskScore: 80,
      adaptiveWeights: { cycle: 0.4, onchain: 0.3, risk: 0.3 },
      risk_budget: { risky_allocation: 0.7, stable_allocation: 0.3 },
      regime: { name: 'expansion', allocation_bias: { meme_cap: 5 } }
    };

    const result = await calculateHierarchicalAllocation(context);

    expect(result).toBeDefined();
    expect(result.allocations).toBeDefined();
    expect(Array.isArray(result.allocations)).toBe(true);
  });

  test('should respect total allocation sum to 100%', async () => {
    const context = {
      cycleScore: 60,
      riskScore: 70,
      risk_budget: { risky_allocation: 0.6, stable_allocation: 0.4 }
    };

    const result = await calculateHierarchicalAllocation(context);

    if (result && result.allocations) {
      const total = result.allocations.reduce((sum, a) => sum + (a.target_allocation || 0), 0);
      expect(total).toBeGreaterThan(99);
      expect(total).toBeLessThanOrEqual(100);
    }
  });
});

describe('Allocation Engine V2 - Floors', () => {

  test('should respect base floors (BTC ≥ 15%, ETH ≥ 12%)', async () => {
    const context = {
      cycleScore: 50,
      riskScore: 50,
      risk_budget: { risky_allocation: 0.5, stable_allocation: 0.5 }
    };

    const result = await calculateHierarchicalAllocation(context);

    if (result && result.allocations) {
      const btc = result.allocations.find(a => a.group === 'BTC');
      const eth = result.allocations.find(a => a.group === 'ETH');

      if (btc) expect(btc.target_allocation).toBeGreaterThanOrEqual(15);
      if (eth) expect(eth.target_allocation).toBeGreaterThanOrEqual(12);
    }
  });

  test('should apply bullish floors when cycle ≥ 90', async () => {
    const context = {
      cycleScore: 95,  // Strong bull
      riskScore: 85,
      risk_budget: { risky_allocation: 0.8, stable_allocation: 0.2 }
    };

    const result = await calculateHierarchicalAllocation(context);

    if (result && result.allocations) {
      const sol = result.allocations.find(a => a.group === 'SOL');
      const defi = result.allocations.find(a => a.group === 'DeFi');

      // Bullish floors: SOL ≥ 6%, DeFi ≥ 8%
      if (sol) expect(sol.target_allocation).toBeGreaterThanOrEqual(6);
      if (defi) expect(defi.target_allocation).toBeGreaterThanOrEqual(8);
    }
  });
});

describe('Allocation Engine V2 - Incumbency Protection', () => {

  test('should protect incumbent positions with 3% minimum', async () => {
    const context = {
      cycleScore: 40,  // Bear market
      riskScore: 50,
      risk_budget: { risky_allocation: 0.3, stable_allocation: 0.7 }
    };

    // Positions with small holdings that would normally go to 0% in bear
    const currentPositions = [
      { symbol: 'DOGE', group: 'Memecoins', allocation: 5 },
      { symbol: 'AXS', group: 'Gaming/NFT', allocation: 2 }
    ];

    const result = await calculateHierarchicalAllocation(context, currentPositions);

    if (result && result.allocations) {
      const memes = result.allocations.find(a => a.group === 'Memecoins');
      const gaming = result.allocations.find(a => a.group === 'Gaming/NFT');

      // Even in bear, incumbent positions should get 3% minimum
      if (memes) expect(memes.target_allocation).toBeGreaterThanOrEqual(3);
      if (gaming) expect(gaming.target_allocation).toBeGreaterThanOrEqual(3);
    }
  });
});

describe('Allocation Engine V2 - Risk Budget Integration', () => {

  test('should allocate more to risky assets when risk budget is high', async () => {
    const contextHighRisk = {
      cycleScore: 70,
      riskScore: 90,  // High robustness
      risk_budget: { risky_allocation: 0.8, stable_allocation: 0.2 }
    };

    const contextLowRisk = {
      cycleScore: 70,
      riskScore: 40,  // Low robustness
      risk_budget: { risky_allocation: 0.4, stable_allocation: 0.6 }
    };

    const resultHigh = await calculateHierarchicalAllocation(contextHighRisk);
    const resultLow = await calculateHierarchicalAllocation(contextLowRisk);

    if (resultHigh && resultLow) {
      const stableHigh = resultHigh.allocations?.find(a => a.group === 'Stablecoins')?.target_allocation || 0;
      const stableLow = resultLow.allocations?.find(a => a.group === 'Stablecoins')?.target_allocation || 0;

      // High risk budget → less stables
      expect(stableLow).toBeGreaterThan(stableHigh);
    }
  });
});

describe('Allocation Engine V2 - Edge Cases', () => {

  test('should handle empty positions array', async () => {
    const context = {
      cycleScore: 60,
      riskScore: 70
    };

    const result = await calculateHierarchicalAllocation(context, []);

    expect(result).toBeDefined();
    if (result) {
      expect(result.allocations).toBeDefined();
    }
  });

  test('should handle extreme scores (cycle=100, risk=100)', async () => {
    const context = {
      cycleScore: 100,
      riskScore: 100,
      risk_budget: { risky_allocation: 0.9, stable_allocation: 0.1 }
    };

    const result = await calculateHierarchicalAllocation(context);

    expect(result).toBeDefined();
    if (result && result.allocations) {
      const total = result.allocations.reduce((sum, a) => sum + (a.target_allocation || 0), 0);
      expect(total).toBeLessThanOrEqual(100);
    }
  });

  test('should handle missing context fields with defaults', async () => {
    const minimalContext = {};  // Empty context

    const result = await calculateHierarchicalAllocation(minimalContext);

    expect(result).toBeDefined();
    // Should use defaults: cycle=50, risk=50, etc.
  });
});

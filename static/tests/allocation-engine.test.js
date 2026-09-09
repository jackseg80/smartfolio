/**
 * Unit tests for Allocation Engine V2
 * Tests hierarchical allocation (macro → sectors → coins) with explicit constraints
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
      onchainScore: 55,
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

describe('Allocation Engine V2 - No implicit floors', () => {

  test('does not apply category floors by default', async () => {
    const context = {
      cycleScore: 50,
      onchainScore: 50,
      riskScore: 50,
      risk_budget: { risky_allocation: 0.5, stable_allocation: 0.5 }
    };

    const result = await calculateHierarchicalAllocation(context);

    expect(result.metadata.floors_applied).toEqual({});
  });

  test('preserves an 80% stablecoin budget', async () => {
    const context = {
      cycleScore: 95,  // Strong bull
      onchainScore: 80,
      riskScore: 85,
      risk_budget: { risky_allocation: 0.2, stable_allocation: 0.8 }
    };

    const result = await calculateHierarchicalAllocation(context);

    const stables = result.allocations.find(a => a.group === 'Stablecoins');
    expect(stables.target_allocation).toBe(80);
  });
});

describe('Allocation Engine V2 - No implicit incumbency', () => {

  test('does not turn held coins into mandatory targets', async () => {
    const context = {
      cycleScore: 60,  // Neutral market (gives more room for alts)
      onchainScore: 55,
      riskScore: 60,
      risk_budget: { risky_allocation: 0.5, stable_allocation: 0.5 }
    };

    // Positions with small holdings that would normally go to 0% in bear
    const currentPositions = [
      { symbol: 'DOGE', group: 'Memecoins', allocation: 5, value_usd: 5000 },
      { symbol: 'AXS', group: 'Gaming/NFT', allocation: 2, value_usd: 2000 }
    ];

    const result = await calculateHierarchicalAllocation(context, currentPositions);

    expect(result.allocations.some(a => a.group === 'DOGE')).toBe(false);
    expect(result.allocations.some(a => a.group === 'AXS')).toBe(false);
  });
});

describe('Allocation Engine V2 - Risk Budget Integration', () => {

  test('should allocate more to risky assets when risk budget is high', async () => {
    const contextHighRisk = {
      cycleScore: 70,
      onchainScore: 75,
      riskScore: 90,  // High robustness
      risk_budget: { risky_allocation: 0.8, stable_allocation: 0.2 }
    };

    const contextLowRisk = {
      cycleScore: 70,
      onchainScore: 55,
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
      onchainScore: 60,
      riskScore: 70,
      risk_budget: { target_stables_pct: 30 }
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
      onchainScore: 100,
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

  test('should reject a missing stablecoin budget', async () => {
    const minimalContext = {};  // Empty context

    const result = await calculateHierarchicalAllocation(minimalContext);

    expect(result).toBeNull();
  });

  test('should reject a missing decision score', async () => {
    const result = await calculateHierarchicalAllocation({
      cycleScore: 60,
      riskScore: 70,
      risk_budget: { target_stables_pct: 30 }
    });

    expect(result).toBeNull();
  });
});

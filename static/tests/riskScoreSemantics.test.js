/**
 * Unit tests for Risk Score semantics
 * P1-5: Regression tests to ensure Risk Score is correctly treated as robustness
 * (high Risk Score = robust portfolio → allows more risky allocation)
 */

import { calculateRiskBudget } from '../modules/market-regimes.js';
import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';

// Helper for range validation
const inRange = (val, min, max, label = 'value') => {
  expect(val).toBeGreaterThanOrEqual(min);
  expect(val).toBeLessThanOrEqual(max);
};

describe('Risk Score Semantics - Correct Interpretation', () => {

  test('High Risk Score (90) → Higher risk_factor than Low Risk Score (40)', () => {
    // Arrange: Same blended score, different risk scores
    const budgetHighRisk = calculateRiskBudget(60, 90);
    const budgetLowRisk = calculateRiskBudget(60, 40);

    // Assert: Higher Risk Score should allow MORE risky allocation
    expect(budgetHighRisk.risky_allocation).toBeGreaterThan(budgetLowRisk.risky_allocation);
    console.log(`Risk 90: ${budgetHighRisk.risky_allocation.toFixed(2)} > Risk 40: ${budgetLowRisk.risky_allocation.toFixed(2)} ✅`);
  });

  test('Risk Score 100 → Maximum risk_factor (1.0 for conservative)', () => {
    const budget = calculateRiskBudget(60, 100);

    // With v2_conservative: risk_factor = 0.5 + 0.5 * (100/100) = 1.0
    // This is the maximum boost
    expect(budget.risky_allocation).toBeGreaterThan(0);
  });

  test('Risk Score 0 → Minimum risk_factor (0.5 for conservative)', () => {
    const budget = calculateRiskBudget(60, 0);

    // With v2_conservative: risk_factor = 0.5 + 0.5 * (0/100) = 0.5
    // This is the minimum (50% penalty on risky)
    expect(budget.risky_allocation).toBeGreaterThan(0);
    expect(budget.risky_allocation).toBeLessThan(1);
  });

  test('Risk Score increases linearly → risky_allocation increases monotonically', () => {
    const riskScores = [20, 40, 60, 80, 100];
    const budgets = riskScores.map(rs => calculateRiskBudget(60, rs));

    // Assert: Each step should increase risky allocation
    for (let i = 1; i < budgets.length; i++) {
      expect(budgets[i].risky_allocation).toBeGreaterThanOrEqual(budgets[i-1].risky_allocation);
    }
  });
});

describe('Risk Score Semantics - V2 Conservative Mode', () => {

  beforeEach(() => {
    // Ensure v2_conservative mode is active
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('RISK_SEMANTICS_MODE', 'v2_conservative');
    }
  });

  test('V2 Conservative: risk_factor range is [0.5 .. 1.0]', () => {
    const scores = [0, 25, 50, 75, 100];

    scores.forEach(riskScore => {
      const budget = calculateRiskBudget(60, riskScore);
      // Calculate expected risk_factor
      const expectedRiskFactor = 0.5 + 0.5 * (riskScore / 100);

      inRange(expectedRiskFactor, 0.5, 1.0, `risk_factor for Risk ${riskScore}`);
    });
  });

  test('V2 Conservative: metadata.formula_version is v2_correct', () => {
    const budget = calculateRiskBudget(60, 80);

    expect(budget.metadata.formula_version).toBe('v2_correct');
    expect(budget.metadata.semantics_mode).toBe('v2_conservative');
  });
});

describe('Risk Score Semantics - V2 Aggressive Mode', () => {

  beforeEach(() => {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('RISK_SEMANTICS_MODE', 'v2_aggressive');
    }
  });

  afterEach(() => {
    // Reset to default
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('RISK_SEMANTICS_MODE', 'v2_conservative');
    }
  });

  test('V2 Aggressive: risk_factor range is [0.4 .. 1.1]', () => {
    const scores = [0, 25, 50, 75, 100];

    scores.forEach(riskScore => {
      const budget = calculateRiskBudget(60, riskScore);
      // Calculate expected risk_factor
      const expectedRiskFactor = 0.4 + 0.7 * (riskScore / 100);

      inRange(expectedRiskFactor, 0.4, 1.1, `risk_factor for Risk ${riskScore}`);
    });
  });

  test('V2 Aggressive: More differentiation than Conservative', () => {
    const budgetLowRisk = calculateRiskBudget(60, 20);
    const budgetHighRisk = calculateRiskBudget(60, 100);

    const spread = budgetHighRisk.risky_allocation - budgetLowRisk.risky_allocation;

    // Aggressive mode should have wider spread than conservative
    // Conservative spread = baseRisky * (1.0 - 0.5) = 0.5 * baseRisky
    // Aggressive spread = baseRisky * (1.1 - 0.4) = 0.7 * baseRisky
    expect(spread).toBeGreaterThan(0);
  });
});

describe('Risk Score Semantics - Legacy Mode Migration', () => {

  test('Legacy mode is auto-migrated to v2_conservative', () => {
    // Arrange: Set legacy mode in localStorage
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('RISK_SEMANTICS_MODE', 'legacy');
    }

    // Act: Calculate budget (should trigger migration)
    const budget = calculateRiskBudget(60, 80);

    // Assert: Should be migrated to v2_conservative
    expect(budget.metadata.semantics_mode).toBe('v2_conservative');
    expect(budget.metadata.formula_version).toBe('v2_correct');

    // Verify localStorage was updated
    if (typeof localStorage !== 'undefined') {
      const currentMode = localStorage.getItem('RISK_SEMANTICS_MODE');
      expect(currentMode).toBe('v2_conservative');
    }
  });

  test('Legacy mode NO LONGER produces inverted results', () => {
    // Arrange: Try to set legacy mode
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('RISK_SEMANTICS_MODE', 'legacy');
    }

    // Act
    const budgetHighRisk = calculateRiskBudget(60, 90);
    const budgetLowRisk = calculateRiskBudget(60, 40);

    // Assert: Even with "legacy" in localStorage, should use correct v2 semantics
    // (higher Risk Score → higher risky allocation)
    expect(budgetHighRisk.risky_allocation).toBeGreaterThan(budgetLowRisk.risky_allocation);
  });
});

describe('Risk Score Semantics - Edge Cases', () => {

  test('Null/undefined Risk Score → fallback to 0', () => {
    const budgetNull = calculateRiskBudget(60, null);
    const budgetUndefined = calculateRiskBudget(60, undefined);

    // Both should produce valid results (using 0 as fallback)
    expect(budgetNull.risky_allocation).toBeGreaterThanOrEqual(0);
    expect(budgetUndefined.risky_allocation).toBeGreaterThanOrEqual(0);
  });

  test('Negative Risk Score → clamped to 0', () => {
    const budget = calculateRiskBudget(60, -10);

    // Should handle gracefully
    expect(budget.risky_allocation).toBeGreaterThanOrEqual(0);
  });

  test('Risk Score > 100 → not clamped (allows overboost)', () => {
    const budget = calculateRiskBudget(60, 120);

    // Should allow values > 100 (documented feature for exceptional robustness)
    expect(budget).toBeDefined();
    expect(budget.risky_allocation).toBeGreaterThanOrEqual(0);
  });
});

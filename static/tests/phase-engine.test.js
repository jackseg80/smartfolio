/**
 * Unit tests for Phase Engine
 * Tests phase inference, memory state, and phase tilts
 */

import {
  inferPhase,
  getPhaseMemoryState,
  resetPhaseMemory,
  forcePhase,
  clearForcePhase,
  getCurrentForce,
  applyPhaseTilts
} from '../core/phase-engine.js';
import { describe, test, expect, beforeEach } from '@jest/globals';

describe('Phase Engine - Phase Inference', () => {

  beforeEach(() => {
    // Reset phase memory before each test
    resetPhaseMemory();
    clearForcePhase();
  });

  test('should infer risk-off or neutral phase for low cycle score', () => {
    const phaseInputs = {
      DI: 25,  // Low DI suggests risk-off
      btc_dom: [60, 60],
      eth_btc: [0.05, 0.05],
      alts_btc: [0.3, 0.3],
      breadth_alts: 0.3,
      dispersion: 0.4,
      corr_alts_btc: 0.7
    };

    const result = inferPhase(phaseInputs);

    expect(result).toBeDefined();
    expect(typeof result).toBe('string');
    expect(['risk_off', 'neutral']).toContain(result);
  });

  test('should infer bullish phase for high DI and good breadth', () => {
    const phaseInputs = {
      DI: 75,  // High DI suggests bull market
      btc_dom: [50, 48],  // BTC dominance declining
      eth_btc: [0.06, 0.065],  // ETH gaining
      alts_btc: [0.4, 0.45],  // Alts gaining
      breadth_alts: 0.70,  // Good breadth
      dispersion: 0.65,
      corr_alts_btc: 0.25
    };

    // Call multiple times to build hysteresis consensus
    inferPhase(phaseInputs);
    inferPhase(phaseInputs);
    const result = inferPhase(phaseInputs);

    expect(result).toBeDefined();
    expect(typeof result).toBe('string');
    // With hysteresis, might return neutral until consensus is reached
    expect(['largecap_altseason', 'eth_expansion', 'full_altseason', 'neutral']).toContain(result);
  });

  test('should infer altseason for extreme conditions', () => {
    const phaseInputs = {
      DI: 80,
      btc_dom: [55, 50],  // BTC dom declining
      eth_btc: [0.06, 0.07],
      alts_btc: [0.4, 0.50],  // Alts surging
      breadth_alts: 0.80,  // Excellent breadth
      dispersion: 0.80,  // High dispersion
      corr_alts_btc: 0.20  // Low correlation
    };

    // Call multiple times to build hysteresis consensus
    inferPhase(phaseInputs);
    inferPhase(phaseInputs);
    const result = inferPhase(phaseInputs);

    expect(result).toBeDefined();
    expect(typeof result).toBe('string');
    // With hysteresis, might return neutral until consensus is reached
    expect(['full_altseason', 'largecap_altseason', 'neutral']).toContain(result);
  });

  test('should handle partial inputs with fallback logic', () => {
    const partialInputs = {
      partial: true,
      DI: 55,
      breadth_alts: 0.5,
      missing: ['btc_dom', 'eth_btc']
    };

    const result = inferPhase(partialInputs);

    expect(result).toBeDefined();
    expect(typeof result).toBe('string');
    // With partial data and DI=55, breadth=0.5, should return neutral or a cautious phase
    expect(['neutral', 'eth_expansion', 'risk_off']).toContain(result);
  });
});

describe('Phase Engine - Memory State', () => {

  beforeEach(() => {
    resetPhaseMemory();
  });

  test('should initialize with empty memory', () => {
    const state = getPhaseMemoryState();

    expect(state).toBeDefined();
    expect(state.history).toBeDefined();
    expect(Array.isArray(state.history)).toBe(true);
  });

  test('should accumulate phase history', () => {
    const phaseInputs1 = { cycle_score: 30 };
    const phaseInputs2 = { cycle_score: 40 };

    inferPhase(phaseInputs1);
    inferPhase(phaseInputs2);

    const state = getPhaseMemoryState();

    expect(state.history.length).toBeGreaterThan(0);
  });

  test('should reset memory correctly', () => {
    const phaseInputs = { cycle_score: 50 };
    inferPhase(phaseInputs);

    const stateBefore = getPhaseMemoryState();
    expect(stateBefore.history.length).toBeGreaterThan(0);

    resetPhaseMemory();

    const stateAfter = getPhaseMemoryState();
    expect(stateAfter.history).toHaveLength(0);
  });
});

describe('Phase Engine - Force Phase', () => {

  beforeEach(() => {
    clearForcePhase();
  });

  test('should force risk_off phase', () => {
    forcePhase('risk_off');

    const force = getCurrentForce();

    expect(force).toBeDefined();
    expect(force).toBe('risk_off');
  });

  test('should force eth_expansion phase', () => {
    forcePhase('eth_expansion');

    const force = getCurrentForce();

    expect(force).toBeDefined();
    expect(force).toBe('eth_expansion');
  });

  test('should override inference when forced', () => {
    forcePhase('full_altseason');

    const phaseInputs = {
      DI: 30,  // Would normally infer risk_off
      btc_dom: [60, 60],
      eth_btc: [0.05, 0.05],
      alts_btc: [0.3, 0.3],
      breadth_alts: 0.3,
      dispersion: 0.4,
      corr_alts_btc: 0.7
    };
    const result = inferPhase(phaseInputs);

    // Should return forced phase (string), not inferred
    expect(result).toBe('full_altseason');
  });

  test('should clear force correctly', () => {
    forcePhase('markup');
    expect(getCurrentForce()).toBeDefined();

    clearForcePhase();
    expect(getCurrentForce()).toBeNull();
  });
});

describe('Phase Engine - Phase Tilts', () => {

  test('should return targets unchanged for neutral phase', async () => {
    const targets = {
      'BTC': 30,
      'ETH': 25,
      'Stablecoins': 20,
      'Memecoins': 5,
      'Others': 20
    };

    const ctx = { DI: 50, breadth_alts: 0.5 };

    const result = await applyPhaseTilts(targets, 'neutral', ctx);

    expect(result).toBeDefined();
    expect(result.targets).toBeDefined();
    expect(result.metadata).toBeDefined();

    // In neutral, targets should be returned unchanged
    expect(result.targets).toEqual(targets);
  });

  test('should apply tilts for full altseason', async () => {
    const targets = {
      'BTC': 35,
      'ETH': 25,
      'SOL': 5,
      'Memecoins': 2,
      'Stablecoins': 20,
      'Others': 13
    };

    const ctx = {
      DI: 80,
      breadth_alts: 0.80
    };

    const result = await applyPhaseTilts(targets, 'full_altseason', ctx);

    expect(result).toBeDefined();
    expect(result.targets).toBeDefined();
    expect(result.metadata).toBeDefined();

    // Total should still be 100%
    const total = Object.values(result.targets).reduce((sum, val) => sum + val, 0);
    expect(total).toBeCloseTo(100, 1);
  });

  test('should return targets unchanged for risk_off phase', async () => {
    const targets = {
      'BTC': 30,
      'ETH': 20,
      'Stablecoins': 15,
      'Memecoins': 10,
      'Others': 25
    };

    const ctx = {
      DI: 25,
      breadth_alts: 0.3
    };

    const result = await applyPhaseTilts(targets, 'risk_off', ctx);

    expect(result).toBeDefined();
    expect(result.targets).toBeDefined();

    // In risk_off, targets should be returned unchanged
    expect(result.targets).toEqual(targets);
  });

  test('should preserve total allocation at 100%', async () => {
    const targets = {
      'BTC': 35,
      'ETH': 25,
      'Stablecoins': 20,
      'SOL': 5,
      'Others': 15
    };

    const result = await applyPhaseTilts(targets, 'eth_expansion', { DI: 70, breadth_alts: 0.65 });

    expect(result).toBeDefined();
    expect(result.targets).toBeDefined();

    if (result.targets && Object.keys(result.targets).length > 0) {
      const total = Object.values(result.targets).reduce((sum, val) => sum + val, 0);
      expect(total).toBeCloseTo(100, 1);  // Allow 0.1% tolerance for rounding
    }
  });
});

describe('Phase Engine - Edge Cases', () => {

  test('should handle extreme DI values (very low and very high)', () => {
    const lowInputs = {
      DI: 10,  // Very low - extreme risk-off
      btc_dom: [60, 60],
      eth_btc: [0.05, 0.05],
      alts_btc: [0.3, 0.3],
      breadth_alts: 0.2,
      dispersion: 0.3,
      corr_alts_btc: 0.8
    };
    const highInputs = {
      DI: 90,  // Very high - extreme bull
      btc_dom: [50, 45],
      eth_btc: [0.06, 0.08],
      alts_btc: [0.4, 0.55],
      breadth_alts: 0.85,
      dispersion: 0.85,
      corr_alts_btc: 0.15
    };

    // DI < 35 triggers immediate risk_off (emergency exit, bypasses hysteresis)
    const resultLow = inferPhase(lowInputs);

    // Build consensus for high inputs
    inferPhase(highInputs);
    inferPhase(highInputs);
    const resultHigh = inferPhase(highInputs);

    expect(resultLow).toBeDefined();
    expect(resultHigh).toBeDefined();
    expect(typeof resultLow).toBe('string');
    expect(typeof resultHigh).toBe('string');
    expect(resultLow).toBe('risk_off');  // Emergency exit
    expect(['full_altseason', 'largecap_altseason', 'neutral']).toContain(resultHigh);
  });

  test('should handle null/undefined inputs gracefully', () => {
    expect(() => inferPhase(null)).not.toThrow();
    expect(() => inferPhase(undefined)).not.toThrow();
    expect(() => inferPhase({})).not.toThrow();
  });
});

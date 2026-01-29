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

  test('should infer accumulation phase for low cycle score', () => {
    const phaseInputs = {
      cycle_score: 30,
      halving_days: 100,
      dominance_btc: 60,
      mvrv: 1.2
    };

    const result = inferPhase(phaseInputs);

    expect(result).toBeDefined();
    expect(result.phase).toBe('accumulation');
    expect(result.confidence).toBeGreaterThan(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });

  test('should infer markup phase for high cycle score', () => {
    const phaseInputs = {
      cycle_score: 80,
      halving_days: 400,
      dominance_btc: 45,
      mvrv: 2.5
    };

    const result = inferPhase(phaseInputs);

    expect(result).toBeDefined();
    expect(['markup', 'distribution']).toContain(result.phase);
  });

  test('should infer distribution phase for extreme cycle score', () => {
    const phaseInputs = {
      cycle_score: 95,
      halving_days: 500,
      dominance_btc: 35,
      mvrv: 3.5
    };

    const result = inferPhase(phaseInputs);

    expect(result).toBeDefined();
    expect(result.phase).toBe('distribution');
  });

  test('should handle missing inputs with defaults', () => {
    const minimalInputs = {
      cycle_score: 60
    };

    const result = inferPhase(minimalInputs);

    expect(result).toBeDefined();
    expect(result.phase).toBeDefined();
    expect(['accumulation', 'markup', 'distribution']).toContain(result.phase);
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

  test('should force accumulation phase', () => {
    forcePhase('accumulation');

    const force = getCurrentForce();

    expect(force).toBeDefined();
    expect(force.phase).toBe('accumulation');
  });

  test('should force markup phase', () => {
    forcePhase('markup');

    const force = getCurrentForce();

    expect(force).toBeDefined();
    expect(force.phase).toBe('markup');
  });

  test('should override inference when forced', () => {
    forcePhase('distribution');

    const phaseInputs = { cycle_score: 30 };  // Would normally infer accumulation
    const result = inferPhase(phaseInputs);

    // Should return forced phase, not inferred
    expect(result.phase).toBe('distribution');
    expect(result.forced).toBe(true);
  });

  test('should clear force correctly', () => {
    forcePhase('markup');
    expect(getCurrentForce()).toBeDefined();

    clearForcePhase();
    expect(getCurrentForce()).toBeNull();
  });
});

describe('Phase Engine - Phase Tilts', () => {

  test('should apply tilts for accumulation phase', async () => {
    const targets = {
      'BTC': 30,
      'ETH': 25,
      'Stablecoins': 20,
      'Memecoins': 5,
      'Others': 20
    };

    const ctx = {
      phase: 'accumulation',
      phaseConfidence: 0.8
    };

    const result = await applyPhaseTilts(targets, 'accumulation', ctx);

    expect(result).toBeDefined();
    expect(result.BTC).toBeDefined();
    expect(result.ETH).toBeDefined();

    // In accumulation, BTC should be boosted
    expect(result.BTC).toBeGreaterThanOrEqual(targets.BTC);
  });

  test('should apply tilts for markup phase (altseason)', async () => {
    const targets = {
      'BTC': 35,
      'ETH': 25,
      'SOL': 5,
      'Memecoins': 2,
      'Stablecoins': 20,
      'Others': 13
    };

    const ctx = {
      phase: 'markup',
      phaseConfidence: 0.9,
      altseasonSignal: true
    };

    const result = await applyPhaseTilts(targets, 'markup', ctx);

    expect(result).toBeDefined();

    // In markup with altseason, alts should be boosted
    // Memecoins might get a tilt
    expect(result.Memecoins).toBeGreaterThanOrEqual(targets.Memecoins);
  });

  test('should apply tilts for distribution phase (risk-off)', async () => {
    const targets = {
      'BTC': 30,
      'ETH': 20,
      'Stablecoins': 15,
      'Memecoins': 10,
      'Others': 25
    };

    const ctx = {
      phase: 'distribution',
      phaseConfidence: 0.85
    };

    const result = await applyPhaseTilts(targets, 'distribution', ctx);

    expect(result).toBeDefined();

    // In distribution, stables should be increased
    expect(result.Stablecoins).toBeGreaterThanOrEqual(targets.Stablecoins);
  });

  test('should preserve total allocation at 100%', async () => {
    const targets = {
      'BTC': 35,
      'ETH': 25,
      'Stablecoins': 20,
      'SOL': 5,
      'Others': 15
    };

    const result = await applyPhaseTilts(targets, 'markup');

    if (result) {
      const total = Object.values(result).reduce((sum, val) => sum + val, 0);
      expect(total).toBeCloseTo(100, 1);  // Allow 0.1% tolerance for rounding
    }
  });
});

describe('Phase Engine - Edge Cases', () => {

  test('should handle extreme cycle scores (0 and 100)', () => {
    const lowInputs = { cycle_score: 0 };
    const highInputs = { cycle_score: 100 };

    const resultLow = inferPhase(lowInputs);
    const resultHigh = inferPhase(highInputs);

    expect(resultLow).toBeDefined();
    expect(resultHigh).toBeDefined();
    expect(resultLow.phase).toBe('accumulation');
    expect(resultHigh.phase).toBe('distribution');
  });

  test('should handle null/undefined inputs gracefully', () => {
    expect(() => inferPhase(null)).not.toThrow();
    expect(() => inferPhase(undefined)).not.toThrow();
    expect(() => inferPhase({})).not.toThrow();
  });
});

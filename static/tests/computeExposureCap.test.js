/**
 * Unit tests for computeExposureCap() function
 * Tests exposure cap calculation across different market regimes and conditions
 */

import { computeExposureCap } from '../modules/targets-coordinator.js';
import { describe, test, expect } from '@jest/globals';

// Helper for range validation
const inRange = (val, min, max, label = 'value') => {
  expect(val).toBeGreaterThanOrEqual(min);
  expect(val).toBeLessThanOrEqual(max);
};

describe('computeExposureCap - Core Functionality', () => {

  test('Expansion + Risk 90 → boost actif, cap ≥ 65%', () => {
    const cap = computeExposureCap({
      blendedScore: 65,
      riskScore: 90,
      decision_score: 0.6,
      confidence: 0.7,
      volatility: 0.30,
      regime: 'expansion',
      backendStatus: 'ok'
    });

    inRange(cap, 65, 75, 'Expansion with Risk 90 cap');
  });

  test('Legacy Euphorie normalizes to canonical Bull Market', () => {
    const cap = computeExposureCap({
      blendedScore: 73,
      riskScore: 90,
      decision_score: 0.75,
      confidence: 0.8,
      volatility: 0.30,
      regime: 'euphorie',
      backendStatus: 'ok'
    });

    inRange(cap, 60, 85, 'Bull Market cap');
  });

  test('Legacy Euphorie + low Risk respects the Bull Market maximum', () => {
    const cap = computeExposureCap({
      blendedScore: 73,
      riskScore: 60,
      decision_score: 0.4,
      confidence: 0.5,
      volatility: 0.35,
      regime: 'euphorie',
      backendStatus: 'ok'
    });

    expect(cap).toBeLessThanOrEqual(85);
  });

  test('Legacy Bear normalizes to canonical Bear Market bounds', () => {
    const cap = computeExposureCap({
      blendedScore: 30,
      riskScore: 40,
      decision_score: 0.3,
      confidence: 0.4,
      volatility: 0.45,
      regime: 'bear',
      backendStatus: 'ok'
    });

    inRange(cap, 20, 40, 'Bear market cap');
  });

  test('Legacy Neutral normalizes to canonical Correction bounds', () => {
    const cap = computeExposureCap({
      blendedScore: 55,
      riskScore: 60,
      decision_score: 0.5,
      confidence: 0.6,
      volatility: 0.25,
      regime: 'neutral',
      backendStatus: 'ok'
    });

    inRange(cap, 40, 70, 'Correction regime cap');
  });
});

describe('computeExposureCap - Backend Status Handling', () => {

  test('Backend error reduces the cap without forcing an Expansion floor', () => {
    const params = {
      blendedScore: 70,
      riskScore: 85,
      decision_score: 0.5,
      confidence: 0.5,
      volatility: 0.32,
      regime: 'expansion',
    };
    const cap = computeExposureCap({ ...params, backendStatus: 'error' });
    const healthyCap = computeExposureCap({ ...params, backendStatus: 'ok' });

    expect(cap).toBeLessThan(healthyCap);
  });

  test('Backend stale applies the full 15 point degradation', () => {
    const cap = computeExposureCap({
      blendedScore: 68,
      riskScore: 90,
      decision_score: 0.55,
      confidence: 0.7,
      volatility: 0.30,
      regime: 'expansion',
      backendStatus: 'stale'
    });

    expect(cap).toBe(56);
  });

  test('Backend status unknown → pas de pénalité appliquée', () => {
    const cap = computeExposureCap({
      blendedScore: 70,
      riskScore: 85,
      decision_score: 0.6,
      confidence: 0.7,
      volatility: 0.30,
      regime: 'expansion',
      backendStatus: 'unknown'
    });

    expect(cap).toBeGreaterThanOrEqual(60);
  });
});

describe('computeExposureCap - Volatility Normalization', () => {

  test('Vol unité mixte: 32 (percent) ≡ 0.32 (decimal) → résultats identiques', () => {
    const params = {
      blendedScore: 70,
      riskScore: 85,
      decision_score: 0.6,
      confidence: 0.7,
      regime: 'expansion',
      backendStatus: 'ok'
    };

    const capDecimal = computeExposureCap({ ...params, volatility: 0.32 });
    const capPercent = computeExposureCap({ ...params, volatility: 32 });

    expect(capDecimal).toBe(capPercent);
  });

  test('Haute volatilité (45%) → pénalité max 10pts appliquée', () => {
    const capLowVol = computeExposureCap({
      blendedScore: 70,
      riskScore: 85,
      decision_score: 0.6,
      confidence: 0.7,
      volatility: 0.20,  // Volatilité de référence
      regime: 'expansion',
      backendStatus: 'ok'
    });

    const capHighVol = computeExposureCap({
      blendedScore: 70,
      riskScore: 85,
      decision_score: 0.6,
      confidence: 0.7,
      volatility: 0.45,  // Volatilité élevée
      regime: 'expansion',
      backendStatus: 'ok'
    });

    // Écart doit être ≤ 10pts (pénalité max)
    expect(capLowVol - capHighVol).toBeLessThanOrEqual(10);
    expect(capLowVol - capHighVol).toBeGreaterThanOrEqual(0);
  });
});

describe('computeExposureCap - Edge Cases', () => {

  test('Unknown regime uses only the generic maximum', () => {
    const cap = computeExposureCap({
      blendedScore: 50,
      riskScore: 70,
      decision_score: 0.5,
      confidence: 0.6,
      volatility: 0.25,
      regime: 'unknown_regime',
      backendStatus: 'ok'
    });

    expect(cap).toBeGreaterThanOrEqual(40);
  });

  test('Missing scores are rejected instead of replaced with zeroes', () => {
    expect(() => computeExposureCap({
      blendedScore: null,
      riskScore: undefined,
      decision_score: null,
      confidence: undefined,
      volatility: null,
      regime: 'expansion',
      backendStatus: 'ok'
    })).toThrow('Exposure cap inputs unavailable');
  });

  test('Regime name avec casse mixte → normalisé correctement', () => {
    const capLower = computeExposureCap({
      blendedScore: 73,
      riskScore: 90,
      decision_score: 0.7,
      confidence: 0.8,
      volatility: 0.30,
      regime: 'euphorie',
      backendStatus: 'ok'
    });

    const capUpper = computeExposureCap({
      blendedScore: 73,
      riskScore: 90,
      decision_score: 0.7,
      confidence: 0.8,
      volatility: 0.30,
      regime: 'EUPHORIE',
      backendStatus: 'ok'
    });

    expect(capLower).toBe(capUpper);
  });

  test('Decision score très faible (0.1) → pénalité signal appliquée', () => {
    const capGoodSignal = computeExposureCap({
      blendedScore: 70,
      riskScore: 85,
      decision_score: 0.8,
      confidence: 0.8,
      volatility: 0.30,
      regime: 'expansion',
      backendStatus: 'ok'
    });

    const capBadSignal = computeExposureCap({
      blendedScore: 70,
      riskScore: 85,
      decision_score: 0.2,
      confidence: 0.5,
      volatility: 0.30,
      regime: 'expansion',
      backendStatus: 'ok'
    });

    // Bad signal devrait avoir un cap plus bas
    expect(capBadSignal).toBeLessThan(capGoodSignal);
    expect(capBadSignal).toBeGreaterThanOrEqual(0);
  });
});

describe('computeExposureCap - No implicit regime floors', () => {

  test('A weak signal can remain below the former regime minimums', () => {
    const regimes = [
      { name: 'expansion', maximum: 95 },
      { name: 'bull market', maximum: 85 },
      { name: 'correction', maximum: 70 },
      { name: 'bear market', maximum: 40 },
    ];

    regimes.forEach(({ name, maximum }) => {
      const cap = computeExposureCap({
        blendedScore: 50,
        riskScore: 50,
        decision_score: 0.3,
        confidence: 0.4,
        volatility: 0.40,  // Pénalités élevées pour tester floor
        regime: name,
        backendStatus: 'stale'  // Pénalité backend pour tester floor
      });

      expect(cap).toBe(22);
      expect(cap).toBeLessThanOrEqual(maximum);
    });
  });
});

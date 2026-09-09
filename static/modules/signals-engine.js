/**
 * CCS Signals Engine - MVP Simple Version
 * Fetches market signals and computes CCS score
 */

import { fetchCached } from '../core/fetcher.js';

// Default CCS weights (model version 1)
export const DEFAULT_CCS_WEIGHTS = {
  fear_greed: 0.25,      // Fear & Greed Index
  btc_dominance: 0.20,   // Bitcoin Dominance
  funding_rate: 0.15,    // Futures funding rates
  eth_btc_ratio: 0.15,   // ETH/BTC strength
  volatility: 0.10,      // Market volatility
  trend: 0.15,           // Price trend momentum
  model_version: 'ccs-1'
};

function unavailableSignal(error) {
  return {
    value: null,
    normalized: null,
    timestamp: Date.now(),
    source: 'unavailable',
    error: error instanceof Error ? error.message : String(error)
  };
}

/**
 * Fetch market signals from multiple sources
 */
export async function fetchSignals() {
  console.debug('🔍 Fetching REAL market signals...');

  const signals = {};

  try {
    // 1. Fear & Greed Index (Alternative.me API)
    try {
      const fearGreedResponse = await fetch('https://api.alternative.me/fng/', {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });

      if (fearGreedResponse.ok) {
        const fearGreedData = await fearGreedResponse.json();
        const fearGreedValue = parseInt(fearGreedData.data[0].value);
        signals.fear_greed = {
          value: fearGreedValue,
          normalized: null,
          timestamp: Date.now(),
          source: 'alternative.me'
        };
        console.debug('✅ Fear & Greed loaded:', fearGreedValue);
      } else {
        throw new Error('Fear & Greed API failed');
      }
    } catch (error) {
      (window.debugLogger?.warn || console.warn)('⚠️ Fear & Greed fallback:', error);
      signals.fear_greed = unavailableSignal(error);
    }

    // 2. BTC Dominance (CoinGecko via proxy)
    try {
      const apiBase = window.getApiBase();
      const proxyUrl = `${apiBase}/api/coingecko-proxy/global`;
      const activeUser = localStorage.getItem('activeUser');

      const dominanceResponse = await fetch(proxyUrl, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'X-User': activeUser
        }
      });

      if (dominanceResponse.ok) {
        const cgData = await dominanceResponse.json();
        // Proxy returns CoinGecko data directly (no wrapper)
        const btcDominance = cgData.data.market_cap_percentage.btc;

        signals.btc_dominance = {
          value: btcDominance,
          normalized: null,
          timestamp: Date.now(),
          source: 'coingecko_proxy'
        };

        console.debug(`✅ BTC Dominance loaded: ${btcDominance.toFixed(1)}%`);
      } else {
        throw new Error(`CoinGecko proxy failed: ${dominanceResponse.status}`);
      }
    } catch (error) {
      (window.debugLogger?.warn || console.warn)('⚠️ BTC Dominance unavailable:', error);
      signals.btc_dominance = unavailableSignal(error);
    }

    // 3. Funding Rate (Binance API)
    try {
      const fundingResponse = await fetch('https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT', {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });

      if (fundingResponse.ok) {
        const fundingData = await fundingResponse.json();
        const fundingRate = parseFloat(fundingData.lastFundingRate); // Already in decimal (e.g., 0.0001 = 0.01%)
        signals.funding_rate = {
          value: fundingRate,
          normalized: null,
          timestamp: Date.now(),
          source: 'binance'
        };
        console.debug('✅ Funding Rate loaded:', (fundingRate * 100).toFixed(4) + '%');
      } else {
        throw new Error('Binance API failed');
      }
    } catch (error) {
      (window.debugLogger?.warn || console.warn)('⚠️ Funding Rate fallback:', error);
      signals.funding_rate = unavailableSignal(error);
    }

    // 4. ETH/BTC Ratio (CoinGecko via proxy)
    try {
      const apiBase = window.getApiBase();
      const proxyUrl = `${apiBase}/api/coingecko-proxy/simple/price?ids=bitcoin,ethereum&vs_currencies=usd`;
      const activeUser = localStorage.getItem('activeUser');

      const pricesResponse = await fetch(proxyUrl, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'X-User': activeUser
        }
      });

      if (pricesResponse.ok) {
        const cgData = await pricesResponse.json();
        // Proxy returns CoinGecko data directly (no wrapper)
        console.debug('🔍 ETH/BTC API response:', cgData);

        const btcPrice = cgData.bitcoin?.usd;
        const ethPrice = cgData.ethereum?.usd;

        if (btcPrice && ethPrice && btcPrice > 0 && ethPrice > 0) {
          const ethBtcRatio = ethPrice / btcPrice;

          signals.eth_btc_ratio = {
            value: ethBtcRatio,
            normalized: null,
            timestamp: Date.now(),
            source: 'coingecko_proxy'
          };

          console.debug(`✅ ETH/BTC Ratio loaded: ${ethBtcRatio.toFixed(6)}`);
        } else {
          throw new Error(`Invalid price data: BTC=${btcPrice}, ETH=${ethPrice}`);
        }
      } else {
        throw new Error(`CoinGecko proxy failed: ${pricesResponse.status}`);
      }
    } catch (error) {
      (window.debugLogger?.warn || console.warn)('⚠️ ETH/BTC Ratio unavailable:', error);
      signals.eth_btc_ratio = unavailableSignal(error);
    }

    // 5. Volatility (calculated from recent BTC price changes via proxy)
    try {
      const apiBase = window.getApiBase();
      const proxyUrl = `${apiBase}/api/coingecko-proxy/market_chart?coin_id=bitcoin&vs_currency=usd&days=7&interval=daily`;
      const activeUser = localStorage.getItem('activeUser');

      const volatilityResponse = await fetch(proxyUrl, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'X-User': activeUser
        }
      });

      if (volatilityResponse.ok) {
        const cgData = await volatilityResponse.json();
        // Proxy returns CoinGecko data directly (no wrapper)
        const prices = cgData.prices.map(p => p[1]);

        // Calculate 7-day volatility
        const returns = [];
        for (let i = 1; i < prices.length; i++) {
          returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
        }

        const volatility = Math.sqrt(returns.reduce((sum, ret) => sum + ret * ret, 0) / returns.length) * Math.sqrt(365); // Annualized volatility as decimal

        signals.volatility = {
          value: volatility,
          normalized: null,
          timestamp: Date.now(),
          source: 'coingecko_calculated_proxy'
        };

        console.debug(`✅ Volatility loaded: ${(volatility * 100).toFixed(1)}%`);
      } else {
        throw new Error(`CoinGecko proxy failed: ${volatilityResponse.status}`);
      }
    } catch (error) {
      (window.debugLogger?.warn || console.warn)('⚠️ Volatility unavailable:', error);
      signals.volatility = unavailableSignal(error);
    }

    // 6. Trend (7-day price momentum)
    try {
      // Use backend proxy to avoid CORS and rate limiting
      const apiBase = window.getApiBase();
      const proxyUrl = `${apiBase}/api/coingecko-proxy/bitcoin?market_data=true`;
      const activeUser = localStorage.getItem('activeUser');

      const trendResponse = await fetch(proxyUrl, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'X-User': activeUser
        }
      });

      if (trendResponse.ok) {
        const cgData = await trendResponse.json();
        // Proxy returns CoinGecko data directly (no wrapper)
        const priceChange7d = cgData.market_data.price_change_percentage_7d / 100; // Convert to decimal

        signals.trend = {
          value: priceChange7d,
          normalized: null,
          timestamp: Date.now(),
          source: 'coingecko_proxy'
        };

        console.debug(`✅ Trend loaded: ${(priceChange7d * 100).toFixed(2)}%`);
      } else {
        throw new Error(`CoinGecko proxy failed: ${trendResponse.status}`);
      }
    } catch (error) {
      (window.debugLogger?.warn || console.warn)('⚠️ Trend unavailable:', error);
      signals.trend = unavailableSignal(error);
    }

  } catch (globalError) {
    debugLogger.error('❌ Global error fetching signals:', globalError);
    return Object.fromEntries(
      Object.keys(DEFAULT_CCS_WEIGHTS)
        .filter(key => key !== 'model_version')
        .map(key => [key, unavailableSignal(globalError)])
    );
  }

  console.debug('🔍 Fetched REAL signals:', signals);
  return signals;
}

/**
 * Normalize individual signal to 0-100 scale
 */
function normalizeSignal(key, rawValue) {
  switch (key) {
    case 'fear_greed':
      // Already 0-100, just clamp
      return Math.max(0, Math.min(100, rawValue));

    case 'btc_dominance':
      // 40-70% dominance → 0-100 scale (higher dominance = lower CCS)
      const domNorm = Math.max(0, Math.min(100, (70 - rawValue) / (70 - 40) * 100));
      return domNorm;

    case 'funding_rate':
      // Negative funding (shorts pay longs) = bullish = higher CCS
      const fundNorm = Math.max(0, Math.min(100, 50 - (rawValue * 2000))); // -1% = 100, +1% = 0
      return fundNorm;

    case 'eth_btc_ratio':
      // Higher ETH/BTC = alt season = higher CCS
      // Handle edge case where rawValue might be 0 or very small
      if (rawValue <= 0) {
        return null;
      }
      const ethNorm = Math.max(0, Math.min(100, (rawValue - 0.025) / (0.06 - 0.025) * 100)); // Adjusted range: 0.025-0.06
      return ethNorm;

    case 'volatility':
      // Lower volatility = more mature market = potentially higher CCS
      const volNorm = Math.max(0, Math.min(100, 100 - (rawValue * 100)));
      return volNorm;

    case 'trend':
      // Positive trend = higher CCS
      const trendNorm = Math.max(0, Math.min(100, 50 + (rawValue * 250))); // -10% = 25, +10% = 75
      return trendNorm;

    default:
      return null;
  }
}

/**
 * Compute CCS score from signals and weights
 */
export function computeCCS(signals, weights = DEFAULT_CCS_WEIGHTS) {
  if (!signals || typeof signals !== 'object') {
    throw new Error('Invalid signals object');
  }

  if (!weights || typeof weights !== 'object') {
    throw new Error('Invalid weights object');
  }

  let weightedSum = 0;
  const normalizedSignals = {};
  const missingSignals = [];
  const requiredKeys = Object.keys(weights).filter(key => key !== 'model_version');

  for (const key of requiredKeys) {
    const signal = signals[key];
    const weight = weights[key];
    if (!Number.isFinite(weight) || weight <= 0 || !signal || !Number.isFinite(signal.value) || signal.source === 'unavailable') {
      missingSignals.push(key);
      continue;
    }

    // Normalize signal
    const normalized = normalizeSignal(key, signal.value);
    if (!Number.isFinite(normalized)) {
      missingSignals.push(key);
      continue;
    }
    normalizedSignals[key] = {
      ...signal,
      normalized,
      weight
    };

    // Add to weighted sum
    weightedSum += normalized * weight;
  }

  if (missingSignals.length > 0) {
    return {
      available: false,
      score: null,
      signals: normalizedSignals,
      weights,
      missing_signals: missingSignals,
      reason: 'Required CCS signals are unavailable',
      calculation_time: new Date().toISOString(),
      model_version: weights.model_version || 'ccs-1'
    };
  }

  const totalWeight = requiredKeys.reduce((sum, key) => sum + weights[key], 0);
  const ccsScore = weightedSum / totalWeight;

  // Validation
  if (isNaN(ccsScore) || ccsScore < 0 || ccsScore > 100) {
    throw new Error(`Invalid CCS score: ${ccsScore}`);
  }

  return {
    available: true,
    score: Math.round(ccsScore * 100) / 100, // Round to 2 decimals
    signals: normalizedSignals,
    weights,
    calculation_time: new Date().toISOString(),
    model_version: weights.model_version || 'ccs-1'
  };
}

/**
 * Fetch signals with cache and compute CCS
 */
export async function fetchAndComputeCCS(weights = DEFAULT_CCS_WEIGHTS) {
  try {
    // Fetch signals (cached)
    const signals = await fetchCached(
      'market-signals',
      () => fetchSignals(),
      'signals'
    );

    // Compute CCS
    const ccs = computeCCS(signals, weights);

    console.debug(`CCS computed: ${ccs.score} (model: ${ccs.model_version})`);

    return ccs;

  } catch (error) {
    debugLogger.error('Failed to fetch and compute CCS:', error);
    throw error;
  }
}

/**
 * Validate CCS score
 */
export function validateCCS(ccs) {
  if (!ccs || typeof ccs !== 'object') {
    return false;
  }

  const { available, score, signals, model_version } = ccs;

  if (available !== true) {
    return false;
  }

  if (typeof score !== 'number' || score < 0 || score > 100) {
    return false;
  }

  if (!signals || typeof signals !== 'object') {
    return false;
  }

  if (!model_version || typeof model_version !== 'string') {
    return false;
  }

  return true;
}

/**
 * Get CCS interpretation
 */
export function interpretCCS(score) {
  if (score >= 80) return { level: 'very_high', label: 'Very Bullish', color: '#10b981' };
  if (score >= 65) return { level: 'high', label: 'Bullish', color: '#059669' };
  if (score >= 50) return { level: 'medium', label: 'Neutral+', color: '#f59e0b' };
  if (score >= 35) return { level: 'low', label: 'Neutral-', color: '#f97316' };
  return { level: 'very_low', label: 'Bearish', color: '#dc2626' };
}

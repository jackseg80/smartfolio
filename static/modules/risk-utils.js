/**
 * Risk Dashboard - Utility Functions
 * Common helpers and formatters used across the Risk Dashboard
 */

// Import shared formatters from core module
import { formatNumber, formatMoney, formatRelativeTime } from '../core/formatters.js';

// Re-export for backward compatibility
export { formatNumber, formatMoney, formatRelativeTime };

// ===== Formatting Functions =====

/**
 * Safely format a number to fixed decimal places
 * @param {number} v - Value to format
 * @param {number} d - Decimal places (default: 2)
 * @returns {string} Formatted value or 'N/A'
 */
export function safeFixed(v, d = 2) {
  return (v == null || isNaN(v)) ? 'N/A' : Number(v).toFixed(d);
}

/**
 * Format a decimal as percentage
 * @param {number} v - Value to format (0.15 → 15%)
 * @returns {string} Formatted percentage or 'N/A'
 */
export function formatPercent(v) {
  return (v == null || isNaN(v)) ? 'N/A' : (v * 100).toFixed(2) + '%';
}

// ===== Risk Score Functions =====

/**
 * Map risk score to risk level category
 * Based on canonical thresholds from docs/RISK_SEMANTICS.md
 * @param {number} score - Risk score (0-100, higher = more robust)
 * @returns {string} Risk level category
 */
export function scoreToRiskLevel(score) {
  if (score == null) return 'unknown';
  if (score >= 80) return 'very-low';     // Très robuste
  if (score >= 65) return 'low';          // Robuste
  if (score >= 50) return 'medium';       // Modéré
  if (score >= 35) return 'high';         // Fragile
  if (score >= 20) return 'very-high';    // Très fragile
  return 'critical';                      // Critique (<20)
}

/**
 * Get color for risk score (positive semantics - higher = better)
 * @param {number} score - Risk score (0-100)
 * @returns {string} CSS color value
 */
export function pickScoreColor(score) {
  if (score == null) return 'var(--theme-text)';
  if (score >= 80) return '#9ece6a';      // Vert foncé (very_low risk)
  if (score >= 65) return '#73daca';      // Vert clair (low risk)
  if (score >= 50) return '#ff9e64';      // Jaune/Orange (medium)
  if (score >= 35) return '#e0af68';      // Orange clair (high)
  if (score >= 20) return '#f7768e';      // Rouge clair (very_high)
  return '#db4b4b';                       // Rouge foncé (critical)
}

/**
 * Get human-readable interpretation of risk score
 * @param {number} score - Risk score (0-100)
 * @returns {string} Interpretation text
 */
export function getScoreInterpretation(score) {
  if (score == null) return 'Metric not available';
  if (score >= 80) return 'Very robust portfolio - excellent';
  if (score >= 65) return 'Robust portfolio - well protected';
  if (score >= 50) return 'Moderate robustness - monitor';
  if (score >= 35) return 'Portfolio fragile - attention requise';
  if (score >= 20) return 'Very fragile portfolio - high risk';
  return 'Critical portfolio - immediate danger';
}

// ===== Metric Health Assessment =====

/**
 * Assess metric health based on predefined rules
 * @param {string} key - Metric key (e.g., 'var_95_1d', 'sharpe_ratio')
 * @param {number} value - Metric value
 * @returns {Object} Health assessment { level, color, interpretation }
 */
export function getMetricHealth(key, value) {
  const healthRules = {
    'var_95_1d': {
      good: [0, 0.04], // 0% to 4%
      warning: [0.04, 0.08], // 4% to 8%
      danger: [0.08, 1], // > 8%
      interpretation: {
        good: "Potential daily loss contained",
        warning: "Moderate potential loss",
        danger: "High potential loss - caution"
      }
    },
    'var_99_1d': {
      good: [0, 0.06],
      warning: [0.06, 0.12],
      danger: [0.12, 1],
      interpretation: {
        good: "Limited extreme loss",
        warning: "Moderate extreme loss",
        danger: "Significant extreme loss"
      }
    },
    'sharpe_ratio': {
      danger: [0, 0.5],
      warning: [0.5, 1.0],
      good: [1.0, 5.0],
      interpretation: {
        danger: "Insufficient return/risk",
        warning: "Acceptable return/risk",
        good: "Excellent risk-adjusted return"
      }
    },
    'sortino_ratio': {
      danger: [0, 0.8],
      warning: [0.8, 1.2],
      good: [1.2, 5.0],
      interpretation: {
        danger: "Insufficient downside protection",
        warning: "Adequate downside protection",
        good: "Excellent downside protection"
      }
    },
    'volatility_annualized': {
      good: [0, 0.4], // 0-40%
      warning: [0.4, 0.8], // 40-80%
      danger: [0.8, 2.0], // >80%
      interpretation: {
        good: "Low volatility",
        warning: "Moderate volatility",
        danger: "High volatility"
      }
    },
    'max_drawdown': {
      good: [0, 0.3], // 0% to 30%
      warning: [0.3, 0.6], // 30% to 60%
      danger: [0.6, 1.0], // > 60%
      interpretation: {
        good: "Limited drawdown",
        warning: "Typical crypto drawdown",
        danger: "Extreme drawdown - diversify"
      }
    },
    'diversification_ratio': {
      danger: [0, 0.4],
      warning: [0.4, 0.7],
      good: [0.7, 2.0], // >1 possible si corrélations négatives
      interpretation: {
        danger: "Very poorly diversified",
        warning: "Limited diversification",
        good: "Well diversified (low or negative correlations)"
      }
    },
    'effective_assets': {
      danger: [0, 10],
      warning: [10, 20],
      good: [20, 999],
      interpretation: {
        danger: "Very few effective assets",
        warning: "Partial diversification",
        good: "Good diversification"
      }
    },
    'risk_score': {
      danger: [0, 40],     // 0-40: faible robustesse
      warning: [40, 65],   // 40-65: robustesse modérée
      good: [65, 100],     // 65-100: bonne robustesse
      interpretation: {
        danger: "Portfolio fragile - high risk",
        warning: "Moderate robustness - monitor",
        good: "Robust portfolio - well protected"
      }
    }
  };

  const rule = healthRules[key];
  if (!rule) return { level: 'unknown', color: '#6b7280', interpretation: 'Metric not evaluated' };

  // Check which range the value falls into
  for (const [level, range] of Object.entries(rule)) {
    if (level === 'interpretation') continue;

    const [min, max] = range;
    if (value >= min && value <= max) {
      const color = level === 'good' ? '#10b981' : level === 'warning' ? '#f59e0b' : '#ef4444';
      return {
        level,
        color,
        interpretation: rule.interpretation[level] || 'No interpretation available'
      };
    }
  }

  return { level: 'unknown', color: '#6b7280', interpretation: 'Value out of bounds' };
}

// ===== Alert Formatting =====

/**
 * Format alert type for display
 * @param {string} alertType - Alert type code
 * @returns {string} Human-readable alert type
 */
export function formatAlertType(alertType) {
  const typeMap = {
    'concentration': '🎯 Concentration Risk',
    'volatility': '📊 High Volatility',
    'drawdown': '📉 Large Drawdown',
    'correlation': '🔗 High Correlation',
    'diversification': '🌐 Low Diversification',
    'rebalance': '⚖️ Rebalance Needed'
  };
  return typeMap[alertType] || alertType;
}

// ===== DOM Utilities =====

/**
 * Show loading state in container
 * @param {HTMLElement} container - Container element
 * @param {string} message - Loading message
 */
export function showLoading(container, message = 'Loading...') {
  container.innerHTML = `
    <div class="loading-center">
      🔄 ${message}
    </div>
  `;
}

/**
 * Show error state in container
 * @param {HTMLElement} container - Container element
 * @param {string} message - Error message
 * @param {string} hint - Optional hint text
 */
export function showError(container, message, hint = '') {
  container.innerHTML = `
    <div class="error">
      <div class="error-title">⚠️ Error</div>
      <div class="error-message">${message}</div>
      ${hint ? `<div class="error-hint">${hint}</div>` : ''}
    </div>
  `;
}

/**
 * Create a metric row HTML element
 * @param {string} label - Metric label
 * @param {string} value - Metric value (already formatted)
 * @param {Object} options - Additional options (color, hint, etc.)
 * @returns {string} HTML string for metric row
 */
export function createMetricRow(label, value, options = {}) {
  const { color, hint, hintKey } = options;
  const hintAttr = hint ? `data-key="${hintKey}" class="hinted"` : '';
  const colorStyle = color ? `style="color: ${color}"` : '';

  return `
    <div class="metric-row">
      <span class="metric-label">${label}</span>
      <span class="metric-value" ${hintAttr} ${colorStyle}>${value}</span>
    </div>
  `;
}

// ===== Cache Utilities =====

/**
 * Set cached data with TTL
 * @param {string} key - Cache key
 * @param {*} data - Data to cache
 * @param {number} ttlMs - Time to live in milliseconds
 */
export function setCachedData(key, data, ttlMs = 12 * 60 * 60 * 1000) {
  try {
    const cacheEntry = {
      data,
      timestamp: Date.now(),
      ttl: ttlMs
    };
    localStorage.setItem(key, JSON.stringify(cacheEntry));
  } catch (error) {
    debugLogger.warn(`Failed to cache data for key: ${key}`, error);
  }
}

/**
 * Get cached data if not expired
 * @param {string} key - Cache key
 * @returns {*} Cached data or null if expired/not found
 */
export function getCachedData(key) {
  try {
    const cached = localStorage.getItem(key);
    if (!cached) return null;

    const entry = JSON.parse(cached);
    const age = Date.now() - entry.timestamp;

    if (age > entry.ttl) {
      localStorage.removeItem(key);
      return null;
    }

    return entry.data;
  } catch (error) {
    debugLogger.warn(`Failed to retrieve cached data for key: ${key}`, error);
    return null;
  }
}

/**
 * Clear all risk dashboard caches
 */
export function clearAllRiskCaches() {
  const keys = [
    'risk_scores_cache',
    'risk_data_cache',
    'alerts_cache',
    'CYCLE_CHART'
  ];

  keys.forEach(key => {
    try {
      localStorage.removeItem(key);
    } catch (e) {
      debugLogger.warn(`Failed to clear cache: ${key}`, e);
    }
  });

  debugLogger.debug('✅ All risk dashboard caches cleared');
}

// ===== Export all utilities =====
export default {
  // Formatting
  safeFixed,
  formatPercent,
  formatNumber,
  formatMoney,
  formatRelativeTime,

  // Risk scoring
  scoreToRiskLevel,
  pickScoreColor,
  getScoreInterpretation,

  // Health assessment
  getMetricHealth,

  // Alert formatting
  formatAlertType,

  // DOM utilities
  showLoading,
  showError,
  createMetricRow,

  // Cache utilities
  setCachedData,
  getCachedData,
  clearAllRiskCaches
};

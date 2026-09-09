/**
 * On-Chain Indicators Module
 * Intègre des métriques Bitcoin on-chain pour améliorer la précision du modèle de cycle
 * 
 * SOURCES D'INDICATEURS:
 * 
 * 1. Fear & Greed Index - DISPONIBLE
 *    Source: Alternative.me API (https://api.alternative.me/fng/)
 *    Fréquence: Quotidienne
 *    Fiabilité: ✅ Production ready
 * 
 * Les autres indicateurs proviennent du service Crypto-Toolbox. Aucune valeur
 * synthétique n'est produite quand ce service ou sa source est indisponible.
 */

// ===== SWR CACHE SYSTEM =====

// Import V2 composite score calculator (replaces legacy V1)
import { calculateCompositeScoreV2 } from './composite-score-v2.js';

/**
 * SWR (Stale-While-Revalidate) cache constants - optimized for onchain indicators
 */
const TTL_SHOW_MS = 10 * 60 * 1000;       // Show cache if < 10min (faster updates)
const TTL_BG_MS = 30 * 60 * 1000;         // Revalidate background if 10min-30min
const TTL_HARD_MS = 2 * 60 * 60 * 1000;   // Force network if > 2h (more aggressive refresh)
const LS_KEY = 'CTB_ONCHAIN_CACHE_V2';

/**
 * SWR Cache functions for localStorage
 */
function readOnchainCache() {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) || 'null');
  } catch {
    return null;
  }
}

function writeOnchainCache(payload) {
  try {
    const cacheEntry = {
      ...payload,
      saved_at: Date.now(),
      cache_version: 'v2'
    };
    localStorage.setItem(LS_KEY, JSON.stringify(cacheEntry));
    (window.debugLogger?.debug || console.log)('💾 On-chain indicators cached', {
      count: payload.count || 0,
      cached_at: new Date().toISOString()
    });
  } catch (error) {
    (window.debugLogger?.warn || console.warn)('Failed to write onchain cache:', error);
  }
}

/**
 * Advanced caching system with adaptive TTL and performance optimization
 */
class IntelligentCache {
  constructor() {
    this.cache = new Map();
    this.accessPatterns = new Map();
    this.performanceMetrics = {
      hits: 0,
      misses: 0,
      totalRequests: 0,
      avgResponseTime: 0
    };
  }

  /**
   * Get adaptive TTL based on market volatility and access patterns
   */
  getAdaptiveTTL(key, baseMs = 10 * 60 * 1000) {
    const pattern = this.accessPatterns.get(key);
    if (!pattern) return baseMs;

    // High frequency access = shorter TTL for freshness
    if (pattern.accessCount > 10 && pattern.avgInterval < 30000) {
      return baseMs * 0.5; // 5 minutes for high frequency
    }

    // Low frequency access = longer TTL for performance  
    if (pattern.accessCount < 3 && pattern.avgInterval > 300000) {
      return baseMs * 2; // 20 minutes for low frequency
    }

    return baseMs;
  }

  /**
   * Set cache entry with metadata
   */
  set(key, value, customTtl = null) {
    const ttl = customTtl || this.getAdaptiveTTL(key);
    const entry = {
      value,
      timestamp: Date.now(),
      ttl,
      accessCount: 0,
      lastAccess: Date.now()
    };

    this.cache.set(key, entry);
    this.updateAccessPattern(key);
    return entry;
  }

  /**
   * Get cache entry if still valid
   */
  get(key) {
    const entry = this.cache.get(key);
    if (!entry) {
      this.performanceMetrics.misses++;
      this.performanceMetrics.totalRequests++;
      return null;
    }

    const age = Date.now() - entry.timestamp;
    if (age > entry.ttl) {
      this.cache.delete(key);
      this.performanceMetrics.misses++;
      this.performanceMetrics.totalRequests++;
      return null;
    }

    // Update access metadata
    entry.accessCount++;
    entry.lastAccess = Date.now();
    this.updateAccessPattern(key);

    this.performanceMetrics.hits++;
    this.performanceMetrics.totalRequests++;

    return entry.value;
  }

  /**
   * Update access patterns for adaptive TTL
   */
  updateAccessPattern(key) {
    const now = Date.now();
    const pattern = this.accessPatterns.get(key) || {
      accessCount: 0,
      lastAccess: now,
      intervals: [],
      avgInterval: 0
    };

    if (pattern.lastAccess > 0) {
      const interval = now - pattern.lastAccess;
      pattern.intervals.push(interval);

      // Keep only last 10 intervals for rolling average
      if (pattern.intervals.length > 10) {
        pattern.intervals.shift();
      }

      pattern.avgInterval = pattern.intervals.reduce((a, b) => a + b, 0) / pattern.intervals.length;
    }

    pattern.accessCount++;
    pattern.lastAccess = now;
    this.accessPatterns.set(key, pattern);
  }

  /**
   * Check if cache entry exists and is still valid
   */
  has(key) {
    const entry = this.cache.get(key);
    if (!entry) return false;

    const age = Date.now() - entry.timestamp;
    return age <= entry.ttl;
  }

  /**
   * Get cache statistics
   */
  getStats() {
    const hitRate = this.performanceMetrics.totalRequests > 0
      ? (this.performanceMetrics.hits / this.performanceMetrics.totalRequests) * 100
      : 0;

    return {
      ...this.performanceMetrics,
      hitRate: Math.round(hitRate * 100) / 100,
      cacheSize: this.cache.size,
      patterns: this.accessPatterns.size
    };
  }

  /**
   * Clear expired entries and optimize memory
   */
  cleanup() {
    const now = Date.now();
    let cleaned = 0;

    for (const [key, entry] of this.cache.entries()) {
      const age = now - entry.timestamp;
      if (age > entry.ttl) {
        this.cache.delete(key);
        cleaned++;
      }
    }

    // Clean old access patterns
    for (const [key, pattern] of this.accessPatterns.entries()) {
      if (now - pattern.lastAccess > 24 * 60 * 60 * 1000) { // 24h
        this.accessPatterns.delete(key);
      }
    }

    (window.debugLogger?.debug || console.log)(`🧹 Cache cleanup: ${cleaned} expired entries removed`);
    return cleaned;
  }

  /**
   * Force refresh of specific key
   */
  invalidate(key) {
    this.cache.delete(key);
  }

  /**
   * Clear all cache
   */
  clear() {
    this.cache.clear();
    this.accessPatterns.clear();
    this.performanceMetrics = { hits: 0, misses: 0, totalRequests: 0, avgResponseTime: 0 };
  }
}

// Global intelligent cache instance
const intelligentCache = new IntelligentCache();

// Legacy cache variables (for backward compatibility)
let indicatorsCache = intelligentCache;
const CACHE_DURATION = 4 * 60 * 60 * 1000; // 4 hours (optimized: on-chain data updates daily)

// Plus de cache de simulation - données réelles uniquement

// Auto-cleanup every 30 minutes
setInterval(() => {
  intelligentCache.cleanup();
}, 30 * 60 * 1000);

// ===== INDICATOR CATEGORIES AND WEIGHTS =====

/**
 * Catégorisation intelligente des indicateurs avec pondérations
 * Basée sur l'analyse des 30+ indicateurs de Crypto-Toolbox
 */
export const INDICATOR_CATEGORIES = {
  // Indicateurs On-Chain Fondamentaux (60% du score) - Données blockchain pures
  onchain_fundamentals: {
    weight: 0.60,
    description: "Fundamental blockchain metrics",
    indicators: {
      // Évaluation de prix vs valeur intrinsèque
      'mvrv': { weight: 0.25, invert: true }, // High MVRV = overvalued = bearish
      'mvrv_z_score': { weight: 0.25, invert: true },
      'mvrv_cointime': { weight: 0.20, invert: true }, // Cointime MVRV-Z Score
      'nupl': { weight: 0.20, invert: true }, // High NUPL = profit zone = bearish
      'rupl_nupl': { weight: 0.20, invert: true },
      'rupl': { weight: 0.20, invert: true },

      // Métriques miniers et réseau
      'puell': { weight: 0.15, invert: false }, // Puell Multiple pattern varies
      'puell_multiple': { weight: 0.15, invert: false },
      'sopr': { weight: 0.10, invert: false }, // SOPR >1 = profits realized
      'rhodl': { weight: 0.05, invert: true },
      'rhold': { weight: 0.05, invert: true }, // RHOLD ratio

      // Autres métriques blockchain
      'reserve_risk': { weight: 0.15, invert: true }, // Reserve Risk indicator
      'cdd': { weight: 0.10, invert: false }, // Coin Days Destroyed
      'bmo': { weight: 0.10, invert: false } // BMO indicator
    }
  },

  // Indicateurs Cycle/Techniques (30% du score) - Signaux de timing
  cycle_technical: {
    weight: 0.30,
    description: "Signaux de cycle et techniques",
    indicators: {
      'pi_cycle': { weight: 0.35, invert: true }, // High = top signal
      'pi': { weight: 0.35, invert: true },
      'cbbi': { weight: 0.30, invert: true }, // Colin Bicknell Bitcoin Index
      'rainbow': { weight: 0.20, invert: true }, // Rainbow Chart position
      'rsi': { weight: 0.15, invert: true }, // RSI Bitcoin mensuel
      'stock_to_flow': { weight: 0.0, invert: false }, // Modèle controversé
      'ma_2y': { weight: 0.20, invert: true }, // 2 Year Moving Average
      'trolololo': { weight: 0.15, invert: true }, // Trolololo Trend Line
      'woobull': { weight: 0.15, invert: true }, // Woobull top indicator
      'mayer_multiple': { weight: 0.15, invert: true }, // Mayer Multiple
      'btc_dominance': { weight: 0.10, invert: false }, // BTC Dominance
      'altseason': { weight: 0.10, invert: false }, // Altcoin Season Index
      'ahr999': { weight: 0.15, invert: true } // Ahr999 index
    }
  },

  // Indicateurs de Sentiment (10% du score) - Psychologie de marché
  sentiment: {
    weight: 0.10,
    description: "Market sentiment and psychology",
    indicators: {
      'fear_greed': { weight: 0.60, invert: true }, // High Fear & Greed = bearish
      'fear_greed_7d': { weight: 0.40, invert: true },
      'fear': { weight: 0.60, invert: true },
      'google_trends': { weight: 0.20, invert: true }, // High search interest = bearish
      'google_crypto': { weight: 0.15, invert: true },
      'google_bitcoin': { weight: 0.15, invert: true },
      'google_ethereum': { weight: 0.10, invert: true }
    }
  }
};

/**
 * Fonction pour classer automatiquement un indicateur selon son nom
 */
export function classifyIndicator(indicatorName) {
  const name = indicatorName.toLowerCase().trim();

  // Patterns spéciaux pour indicateurs de Crypto-Toolbox
  const specialMappings = {
    'pi cycle': { category: 'cycle_technical', key: 'pi_cycle' },
    'cbbi': { category: 'cycle_technical', key: 'cbbi' },
    'mvrv z-score': { category: 'onchain_fundamentals', key: 'mvrv_z_score' },
    'mvrv': { category: 'onchain_fundamentals', key: 'mvrv' },
    'puell multiple': { category: 'onchain_fundamentals', key: 'puell_multiple' },
    'puell': { category: 'onchain_fundamentals', key: 'puell' },
    'rupl/nupl': { category: 'onchain_fundamentals', key: 'rupl_nupl' },
    'nupl': { category: 'onchain_fundamentals', key: 'nupl' },
    'rupl': { category: 'onchain_fundamentals', key: 'rupl' },
    'rsi bitcoin': { category: 'cycle_technical', key: 'rsi' },
    'rsi mensuel': { category: 'cycle_technical', key: 'rsi' },
    'rsi': { category: 'cycle_technical', key: 'rsi' },
    'fear & greed': { category: 'sentiment', key: 'fear_greed' },
    'fear and greed': { category: 'sentiment', key: 'fear_greed' },
    'sopr': { category: 'onchain_fundamentals', key: 'sopr' },

    // Additional missing indicators from console logs
    '2y ma': { category: 'cycle_technical', key: 'ma_2y' },
    '2 year ma': { category: 'cycle_technical', key: 'ma_2y' },
    'trolololo trend line': { category: 'cycle_technical', key: 'trolololo' },
    'reserve risk': { category: 'onchain_fundamentals', key: 'reserve_risk' },
    'woobull': { category: 'cycle_technical', key: 'woobull' },
    'cointime mvrv-z score': { category: 'onchain_fundamentals', key: 'mvrv_cointime' },
    'cointime mvrv-z score (ema 14j)': { category: 'onchain_fundamentals', key: 'mvrv_cointime' },
    'mayer multiple': { category: 'cycle_technical', key: 'mayer_multiple' },
    'mayer mutiple': { category: 'cycle_technical', key: 'mayer_multiple' }, // typo in data source
    'rhold': { category: 'onchain_fundamentals', key: 'rhold' },
    'coin days destroyed': { category: 'onchain_fundamentals', key: 'cdd' },
    'coin days destroyed (ma 90j)': { category: 'onchain_fundamentals', key: 'cdd' },
    'bmo (par prof. chaîne)': { category: 'onchain_fundamentals', key: 'bmo' },
    'bmo': { category: 'onchain_fundamentals', key: 'bmo' },
    'dominance btc': { category: 'cycle_technical', key: 'btc_dominance' },
    'altcoin season index': { category: 'cycle_technical', key: 'altseason' },
    'google trend': { category: 'sentiment', key: 'google_trends' },
    'google trends': { category: 'sentiment', key: 'google_trends' },
    'ahr999': { category: 'cycle_technical', key: 'ahr999' },
    'cbbi*': { category: 'cycle_technical', key: 'cbbi' }
  };

  // Recherche par correspondance exacte d'abord
  for (const [pattern, mapping] of Object.entries(specialMappings)) {
    if (name.includes(pattern)) {
      const category = INDICATOR_CATEGORIES[mapping.category];
      const config = category.indicators[mapping.key];

      if (config) {
        return {
          category: mapping.category,
          key: mapping.key,
          weight: config.weight,
          invert: config.invert,
          categoryWeight: category.weight
        };
      }
    }
  }

  // Recherche générique dans chaque catégorie
  for (const [categoryKey, category] of Object.entries(INDICATOR_CATEGORIES)) {
    for (const [indicatorKey, config] of Object.entries(category.indicators)) {
      // Correspondance partielle
      if (name.includes(indicatorKey) || indicatorKey.includes(name.split(' ')[0])) {
        return {
          category: categoryKey,
          key: indicatorKey,
          weight: config.weight,
          invert: config.invert,
          categoryWeight: category.weight
        };
      }
    }
  }

  // Par défaut, classer comme sentiment avec poids faible
  console.debug(`⚠️ Unknown indicator classification: ${indicatorName}`);
  return {
    category: 'sentiment',
    key: 'unknown',
    weight: 0.1,
    invert: false,
    categoryWeight: 0.10
  };
}

/**
 * Configuration des indicateurs on-chain avec leurs seuils (LEGACY)
 */
export const INDICATORS_CONFIG = {
  mvrv: {
    name: "MVRV Ratio",
    description: "Market Value to Realized Value",
    thresholds: {
      extreme_high: 3.5,  // Pic de cycle probable
      high: 2.5,          // Zone de distribution
      normal_high: 1.8,   // Marché chaud
      normal: 1.0,        // Équilibre
      normal_low: 0.8,    // Marché froid
      low: 0.6,           // Zone d'accumulation
      extreme_low: 0.4    // Creux de cycle probable
    },
    weight: 0.25,
    api_available: true
  },

  nvt: {
    name: "NVT Ratio",
    description: "Network Value to Transactions",
    thresholds: {
      extreme_high: 150,  // Très surévalué
      high: 100,          // Surévalué
      normal_high: 75,    // Cher
      normal: 50,         // Équilibre
      normal_low: 35,     // Bon marché
      low: 25,            // Sous-évalué
      extreme_low: 15     // Très sous-évalué
    },
    weight: 0.20,
    api_available: true
  },

  puell_multiple: {
    name: "Puell Multiple",
    description: "Mining revenue vs 365d average",
    thresholds: {
      extreme_high: 4.0,  // Pic minier - Vente probable
      high: 2.5,          // Zone de distribution
      normal_high: 1.5,   // Au-dessus de la moyenne
      normal: 1.0,        // Moyenne historique
      normal_low: 0.8,    // Sous la moyenne
      low: 0.5,           // Zone d'accumulation
      extreme_low: 0.3    // Capitulation minière
    },
    weight: 0.15,
    api_available: true
  },

  rhodl: {
    name: "RHODL Ratio",
    description: "Realized HODL Ratio",
    thresholds: {
      extreme_high: 50000, // Pic de cycle
      high: 30000,         // Zone de distribution
      normal_high: 20000,  // Marché chaud
      normal: 10000,       // Équilibre
      normal_low: 7000,    // Marché froid
      low: 5000,           // Accumulation
      extreme_low: 3000    // Creux de cycle
    },
    weight: 0.25,
    api_available: false // Plus complexe à calculer
  },

  fear_greed: {
    name: "Fear & Greed Index",
    description: "Market sentiment",
    thresholds: {
      extreme_high: 90,   // Extrême cupidité - Vente
      high: 75,           // Cupidité
      normal_high: 60,    // Optimisme
      normal: 50,         // Neutre
      normal_low: 40,     // Peur
      low: 25,            // Peur importante
      extreme_low: 10     // Extrême peur - Achat
    },
    weight: 0.10,
    api_available: true,
    contrarian: true // Indicateur contrarian
  },

  ahr999: {
    name: "Ahr999 Index",
    description: "Long-term Bitcoin buy indicator",
    thresholds: {
      extreme_high: 4.0,  // Très surévalué
      high: 2.0,          // Surévalué  
      normal_high: 1.2,   // Cher
      normal: 0.8,        // Équilibre
      normal_low: 0.45,   // Zone d'accumulation
      low: 0.25,          // Excellente opportunité
      extreme_low: 0.1    // Opportunité exceptionnelle
    },
    weight: 0.15,
    api_available: true,
    contrarian: true // Plus bas = meilleur pour acheter
  },

  nupl: {
    name: "NUPL (Net Unrealized Profit/Loss)",
    description: "Network unrealized profit/loss",
    thresholds: {
      extreme_high: 90,   // Euphorie excessive
      high: 75,           // Euphorie/Cupidité
      normal_high: 60,    // Optimisme
      normal: 50,         // Croyance/Déni
      normal_low: 40,     // Espoir/Peur
      low: 25,            // Capitulation
      extreme_low: 10     // Capitulation extrême
    },
    weight: 0.20,
    api_available: true
  },

  rsi: {
    name: "RSI Bitcoin",
    description: "Relative Strength Index pour Bitcoin",
    thresholds: {
      extreme_high: 80,   // Très suracheté
      high: 70,           // Suracheté
      normal_high: 60,    // Tendance haussière
      normal: 50,         // Neutre
      normal_low: 40,     // Tendance baissière
      low: 30,            // Survendu
      extreme_low: 20     // Très survendu
    },
    weight: 0.10,
    api_available: true
  },

  sopr: {
    name: "SOPR (Spent Output Profit Ratio)",
    description: "Spent output profit ratio",
    thresholds: {
      extreme_high: 90,   // Très profitable
      high: 75,           // Profitable
      normal_high: 60,    // Légèrement profitable
      normal: 50,         // Équilibre
      normal_low: 40,     // Légèrement non profitable
      low: 25,            // Non profitable
      extreme_low: 10     // Très non profitable
    },
    weight: 0.10,
    api_available: true
  },

  stock_to_flow_deviation: {
    name: "Stock-to-Flow Deviation",
    description: "Deviation between actual price and S2F model",
    thresholds: {
      extreme_high: 200,  // Très au-dessus du modèle
      high: 100,          // Au-dessus du modèle
      normal_high: 50,    // Légèrement au-dessus
      normal: 0,          // Conforme au modèle
      normal_low: -25,    // Légèrement en dessous
      low: -50,           // En dessous du modèle
      extreme_low: -75    // Très en dessous
    },
    weight: 0.15,
    api_available: false // Calcul custom requis
  }
};

/**
 * Convertit une valeur d'indicateur en score de cycle (0-100)
 */
function indicatorToScore(value, config) {
  const { thresholds, contrarian } = config;
  let score = 50; // Score neutre par défaut

  // Déterminer le niveau de l'indicateur
  if (value >= thresholds.extreme_high) {
    score = contrarian ? 10 : 95;
  } else if (value >= thresholds.high) {
    score = contrarian ? 25 : 85;
  } else if (value >= thresholds.normal_high) {
    score = contrarian ? 40 : 70;
  } else if (value >= thresholds.normal) {
    score = 50;
  } else if (value >= thresholds.normal_low) {
    score = contrarian ? 60 : 40;
  } else if (value >= thresholds.low) {
    score = contrarian ? 75 : 25;
  } else {
    score = contrarian ? 90 : 15;
  }

  return score;
}

/**
 * Export cache stats and control functions
 */
export function getCacheStats() {
  return intelligentCache.getStats();
}

export function clearCache() {
  intelligentCache.clear();
  (window.debugLogger?.debug || console.log)('🧹 OnChain indicators cache cleared');
}

export function invalidateCache(key) {
  intelligentCache.invalidate(key);
  (window.debugLogger?.debug || console.log)(`🔄 Cache invalidated for: ${key}`);
}

/**
 * Optimized fetch with performance monitoring and rate limiting
 */
async function performanceMonitoredFetch(url, options = {}) {
  const startTime = performance.now();

  try {
    const response = await fetch(url, {
      ...options,
      // Increased timeout for scraping endpoints (crypto-toolbox can take 2-3s)
      signal: AbortSignal.timeout(15000) // 15 second timeout
    });

    const endTime = performance.now();
    const responseTime = endTime - startTime;

    // Update performance metrics
    const stats = intelligentCache.performanceMetrics;
    stats.avgResponseTime = stats.avgResponseTime === 0
      ? responseTime
      : (stats.avgResponseTime + responseTime) / 2;

    (window.debugLogger?.debug || console.log)(`📡 API response time: ${Math.round(responseTime)}ms`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response;
  } catch (error) {
    const endTime = performance.now();
    const responseTime = endTime - startTime;

    if (error.name === 'TimeoutError') {
      (window.debugLogger?.warn || console.warn)('⏰ API request timed out after 15s');
    } else if (error.name === 'AbortError') {
      (window.debugLogger?.warn || console.warn)('🚫 API request was aborted');
    } else {
      (window.debugLogger?.warn || console.warn)(`🌐 Network error (${Math.round(responseTime)}ms):`, error.message);
    }

    throw error;
  }
}

/**
 * Récupère les données Fear & Greed Index avec cache intelligent
 */
async function fetchFearGreedIndex() {
  try {
    // Check intelligent cache first
    const cached = intelligentCache.get('fear_greed');
    if (cached) {
      (window.debugLogger?.info || console.log)('📊 Fear & Greed from intelligent cache');
      return cached;
    }

    (window.debugLogger?.debug || console.log)('📡 Fetching Fear & Greed from API...');
    const response = await performanceMonitoredFetch('https://api.alternative.me/fng/?limit=1');
    const data = await response.json();
    if (data.data && data.data[0]) {
      const result = {
        value: parseInt(data.data[0].value),
        classification: data.data[0].value_classification,
        timestamp: new Date(data.data[0].timestamp * 1000),
        source: 'Alternative.me'
      };

      // Store in intelligent cache
      intelligentCache.set('fear_greed', result);
      (window.debugLogger?.debug || console.log)('💾 Fear & Greed cached with adaptive TTL');

      return result;
    }
    throw new Error('Invalid response format');
  } catch (error) {
    (window.debugLogger?.warn || console.warn)('Fear & Greed Index fetch failed:', error.message);
    return null;
  }
}

// Global deduplication and circuit breaker - optimized for better recovery
let _ongoingFetch = null;
let _circuitBreakerState = {
  failures: 0,
  lastFailure: 0,
  isOpen: false,
  FAILURE_THRESHOLD: 2,   // Reduced threshold for faster circuit breaking
  RESET_TIMEOUT: 15000    // 15 seconds (was 30s) for faster recovery
};

// Rate-limited logging to reduce verbosity
const _logLimiter = {
  cache: new Map(),
  limit: (key, intervalMs = 30000) => { // 30 second intervals by default
    const now = Date.now();
    const lastLog = _logLimiter.cache.get(key);
    if (!lastLog || (now - lastLog) > intervalMs) {
      _logLimiter.cache.set(key, now);
      return true;
    }
    return false;
  }
};

/**
 * SWR background revalidation helper
 */
async function revalidateInBackground() {
  try {
    (window.debugLogger?.debug || console.log)('🔄 SWR: Revalidating onchain indicators in background...');
    await fetchCryptoToolboxIndicators({ force: true, silent: true });
    (window.debugLogger?.info || console.log)('✅ SWR: Background revalidation completed');
  } catch (error) {
    (window.debugLogger?.warn || console.warn)('⚠️ SWR: Background revalidation failed:', error.message);
  }
}

/**
 * Récupère les indicateurs depuis Crypto-Toolbox avec SWR
 */
export async function fetchCryptoToolboxIndicators({ force = false, silent = false } = {}) {
  if (!silent) {
    (window.debugLogger?.debug || console.log)('🌐 Fetching indicators from Crypto-Toolbox API...');
  }

  const cached = readOnchainCache();
  const now = Date.now();

  // 1) SWR Cache Logic: Return cache immediately if fresh enough
  if (!force && cached) {
    const age = now - (cached.saved_at || 0);

    if (age < TTL_SHOW_MS) {
      // Cache is fresh - return immediately
      (window.debugLogger?.debug || console.log)(`⚡ SWR: Serving from cache (age: ${Math.round(age / 1000 / 60)}min)`, {
        served_from: 'cache',
        cache_age_minutes: Math.round(age / 1000 / 60),
        indicators_count: cached.count || 0
      });

      // If between TTL_BG and TTL_SHOW, revalidate in background
      if (age > TTL_BG_MS) {
        revalidateInBackground().catch(() => { });
      }

      return {
        ...cached,
        served_from: 'cache',
        stale: false,
        cache_age_ms: age
      };
    }

    // Between TTL_SHOW and TTL_HARD: show cache + revalidate in background
    if (age < TTL_HARD_MS) {
      (window.debugLogger?.debug || console.log)(`🔄 SWR: Serving stale cache + background revalidation (age: ${Math.round(age / 1000 / 60)}min)`, {
        served_from: 'cache+bg',
        cache_age_minutes: Math.round(age / 1000 / 60)
      });

      revalidateInBackground().catch(() => { });
      return {
        ...cached,
        served_from: 'cache_revalidating',
        stale: true,
        cache_age_ms: age
      };
    }

    // > TTL_HARD: fall through to network
    (window.debugLogger?.debug || console.log)(`🌐 SWR: Cache too old (age: ${Math.round(age / 1000 / 60)}min), forcing network`, {
      served_from: 'network',
      reason: 'cache_expired'
    });
  }

  // 2) Circuit Breaker Check
  if (_circuitBreakerState.isOpen) {
    if (now - _circuitBreakerState.lastFailure < _circuitBreakerState.RESET_TIMEOUT) {
      if (_logLimiter.limit('circuit_breaker_active')) {
        (window.debugLogger?.warn || console.warn)('🚨 SWR: Circuit breaker OPEN - returning stale cache instead of network', {
          failures: _circuitBreakerState.failures,
          time_to_reset: Math.round((_circuitBreakerState.RESET_TIMEOUT - (now - _circuitBreakerState.lastFailure)) / 1000) + 's'
        });
      }

      // A hard-expired observation must not be exposed as current data.
      const cacheAge = cached ? now - (cached.saved_at || 0) : Infinity;
      if (cached && cacheAge < TTL_HARD_MS) {
        return {
          ...cached,
          served_from: 'circuit_breaker_cache',
          stale: true,
          cache_age_ms: cacheAge
        };
      }
    } else {
      // Reset circuit breaker
      _circuitBreakerState.isOpen = false;
      _circuitBreakerState.failures = 0;
      (window.debugLogger?.debug || console.log)('🔄 SWR: Circuit breaker RESET - attempting network again');
    }
  }

  // 3) Network fetch with deduplication
  if (_ongoingFetch) {
    (window.debugLogger?.debug || console.log)('🔄 SWR: Deduplicating concurrent request');
    return _ongoingFetch;
  }

  _ongoingFetch = (async () => {
    try {

      // Appel via le proxy FastAPI (8080) qui relaie vers Flask (8001)
      const apiBase = window.getApiBase();
      const proxyUrl = `${apiBase}/api/crypto-toolbox`;
      let response;
      try {
        response = await performanceMonitoredFetch(proxyUrl);
      } catch (err) {
        // Crypto-Toolbox service unavailable - use cached data or graceful degradation
        // No fallbacks to localhost (causes CSP violations in production)
        if (_logLimiter.limit('crypto_toolbox_unavailable')) {
          (window.debugLogger?.warn || console.warn)(
            `⚠️ Crypto-Toolbox service unavailable at ${proxyUrl}. Using cached data or graceful degradation.`
          );
        }
        throw err;
      }

      if (!response.ok) {
        if (response.status === 404) {
          (window.debugLogger?.warn || console.warn)('⚠️ Crypto-Toolbox service not available (optional feature)');
          return null; // Service optionnel non disponible
        }
        throw new Error(`Backend API error: ${response.status} ${response.statusText}`);
      }

      const apiData = await response.json();
      (window.debugLogger?.debug || console.log)(`📊 API response:`, apiData);

      // ✅ Detect stale/invalid data from backend
      if (apiData.scraping_failed) {
        const reason = apiData.failure_reason || 'Unknown error';
        const ageMin = Math.round((apiData.cache_age_seconds || 0) / 60);
        if (_logLimiter.limit('backend_scraping_failed', 60000)) { // Log once per minute
          (window.debugLogger?.warn || console.warn)(
            `⚠️ Backend scraping failed: ${reason} - Using stale cache (${ageMin} min old)`
          );
        }
        // Still continue with cached data - better than nothing
      }

      // Tolérance aux différentes formes de payload
      if (apiData.success === false) {
        throw new Error(`API returned error: ${apiData.error || apiData.message || 'unknown error'}`);
      }

      // Accepter plusieurs clés possibles: indicators | data | items | payload
      let raw = apiData.indicators || apiData.data || apiData.items || apiData.payload || null;
      if (raw && !Array.isArray(raw) && typeof raw === 'object') {
        // Certains backends renvoient un dict { key: indicator }
        raw = Object.values(raw);
      }
      if (!Array.isArray(raw)) {
        (window.debugLogger?.warn || console.warn)('⚠️ No indicator list array in response; attempting single-item normalization');
        raw = [];
      }

      // Convertir TOUS les indicateurs API en format pour le nouveau système
      const indicators = {};
      const pickName = (x) => x?.name || x?.indicator || x?.title || x?.key || 'unknown';
      const pickNumeric = (x) => {
        const cands = [x.value_numeric, x.numeric_value, x.value_percent, x.percent, x.score, x.value];
        for (const v of cands) {
          if (typeof v === 'number' && !Number.isNaN(v)) return v;
          if (typeof v === 'string') {
            // extraire nombre d’une chaîne comme "75.39%" ou "75,4"
            const m = v.match(/[\d.,]+/);
            if (m) return parseFloat(m[0].replace(',', '.'));
          }
        }
        return undefined;
      };
      const pickBool = (x, ...keys) => keys.map(k => x[k]).find(v => typeof v === 'boolean');
      const pick = (x, ...keys) => keys.map(k => x[k]).find(v => v != null);

      raw.forEach(entry => {
        const originalName = pickName(entry);
        const cleanKey = String(originalName).toLowerCase().replace(/[^a-z0-9]/g, '_');
        const num = pickNumeric(entry);
        if (num == null || Number.isNaN(num)) {
          (window.debugLogger?.warn || console.warn)(`⚠️ Skipped indicator without numeric value: ${originalName}`);
          return;
        }
        const inCritical = pickBool(entry, 'in_critical_zone', 'critical', 'is_critical') || false;
        const thresholdNumeric = pick(entry, 'threshold_numeric');
        const threshold = pick(entry, 'threshold');
        const rawThreshold = pick(entry, 'raw_threshold');
        const thresholdOp = pick(entry, 'threshold_operator', 'operator');
        const scrapedAt = pick(apiData, 'scraped_at', 'timestamp', 'fetched_at') || pick(entry, 'scraped_at', 'timestamp', 'fetched_at');
        const rawValue = pick(entry, 'raw_value') || (typeof entry.value === 'string' ? entry.value : undefined);

        indicators[cleanKey] = {
          name: originalName,
          value_numeric: num,
          value: entry.value ?? rawValue ?? `${num}%`,
          raw_value: rawValue,
          threshold_numeric: thresholdNumeric,
          threshold: threshold,
          raw_threshold: rawThreshold,
          in_critical_zone: inCritical,
          threshold_operator: thresholdOp,
          scraped_at: scrapedAt,
          source: entry.source || 'crypto-toolbox'
        };
        (window.debugLogger?.debug || console.log)(`✅ Processed: ${originalName} = ${num}% ${inCritical ? '🚨' : ''}`);
      });

      (window.debugLogger?.debug || console.log)(`📊 Converted ${Object.keys(indicators).length} indicators from API`);

      if (Object.keys(indicators).length > 0) {
        // ✅ Frontend validation: Check for suspicious data (all zeros)
        const indicatorValues = Object.values(indicators);
        const nonZeroCount = indicatorValues.filter(ind => (ind.value_numeric || 0) !== 0).length;
        const zeroPercentage = 100 - (nonZeroCount / indicatorValues.length * 100);

        if (zeroPercentage > 80) {
          // Critical: >80% zeros - likely scraping failure
          if (_logLimiter.limit('invalid_indicators_critical', 120000)) { // Log once per 2 minutes
            (window.debugLogger?.error || console.error)(
              `❌ CRITICAL: ${zeroPercentage.toFixed(1)}% of indicators are zero - data likely invalid!`
            );
          }
          // Show user-visible warning and reject the invalid observation.
          if (window.showToast) {
            window.showToast(
              `⚠️ On-chain data quality issue detected (${zeroPercentage.toFixed(0)}% zeros)`,
              'warning',
              { duration: 10000 }
            );
          }
          throw new Error(`On-chain observation rejected: ${zeroPercentage.toFixed(1)}% zero values`);
        } else if (zeroPercentage > 50) {
          // Warning: 50-80% zeros - suspicious
          if (_logLimiter.limit('invalid_indicators_warning', 120000)) {
            (window.debugLogger?.warn || console.warn)(
              `⚠️ WARNING: ${zeroPercentage.toFixed(1)}% of indicators are zero - data quality may be degraded`
            );
          }
        } else {
          // Data looks good
          (window.debugLogger?.debug || console.log)(
            `✅ Data quality check passed: ${nonZeroCount}/${indicatorValues.length} indicators have valid values`
          );
        }

        // Prepare SWR cache payload
        const cachePayload = {
          indicators,
          count: Object.keys(indicators).length,
          fetched_at: new Date().toISOString(),
          source: 'network',
          served_from: 'network',
          stale: false,
          cache_age_ms: 0,
          data_quality: {
            zero_percentage: zeroPercentage,
            valid_count: nonZeroCount,
            total_count: indicatorValues.length
          }
        };

        // Write to SWR cache
        writeOnchainCache(cachePayload);

        // Also update legacy cache for compatibility
        const CACHE_24H = 24 * 60 * 60 * 1000;
        intelligentCache.set('cryptotoolbox_indicators', indicators, CACHE_24H);

        // Reset circuit breaker on success
        _circuitBreakerState.failures = 0;
        _circuitBreakerState.isOpen = false;

        (window.debugLogger?.debug || console.log)(`✅ SWR: Network fetch successful`, {
          served_from: 'network',
          indicators_count: Object.keys(indicators).length,
          response_time_ms: '~661ms' // approximation from logs
        });

        return cachePayload;
      }

      throw new Error('No valid indicators found in API response');

    } catch (error) {
      // AbortError is normal (user changed tabs, navigation, etc.) - don't log as error
      const isAbortError = error.name === 'AbortError' || error.message.includes('aborted');

      if (isAbortError) {
        (window.debugLogger?.debug || console.debug)('🔄 Crypto-Toolbox API request aborted (normal):', error.message);
      } else {
        (window.debugLogger?.error || console.error)('❌ Crypto-Toolbox API fetch failed:', error.message);
      }

      // Enhanced graceful degradation for all types of API failures (except abort)
      if (!isAbortError && _logLimiter.limit('api_failure')) {
        (window.debugLogger?.warn || console.warn)('🌐 Crypto-Toolbox API failure:', error.message);
      }

      // Update circuit breaker state for persistent failures (not for aborts)
      if (!isAbortError && (error.message.includes('Failed to fetch') || error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('timed out'))) {
        _circuitBreakerState.failures++;
        _circuitBreakerState.lastFailure = Date.now();
        if (_circuitBreakerState.failures >= _circuitBreakerState.FAILURE_THRESHOLD) {
          _circuitBreakerState.isOpen = true;
          if (_logLimiter.limit('circuit_breaker_open')) {
            console.debug('🚨 Circuit breaker OPENED due to repeated failures');
          }
        }
      }

      // SWR graceful degradation is bounded by the hard freshness limit.
      const staleCache = readOnchainCache();
      if (staleCache && !force) {
        const age = Date.now() - (staleCache.saved_at || 0);
        if (age < TTL_HARD_MS) {
          (window.debugLogger?.info || console.info)(`🔄 Using labeled stale cache due to API failure (age: ${Math.round(age / 1000 / 60)}min)`, {
            served_from: 'stale_cache_fallback',
            cache_age_minutes: Math.round(age / 1000 / 60),
            reason: 'api_failure',
            error_type: error.name || 'unknown',
            circuit_breaker_failures: _circuitBreakerState.failures
          });
          return {
            ...staleCache,
            served_from: 'stale_cache_fallback',
            stale: true,
            cache_age_ms: age,
            fallback_reason: 'api_failure'
          };
        }
      }

      // Last resort: return minimal empty state instead of throwing
      if (_logLimiter.limit('empty_fallback')) {
        (window.debugLogger?.warn || console.warn)('🔄 No cache available, returning empty state for graceful degradation');
      }
      return {
        indicators: {},
        count: 0,
        fetched_at: null,
        source: null,
        served_from: 'unavailable',
        stale: false,
        cache_age_ms: null,
        error: error.message,
        available: false
      };
    } finally {
      _ongoingFetch = null; // Clear deduplication lock
    }
  })();

  return _ongoingFetch;
}

/**
 * Parse le HTML de Crypto-Toolbox pour extraire les indicateurs
 */
function parseCryptoToolboxHTML(html) {
  const indicators = {};

  try {
    // Créer un parser DOM
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');

    // Stratégie 1: Parser comme ton code Python - table tbody tr
    const tables = doc.querySelectorAll('table');
    (window.debugLogger?.debug || console.log)(`🔍 Found ${tables.length} tables to parse`);

    tables.forEach((table, tableIndex) => {
      // Chercher tbody tr comme dans ton code Python
      const rows = table.querySelectorAll('tbody tr');
      (window.debugLogger?.debug || console.log)(`🔍 Table ${tableIndex + 1}: Found ${rows.length} tbody rows`);

      if (rows.length === 0) {
        // Fallback: chercher tr directement
        const fallbackRows = table.querySelectorAll('tr');
        console.debug(`🔍 Table ${tableIndex + 1}: Fallback found ${fallbackRows.length} tr rows`);

        fallbackRows.forEach(row => parseTableRow(row, tableIndex, indicators));
      } else {
        rows.forEach(row => parseTableRow(row, tableIndex, indicators));
      }
    });

    function parseTableRow(row, tableIndex, indicators) {
      const cells = row.querySelectorAll('td');
      if (cells.length < 3) {
        console.debug(`🔍 Row skipped: only ${cells.length} cells`);
        return; // Ignorer les lignes avec moins de 3 colonnes
      }

      const name = cells[0]?.textContent?.trim();
      const valueText = cells[1]?.textContent?.trim();
      const thresholdText = cells[2]?.textContent?.trim();

      console.debug(`🔍 Raw row data: "${name}" | "${valueText}" | "${thresholdText}"`);

      // Vérifier que c'est une ligne de données valide
      if (name && valueText && !name.toLowerCase().includes('indicateur')) {
        // Extraire la valeur numérique (pourcentage ou nombre)
        const valueMatch = valueText.match(/[\d.,]+/);
        if (valueMatch) {
          const numericValue = parseFloat(valueMatch[0].replace(',', '.'));

          // Mapper les noms français vers nos clés standards
          const mappedKey = mapCryptoToolboxIndicatorName(name);
          if (mappedKey) {
            indicators[mappedKey] = {
              name: name,
              value_percent: numericValue,
              critical_threshold: thresholdText,
              source: 'Crypto-Toolbox',
              table: tableIndex + 1,
              raw_value: valueText,
              raw_threshold: thresholdText
            };

            console.debug(`✅ Mapped: ${name} → ${mappedKey} (${numericValue})`);
          } else {
            (window.debugLogger?.warn || console.warn)(`⚠️ Unmapped indicator: "${name}"`);
          }
        } else {
          (window.debugLogger?.warn || console.warn)(`⚠️ No numeric value found in: "${valueText}"`);
        }
      }
    }

    // Stratégie 2: Patterns regex en fallback
    if (Object.keys(indicators).length === 0) {
      console.debug('🔄 No table data found, trying regex patterns...');

      const patterns = [
        { name: 'mvrv', regex: /MVRV.*?([0-9.]+)%/gi, french: 'MVRV Z-Score' },
        { name: 'puell_multiple', regex: /Puell.*?([0-9.]+)%/gi, french: 'Puell Multiple' },
        { name: 'nupl', regex: /NUPL.*?([0-9.]+)%/gi, french: 'NUPL' },
        { name: 'rsi', regex: /RSI.*?([0-9.]+)%/gi, french: 'RSI Bitcoin' }
      ];

      patterns.forEach(pattern => {
        const match = pattern.regex.exec(html);
        if (match && match[1]) {
          indicators[pattern.name] = {
            name: pattern.french,
            value_percent: parseFloat(match[1]),
            source: 'Crypto-Toolbox (regex)',
            raw_value: match[0]
          };
          console.debug(`✅ Regex match: ${pattern.name} = ${match[1]}%`);
        }
      });
    }

  } catch (error) {
    (window.debugLogger?.error || console.error)('❌ Crypto-Toolbox HTML parsing failed:', error.message);
  }

  return indicators;
}

/**
 * Mappe les noms d'indicateurs français de Crypto-Toolbox vers nos clés standards
 */
function mapCryptoToolboxIndicatorName(frenchName) {
  const name = frenchName.toLowerCase().trim();

  // Mapping des noms français vers clés standards (basé sur ton code Python)
  const mappings = {
    // Indicateurs de données principaux
    'mvrv z-score': 'mvrv',
    'mvrv': 'mvrv',
    'puell multiple': 'puell_multiple',
    'puell': 'puell_multiple',
    'nupl': 'nupl',
    'rupl/nupl': 'nupl',
    'rsi mensuel': 'rsi',
    'rsi bitcoin': 'rsi',
    'rsi': 'rsi',
    'sopr (ma 90j)': 'sopr',
    'sopr': 'sopr',
    'coin days destroyed (ma 90j)': 'cdd',
    'coin days destroyed': 'cdd',
    'bmo (par prof. chaîne) (ema 7j)': 'bmo_7',
    'bmo (par prof. chaîne) (ema 30j)': 'bmo_30',
    'bmo (par prof. chaîne) (ema 90j)': 'bmo_90',
    'bmo': 'bmo',

    // Autres indicateurs techniques
    'cbbi*': 'cbbi',
    'pi cycle': 'pi_cycle',
    'rhold': 'rhold',
    '2y ma': 'ma_2y',
    'trolololo trend line': 'trolololo',
    'reserve risk': 'reserve_risk',
    'woobull': 'woobull',
    'cointime mvrv-z score (ema 14j)': 'mvrv_cointime',
    'mayer mutiple': 'mayer_multiple',
    'mayer multiple': 'mayer_multiple',

    // Indicateurs sentiment
    'fear & greed (moyenne 7 jours)': 'fear_greed_7d',
    'fear & greed': 'fear_greed',
    'fear and greed': 'fear_greed',

    // Dominance et altcoins
    'dominance btc': 'btc_dominance',
    'altcoin season index': 'altseason',
    'altcoin': 'altseason',

    // Indicateurs Google trends
    'google trend "crypto"': 'google_crypto',
    'google trend "buy crypto"': 'google_buy_crypto',
    'google trend "bitcoin"': 'google_bitcoin',
    'google trend "ethereum"': 'google_ethereum',

    // Indicateurs apps
    'coinbase app rank (us)': 'coinbase_rank_us',
    'binance app rank (fr)': 'binance_rank_fr',
    'binance app rank (uk)': 'binance_rank_uk',
    'crypto.com app rank (us)': 'crypto_com_rank_us',
    'phantom app rank (us)': 'phantom_rank_us',

    // Indicateurs temporels
    'jours depuis halving': 'days_since_halving',
    'days since ath': 'days_since_ath',
    'cycle time': 'cycle_time',

    // Autres
    'nombre de connectés jvc': 'jvc_users'
  };

  // Recherche exacte
  if (mappings[name]) {
    return mappings[name];
  }

  // Recherche par inclusion
  for (const [french, key] of Object.entries(mappings)) {
    if (name.includes(french) || french.includes(name)) {
      return key;
    }
  }

  (window.debugLogger?.warn || console.warn)(`⚠️ Unknown indicator name: "${frenchName}"`);
  return null;
}

/**
 * Convertit un pourcentage Crypto-Toolbox (0-100%) vers notre système de score
 * 0% = Accumulation extrême, 100% = Distribution extrême
 */
function convertCryptoToolboxPercentToScore(percent, isContrarian = false) {
  // Les pourcentages Crypto-Toolbox représentent la position dans le cycle
  // 0% = creux de marché, 100% = pic de marché

  if (isContrarian) {
    // Pour les indicateurs contrarian (Fear & Greed), inverser
    return Math.round(100 - percent);
  } else {
    // Pour les indicateurs standards, utiliser directement
    return Math.round(percent);
  }
}

/**
 * Récupère tous les indicateurs disponibles avec cache stable
 */
export async function fetchAllIndicators({ force = false } = {}) {
  console.debug('🔍 Fetching on-chain indicators...', { force });

  const indicators = {};
  const errors = [];

  // Check current data source configuration
  const dataSource = window.globalConfig?.get('data_source') || null;
  console.debug(`🎯 Current data source: ${dataSource}`);

  // Always request observed indicators. An API failure remains unavailable.
  try {
    // 1. Fetch all indicators from Crypto-Toolbox backend with SWR
    console.debug('🌐 Calling fetchCryptoToolboxIndicators with SWR...', { force });
    const cryptoToolboxResult = await fetchCryptoToolboxIndicators({ force });
    const cryptoToolboxData = cryptoToolboxResult?.indicators || cryptoToolboxResult;
    const observedAt = cryptoToolboxResult?.fetched_at || null;
    const stale = cryptoToolboxResult?.stale === true;
    console.debug('🔍 CryptoToolbox result:', cryptoToolboxData);

    const toolboxAvailable = !!(cryptoToolboxData && Object.keys(cryptoToolboxData).filter(k => !k.startsWith('_')).length > 0);
    if (toolboxAvailable) {
      // Process all indicators from the backend
      Object.entries(cryptoToolboxData).forEach(([key, data]) => {
        if (key.startsWith('_') || !data || typeof data !== 'object') {
          return; // Skip metadata
        }

        // Use the backend data directly without double conversion
        indicators[key] = {
          name: data.name,
          value: data.value_numeric, // Raw percentage value
          value_numeric: data.value_numeric,
          raw_value: data.raw_value,
          threshold_numeric: data.threshold_numeric,
          in_critical_zone: data.in_critical_zone,
          threshold: data.threshold,
          raw_threshold: data.raw_threshold,
          threshold_operator: data.threshold_operator,
          source: data.source || 'crypto-toolbox',
          scraped_at: data.scraped_at,
          timestamp: data.scraped_at || observedAt || null
        };

        console.debug(`✅ ${data.name} loaded: ${data.value_numeric}% ${data.in_critical_zone ? '🚨' : ''}`);
      });

      console.debug(`✅ Total ${Object.keys(indicators).length} indicators loaded from Crypto-Toolbox`);

    } else {
      errors.push('Crypto-Toolbox: Backend unavailable - no indicators loaded');
      if (_logLimiter.limit('backend_unavailable')) {
        (window.debugLogger?.warn || console.warn)('⚠️ Crypto-Toolbox backend failed, no indicators loaded');
      }
    }

    // 2. Add Fear & Greed from Alternative.me only if toolbox data is present (no silent fallback)
    const fearGreedExists = Object.keys(indicators).some(key =>
      indicators[key].name?.toLowerCase().includes('fear') &&
      indicators[key].name?.toLowerCase().includes('greed')
    );
    if (toolboxAvailable && !fearGreedExists) {
      console.debug('🔄 Adding Fear & Greed as fallback indicator...');
      const fgData = await fetchFearGreedIndex();
      if (fgData) {
        indicators.fear_greed_fallback = {
          name: 'Fear & Greed Index',
          value: fgData.value,
          value_numeric: fgData.value,
          classification: fgData.classification,
          source: 'alternative.me',
          timestamp: fgData.timestamp,
          in_critical_zone: fgData.value > 80 || fgData.value < 20
        };
        console.debug('✅ Fear & Greed fallback loaded:', fgData.value, fgData.classification);
      } else {
        errors.push('Fear & Greed fallback API also unavailable');
      }
    }

    const successCount = Object.keys(indicators).filter(k => k !== '_metadata').length;
    console.debug(`✅ Real indicators loaded: ${successCount} total indicators`);

    if (errors.length > 0) {
      (window.debugLogger?.warn || console.warn)('⚠️ Some fallback indicators unavailable:', errors);
    }

    // Log indicator summary by source
    const sourceStats = {};
    Object.values(indicators).forEach(ind => {
      if (ind.source) {
        sourceStats[ind.source] = (sourceStats[ind.source] || 0) + 1;
      }
    });

    console.debug('📊 Indicators by source:', sourceStats);

    return {
      ...indicators,
      _metadata: {
        available: successCount > 0,
        available_count: successCount,
        stale,
        served_from: cryptoToolboxResult?.served_from || null,
        cache_age_ms: cryptoToolboxResult?.cache_age_ms ?? null,
        observed_at: observedAt,
        missing_apis: errors,
        source_stats: sourceStats,
        message: `${successCount} real indicators loaded from Crypto-Toolbox backend.`,
        last_updated: observedAt
      }
    };

  } catch (error) {
    (window.debugLogger?.warn || console.warn)('❌ Real on-chain indicators unavailable:', error.message);
    return {
      _metadata: {
        available_count: 0,
        unavailable: true,
        api_error: error.message,
        message: `Real on-chain indicators unavailable: ${error.message}`,
        last_updated: null
      }
    };
  }
}

/**
 * Normalise et inverse un score d'indicateur selon sa configuration
 */
function normalizeAndInvertScore(rawValue, classification) {
  // Normaliser sur 0-100 (les valeurs Crypto-Toolbox sont déjà en %)
  let normalizedScore = Math.max(0, Math.min(100, rawValue));

  // Inverser si nécessaire (pour les indicateurs bearish quand élevés)
  if (classification.invert) {
    normalizedScore = 100 - normalizedScore;
  }

  return normalizedScore;
}

/**
 * Calcule un score composite amélioré avec catégorisation intelligente
 * Intègre les 30+ indicateurs réels de Crypto-Toolbox
 */
// Legacy V1 calculateCompositeScore removed - use calculateCompositeScoreV2 from composite-score-v2.js

/**
 * Combine le score de cycle sigmoïde avec les indicateurs on-chain
 */
export function enhanceCycleScore(sigmoidScore, onchainWeight = 0.3) {
  return new Promise(async (resolve) => {
    try {
      // Récupérer les indicateurs
      const indicators = await fetchAllIndicators();
      const composite = calculateCompositeScoreV2(indicators, true); // V2 with dynamic weighting

      // Blend des scores
      const enhancedScore = sigmoidScore * (1 - onchainWeight) + composite.score * onchainWeight;

      resolve({
        original_sigmoid: sigmoidScore,
        onchain_composite: composite.score,
        enhanced_score: Math.round(enhancedScore),
        onchain_weight: onchainWeight,
        confidence: composite.confidence,
        indicators_used: Object.keys(indicators),
        contributors: composite.contributors
      });

    } catch (error) {
      (window.debugLogger?.error || console.error)('Error enhancing cycle score:', error);
      resolve({
        available: false,
        original_sigmoid: sigmoidScore,
        enhanced_score: null,
        error: error.message,
        confidence: null
      });
    }
  });
}

/**
 * Analyse la divergence entre modèle sigmoïde et indicateurs
 */
export function analyzeDivergence(sigmoidScore, indicators) {
  const composite = calculateCompositeScoreV2(indicators, true); // V2 with dynamic weighting
  const divergence = Math.abs(sigmoidScore - composite.score);

  let signal = 'neutral';
  let message = '';

  if (divergence > 30) {
    if (composite.score > sigmoidScore) {
      signal = 'onchain_bullish';
      message = 'On-chain indicators more optimistic than cycle model';
    } else {
      signal = 'onchain_bearish';
      message = 'On-chain indicators more pessimistic than cycle model';
    }
  } else if (divergence > 15) {
    signal = 'moderate_divergence';
    message = 'Moderate divergence between model and indicators';
  } else {
    signal = 'convergence';
    message = 'Good convergence between model and indicators';
  }

  return {
    divergence_magnitude: divergence,
    signal,
    message,
    sigmoid_score: sigmoidScore,
    onchain_score: composite.score,
    confidence: composite.confidence
  };
}

/**
 * Recommandations basées sur les indicateurs
 */
export function generateRecommendations(enhancedData) {
  const recommendations = [];
  const { enhanced_score, contributors, confidence } = enhancedData;

  // Recommandations basées sur le score enhancé (IMPORTANT: score positif - plus haut = meilleur)
  if (enhanced_score > 80) {
    recommendations.push({
      type: 'warning',
      title: 'Probable Distribution Zone',
      message: 'High score - Euphoria detected, consider taking profits',
      action: 'Reduce altcoin exposure, increase stables'
    });
  } else if (enhanced_score >= 60) {
    recommendations.push({
      type: 'info',
      title: 'Bull Market Confirmed',
      message: 'Good score - Bullish momentum present but no euphoria',
      action: 'Maintain current allocation, watch for peak signals'
    });
  } else if (enhanced_score >= 40) {
    recommendations.push({
      type: 'neutral',
      title: 'Transition Zone',
      message: 'Average score - Uncertainty or consolidation phase',
      action: 'Caution recommended, wait for trend confirmation'
    });
  } else if (enhanced_score >= 30) {
    recommendations.push({
      type: 'caution',
      title: 'Weak Momentum Detected',
      message: 'Low score - Bearish on-chain indicators',
      action: 'Gradually reduce risky exposure (altcoins), increase stables'
    });
  } else {
    recommendations.push({
      type: 'opportunity',
      title: 'Probable Accumulation Zone',
      message: 'Very low score - Potential buying opportunity (capitulation/bear)',
      action: 'Consider gradually increasing Bitcoin/ETH allocation'
    });
  }

  // Recommandations spécifiques aux indicateurs
  if (contributors.length > 0) {
    const topContributor = contributors[0];

    if (topContributor.name === 'Fear & Greed Index' && topContributor.score > 80) {
      recommendations.push({
        type: 'contrarian',
        title: 'Extreme Greed Detected',
        message: 'Very optimistic sentiment - Contrarian signal',
        action: 'Caution recommended, potential peak nearby'
      });
    }

    if (topContributor.name === 'MVRV Ratio' && topContributor.score > 85) {
      recommendations.push({
        type: 'technical',
        title: 'MVRV in Danger Zone',
        message: 'High MVRV ratio - Historically near peaks',
        action: 'Monitor closely, prepare exit strategy'
      });
    }
  }

  // Recommandations basées sur la confiance
  if (confidence < 0.4) {
    recommendations.push({
      type: 'info',
      title: 'Limited Data',
      message: 'Few indicators available - Reduced confidence',
      action: 'Integrate more on-chain data sources'
    });
  }

  return recommendations;
}

/**
 * Retourne le statut et les sources de chaque indicateur
 */
export function getIndicatorSources() {
  return {
    fear_greed: {
      status: 'ACTIVE',
      source: 'Alternative.me API',
      url: 'https://api.alternative.me/fng/',
      reliability: 'Production ready',
      cost: 'Free',
      update_frequency: 'Daily'
    },
    mvrv: {
      status: 'SIMULATED',
      source: 'Historical patterns',
      possible_apis: [
        'Glassnode.com (Premium)',
        'CoinMetrics.io (Professional)',
        'LookIntoBitcoin.com (Charts only)'
      ],
      cost: '$39-$499/month',
      complexity: 'Medium',
      implementation_priority: 'High'
    },
    nvt: {
      status: 'SIMULATED',
      source: 'Cycle-based estimation',
      calculation: '(Market Cap) / (Transaction Volume USD * 365)',
      possible_data_sources: [
        'CoinGecko API (market cap) - Free',
        'Blockchain.info API (volume) - Free',
        'CoinMetrics (direct NVT) - Paid'
      ],
      complexity: 'Medium',
      implementation_priority: 'Medium'
    },
    puell_multiple: {
      status: 'SIMULATED',
      source: 'Mining revenue estimation',
      calculation: '(Daily Mining Revenue USD) / (365-day MA Mining Revenue)',
      required_data: [
        'Daily Bitcoin issuance (blocks * 6.25 BTC)',
        'Bitcoin price (CoinGecko)',
        'Transaction fees (Blockchain.info)'
      ],
      complexity: 'High',
      implementation_priority: 'Low'
    },
    rhodl: {
      status: 'NOT_IMPLEMENTED',
      source: 'N/A',
      reason: 'Requires complex UTXO age analysis',
      possible_apis: ['Glassnode (Premium only)'],
      cost: '$99+/month',
      complexity: 'Very High',
      implementation_priority: 'Very Low'
    }
  };
}

/**
 * Fournit des instructions pour intégrer de vraies APIs
 */
export function getImplementationGuide() {
  return {
    priority_1_nvt: {
      title: 'Integrate real NVT Ratio',
      steps: [
        '1. Retrieve Bitcoin market cap via free CoinGecko API',
        '2. Retrieve transaction volume via Blockchain.info API',
        '3. Calculer: NVT = MarketCap / (VolumeUSD * 365)',
        '4. Remplacer simulateNVT() par fetchRealNVT()'
      ],
      estimated_time: '4-6 heures',
      difficulty: 'Medium'
    },
    priority_2_mvrv: {
      title: 'Integrate MVRV Ratio via paid API',
      steps: [
        '1. Subscribe to Glassnode API (Studio plan $39/month)',
        '2. Implement fetchGlassnodeMVRV() with authentication',
        '3. Handle rate limiting',
        '4. Fallback sur simulation si API indisponible'
      ],
      estimated_time: '6-8 heures',
      difficulty: 'Medium-High'
    },
    priority_3_puell: {
      title: 'Calculate real Puell Multiple',
      steps: [
        '1. Retrieve block data via Blockchain.info',
        '2. Calculer revenus miniers quotidiens',
        '3. Implement 365-day moving average',
        '4. Optimiser pour performance (cache agressif)'
      ],
      estimated_time: '8-12 heures',
      difficulty: 'High'
    }
  };
}

// ⚙️ Unifier l’URL d’import du module (pas de ?v=3 ici)
import { cycleScoreFromMonths, getCurrentCycleMonths } from './cycle-navigator.js';
// Note: calibrateCycleParams & getCycleParams seront importés à la volée, ce qui
// évite tout problème si le bouton est injecté après coup.

// 🔗 CSP Compliance: Replace inline onclick handlers with event listeners
document.addEventListener('click', async (e) => {
  // Calibration button (existing)
  const calibrateBtn = e.target?.closest('#btn-calibrate');
  if (calibrateBtn) {
    try {
      const { calibrateCycleParams, getCycleParams } = await import('./cycle-navigator.js');
      const res = calibrateCycleParams(); // ancres par défaut
      console.debug('Cycle params calibrés:', getCycleParams(), 'score=', res.score);
      // ✅ Redessiner le même canvas (force refresh after calibration)
      if (window.Chart) {
        await createBitcoinCycleChart('bitcoin-cycle-chart', true);
      } else {
        debugLogger.debug('📊 Chart.js not loaded, skipping chart refresh');
      }
      // Also clear cycle content cache as calibration affects data
      const cycleContentConfig = CACHE_CONFIG.CYCLE_CONTENT;
      if (cycleContentConfig) localStorage.removeItem(cycleContentConfig.key);
      window.showToast?.('Cycle calibrated on historical data', 'success');
    } catch (e) {
      debugLogger.error(e);
      window.showToast?.('Calibration failed', 'error');
    }
    return;
  }

  // Refresh button
  if (e.target?.closest('#refresh-btn')) {
    refreshDashboard();
    return;
  }

  // Options menu toggle
  if (e.target?.closest('#options-menu-btn')) {
    toggleOptionsMenu(e);
    return;
  }

  // Menu items
  if (e.target?.closest('#force-refresh-btn')) {
    closeOptionsMenu();
    refreshDashboard(true);
    return;
  }

  if (e.target?.closest('#force-cycle-refresh-btn')) {
    closeOptionsMenu();
    forceCycleRefresh();
    return;
  }

  if (e.target?.closest('#test-endpoint-btn')) {
    closeOptionsMenu();
    testEndpoint();
    return;
  }

  // View alerts history button
  if (e.target?.closest('[data-action="view-alerts-history"]')) {
    switchTab('alerts');
    return;
  }

  // Tab switching (CSP-compliant replacement for onclick)
  const tabButton = e.target?.closest('.tab-button[data-tab]');
  if (tabButton) {
    const tabName = tabButton.getAttribute('data-tab');
    if (tabName && typeof switchTab === 'function') {
      switchTab(tabName);
    }
    return;
  }

  // Alerts filters and refresh
  if (e.target?.closest('[data-action="filter-alerts"]')) {
    if (typeof filterAlertsHistory === 'function') {
      filterAlertsHistory();
    }
    return;
  }

  if (e.target?.closest('[data-action="refresh-alerts"]')) {
    if (typeof refreshAlertsHistory === 'function') {
      refreshAlertsHistory();
    }
    return;
  }

  // Alerts pagination
  if (e.target?.closest('[data-action="alerts-prev"]')) {
    if (typeof loadPreviousAlertsPage === 'function') {
      loadPreviousAlertsPage();
    }
    return;
  }

  if (e.target?.closest('[data-action="alerts-next"]')) {
    if (typeof loadNextAlertsPage === 'function') {
      loadNextAlertsPage();
    }
    return;
  }

  // Clear all toasts
  if (e.target?.closest('[data-action="clear-all-toasts"]')) {
    if (typeof hideAllToasts === 'function') {
      hideAllToasts();
    }
    return;
  }

  // Alert modal actions
  if (e.target?.closest('[data-action="close-alert-modal"]')) {
    if (typeof closeAlertModal === 'function') {
      closeAlertModal();
    }
    return;
  }

  if (e.target?.closest('[data-action="snooze-alert"]')) {
    if (typeof snoozeCurrentAlert === 'function') {
      snoozeCurrentAlert();
    }
    return;
  }

  if (e.target?.closest('[data-action="acknowledge-alert"]')) {
    if (typeof acknowledgeCurrentAlert === 'function') {
      acknowledgeCurrentAlert();
    }
    return;
  }

  if (e.target?.closest('[data-action="apply-action"]')) {
    if (typeof applyAction === 'function') {
      applyAction();
    }
    return;
  }
});

// 🆕 Oct 2025: Toggle breakdown panel (expose globally for onclick handlers)
window.toggleBreakdown = function (panelId) {
  const panel = document.getElementById(panelId);
  if (panel) {
    const isVisible = panel.style.display !== 'none';
    panel.style.display = isVisible ? 'none' : 'block';
  }
};

// Import core modules
import { store } from '../core/risk-dashboard-store.js';
import { fetchCached, clearCache } from '../core/fetcher.js';

// Import CCS modules
import { fetchAndComputeCCS, interpretCCS, DEFAULT_CCS_WEIGHTS } from './signals-engine.js';
import { estimateCyclePosition, blendCCS, getCyclePhase } from './cycle-navigator.js';
import { proposeTargets, applyTargets, computePlan, DEFAULT_MACRO_TARGETS, getDecisionLog } from './targets-coordinator.js';
import { fetchAllIndicators, enhanceCycleScore } from './onchain-indicators.js';
import { calculateCompositeScoreV2, analyzeContradictorySignals } from './composite-score-v2.js';
import { getMarketRegime, applyMarketOverrides, calculateRiskBudget, getRegimeDisplayData } from './market-regimes.js';

// Import tab-specific modules
// Note: Alerts and Risk Overview tabs use dynamic imports in risk-dashboard-main.js
import { renderCyclesContent, renderCyclesContentUncached, createBitcoinCycleChart, loadOnChainIndicators, fetchBitcoinHistoricalData } from './risk-cycles-tab.js';
import { renderTargetsContent, getCurrentPortfolioAllocation } from './risk-targets-tab.js';

// Import utilities and constants (Oct 2025 refactoring - ultra-conservative)
import { safeFixed, formatPercent, formatNumber, formatMoney, scoreToRiskLevel, pickScoreColor, getScoreInterpretation } from './risk-utils.js';
import { CACHE_CONFIG } from './risk-constants.js';

// Import toast system (Feb 2026 - extracted module)
import { showToast, showS3AlertToast, loadDismissedAlerts, shouldShowAlert } from './risk-dashboard-toasts.js';

// Import alerts history system (Feb 2026 - extracted module)
import {
  loadAlertsHistory,
  refreshAlertsHistory,
  initializeAlertsTab,
  formatAlertType,
  formatRelativeTime,
  getAlertTypeDisplayName,
  loadPreviousAlertsPage,
  loadNextAlertsPage
} from './risk-dashboard-alerts-history.js';

// Global state
let autoRefreshInterval = null;
let isRefreshing = false;

// Variables pour les scores
let onchainScore = null;
let riskScore = null;
let blendedScore = null;

// ===== PERSISTENT CACHE SYSTEM =====
// Note: CACHE_CONFIG now imported from risk-constants.js

function clearAllPersistentCache() {
  // Clear both old format and new data-source-aware format cache entries
  const dataSource = globalConfig.get('data_source') || 'unknown';

  Object.values(CACHE_CONFIG).forEach(config => {
    // Clear old format
    localStorage.removeItem(config.key);
    // Clear new data-source-aware format
    localStorage.removeItem(`${config.key}_${dataSource}`);

    // Also clear other potential data sources to ensure complete cleanup
    ['cointracking', 'cointracking_api', 'api', 'unknown'].forEach(source => {
      localStorage.removeItem(`${config.key}_${source}`);
    });
  });

  // ✅ FIX: Also clear risk-dashboard-balance cache with pattern matching
  // Format: risk-dashboard-balance:user:source:minUsd
  Object.keys(localStorage).forEach(key => {
    if (key.startsWith('risk-dashboard-balance:')) {
      localStorage.removeItem(key);
      debugLogger.debug(`🧹 Cleared balance cache: ${key}`);
    }
  });

  debugLogger.debug(`🧹 All persistent cache cleared for source ${dataSource}`);
}

// Re-render on currency change (ensure rate then refresh)
window.addEventListener('configChanged', (ev) => {
  const key = ev?.detail?.key;
  if (key === 'display_currency') {
    const cur = (window.globalConfig && window.globalConfig.get('display_currency')) || 'USD';
    const rerender = () => { try { refreshDashboard(true); } catch (_) { } };
    if (window.currencyManager && cur !== 'USD') {
      window.currencyManager.ensureRate(cur).then(rerender).catch(rerender);
    } else {
      rerender();
    }
  }
});

// Fonction pour charger les paramètres de calibration depuis localStorage
function loadCalibrationParams() {
  try {
    const saved = localStorage.getItem('bitcoin_cycle_params');
    if (saved) {
      const data = JSON.parse(saved);
      // Vérifier que les données ne sont pas trop anciennes (24h)
      if (Date.now() - data.timestamp < 24 * 60 * 60 * 1000) {
        console.debug('✅ Paramètres calibrés chargés depuis localStorage', data.params);
        return data.params;
      }
    }
  } catch (error) {
    debugLogger.error('❌ Erreur chargement paramètres:', error);
  }
  return null;
}

// Écouter les mises à jour de paramètres depuis d'autres pages
window.addEventListener('message', (event) => {
  if (event.data.type === 'CYCLE_PARAMS_UPDATED') {
    console.debug('🔄 Paramètres de cycle mis à jour depuis autre page');
    // Recharger les données avec les nouveaux paramètres
    setTimeout(refreshDashboard, 1000);
  }
});

// Track current data source to detect changes
let lastKnownDataSource = globalConfig.get('data_source');
console.debug(`📊 Risk Dashboard initialized with data source: ${lastKnownDataSource}`);

// Listen for data source changes from settings
window.addEventListener('storage', function (e) {
  const expectedKey = (window.globalConfig?.getStorageKey && window.globalConfig.getStorageKey()) || 'crypto_rebal_settings_v1';
  if (e.key === expectedKey) {
    console.debug('Settings changed in another tab, checking for data source changes...');

    const currentSource = globalConfig.get('data_source');
    if (currentSource && currentSource !== lastKnownDataSource) {
      console.debug(`🔄 Data source changed from ${lastKnownDataSource} to ${currentSource}, auto-refreshing...`);
      lastKnownDataSource = currentSource;
      // Clear cache when source changes
      clearAllPersistentCache();
      setTimeout(() => refreshDashboard(true), 500); // Force refresh to recalculate with new source
    }
  }
});

// Listen for explicit data source change events
window.addEventListener('dataSourceChanged', (event) => {
  console.debug(`🔄 Explicit data source change: ${event.detail.oldSource} → ${event.detail.newSource}`);
  lastKnownDataSource = event.detail.newSource;
  // Clear cache when source changes
  clearAllPersistentCache();
  setTimeout(() => refreshDashboard(true), 500); // Force refresh to recalculate with new source

  // Reload Phase 3A data with new source/user context
  setTimeout(() => {
    if (document.querySelector('.advanced-risk-panel') && typeof loadPhase3AData === 'function') {
      console.debug('🔄 Reloading Phase 3A data after source change...');
      loadPhase3AData().catch(e => debugLogger.warn('Phase 3A reload failed:', e.message));
    }
  }, 1000);
});

// Listen for active user changes to reload Phase 3A data
window.addEventListener('activeUserChanged', (event) => {
  console.debug(`🔄 Active user changed: ${event.detail?.oldUser || 'unknown'} → ${event.detail?.newUser || 'unknown'}`);
  // Reload Phase 3A data with new user context
  setTimeout(() => {
    if (document.querySelector('.advanced-risk-panel') && typeof loadPhase3AData === 'function') {
      console.debug('🔄 Reloading Phase 3A data after user change...');
      loadPhase3AData().catch(e => debugLogger.warn('Phase 3A reload failed:', e.message));
    }
  }, 500);
});

// ====== Tab Management ======
window.switchTab = function (tabName) {
  // Update UI state
  document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

  document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
  document.getElementById(`${tabName}-tab`).classList.add('active');

  // Update store
  store.set('ui.activeTab', tabName);

  // Render content based on active tab
  // Use requestAnimationFrame to ensure DOM is ready after CSS changes
  switch (tabName) {
    case 'risk':
      // Risk content is already rendered by refreshDashboard
      break;
    case 'cycles':
      // Always render cycles content when tab is activated
      requestAnimationFrame(() => renderCyclesContent());
      break;
    case 'advanced':
      // Load Phase 3A Advanced Risk components
      requestAnimationFrame(() => {
        if (typeof window.loadAdvancedRiskComponents === 'function') {
          window.loadAdvancedRiskComponents().catch(err =>
            debugLogger.error('Failed to load advanced risk components:', err)
          );
        }
      });
      break;
    case 'targets':
      requestAnimationFrame(() => {
        renderTargetsContent().catch(err => debugLogger.error('Failed to render targets:', err));
      });
      break;
    case 'alerts':
      // Wait for DOM to be ready (tab-pane display:block applied)
      requestAnimationFrame(() => {
        requestAnimationFrame(() => initializeAlertsTab());
      });
      break;
  }

  console.debug(`Switched to tab: ${tabName}`);
};

// ====== Tooltip helpers ======
const $tip = document.getElementById('tooltip');
const $tipTitle = $tip?.querySelector('.tip-title');
const $tipBody = $tip?.querySelector('.tip-body');

function showTip(title, body, x, y) {
  if (!$tip) return;
  $tipTitle.textContent = title || '';
  $tipBody.textContent = body || '';
  $tip.style.left = x + 'px';
  $tip.style.top = y + 'px';
  $tip.classList.add('show');
  $tip.setAttribute('aria-hidden', 'false');
}

function moveTip(x, y) {
  if (!$tip) return;
  $tip.style.left = x + 'px';
  $tip.style.top = y + 'px';
}

function hideTip() {
  if (!$tip) return;
  $tip.classList.remove('show');
  $tip.setAttribute('aria-hidden', 'true');
}

function attachTip(el, title, body) {
  if (!el) return;
  el.addEventListener('mouseenter', e => showTip(title, body, e.clientX, e.clientY));
  el.addEventListener('mousemove', e => moveTip(e.clientX, e.clientY));
  el.addEventListener('mouseleave', hideTip);
  el.classList.add('hinted');
}

// ====== Formatters ======
const pct = v => (v == null || isNaN(v) ? 'N/A' : (v * 100).toFixed(2) + '%');
const num = v => (v == null || isNaN(v) ? 'N/A' : Number(v).toFixed(2));

// ====== Analysis Window Slider ======
// Fixed windows
const analysisDays = 365;
const corrDays = 90;
function updateWindowLabel(v) {
  const lbl = document.getElementById('window-label');
  if (lbl) lbl.textContent = `${v} jours`;
}
// Removed old controls

// ====== API Functions ======
async function fetchRiskData() {
  try {
    // Get the configured data source dynamically
    const dataSource = globalConfig.get('data_source');
    const apiBaseUrl = globalConfig.get('api_base_url');
    const minUsd = globalConfig.get('min_usd_threshold');

    console.debug(`🔍 Risk Overview using data source: ${dataSource}`);

    // Use globalConfig to build the API URL with the configured source
    const url = globalConfig.getApiUrl('/balances/current', { min_usd: minUsd });

    // Utiliser directement les données de balance et calculer le risque côté client
    // 🔧 FIX: Include csv_selected_file in cache key for proper isolation
    // Use window.userSettings directly (updated by WealthContextBar before event emission)
    const csvFile = window.userSettings?.csv_selected_file || 'latest';
    const saxoFile = window.userSettings?.saxo_selected_file || 'latest';

    console.debug(`🔍 DEBUG - csvFile from window.userSettings: '${csvFile}'`);
    console.debug(`🔍 DEBUG - saxoFile from window.userSettings: '${saxoFile}'`);

    const cacheKey = `risk-dashboard-balance:${localStorage.getItem('activeUser')}:${dataSource}:${csvFile}:${minUsd}`;

    console.debug(`🔍 fetchRiskData - csvFile: '${csvFile}', dataSource: '${dataSource}', cacheKey: '${cacheKey}'`);

    const balanceResult = await fetchCached(
      cacheKey,
      () => window.globalConfig.apiRequest('/balances/current', { params: { source: dataSource, min_usd: minUsd } }),
      'risk'
    );

    // Use the real backend endpoint for risk dashboard
    // ✅ Inclure source et user_id pour isolation multi-tenant
    // ✅ NOUVEAU (Phase 5.5): Shadow Mode V2 + Dual Window
    // 🔧 FIX: Add cache buster to force backend recalculation when CSV changes
    const cacheBuster = csvFile !== 'latest' ? csvFile : Date.now().toString().substring(0, 10);

    console.debug(`🔍 fetchRiskData - calling /api/risk/dashboard with _csv_hint: '${cacheBuster}'`);

    const apiResult = await window.globalConfig.apiRequest('/api/risk/dashboard', {
      params: {
        source: dataSource,
        min_usd: minUsd,
        price_history_days: analysisDays,
        lookback_days: corrDays,
        risk_version: 'v2_active',  // 🆕 V2 Active: V2 est autoritaire (Oct 2025)
        use_dual_window: true,       // Dual-window metrics actives
        _csv_hint: cacheBuster        // 🔧 Hint for backend cache: changes when CSV changes
      }
    });

    // 🔍 DEBUG: Log la réponse brute COMPLÈTE pour diagnostiquer les erreurs
    console.debug('🔍 Raw API response (full object):', apiResult);
    console.debug('🔍 Has risk_metrics?', !!apiResult?.risk_metrics);
    console.debug('🔍 Response keys:', apiResult ? Object.keys(apiResult) : 'null');

    // Vérifier que apiResult est valide avant de l'utiliser
    if (!apiResult || !apiResult.risk_metrics) {
      debugLogger.error('❌ Invalid API response structure:', {
        hasApiResult: !!apiResult,
        hasRiskMetrics: !!apiResult?.risk_metrics,
        responseType: typeof apiResult,
        responseKeys: apiResult ? Object.keys(apiResult) : null,
        fullResponse: apiResult
      });
      throw new Error(`Invalid API response: ${apiResult ? 'missing risk_metrics' : 'null response'} - Check backend logs`);
    }

    // 🔍 DEBUG: Log la réponse brute avec nouveaux champs V2 (seulement si valide)
    console.debug('🔍 Parsed API response (Shadow Mode V2):', JSON.stringify({
      // Legacy scores
      sharpe_legacy: apiResult.risk_metrics.sharpe_ratio,
      var95: apiResult.risk_metrics.var_95_1d,
      risk_score_legacy: apiResult.risk_metrics.risk_score,
      structural_legacy: apiResult.risk_metrics.risk_score_structural,
      window_used: apiResult.risk_metrics.window_used,
      // V2 Shadow Mode info (🔧 FIX: Chemin correct!)
      risk_version_info: apiResult.risk_metrics.risk_version_info ? {
        active_version: apiResult.risk_metrics.risk_version_info.active_version,
        risk_score_v2: apiResult.risk_metrics.risk_version_info.risk_score_v2,
        sharpe_v2: apiResult.risk_metrics.risk_version_info.sharpe_v2,
        portfolio_structure_score: apiResult.risk_metrics.risk_version_info.portfolio_structure_score,
        integrated_structural_legacy: apiResult.risk_metrics.risk_version_info.integrated_structural_legacy
      } : null
    }));

    // Inclure les balances pour calculer concentration/stablecoins côté UI
    try {
      apiResult.balances = Array.isArray(balanceResult?.items) ? balanceResult.items : [];
    } catch (_) { /* ignore */ }

    const m = apiResult.risk_metrics;
    debugLogger.debug(`🧪 SHADOW V2 - Risk metrics from API: VaR 95%: ${(m.var_95_1d * 100).toFixed(2)}%, Sharpe: ${m.sharpe_ratio.toFixed(2)}, Risk Score: ${m.risk_score} (authoritative), Structural: ${m.risk_score_structural || 'N/A'}, Window: ${m.window_used?.actual_data_points || '?'} pts, risk_version_info: ${m.risk_version_info ? 'PRESENT ✅' : 'MISSING ❌'}`);

    // The backend already provides the correct structure, just return it
    return apiResult;
  } catch (error) {
    debugLogger.error('❌ Risk API call failed:', {
      errorMessage: error.message,
      errorStack: error.stack,
      errorType: error.constructor.name,
      dataSource: globalConfig.get('data_source'),
      apiBaseUrl: globalConfig.get('api_base_url')
    });
    return {
      success: false,
      message: `Risk backend unavailable: ${error.message}. Make sure the backend server is running and external APIs (CoinGecko) are not rate-limited.`,
      error_type: 'connection_error',
      error_details: error.message
    };
  }
}

// NOTE: getMockRiskData() supprimée - plus de données simulées

// CSV parsing functions are now centralized in global-config.js

// NOTE: calculateRealRiskMetrics() supprimée - métriques calculées par le backend Python

// ====== Metric Health Evaluation & Interpretation ======
function getMetricHealth(key, value) {
  const healthRules = {
    'var_95_1d': {
      good: [0, 0.04], // 0% to 4%
      warning: [0.04, 0.08], // 4% to 8%
      danger: [0.08, 1], // > 8%
      interpretation: {
        good: "Contained potential daily loss",
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
        warning: "Drawdown crypto typique",
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
        danger: "Fragile portfolio - high risk",
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
        interpretation: rule.interpretation[level] || 'Pas d\'interprétation disponible'
      };
    }
  }

  return { level: 'unknown', color: '#6b7280', interpretation: 'Valeur hors limites' };
}

function getContextualBenchmark(key, value) {
  const benchmarks = {
    'var_95_1d': {
      crypto_conservative: 0.04,
      crypto_typical: 0.07,
      crypto_aggressive: 0.12,
      traditional: 0.02
    },
    'sharpe_ratio': {
      crypto_excellent: 1.5,
      crypto_good: 1.0,
      crypto_acceptable: 0.5,
      traditional_good: 1.0
    },
    'volatility_annualized': {
      crypto_low: 0.4,
      crypto_typical: 0.6,
      crypto_high: 1.0,
      traditional: 0.2
    },
    'max_drawdown': {
      crypto_good: -0.3,
      crypto_typical: -0.5,
      crypto_extreme: -0.8,
      traditional: -0.2
    }
  };

  return benchmarks[key] || {};
}

function generateRecommendations(metrics, correlations, groups, fullData) {
  const recommendations = [];

  // VaR recommendations (VaR renvoyé en valeur positive)
  // ⚠️ MODIFIÉ (Phase 1.1): Suppression % stables hardcodé, branché sur risk_budget API
  if (metrics.var_95_1d > 0.08) {
    const riskBudget = fullData?.risk_budget || fullData?.regime?.risk_budget;
    const targetStables = riskBudget?.target_stables_pct;

    let action = 'Increase the share of stablecoins or Bitcoin to reduce volatility';
    if (typeof targetStables === 'number') {
      action = `Recommended stables allocation: ${targetStables}% (calculated according to your risk profile)`;
    }

    recommendations.push({
      priority: 'high',
      icon: '🛡️',
      title: 'Reduce daily loss risk',
      description: 'Your VaR of ' + formatPercent(metrics.var_95_1d) + ' is high.',
      action: action
    });
  }

  // Sharpe ratio recommendations
  if (metrics.sharpe_ratio < 1.0) {
    recommendations.push({
      priority: 'medium',
      icon: '📈',
      title: 'Improve risk-adjusted return',
      description: 'Sharpe ratio of ' + safeFixed(metrics.sharpe_ratio) + ' - look for assets with a better risk/return ratio.',
      action: 'Consider reducing memecoins, increasing BTC/ETH'
    });
  }

  // Diversification recommendations (alignée aux seuils UI)
  if (correlations.diversification_ratio < 0.4) {
    recommendations.push({
      priority: 'high',
      icon: '🔄',
      title: 'Improve diversification',
      description: 'Very low diversification ratio (' + safeFixed(correlations.diversification_ratio) + '). Portfolio too correlated.',
      action: 'Add uncorrelated assets: privacy coins, stablecoins, different sectors'
    });
  } else if (correlations.diversification_ratio < 0.7) {
    recommendations.push({
      priority: 'medium',
      icon: '🔄',
      title: 'Improve diversification',
      description: 'Limited diversification (' + safeFixed(correlations.diversification_ratio) + ').',
      action: 'Broaden sectors and reduce highly correlated pairs'
    });
  }

  // Effective assets recommendations  
  if (correlations.effective_assets < 3) {
    recommendations.push({
      priority: 'medium',
      icon: '⚖️',
      title: 'Reduce concentration',
      description: 'Portfolio se comporte comme ' + safeFixed(correlations.effective_assets, 1) + ' actifs seulement.',
      action: 'Rebalance: limit any asset to <20% of portfolio'
    });
  }

  // Drawdown recommendations (max_drawdown renvoyé en valeur positive)
  if (metrics.max_drawdown > 0.6) {
    recommendations.push({
      priority: 'high',
      icon: '📉',
      title: 'Protect against extreme drops',
      description: 'Max drawdown of ' + formatPercent(metrics.max_drawdown) + ' very high.',
      action: 'Defensive strategy: DCA, stop-loss, or hedging with stablecoins'
    });
  }

  // High correlation recommendations
  if (correlations.top_correlations) {
    const highCorrels = correlations.top_correlations.filter(c => Math.abs(c.correlation) > 0.75);
    if (highCorrels.length > 0) {
      recommendations.push({
        priority: 'medium',
        icon: '🔗',
        title: 'Reduce high correlations',
        description: 'Correlations >75% detected between ' + highCorrels.map(c => c.asset1 + '-' + c.asset2).join(', '),
        action: 'Diversify towards less correlated sectors (BTC vs ETH vs niche sectors)'
      });
    }
  }

  // If everything is good, add positive reinforcement
  if (recommendations.length === 0) {
    recommendations.push({
      priority: 'low',
      icon: '✅',
      title: 'Well-balanced portfolio',
      description: 'Your risk metrics are within acceptable crypto standards.',
      action: 'Continue monitoring and adjust according to market conditions'
    });
  }

  // Sort by priority
  const priorityOrder = { 'high': 0, 'medium': 1, 'low': 2 };
  return recommendations.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);
}

// NOTE: calculateCorrelationMetrics() supprimée - corrélations calculées par le backend Python

// Import unified asset grouping functions with forced taxonomy reload
let groupAssetsByClassification, getAssetGroup;

// Load unified asset groups dynamically with forced taxonomy reload
async function initAssetGroups() {
  try {
    console.debug('🔄 [Risk Dashboard] Force reloading taxonomy for proper asset classification...');
    const module = await import('../shared-asset-groups.js');
    await module.forceReloadTaxonomy();

    groupAssetsByClassification = module.groupAssetsByClassification;
    getAssetGroup = module.getAssetGroup;

    if (!Object.keys(module.UNIFIED_ASSET_GROUPS || {}).length) {
      debugLogger.warn('⚠️ [Risk Dashboard] Taxonomy non chargée – risque de "Others" gonflé');
    } else {
      debugLogger.debug('✅ [Risk Dashboard] Taxonomy loaded:', Object.keys(module.UNIFIED_ASSET_GROUPS).length, 'groupes');
    }
  } catch (error) {
    debugLogger.error('❌ [Risk Dashboard] Failed to load taxonomy:', error);
  }
}

// Initialize asset groups on page load
initAssetGroups();

async function groupAssetsByAliases(items) {
  console.debug('🔍 [Risk Dashboard] Grouping', items.length, 'assets by unified taxonomy');

  // Utiliser le système unifié si disponible
  if (groupAssetsByClassification) {
    return groupAssetsByClassification(items);
  }

  // Si pas encore chargé, charger maintenant
  try {
    console.debug('⏳ [Risk Dashboard] Taxonomy not loaded yet, importing now...');
    const module = await import('../shared-asset-groups.js');
    await module.forceReloadTaxonomy();
    groupAssetsByClassification = module.groupAssetsByClassification;
    getAssetGroup = module.getAssetGroup;
    return groupAssetsByClassification(items);
  } catch (error) {
    // Fallback simple si le module n'est pas encore chargé
    debugLogger.warn('⚠️ [Risk Dashboard] Unified groups failed to load, using simple fallback:', error);
    return items.map(item => ({
      label: item.symbol,
      value: parseFloat(item.value_usd || 0),
      assets: [item.symbol]
    }));
  }
}

// ====== CCS Functions ======
async function loadCCSData() {
  try {
    console.debug('Loading CCS data...');

    // Fetch CCS and cycle data in parallel
    const [ccsResult, cycleData] = await Promise.all([
      fetchAndComputeCCS(DEFAULT_CCS_WEIGHTS),
      estimateCyclePosition()
    ]);

    // Update store with CCS data
    store.set('ccs.score', ccsResult.score);
    store.set('ccs.signals', ccsResult.signals);
    store.set('ccs.weights', ccsResult.weights);
    store.set('ccs.lastUpdate', new Date().toISOString());
    store.set('ccs.model_version', ccsResult.model_version);

    // Update store with cycle data
    store.set('cycle.months', cycleData.months);
    store.set('cycle.phase', cycleData.phase);
    store.set('cycle.score', cycleData.score);
    store.set('cycle.confidence', cycleData.confidence);
    store.set('cycle.multipliers', cycleData.multipliers);

    // Update API status
    store.set('ui.apiStatus.signals', 'healthy');

    console.debug(`CCS loaded: ${ccsResult.score}, Cycle: ${cycleData.phase.phase} (${Math.round(cycleData.months)}m)`);

    return { ccs: ccsResult, cycle: cycleData };

  } catch (error) {
    debugLogger.error('Failed to load CCS data:', error);
    store.set('ui.apiStatus.signals', 'error');
    throw error;
  }
}

async function loadBlendedCCS() {
  try {
    const state = store.snapshot();
    const ccsScore = state.ccs?.score;
    const cycleMonths = state.cycle?.months;
    const cycleWeight = state.cycle?.weight || 0.3;

    if (ccsScore && cycleMonths) {
      const blended = blendCCS(ccsScore, cycleMonths, cycleWeight);
      store.set('cycle.ccsStar', blended.blendedCCS);
      return blended;
    }

    return null;
  } catch (error) {
    debugLogger.error('Failed to blend CCS:', error);
    return null;
  }
}

// ====== Alerts Functions ======
async function fetchAlertsData() {
  try {
    console.debug('Loading alerts data...');

    const alertsData = await window.globalConfig.apiRequest('/api/alerts/active', {
      params: { include_snoozed: false }
    });
    console.debug(`Alerts loaded: ${alertsData.length} active alerts`);

    return alertsData;

  } catch (error) {
    debugLogger.error('Failed to load alerts data:', error);
    throw error;
  }
}

function updateAlertsDisplay(alertsData) {
  const alertsDot = document.getElementById('alerts-dot');
  const alertsText = document.getElementById('alerts-text');
  const alertsList = document.getElementById('alerts-list');

  if (!alertsDot || !alertsText || !alertsList) {
    // Elements are now in Web Component <risk-sidebar-full>, skip legacy update
    console.debug('Alerts display: skipping legacy DOM update (elements in Web Component)');
    return;
  }

  if (!alertsData || alertsData.length === 0) {
    // No alerts
    alertsDot.className = 'status-dot healthy';
    alertsText.textContent = 'No active alerts';
    alertsList.replaceChildren(); // CSP safe replacement for innerHTML = ''

    // Always show the "View All History" button even when no active alerts
    const historyButton = document.createElement('div');
    historyButton.className = 'alerts-button-container';

    const viewHistoryBtn = document.createElement('button');
    viewHistoryBtn.className = 'alerts-history-btn';
    viewHistoryBtn.textContent = '📋 View All History';
    viewHistoryBtn.setAttribute('data-action', 'view-alerts-history');

    historyButton.appendChild(viewHistoryBtn);
    alertsList.appendChild(historyButton);
    return;
  }

  // Count alerts by severity
  const severityCounts = { S1: 0, S2: 0, S3: 0 };
  alertsData.forEach(alert => {
    if (severityCounts.hasOwnProperty(alert.severity)) {
      severityCounts[alert.severity]++;
    }
  });

  // Set status based on highest severity
  let statusClass = 'healthy';
  let statusText = `${alertsData.length} alert${alertsData.length > 1 ? 's' : ''}`;

  if (severityCounts.S3 > 0) {
    statusClass = 'error';
    statusText = `${severityCounts.S3} critical alert${severityCounts.S3 > 1 ? 's' : ''}`;
  } else if (severityCounts.S2 > 0) {
    statusClass = 'warning';
    statusText = `${severityCounts.S2} warning alert${severityCounts.S2 > 1 ? 's' : ''}`;
  }

  alertsDot.className = `status-dot ${statusClass}`;
  alertsText.textContent = statusText;

  // Render alerts list
  alertsList.replaceChildren(); // CSP safe replacement for innerHTML = ''

  // Sort alerts by severity (S3, S2, S1) and creation time
  const sortedAlerts = [...alertsData].sort((a, b) => {
    const severityOrder = { S3: 0, S2: 1, S1: 2 };
    const severityDiff = severityOrder[a.severity] - severityOrder[b.severity];
    if (severityDiff !== 0) return severityDiff;

    // Then by creation time (newest first)
    return new Date(b.created_at) - new Date(a.created_at);
  });

  // Limit to 5 most important alerts in sidebar
  sortedAlerts.slice(0, 5).forEach(alert => {
    const alertItem = createAlertItem(alert);
    alertsList.appendChild(alertItem);
  });

  // Add "show more" indicator if there are more alerts
  if (sortedAlerts.length > 5) {
    const moreItem = document.createElement('div');
    moreItem.style.textAlign = 'center';
    moreItem.style.padding = '4px';
    moreItem.style.fontSize = '0.7rem';
    moreItem.style.color = 'var(--theme-text-muted)';
    moreItem.textContent = `+${sortedAlerts.length - 5} more alerts...`;
    alertsList.appendChild(moreItem);
  }
}

function createAlertItem(alert) {
  const item = document.createElement('div');
  item.className = `alert-item severity-${alert.severity}`;

  const severity = document.createElement('div');
  severity.className = `alert-severity ${alert.severity}`;
  severity.textContent = alert.severity;

  const content = document.createElement('div');
  content.className = 'alert-content';

  const type = document.createElement('div');
  type.className = 'alert-type';
  type.textContent = formatAlertType(alert.alert_type);

  const message = document.createElement('div');
  message.className = 'alert-message';
  message.textContent = formatAlertMessage(alert);

  const timestamp = document.createElement('div');
  timestamp.className = 'alert-timestamp';
  timestamp.textContent = formatRelativeTime(alert.created_at);

  content.appendChild(type);
  content.appendChild(message);
  content.appendChild(timestamp);

  // Add action buttons for major alerts
  if (alert.severity === 'S2' || alert.severity === 'S3') {
    const actions = document.createElement('div');
    actions.className = 'alert-actions';

    const ackButton = document.createElement('button');
    ackButton.className = 'alert-action-btn';
    ackButton.textContent = 'Ack';
    ackButton.title = 'Acknowledge alert';
    ackButton.onclick = () => acknowledgeAlert(alert.id);

    const snoozeButton = document.createElement('button');
    snoozeButton.className = 'alert-action-btn';
    snoozeButton.textContent = 'Snooze';
    snoozeButton.title = 'Snooze for 30 minutes';
    snoozeButton.onclick = () => snoozeAlert(alert.id, 30);

    actions.appendChild(ackButton);
    actions.appendChild(snoozeButton);
    content.appendChild(actions);
  }

  item.appendChild(severity);
  item.appendChild(content);

  return item;
}

// formatAlertType, formatRelativeTime: imported from risk-dashboard-alerts-history.js

async function acknowledgeAlert(alertId) {
  try {
    const response = await window.globalConfig.apiRequest(`/api/alerts/test/acknowledge/${alertId}`, {
      method: 'POST',
      body: JSON.stringify({ notes: 'Acknowledged from dashboard' })
    });

    if (response && (response.ok || response.success || !response.error)) {
      // Refresh alerts display
      const alertsData = await fetchAlertsData();
      updateAlertsDisplay(alertsData);
      debugLogger.debug(`Alert ${alertId} acknowledged`);
    } else {
      debugLogger.error('Failed to acknowledge alert:', response.status);
    }
  } catch (error) {
    debugLogger.error('Error acknowledging alert:', error);
  }
}

async function snoozeAlert(alertId, minutes) {
  try {
    const response = await window.globalConfig.apiRequest(`/api/alerts/test/snooze/${alertId}`, {
      method: 'POST',
      body: JSON.stringify({ minutes })
    });

    if (response && (response.ok || response.success || !response.error)) {
      // Refresh alerts display
      const alertsData = await fetchAlertsData();
      updateAlertsDisplay(alertsData);
      debugLogger.debug(`Alert ${alertId} snoozed for ${minutes} minutes`);
    } else {
      debugLogger.error('Failed to snooze alert:', response.status);
    }
  } catch (error) {
    debugLogger.error('Error snoozing alert:', error);
  }
}

// ====== Alerts History Functions ======
// Extracted to risk-dashboard-alerts-history.js (Feb 2026)
// Functions imported: loadAlertsHistory, refreshAlertsHistory, initializeAlertsTab,
//                     formatAlertType, formatRelativeTime, getAlertTypeDisplayName,
//                     loadPreviousAlertsPage, loadNextAlertsPage

// ====== Sidebar Updates ======
// Note: Sidebar now handled by Web Component <risk-sidebar-full> in flyout-panel


// Risk rating rules for different metrics
const RISK_RULES = {
  sharpe: { good: [0.5, 999], warn: [0.2, 0.5] },
  sortino: { good: [0.5, 999], warn: [0.2, 0.5] },
  volatility: { good: [0, 0.3], warn: [0.3, 0.6] },
  max_drawdown: { good: [0, 0.2], warn: [0.2, 0.4] },
  var95_1d: { good: [0, 0.05], warn: [0.05, 0.10] },
  var99_1d: { good: [0, 0.08], warn: [0.08, 0.15] },
  cvar95_1d: { good: [0, 0.07], warn: [0.07, 0.12] },
  cvar99_1d: { good: [0, 0.10], warn: [0.10, 0.18] },
  diversification_ratio: { good: [0.7, 2.0], warn: [0.4, 0.7] },
  effective_assets: { good: [10, 999], warn: [5, 10] }
};

function rate(key, value) {
  const r = RISK_RULES[key];
  if (!r || value == null || isNaN(value)) return { dot: 'orange', verdict: 'Unavailable', body: 'Data unavailable.' };
  const signed = value;
  let v = signed;
  // Pour ces métriques, on évalue la magnitude (valeur absolue) :
  if (['volatility', 'max_drawdown', 'var95_1d', 'var99_1d', 'cvar95_1d', 'cvar99_1d'].includes(key)) v = Math.abs(signed);
  const inR = ([a, b]) => v >= a && v < b;
  let dot = 'red', verdict = 'High / risky';
  if (inR(r.good)) { dot = 'green'; verdict = 'Rather low / controlled'; }
  else if (inR(r.warn)) { dot = 'orange'; verdict = 'Intermediate / to monitor'; }
  return { dot, verdict, body: '', label: key };
}

// Note: Initialization moved to the bottom of the script

// ====== Main Functions ======
// ====== Score Calculation Functions ======

// Calculate On-Chain Score (V2 System with optional dynamic weighting)
async function calculateOnChainScore() {
  try {
    const indicators = await fetchAllIndicators();
    // Aucun indicateur exploitable → retourner null (et laisser l'UI gérer)
    if (!indicators || (indicators._metadata && indicators._metadata.available_count === 0)) {
      return null;
    }

    // Dynamic weighting always enabled (V2 production mode)
    const composite = calculateCompositeScoreV2(indicators, true);

    console.debug(`📊 Composite result: score=${composite.score}, version=${composite.version}, hasDynamic=${!!composite.dynamicWeighting}`);

    if (composite.dynamicWeighting) {
      console.debug(`🤖 Dynamic weighting applied: ${composite.dynamicWeighting.phase.name} phase`);
    }

    if (composite.score === null) {
      debugLogger.warn('⚠️ No real on-chain data available - returning null score');
      return null;
    }

    console.debug(`📊 On-Chain Score: ${composite.score} (${composite.message})`);

    // Stocker les métadonnées enrichies V2 pour utilisation par SMART
    store.set('scores.onchain_metadata', {
      categoryBreakdown: composite.categoryBreakdown,
      criticalZoneCount: composite.criticalZoneCount,
      totalIndicators: composite.totalIndicators,
      activeCategories: composite.activeCategories,
      contributors: composite.contributors.slice(0, 10), // Top 10 contributeurs
      confidence: composite.confidence,
      // V2 specific properties
      consensusSignals: composite.consensusSignals,
      correlationAnalysis: composite.correlationAnalysis,
      version: composite.version,
      improvements: composite.improvements,
      // Dynamic weighting properties (if enabled)
      dynamicWeighting: composite.dynamicWeighting
    });

    // Analyze contradictory signals V2
    const contradictions = analyzeContradictorySignals(composite.categoryBreakdown);
    if (contradictions.length > 0) {
      debugLogger.warn(`⚠️ ${contradictions.length} signaux contradictoires détectés:`, contradictions);
      store.set('scores.contradictory_signals', contradictions);
    }

    // Log des alertes critiques
    if (composite.criticalZoneCount > 0) {
      debugLogger.warn(`🚨 ${composite.criticalZoneCount} indicateur(s) en zone critique!`);

      const criticalIndicators = composite.contributors.filter(c => c.inCriticalZone);
      criticalIndicators.forEach(indicator => {
        debugLogger.warn(`  ⚠️ ${indicator.name}: ${indicator.originalValue}% (seuil: ${indicator.raw_threshold})`);
      });
    }

    // Log de la répartition par catégorie
    Object.entries(composite.categoryBreakdown).forEach(([category, data]) => {
      const emoji = category === 'onchain_fundamentals' ? '🔗' :
        category === 'cycle_technical' ? '📊' : '😨';
      console.debug(`  ${emoji} ${data.description}: ${data.score}/100 (${data.contributorsCount} indicateurs)`);
    });

    if (indicators._metadata?.missing_apis?.length > 0) {
      debugLogger.warn('⚠️ Missing APIs:', indicators._metadata.missing_apis);
    }

    return composite.score;
  } catch (error) {
    debugLogger.error('❌ Erreur calcul On-Chain Score:', error);
    return null; // Retourner null au lieu de fallback trompeur
  }
}

// ❌ SUPPRIMÉ: calculateRiskScore() locale (divergeait de l'orchestrator)
// ✅ Maintenant calculé par risk-data-orchestrator.js (source unique SSOT)

// Calculate strategic Blended Score (nouvelle formule market-aware)
function calculateBlendedScore(ccsMixteScore, onchainScore, riskScore) {
  // Score de Régime: 0.5×CCS + 0.3×OnChain + 0.2×Risk (positif, plus haut = plus robuste)
  if (ccsMixteScore == null && onchainScore == null && riskScore == null) {
    return 50; // Fallback neutre
  }

  console.debug('🎯 Calcul Blended Score stratégique:', {
    ccsMixte: ccsMixteScore,
    onchain: onchainScore,
    risk: riskScore
  });

  let totalScore = 0;
  let totalWeight = 0;

  // CCS Mixte : 50% (score directeur du marché)
  if (ccsMixteScore != null) {
    totalScore += ccsMixteScore * 0.50;
    totalWeight += 0.50;
    console.debug('  → CCS Mixte contributtion:', ccsMixteScore * 0.50);
  }

  // On-Chain Composite : 30% (fondamentaux réseau)
  if (onchainScore != null) {
    totalScore += onchainScore * 0.30;
    totalWeight += 0.30;
    console.debug('  → On-Chain contribution:', onchainScore * 0.30);
  } else {
    debugLogger.warn('  ⚠️ On-Chain Score non disponible (APIs payantes requises) - poids redistribué');
  }

  // Risk contribution : 20% (score direct, plus haut = plus robuste)
  // ✅ Respecte docs/RISK_SEMANTICS.md - NE PAS inverser
  if (riskScore != null) {
    totalScore += riskScore * 0.20;
    totalWeight += 0.20;
    console.debug('  → Risk contribution (direct):', riskScore * 0.20);
  }

  const finalScore = totalWeight > 0 ? totalScore / totalWeight : 50;
  console.debug('🎯 Final Blended Score:', finalScore, '(weight:', totalWeight, ')');

  return Math.max(0, Math.min(100, finalScore));
}

// Apply UI state based on blended decision score
function applyDecisionState(score) {
  try {
    const el = document.getElementById('blended-gauge');
    if (!el || typeof score !== 'number') return;
    el.classList.remove('decision--bullish', 'decision--neutral', 'decision--defensive');
    let state = 'neutral';
    if (score >= 70) state = 'bullish';
    else if (score < 45) state = 'defensive';
    el.classList.add(`decision--${state}`);
  } catch (e) { /* no-op */ }
}

// ✅ REMPLACÉ: Load scores from orchestrator (SSOT)
async function loadScoresFromStore() {
  try {
    console.debug('🔄 Waiting for orchestrator hydration...');

    // ✅ Attendre hydratation complète du store par l'orchestrator
    if (!store.getState()?._hydrated) {
      await new Promise(resolve => {
        const handler = (e) => {
          if (e.detail?.hydrated) {
            debugLogger.debug('✅ Store hydrated by orchestrator, source:', e.detail.source);
            resolve();
          }
        };
        window.addEventListener('riskStoreReady', handler, { once: true });
      });
    }

    // ✅ Lire scores depuis store (source unique)
    const state = store.snapshot();
    const onchainScore = state.scores?.onchain;
    const riskScore = state.scores?.risk;
    const blendedScore = state.scores?.blended;
    const ccsScore = state.ccs?.score;
    const ccsMixteScore = state.cycle?.ccsStar;

    console.debug('📊 Scores loaded from orchestrator:', {
      onchain: onchainScore,
      risk: riskScore,
      blended: blendedScore,
      ccs: ccsScore,
      ccsMixte: ccsMixteScore,
      source: state._hydration_source,
      duration: state._hydration_duration_ms + 'ms'
    });

    // ✅ Mettre à jour UI
    updateScoreDisplays(onchainScore, riskScore, blendedScore, ccsScore);
    const cycleScore = state.scores?.cycle ?? null;
    updateMarketRegime(blendedScore, onchainScore, riskScore, cycleScore);

    // ✅ Compatibility: Store in localStorage for legacy cross-page access
    const dataSource = globalConfig.get('data_source') || 'unknown';
    const __user = localStorage.getItem('activeUser');
    const __prefix = (k) => `${k}:${__user}`;
    try {
      // ✅ FIX: Ne pas stocker de strings vides - seulement stocker si la valeur existe
      // ⚠️ IMPORTANT: Toujours stocker le timestamp, même si certains scores sont null
      if (onchainScore !== null && onchainScore !== undefined) {
        localStorage.setItem(__prefix('risk_score_onchain'), onchainScore.toString());
      } else {
        localStorage.removeItem(__prefix('risk_score_onchain'));
      }
      if (riskScore !== null && riskScore !== undefined) {
        localStorage.setItem(__prefix('risk_score_risk'), riskScore.toString());
      } else {
        localStorage.removeItem(__prefix('risk_score_risk'));
      }
      if (blendedScore !== null && blendedScore !== undefined) {
        localStorage.setItem(__prefix('risk_score_blended'), blendedScore.toString());
      } else {
        localStorage.removeItem(__prefix('risk_score_blended'));
      }
      if (ccsScore !== null && ccsScore !== undefined) {
        localStorage.setItem(__prefix('risk_score_ccs'), ccsScore.toString());
      } else {
        localStorage.removeItem(__prefix('risk_score_ccs'));
      }
      // Toujours stocker le timestamp pour marquer la fraîcheur des données
      localStorage.setItem(__prefix('risk_score_timestamp'), Date.now().toString());
      localStorage.setItem(__prefix('risk_score_data_source'), dataSource);
    } catch (_) { }

    return { onchainScore, riskScore, blendedScore, ccsScore, ccsMixteScore };

  } catch (error) {
    debugLogger.error('❌ Erreur chargement scores depuis store:', error);
    // ✅ Fallback: Use partial data if available
    const state = store.snapshot();
    return {
      onchainScore: state.scores?.onchain ?? null,
      riskScore: state.scores?.risk ?? null,
      blendedScore: state.scores?.blended ?? null,
      ccsScore: state.ccs?.score ?? null,
      ccsMixteScore: state.cycle?.ccsStar ?? null
    };
  }
}

// ❌ OBSOLÈTE: calculateAllScores() supprimée (remplacée par loadScoresFromStore)
// L'orchestrator est maintenant la source unique de vérité (SSOT)

// ====== Market Regime Functions ======
function updateMarketRegime(blendedScore, onchainScore, riskScore, cycleScore = null) {
  try {
    // Calculate regime data (pass cycleScore to avoid resetting Schmitt trigger flags)
    const regimeData = getRegimeDisplayData(blendedScore, onchainScore, riskScore, cycleScore);
    const regime = regimeData.regime;

    console.debug('📊 Market Regime calculated:', regime);

    // Update Market Regime display
    const regimeDot = document.getElementById('regime-dot');
    const regimeText = document.getElementById('regime-text');

    if (regimeDot && regimeText) {
      regimeDot.style.backgroundColor = regime.color;
      regimeDot.className = 'status-dot active';

      const confidenceText = `${Math.round(regime.confidence * 100)}%`;
      regimeText.innerHTML = `
            <div style="font-weight: bold; color: ${regime.color};">
              ${regime.emoji} ${regime.name}
            </div>
            <div style="font-size: 0.8rem; margin-top: 2px;">
              Score: ${regime.score.toFixed(2)} | Conf: ${confidenceText}
            </div>
          `;

      // Store regime data for other components
      store.set('market.regime', regimeData);
    }

  } catch (error) {
    debugLogger.error('❌ Error updating market regime:', error);

    const regimeText = document.getElementById('regime-text');
    if (regimeText) {
      regimeText.textContent = 'Error calculating regime';
    }
  }
}

// ====== Dynamic Weighting Functions ======
// Fallback banner toggle (backend status)
// Nov 2025: Only show banner for critical failures (error), not degraded state
function updateBackendBanner() {
  try {
    const status = store.get('ui.apiStatus.backend');
    const banner = document.getElementById('backend-fallback-banner');
    if (!banner) return;
    // Only show banner for 'error' status (not 'degraded' or 'unknown')
    // 'degraded' means optional APIs failed but core functionality works
    banner.style.display = (status === 'error') ? 'block' : 'none';
  } catch { }
}

// Utiliser managed interval si disponible, sinon fallback setInterval
if (window.networkStateManager) {
  window.networkStateManager.createManagedInterval(updateBackendBanner, 2000, 'Backend Banner Toggle');
} else {
  setInterval(updateBackendBanner, 2000);
}

// Dynamic weighting is now always enabled (V2 production mode)
// Legacy functions removed: toggleDynamicWeighting, initializeDynamicWeightingToggle

// ====== Section Collapse Functions ======


window.toggleSection = function (sectionId) {
  const content = document.getElementById(sectionId + '-content');
  const arrow = document.getElementById(sectionId + '-arrow');

  if (!content || !arrow) return;

  const isCollapsed = content.style.display === 'none';

  if (isCollapsed) {
    content.style.display = '';
    arrow.textContent = '▼';
    localStorage.setItem(`section_${sectionId}_collapsed`, 'false');

    // Observe lazy elements; only force-load if they are near viewport
    setTimeout(() => {
      const lazyElements = content.querySelectorAll('[data-lazy-load]');
      debugLogger.debug(`🔍 Section ${sectionId} expanded, checking ${lazyElements.length} lazy elements`);

      const isNearViewport = (el, margin = 150) => {
        try {
          const r = el.getBoundingClientRect();
          const vh = window.innerHeight || document.documentElement.clientHeight;
          return r.top < vh + margin && r.bottom > -margin;
        } catch (_) { return false; }
      };

      lazyElements.forEach(element => {
        if (!element.classList.contains('lazy-loaded') && !element.classList.contains('lazy-error')) {
          if (isNearViewport(element)) {
            debugLogger.debug(`📊 Element near viewport; loading now in section ${sectionId}`);
            window.lazyLoader?.loadVisibleElement(element);
          } else {
            debugLogger.debug(`👁️ Element not near viewport; ensuring observer is attached`);
            if (window.lazyLoader?.intersectionObserver) {
              try { window.lazyLoader.intersectionObserver.observe(element); } catch (_) { }
            }
          }
        }
      });
    }, 100);
  } else {
    content.style.display = 'none';
    arrow.textContent = '▶';
    localStorage.setItem(`section_${sectionId}_collapsed`, 'true');
  }
}

// Initialize section states on page load
function initializeSectionStates() {
  const sections = ['onchain-indicators', 'bitcoin-cycle', 'dynamic-weighting-info'];

  sections.forEach(sectionId => {
    const isCollapsed = localStorage.getItem(`section_${sectionId}_collapsed`) === 'true';
    const content = document.getElementById(sectionId + '-content');
    const arrow = document.getElementById(sectionId + '-arrow');

    if (content && arrow) {
      if (isCollapsed) {
        content.style.display = 'none';
        arrow.textContent = '▶';
      } else {
        content.style.display = '';
        arrow.textContent = '▼';

        // Ensure lazy elements are observed; only force-load if near viewport
        setTimeout(() => {
          const lazyElements = content.querySelectorAll('[data-lazy-load]');
          if (lazyElements.length > 0) {
            debugLogger.debug(`🔍 Section ${sectionId} is open by default, checking ${lazyElements.length} lazy elements`);

            lazyElements.forEach(element => {
              if (!element.classList.contains('lazy-loaded') && !element.classList.contains('lazy-error')) {
                const isNearViewport = (el, margin = 150) => {
                  try {
                    const r = el.getBoundingClientRect();
                    const vh = window.innerHeight || document.documentElement.clientHeight;
                    return r.top < vh + margin && r.bottom > -margin;
                  } catch (_) { return false; }
                };

                if (isNearViewport(element)) {
                  debugLogger.debug(`📊 Element near viewport; loading now in open section ${sectionId}`);
                  setTimeout(() => { window.lazyLoader?.loadVisibleElement(element); }, 300);
                } else {
                  debugLogger.debug(`👁️ Element not near viewport; ensuring observer is attached`);
                  if (window.lazyLoader?.intersectionObserver) {
                    try { window.lazyLoader.intersectionObserver.observe(element); } catch (_) { }
                  }
                }
              }
            });
          }
        }, 100);
      }
    }
  });
}

async function refreshDashboard(forceRefresh = false) {
  if (isRefreshing) return;
  isRefreshing = true;
  // Ensure currency rate loaded before render when not USD
  try {
    const cur = (window.globalConfig && window.globalConfig.get('display_currency')) || 'USD';
    if (window.currencyManager && cur !== 'USD') {
      await window.currencyManager.ensureRate(cur);
    }
  } catch (_) { }

  const refreshBtn = document.getElementById('refresh-btn');
  refreshBtn.disabled = true;
  // Optional: disable dropdown trigger if present
  const refreshMenuBtn = document.getElementById('refresh-menu-btn');
  if (refreshMenuBtn) refreshMenuBtn.disabled = true;
  const forceRefreshBtn = document.getElementById('force-refresh-btn');
  if (forceRefreshBtn) forceRefreshBtn.disabled = true;
  refreshBtn.textContent = forceRefresh ? '🔄 Force Refreshing…' : '🔄 Refreshing…';

  // Clear cache if force refresh is requested
  if (forceRefresh) {
    clearAllPersistentCache();
    debugLogger.debug('🧹 Force refresh: all cache cleared');
  }

  // Update loading state
  store.set('ui.loading', true);
  store.set('ui.apiStatus.backend', 'unknown');

  try {
    // Fetch risk data, CCS data, and alerts in parallel
    const [riskData, ccsData, alertsData] = await Promise.all([
      fetchRiskData(),
      loadCCSData().catch(err => {
        debugLogger.warn('CCS data failed, continuing with risk only:', err);
        return null;
      }),
      fetchAlertsData().catch(err => {
        debugLogger.warn('Alerts data failed, continuing without alerts:', err);
        return null;
      })
    ]);

    if (riskData && riskData.success) {
      // Update store with risk data
      store.set('riskMetrics', riskData.risk_metrics);
      store.set('portfolioSummary', riskData.portfolio_summary);
      store.set('correlationMetrics', riskData.correlation_metrics);
      store.set('ui.apiStatus.backend', 'healthy');

      // Render risk dashboard in the active tab
      renderRiskDashboard(riskData);

      // Update timestamp
      if (riskData.timestamp && riskData.calculation_time) {
        updateTimestamp(riskData.timestamp, riskData.calculation_time);
      }

      // Advanced components are loaded inline in this controller
      // (GRI, Phase 3A, Structure Modulation, etc.)

      // If CCS data loaded successfully, compute blended CCS
      if (ccsData) {
        await loadBlendedCCS();
      }

      // ✅ Load scores from orchestrator (SSOT)
      await loadScoresFromStore();

      // Force sidebar update after all scores are calculated
      const finalState = store.snapshot();
      // Update alerts in store (for risk-sidebar-full component)
      store.set('alerts', alertsData || []);

      // Update alerts display
      updateAlertsDisplay(alertsData);

      // Also update cycles content (contains composite score display) 
      if (finalState.ccs?.score && finalState.cycle?.months) {
        await renderCyclesContent();
        console.debug('🔄 Cycles content updated with latest data');
      }

      debugLogger.debug('Risk dashboard refreshed successfully');

    } else {
      // Handle API error with informative message, but still compute partial scores (on-chain + defaults)
      store.set('ui.apiStatus.backend', 'error');
      const errorType = riskData?.error_type || 'unknown';
      const errorMessage = riskData?.message || 'Failed to load risk dashboard';

      if (errorType === 'connection_error') {
        renderBackendUnavailable(errorMessage);
      } else {
        renderApiError(errorMessage);
      }

      // Attempt partial scoring so risk-sidebar-full component is still populated
      try {
        if (ccsData) {
          await loadBlendedCCS();
        }
        await loadScoresFromStore();
      } catch (e) {
        debugLogger.warn('Partial score calculation failed:', e);
      }
      return; // Don't throw, just render error state
    }

  } catch (err) {
    debugLogger.error('Dashboard error:', err);
    store.set('ui.apiStatus.backend', 'error');
    renderBackendUnavailable('Error connecting to risk backend');
  } finally {
    isRefreshing = false;
    store.set('ui.loading', false);
    refreshBtn.disabled = false;
    if (refreshMenuBtn) refreshMenuBtn.disabled = false;
    if (forceRefreshBtn) forceRefreshBtn.disabled = false;
    refreshBtn.textContent = '🔄 Refresh';
  }
}

// Update badges with current risk data
function updateRiskDashboardBadges(data) {
  if (!window.riskDashboardBadges) return;

  try {
    const now = new Date();

    // Risk Overview badge
    const riskMetrics = data?.risk_metrics || {};
    const portfolioSummary = data?.portfolio_summary || {};
    const contradiction = Math.round((data?.correlation_metrics?.average_correlation || 0.3) * 100);
    const varCap = Math.round((riskMetrics.var_95_1d || 0.05) * 100);

    if (window.riskDashboardBadges.riskOverview) {
      window.riskDashboardBadges.riskOverview.updateData({
        source: 'Risk Engine',
        updated: now,
        contradiction: contradiction,
        cap: varCap,
        overrides: data?.overrides_count || 0,
        status: data ? 'ok' : 'error'
      });
    }

    // Cycles badge
    if (window.riskDashboardBadges.cycles) {
      window.riskDashboardBadges.cycles.updateData({
        source: 'Market Cycles',
        updated: now,
        contradiction: Math.round(Math.random() * 50), // Placeholder
        status: 'ok'
      });
    }

    // Targets badge
    if (window.riskDashboardBadges.targets) {
      window.riskDashboardBadges.targets.updateData({
        source: 'Targets',
        updated: now,
        contradiction: Math.round(Math.random() * 30), // Placeholder
        status: 'ok'
      });
    }

    debugLogger.debug('🏷️ Risk dashboard badges updated');
  } catch (error) {
    debugLogger.warn('Risk badge update failed:', error);
  }
}

function renderRiskDashboard(data) {
  const container = document.getElementById('risk-dashboard-content');
  if (!container) {
    debugLogger.warn('⚠️ risk-dashboard-content not found in DOM, skipping legacy render');
    return;
  }
  if (!data || !data.risk_metrics || !data.correlation_metrics || !data.portfolio_summary) {
    renderError('Incomplete data received from API'); return;
  }

  // Afficher un bandeau pour le mode test
  let testModeBanner = '';
  if (data.test_mode) {
    testModeBanner = `
          <div style="background: var(--info-bg); border: 1px solid var(--info); border-radius: var(--radius-md); padding: 1rem; margin-bottom: 1.5rem; text-align: center;">
            <div style="color: var(--info); font-weight: 600; margin-bottom: 0.5rem;">🧪 TEST MODE - Real Data</div>
            <div style="color: var(--theme-text-muted); font-size: 0.9rem;">
              Demo portfolio using the real price history cache (${data.test_holdings?.length || 0} assets, ${formatMoney(data.portfolio_summary.total_value)})
            </div>
          </div>
        `;
  }
  const m = data.risk_metrics;
  const c = data.correlation_metrics;
  const p = data.portfolio_summary;
  const balances = Array.isArray(data.balances) ? data.balances : [];

  // Quick insights from balances for concentration and stablecoins
  const insights = (() => {
    const total = Number(p?.total_value) || balances.reduce((a, b) => a + Number(b.value_usd || 0), 0);
    if (!total || (!balances || balances.length === 0)) {
      return { top5Share: null, hhi: null, stableShare: null };
    }
    const sorted = balances
      .filter(x => Number(x.value_usd) > 0)
      .sort((a, b) => Number(b.value_usd) - Number(a.value_usd));
    const weights = sorted.map(x => Number(x.value_usd) / total);
    const top5Share = weights.slice(0, 5).reduce((a, b) => a + b, 0);
    const hhi = weights.reduce((a, b) => a + b * b, 0);
    // Stablecoins share
    const STABLES = new Set(['USDC', 'USDT', 'USD', 'DAI', 'USTC']);
    const stableValue = sorted
      .filter(x => STABLES.has(String(x.symbol || '').toUpperCase()))
      .reduce((a, b) => a + Number(b.value_usd || 0), 0);
    const stableShare = stableValue / total;
    return { top5Share, hhi, stableShare };
  })();

  // Helper functions are defined at the top of the script

  // Prépare: HTML recommandations et alertes pour la section top-summary
  // ✅ MODIFIÉ (Phase 1.1): Passe fullData pour accès à risk_budget
  const recos = generateRecommendations(m, c, p.groups || {}, data);
  const recommendationsHtml = (() => {
    return recos.map(rec => `
          <div class="recommendation recommendation-${rec.priority}">
            <div class="recommendation-header">
              <span class="recommendation-icon">${rec.icon}</span>
              <span class="recommendation-title">${rec.title}</span>
              <span class="recommendation-priority">${rec.priority === 'high' ? 'PRIORITÉ' : rec.priority === 'medium' ? 'Important' : 'Info'}</span>
            </div>
            <div class="recommendation-description">${rec.description}</div>
            <div class="recommendation-action">▶️ ${rec.action}</div>
          </div>
        `).join('');
  })();

  const alertCount = (data.alerts && data.alerts.length) ? data.alerts.length : 0;
  const severityCounts = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
  (data.alerts || []).forEach(a => {
    const lvl = String(a.level || '').toLowerCase();
    if (severityCounts.hasOwnProperty(lvl)) severityCounts[lvl]++;
  });
  const hasSevere = (severityCounts.critical + severityCounts.high) > 0;
  const breakdown = (() => {
    const parts = [];
    if (severityCounts.critical) parts.push(`${severityCounts.critical} critical`);
    if (severityCounts.high) parts.push(`${severityCounts.high} high`);
    if (severityCounts.medium) parts.push(`${severityCounts.medium} medium`);
    if (parts.length === 0) return '';
    return ` (${parts.join(', ')})`;
  })();
  const alertsHtml = (alertCount) ? (
    data.alerts.map(a => `
          <div class="alert alert-${a.level}">
            <strong>${a.message}</strong><br>
            <em>Recommendation: ${a.recommendation}</em>
          </div>
        `).join('')
  ) : `
        <div class="alert alert-low">
          <strong>✅ All Clear</strong><br>
          <em>No significant risk alerts at this time.</em>
        </div>
      `;

  container.innerHTML = `
        ${testModeBanner}
        <!-- Top Summary: Collapsible container -->
        <details class="top-collapsible" ${hasSevere ? 'open' : ''}>
          <summary>
            <div>Risk Overview & Recommendations</div>
            <div class="summary-right">
              <span class="badge badge-alerts">⚠️ ${alertCount} alerts${breakdown}</span>
              <span class="badge badge-recos">💡 ${recos.length} recs</span>
              <span class="chevron">›</span>
            </div>
          </summary>
          <div class="top-summary">
          <!-- Points clés -->
          <div class="risk-card">
            <h3>📋 Key points of your portfolio</h3>
            <div class="insights-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .75rem;">
              <div class="insight-item">
                <div style="font-weight: 600; color: var(--theme-text);">🎯 Risk Level</div>
                <div style="color: var(--theme-text-muted); margin-top: 0.25rem;">
                  ${(() => {
      const riskScore = m.risk_score || 0;
      // IMPORTANT: Risk Score positif - plus haut = meilleur (plus robuste)
      if (riskScore > 70) return 'Excellent - Very robust portfolio';
      if (riskScore > 50) return 'Good - Robustness/return balance';
      return 'Low - Watch out for high volatility';
    })()}
                </div>
              </div>
              <div class="insight-item">
                <div style="font-weight: 600; color: var(--theme-text);">📊 Diversification</div>
                <div style="color: var(--theme-text-muted); margin-top: 0.25rem;">
                  ${(() => {
      const div = c.diversification_ratio || 0;
      if (div > 0.7) return 'Excellent - Well-distributed portfolio';
      if (div > 0.4) return 'Limited - Room for improvement';
      return 'Low - Too correlated, diversify';
    })()}
                </div>
              </div>
              <div class="insight-item">
                <div style="font-weight: 600; color: var(--theme-text);">⚡ Performance/Risk</div>
                <div style="color: var(--theme-text-muted); margin-top: 0.25rem;">
                  ${(() => {
      const sharpe = m.sharpe_ratio || 0;
      if (sharpe > 1.2) return 'Excellent - Superior return for the risk taken';
      if (sharpe > 0.8) return 'Good - Acceptable return for the risk';
      return 'Needs improvement - High risk vs return';
    })()}
                </div>
              </div>
              <div class="insight-item">
                <div style="font-weight: 600; color: var(--theme-text);">🔝 Concentration</div>
                <div style="color: var(--theme-text-muted); margin-top: 0.25rem;">
                  ${(() => {
      const t5 = insights.top5Share;
      const hhi = insights.hhi;
      if (t5 == null || hhi == null) return 'N/A';
      return `Top 5: ${(t5 * 100).toFixed(1)}% • HHI: ${hhi.toFixed(2)}`;
    })()}
                </div>
              </div>
              <div class="insight-item">
                <div style="font-weight: 600; color: var(--theme-text);">💵 Stablecoins</div>
                <div style="color: var(--theme-text-muted); margin-top: 0.25rem;">
                  ${(() => {
      const s = insights.stableShare;
      return (s == null) ? 'N/A' : `${(s * 100).toFixed(1)}% of portfolio`;
    })()}
                </div>
              </div>
              <div class="insight-item">
                <div style="font-weight: 600; color: var(--theme-text);">🧪 Calculation Data</div>
                <div style="color: var(--theme-text-muted); margin-top: 0.25rem;">
                  ${p.num_assets || (balances?.length || 'N/A')} assets used
                </div>
              </div>
            </div>
          </div>

          <!-- Risk Alerts -->
          <div class="risk-card">
            <h3>⚠️ Risk Alerts</h3>
            ${alertsHtml}
          </div>

          <!-- Recommandations d'amélioration -->
          <div class="risk-card">
            <h3>💡 Improvement Recommendations</h3>
            ${recommendationsHtml}
          </div>
          </div>
        </details>

        <!-- Portfolio Summary - Compact KPI Cards -->
        <div class="portfolio-summary-compact">
          <div class="kpi-card kpi-primary">
            <div class="kpi-content">
              <div class="kpi-label">Total Value</div>
              <div class="kpi-value">${formatMoney(p.total_value)}</div>
            </div>
          </div>

          <div class="kpi-card">
            <div class="kpi-content">
              <div class="kpi-label">Assets</div>
              <div class="kpi-value">${p.num_assets}</div>
              <div class="kpi-subtext">${(() => {
      const effective = c.effective_assets;
      return effective ? `${safeFixed(effective, 1)} effective` : '';
    })()}</div>
            </div>
          </div>

          <div class="kpi-card">
            <div class="kpi-content">
              <div class="kpi-label">Diversification</div>
              <div class="kpi-value">${safeFixed(c.diversification_ratio, 2)}</div>
              <div class="kpi-subtext">${(() => {
      const div = c.diversification_ratio || 0;
      if (div > 0.7) return 'Excellent';
      if (div > 0.4) return 'Moderate';
      return 'Low';
    })()}</div>
            </div>
          </div>

          <div class="kpi-card">
            <div class="kpi-content">
              <div class="kpi-label">Top 5 Share</div>
              <div class="kpi-value">${(() => {
      const t5 = insights.top5Share;
      return t5 != null ? `${(t5 * 100).toFixed(1)}%` : 'N/A';
    })()}</div>
              <div class="kpi-subtext">Concentration</div>
            </div>
          </div>

          <div class="kpi-card">
            <div class="kpi-content">
              <div class="kpi-label">Stablecoins</div>
              <div class="kpi-value">${(() => {
      const s = insights.stableShare;
      return s != null ? `${(s * 100).toFixed(1)}%` : 'N/A';
    })()}</div>
              <div class="kpi-subtext">Protection</div>
            </div>
          </div>

          <div class="kpi-card">
            <div class="kpi-content">
              <div class="kpi-label">Data Quality</div>
              <div class="kpi-value">${safeFixed((p.confidence_level || 0) * 100, 1)}%</div>
              <div class="kpi-subtext">Confidence</div>
            </div>
          </div>
        </div>

        <!-- Key Insights moved to Top Summary -->

        <div class="risk-grid">
          <!-- 🆕 Risk Score Card (Oct 2025) -->
          <div class="risk-card">
            <h3>Risk Score <span style="font-size:.8rem; color: var(--theme-text); opacity:.7; font-weight:500; margin-left:.5rem;"><br>Robustness Indicator [0-100]</span></h3>

            <!-- Risk Score Principal -->
            <div class="metric-row">
              <span class="metric-label">Risk Score</span>
              <span class="metric-value hinted" data-key="risk_score" data-value="${m.risk_score}" data-score="risk-display" style="color: ${pickScoreColor(m.risk_score)}">
                ${safeFixed(m.risk_score, 1)}/100
              </span>
              <button class="btn-breakdown-toggle" onclick="toggleBreakdown('risk-score-breakdown')" title="View penalty details" aria-label="Show Risk Score calculation details" style="margin-left: 8px; padding: 2px 8px; font-size: 0.75em; background: rgba(125, 207, 255, 0.15); border: 1px solid var(--brand-primary); border-radius: 4px; color: var(--brand-primary); cursor: pointer;">
                🔍 Details
              </button>
            </div>

            <!-- Breakdown Panel -->
            <div id="risk-score-breakdown" class="breakdown-panel" style="display: none; margin: 8px 0; padding: 12px; background: rgba(30, 30, 46, 0.6); border-radius: 8px; border: 1px solid rgba(125, 207, 255, 0.2); font-size: 0.85em;">
              <div class="breakdown-header" style="font-weight: 600; color: var(--brand-primary); margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                <span>Calculation Detail (Base = 50) ${m.risk_version_info ? `— ${m.risk_version_info.active_version === 'v2' ? 'V2' : 'Legacy'}` : ''}</span>
                <button onclick="toggleBreakdown('risk-score-breakdown')" style="background: none; border: none; color: var(--text-secondary); cursor: pointer; font-size: 1.2em;" aria-label="Close">×</button>
              </div>
              <div class="breakdown-table" style="display: flex; flex-direction: column; gap: 4px;">
                <div class="breakdown-row breakdown-base" style="display: grid; grid-template-columns: 1fr auto auto; gap: 8px; padding: 4px; background: rgba(125, 207, 255, 0.05); border-radius: 4px;">
                  <span class="breakdown-label" style="color: var(--text-secondary);">Neutral base</span>
                  <span class="breakdown-value" style="color: var(--text-primary); font-weight: 600;">+50.0</span>
                  <span class="breakdown-cumul" style="color: var(--brand-primary); font-weight: 600; min-width: 50px; text-align: right;">50.0</span>
                </div>
                ${(() => {
      // 🔧 Oct 2025: Use V2 breakdown when v2_active, otherwise legacy
      const breakdown = m.structural_breakdown || {};
      let cumul = 50.0;
      const rows = [];
      const order = ['var_95', 'sharpe', 'drawdown', 'volatility', 'memecoins', 'concentration', 'group_risk', 'diversification'];
      const labels = {
        var_95: 'VaR 95%',
        sharpe: 'Sharpe Ratio',
        drawdown: 'Max Drawdown',
        volatility: 'Volatility',
        memecoins: 'Memecoins %',
        concentration: 'Concentration (HHI)',
        group_risk: 'Group Risk Index',
        diversification: 'Diversification'
      };
      for (const key of order) {
        if (breakdown[key] !== undefined) {
          const delta = breakdown[key];
          cumul += delta;
          const color = delta > 0 ? '#9ece6a' : delta < 0 ? '#f7768e' : 'var(--text-secondary)';
          rows.push(`
                        <div class="breakdown-row" style="display: grid; grid-template-columns: 1fr auto auto; gap: 8px; padding: 4px; border-radius: 4px;">
                          <span class="breakdown-label" style="color: var(--text-secondary);">${labels[key] || key}</span>
                          <span class="breakdown-value" style="color: ${color}; font-weight: 600;">${delta > 0 ? '+' : ''}${delta.toFixed(1)}</span>
                          <span class="breakdown-cumul" style="color: var(--text-primary); min-width: 50px; text-align: right;">${cumul.toFixed(1)}</span>
                        </div>
                      `);
        }
      }
      return rows.join('');
    })()}
                <div class="breakdown-row breakdown-total" style="display: grid; grid-template-columns: 1fr auto auto; gap: 8px; padding: 6px 4px; margin-top: 4px; border-top: 1px solid rgba(125, 207, 255, 0.3); background: rgba(125, 207, 255, 0.08); border-radius: 4px;">
                  <span class="breakdown-label" style="color: var(--text-primary); font-weight: 700;">Total (clamped [0,100])</span>
                  <span class="breakdown-value" style="color: var(--text-tertiary);">—</span>
                  <span class="breakdown-cumul" style="color: var(--brand-primary); font-weight: 700; font-size: 1.1em; min-width: 50px; text-align: right;">${safeFixed(m.risk_score, 1)}</span>
                </div>
              </div>
            </div>

            <!-- Metric Interpretation -->
            <div class="metric-interpretation">
              💡 ${getScoreInterpretation(m.risk_score)}
            </div>

            <!-- Dual Window Badges -->
            ${m.dual_window?.enabled ? `
            <div style="margin: 8px 0; padding: 8px; background: rgba(122, 162, 247, 0.1); border-radius: 6px; border-left: 3px solid var(--brand-primary);">
              ${m.dual_window.long_term?.available ? `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                  <span style="font-size: 0.85em; color: var(--text-secondary); cursor: help;" title="Long-Term window: Computes the Risk Score over ${m.dual_window.long_term.window_days} days of history excluding recent assets. Covers ${(m.dual_window.long_term.coverage_pct * 100).toFixed(0)}% of portfolio value with ${m.dual_window.long_term.asset_count} assets with sufficient history. More stable and reliable metrics than full intersection.">
                    📈 Long-Term (${m.dual_window.long_term.window_days}d, ${m.dual_window.long_term.asset_count} assets, ${(m.dual_window.long_term.coverage_pct * 100).toFixed(0)}%) <span style="color: var(--brand-primary); opacity: 0.6;">ℹ️</span>
                  </span>
                  <span style="font-size: 0.85em; font-weight: 600; color: var(--brand-primary);">
                    Sharpe: ${safeFixed(m.dual_window.long_term.metrics?.sharpe_ratio, 2)}
                  </span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                  <span style="font-size: 0.85em; color: var(--text-secondary); cursor: help;" title="Full Intersection window: Minimum common period including ALL assets (${m.dual_window.full_intersection.asset_count} assets). Over ${m.dual_window.full_intersection.window_days} days only because recent assets limit the history. Metrics may be unstable if window is short. Used for comparison and divergence detection.">
                    🔍 Full Intersection (${m.dual_window.full_intersection.window_days}d, ${m.dual_window.full_intersection.asset_count} assets) <span style="color: var(--text-secondary); opacity: 0.6;">ℹ️</span>
                  </span>
                  <span style="font-size: 0.85em; color: ${Math.abs(m.dual_window.full_intersection.metrics?.sharpe_ratio - m.dual_window.long_term.metrics?.sharpe_ratio) > 0.5 ? 'var(--theme-error)' : 'var(--text-secondary)'};">
                    Sharpe: ${safeFixed(m.dual_window.full_intersection.metrics?.sharpe_ratio, 2)}
                  </span>
                </div>
                ${m.dual_window.exclusions?.excluded_pct > 0.2 ? `
                <div style="margin-top: 6px; padding: 4px 8px; background: rgba(247, 118, 142, 0.15); border-radius: 4px; cursor: help;" title="Assets excluded from Long-Term window due to history < ${m.dual_window.long_term.window_days}j : ${m.dual_window.exclusions.excluded_assets.map(a => a.symbol).join(', ')}. Represent ${(m.dual_window.exclusions.excluded_pct * 100).toFixed(1)}% of total value. The Risk Score is calculated only on the ${m.dual_window.long_term.asset_count} assets with sufficient history for more stability.">
                  <span style="font-size: 0.8em; color: var(--theme-error);">
                    ⚠️ ${m.dual_window.exclusions.excluded_assets.length} assets excluded (${(m.dual_window.exclusions.excluded_pct * 100).toFixed(0)}% value) - short history <span style="opacity: 0.6;">ℹ️</span>
                  </span>
                </div>
                ` : ''}
                <div style="margin-top: 6px; font-size: 0.75em; color: var(--text-tertiary); font-style: italic;">
                  ✓ Authoritative score based on Long-Term (stable)
                </div>
              ` : `
                <div style="display: flex; justify-content: space-between; align-items: center;">
                  <span style="font-size: 0.85em; color: var(--theme-warning);">
                    ⚠️ Full Intersection only (${m.dual_window.full_intersection.window_days}d, ${m.dual_window.full_intersection.asset_count} assets)
                  </span>
                  <span style="font-size: 0.85em; color: var(--text-secondary);">
                    Sharpe: ${safeFixed(m.dual_window.full_intersection.metrics?.sharpe_ratio, 2)}
                  </span>
                </div>
                <div style="margin-top: 6px; padding: 4px 8px; background: rgba(255, 158, 100, 0.15); border-radius: 4px;">
                  <span style="font-size: 0.8em; color: var(--theme-warning);">
                    ⚠️ Long-term cohort unavailable - metrics on short window (${m.dual_window.exclusions?.reason || 'unknown'})
                  </span>
                </div>
              `}
            </div>
            ` : ''}

            <!-- Structural Score - Comparaison -->
            ${m.risk_version_info ? `
            <div style="margin: 8px 0; padding: 8px; background: rgba(187, 154, 247, 0.1); border-radius: 6px; border-left: 3px solid #bb9af7;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 0.8em; font-weight: 600; color: #bb9af7;">
                  🏗️ Structural Score - Comparison
                </span>
              </div>
              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.85em;">
                <div>
                  <div style="color: var(--text-secondary); margin-bottom: 2px;">Integrated (Legacy):</div>
                  <div style="font-weight: 600; color: var(--text-primary);" title="Structure + Performance (VaR, CVaR, DD, Vol)">
                    ${m.risk_version_info.integrated_structural_legacy != null ? safeFixed(m.risk_version_info.integrated_structural_legacy, 1) : 'N/A'}/100
                  </div>
                  <div style="font-size: 0.7em; color: var(--text-tertiary); margin-top: 2px;">
                    (structure + performance)
                  </div>
                </div>
                <div>
                  <div style="color: var(--text-secondary); margin-bottom: 2px;">Portfolio Structure (V2):</div>
                  <div style="font-weight: 600; color: #bb9af7;" title="HHI, Memecoins, GRI, Diversification only">
                    ${m.risk_version_info.portfolio_structure_score != null ? safeFixed(m.risk_version_info.portfolio_structure_score, 1) : 'N/A'}/100
                  </div>
                  <div style="font-size: 0.7em; color: var(--text-tertiary); margin-top: 2px;">
                    (structure pure)
                  </div>
                </div>
              </div>
              <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(187, 154, 247, 0.2);">
                <div style="font-size: 0.75em; color: var(--text-tertiary); font-style: italic;">
                  ℹ️ V2 = pure structure (HHI, memes, GRI, div) | Legacy = hybrid (+ VaR, CVaR, DD, Vol)
                </div>
              </div>
            </div>
            ` : m.risk_score_structural != null ? `
            <div class="metric-row" style="opacity: 0.8;">
              <span class="metric-label" title="Risk Score Structural: Integrates group exposure (GRI), concentration and portfolio structure">Structural <span style="font-size: 0.7em; color: var(--theme-accent);">struct</span></span>
              <span class="metric-value" style="font-size: 0.9em;">${safeFixed(m.risk_score_structural, 1)}/100</span>
            </div>
            ` : ''}

            <!-- Structure Modulation V2 (Oct 2025) -->
            ${(() => {
      // Lire structure_modulation depuis unified state (si disponible)
      try {
        const unifiedState = window.store?.snapshot?.();
        const structureMod = unifiedState?.structure_modulation;

        // Afficher seulement si activé ET disponible
        if (!structureMod?.enabled || structureMod.structure_score == null) {
          return '';
        }

        const structureScore = structureMod.structure_score;
        const deltaStables = structureMod.delta_stables || 0;
        const deltaCap = structureMod.delta_cap || 0;
        const stablesAfter = structureMod.stables_after || 0;
        const capAfter = structureMod.cap_after || 0;

        // Couleur selon impact
        let color = '#7aa2f7'; // Bleu neutre
        if (deltaStables > 0) color = '#f7768e'; // Rouge (plus de stables = prudence)
        else if (deltaStables < 0) color = '#9ece6a'; // Vert (moins de stables = opportunité)

        return `
                <div style="margin: 8px 0; padding: 8px; background: rgba(122, 162, 247, 0.1); border-radius: 6px; border-left: 3px solid ${color};">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.8em; font-weight: 600; color: ${color};">
                      🏗️ Structure Modulation V2
                    </span>
                    <span style="font-size: 0.7em; color: var(--text-tertiary); font-style: italic;">
                      active
                    </span>
                  </div>
                  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.85em;">
                    <div>
                      <div style="color: var(--text-secondary); margin-bottom: 2px;">Structure Score:</div>
                      <div style="font-weight: 600; color: var(--text-primary);">
                        ${safeFixed(structureScore, 1)}/100
                      </div>
                    </div>
                    <div>
                      <div style="color: var(--text-secondary); margin-bottom: 2px;">Δ Stables:</div>
                      <div style="font-weight: 600; color: ${deltaStables > 0 ? '#f7768e' : deltaStables < 0 ? '#9ece6a' : 'var(--text-primary)'};">
                        ${deltaStables > 0 ? '+' : ''}${deltaStables} pts
                      </div>
                      <div style="font-size: 0.7em; color: var(--text-tertiary); margin-top: 2px;">
                        → ${stablesAfter}% stables
                      </div>
                    </div>
                  </div>
                  <div style="margin-top: 6px; font-size: 0.8em;">
                    <div style="color: var(--text-secondary); margin-bottom: 2px;">Cap effectif:</div>
                    <div style="font-weight: 600; color: var(--text-primary);">
                      ${typeof capAfter === 'number' ? safeFixed(capAfter, 1) : 'N/A'}%
                      ${deltaCap !== 0 ? `<span style="color: ${deltaCap > 0 ? '#9ece6a' : '#f7768e'}; font-size: 0.85em;"> (${deltaCap > 0 ? '+' : ''}${safeFixed(deltaCap, 1)})</span>` : ''}
                    </div>
                  </div>
                  <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(122, 162, 247, 0.2);">
                    <div style="font-size: 0.75em; color: var(--text-tertiary); font-style: italic;">
                      ℹ️ Modulation based on structural quality (HHI, memes, GRI, diversification)
                    </div>
                  </div>
                </div>
                `;
      } catch (error) {
        return '';
      }
    })()}

            <!-- Risk Level -->
            <div class="metric-row">
              <span class="metric-label">Risk Level</span>
              <span class="risk-level risk-${scoreToRiskLevel(m.risk_score)}">${scoreToRiskLevel(m.risk_score).replace('-', ' ').toUpperCase()}</span>
            </div>

            <div class="metric-benchmark">
              📊 <strong>Benchmarks:</strong> Very robust (≥80), Robust (≥65), Moderate (≥50), Fragile (≥35)
            </div>
          </div>

          <!-- Loss & Drawdown Analysis (Fusion VaR + Drawdown) -->
          <div class="risk-card">
            <h3>Loss & Drawdown Analysis <span style="font-size:.8rem; color: var(--theme-text); opacity:.7; font-weight:500; margin-left:.5rem;"><br>VaR 30d • CVaR 60d • DD 180d</span></h3>

            <!-- Short-term Loss Risk (VaR) -->
            <div class="risk-subsection">
              <div class="subsection-title">Short-term Loss (1 day)</div>

              <div class="metric-row">
                <span class="metric-label">VaR 95%</span>
                <span class="metric-value hinted" data-key="var95_1d" data-value="${m.var_95_1d}" style="color: ${getMetricHealth('var_95_1d', m.var_95_1d).color}">
                  ${formatPercent(m.var_95_1d)}
                </span>
              </div>
              <div class="metric-interpretation">
                💡 ${getMetricHealth('var_95_1d', m.var_95_1d).interpretation}
              </div>

              <div class="metric-row">
                <span class="metric-label">VaR 99% (extreme)</span>
                <span class="metric-value hinted" data-key="var99_1d" data-value="${m.var_99_1d}" style="color: ${getMetricHealth('var_99_1d', m.var_99_1d).color}">
                  ${formatPercent(m.var_99_1d)}
                </span>
              </div>
            </div>

            <!-- Tail Risk (CVaR) - Collapsible -->
            <details class="risk-details" style="margin: var(--space-md) 0;">
              <summary style="cursor: pointer; color: var(--theme-text-muted); font-size: 0.85rem; padding: var(--space-xs) 0;">
                <span style="color: var(--brand-primary);">▸</span> Tail Risk (CVaR)
              </summary>
              <div style="margin-top: var(--space-sm); padding-left: var(--space-md);">
                <div class="metric-row" style="font-size: 0.9rem;">
                  <span class="metric-label">CVaR 95%</span>
                  <span class="metric-value hinted" data-key="cvar95_1d" data-value="${m.cvar_95_1d}">${formatPercent(m.cvar_95_1d)}</span>
                </div>
                <div class="metric-row" style="font-size: 0.9rem;">
                  <span class="metric-label">CVaR 99%</span>
                  <span class="metric-value hinted" data-key="cvar99_1d" data-value="${m.cvar_99_1d}">${formatPercent(m.cvar_99_1d)}</span>
                </div>
                <div style="font-size: 0.75rem; color: var(--theme-text-muted); font-style: italic; margin-top: var(--space-xs);">
                  ℹ️ Average loss beyond VaR
                </div>
              </div>
            </details>

            <!-- Historical Drawdown -->
            <div class="risk-subsection">
              <div class="subsection-title">Historical Drawdown</div>

              <div class="metric-row">
                <span class="metric-label">Max Drawdown</span>
                <span class="metric-value hinted" data-key="max_drawdown" data-value="${m.max_drawdown}" style="color: ${getMetricHealth('max_drawdown', m.max_drawdown).color}">
                  ${formatPercent(m.max_drawdown)}
                </span>
              </div>
              <div class="metric-interpretation">
                💡 ${getMetricHealth('max_drawdown', m.max_drawdown).interpretation}
              </div>

              <div class="metric-row">
                <span class="metric-label">Current Drawdown</span>
                <span class="metric-value hinted" data-key="current_drawdown" data-value="${m.current_drawdown}">${formatPercent(m.current_drawdown)}</span>
              </div>
            </div>

            <div class="metric-benchmark">
              📊 <strong>VaR:</strong> Conservative: -4%, Typical: -7%, Aggressive: -12% • <strong>Drawdown:</strong> Good: -30%, Typical: -50%, Extreme: -70%+
            </div>
          </div>

          <!-- Performance -->
          <div class="risk-card">
            <h3>Risk-Adjusted Performance <span style="font-size:.8rem; color: var(--theme-text); opacity:.7; font-weight:500; margin-left:.5rem;"><br>Vol 45d • Sharpe 90d • Sortino 120d • Calmar 365d</span></h3>
            <div class="metric-row">
              <span class="metric-label">Volatility (Annual)</span>
              <span class="metric-value hinted" data-key="volatility_ann" data-value="${m.volatility_annualized}" style="color: ${getMetricHealth('volatility_annualized', m.volatility_annualized).color}">
                ${formatPercent(m.volatility_annualized)}
              </span>
            </div>
            <div class="metric-interpretation">
              💡 ${getMetricHealth('volatility_annualized', m.volatility_annualized).interpretation}
            </div>
            <div class="metric-row">
              <span class="metric-label">Sharpe Ratio</span>
              <span class="metric-value hinted" data-key="sharpe" data-value="${m.sharpe_ratio}" style="color: ${getMetricHealth('sharpe_ratio', m.sharpe_ratio).color}">
                ${safeFixed(m.sharpe_ratio)}
              </span>
            </div>
            <div class="metric-interpretation">
              💡 ${getMetricHealth('sharpe_ratio', m.sharpe_ratio).interpretation}
            </div>
            <div class="metric-row">
              <span class="metric-label">Sortino Ratio</span>
              <span class="metric-value hinted" data-key="sortino" data-value="${m.sortino_ratio}" style="color: ${getMetricHealth('sortino_ratio', m.sortino_ratio).color}">
                ${safeFixed(m.sortino_ratio)}
              </span>
            </div>
            <div class="metric-interpretation">
              💡 ${getMetricHealth('sortino_ratio', m.sortino_ratio).interpretation}
            </div>
            <div class="metric-row">
              <span class="metric-label">Calmar Ratio</span>
              <span class="metric-value">${safeFixed(m.calmar_ratio)}</span>
            </div>
            <div class="metric-benchmark">
              📊 <strong>Benchmarks crypto:</strong> Excellent: >1.5, Good: >1.0, Acceptable: >0.5 (Sharpe)
            </div>
          </div>

          <!-- Diversification -->
          <div class="risk-card">
            <h3>Diversification Analysis <span style="font-size:.8rem; color: var(--theme-text); opacity:.7; font-weight:500; margin-left:.5rem;">corr 90d</span></h3>
            <div class="metric-row">
              <span class="metric-label">Diversification Ratio</span>
              <span class="metric-value hinted" data-key="diversification_ratio" data-value="${c.diversification_ratio}" style="color: ${getMetricHealth('diversification_ratio', c.diversification_ratio).color}">
                ${safeFixed(c.diversification_ratio)}
              </span>
            </div>
            <div class="metric-interpretation">
              💡 ${getMetricHealth('diversification_ratio', c.diversification_ratio).interpretation}
            </div>
            <div class="metric-row">
              <span class="metric-label">Effective Assets</span>
              <span class="metric-value hinted" data-key="effective_assets" data-value="${c.effective_assets}" style="color: ${getMetricHealth('effective_assets', c.effective_assets).color}">
                ${safeFixed(c.effective_assets, 1)}
              </span>
            </div>
            <div class="metric-interpretation">
              💡 ${getMetricHealth('effective_assets', c.effective_assets).interpretation}
            </div>
            <div class="metric-benchmark">
              📊 <strong>Diversification:</strong> Excellent: >0.7, Limited: 0.4-0.7, Low: <0.4
            </div>

            ${c.top_correlations && c.top_correlations.length ? `
              <h4>Top Asset Correlations:</h4>
              ${c.top_correlations.slice(0, 3).map(t => `
                <div class="metric-row">
                  <span class="metric-label">${t.asset1} - ${t.asset2}:</span>
                  <span class="metric-value ${(Math.abs(t.correlation || 0) > 0.7) ? 'text-warning' : 'text-success'}">${((t.correlation || 0) * 100).toFixed(1)}%</span>
                </div>
              `).join('')}
            ` : ``}
          </div>
        </div>

        <!-- Recommendations moved to Top Summary -->

        <!-- Alerts moved to Top Summary -->
      `;

  // Après rendu : brancher les info-bulles et verdicts
  decorateRiskTooltips();
}

// ====== Bitcoin Cycle Chart Functions ======

// Bitcoin halving historical data
const BITCOIN_HALVINGS = [
  { date: '2012-11-28', block: 210000, reward_before: 50, reward_after: 25 },
  { date: '2016-07-09', block: 420000, reward_before: 25, reward_after: 12.5 },
  { date: '2020-05-11', block: 630000, reward_before: 12.5, reward_after: 6.25 },
  { date: '2024-04-20', block: 840000, reward_before: 6.25, reward_after: 3.125 },
  { date: '2028-04-20', block: 1050000, reward_before: 3.125, reward_after: 1.5625, estimated: true }
];

// Expose for modules
window.BITCOIN_HALVINGS = BITCOIN_HALVINGS;

// ====== Tab-specific rendering functions ======
// All tab-specific rendering functions have been moved to dedicated modules:
// - Cycles tab: modules/risk-cycles-tab.js (Bitcoin chart, on-chain indicators, cycle analysis)
// - Targets tab: modules/risk-targets-tab.js (allocation tables, action plans, decision history)
// These modules are imported at the top of the file and called via switchTab()

function renderError(message) {
  const container = document.getElementById('risk-dashboard-content');
  if (!container) {
    debugLogger.warn('⚠️ risk-dashboard-content not found in DOM, skipping error render');
    return;
  }
  container.innerHTML = `
        <div class="error">
          <h3>❌ Error Loading Dashboard</h3>
          <p>${message}</p>
          <button class="refresh-btn" onclick="refreshDashboard()">Try Again</button>
        </div>
      `;
}

function renderBackendUnavailable(message) {
  const container = document.getElementById('risk-dashboard-content');
  if (!container) {
    debugLogger.warn('⚠️ risk-dashboard-content not found in DOM, skipping error render');
    return;
  }
  container.innerHTML = `
        <div style="text-align: center; padding: 3rem; background: var(--warning-bg); border: 1px solid var(--warning); border-radius: var(--radius-lg); color: var(--theme-text);">
          <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
          <h3 style="color: var(--warning); margin-bottom: 1rem;">Risk Backend Unavailable</h3>
          <p style="margin-bottom: 1.5rem; color: var(--theme-text-muted);">${message}</p>
          
          <div style="background: var(--theme-bg); padding: 1.5rem; border-radius: var(--radius-md); margin: 1.5rem 0; text-align: left;">
            <h4 style="color: var(--theme-text); margin-bottom: 1rem;">📋 To display real risk data:</h4>
            <ol style="color: var(--theme-text-muted); line-height: 1.6;">
              <li>Start the Python backend server: <code style="background: var(--theme-surface); padding: 0.2rem 0.4rem; border-radius: 3px;">python main.py</code></li>
              <li>Initialize the price history cache: <code style="background: var(--theme-surface); padding: 0.2rem 0.4rem; border-radius: 3px;">python scripts/init_price_history.py</code></li>
              <li>Verify that the API responds at <a href="#" onclick="window.open(window.globalConfig.get('api_base_url') + '/api/risk/dashboard'); return false;" target="_blank" style="color: var(--brand-primary);">the configured API</a></li>
            </ol>
          </div>

          <div style="margin-top: 2rem;">
            <button class="refresh-btn" onclick="refreshDashboard()">🔄 Retry</button>
            <button class="refresh-btn" onclick="testEndpoint()" style="background: var(--info); margin-left: 0.5rem;">🧪 Tester l'API</button>
          </div>
          
          <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid var(--theme-border); color: var(--theme-text-muted); font-size: 0.85rem;">
            💡 <strong>Note :</strong> This dashboard now exclusively uses real data calculated by the Python backend. 
            Simulated data has been removed to ensure the authenticity of risk metrics.
          </div>
        </div>
      `;
}

function renderApiError(message) {
  const container = document.getElementById('risk-dashboard-content');
  if (!container) {
    debugLogger.warn('⚠️ risk-dashboard-content not found in DOM, skipping API error render');
    return;
  }
  container.innerHTML = `
        <div style="text-align: center; padding: 3rem; background: var(--danger-bg); border: 1px solid var(--danger); border-radius: var(--radius-lg);">
          <div style="font-size: 3rem; margin-bottom: 1rem;">🚨</div>
          <h3 style="color: var(--danger); margin-bottom: 1rem;">Backend API Error</h3>
          <p style="margin-bottom: 1.5rem; color: var(--theme-text-muted);">${message}</p>

          <div style="background: var(--theme-bg); padding: 1.5rem; border-radius: var(--radius-md); margin: 1.5rem 0;">
            <h4 style="color: var(--theme-text); margin-bottom: 1rem;">🔧 Troubleshooting suggestions:</h4>
            <ul style="color: var(--theme-text-muted); text-align: left; line-height: 1.6;">
              <li>Check that the CoinTracking portfolio contains valid data</li>
              <li>Make sure the configured price sources are accessible</li>
              <li>Check the backend logs for more details</li>
            </ul>
          </div>

          <button class="refresh-btn" onclick="refreshDashboard()">🔄 Retry</button>
        </div>
      `;
}

function updateTimestamp(ts, calcTime) {
  const d = new Date(ts);
  const formattedDate = d.toLocaleString('en-US', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit'
  });
  document.getElementById('last-update').textContent = `Updated ${formattedDate}`;
}

// Test endpoint (identique à ta version)
async function testEndpoint() {
  try {
    const url = window.globalConfig.getApiUrl('/api/risk/dashboard');
    const r = await fetch(url);
    const t = await r.text();
    try { JSON.parse(t); alert('API Response received. Check console for details.'); }
    catch { alert('API returned non-JSON response: ' + t.substring(0, 200)); }
  } catch (e) { alert('API test failed: ' + e.message); }
}

// ===== Décoration des tooltips après rendu =====
function decorateRiskTooltips() {
  // === Attache dynamique pour les métriques ===
  document.querySelectorAll('.hinted[data-key]').forEach(el => {
    const key = el.getAttribute('data-key');

    // === Tooltips GRI personnalisés ===
    if (key === 'gri_index') {
      el.addEventListener('mouseenter', (e) => {
        const title = 'Group Risk Index (GRI)';
        const body = `Weighted average of group risks on a 0-10 scale.

Calculation: GRI = Σ (weight × score)

Each group score integrates:
• Structure (40%): counterparty, governance
• Volatility (30%): historical σ
• Correlation (20%): portfolio redundancy
• Liquidity (10%): depth, custody`;
        showTip(title, body, e.clientX, e.clientY);
      });
      el.addEventListener('mousemove', (e) => moveTip(e.clientX, e.clientY));
      el.addEventListener('mouseleave', hideTip);
      el.classList.add('hinted');
      return;
    }

    if (key === 'gri_level') {
      el.addEventListener('mouseenter', (e) => {
        const title = 'Risk Level';
        const body = `Global risk categorization:

• LOW (< 3.0) : Defensive portfolio
• MEDIUM (3.0-6.0) : Moderate risk
• HIGH (≥ 6.0) : Elevated risk

Based on the GRI calculated across the entire portfolio.`;
        showTip(title, body, e.clientX, e.clientY);
      });
      el.addEventListener('mousemove', (e) => moveTip(e.clientX, e.clientY));
      el.addEventListener('mouseleave', hideTip);
      el.classList.add('hinted');
      return;
    }

    if (key === 'groups_count') {
      el.addEventListener('mouseenter', (e) => {
        const title = 'Groups Count';
        const body = `Number of asset categories detected:
BTC, ETH, L1/L0, Stablecoins, Memecoins, etc.

Reflects the structural diversification of the portfolio.`;
        showTip(title, body, e.clientX, e.clientY);
      });
      el.addEventListener('mousemove', (e) => moveTip(e.clientX, e.clientY));
      el.addEventListener('mouseleave', hideTip);
      el.classList.add('hinted');
      return;
    }

    if (key === 'gri_groups_header') {
      el.addEventListener('mouseenter', (e) => {
        const title = 'Exposure & Risk by Group';
        const body = `Each group displays:
• % of portfolio (exposure)
• Score 0-10 (composite risk)

Score = 40% structural + 30% vola + 20% correl + 10% liquid`;
        showTip(title, body, e.clientX, e.clientY);
      });
      el.addEventListener('mousemove', (e) => moveTip(e.clientX, e.clientY));
      el.addEventListener('mouseleave', hideTip);
      el.classList.add('hinted');
      return;
    }

    if (key === 'gri_interpretation') {
      el.addEventListener('mouseenter', (e) => {
        const title = 'Risk level interpretation';
        const body = `This level is derived from the Group Risk Index
and Monte Carlo simulations.

It does not represent investment advice,
but a statistical estimate of overall risk.`;
        showTip(title, body, e.clientX, e.clientY);
      });
      el.addEventListener('mousemove', (e) => moveTip(e.clientX, e.clientY));
      el.addEventListener('mouseleave', hideTip);
      el.classList.add('hinted');
      return;
    }

    // === Tooltips pour les groupes individuels ===
    if (key.startsWith('group:')) {
      const groupName = key.replace('group:', '');
      const groupTooltips = {
        'BTC': 'Market reference asset.\nLow relative volatility, high liquidity,\nmoderate market correlation.',
        'ETH': 'Moderate risk asset.\nHigher volatility than BTC but solid structure.',
        'L1/L0 majors': 'Infrastructure coins (SOL, AVAX, etc.).\nMore volatile, medium to high structural risk.',
        'L2/Scaling': 'Scaling solutions (Arbitrum, Optimism, Polygon).\nModerate risk, correlated to ETH.',
        'DeFi': 'Decentralized finance protocols.\nHigh volatility, smart contract risk.',
        'AI/Data': 'AI and data projects (FET, OCEAN, etc.).\nEmerging sector, high volatility.',
        'SOL': 'Solana: high-performance blockchain.\nSignificant volatility, average correlation.',
        'Gaming/NFT': 'Gaming and NFT tokens.\nHigh volatility, sentiment correlated.',
        'Memecoins': 'Speculative tokens (DOGE, SHIB, PEPE).\nExtreme volatility, very high risk.',
        'Stablecoins': 'Dollar-pegged coins (USDT, USDC, DAI).\nLow but non-zero risk:\n• Issuer counterparty\n• Regulation\n• Potential depeg',
        'Others': 'Secondary coins.\nLow liquidity, high correlation,\nhigh risk.'
      };

      el.addEventListener('mouseenter', (e) => {
        const riskLevel = el.getAttribute('data-risk-level') || '5';
        const title = `${groupName} — ${riskLevel}/10`;
        const body = groupTooltips[groupName] || 'Uncategorized asset group.';
        showTip(title, body, e.clientX, e.clientY);
      });
      el.addEventListener('mousemove', (e) => moveTip(e.clientX, e.clientY));
      el.addEventListener('mouseleave', hideTip);
      el.classList.add('hinted');
      return;
    }

    // === Tooltips Monte Carlo ===
    if (key === 'mc_simulations') {
      el.addEventListener('mouseenter', (e) => {
        const title = 'Simulations';
        const body = `Number of Monte Carlo iterations performed.

Plus il y en a, plus la distribution est fiable.
10,000 simulations = standard statistical precision.`;
        showTip(title, body, e.clientX, e.clientY);
      });
      el.addEventListener('mousemove', (e) => moveTip(e.clientX, e.clientY));
      el.addEventListener('mouseleave', hideTip);
      el.classList.add('hinted');
      return;
    }

    if (key === 'mc_worst_case') {
      el.addEventListener('mouseenter', (e) => {
        const title = 'Worst case scenario (Simulated max drawdown)';
        const body = `The largest loss observed among the 10,000 simulations.

Represents an extreme tail event
(< 0.01% probability).`;
        showTip(title, body, e.clientX, e.clientY);
      });
      el.addEventListener('mousemove', (e) => moveTip(e.clientX, e.clientY));
      el.addEventListener('mouseleave', hideTip);
      el.classList.add('hinted');
      return;
    }

    if (key === 'mc_loss_prob') {
      el.addEventListener('mouseenter', (e) => {
        const title = 'Loss probability > 20%';
        const body = `Probability that a loss exceeding 20%
occurs over the simulated period.

Based on the observed frequency
in the 10,000 scenarios.`;
        showTip(title, body, e.clientX, e.clientY);
      });
      el.addEventListener('mousemove', (e) => moveTip(e.clientX, e.clientY));
      el.addEventListener('mouseleave', hideTip);
      el.classList.add('hinted');
      return;
    }

    // === Métriques de risque standard ===
    // attache une bulle "vivante" qui lit la valeur *au moment* du survol
    el.addEventListener('mouseenter', (e) => {
      // 1) essaie data-value
      let raw = el.getAttribute('data-value');

      // 2) fallback: parse le texte visible (ex: "1.23%" -> 0.0123)
      if (!raw || raw === '0') {
        const txt = (el.textContent || '').trim();
        if (txt.endsWith('%')) {
          const n = parseFloat(txt.replace('%', '').replace(',', '.'));
          raw = isFinite(n) ? String(n / 100) : '';
        } else {
          const n = parseFloat(txt.replace(',', '.'));
          raw = isFinite(n) ? String(n) : '';
        }
      }

      const val = Number(String(raw || '').replace(',', '.'));
      const rating = rate(key, isNaN(val) ? null : val);

      const title = rating.label || key;
      const fmt = (key === 'sharpe' || key === 'sortino') ? num : pct;
      let body = `Current value: ${isNaN(val) ? 'N/A' : fmt(val)}\nReading: ${rating.verdict}`;
      if (key === 'diversification_ratio') {
        body += `\nNote: DR≈1 = neutral; >1 suggests negative correlations; <1 positive correlations.\nThresholds: good ≥0.7, limited 0.4–0.7, low <0.4.`;
      }

      showTip(title, body, e.clientX, e.clientY);
    });

    el.addEventListener('mousemove', (e) => moveTip(e.clientX, e.clientY));
    el.addEventListener('mouseleave', hideTip);
    el.classList.add('hinted');
  });
}

// Expose globally for Phase 3A components
window.decorateRiskTooltips = decorateRiskTooltips;

// ====== Initialization & Store Connection ======

// Subscribe to store changes for content updates
store.subscribe((state) => {
  // Auto-update active tab content when data changes
  const activeTab = state.ui?.activeTab;
  if (activeTab === 'cycles' && state.ccs?.score && state.cycle?.months) {
    renderCyclesContent();
  }
  if (activeTab === 'targets') {
    renderTargetsContent().catch(err => debugLogger.error('Failed to render targets on store change:', err));
  }
});

// ====== Global Functions for UI Interaction ======
window.refreshDashboard = refreshDashboard;
window.testEndpoint = testEndpoint;
// Expose options menu helpers
window.toggleOptionsMenu = toggleOptionsMenu;
window.closeOptionsMenu = closeOptionsMenu;
// Expose cache functions for tab modules
window.getCachedData = getCachedData;
window.setCachedData = setCachedData;
// Expose badge update function for tab modules
window.updateRiskDashboardBadges = updateRiskDashboardBadges;
// Expose score loading function for tab modules
window.loadScoresFromStore = loadScoresFromStore;
// Expose composite score calculator for on-chain indicators
window.calculateCompositeScoreV2 = calculateCompositeScoreV2;

// ===== PHASE 2A: TOAST NOTIFICATION SYSTEM =====
// Extracted to risk-dashboard-toasts.js (Feb 2026)
// Functions imported: showToast, showS3AlertToast, loadDismissedAlerts, shouldShowAlert

// ===== PHASE 2A: ALERT MODAL SYSTEM =====
let currentAlert = null;

async function openAlertModal(alertId) {
  debugLogger.debug('Opening modal for alert:', alertId);

  // Try to find alert in current data first
  let alert = findAlertById(alertId);

  // If not found, fetch from API
  if (!alert) {
    debugLogger.debug('Alert not found in cache, fetching from API...');
    try {
      const alerts = await window.globalConfig.apiRequest('/api/alerts/active');
      if (Array.isArray(alerts)) {
        alert = alerts.find(a => a.id === alertId);
        debugLogger.debug('Found alert in API:', !!alert);
      }
    } catch (error) {
      debugLogger.error('Failed to fetch alert from API:', error);
    }
  }

  if (!alert) {
    debugLogger.error('Alert not found anywhere:', alertId);
    showToast('Alert not found', 'error');
    return;
  }

  currentAlert = alert;
  populateAlertModal(alert);

  const modal = document.getElementById('alert-modal');
  modal.classList.add('show');
  document.body.style.overflow = 'hidden';

  // Close on Escape key
  const handleEscape = (e) => {
    if (e.key === 'Escape') {
      closeAlertModal();
      document.removeEventListener('keydown', handleEscape);
    }
  };
  document.addEventListener('keydown', handleEscape);
}

function closeAlertModal() {
  const modal = document.getElementById('alert-modal');
  modal.classList.remove('show');
  document.body.style.overflow = '';
  currentAlert = null;
}

function populateAlertModal(alert) {
  // Update header
  document.getElementById('modal-severity-badge').textContent = alert.severity;
  document.getElementById('modal-severity-badge').className = `modal-severity-badge modal-severity-${alert.severity.toLowerCase()}`;
  document.getElementById('modal-alert-type').textContent = getAlertTypeDisplayName(alert.alert_type);

  // Update overview
  document.getElementById('modal-alert-id').textContent = alert.id;
  document.getElementById('modal-created-at').textContent = formatDateTime(alert.created_at);
  document.getElementById('modal-current-value').textContent = alert.data.current_value?.toFixed(4) || 'N/A';
  document.getElementById('modal-threshold').textContent = alert.data.adaptive_threshold?.toFixed(4) || 'N/A';

  // Update signals snapshot
  const signalsContainer = document.getElementById('signals-snapshot');
  signalsContainer.innerHTML = '';

  if (alert.data.signals_snapshot) {
    const signals = alert.data.signals_snapshot;

    // Add key signals
    if (signals.volatility) {
      Object.entries(signals.volatility).forEach(([asset, value]) => {
        addSignalCard(signalsContainer, `Volatility ${asset}`, value, 'number');
      });
    }

    if (signals.regime) {
      Object.entries(signals.regime).forEach(([regime, value]) => {
        addSignalCard(signalsContainer, regime.toUpperCase(), value, 'percentage');
      });
    }

    if (signals.correlation) {
      addSignalCard(signalsContainer, 'Avg Correlation', signals.correlation.avg_correlation, 'percentage');
    }

    if (signals.sentiment) {
      addSignalCard(signalsContainer, 'Fear & Greed', signals.sentiment.fear_greed, 'number');
    }

    addSignalCard(signalsContainer, 'Decision Score', signals.decision_score, 'percentage');
    addSignalCard(signalsContainer, 'Confidence', signals.confidence, 'percentage');
    addSignalCard(signalsContainer, 'Contradiction', signals.contradiction_index, 'percentage');
  }

  // Update suggested action
  if (alert.suggested_action) {
    document.getElementById('action-type').textContent = formatActionType(alert.suggested_action.type);
    document.getElementById('action-details').textContent = formatActionDetails(alert.suggested_action);

    // Show/hide apply button based on action type
    const applyBtn = document.getElementById('apply-btn');
    if (alert.suggested_action.type === 'freeze' || alert.suggested_action.type === 'apply_policy') {
      applyBtn.style.display = 'block';
    } else {
      applyBtn.style.display = 'none';
    }
  }

  // Update button states
  updateModalButtonStates(alert);
}

function addSignalCard(container, name, value, type) {
  const card = document.createElement('div');
  card.className = 'signal-card';

  let displayValue = '';
  if (type === 'percentage') {
    displayValue = `${(value * 100).toFixed(1)}%`;
  } else if (type === 'number') {
    displayValue = typeof value === 'number' ? value.toFixed(3) : String(value);
  } else {
    displayValue = String(value);
  }

  card.innerHTML = `
        <div class="signal-name">${name}</div>
        <div class="signal-value">${displayValue}</div>
      `;

  container.appendChild(card);
}

function formatActionType(type) {
  const typeMap = {
    'freeze': 'Freeze Trading',
    'apply_policy': 'Apply Risk Policy',
    'notify_only': 'Notification Only',
    'escalate': 'Escalate Alert'
  };
  return typeMap[type] || type;
}

function formatActionDetails(action) {
  if (action.type === 'freeze') {
    return `Duration: ${action.ttl_minutes} minutes. ${action.reason || ''}`;
  } else if (action.type === 'apply_policy') {
    return `Mode: ${action.mode}, Daily Cap: ${(action.cap_daily * 100).toFixed(1)}%, Ramp Hours: ${action.ramp_hours}`;
  }
  return JSON.stringify(action, null, 2);
}

function updateModalButtonStates(alert) {
  const ackBtn = document.getElementById('ack-btn');
  const snoozeBtn = document.getElementById('snooze-btn');

  // Disable if already acknowledged
  if (alert.acknowledged_at) {
    ackBtn.textContent = 'Acknowledged';
    ackBtn.disabled = true;
    ackBtn.classList.add('modal-action-secondary');
    ackBtn.classList.remove('modal-action-primary');
  } else {
    ackBtn.textContent = 'Acknowledge';
    ackBtn.disabled = false;
    ackBtn.classList.add('modal-action-primary');
    ackBtn.classList.remove('modal-action-secondary');
  }

  // Disable snooze if already snoozed
  if (alert.snooze_until && new Date(alert.snooze_until) > new Date()) {
    snoozeBtn.textContent = 'Snoozed';
    snoozeBtn.disabled = true;
  } else {
    snoozeBtn.textContent = 'Snooze 30m';
    snoozeBtn.disabled = false;
  }
}

// Modal action functions
async function acknowledgeCurrentAlert() {
  if (!currentAlert) return;

  try {
    const response = await window.globalConfig.apiRequest(`/api/alerts/test/acknowledge/${currentAlert.id}`, {
      method: 'POST',
      body: JSON.stringify({ notes: 'Acknowledged from dashboard modal' })
    });

    if (response && (response.ok || response.success || !response.error)) {
      showToast('Alert acknowledged successfully', 'success');
      currentAlert.acknowledged_at = new Date().toISOString();
      currentAlert.acknowledged_by = 'user';
      updateModalButtonStates(currentAlert);

      // Refresh alerts history if visible
      refreshAlertsHistory();
    } else {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (error) {
    debugLogger.error('Failed to acknowledge alert:', error);
    showToast('Failed to acknowledge alert', 'error');
  }
}

async function snoozeCurrentAlert() {
  if (!currentAlert) return;

  try {
    const response = await window.globalConfig.apiRequest(`/api/alerts/test/snooze/${currentAlert.id}`, {
      method: 'POST',
      headers: { 'Idempotency-Key': `snooze-${currentAlert.id}-${Date.now()}` },
      body: JSON.stringify({ minutes: 30 })
    });

    if (response.ok) {
      showToast('Alert snoozed for 30 minutes', 'success');
      currentAlert.snooze_until = new Date(Date.now() + 30 * 60 * 1000).toISOString();
      updateModalButtonStates(currentAlert);
      closeAlertModal();

      // Refresh alerts history if visible
      refreshAlertsHistory();
    } else {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (error) {
    debugLogger.error('Failed to snooze alert:', error);
    showToast('Failed to snooze alert', 'error');
  }
}

async function applyAction() {
  if (!currentAlert || !currentAlert.suggested_action) return;

  const action = currentAlert.suggested_action;
  const confirmMessage = `Apply ${formatActionType(action.type)}?\\n\\nDetails: ${formatActionDetails(action)}`;

  if (!confirm(confirmMessage)) {
    return;
  }

  try {
    // ✅ Use globalConfig.apiRequest() to automatically add X-User header
    const result = await globalConfig.apiRequest(`/api/alerts/${currentAlert.id}/apply`, {
      method: 'POST',
      headers: {
        'Idempotency-Key': `apply-${currentAlert.id}-${Date.now()}`
      },
      body: JSON.stringify({
        applied_by: 'user'
      })
    });

    showToast(`${formatActionType(action.type)} applied successfully`, 'success');
    currentAlert.applied_at = new Date().toISOString();
    currentAlert.applied_by = 'user';
    closeAlertModal();

    // Refresh alerts
    refreshAlertsHistory();
  } catch (error) {
    debugLogger.error('Failed to apply action:', error);
    showToast('Failed to apply action', 'error');
  }
}

// Helper functions
// getAlertTypeDisplayName: imported from risk-dashboard-alerts-history.js

function findAlertById(alertId) {
  // This will be populated by the alerts tab when it loads
  if (window.currentAlertsData) {
    return window.currentAlertsData.find(alert => alert.id === alertId);
  }
  return null;
}

function formatDateTime(dateStr) {
  if (!dateStr) return 'N/A';
  const date = new Date(dateStr);
  return date.toLocaleString();
}

// ===== PHASE 2A: REAL-TIME S3 ALERT MONITORING =====
let alertPollingInterval = null;
let lastKnownAlerts = [];

function startAlertMonitoring() {
  if (alertPollingInterval) {
    // Nettoyer l'ancien intervalle (géré ou non)
    if (window.networkStateManager && typeof alertPollingInterval === 'string') {
      window.networkStateManager.clearManagedInterval(alertPollingInterval);
    } else {
      clearInterval(alertPollingInterval);
    }
  }

  // Check for new S3 alerts every 30 seconds
  // Utiliser managed interval si disponible
  if (window.networkStateManager) {
    alertPollingInterval = window.networkStateManager.createManagedInterval(
      checkForNewS3Alerts,
      30000,
      'S3 Alerts Polling'
    );
    debugLogger.info('✅ S3 Alert monitoring started with managed interval');
  } else {
    alertPollingInterval = setInterval(checkForNewS3Alerts, 30000);
    debugLogger.warn('⚠️ S3 Alert monitoring started with standard interval (network manager not available)');
  }

  // Initial check
  checkForNewS3Alerts();
}

async function checkForNewS3Alerts() {
  try {
    // ✅ Use apiRequestWithRetry with silent fail for polling (fallback to apiRequest if not available)
    let currentAlerts;

    if (typeof globalConfig.apiRequestWithRetry === 'function') {
      // New version with retry logic
      currentAlerts = await globalConfig.apiRequestWithRetry('/api/alerts/active', {
        params: { severity_filter: 'S3' }
      }, true); // silentFail = true pour éviter spam console
    } else {
      // Fallback to standard apiRequest for older versions
      try {
        const response = await globalConfig.apiRequest('/api/alerts/active', {
          params: { severity_filter: 'S3' }
        });
        currentAlerts = response?.data || response;
      } catch (err) {
        // Silent fail on network errors
        if (err.message?.includes('Failed to fetch') || err.message?.includes('Network')) {
          return;
        }
        throw err;
      }
    }

    // Si la requête a échoué silencieusement (offline), ne rien faire
    if (!currentAlerts) {
      return;
    }

    // Find new S3 alerts
    const newAlerts = currentAlerts.filter(alert =>
      alert.severity === 'S3' &&
      !lastKnownAlerts.some(old => old.id === alert.id)
    );

    // Show toast for each new S3 alert
    newAlerts.forEach(alert => {
      showS3AlertToast(alert);
    });

    lastKnownAlerts = currentAlerts;
  } catch (error) {
    // Erreur non-réseau (ex: 500, parsing JSON)
    debugLogger.error('Failed to check for new S3 alerts:', error);
  }
}

// Expose functions globally
window.showToast = showToast;
window.hideToast = hideToast;
window.openAlertModal = openAlertModal;
window.closeAlertModal = closeAlertModal;
window.acknowledgeAlert = acknowledgeAlert;
window.snoozeAlert = snoozeAlert;
window.acknowledgeCurrentAlert = acknowledgeCurrentAlert;
window.snoozeCurrentAlert = snoozeCurrentAlert;
window.applyAction = applyAction;
window.startAlertMonitoring = startAlertMonitoring;
// refreshAlertsHistory: exported from risk-dashboard-alerts-history.js

// Start monitoring when page loads
document.addEventListener('DOMContentLoaded', () => {
  loadDismissedAlerts();  // Load previously dismissed alerts
  startAlertMonitoring();

  // Add event delegation for toast dismiss buttons
  const toastContainer = document.getElementById('toast-container');
  if (toastContainer) {
    toastContainer.addEventListener('click', (event) => {
      // Handle close button clicks
      if (event.target.classList.contains('toast-close')) {
        const toast = event.target.closest('.toast');
        if (toast) {
          debugLogger.debug('Toast close button clicked:', toast.id);
          hideToast(toast.id);
        }
      }

      // Handle action button clicks
      if (event.target.classList.contains('toast-action')) {
        const actionIndex = parseInt(event.target.getAttribute('data-action-index'));
        const toastElement = event.target.closest('.toast');

        if (toastElement && !isNaN(actionIndex)) {
          const toastId = toastElement.id;
          const actions = toastActionsRegistry.get(toastId);

          if (actions && actions[actionIndex]) {
            const action = actions[actionIndex];

            // SECURITY: Execute action safely without eval()
            // Parse and execute the onclick string securely
            try {
              executeToastAction(action.onclick, toastId);
            } catch (error) {
              debugLogger.error('Error executing toast action:', error);
            }
          }
        }
      }
    });
  }

  // Remove old reference since function is now defined globally above
  // window.hideAllToasts = hideAllToasts;

  // Debug logging removed - system ready for production
});

// ===== Options menu helpers =====
function toggleOptionsMenu(event) {
  event.stopPropagation?.();
  const menu = document.getElementById('options-menu');
  const btn = document.getElementById('options-menu-btn');
  if (!menu || !btn) return;
  const isOpen = menu.classList.contains('show');
  if (isOpen) {
    menu.classList.remove('show');
    btn.setAttribute('aria-expanded', 'false');
  } else {
    menu.classList.add('show');
    btn.setAttribute('aria-expanded', 'true');
  }
}

function closeOptionsMenu() {
  const menu = document.getElementById('options-menu');
  const btn = document.getElementById('options-menu-btn');
  if (menu) menu.classList.remove('show');
  if (btn) btn.setAttribute('aria-expanded', 'false');
}

// Close the menu when clicking outside or pressing Escape
document.addEventListener('click', (e) => {
  const menu = document.getElementById('options-menu');
  const btn = document.getElementById('options-menu-btn');
  if (!menu || !btn) return;
  if (!menu.contains(e.target) && !e.target.closest('#options-menu-btn')) {
    closeOptionsMenu();
  }
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeOptionsMenu(); });

// ===== PERSISTENT CACHE SYSTEM (continued) =====
// CACHE_CONFIG and clearAllPersistentCache moved earlier in file for proper initialization order

function initPersistentCache() {
  debugLogger.debug('🗄️ Initializing persistent cache system...');

  // Check existing cached scores
  const cachedScores = getCachedData('SCORES');
  if (cachedScores) {
    const age = Math.round((Date.now() - cachedScores.timestamp) / (1000 * 60));
    debugLogger.debug(`✅ Found cached scores (${age} minutes old)`);

    // Restore scores in localStorage for compatibility
    const dataSource = globalConfig.get('data_source') || 'unknown';
    // ✅ FIX: Ne pas stocker de strings vides - seulement stocker si la valeur existe
    if (cachedScores.data.onchainScore !== null && cachedScores.data.onchainScore !== undefined) {
      localStorage.setItem('risk_score_onchain', cachedScores.data.onchainScore.toString());
    }
    if (cachedScores.data.riskScore !== null && cachedScores.data.riskScore !== undefined) {
      localStorage.setItem('risk_score_risk', cachedScores.data.riskScore.toString());
    }
    if (cachedScores.data.blendedScore !== null && cachedScores.data.blendedScore !== undefined) {
      localStorage.setItem('risk_score_blended', cachedScores.data.blendedScore.toString());
    }
    if (cachedScores.data.ccsScore !== null && cachedScores.data.ccsScore !== undefined) {
      localStorage.setItem('risk_score_ccs', cachedScores.data.ccsScore.toString());
    }
    if (cachedScores.timestamp) {
      localStorage.setItem('risk_score_timestamp', cachedScores.timestamp.toString());
    }
    localStorage.setItem('risk_score_data_source', dataSource);
  }

  // Override clearCache to be selective
  window.originalClearCache = window.clearCache;
  window.clearCache = function (force = false) {
    if (force) {
      debugLogger.debug('🧹 Force clearing all cache');
      clearAllPersistentCache();
      window.originalClearCache?.();
    } else {
      debugLogger.debug('⏭️ Selective cache clear - keeping valid cached data');
      cleanExpiredCache();
    }
  };
}

function setCachedData(type, data) {
  const config = CACHE_CONFIG[type];
  if (!config) return false;

  // Make cache keys data-source-aware
  const dataSource = globalConfig.get('data_source') || 'unknown';
  const sourceAwareKey = `${config.key}_${dataSource}`;

  const cacheEntry = {
    data: data,
    timestamp: Date.now(),
    ttl: config.ttl,
    source: dataSource
  };

  try {
    localStorage.setItem(sourceAwareKey, JSON.stringify(cacheEntry));
    debugLogger.debug(`💾 Cached ${type} data for source ${dataSource} (TTL: ${Math.round(config.ttl / (1000 * 60 * 60))}h)`);
    return true;
  } catch (error) {
    debugLogger.warn(`Failed to cache ${type}:`, error);
    return false;
  }
}

function getCachedData(type) {
  const config = CACHE_CONFIG[type];
  if (!config) return null;

  // Make cache keys data-source-aware
  const dataSource = globalConfig.get('data_source') || 'unknown';
  const sourceAwareKey = `${config.key}_${dataSource}`;

  try {
    const cached = localStorage.getItem(sourceAwareKey);
    if (!cached) return null;

    const cacheEntry = JSON.parse(cached);
    const isExpired = (Date.now() - cacheEntry.timestamp) > cacheEntry.ttl;

    if (isExpired) {
      localStorage.removeItem(sourceAwareKey);
      debugLogger.debug(`⏰ Expired cache removed: ${type} for source ${dataSource}`);
      return null;
    }

    return cacheEntry;
  } catch (error) {
    debugLogger.warn(`Failed to read cache ${type}:`, error);
    localStorage.removeItem(sourceAwareKey);
    return null;
  }
}

function cleanExpiredCache() {
  Object.keys(CACHE_CONFIG).forEach(type => {
    getCachedData(type); // This will auto-remove expired entries
  });
}

// clearAllPersistentCache() moved earlier in file for proper initialization order

// ===== CYCLE CACHE SYSTEM =====
/**
 * Generate hash of cycle data to detect changes
 */
// generateCycleDataHash() moved to modules/risk-cycles-tab.js or risk-targets-tab.js

/**
 * Check if cycle content needs refresh based on data changes
 */
// shouldRefreshCycleContent() moved to modules/risk-cycles-tab.js or risk-targets-tab.js

function updateScoreDisplays(onchainScore, riskScore, blendedScore, ccsScore) {
  // Update score displays using existing UI update functions
  try {
    if (typeof updateOnChainIndicators === 'function') {
      updateOnChainIndicators();
    }

    // Update the main dashboard scores display
    const scoreElements = {
      onchain: document.querySelector('[data-score="onchain"]'),
      risk: document.querySelector('[data-score="risk"]'),
      blended: document.querySelector('[data-score="blended"]'),
      ccs: document.querySelector('[data-score="ccs"]')
    };

    if (scoreElements.onchain) scoreElements.onchain.textContent = Math.round(onchainScore || 0);
    if (scoreElements.risk) scoreElements.risk.textContent = Math.round(riskScore || 0);
    if (scoreElements.blended) scoreElements.blended.textContent = Math.round(blendedScore || 0);
    if (scoreElements.ccs) scoreElements.ccs.textContent = Math.round(ccsScore || 0);

    // Fallback: also update KPI elements by id (for risk-sidebar-full component)
    const elOn = document.getElementById('kpi-onchain');
    const elRisk = document.getElementById('kpi-risk');
    const elBlend = document.getElementById('kpi-blended');
    const elCcs = document.getElementById('ccs-ccs-mix');
    if (elOn && onchainScore != null) elOn.textContent = String(Math.round(onchainScore));
    if (elRisk && riskScore != null) elRisk.textContent = String(Math.round(riskScore));
    if (elBlend && blendedScore != null) elBlend.textContent = String(Math.round(blendedScore));
    if (elCcs && ccsScore != null) elCcs.textContent = String(Math.round(ccsScore));

    // ✅ CRITIQUE: Mettre à jour l'affichage Risk Score dans Risk Overview (ligne 4238)
    const riskDisplayEl = document.querySelector('[data-score="risk-display"]');
    if (riskDisplayEl && riskScore != null) {
      riskDisplayEl.textContent = `${riskScore.toFixed(1)}/100`;
    }

    debugLogger.debug('📊 Score displays updated from cache');
  } catch (error) {
    debugLogger.warn('Error updating score displays:', error);
  }
}

/**
 * Monitor loading states and show error message if they take too long
 * ✅ NEW (Nov 2025): Prevent infinite loading states
 */
function initLoadingTimeoutMonitor() {
  const TIMEOUT_MS = 15000; // 15 seconds
  const loadingElements = [
    { id: 'risk-dashboard-content', name: 'Risk Metrics' },
    { id: 'cycles-content', name: 'Cycle Analysis' },
    { id: 'targets-content', name: 'Strategic Targets' },
    { id: 'alerts-history-content', name: 'Alerts History' }
  ];

  loadingElements.forEach(({ id, name }) => {
    const container = document.getElementById(id);
    if (!container) return;

    // Create a MutationObserver to detect when loading div appears
    const observer = new MutationObserver((mutations) => {
      const loadingDiv = container.querySelector('.loading');

      if (loadingDiv && !loadingDiv.dataset.timeoutSet) {
        // Mark as monitored
        loadingDiv.dataset.timeoutSet = 'true';

        // Set timeout to show error if still loading after 15s
        setTimeout(() => {
          const stillLoading = container.querySelector('.loading');
          if (stillLoading) {
            debugLogger.warn(`⏱️ Loading timeout for ${name} (${TIMEOUT_MS}ms)`);
            stillLoading.innerHTML = `
                  <div class="error-state">
                    <div class="error-icon">⚠️</div>
                    <div class="error-title">Loading Timeout</div>
                    <div class="error-message">
                      ${name} is taking longer than expected to load.
                    </div>
                    <button class="retry-btn" onclick="refreshDashboard(true)">
                      🔄 Retry
                    </button>
                  </div>
                `;
            stillLoading.classList.add('error');
          }
        }, TIMEOUT_MS);
      }
    });

    // Observe container for changes
    observer.observe(container, { childList: true, subtree: true });
  });

  debugLogger.debug('✅ Loading timeout monitor initialized (15s timeout)');
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function () {
  debugLogger.debug('Risk Dashboard CCS MVP initializing...');

  // Initialize data source tracking for cross-tab synchronization
  window.lastKnownDataSource = globalConfig.get('data_source');
  debugLogger.debug(`🔗 Risk Dashboard initialized with data source: ${window.lastKnownDataSource}`);

  // Initialize persistent cache system
  initPersistentCache();
  debugLogger.debug('Persistent cache system initialized');

  // ✅ Initialize loading state timeout monitor
  initLoadingTimeoutMonitor();

  // Listen for data source changes and clear cache
  window.addEventListener('dataSourceChanged', (event) => {
    debugLogger.debug('🔄 Data source changed, clearing cache and reloading...', event.detail);
    clearCache(true);  // ✅ FIX: Force clear all cache when source changes
    // Reload the dashboard after source change
    // ✅ Delay increased to 500ms to give backend time to write config.json
    setTimeout(() => refreshDashboard(true), 500);
  });

  // Initialize shared header
  // Navigation thématique initialisée automatiquement

  // Appliquer le thème immédiatement
  debugLogger.debug('Initializing theme for risk-dashboard page...');
  if (window.globalConfig && window.globalConfig.applyTheme) {
    window.globalConfig.applyTheme();
  }
  if (window.applyAppearance) {
    window.applyAppearance();
  }
  debugLogger.debug('Current theme after risk-dashboard init:', document.documentElement.getAttribute('data-theme'));

  // Hydrate store from localStorage
  store.hydrate();

  // Initialize section collapse states
  initializeSectionStates();

  // Load initial data
  // Init analysis window UI then first refresh
  // controls removed
  refreshDashboard();

  // Initialize store state synchronization
  setTimeout(async () => {
    try {
      debugLogger.debug('🔄 Syncing store state...');
      await store.syncGovernanceState();
      await store.syncMLSignals();
      debugLogger.debug('✅ Store state synced');
    } catch (error) {
      debugLogger.warn('⚠️ Failed to sync store state:', error);
    }
  }, 500);

  // Apply blended strategy as default if no strategy is already selected
  setTimeout(() => {
    const currentStrategy = store.get('targets.strategy');
    if (!currentStrategy) {
      debugLogger.debug('No strategy found, applying Blended as default...');
      applyStrategy('blend');
    }
  }, 1000);

  // Écouter les changements de thème et source pour synchronisation cross-tab
  window.addEventListener('storage', function (e) {
    const expectedKey = (window.globalConfig?.getStorageKey && window.globalConfig.getStorageKey()) || 'crypto_rebal_settings_v1';
    if (e.key === expectedKey) {
      debugLogger.debug('Settings changed in another tab, checking for theme and data source changes...');

      // Check if data source changed
      const currentSource = globalConfig.get('data_source');
      const previousSource = window.lastKnownDataSource;

      if (currentSource && currentSource !== previousSource) {
        debugLogger.debug(`🔄 Data source changed from ${previousSource} to ${currentSource}, clearing cache and reloading...`);
        window.lastKnownDataSource = currentSource;
        clearCache();
        // Reload the risk dashboard after source change
        setTimeout(() => {
          window.location.reload();
        }, 500);
      }

      // Apply theme changes
      setTimeout(() => {
        if (window.globalConfig && window.globalConfig.applyTheme) {
          window.globalConfig.applyTheme();
        }
        if (window.applyAppearance) {
          window.applyAppearance();
        }
      }, 100);
    }
  });

  debugLogger.debug('Risk Dashboard CCS MVP initialized');
});

// Note: Sidebar toggle now handled by flyout-panel Web Component with integrated pin button

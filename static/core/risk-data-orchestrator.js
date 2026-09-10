// static/core/risk-data-orchestrator.js
// Orchestrateur centralisé pour hydrater le risk store avec toutes les métriques calculées
// Utilisé par rebalance.html, analytics-unified.html, execution.html pour parité avec risk-dashboard.html

import { fetchAndComputeCCS, interpretCCS, DEFAULT_CCS_WEIGHTS } from '../modules/signals-engine.js';
import { estimateCyclePosition, blendCCS, getCyclePhase } from '../modules/cycle-navigator.js';
import { fetchAllIndicators, enhanceCycleScore } from '../modules/onchain-indicators.js';
import { calculateCompositeScoreV2 } from '../modules/composite-score-v2.js';
import { getRegimeDisplayData } from '../modules/market-regimes.js';

// ✅ Singleton guard: empêche doubles initialisations
if (window.__risk_orchestrator_init) {
  debugLogger.debug('⚠️ Risk orchestrator already initialized, skipping duplicate');
} else {
  window.__risk_orchestrator_init = true;
  debugLogger.debug('✅ Risk orchestrator initialized (singleton)');
}

/**
 * Hydrate le risk store avec toutes les métriques calculées
 * Appelé après chargement du store pour peupler CCS, Cycle, On-Chain, Regime
 * Ré-émet riskStoreReady après hydratation complète
 *
 * @returns {Promise<void>}
 * @throws {Error} Si riskStore n'est pas disponible ou si calculs échouent
 */
export async function hydrateRiskStore() {
  if (!window.riskStore) {
    throw new Error('riskStore not available - ensure core/risk-dashboard-store.js is loaded first');
  }

  debugLogger.debug('🔄 Starting risk store hydration...');
  const startTime = performance.now();

  // ✅ Détecter hard refresh (Ctrl+Shift+R) pour forcer cache bust
  const isHardRefresh = performance.navigation?.type === 1 ||
                        performance.getEntriesByType?.('navigation')?.[0]?.type === 'reload';
  const forceRefresh = isHardRefresh || false;
  if (forceRefresh) {
    debugLogger.debug('🔄 Hard refresh detected, forcing cache refresh');
  }

  try {
    // Fetch alerts (asynchrone, indépendant des autres calculs)
    const fetchAlerts = async () => {
      try {
        if (!window.globalConfig?.apiRequest) {
          debugLogger.warn('⚠️ globalConfig.apiRequest not available for alerts');
          return [];
        }
        const alertsData = await window.globalConfig.apiRequest('/api/alerts/active', {
          params: { include_snoozed: false }
        });
        return Array.isArray(alertsData) ? alertsData : [];
      } catch (err) {
        debugLogger.warn('⚠️ Alerts fetch failed:', err);
        return [];
      }
    };

    // Fetch risk data from API (pour risk score)
    const fetchRiskData = async () => {
      try {
        if (!window.globalConfig?.apiRequest) {
          debugLogger.warn('⚠️ globalConfig.apiRequest not available for risk data');
          return null;
        }

        // 🔧 FIX: Get current source from globalConfig (MULTI-TENANT CRITICAL - Nov 2025)
        const currentSource = window.globalConfig.get('data_source');
        if (!currentSource) {
          debugLogger.warn('Risk data unavailable: no portfolio source is selected');
          return null;
        }

        // 🔧 FIX: Add _csv_hint to invalidate backend cache when CSV changes (Nov 2025)
        const csvFile = window.userSettings?.csv_selected_file || 'latest';
        const cacheBuster = csvFile !== 'latest' ? csvFile : Date.now().toString().substring(0, 10);

        debugLogger.debug(`🔍 hydrateRiskStore - fetching risk data with source: '${currentSource}', _csv_hint: '${cacheBuster}'`);

        const riskData = await window.globalConfig.apiRequest('/api/risk/dashboard', {
          params: {
            source: currentSource,  // 🔧 FIX: Pass source parameter for multi-tenant isolation
            min_usd: 1.0,
            price_history_days: 365,
            lookback_days: 90,
            use_dual_window: true,  // Cohérent avec risk-dashboard-main-controller.js
            risk_version: 'v2_active',
            _csv_hint: cacheBuster  // 🔧 Invalide cache backend quand CSV change
          }
        });
        return riskData;
      } catch (err) {
        debugLogger.warn('⚠️ Risk data fetch failed:', err);
        return null;
      }
    };

    // Fetch governance state (pour contradiction_index autoritaire)
    const fetchGovernanceState = async () => {
      try {
        if (!window.globalConfig?.apiRequest) {
          debugLogger.warn('⚠️ globalConfig.apiRequest not available for governance state');
          return null;
        }
        return await window.globalConfig.apiRequest('/execution/governance/state');
      } catch (err) {
        debugLogger.warn('⚠️ Governance state fetch failed:', err);
        return null;
      }
    };

    // ✅ Utiliser risk_score déjà calculé par l'API backend (source de vérité)
    const calculateRiskScore = (riskData) => {
      if (!riskData?.risk_metrics) return null;

      // L'API /api/risk/dashboard retourne déjà un risk_score calculé côté Python
      // avec la formule autoritaire depuis services/risk_management.py
      const riskScore = riskData.risk_metrics.risk_score;

      if (riskScore != null && typeof riskScore === 'number') {
        console.debug('✅ Risk score from backend API:', riskScore);
        return Math.max(0, Math.min(100, riskScore));
      }

      // ❌ Fallback: si risk_score manque, retourner null (ne pas calculer côté client)
      debugLogger.warn('⚠️ risk_score missing from API response, using fallback');
      return null;
    };

    // Calculer toutes les métriques en parallèle pour performance optimale
    // NOTE: estimateCyclePosition() est SYNCHRONE, on le wrap dans Promise.resolve()
    const [ccsResult, cycleResult, indicatorsResult, alertsResult, riskResult, governanceResult] = await Promise.allSettled([
      fetchAndComputeCCS().catch(err => {
        debugLogger.warn('⚠️ CCS calculation failed:', err);
        return null;
      }),
      Promise.resolve().then(() => {
        try {
          return estimateCyclePosition();
        } catch (err) {
          debugLogger.warn('⚠️ Cycle estimation failed:', err);
          return null;
        }
      }),
      fetchAllIndicators({ force: forceRefresh }).catch(err => {
        debugLogger.warn('⚠️ On-chain indicators fetch failed:', err);
        return null;
      }),
      fetchAlerts(),
      fetchRiskData(),
      fetchGovernanceState()
    ]);

    // Extraire les résultats (null si échec)
    let ccs = ccsResult.status === 'fulfilled' ? ccsResult.value : null;
    const cycle = cycleResult.status === 'fulfilled' ? cycleResult.value : null;
    const indicators = indicatorsResult.status === 'fulfilled' ? indicatorsResult.value : null;
    const alerts = alertsResult.status === 'fulfilled' ? alertsResult.value : [];
    const riskData = riskResult.status === 'fulfilled' ? riskResult.value : null;
    const governanceState = governanceResult.status === 'fulfilled' ? governanceResult.value : null;

    // Calculer risk score depuis riskData
    const riskScore = riskData ? calculateRiskScore(riskData) : null;

    // Utiliser contradiction_index autoritaire depuis governance state (source de vérité backend)
    const contradiction = governanceState?.contradiction_index ?? null;

    // Ajouter interpretation au CCS si manquant
    if (Number.isFinite(ccs?.score) && !ccs.interpretation) {
      ccs = { ...ccs, interpretation: interpretCCS(ccs.score) };
    }

    // Calculer score composite on-chain avec V2 (plus précis, dynamic weighting always enabled)
    let onchainScore = null;
    if (indicators && Object.keys(indicators).length > 0) {
      try {
        const compositeResult = calculateCompositeScoreV2(indicators, true);
        // calculateCompositeScoreV2 returns { score, confidence, contributors, ... }
        onchainScore = compositeResult?.score ?? null;
      } catch (err) {
        debugLogger.warn('⚠️ On-chain composite score calculation failed:', err);
      }
    }

    // Récupérer état actuel AVANT tout calcul (pour préserver données existantes)
    const currentState = window.riskStore.getState();

    // Calculer blended score (CCS + Cycle)
    // Blend CCS with Cycle to get ccsStar
    let ccsStar = null;
    if (Number.isFinite(ccs?.score) && cycle) {
      try {
        const blendResult = blendCCS(ccs.score, cycle.months || 18);
        // blendCCS returns { originalCCS, cycleScore, blendedCCS, cycleWeight, phase }
        ccsStar = blendResult?.blendedCCS ?? null;
      } catch (err) {
        debugLogger.warn('⚠️ CCS blend calculation failed:', err);
      }
    }

    // Calculate final blended score (CCS*0.5 + OnChain*0.3 + Risk*0.2)
    let blendedScore = null;
    if ([ccsStar, onchainScore, riskScore].every(Number.isFinite)) {
      const wCCS = 0.50;
      const wOnchain = 0.30;
      const wRisk = 0.20;
      blendedScore = Math.round(ccsStar * wCCS + onchainScore * wOnchain + riskScore * wRisk);
    }

    // Calculer market regime (nécessite blended + onchain + risk scores)
    let regime = null;
    if ([blendedScore, onchainScore, riskScore].every(Number.isFinite)) {
      try {
        const regimeData = getRegimeDisplayData(
          blendedScore,
          onchainScore,
          riskScore,
          cycle?.score ?? null,
          cycle?.direction ?? null,
          cycle?.confidence ?? null
        );
        // getRegimeDisplayData returns { regime: {...}, risk_budget, allocation, recommendations }
        regime = regimeData?.regime ?? null;
      } catch (err) {
        debugLogger.warn('⚠️ Market regime calculation failed:', err);
      }
    }

    // Construire nouveau état avec métriques calculées
    const newState = {
      ...currentState,
      // CCS Mixte
      ccs: ccs || { available: false, score: null, reason: 'CCS unavailable' },

      // Cycle position
      cycle: cycle ? {
        ...cycle,
        ccsStar: ccsStar // CCS blended with cycle
      } : (currentState.cycle || {
        ccsStar: null,
        months: null,
        phase: null
      }),

      // Market regime
      regime: regime || {
        phase: null,
        confidence: null,
        divergence: null
      },

      // Risk metrics complets (pour analytics-unified.html: var_95_1d, max_drawdown, etc.)
      risk: riskData || null,

      // Scores unifiés
      scores: {
        ...(currentState.scores || {}),
        onchain: onchainScore,
        blended: blendedScore,
        // Risk score calculé depuis API /api/risk/dashboard (TOUJOURS utiliser API si disponible)
        risk: riskScore,
        cycle: cycle?.score ?? (currentState.scores?.cycle ?? null)
      },

      // Governance (pour compatibilité avec risk-sidebar-full.js)
      governance: {
        ...(currentState.governance || {}),
        // Contradiction doit être dans governance.contradiction_index (0..1)
        contradiction_index: contradiction ?? currentState.governance?.contradiction_index
      },

      // Contradiction (stockage racine pour compatibilité legacy)
      contradiction: contradiction ?? currentState.contradiction,

      // Alerts (IMPORTANT: doit être un tableau pour risk-sidebar-full.js)
      alerts: alerts || [],

      // Metadata hydratation
      _hydrated: true,
      _hydration_timestamp: new Date().toISOString(),
      _hydration_duration_ms: Math.round(performance.now() - startTime),
      _hydration_source: 'risk-data-orchestrator'  // ✅ Traçabilité source
    };

    // Mise à jour atomique du store
    window.riskStore.setState(newState);

    // Ré-émettre riskStoreReady APRÈS hydratation complète
    // Detail inclut flag hydrated:true pour différencier du premier event (store vide)
    window.dispatchEvent(new CustomEvent('riskStoreReady', {
      detail: {
        store: window.riskStore,
        hydrated: true,
        timestamp: Date.now(),
        metrics: {
          ccs: ccs !== null,
          cycle: cycle !== null,
          onchain: onchainScore !== null,
          blended: blendedScore !== null,
          risk: riskScore !== null,
          regime: regime !== null,
          contradiction: contradiction !== null,
          alerts: alerts.length > 0
        }
      }
    }));

    const duration = Math.round(performance.now() - startTime);
    debugLogger.debug(`✅ Risk store hydrated successfully in ${duration}ms`, {
      ccs: ccs ? `${ccs.score} (${ccs.interpretation?.label || ccs.interpretation})` : 'N/A',
      cycle: cycle ? `${cycle.phase?.phase || cycle.phase} (${cycle.months}mo)` : 'N/A',
      onchain: onchainScore !== null && typeof onchainScore === 'number' ? onchainScore.toFixed(1) : (onchainScore || 'N/A'),
      blended: blendedScore !== null && typeof blendedScore === 'number' ? blendedScore.toFixed(1) : (blendedScore || 'N/A'),
      risk: riskScore !== null && typeof riskScore === 'number' ? riskScore.toFixed(1) : (riskScore || 'N/A'),
      regime: regime ? regime.phase : 'N/A',
      contradiction: contradiction !== null && typeof contradiction === 'number' ? contradiction.toFixed(2) : (contradiction || 'N/A'),
      alerts: `${alerts.length} alerts`
    });

  } catch (err) {
    debugLogger.error('❌ Failed to hydrate risk store:', err);

    // Marquer échec d'hydratation dans le store
    const currentState = window.riskStore.getState();
    window.riskStore.setState({
      ...currentState,
      _hydrated: false,
      _hydration_error: err.message,
      _hydration_timestamp: new Date().toISOString()
    });

    throw err;
  }
}

/**
 * Auto-init : Hydrate le store dès que le DOM est prêt
 * Garantit que les modules de calcul sont exécutés et le store rempli
 */
async function autoInit() {
  if (window.__smartfolioAuthReady && !await window.__smartfolioAuthReady) return;

  // Attendre que riskStore soit disponible (chargé par risk-dashboard-store.js)
  if (window.riskStore) {
    hydrateRiskStore().catch(err => {
      debugLogger.error('Auto-init hydration failed:', err);
    });
  } else {
    // Retry après 100ms si store pas encore chargé
    debugLogger.debug('⏳ Waiting for riskStore to be available...');
    setTimeout(autoInit, 100);
  }
}

// Listen for data source changes and re-hydrate store
window.addEventListener('dataSourceChanged', (event) => {
  debugLogger.debug(`🔄 Data source changed in orchestrator: ${event.detail.oldSource} → ${event.detail.newSource}`);

  // Clear risk store to force fresh data fetch
  if (window.riskStore) {
    const clearedState = {
      ccs: { score: null },
      cycle: { ccsStar: null, months: null, phase: null },
      regime: { phase: null, confidence: null, divergence: null },
      risk: null,
      scores: { onchain: null, blended: null, risk: null },
      governance: { contradiction_index: null },
      contradiction: null,
      alerts: [],
      _hydrated: false,
      _cleared_for_source_change: true,
      _cleared_timestamp: new Date().toISOString()
    };
    window.riskStore.setState(clearedState);
    debugLogger.debug('✅ Risk store cleared for source change');
  }

  // Re-hydrate store with new source data
  setTimeout(() => {
    hydrateRiskStore().catch(err => {
      debugLogger.error('Failed to re-hydrate store after source change:', err);
    });
  }, 100);
});

// Démarrer auto-init selon état du DOM
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', autoInit);
} else {
  // DOM déjà prêt (module chargé tardivement)
  autoInit();
}

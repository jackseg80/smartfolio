import { selectCapPercent } from '../selectors/governance.js';
import { calculateAdaptiveWeights } from '../governance/contradiction-policy.js';

/**
 * Simulation Engine - Pipeline exact d'Analytics Unified en mode simulation
 * Version simplifiée avec fallbacks pour éviter les blocages d'imports
 */

console.debug('🎭 SIM: Simulation Engine loaded');

// Imports pour système de contradiction unifié
let contradictionModules = null;

async function loadContradictionModules() {
  if (!contradictionModules) {
    // Use unified calculateAdaptiveWeights from contradiction-policy.js
    console.debug('✅ SIM: Using unified contradiction modules from contradiction-policy.js');
    contradictionModules = {
      smoothContradiction: (value, prevValue, config, state) => {
        // Simple fallback - just apply basic smoothing
        if (prevValue !== undefined && prevValue !== null) {
          const alpha = config?.ema_alpha ?? 0.25;
          const smoothed = alpha * value + (1 - alpha) * prevValue;
          return { value01: smoothed, level: 'medium', persistCount: 0 };
        }
        return { value01: value, level: 'medium', persistCount: 0 };
      },
      getEffectiveContradiction01: (opts) => ({
        value01: opts.state?.governance?.contradiction_index ?? 0.5,
        stale: false,
        useBaseWeights: false
      }),
      // ✅ UNIFIED: Use centralized calculateAdaptiveWeights from contradiction-policy.js
      // No more duplication - single source of truth for weight calculations
      calculateAdaptiveWeights: (base, state) => {
        // Delegate to centralized implementation
        const result = calculateAdaptiveWeights(base, state);
        // Return format compatible with simulation engine expectations
        return {
          cycle: result.cycle,
          onchain: result.onchain,
          risk: result.risk,
          wCycle: result.cycle,
          wOnchain: result.onchain,
          wRisk: result.risk
        };
      },
      applyContradictionCaps: (policy, state) => policy
    };
  }
}

// Use window.store if available, fallback to simple implementation
const store = window.store || {
  get: (path) => {
    // Simuler des valeurs par défaut
    if (path === 'scores.onchain') return 50;
    if (path === 'scores.risk') return 50;
    if (path === 'scores.blended') return 50;
    if (path === 'cycle.score') return 50;
    if (path === 'cycle.months') return 18;
    if (path === 'cycle.phase') return 'expansion';
    return null;
  },
  set: (path, value) => {},
  snapshot: () => ({
    scores: { onchain: 50, risk: 50, blended: 50, cycle: 50 },
    wallet: { balances: [], total: 0 },
    cycle: { months: 18, phase: 'expansion', score: 50 }
  })
};

const getRegimeDisplayData = () => null;
const analyzeContradictorySignals = () => [];
const estimateCyclePosition = () => ({ months: 18, score: 50, phase: 'expansion' });
const getCyclePhase = () => 'expansion';

// Import des modules réels
let realComputeMacroTargetsDynamic = null;
let assetGroupsModule = null;
let phaseEngineModule = null;

// Fonction pour charger les vraies implementations
async function loadRealComputeFunction() {
  if (!realComputeMacroTargetsDynamic) {
    try {
      const module = await import('../core/unified-insights-v2.js');
      realComputeMacroTargetsDynamic = module.computeMacroTargetsDynamic;
      console.debug('✅ SIM: Real computeMacroTargetsDynamic loaded from unified-insights-v2.js');
    } catch (error) {
      (window.debugLogger?.warn || console.warn)('⚠️ SIM: Failed to load real computeMacroTargetsDynamic, using fallback:', error.message);
    }
  }

  // Charger aussi les modules de contradiction
  await loadContradictionModules();

  // Charger les groupes d'assets pour position réelle
  if (!assetGroupsModule) {
    try {
      assetGroupsModule = await import('../shared-asset-groups.js');
      console.debug('✅ SIM: Asset groups module loaded');
    } catch (error) {
      (window.debugLogger?.warn || console.warn)('⚠️ SIM: Failed to load shared-asset-groups.js:', error.message);
    }
  }

  // Charger le Phase Engine unifié
  if (!phaseEngineModule) {
    try {
      phaseEngineModule = await import('../core/phase-engine.js');
      console.debug('✅ SIM: Real Phase Engine loaded');
    } catch (error) {
      (window.debugLogger?.warn || console.warn)('⚠️ SIM: Failed to load phase-engine.js, using fallback tilts:', error.message);
    }
  }
}

// Wrapper function qui utilise la vraie fonction ou fallback
function computeMacroTargetsDynamic(ctx, rb, walletStats) {
  if (realComputeMacroTargetsDynamic) {
    console.debug('✅ SIM: Using real computeMacroTargetsDynamic');
    return realComputeMacroTargetsDynamic(ctx, rb, walletStats);
  }

  // Fallback si la vraie fonction n'est pas disponible
  console.debug('🎭 SIM: Using fallback computeMacroTargetsDynamic');

  const stables = rb?.target_stables_pct || 25;
  const riskyPool = 100 - stables;

  // Poids de base relatifs (hors stables) - Portfolio neutre
  let base = {
    BTC: 0.42,           // 42% du risky pool
    ETH: 0.28,           // 28% du risky pool
    'L1/L0 majors': 0.06,
    SOL: 0.06,
    'L2/Scaling': 0.06,
    DeFi: 0.05,
    'AI/Data': 0.04,
    'Gaming/NFT': 0.02,
    Memecoins: 0.01,
    Others: 0.00
  };

  // Modulateurs simples par régime/sentiment
  const bull = ctx?.regime === 'bull' || (ctx?.cycle_score >= 70);
  const bear = ctx?.regime === 'bear' || (ctx?.cycle_score <= 30);

  if (bull) {
    // Mode bull: moins BTC, plus ETH/L2/SOL
    base.BTC *= 0.95;
    base.ETH *= 1.08;
    base['L2/Scaling'] *= 1.15;
    base.SOL *= 1.10;
  }

  if (bear) {
    // Mode prudent: réduire long tail
    base.Memecoins *= 0.5;
    base['Gaming/NFT'] *= 0.7;
    base.DeFi *= 0.85;
  }

  // Normaliser la somme (hors stables)
  const sumBase = Object.values(base).reduce((s, v) => s + v, 0) || 1;
  for (const k in base) {
    base[k] = base[k] / sumBase;
  }

  // Convertir en points (%) sur le riskyPool
  const targets = { Stablecoins: +stables.toFixed(1) };
  for (const [k, v] of Object.entries(base)) {
    targets[k] = +(v * riskyPool).toFixed(1);
  }

  // Ajustement somme=100 (gestion arrondis)
  const sum = Object.values(targets).reduce((a, b) => a + b, 0);
  const diff = +(100 - sum).toFixed(1);
  if (Math.abs(diff) >= 0.1) {
    const heavy = 'BTC'; // Ajuster sur BTC
    targets[heavy] = +(targets[heavy] + diff).toFixed(1);
  }

  (window.debugLogger?.debug || console.log)('🎯 Fallback targets computed:', targets);
  return targets;
}

/**
 * Calculer la position actuelle réelle depuis wallet.balances
 */
async function computeCurrentAllocation(wallet) {
  if (!assetGroupsModule || !wallet?.balances?.length) {
    // Fallback sur position simulée
    (window.debugLogger?.warn || console.warn)('⚠️ SIM: Using fallback allocation (no real wallet data)');
    return {
      Stablecoins: 25,
      BTC: 40,
      ETH: 20,
      'L1/L0 majors': 5,
      SOL: 5,
      'L2/Scaling': 3,
      DeFi: 2,
      totalValue: 10000
    };
  }

  try {
    const { groupAssetsByClassification } = assetGroupsModule;
    const groupedData = groupAssetsByClassification(wallet.balances);
    const totalValue = wallet.total || wallet.balances.reduce((sum, item) => sum + (parseFloat(item.value_usd) || 0), 0);

    // Convertir en pourcentages
    const allocation = {};
    groupedData.forEach(group => {
      const percentage = totalValue > 0 ? (group.value / totalValue) * 100 : 0;
      allocation[group.label] = +percentage.toFixed(1);
    });

    allocation.totalValue = totalValue;

    console.debug('✅ SIM: Real allocation computed from wallet:', {
      totalValue,
      groups: Object.keys(allocation).length - 1,
      top3: Object.entries(allocation)
        .filter(([key]) => key !== 'totalValue')
        .sort(([,a], [,b]) => b - a)
        .slice(0, 3)
    });

    return allocation;
  } catch (error) {
    debugLogger.error('❌ SIM: Failed to compute real allocation:', error);
    // Fallback
    return {
      Stablecoins: 25,
      BTC: 40,
      ETH: 20,
      'L1/L0 majors': 5,
      SOL: 5,
      'L2/Scaling': 3,
      DeFi: 2,
      totalValue: 10000
    };
  }
}

// État global de simulation
let simulationState = {
  sourceData: null,
  lastSnapshot: null,
  isLoaded: false,
  // États déterministes pour remplacer Math.random()
  deterministicState: {
    upDays: 0,      // Compteur pour hystérésis up
    downDays: 0,    // Compteur pour hystérésis down
    volStressCounter: 0,  // Pour circuit breaker vol
    ddStressCounter: 0,   // Pour circuit breaker drawdown
    contradictionLevel: 0  // Niveau de contradictions simulé
  },
  // État de smoothing/hystérésis pour contradiction
  smoothState: {
    prevLevel: undefined,
    prevValue: undefined,
    persistCount: 0
  }
};

// Utilitaire pour initialiser deterministicState si nécessaire
function ensureDeterministicState() {
  if (!simulationState.deterministicState) {
    simulationState.deterministicState = {
      upDays: 0,
      downDays: 0,
      volStressCounter: 0,
      ddStressCounter: 0,
      contradictionLevel: 0
    };
  }
}

/**
 * 1. INITIALISATION - Charge le snapshot une fois
 */
export async function initSimulation({ sourceId }) {
  console.debug('🎭 SIM: initSimulation called:', { sourceId });

  try {
    // Charger la vraie fonction computeMacroTargetsDynamic
    await loadRealComputeFunction();

    // Utiliser la même logique que analytics-unified.html pour charger les données
    const snapshot = await loadSourceSnapshot(sourceId);

    simulationState = {
      sourceData: snapshot,
      lastSnapshot: Date.now(),
      isLoaded: true,
      sourceId
    };

    (window.debugLogger?.debug || console.log)('🎭 SIM: sourceLoaded -', {
      timestamp: new Date().toISOString(),
      walletItems: snapshot.wallet?.balances?.length || 0,
      totalValue: snapshot.wallet?.total || 0,
      hasRealComputeFunction: !!realComputeMacroTargetsDynamic
    });

    return { success: true, data: snapshot };
  } catch (error) {
    debugLogger.error('🎭 SIM: initSimulation failed:', error);
    return { success: false, error: error.message };
  }
}

/**
 * Charge le snapshot depuis la source (même logique qu'Analytics Unified)
 */
async function loadSourceSnapshot(sourceId) {
  // Réutiliser la logique de chargement d'analytics-unified.html
  if (typeof window.loadBalanceData === 'function') {
    try {
      const balanceResult = await window.loadBalanceData();
      if (balanceResult.success) {
        return {
          wallet: {
            balances: balanceResult.data?.items || [],
            total: balanceResult.data?.items?.reduce((sum, item) => sum + (parseFloat(item.value_usd) || 0), 0) || 0
          },
          scores: {
            onchain: store.get('scores.onchain') || 50,
            risk: store.get('scores.risk') || 50,
            blended: store.get('scores.blended') || 50,
            cycle: store.get('cycle.score') || 50
          },
          regime: store.get('market.regime'),
          cycle: {
            months: store.get('cycle.months'),
            phase: store.get('cycle.phase'),
            score: store.get('cycle.score')
          }
        };
      }
    } catch (error) {
      (window.debugLogger?.warn || console.warn)('🎭 SIM: loadBalanceData failed, using store fallback:', error);
    }
  }

  // Fallback: utiliser le store actuel ou données simulées
  const storeSnapshot = store.snapshot();
  return {
    wallet: storeSnapshot.wallet || {
      balances: [
        { symbol: 'BTC', value_usd: 4000 },
        { symbol: 'ETH', value_usd: 2000 },
        { symbol: 'USDC', value_usd: 2500 },
        { symbol: 'SOL', value_usd: 1000 },
        { symbol: 'MATIC', value_usd: 500 }
      ],
      total: 10000
    },
    scores: storeSnapshot.scores || {
      onchain: 50,
      risk: 50,
      blended: 50,
      cycle: 50
    },
    regime: storeSnapshot.market?.regime,
    cycle: storeSnapshot.cycle || { months: 18, phase: 'expansion', score: 50 }
  };
}

/**
 * 2. CONSTRUCTION DU CONTEXTE DE SIMULATION
 */
export function buildSimulationContext(liveContext, uiOverrides = {}) {
  console.debug('🎭 SIM: buildSimulationContext called');

  if (!simulationState.isLoaded) {
    throw new Error('Simulation not initialized - call initSimulation first');
  }

  const context = {
    // Sources de base
    wallet: simulationState.sourceData.wallet,
    sourceScores: simulationState.sourceData.scores,

    // Scores avec overrides UI
    scores: {
      cycle: uiOverrides.cycleScore ?? simulationState.sourceData.scores.cycle ?? 50,
      onchain: uiOverrides.onChainScore ?? simulationState.sourceData.scores.onchain ?? 50,
      risk: uiOverrides.riskScore ?? simulationState.sourceData.scores.risk ?? 50
    },

    // Confiances (réalistes: cycle ~0.46, onchain ~0.6)
    confidences: {
      cycle: uiOverrides.cycleConf ?? 0.46,
      onchain: uiOverrides.onchainConf ?? 0.6
    },

    // Sentiment score (0-100) - 4th component per DECISION_INDEX_V2.md
    sentimentScore: uiOverrides.sentimentScore ?? 50,

    // Backend decision override
    backendDecision: uiOverrides.backendDecision || null,

    // Pénalité contradiction
    contradictionPenalty: uiOverrides.contradictionPenalty ?? 0.1,

    // Phase Engine
    phaseEngine: {
      enabled: uiOverrides.phaseEngine?.enabled ?? false,
      mode: uiOverrides.phaseEngine?.mode ?? 'shadow',
      forcedPhase: uiOverrides.phaseEngine?.forcedPhase ?? null,
      offset: uiOverrides.phaseEngine?.offset ?? 0
    },

    // Métadonnées
    presetInfo: uiOverrides.presetInfo ?? { name: 'Custom', desc: '' },
    timestamp: new Date().toISOString(),
    source: 'simulation'
  };

  console.debug('🎭 SIM: Context built:', context);
  return context;
}

/**
 * 3. CALCUL DECISION INDEX (aligné sur unified-insights-v2.js - Production)
 *
 * ⚠️ ALIGNÉ avec unified-insights-v2.js - 3 composantes:
 * raw_decision_score = (cycle × wCycle) + (onchain × wOnchain) + (risk × wRisk)
 * final_score = raw_decision_score × phase_factor
 *
 * NOTE: Le sentiment n'est PAS une composante du DI (poids 0.33/0.39/0.28).
 * Le sentiment est utilisé comme OVERRIDE contextuel via applySentimentOverride().
 *
 * ⚠️ IMPORTANT — Sémantique Risk:
 * Risk est un score POSITIF (0..100, plus haut = mieux/robuste).
 * Ne jamais inverser (pas de 100 - risk).
 */
export function computeDecisionIndex(context) {
  console.debug('🎭 SIM: computeDecisionIndex called');

  const { scores, confidences, backendDecision, contradictionPenalty } = context;

  // PRIORITÉ 1: Backend Decision (si forcé)
  if (backendDecision && typeof backendDecision.score === 'number') {
    const result = {
      di: Math.max(0, Math.min(100, backendDecision.score)),
      source: 'backend_forced',
      confidence: backendDecision.confidence || 0.8,
      penalties: { contradiction: 0 },
      reasoning: 'Backend decision forced via UI override'
    };

    (window.debugLogger?.debug || console.log)('🎭 SIM: diComputed -', result);
    return result;
  }

  // PRIORITÉ 2: Formule 3-composantes (aligné unified-insights-v2.js)
  // Poids par défaut alignés sur production: 0.33/0.39/0.28
  let wCycle = context.weights?.cycle ?? context.weights?.wCycle ?? 0.33;
  let wOnchain = context.weights?.onchain ?? context.weights?.wOnchain ?? 0.39;
  let wRisk = context.weights?.risk ?? context.weights?.wRisk ?? 0.28;

  // Règle adaptative: Cycle ≥ 90 → boost wCycle (comme unified-insights-v2.js)
  if (scores.cycle >= 90) {
    wCycle = 0.45;
    wOnchain = 0.35;
    wRisk = 0.20;
    console.debug('🚀 SIM: Adaptive weights: Cycle ≥ 90 → cycle boost');
  } else if (scores.cycle >= 70) {
    wCycle = 0.40;
    wOnchain = 0.37;
    wRisk = 0.23;
  }

  // Ajuster poids selon confiances (cycle et onchain uniquement)
  wCycle *= (0.8 + 0.4 * (confidences?.cycle ?? 0.46));
  wOnchain *= (0.8 + 0.4 * (confidences?.onchain ?? 0.6));
  // Risk n'a pas de "confidence" séparée - le score EST la mesure de robustesse

  // Initialiser deterministicState si nécessaire
  ensureDeterministicState();

  // Pénalité contradiction (déterministe basée sur la divergence entre scores)
  let contradictionCount = simulationState.deterministicState.contradictionLevel;
  const scoreSpread = Math.abs(scores.cycle - scores.onchain);
  if (scoreSpread > 30) {
    contradictionCount = Math.min(2, contradictionCount + 1);
  } else if (scoreSpread < 10) {
    contradictionCount = Math.max(0, contradictionCount - 1);
  }

  simulationState.deterministicState.contradictionLevel = contradictionCount;
  const contraFactor = Math.max(0.8, 1 - contradictionPenalty * contradictionCount);
  wOnchain *= contraFactor;

  // Normaliser les poids pour somme = 1
  const sum = wCycle + wOnchain + wRisk;
  wCycle /= sum;
  wOnchain /= sum;
  wRisk /= sum;

  // Calcul DI brut (3 composantes - aligné production)
  // ✅ Risk est positif (0-100, plus haut = mieux) - pas d'inversion
  const rawDI =
    (scores.cycle * wCycle) +
    (scores.onchain * wOnchain) +
    (scores.risk * wRisk);

  // Phase factor (basé uniquement sur cycle, pas sentiment)
  // bullish: cycle ≥ 70 → factor 1.0-1.05
  // bearish: cycle < 40 → factor 0.90
  // moderate: sinon → factor 0.95
  let phaseFactor = 1.0;
  if (scores.cycle >= 70) {
    phaseFactor = 1.0 + (scores.cycle - 70) * 0.001; // Slight boost for strong bull
  } else if (scores.cycle < 40) {
    phaseFactor = 0.90; // Bearish phase penalty
  } else {
    phaseFactor = 0.95; // Moderate
  }

  const di = Math.round(rawDI * phaseFactor);

  // Confidence basée sur cycle + onchain (les 2 seules métriques avec confidence réelle)
  const confidence = Math.min(1, ((confidences?.cycle ?? 0.46) + (confidences?.onchain ?? 0.6)) / 2);

  const result = {
    di: Math.max(0, Math.min(100, di)),
    source: 'decision_index_v2',
    confidence,
    weights: { wCycle, wOnchain, wRisk },
    phaseFactor,
    penalties: {
      contradiction: contradictionCount,
      contradictionFactor: contraFactor
    },
    reasoning: `DI V2 (aligned): Cycle(${scores.cycle}×${wCycle.toFixed(2)}) + OnChain(${scores.onchain}×${wOnchain.toFixed(2)}) + Risk(${scores.risk}×${wRisk.toFixed(2)}) × phase(${phaseFactor.toFixed(2)})`
  };

  (window.debugLogger?.debug || console.log)('🎭 SIM: diComputed -', result);
  return result;
}

/**
 * Applique l'override de sentiment sur les targets (comme unified-insights-v2.js)
 *
 * Le sentiment n'est PAS une composante du DI mais un OVERRIDE contextuel:
 * - Extreme Fear (<25): opportunité en bull, protection en bear
 * - Extreme Greed (>75): prise de profits systématique
 *
 * @param {Object} targets - Targets d'allocation { BTC: 30, ETH: 25, ... }
 * @param {number} sentimentScore - Score sentiment ML (0-100)
 * @param {string} regime - Régime de marché ('bull', 'bear', 'neutral')
 * @param {number} cycleScore - Score cycle pour déterminer le contexte
 * @returns {Object} - Targets modifiées avec override info
 */
export function applySentimentOverride(targets, sentimentScore, regime = 'neutral', cycleScore = 50) {
  const extremeFear = sentimentScore < 25;
  const extremeGreed = sentimentScore > 75;

  // Si pas de sentiment extrême, retourner les targets inchangées
  if (!extremeFear && !extremeGreed) {
    return {
      targets,
      overrideApplied: false,
      overrideReason: null
    };
  }

  // Copie des targets pour modification
  const modified = { ...targets };
  let overrideReason = null;

  // Déterminer le contexte bull/bear
  const bullContext = (regime === 'bull') || (cycleScore >= 70);
  const bearContext = (regime === 'bear') || (cycleScore <= 30);

  if (extremeFear) {
    if (bullContext) {
      // 🐂 Bull + Fear = OPPORTUNITÉ (contrarian buy)
      if (modified.ETH) modified.ETH *= 1.15;
      if (modified.SOL) modified.SOL *= 1.20;
      if (modified['L2/Scaling']) modified['L2/Scaling'] *= 1.20;
      if (modified.DeFi) modified.DeFi *= 1.10;
      if (modified.Memecoins) modified.Memecoins = Math.max(modified.Memecoins * 1.5, 2);
      overrideReason = `🐂 Bull + Extreme Fear (${sentimentScore}) → Buying opportunity`;
      console.debug('💎 SIM OVERRIDE: Opportunistic allocation (Bull + Fear)');
    } else if (bearContext) {
      // 🐻 Bear + Fear = DANGER (capitulation)
      if (modified.Memecoins) modified.Memecoins *= 0.3;
      if (modified['Gaming/NFT']) modified['Gaming/NFT'] *= 0.5;
      if (modified.DeFi) modified.DeFi *= 0.7;
      if (modified['AI/Data']) modified['AI/Data'] *= 0.8;
      overrideReason = `🐻 Bear + Extreme Fear (${sentimentScore}) → Protection`;
      console.debug('🛡️ SIM OVERRIDE: Defensive allocation (Bear + Fear)');
    } else {
      // 😐 Neutral + Fear = Prudence légère
      if (modified.Memecoins) modified.Memecoins *= 0.7;
      if (modified['Gaming/NFT']) modified['Gaming/NFT'] *= 0.8;
      overrideReason = `😐 Neutral + Extreme Fear (${sentimentScore}) → Prudence`;
      console.debug('⚖️ SIM OVERRIDE: Cautious allocation (Neutral + Fear)');
    }
  }

  if (extremeGreed) {
    // ⚠️ Extreme Greed = TOUJOURS prise de profits
    if (modified.Memecoins) modified.Memecoins *= 0.3;
    if (modified['Gaming/NFT']) modified['Gaming/NFT'] *= 0.5;
    if (modified['AI/Data']) modified['AI/Data'] *= 0.7;
    if (modified.DeFi) modified.DeFi *= 0.8;
    overrideReason = overrideReason
      ? `${overrideReason} + Extreme Greed (${sentimentScore}) → Prise de profits`
      : `⚠️ Extreme Greed (${sentimentScore}) → Prise de profits`;
    console.debug('⚠️ SIM OVERRIDE: Profit-taking (Extreme Greed)');
  }

  // Renormaliser pour que la somme = 100%
  const stables = modified.Stablecoins || 0;
  const riskyTotal = Object.entries(modified)
    .filter(([k]) => k !== 'Stablecoins')
    .reduce((sum, [, v]) => sum + v, 0);

  if (riskyTotal > 0) {
    const targetRisky = 100 - stables;
    const scale = targetRisky / riskyTotal;
    for (const key of Object.keys(modified)) {
      if (key !== 'Stablecoins') {
        modified[key] = +(modified[key] * scale).toFixed(1);
      }
    }
  }

  return {
    targets: modified,
    overrideApplied: true,
    overrideReason
  };
}

/**
 * 4. CALCUL RISK BUDGET avec options avancées
 */
export function computeRiskBudget(di, options = {}, marketOverlays = {}) {
  console.debug('🎭 SIM: computeRiskBudget called:', { di, options, marketOverlays });

  const {
    curve = 'linear',
    min_stables = 10,
    max_stables = 60,
    hysteresis = { on: false, upDays: 3, downDays: 5 },
    circuit_breakers = { vol_z_gt: 2.5, dd_90d_pct_lt: -20, floor_stables_if_trigger: 70 }
  } = options;

  let target_stables_pct;

  // Calcul selon la courbe
  if (curve === 'sigmoid') {
    // Courbe sigmoïde: plus agressive aux extrêmes
    const x = (di - 50) / 50; // Centré sur 0, range [-1, 1]
    const sigmoid = 1 / (1 + Math.exp(-3 * x)); // Sigmoid agressif
    target_stables_pct = min_stables + (max_stables - min_stables) * (1 - sigmoid);
  } else {
    // Courbe linéaire inversée (DI=0 → max_stables, DI=100 → min_stables)
    target_stables_pct = max_stables - ((di / 100) * (max_stables - min_stables));
  }

  // Initialiser deterministicState si nécessaire
  ensureDeterministicState();

  // Hysteresis déterministe
  let hysteresisApplied = false;
  if (hysteresis.on && simulationState.lastRiskBudget) {
    const lastValue = simulationState.lastRiskBudget.target_stables_pct;
    const delta = Math.abs(target_stables_pct - lastValue);
    const trend = target_stables_pct > lastValue ? 'up' : 'down';

    if (delta < 5) {
      // Mise à jour des compteurs de jours
      if (trend === 'up') {
        simulationState.deterministicState.upDays += 1;
        simulationState.deterministicState.downDays = 0;
      } else {
        simulationState.deterministicState.downDays += 1;
        simulationState.deterministicState.upDays = 0;
      }

      // Appliquer hystérésis selon les seuils
      const requiredDays = trend === 'up' ? hysteresis.upDays : hysteresis.downDays;
      const currentDays = trend === 'up' ? simulationState.deterministicState.upDays : simulationState.deterministicState.downDays;

      if (currentDays < requiredDays) {
        target_stables_pct = lastValue;
        hysteresisApplied = true;
      }
    } else {
      // Reset compteurs si changement important
      simulationState.deterministicState.upDays = 0;
      simulationState.deterministicState.downDays = 0;
    }
  }

  // Circuit breakers avec market overlays
  let cbVolTriggered = false;
  let cbDdTriggered = false;

  // CB Volatilité: utiliser market overlay ou fallback neutre (0) si absent
  const volZ = (marketOverlays && typeof marketOverlays.vol_z === 'number') ? marketOverlays.vol_z : 0;
  if (volZ >= circuit_breakers.vol_z_gt) {
    cbVolTriggered = true;
    target_stables_pct = Math.max(target_stables_pct, circuit_breakers.floor_stables_if_trigger);
    console.debug('🚨 SIM: CB Volatilité déclenché:', { vol_z: volZ, threshold: circuit_breakers.vol_z_gt });
  }

  // CB Drawdown: utiliser market overlay ou fallback sur DI
  const ddPct = marketOverlays.dd_90d_pct !== undefined ? marketOverlays.dd_90d_pct : ((di - 50) / 10);
  if (ddPct <= circuit_breakers.dd_90d_pct_lt) {
    cbDdTriggered = true;
    target_stables_pct = Math.max(target_stables_pct, circuit_breakers.floor_stables_if_trigger);
    console.debug('🚨 SIM: CB Drawdown déclenché:', { dd_90d_pct: ddPct, threshold: circuit_breakers.dd_90d_pct_lt });
  }

  // Clamps finaux
  target_stables_pct = Math.max(min_stables, Math.min(max_stables, target_stables_pct));

  const result = {
    target_stables_pct: Math.round(target_stables_pct * 10) / 10, // 1 décimale
    flags: {
      hysteresis: hysteresisApplied,
      cb_vol: cbVolTriggered,
      cb_dd: cbDdTriggered
    },
    curve,
    clamps: { min_stables, max_stables },
    options
  };

  // Sauvegarder pour hystérésis suivante
  simulationState.lastRiskBudget = result;

  (window.debugLogger?.debug || console.log)('🎭 SIM: riskBudgetUpdated -', result);
  return result;
}

/**
 * 5. CALCUL TARGETS (réutilise computeMacroTargetsDynamic)
 */
export function computeTargets(riskBudget, context) {
  console.debug('🎭 SIM: computeTargets called');

  // Construire le contexte pour computeMacroTargetsDynamic (même format)
  const ctx = {
    cycle_score: context.scores.cycle,
    regime: context.scores.onchain > 70 ? 'bull' : context.scores.onchain < 30 ? 'bear' : 'neutral',
    sentiment: context.scores.onchain < 25 ? 'extreme_fear' : 'neutral',
    governance_mode: 'Standard',
    flags: {
      phase_engine: context.phaseEngine?.enabled && context.phaseEngine?.mode === 'apply' ? 'apply' : 'off'
    }
  };

  // Stats wallet simplifiées
  const walletStats = {
    topWeightSymbol: 'BTC', // Simulation
    topWeightPct: 35 // Simulation
  };

  // RÉUTILISER la fonction existante
  const targets = computeMacroTargetsDynamic(ctx, riskBudget, walletStats);

  (window.debugLogger?.debug || console.log)('🎭 SIM: targetsComputed -', targets);
  return targets;
}

/**
 * 6. APPLICATION PHASE ENGINE TILTS
 */
export async function applyPhaseEngineTilts(targets, phaseConfig) {
  console.debug('🎭 SIM: applyPhaseEngineTilts called:', phaseConfig);

  if (!phaseConfig?.enabled) {
    console.debug('🎭 SIM: Phase Engine disabled, skipping tilts');
    return { ...targets };
  }

  // Utiliser le vrai Phase Engine si disponible et en mode apply
  if (phaseEngineModule && phaseConfig.mode === 'apply') {
    try {
      const { applyPhaseTilts } = phaseEngineModule;

      // Appeler avec une phase string et déballer le retour {targets, metadata}
      const result = await applyPhaseTilts(targets, (phaseConfig.forcedPhase || 'neutral'));
      const unwrapped = (result && result.targets) ? result.targets : result;
      console.debug('✅ SIM: Real Phase Engine tilts applied:', {
        phase: phaseConfig.forcedPhase || 'auto',
        originalStables: targets?.Stablecoins,
        tiltedStables: unwrapped?.Stablecoins
      });

      return unwrapped;
    } catch (error) {
      (window.debugLogger?.warn || console.warn)('⚠️ SIM: Real Phase Engine failed, using fallback:', error.message);
    }
  }

  // Fallback sur tilts simplifiés
  const tiltedTargets = { ...targets };
  const phase = phaseConfig.forcedPhase || 'neutral';

  switch (phase) {
    case 'risk_off':
      tiltedTargets.Stablecoins = Math.min(80, tiltedTargets.Stablecoins + 15);
      tiltedTargets.Memecoins = Math.max(0, tiltedTargets.Memecoins * 0.1);
      break;

    case 'eth_expansion':
      tiltedTargets.ETH = tiltedTargets.ETH + 5;
      tiltedTargets['L2/Scaling'] = tiltedTargets['L2/Scaling'] + 3;
      tiltedTargets.Stablecoins = Math.max(5, tiltedTargets.Stablecoins - 2);
      break;

    case 'full_altseason':
      tiltedTargets.Memecoins = tiltedTargets.Memecoins * 2.5;
      tiltedTargets.Others = tiltedTargets.Others * 2;
      tiltedTargets.Stablecoins = Math.max(5, tiltedTargets.Stablecoins - 15);
      break;
  }

  // Renormaliser à 100%
  const sum = Object.values(tiltedTargets).reduce((a, b) => a + b, 0);
  if (sum !== 100) {
    const factor = 100 / sum;
    for (const key in tiltedTargets) {
      tiltedTargets[key] = Math.round(tiltedTargets[key] * factor * 10) / 10;
    }
  }

  (window.debugLogger?.debug || console.log)('🎭 SIM: phaseTiltsApplied -', { phase, original: targets, tilted: tiltedTargets });
  return tiltedTargets;
}

/**
 * 7. APPLICATION GOVERNANCE CAPS/FLOORS
 */
export function applyGovernanceCaps(targets, govSettings = {}) {
  console.debug('🎭 SIM: applyGovernanceCaps called:', govSettings);

  const cappedTargets = { ...targets };
  let capsTriggered = [];

  // Caps par défaut
  const caps = {
    'L2/Scaling': govSettings.caps?.L2 ?? 15,
    'DeFi': govSettings.caps?.DeFi ?? 10,
    'Gaming/NFT': govSettings.caps?.Gaming ?? 5,
    'Memecoins': govSettings.caps?.Memes ?? 8,
    'Others': govSettings.caps?.Others ?? 5,
    'BTC': govSettings.max_btc ?? 50,
    'ETH': govSettings.max_eth ?? 35,
    ...govSettings.caps
  };

  // Appliquer les caps
  for (const [group, cap] of Object.entries(caps)) {
    if (cappedTargets[group] > cap) {
      const excess = cappedTargets[group] - cap;
      cappedTargets[group] = cap;
      capsTriggered.push({ group, excess, cap });

      // Redistribuer l'excès sur les stables
      cappedTargets.Stablecoins += excess;
    }
  }

  if (capsTriggered.length > 0) {
    (window.debugLogger?.debug || console.log)('🎭 SIM: capsTriggered -', capsTriggered);
  }

  return { targets: cappedTargets, capsTriggered };
}

/**
 * 8. PLAN D'EXÉCUTION SIMULÉ
 */
export function planOrdersSimulated(current, targets, execPolicy = {}) {
  console.debug('🎭 SIM: planOrdersSimulated called');

  const {
    global_delta_threshold_pct = 2,
    bucket_delta_threshold_pct = 1,
    min_lot_eur = 10,
    slippage_bps = 20,
    cap01 = null
  } = execPolicy;

  const orders = [];
  let totalDelta = 0;

  // Calculer les ordres nécessaires
  for (const [group, targetPct] of Object.entries(targets)) {
    const currentPct = current[group] || 0;
    let delta = targetPct - currentPct;

    if (typeof cap01 === 'number' && Number.isFinite(cap01) && cap01 > 0) {
      const capPP = cap01 * 100;
      if (delta > capPP) {
        delta = capPP;
      } else if (delta < -capPP) {
        delta = -capPP;
      }
    }

    const absDelta = Math.abs(delta);


    // Créer ordre si delta > seuil bucket
    if (absDelta >= bucket_delta_threshold_pct) {
      totalDelta += absDelta;
      const estimatedLot = (absDelta / 100) * (current.totalValue || 10000); // Simulation

      if (estimatedLot >= min_lot_eur) {
        orders.push({
          group,
          action: delta > 0 ? 'BUY' : 'SELL',
          currentPct: Math.round(currentPct * 10) / 10,
          targetPct: Math.round(targetPct * 10) / 10,
          deltaPct: Math.round(delta * 10) / 10,
          estimatedLot: Math.round(estimatedLot),
          slippageBps: slippage_bps,
          priority: absDelta > 5 ? 'HIGH' : 'NORMAL'
        });
      }
    }
  }

  // Vérifier seuil global
  const shouldExecute = totalDelta >= global_delta_threshold_pct;

  const result = {
    orders: shouldExecute ? orders : [],
    summary: {
      totalDelta: Math.round(totalDelta * 10) / 10,
      globalThreshold: global_delta_threshold_pct,
      shouldExecute,
      ordersCount: shouldExecute ? orders.length : 0,
      estimatedValue: orders.reduce((sum, o) => sum + o.estimatedLot, 0)
    },
    policy: execPolicy
  };

  (window.debugLogger?.debug || console.log)('🎭 SIM: ordersPlanned -', result);
  return result;
}

/**
 * 9. EXPLICATION PIPELINE
 */
export function explainPipeline(context, steps) {
  console.debug('🎭 SIM: explainPipeline called');

  const { di, riskBudget, targets, finalTargets, cappedResult, orders } = steps;

  // Arbre hiérarchique pour SimInspector
  const explainTree = {
    root: {
      label: 'Pipeline de Simulation',
      status: 'completed',
      children: {
        inputs: {
          label: 'Decision Inputs (4 composantes)',
          status: 'completed',
          data: {
            scores: context.scores,
            sentimentScore: context.sentimentScore ?? 50,
            confidences: context.confidences,
            overrides: context.backendDecision ? 'Backend forced' : 'DI V2 (4 comp.)'
          }
        },

        di: {
          label: `Decision Index: ${di.di}/100`,
          status: 'completed',
          data: {
            source: di.source,
            confidence: `${Math.round(di.confidence * 100)}%`,
            reasoning: di.reasoning
          }
        },

        riskBudget: {
          label: `Risk Budget: ${riskBudget.target_stables_pct}% Stables`,
          status: riskBudget.flags.hysteresis || riskBudget.flags.cb_vol || riskBudget.flags.cb_dd ? 'warning' : 'completed',
          data: {
            curve: riskBudget.curve,
            flags: riskBudget.flags,
            clamps: riskBudget.clamps
          }
        },

        targets: {
          label: 'Allocation Targets',
          status: 'completed',
          data: targets
        },

        phase: context.phaseEngine?.enabled ? {
          label: 'Phase Engine Tilts',
          status: 'completed',
          data: {
            mode: context.phaseEngine.mode,
            phase: context.phaseEngine.forcedPhase || 'auto',
            applied: context.phaseEngine.mode === 'apply'
          }
        } : null,

        governance: {
          label: 'Governance Caps',
          status: cappedResult.capsTriggered?.length > 0 ? 'warning' : 'completed',
          data: {
            capsTriggered: cappedResult.capsTriggered || [],
            finalTargets: cappedResult.targets
          }
        },

        execution: {
          label: `Execution Plan: ${orders.orders.length} ordres`,
          status: orders.summary.shouldExecute ? 'action' : 'idle',
          data: {
            shouldExecute: orders.summary.shouldExecute,
            totalDelta: orders.summary.totalDelta,
            threshold: orders.summary.globalThreshold
          }
        }
      }
    }
  };

  // Résumé en langage naturel
  const summaryNL = generateNaturalLanguageSummary(context, steps);

  return { explainTree, summaryNL };
}

function generateNaturalLanguageSummary(context, steps) {
  const { di, riskBudget, orders } = steps;

  let summary = `Simulation avec Decision Index de ${di.di}/100 (${di.source}). `;

  summary += `Budget risk recommande ${riskBudget.target_stables_pct}% en stables `;

  if (riskBudget.flags.hysteresis) {
    summary += `(hysteresis applied) `;
  }

  if (riskBudget.flags.cb_vol || riskBudget.flags.cb_dd) {
    summary += `(circuit-breakers triggered) `;
  }

  summary += `. Plan d'exécution: `;

  if (orders.summary.shouldExecute) {
    summary += `${orders.orders.length} ordres requis (delta total ${orders.summary.totalDelta}% > seuil ${orders.summary.globalThreshold}%).`;
  } else {
    summary += `aucun ordre nécessaire (delta ${orders.summary.totalDelta}% < seuil ${orders.summary.globalThreshold}%).`;
  }

  return summary;
}

/**
 * 10. GESTION PRESETS
 */
export function loadPreset(presetObj) {
  console.debug('🎭 SIM: loadPreset called:', presetObj.name);

  // Convertir preset vers format UI
  const uiState = {
    // Decision Inputs
    cycleScore: presetObj.inputs?.cycleScore ?? 50,
    onChainScore: presetObj.inputs?.onChainScore ?? 50,
    riskScore: presetObj.inputs?.riskScore ?? 50,
    cycleConf: presetObj.inputs?.cycleConf ?? 0.7,
    onchainConf: presetObj.inputs?.onchainConf ?? 0.6,
    regimeConf: presetObj.inputs?.regimeConf ?? 0.5,
    contradictionPenalty: presetObj.inputs?.contradictionPenalty ?? 0.1,
    backendDecision: presetObj.inputs?.backendDecision || null,

    // Phase Engine
    phaseEngine: presetObj.regime_phase || { enabled: false, mode: 'shadow' },

    // Risk Budget
    riskBudget: presetObj.risk_budget || { curve: 'linear', min_stables: 10, max_stables: 60 },

    // Governance
    governance: presetObj.governance || { caps: {}, max_btc: 50, max_eth: 35 },

    // Execution
    execution: presetObj.execution || { global_delta_threshold_pct: 2, bucket_delta_threshold_pct: 1 }
  };

  (window.debugLogger?.debug || console.log)('🎭 SIM: presetLoaded -', { name: presetObj.name, version: presetObj.version });
  return uiState;
}

export function exportPreset(uiState, name, description) {
  console.debug('🎭 SIM: exportPreset called:', name);

  const preset = {
    version: '1.0',
    created_with: 'simulator-v1',
    name,
    desc: description,
    created_at: new Date().toISOString(),

    inputs: {
      cycleScore: uiState.cycleScore,
      onChainScore: uiState.onChainScore,
      riskScore: uiState.riskScore,
      cycleConf: uiState.cycleConf,
      onchainConf: uiState.onchainConf,
      regimeConf: uiState.regimeConf,
      contradictionPenalty: uiState.contradictionPenalty,
      backendDecision: uiState.backendDecision
    },

    regime_phase: uiState.phaseEngine,
    risk_budget: uiState.riskBudget,
    governance: uiState.governance,
    execution: uiState.execution
  };

  return preset;
}

/**
 * 11. URL HASH STATE
 */
export function stateToUrlHash(uiState) {
  try {
    // UTF-8 safe encoding (btoa only supports Latin1)
    const jsonString = JSON.stringify(uiState);
    const compressed = btoa(unescape(encodeURIComponent(jsonString)));
    return `#sim=${compressed}`;
  } catch (error) {
    (window.debugLogger?.warn || console.warn)('🎭 SIM: Failed to encode state to URL:', error);
    return '#sim=error';
  }
}

export function stateFromUrlHash() {
  try {
    const hash = window.location.hash;
    if (!hash.startsWith('#sim=')) return null;

    const compressed = hash.substring(5);
    // UTF-8 safe decoding (reverse of stateToUrlHash)
    const state = JSON.parse(decodeURIComponent(escape(atob(compressed))));

    (window.debugLogger?.debug || console.log)('🎭 SIM: State restored from URL hash');
    return state;
  } catch (error) {
    (window.debugLogger?.warn || console.warn)('🎭 SIM: Failed to decode state from URL:', error);
    return null;
  }
}

/**
 * 12. PIPELINE COMPLET
 */
export async function simulateFullPipeline(uiOverrides = {}) {
  console.debug('🎭 SIM: simulateFullPipeline called with overrides:', uiOverrides);

  try {
    // 1. Contexte de base
    const baseContext = buildSimulationContext(simulationState.sourceData, uiOverrides);
    const presetInfo = uiOverrides?.presetInfo ?? baseContext?.presetInfo ?? null;
    const executionOverrides = uiOverrides?.execution ?? {};

    // 2. Système de Contradiction Unifié
    // ⚠️ ALIGNÉ avec unified-insights-v2.js (lignes 50-52)
    const BASE_WEIGHTS = { cycle: 0.5, onchain: 0.3, risk: 0.2 };
    const SMOOTHING_CFG = { ema_alpha: 0.25, deadband: 2, persistence: 3 };

    // Construire snapshot d'état pour contradiction unifié
    const snapshot = {
      governance: {
        contradiction_index: uiOverrides.contradictionIndex ?? 0.5,
        updated_at: uiOverrides.forceStale ? new Date(Date.now() - 45 * 60 * 1000).toISOString() : new Date().toISOString(),
        overrides_count: Object.keys(uiOverrides).length
      },
      scores: baseContext.scores
    };

    // Effective contradiction avec staleness gating
    const eff = contradictionModules.getEffectiveContradiction01({
      state: snapshot,
      baseWeights: BASE_WEIGHTS,
      ttlMin: 30
    });

    // Smoothing/hystérésis - ensure smoothState is initialized
    if (!simulationState.smoothState) {
      simulationState.smoothState = {
        prevLevel: undefined,
        prevValue: undefined,
        persistCount: 0
      };
    }

    const sm = contradictionModules.smoothContradiction(eff.value01, simulationState.smoothState.prevValue, SMOOTHING_CFG, simulationState.smoothState);
    simulationState.smoothState.prevLevel = sm.level;
    simulationState.smoothState.prevValue = sm.value01;
    simulationState.smoothState.persistCount = sm.persistCount;

    // Construire état unifié pour engine + données governance utiles
    const governanceSource = simulationState.sourceData?.governance ?? {};

    const takeNumber = (...values) => {
      for (const value of values) {
        if (typeof value === 'number' && Number.isFinite(value)) {
          return value;
        }
      }
      return null;
    };

    const policyCap01 = takeNumber(
      executionOverrides.cap01,
      governanceSource?.active_policy?.cap_daily,
      governanceSource?.policy?.cap_daily,
      store.get('governance.active_policy.cap_daily'),
      store.get('governance.policy.cap_daily')
    );

    const engineCapDaily = takeNumber(
      governanceSource?.caps?.engine_cap,
      governanceSource?.engine_cap_daily,
      store.get('governance.caps.engine_cap'),
      store.get('governance.engine_cap_daily')
    );

    const stateForEngine = {
      ...snapshot,
      governance: {
        ...governanceSource,
        ...snapshot.governance,
        contradiction_index: sm.value01,
        last_fresh_contrad01: eff.stale ? (snapshot?.governance?.last_fresh_contrad01 ?? sm.value01) : sm.value01
      }
    };

    if (policyCap01 != null) {
      stateForEngine.governance.active_policy = {
        ...(governanceSource.active_policy ?? {}),
        ...stateForEngine.governance.active_policy,
        cap_daily: policyCap01
      };
    }

    if (engineCapDaily != null) {
      stateForEngine.governance.caps = {
        ...(governanceSource.caps ?? {}),
        ...stateForEngine.governance.caps,
        engine_cap: engineCapDaily
      };
    }


    // Poids adaptatifs
    const weights = eff.useBaseWeights
      ? BASE_WEIGHTS
      : contradictionModules.calculateAdaptiveWeights(BASE_WEIGHTS, stateForEngine);

    // 3. Decision Index avec poids adaptatifs
    const di = computeDecisionIndex({ ...baseContext, weights });

    // 4. Risk Budget
    // ⚠️ PRIORITÉ: regimeData.risk_budget si disponible (source unique comme Analytics)
    let riskBudget;
    if (stateForEngine.regimeData?.risk_budget?.target_stables_pct != null) {
      riskBudget = {
        target_stables_pct: stateForEngine.regimeData.risk_budget.target_stables_pct,
        source: 'market-regimes (v2)',
        regime_based: true
      };
      console.debug('✅ SIM: Using regimeData.risk_budget as source of truth:', riskBudget);
    } else {
      riskBudget = computeRiskBudget(di.di, uiOverrides.riskBudget, uiOverrides.marketOverlays);
      console.debug('⚠️ SIM: Fallback to computed risk budget (no regimeData):', riskBudget);
    }

    // 5. Targets de base
    const targets = computeTargets(riskBudget, { ...baseContext, weights });

    // 6. Phase tilts
    const finalTargets = await applyPhaseEngineTilts(targets, uiOverrides.phaseEngine);

    // 7. Caps de contradiction
    const contradictionCaps = contradictionModules.applyContradictionCaps(finalTargets, stateForEngine);

    // 8. Governance caps
    const cappedResult = applyGovernanceCaps(contradictionCaps, uiOverrides.governance);

    // 8.5 Sentiment Override (nouveau - comme unified-insights-v2.js)
    // Le sentiment n'est PAS une composante du DI mais un OVERRIDE contextuel
    const sentimentScore = uiOverrides.sentimentScore ?? baseContext.sentimentScore ?? 50;
    const cycleScore = baseContext.scores?.cycle ?? 50;
    const regime = cycleScore >= 70 ? 'bull' : (cycleScore <= 30 ? 'bear' : 'neutral');

    const sentimentResult = applySentimentOverride(cappedResult.targets, sentimentScore, regime, cycleScore);
    if (sentimentResult.overrideApplied) {
      cappedResult.targets = sentimentResult.targets;
      cappedResult.sentimentOverride = {
        applied: true,
        reason: sentimentResult.overrideReason,
        sentiment: sentimentScore,
        regime
      };
      console.debug('🎭 SIM: Sentiment override applied:', sentimentResult.overrideReason);
    }

    const effectiveCap01 = (typeof policyCap01 === 'number' && Number.isFinite(policyCap01) && policyCap01 > 0)
      ? policyCap01
      : null;

    const executionPolicy = {
      ...executionOverrides,
      global_delta_threshold_pct: executionOverrides.global_delta_threshold_pct ?? 2,
      bucket_delta_threshold_pct: executionOverrides.bucket_delta_threshold_pct ?? 1,
      min_lot_eur: executionOverrides.min_lot_eur ?? 10,
      slippage_bps: executionOverrides.slippage_bps ?? 20,
      cap01: effectiveCap01
    };

    // 9. Plan d'exécution - position réelle calculée depuis wallet
    const currentAllocation = await computeCurrentAllocation(baseContext.wallet);
    const orders = planOrdersSimulated(currentAllocation, cappedResult.targets, executionPolicy);

    // 10. Explication
    const explanation = explainPipeline(baseContext, {
      di, riskBudget, targets, finalTargets, cappedResult, orders
    });

    const capPercentFromState = selectCapPercent(stateForEngine);
    const capPercentForUi = effectiveCap01 != null
      ? Math.round(effectiveCap01 * 100)
      : capPercentFromState;
    const capPct01ForUi = effectiveCap01 != null
      ? effectiveCap01
      : (capPercentForUi != null ? capPercentForUi / 100 : undefined);

    const fullResult = {
      context: baseContext,
      di,
      riskBudget,
      targets,
      finalTargets,
      cappedTargets: cappedResult.targets,
      capsTriggered: cappedResult.capsTriggered,
      currentAllocation,
      orders,
      explanation,
      timestamp: new Date().toISOString(),
      presetInfo,
      // Infos UI pour badges
      ui: {
        stateForEngine,
        contradictionPct: Math.round((stateForEngine.governance.contradiction_index ?? 0) * 100),
        capPercent: capPercentForUi ?? null,
        capPct01: capPct01ForUi,
        stale: eff.stale === true,
        mode: eff.useBaseWeights ? 'FROZEN' : 'ACTIVE'
      }
    };

    (window.debugLogger?.debug || console.log)('🎭 SIM: Full pipeline completed successfully');
    return fullResult;

  } catch (error) {
    debugLogger.error('🎭 SIM: Pipeline simulation failed:', error);
    throw error;
  }
}

// Export de l'état pour debug
export function getSimulationState() {
  return { ...simulationState };
}


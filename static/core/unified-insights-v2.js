// Unified Insights V2 - Migration vers Strategy API (PR-C)
// Nouvelle version qui utilise l'API Strategy tout en gardant la compatibilité
// Remplace progressivement unified-insights.js
(window.debugLogger?.debug || console.debug)('🔄 UNIFIED-INSIGHTS-V2.JS LOADED - FORCE CACHE RELOAD TIMESTAMP:', new Date().toISOString());

import { store } from './risk-dashboard-store.js';
import { getRegimeDisplayData, getMarketRegime } from '../modules/market-regimes.js';
import { estimateCyclePosition } from '../modules/cycle-navigator.js';
import { interpretCCS } from '../modules/signals-engine.js';
import { analyzeContradictorySignals } from '../modules/composite-score-v2.js';
import { calculateIntelligentDecisionIndexAPI, StrategyConfig } from './strategy-api-adapter.js';
import { calculateAdaptiveWeights as calculateAdaptiveWeightsV2 } from '../governance/contradiction-policy.js';

// Lightweight helpers (conservés pour compatibilité)
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const pct = (x) => Math.round(clamp01(x) * 100);
// Risk Score semantics: 0-100, higher = more robust → GREEN
const colorForScore = (s) => s > 70 ? 'var(--success)' : s >= 40 ? 'var(--warning)' : 'var(--danger)';

/**
 * 🆕 STRUCTURE MODULATION V2 (Oct 2025)
 * Moduler la cible de stables et le cap selon la qualité structurelle du portfolio
 *
 * @param {number} structureScore - Portfolio Structure Score V2 (0-100)
 * @returns {object} { deltaStables, deltaCap }
 *
 * Règles:
 * - Structure faible (< 50) → +10 pts stables, cap -0.5
 * - Structure moyenne (50-60) → +5 pts stables, cap 0
 * - Structure forte (≥ 80) → -5 pts stables, cap +0.5
 * - Sinon (60-80) → Neutre (0, 0)
 */
export function computeStructureModulation(structureScore) {
  if (structureScore == null || Number.isNaN(structureScore)) {
    return { deltaStables: 0, deltaCap: 0 };
  }

  // ✅ DÉSACTIVÉ (Jan 2026): Risk Budget est la source canonique INTOUCHABLE pour stables
  // La Structure Modulation ne doit PAS modifier le % stables calculé par Risk Budget
  // On conserve seulement deltaCap pour ajuster le plafond d'exposition risque

  let deltaCap = 0;

  if (structureScore < 50) {
    deltaCap = -0.5; // Portfolio fragile → réduire cap
  } else if (structureScore >= 80) {
    deltaCap = +0.5; // Portfolio robuste → augmenter cap
  }

  // deltaStables TOUJOURS 0 - Risk Budget est intouchable
  return { deltaStables: 0, deltaCap: deltaCap };
}

// ============================================================================
// CRITICAL FIX (Feb 2026): Fallback supprimé pour éviter split-brain
// Audit Gemini + Claude: Les poids JS (0.35/0.25/0.2/0.2) étaient différents
// du backend (0.2/0.3/0.4/0.1), causant des scores incohérents.
// Maintenant: si l'API échoue → erreur explicite, pas de score fallback.
// ============================================================================
function simpleFallbackCalculation(_context) {
  // SUPPRIMÉ: Ne plus calculer de score avec des poids différents du backend
  // Retourner un état d'erreur explicite pour que l'UI affiche un message clair
  console.error('❌ SPLIT-BRAIN PREVENTION: Strategy API failed, no fallback calculation');
  return {
    score: null,  // null = pas de score disponible
    confidence: null,
    action: 'API_ERROR',
    source: 'api_failed_no_fallback',
    error: 'Strategy API unavailable - decision score cannot be calculated without backend'
  };
}

/**
 * Calcule les pondérations adaptatives selon le contexte de marché
 * Cycle ≥ 90 → augmente wCycle, plafonne pénalité On-Chain
 *
 * ⚠️ IMPORTANT — Sémantique Risk:
 * Risk est un score POSITIF (0..100, plus haut = mieux).
 * Ne jamais inverser (pas de 100 - risk).
 * Contributions UI: (w * score) / Σ(w * score).
 */
function calculateAdaptiveWeights(cycleData, onchainScore, contradictions, governanceContradiction = 0) {
  const cycleScore = cycleData?.score;
  if (!Number.isFinite(cycleScore) || !Number.isFinite(onchainScore)) {
    throw new Error('Cycle and on-chain scores are required for adaptive weights');
  }
  // Utiliser governance.contradiction_index comme source primaire, fallback sur on-chain
  const contradictionLevel = governanceContradiction > 0 ?
    Math.round(governanceContradiction * 100) :
    (contradictions?.length ?? 0);

  // ============================================================================
  // CRITICAL FIX (Feb 2026): Harmonisation poids frontend/backend
  // Source de vérité: services/execution/strategy_registry.py template "balanced"
  // Backend: cycle=0.3, onchain=0.35, risk=0.25 (sentiment ignoré côté frontend)
  // Renormalisé sur 0.9 total: cycle≈0.33, onchain≈0.39, risk≈0.28
  // ============================================================================
  let wCycle = 0.33;
  let wOnchain = 0.39;
  let wRisk = 0.28;

  // RÈGLE 1: Cycle ≥ 90 → boost modéré wCycle (calibration Feb 2026)
  // Boosters réduits pour éviter amplification de scores potentiellement contaminés
  if (cycleScore >= 90) {
    wCycle = 0.45; // Boost modéré (était 0.65)
    wOnchain = 0.35; // Préserve on-chain (était 0.25)
    wRisk = 0.20; // Maintient risque significatif (était 0.1)
    console.debug('🚀 Adaptive weights: Cycle ≥ 90 → moderate cycle boost (calibrated)');
  } else if (cycleScore >= 70) {
    wCycle = 0.40;
    wOnchain = 0.37;
    wRisk = 0.23;
  }

  // RÈGLE 2: Plafond de pénalité On-Chain pour préserver floors Alts
  const onchainPenaltyFloor = cycleScore >= 90 ? 0.3 : 0.0; // Pas moins de 30% si cycle fort
  const adjustedOnchainScore = Math.max(onchainPenaltyFloor * 100, onchainScore);

  // RÈGLE 3: Contradiction → affecte vitesse (cap), pas objectif
  let speedMultiplier = 1.0;
  if (contradictionLevel >= 3) {
    speedMultiplier = 0.6; // Ralentit exécution
  } else if (contradictionLevel >= 2) {
    speedMultiplier = 0.8;
  }

  const result = {
    wCycle,
    wOnchain,
    wRisk,
    onchainFloor: onchainPenaltyFloor,
    adjustedOnchainScore,
    speedMultiplier,
    reasoning: {
      cycleBoost: cycleScore >= 90,
      onchainFloorApplied: adjustedOnchainScore > onchainScore,
      contradictionSlowdown: speedMultiplier < 1.0
    }
  };

  console.debug('⚖️ Adaptive weights calculated:', result);
  return result;
}

/**
 * DYNAMIQUE - Calcule les cibles d'allocation macro selon le contexte réel
 * Remplace les presets hardcodés par un calcul adaptatif
 * @param {object} ctx - Contexte (cycle, regime, sentiment, governance)
 * @param {object} rb - Risk budget avec target_stables_pct, min_stables, max_stables
 * @param {object} walletStats - Stats wallet (concentration, volatilité)
 * @param {object} data - Données complètes (pour accès risk_metrics.risk_version_info)
 * @returns {object} Targets par groupe, somme = 100%
 */
function computeMacroTargetsDynamic(ctx, rb, walletStats, data = null) {
  console.debug('🎯 computeMacroTargetsDynamic called:', { ctx, rb, walletStats, hasData: !!data });

  // 0) Stables = SOURCE DE VÉRITÉ (risk budget) avec Structure Modulation V2
  let stablesBase = rb?.target_stables_pct;
  if (typeof stablesBase !== 'number' || stablesBase < 0 || stablesBase > 100) {
    throw new Error('Verified stablecoin risk budget is unavailable');
  }

  // 🆕 Structure Modulation V2 (Oct 2025)
  const structureScore = data?.risk?.risk_metrics?.risk_version_info?.portfolio_structure_score;
  const { deltaStables, deltaCap } = computeStructureModulation(structureScore);

  // Respect the risk-budget target. Bounds are only applied when the budget
  // explicitly provides them; hidden defaults must not weaken a defensive target.
  const minStables = rb?.min_stables ?? 0;
  const maxStables = rb?.max_stables ?? 100;
  const stablesModulated = Math.max(minStables, Math.min(maxStables, stablesBase + deltaStables));

  // Métadonnées pour UI/logs
  ctx.structure_modulation = {
    structure_score: structureScore ?? null,
    delta_stables: deltaStables,
    delta_cap: deltaCap,
    stables_before: stablesBase,
    stables_after: stablesModulated,
    note: 'V2 portfolio structure modulation',
    enabled: structureScore != null
  };

  console.debug('🏗️ Structure Modulation V2:', ctx.structure_modulation);

  const stables = stablesModulated;
  const riskyPool = Math.max(0, 100 - stables); // Espace pour assets risqués

  // 1) Poids de base relatifs (hors stables) - Portfolio neutre
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

  // 2) Modulateurs avec HIÉRARCHIE DE PRIORITÉS (Option 1)
  //
  // ARCHITECTURE:
  // 1. Sentiments EXTRÊMES (Fear <25 ou Greed >75) → TOUJOURS actifs (override TOUT)
  // 2. Phase Engine → Tilts tactiques (ETH expansion, altseason, etc.)
  // 3. Modulateurs de base (bull/bear) → Désactivés si Phase Engine actif
  //
  // RATIONALE:
  // - Extreme Fear/Greed = situations critiques qui nécessitent action immédiate
  // - Phase Engine = optimisations tactiques normales
  // - Les deux peuvent coexister: Phase Engine gère macro, Sentiment gère extrêmes

  const phaseEngineActive = ctx?.flags?.phase_engine === 'apply';

  // NIVEAU 1: Sentiments extrêmes (TOUJOURS actifs)
  const mlSentiment = Number.isFinite(ctx?.sentiment_value) ? ctx.sentiment_value : null;
  const extremeFear = mlSentiment !== null && mlSentiment < 25;
  const extremeGreed = mlSentiment !== null && mlSentiment > 75;

  // NIVEAU 2 & 3: Modulateurs de base (désactivés si Phase Engine actif)
  const isBull = !phaseEngineActive && ((ctx?.regime === 'bull') || (ctx?.cycle_score >= 70));
  const isBear = !phaseEngineActive && ((ctx?.regime === 'bear') || (ctx?.cycle_score <= 30));
  const isHedge = !phaseEngineActive && (ctx?.governance_mode === 'Hedge');

  console.debug('🔍 Market conditions (Hierarchical):', {
    level1_extremes: { extremeFear, extremeGreed, mlSentiment },
    level2_phase_engine: { active: phaseEngineActive, mode: ctx?.flags?.phase_engine },
    level3_base_modulators: { isBull, isBear, isHedge, disabled: phaseEngineActive },
    context: { cycle_score: ctx?.cycle_score, regime: ctx?.regime }
  });

  // Variable pour logs d'override
  let overrideReason = null;

  // NIVEAU 1 - PRIORITÉ ABSOLUE: Sentiments Extrêmes (toujours exécutés en premier)
  if (extremeFear || extremeGreed) {
    // Déterminer le régime pour contextualiser
    const bullContext = (ctx?.regime === 'bull') || (ctx?.cycle_score >= 70);
    const bearContext = (ctx?.regime === 'bear') || (ctx?.cycle_score <= 30);

    if (extremeFear && bullContext) {
      // 🐂 Bull + Fear = OPPORTUNITÉ (contrarian buy)
      base.ETH *= 1.15;
      base.SOL *= 1.20;
      base['L2/Scaling'] *= 1.20;
      base.DeFi *= 1.10;
      base.Memecoins = Math.max(base.Memecoins * 1.5, 0.02);
      overrideReason = `🐂 Bull Market + Extreme Fear (${mlSentiment}) → Buying opportunity`;
      console.debug('💎 LEVEL 1 OVERRIDE: Opportunistic allocation (Bull + Fear)');
    }
    else if (extremeFear && bearContext) {
      // 🐻 Bear + Fear = DANGER (capitulation)
      base.Memecoins *= 0.3;
      base['Gaming/NFT'] *= 0.5;
      base.DeFi *= 0.7;
      base['AI/Data'] *= 0.8;
      overrideReason = `🐻 Bear Market + Extreme Fear (${mlSentiment}) → Protection`;
      console.debug('🛡️ LEVEL 1 OVERRIDE: Defensive allocation (Bear + Fear)');
    }
    else if (extremeFear) {
      // 😐 Neutral + Fear = Prudence légère
      base.Memecoins *= 0.7;
      base['Gaming/NFT'] *= 0.8;
      overrideReason = `😐 Neutral + Fear (${mlSentiment}) → Prudence`;
      console.debug('⚖️ LEVEL 1 OVERRIDE: Cautious allocation (Neutral + Fear)');
    }

    if (extremeGreed) {
      // ⚠️ Extreme Greed = TOUJOURS prise de profits
      base.Memecoins *= 0.3;
      base['Gaming/NFT'] *= 0.5;
      base['AI/Data'] *= 0.7;
      base.DeFi *= 0.8;
      overrideReason = overrideReason
        ? `${overrideReason} + Extreme Greed (${mlSentiment}) → Prise de profits`
        : `⚠️ Extreme Greed (${mlSentiment}) → Prise de profits`;
      console.debug('⚠️ LEVEL 1 OVERRIDE: Profit-taking (Extreme Greed)');
    }
  }
  // NIVEAU 3: Modulateurs de base (seulement si Phase Engine inactif ET pas d'extrême)
  else if (!phaseEngineActive) {
    if (isBull) {
      // Mode bull: moins BTC, plus ETH/L2/SOL
      base.BTC *= 0.95;
      base.ETH *= 1.08;
      base['L2/Scaling'] *= 1.15;
      base.SOL *= 1.10;
      console.debug('🚀 LEVEL 3: Bull mode (standard boost ETH/L2/SOL)');
    }

    if (isBear || isHedge) {
      // Mode prudent: réduire long tail
      base.Memecoins *= 0.5;
      base['Gaming/NFT'] *= 0.7;
      base.DeFi *= 0.85;
      console.debug('🛡️ LEVEL 3: Standard defensive mode');
    }
  } else {
    console.debug('🎯 LEVEL 2: Phase Engine active, delegating tilts to Phase Engine');
  }

  // Stocker reason dans ctx pour UI
  if (overrideReason) {
    ctx.allocation_override_reason = overrideReason;
  }

  // 3) Diversification basée sur concentration wallet
  if (walletStats?.topWeightSymbol === 'BTC' && walletStats?.topWeightPct > 35) {
    base.BTC *= 0.92;
    base.ETH *= 1.06;
    base['L2/Scaling'] *= 1.06;
    console.debug('⚖️ BTC over-concentration: rebalance to ETH/L2');
  }

  // 4) Normaliser la somme (hors stables)
  const sumBase = Object.values(base).reduce((s, v) => s + v, 0) || 1;
  for (const k in base) {
    base[k] = base[k] / sumBase;
  }

  // 5) Convertir en points (%) sur le riskyPool
  const targets = { Stablecoins: +stables.toFixed(1) };
  for (const [k, v] of Object.entries(base)) {
    targets[k] = +(v * riskyPool).toFixed(1);
  }

  // 6) Ajustement somme=100 (gestion arrondis)
  const sum = Object.values(targets).reduce((a, b) => a + b, 0);
  const diff = +(100 - sum).toFixed(1);
  if (Math.abs(diff) >= 0.1) {
    const heavy = 'BTC'; // Ajuster sur BTC
    targets[heavy] = +(targets[heavy] + diff).toFixed(1);
    console.debug('🔧 Sum adjustment applied:', { diff, heavy });
  }

  // Debug logs removed (obsolete, redundant with comprehensive log below)
  (window.debugLogger?.debug || console.log)('🎯 Dynamic targets computed:', targets);
  console.debug('📊 Target breakdown: stables=' + stables + '%, risky=' + riskyPool + '%');

  return targets;
}

/**
 * Version améliorée de getUnifiedState qui utilise l'API Strategy
 * Garde la même interface pour la compatibilité ascendante
 */
export async function getUnifiedState() {
  console.debug('🔄 getUnifiedState called - starting unified state construction');
  const state = store.snapshot();

  // Extract base scores (identique à la version legacy)
  const onchainScore = state.scores?.onchain ?? null;
  const riskScore = state.scores?.risk ?? null;
  const blendedScore = state.scores?.blended ?? null;
  const activeCapFraction = state.governance?.active_policy?.cap_daily;
  const movementCapPct = Number.isFinite(activeCapFraction) && activeCapFraction > 0
    ? activeCapFraction * 100
    : null;
  const ocMeta = state.scores?.onchain_metadata || {};
  const risk = state.risk?.risk_metrics || {};

  // Extract categories for contradictions analysis (moved up to avoid initialization error)
  const ocCategories = ocMeta.categoryBreakdown || {};

  // CONTRADICTIONS ANALYSIS (moved up to avoid initialization error)
  let contradictions = [];
  try {
    contradictions = analyzeContradictorySignals(ocCategories).slice(0, 2);
    console.debug('✅ Contradictions Intelligence loaded:', contradictions.length);
  } catch (error) {
    contradictions = (state.scores?.contradictory_signals || []).slice(0, 2);
    (window.debugLogger?.warn || console.warn)('⚠️ Contradictions fallback:', error);
  }

  console.debug('🧠 UNIFIED STATE V2 - Using Strategy API + sophisticated modules');

  // 1. CYCLE INTELLIGENCE (conservé identique)
  let cycleData;
  try {
    cycleData = estimateCyclePosition();
    console.debug('✅ Cycle Intelligence loaded:', cycleData.phase?.phase, cycleData.score);
  } catch (error) {
    (window.debugLogger?.warn || console.warn)('⚠️ Cycle Intelligence unavailable:', error);
    cycleData = {
      available: false,
      months: null,
      score: null,
      phase: { phase: 'unknown' },
      confidence: null,
      multipliers: {}
    };
  }

  // 2. REGIME INTELLIGENCE avec scores arrondis pour stabilité
  let regimeData;
  try {
    if (blendedScore != null) {
      // ARRONDIR les scores pour éviter micro-variations (68.3 vs 68.7 → même regime)
      const blendedRounded = Math.round(blendedScore);
      const onchainRounded = onchainScore != null ? Math.round(onchainScore) : null;
      const riskRounded = riskScore != null ? Math.round(riskScore) : null;

      regimeData = getRegimeDisplayData(
        blendedRounded, onchainRounded, riskRounded,
        cycleData.score ?? null,
        cycleData.direction ?? null,
        cycleData.confidence ?? null
      );
      console.debug('✅ Regime Intelligence loaded:', {
        regimeName: regimeData.regime?.name,
        recommendationsCount: regimeData.recommendations?.length,
        hasRiskBudget: !!regimeData.risk_budget,
        riskBudgetKeys: regimeData.risk_budget ? Object.keys(regimeData.risk_budget) : null,
        stablesAllocation: regimeData.risk_budget?.stables_allocation,
        targetStablesPct: regimeData.risk_budget?.target_stables_pct,
        scoresUsed: { blended: blendedRounded, onchain: onchainRounded, risk: riskRounded }
      });
    } else {
      regimeData = { regime: { name: 'Unknown', emoji: '❓' }, recommendations: [], risk_budget: null };
    }
  } catch (error) {
    (window.debugLogger?.warn || console.warn)('⚠️ Regime Intelligence fallback:', error);
    regimeData = { regime: { name: 'Unknown', emoji: '❓' }, recommendations: [], risk_budget: null };
  }

  // 3. SIGNALS INTELLIGENCE (conservé identique pour compatibilité)
  let signalsData;
  let sentimentData = null;
  
  try {
    const globalConfig = window.globalConfig;
    if (globalConfig?.apiRequest) {
      const sentimentResult = await globalConfig.apiRequest('/api/ml/sentiment/symbol/BTC', {
        params: { days: 1 }
      });
        if (sentimentResult?.available === true && sentimentResult.aggregated_sentiment) {
          const fearGreedSource = sentimentResult.aggregated_sentiment.source_breakdown?.fear_greed;
          if (Number.isFinite(fearGreedSource?.average_sentiment)) {
            const fearGreedValue = Math.max(0, Math.min(100, Math.round(50 + (fearGreedSource.average_sentiment * 50))));
            sentimentData = {
              value: fearGreedValue,
              sources: sentimentResult.sources_used || [],
              interpretation: fearGreedValue < 25 ? 'extreme_fear' : fearGreedValue < 45 ? 'fear' : fearGreedValue < 55 ? 'neutral' : fearGreedValue < 75 ? 'greed' : 'extreme_greed'
            };
            console.debug('✅ Multi-source sentiment loaded:', sentimentData.sources, fearGreedValue);
          }
        }
    }
  } catch (e) {
    (window.debugLogger?.warn || console.warn)('⚠️ Multi-source sentiment fallback to store data');
  }
  
  try {
    if (!Number.isFinite(blendedScore)) {
      signalsData = {
        available: false,
        interpretation: 'unknown',
        confidence: null,
        signals_strength: 'unavailable',
        reason: 'Blended score unavailable'
      };
    } else {
      const ccsInterpretation = interpretCCS(blendedScore);
      signalsData = {
        available: true,
        interpretation: ccsInterpretation?.label?.toLowerCase?.() || 'unknown',
        confidence: null,
        signals_strength: 'unrated',
        ccs_level: ccsInterpretation?.level,
        ccs_color: ccsInterpretation?.color
      };
    }

    if (sentimentData && ['extreme_fear', 'extreme_greed'].includes(sentimentData.interpretation)) {
      signalsData.interpretation = sentimentData.interpretation;
      signalsData.available = true;
      signalsData.signals_strength = 'strong';
    }
    
    console.debug('✅ Signals Intelligence loaded:', signalsData.interpretation, signalsData.confidence);
  } catch (error) {
    (window.debugLogger?.warn || console.warn)('⚠️ Signals Intelligence fallback:', error);
    signalsData = { available: false, interpretation: 'unknown', confidence: null, signals_strength: 'unavailable', reason: error.message };
  }

  // 5. SOPHISTICATED ANALYSIS (conservé identique) - using ocCategories already declared above
  const drivers = Object.entries(ocCategories)
    .filter(([, data]) => Number.isFinite(data?.score))
    .map(([key, data]) => ({ 
      key, 
      score: data.score,
      desc: data?.description, 
      contributors: data?.contributorsCount ?? 0,
      consensus: data?.consensus
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);

  // INTELLIGENT CONTRADICTIONS ANALYSIS (already calculated above)

  // 4. NOUVELLE LOGIQUE - DECISION INDEX VIA STRATEGY API (moved after contradictions)
  let decision;
  try {
    // BLENDING ADAPTATIF - Pondérations contextuelles avec governance unifiée
    const governanceContradiction = Number.isFinite(state.governance?.contradiction_index)
      ? state.governance.contradiction_index
      : null;
    const adaptiveWeights = calculateAdaptiveWeights(cycleData, onchainScore, contradictions, governanceContradiction);

    // Préparer le contexte pour l'API Strategy
    const context = {
      blendedScore,
      cycleData,
      regimeData,
      signalsData,
      onchainScore,
      onchainConfidence: ocMeta?.confidence ?? null,
      riskScore,
      contradiction: contradictions?.length > 0
        ? Math.min(contradictions.length * 0.15, 0.48)
        : governanceContradiction,
      execution: { cap_pct_per_iter: movementCapPct },
      adaptiveWeights // Nouveau - utilisé par strategy-api-adapter
    };

    // Utiliser l'adaptateur Strategy API
    decision = await calculateIntelligentDecisionIndexAPI(context);

    console.debug('🚀 Strategy API decision:', {
      score: decision.score,
      confidence: decision.confidence,
      source: decision.source,
      template: decision.template_used
    });

    // Comparaison désactivée (legacy archivé)

  } catch (error) {
    (window.debugLogger?.warn || console.warn)('⚠️ Strategy API unavailable:', error.message);

    // Preserve the unavailable contract through the compatibility wrapper.
    const context = {
      blendedScore, cycleData, regimeData, signalsData,
      onchainScore, onchainConfidence: ocMeta?.confidence ?? null, riskScore
    };
    decision = simpleFallbackCalculation(context);
  }

  // ENHANCED HEALTH (conservé + ajout info Strategy API)
  const health = {
    backend: state.ui?.apiStatus?.backend || 'unknown',
    signals: state.ui?.apiStatus?.signals || 'unknown',
    lastUpdate: state.ccs?.lastUpdate || null,
    intelligence_modules: {
      cycle: (Number.isFinite(cycleData.confidence) && cycleData.confidence > 0.5)
        || (Number.isFinite(cycleData.score) && cycleData.score > 85) ? 'active' : 'limited',
      regime: regimeData.recommendations?.length > 0 ? 'active' : 'limited',
      signals: signalsData.confidence > 0.6 ? 'active' : 'limited',
      strategy_api: decision.available === true ? 'active' : 'unavailable'
    }
  };

  // Adjust decision confidence (conservé)
  try {
    const contraPenalty = Math.min((contradictions?.length || 0) * 0.05, 0.15);
    if (typeof decision.confidence === 'number') {
      decision.confidence = Math.max(0, Math.min(0.95, decision.confidence - contraPenalty));
    }
  } catch {}

  // ASSERTIONS V2 - Invariants critiques
  if (typeof window !== 'undefined' && window.location?.hostname === 'localhost') {
    // Vérifier l'invariant risky + stables = 100%
    const riskyPct = regimeData?.risk_budget?.percentages?.risky ?? 0;
    const stablesPct = regimeData?.risk_budget?.percentages?.stables ?? 0;
    const sum = riskyPct + stablesPct;

    // Seulement vérifier l'assertion si on a des données valides
    if (riskyPct > 0 && stablesPct > 0) {
      console.assert(
        Math.abs(sum - 100) <= 0.1,
        'Invariant failed: risky+stables must equal 100%',
        { risky: riskyPct, stables: stablesPct, sum, regimeData: regimeData?.risk_budget }
      );
    } else {
      console.debug('⚠️ Skipping risky+stables assertion: missing valid percentages',
        { risky: riskyPct, stables: stablesPct });
    }

    // DEBUG - Analyser regimeData avant assertion
    console.debug('🔍 REGIME DATA DEBUG DETAILLE:', {
      hasRegimeData: !!regimeData,
      hasRiskBudget: !!regimeData?.risk_budget,
      riskBudgetKeys: regimeData?.risk_budget ? Object.keys(regimeData.risk_budget) : null,
      targetStablesPct: regimeData?.risk_budget?.target_stables_pct,
      percentages: regimeData?.risk_budget?.percentages,
      riskBudgetFull: regimeData?.risk_budget,
      // Vérifier si les champs raw existent
      rawRiskyAllocation: regimeData?.risk_budget?.risky_allocation,
      rawStablesAllocation: regimeData?.risk_budget?.stables_allocation,
      generatedAt: regimeData?.risk_budget?.generated_at
    });

    console.debug('✅ V2 invariants validated:', {
      sum: `${sum}%`,
      target_stables: regimeData?.risk_budget?.target_stables_pct,
      timestamp: regimeData?.risk_budget?.generated_at ? '✅' : '⚠️'
    });
  }

  // RETURN ENHANCED UNIFIED STATE
  const unifiedState = {
    decision,
    cycle: {
      months: cycleData.months,
      score: Number.isFinite(cycleData.score) ? Math.round(cycleData.score) : null,
      weight: state.cycle?.weight ?? 0.3,
      phase: cycleData.phase,
      confidence: cycleData.confidence,
      multipliers: cycleData.multipliers
    },
    onchain: {
      score: onchainScore != null ? Math.round(onchainScore) : null,
      confidence: Number.isFinite(ocMeta.confidence)
        ? Math.round(ocMeta.confidence * 100) / 100
        : null,
      drivers,
      criticalCount: ocMeta.criticalZoneCount || 0,
    },
    risk: {
      score: riskScore != null ? Math.round(riskScore) : null,
      sharpe: risk?.sharpe_ratio ?? null,
      volatility: risk?.volatility_annualized ?? risk?.volatility ?? null,
      var95_1d: risk?.var_95_1d ?? risk?.var95_1d ?? null,
      budget: regimeData.risk_budget
    },
    // Scores consolidés pour Decision Index Panel
    scores: {
      cycle: Number.isFinite(cycleData.score) ? Math.round(cycleData.score) : null,
      onchain: onchainScore != null ? Math.round(onchainScore) : null,
      risk: riskScore != null ? Math.round(riskScore) : null,
      blended: blendedScore != null ? Math.round(blendedScore) : null
    },

    // NOUVEAUX EXPOSÉS - Budget vs Exécution (Hard-switch V2)
    risk_budget: {
      // % entiers (0–100) - SOURCE UNIQUE depuis market-regimes.js
      target_stables_pct: regimeData.risk_budget?.target_stables_pct ??
                          regimeData.risk_budget?.percentages?.stables ??
                          (regimeData.risk_budget?.stables_allocation != null
                            ? Math.round(regimeData.risk_budget.stables_allocation * 100)
                            : null),
      risky_target_pct: regimeData.risk_budget?.percentages?.risky ??
                        (regimeData.risk_budget?.risky_allocation != null
                          ? Math.round(regimeData.risk_budget.risky_allocation * 100)
                          : null),
      methodology: regimeData.risk_budget?.methodology ?? null,
      confidence: regimeData.risk_budget?.confidence ?? null,
      percentages: regimeData.risk_budget?.percentages || null,
      // Timestamp fiable depuis market-regimes
      generated_at: regimeData.risk_budget?.generated_at ?? regimeData.timestamp ?? null
    },

    // SOURCE CANONIQUE UNIQUE - Cibles dynamiques calculées selon contexte réel
    targets_by_group: await (async () => {
      // Construire le contexte pour calcul dynamique
      const ctx = {
        regime: regimeData.regime?.name?.toLowerCase(),
        cycle_score: cycleData.score,
        cycle_direction: cycleData.direction ?? null,
        cycle_confidence: cycleData.confidence ?? null,
        governance_mode: decision.governance_mode ?? state.governance?.active_policy?.mode ?? null,
        sentiment: sentimentData?.interpretation,
        sentiment_value: sentimentData?.value ?? null,
        // NOUVEAU: Feature flags pour phase engine
        flags: {
          phase_engine: typeof window !== 'undefined' ?
            localStorage.getItem('PHASE_ENGINE_ENABLED') || 'off' : 'off'
        }
      };

      // Risk budget (SOURCE DE VÉRITÉ pour stables)
      const rb = regimeData.risk_budget;
      if (!Number.isFinite(rb?.target_stables_pct)) {
        return {};
      }

      // Stats wallet basiques (calculés depuis allocations réelles)
      const currentAllocations = window.store?.get('allocations.current') || {};
      const sortedByWeight = Object.entries(currentAllocations).sort((a, b) => b[1] - a[1]);

      const walletStats = {
        topWeightSymbol: sortedByWeight[0]?.[0] || null,
        topWeightPct: sortedByWeight[0]?.[1] || null,
        volatility: null // Requires historical data, defer to risk metrics
      };

      // CALCUL DYNAMIQUE: remplace les presets hardcodés
      // 🆕 Passer risk_metrics pour Structure Modulation V2
      const dataForModulation = { risk: { risk_metrics: risk } };
      let dynamicTargets = computeMacroTargetsDynamic(ctx, rb, walletStats, dataForModulation);

      // PHASE ENGINE INTEGRATION (shadow/apply modes)
      if (ctx.flags.phase_engine === 'shadow' || ctx.flags.phase_engine === 'apply') {
        console.debug('🧪 PhaseEngine: Flag detected:', ctx.flags.phase_engine);

        // Store config in global scope for debugging
        if (typeof window !== 'undefined') {
          window._phaseEngineConfig = {
            mode: ctx.flags.phase_engine,
            enabled: true,
            targets: { ...dynamicTargets },
            context: { DI: decision.score ?? null, breadth_alts: null }
          };
        }

        // CRITICAL FIX: Make Phase Engine awaitable instead of fire-and-forget
        const phaseEnginePromise = (async () => {
          try {
            console.debug('🔄 PhaseEngine: Starting dynamic import...');

            const [
              { extractPhaseInputs },
              { inferPhase, applyPhaseTilts, forcePhase, clearForcePhase }
            ] = await Promise.all([
              import('./phase-inputs-extractor.js'),
              import('./phase-engine.js')
            ]);

            console.debug('✅ PhaseEngine: Modules loaded successfully');

            // Expose debug controls globally after import
            if (typeof window !== 'undefined' && window.location?.hostname === 'localhost') {
              if (!window.debugPhaseEngine) {
                window.debugPhaseEngine = {};
              }
              window.debugPhaseEngine.forcePhase = forcePhase;
              window.debugPhaseEngine.clearForcePhase = clearForcePhase;
              window.debugPhaseEngine.getCurrentForce = () => {
                // Import fresh to get current state
                return import('./phase-engine.js').then(m => m.getCurrentForce());
              };
            }

            const phaseInputs = extractPhaseInputs(store);
            console.debug('📊 PhaseEngine: Inputs extracted:', {
              DI: phaseInputs.DI,
              btc_dom: phaseInputs.btc_dom,
              partial: phaseInputs.partial,
              missing: phaseInputs.missing
            });

            const phase = inferPhase(phaseInputs);
            console.debug('🔍 PhaseEngine: Phase detected:', phase);

            const phaseResult = await applyPhaseTilts(dynamicTargets, phase, {
              DI: phaseInputs.DI,
              breadth_alts: phaseInputs.breadth_alts
            });

            console.debug('🔍 PhaseEngine RESULT DEBUG:', {
              phaseResult,
              hasTargets: !!phaseResult?.targets,
              targetsKeys: phaseResult?.targets ? Object.keys(phaseResult.targets) : [],
              hasMetadata: !!phaseResult?.metadata,
              type: typeof phaseResult
            });

            console.debug('⚡ PhaseEngine: Tilts calculated:', {
              phase,
              tiltsApplied: phaseResult.metadata?.tiltsApplied ?? 'unknown',
              capsTriggered: phaseResult.metadata?.capsTriggered ?? 'unknown',
              hasMetadata: !!phaseResult.metadata
            });

            if (ctx.flags.phase_engine === 'shadow') {
              // Shadow mode: log detailed results
              (window.debugLogger?.debug || console.log)('🧪 PhaseEngine Shadow Mode:', {
                phase,
                inputsQuality: phaseInputs.partial ? 'partial' : 'complete',
                originalTargets: Object.keys(dynamicTargets).reduce((acc, k) => {
                  acc[k] = (dynamicTargets[k] || 0).toFixed(1) + '%';
                  return acc;
                }, {}),
                phaseTiltedTargets: Object.keys(phaseResult.targets || {}).reduce((acc, k) => {
                  acc[k] = ((phaseResult.targets || {})[k] || 0).toFixed(1) + '%';
                  return acc;
                }, {}),
                deltas: Object.keys(dynamicTargets).reduce((acc, k) => {
                  const original = dynamicTargets[k] || 0;
                  const tilted = ((phaseResult.targets || {})[k]) || 0;
                  const delta = tilted - original;
                  if (Math.abs(delta) > 0.1) {
                    acc[k] = (delta > 0 ? '+' : '') + delta.toFixed(2) + '%';
                  }
                  return acc;
                }, {}),
                metadata: phaseResult.metadata || {}
              });

              // Store shadow result for UI consumption
              if (typeof window !== 'undefined') {
                window._phaseEngineShadowResult = {
                  phase,
                  inputs: phaseInputs,
                  original: dynamicTargets,
                  tilted: phaseResult.targets || {},
                  metadata: phaseResult.metadata || {},
                  timestamp: new Date().toISOString()
                };
              }

            } else if (ctx.flags.phase_engine === 'apply') {
              // Apply mode: Actually use the phase-tilted targets
              if (phaseResult.targets) {
                dynamicTargets = phaseResult.targets;
              } else {
                (window.debugLogger?.warn || console.warn)('⚠️ PhaseEngine: No targets returned, keeping original');
              }

              // Calculate sums properly for logging
              const originalSum = Object.values(phaseResult.original || {}).reduce((a, b) => a + (Number(b) || 0), 0);
              const newSum = Object.values(dynamicTargets || {}).reduce((a, b) => a + (Number(b) || 0), 0);

              (window.debugLogger?.info || console.log)('✅ PhaseEngine Apply Mode - TARGETS MODIFIED:', {
                phase,
                tiltsApplied: phaseResult.metadata?.tiltsApplied ?? 'unknown',
                capsTriggered: phaseResult.metadata?.capsTriggered ?? 'unknown',
                stablesFloorHit: phaseResult.metadata?.stablesFloorHit ?? 'unknown',
                originalSum: originalSum.toFixed(1) + '%',
                newSum: newSum.toFixed(1) + '%',
                note: 'Phase tilts REALLY applied to targets'
              });

              // Store applied tilts for debugging AND sync storage for immediate access
              if (typeof window !== 'undefined') {
                window._phaseEngineAppliedResult = {
                  phase,
                  original: phaseResult.original || {},
                  modified: dynamicTargets,
                  metadata: phaseResult.metadata || {},
                  timestamp: new Date().toISOString()
                };

                // Store in sync cache for immediate reuse
                window._phaseEngineCurrentTargets = { ...dynamicTargets };
              }
            }

          } catch (error) {
            debugLogger.error('❌ PhaseEngine: Import/execution failed:', error);

            // Fallback notification
            if (typeof window !== 'undefined') {
              window._phaseEngineError = {
                error: error.message,
                timestamp: new Date().toISOString(),
                mode: ctx.flags.phase_engine
              };
            }
          }

          // Return the final targets after phase processing
          return dynamicTargets;
        })();

        // CRITICAL: Wait for Phase Engine to complete before continuing
        dynamicTargets = await phaseEnginePromise;
        console.debug('🔥 Phase Engine completed, final targets applied:', dynamicTargets);

      }

      // Sync cache no longer needed since Phase Engine is now awaitable

      (window.debugLogger?.debug || console.log)('🎯 DYNAMIC TARGETS' + (ctx.flags.phase_engine !== 'off' ? ' + PHASE ENGINE' : '') + ':', {
        old_method: 'preset_from_api',
        new_method: 'dynamic_computation' + (ctx.flags.phase_engine !== 'off' ? ' + phase_tilts' : ''),
        phase_engine_mode: ctx.flags.phase_engine,
        targets: dynamicTargets,
        stables_source: rb?.target_stables_pct
      });

      return dynamicTargets;
    })(),

    execution: {
      cap_pct_per_iter: movementCapPct,
      estimated_iters_to_target: decision.execution_plan?.estimated_iters ?? null, // From allocation engine V2
      current_iteration: null,
      convergence_strategy: decision.policy_hint
        ? (decision.policy_hint.toLowerCase() === 'slow' ? 'gradual' : 'standard')
        : null,
      // Plan d'exécution calculé depuis targets_by_group (même source que cartes)
      plan_iter1: decision.execution_plan || null
    },
    regime: {
      name: regimeData.regime?.name,
      key: regimeData.regime?.key,
      emoji: regimeData.regime?.emoji,
      confidence: regimeData.regime?.confidence,
      strategy: regimeData.regime?.strategy,
      recommendations: regimeData.recommendations || []
    },
    signals: {
      interpretation: signalsData.interpretation,
      confidence: signalsData.confidence,
      strength: signalsData.signals_strength,
      sentiment: sentimentData
    },
    contradictions,
    health,
    
    // NOUVELLES DONNÉES STRATEGY API
    strategy: {
      enabled: StrategyConfig.getConfig().enabled,
      template_used: decision.template_used || null,
      policy_hint: decision.policy_hint || null,
      targets: decision.targets || [],
      api_version: decision.api_version || null,
      generated_at: decision.generated_at || null
    },
    
    // Intelligence metadata (conservé + enrichi)
    intelligence: {
      cycleData,
      regimeData,
      regimeRecommendations: regimeData.recommendations,
      signalsData,
      sentimentData,
      version: 'v2',  // NOUVEAU
      migration_status: decision.available === true ? 'migrated' : 'unavailable',
      // Legacy allocation support - convert strategy targets to old format
      allocation: decision.targets?.length > 0 ?
        decision.targets.reduce((acc, target) => {
          acc[target.symbol] = target.weight * 100; // Convert to percentage
          return acc;
        }, {}) : null
    }
  };

  // GARDE-FOUS & COHÉRENCE (ajouté)
  const rb = unifiedState?.risk?.budget || {};
  const riskyPct = rb?.percentages?.risky ?? 0;
  const stablesPct = rb?.percentages?.stables ?? 0;
  const sum = riskyPct + stablesPct;

  // Seulement vérifier si on a des données valides
  if (riskyPct > 0 && stablesPct > 0) {
    console.assert(
      Math.abs(sum - 100) <= 0.1,
      'Invariant cassé: risky+stables doit faire 100', rb?.percentages
    );
  } else {
    console.debug('⚠️ Skipping second risky+stables assertion: missing budget data',
      { risky: riskyPct, stables: stablesPct, rb: rb?.percentages });
  }

  // Aligner stables du groupe sur le risk budget (si design l'exige)
  if (typeof rb?.target_stables_pct === 'number' && unifiedState.targets_by_group?.Stablecoins != null) {
    const stablesFinal = Math.round(unifiedState.targets_by_group.Stablecoins * 10) / 10;
    const stablesBudget = Math.round(rb.target_stables_pct * 10) / 10;
    const diff = Math.abs(stablesFinal - stablesBudget);

    // Log as warning if divergence > 15% (frontend vs backend can differ legitimately)
    if (diff >= 15) {
      (window.debugLogger?.warn || console.warn)('⚠️ Large stablecoins divergence between targets_by_group and risk_budget',
        { stablesFinal, stablesBudget, diff: `${diff.toFixed(1)}%` });
    } else if (diff >= 5) {
      console.debug('ℹ️ Moderate stablecoins divergence (expected):', { stablesFinal, stablesBudget, diff: `${diff.toFixed(1)}%` });
    }
  }

  // 🆕 Exposer Structure Modulation V2 (Oct 2025)
  // ctx.structure_modulation est défini dans computeMacroTargetsDynamic()
  // On le récupère depuis le contexte utilisé pour les targets
  const canComputeStructureModulation = decision.available === true
    && unifiedState.targets_by_group
    && Object.keys(unifiedState.targets_by_group).length > 0
    && regimeData?.risk_budget;
  const structureMod = canComputeStructureModulation ? await (async () => {
    try {
      // Reconstruire le contexte (même que pour targets_by_group)
      const ctx = {
        regime: regimeData.regime?.name?.toLowerCase(),
        cycle_score: cycleData.score,
        cycle_direction: cycleData.direction ?? null,
        cycle_confidence: cycleData.confidence ?? null,
        governance_mode: decision.governance_mode ?? state.governance?.active_policy?.mode ?? null,
        sentiment: sentimentData?.interpretation,
        flags: {
          phase_engine: typeof window !== 'undefined' ?
            localStorage.getItem('PHASE_ENGINE_ENABLED') || 'off' : 'off'
        }
      };

      const rb = regimeData.risk_budget;
      const dataForModulation = { risk: { risk_metrics: risk } };

      // Appeler computeMacroTargetsDynamic pour obtenir ctx.structure_modulation
      computeMacroTargetsDynamic(ctx, rb, {}, dataForModulation);

      return ctx.structure_modulation || null;
    } catch (error) {
      debugLogger.debug('Structure modulation unavailable:', error);
      return null;
    }
  })() : null;

  unifiedState.structure_modulation = structureMod;

  // Timestamp fiable
  unifiedState.lastUpdate = rb?.generated_at || unifiedState?.lastUpdate || null;

  return unifiedState;
}

/**
 * Génère un ID de snapshot basé sur les données stables (pas les timestamps auto-générés)
 */
function snapshotId(u) {
  return JSON.stringify({
    user: u.user?.id || localStorage.getItem('activeUser'),
    source: u.meta?.data_source,
    // Scores arrondis pour stabilité (pas de timestamps qui changent)
    blended: Number.isFinite(u.decision?.score) ? Math.round(u.decision.score) : null,
    onchain: Number.isFinite(u.scores?.onchain) ? Math.round(u.scores.onchain) : null,
    risk: Number.isFinite(u.scores?.risk) ? Math.round(u.scores.risk) : null,
    cycle: Number.isFinite(u.scores?.cycle) ? Math.round(u.scores.cycle) : null,
    // Governance stable
    contradiction: Number.isFinite(u.governance?.contradiction_index)
      ? Math.round(u.governance.contradiction_index * 100)
      : null,
    // Risk budget stable (arrondi)
    stables_alloc: Number.isFinite(u.risk?.budget?.stables_allocation)
      ? Math.round(u.risk.budget.stables_allocation * 100)
      : null,
    // Regime key (pas timestamp)
    regime_key: u.regime?.key,
    // Strategy template (pas generated_at)
    strategy_template: u.strategy?.template_used
  });
}

// Cache snapshot-based avec TTL 30s
let _recoCache = { snapshotId: null, recos: null, timestamp: 0 };

/**
 * Dérivation des recommandations avec cache snapshot-based et stabilité renforcée
 */
export function deriveRecommendations(u) {
  // Vérifier cache snapshot d'abord
  const currentSnapshotId = snapshotId(u);
  const now = Date.now();

  if (_recoCache.snapshotId === currentSnapshotId && now - _recoCache.timestamp < 30000) {
    console.debug('🎯 Recommendations from snapshot cache:', _recoCache.recos.length);
    return _recoCache.recos;
  }

  console.debug('🧠 DERIVING INTELLIGENT RECOMMENDATIONS V2 - Snapshot:', currentSnapshotId.substring(0, 120) + '...');
  console.debug('📊 Snapshot Key Factors:', {
    blended: Number.isFinite(u.decision?.score) ? Math.round(u.decision.score) : null,
    onchain: Number.isFinite(u.scores?.onchain) ? Math.round(u.scores.onchain) : null,
    risk: Number.isFinite(u.scores?.risk) ? Math.round(u.scores.risk) : null,
    cycle: Number.isFinite(u.scores?.cycle) ? Math.round(u.scores.cycle) : null,
    contradiction: Number.isFinite(u.governance?.contradiction_index)
      ? Math.round(u.governance.contradiction_index * 100)
      : null,
    stables_alloc: Number.isFinite(u.risk?.budget?.stables_allocation)
      ? Math.round(u.risk.budget.stables_allocation * 100)
      : null,
    regime_key: u.regime?.key
  });

  let recos = [];

  // 1. USE STRATEGY API TARGETS avec primary stable (tie-breaker)
  // ⚠️ FILTRE: Ne pas afficher si contredit fortement le Risk Budget (source de vérité)
  if (u.strategy?.targets?.length > 0) {
    // Tri stable: poids DESC puis symbol ASC
    const targets = [...u.strategy.targets].sort((a,b) =>
      (b.weight - a.weight) || (a.symbol||'').localeCompare(b.symbol||'')
    );

    let primaryTarget = targets[0];
    const prevPrimary = window.__prevPrimaryTarget;

    // Hysteresis: si écart < 0.5% avec 2e, garder l'ancien (éviter flip visuel)
    if (prevPrimary && targets[1] && Math.abs(primaryTarget.weight - targets[1].weight) < 0.005) {
      const prevStillTop = targets.find(t => t.symbol === prevPrimary.symbol);
      if (prevStillTop && prevStillTop.weight >= targets[0].weight - 0.005) {
        primaryTarget = prevStillTop;
      }
    }
    window.__prevPrimaryTarget = primaryTarget;

    // Détection stables élargie (EUR, USDT, USDC, DAI, Stablecoins groupe, etc.)
    const stablesPattern = /stablecoin|usdt|usdc|busd|dai|eur|usd\b|tusd|gusd|pax|husd/i;
    const isStablesTarget = stablesPattern.test(primaryTarget.symbol);
    const allocPct = Math.round(primaryTarget.weight * 100);

    // ✅ VALIDATION: Comparer avec Risk Budget si disponible (source canonique)
    const riskBudgetStables = u.risk?.budget?.percentages?.stables;
    const shouldSkipStrategyReco = isStablesTarget && riskBudgetStables != null && Math.abs(allocPct - riskBudgetStables) > 15;

    if (shouldSkipStrategyReco) {
      debugLogger.debug('[RECOMMENDATIONS] Strategy API stables reco skipped - conflicts with Risk Budget:', {
        strategyValue: allocPct,
        riskBudgetValue: riskBudgetStables,
        delta: Math.abs(allocPct - riskBudgetStables)
      });
    } else {
      recos.push({
        key: `reco:strategy:primary:${primaryTarget.symbol}`,  // Clé canonique stable
        topic: isStablesTarget ? 'stables_allocation' : undefined,
        value: isStablesTarget ? allocPct : undefined,
        priority: 'high',
        title: `Allocation ${primaryTarget.symbol}: ${allocPct}%`,
        reason: primaryTarget.rationale || `Suggestion ${u.strategy.template_used}`,
        icon: '🎯',
        source: 'strategy-api'
      });
    }
  }

  // 2. USE REGIME RECOMMENDATIONS avec clés canoniques
  if (u.intelligence?.regimeRecommendations?.length > 0) {
    u.intelligence.regimeRecommendations.forEach(rec => {
      // Générer clé stable basée sur type + message
      const regimeKey = rec.type || 'general';
      const msgHash = (rec.message || rec.action || '').toLowerCase().replace(/[^a-z0-9]/g, '_').substring(0, 30);

      // Extract stables percentage if mentioned (check BOTH message AND action)
      const messageText = rec.message || '';
      const actionText = rec.action || '';
      const combinedText = messageText + ' ' + actionText;
      const stablesMatch = combinedText.match(/(\d+)%/);
      const isStablesReco = /stables?/i.test(combinedText);

      // DEBUG: Log detection
      if (isStablesReco) {
        debugLogger.debug('[CONSOLIDATION DEBUG] Regime stables reco detected:', {
          message: rec.message,
          action: rec.action,
          stablesMatch: stablesMatch?.[1],
          isStablesReco
        });
      }

      recos.push({
        key: `reco:regime:${regimeKey}:${msgHash}`,
        topic: isStablesReco ? 'stables_allocation' : undefined,
        value: isStablesReco && stablesMatch ? parseInt(stablesMatch[1]) : undefined,
        priority: rec.priority || 'medium',
        title: rec.message || rec.title || rec.action,
        reason: rec.action || rec.message || 'Market regime recommendation',
        icon: rec.type === 'warning' ? '⚠️' : rec.type === 'alert' ? '🚨' : '💡',
        source: 'regime-intelligence'
      });
    });
  }

  // 3. CYCLE-BASED RECOMMENDATIONS avec clés canoniques
  // ✅ VALIDATION: Tenir compte du RÉGIME RÉEL (blended) pas seulement du cycle position
  if (u.cycle?.phase?.phase) {
    const cyclePhase = u.cycle.phase.phase;
    const blendedScore = u.scores?.blended;
    const regimeKey = u.regime?.key ||
                      u.market?.regime?.key ||
                      (Number.isFinite(blendedScore)
                        ? (blendedScore >= 76 ? 'expansion' :
                          blendedScore >= 51 ? 'bull_market' :
                          blendedScore >= 26 ? 'correction' : 'bear_market')
                        : null);
    const hasDecisionScore = Number.isFinite(u.decision?.score);

    // Take profits ONLY if cycle=peak AND (regime=bull_market OR expansion) AND DI>75
    if (cyclePhase === 'peak' && hasDecisionScore && u.decision.score > 75 && (regimeKey === 'bull_market' || regimeKey === 'expansion')) {
      recos.push({
        key: 'reco:cycle:peak_profits',
        priority: 'high',
        title: 'Take progressive profits',
        reason: `Cycle peak + Regime ${regimeKey} + High DI (${u.decision.score})`,
        icon: '📈',
        source: 'cycle-intelligence'
      });
    }
    // Vigilance if cycle=peak but regime is only correction (divergence)
    else if (cyclePhase === 'peak' && hasDecisionScore && u.decision.score > 75 && regimeKey === 'correction') {
      recos.push({
        key: 'reco:cycle:peak_but_correction',
        priority: 'medium',
        title: 'Increased vigilance recommended',
        reason: `Cycle at peak but market in ${regimeKey.replace('_', ' ')} (blended: ${blendedScore}) - possible divergence`,
        icon: '⚠️',
        source: 'cycle-intelligence'
      });
    }
    // Accumulation classique
    else if (cyclePhase === 'accumulation' && hasDecisionScore && u.decision.score < 40) {
      recos.push({
        key: 'reco:cycle:accumulation',
        priority: 'medium',
        title: 'Accumulate quality positions',
        reason: `Cycle accumulation + DI bas (${u.decision.score})`,
        icon: '🔵',
        source: 'cycle-intelligence'
      });
    }
  }

  // 4. STRATEGY API POLICY HINTS avec clés canoniques
  if (u.strategy?.policy_hint) {
    const policyHint = u.strategy.policy_hint;
    if (policyHint === 'Slow') {
      recos.push({
        key: 'reco:policy:slow',
        priority: 'medium',
        title: 'Cautious approach recommended',
        reason: 'Contradictory signals or low confidence detected',
        icon: '🐌',
        source: 'strategy-api-policy'
      });
    } else if (policyHint === 'Aggressive') {
      recos.push({
        key: 'reco:policy:aggressive',
        priority: 'high',
        title: 'Aggressive allocation opportunity',
        reason: 'High score and coherent signals',
        icon: '⚡',
        source: 'strategy-api-policy'
      });
    }
  }

  // 5. CONTRADICTION ALERTS avec hysteresis + clés canoniques
  const governanceContradiction = u.governance?.contradiction_index || 0;
  const onchainContradictions = u.contradictions?.length || 0;

  // Init flags hysteresis
  if (!window.__recoFlags) window.__recoFlags = {};
  const flags = window.__recoFlags;

  // Fonction flip pour Schmitt trigger
  const flip = (prev, val, up, down) => prev ? (val > down) : (val >= up);

  // Hysteresis sur contradiction governance (up=0.35, down=0.25)
  flags.contradiction_high = flip(flags.contradiction_high, governanceContradiction, 0.35, 0.25);

  if (flags.contradiction_high) {
    const isVeryHigh = governanceContradiction > 0.7;
    recos.push({
      key: isVeryHigh ? 'reco:gov:contradiction_very_high' : 'reco:gov:contradiction_high',
      priority: isVeryHigh ? 'high' : 'medium',
      title: `Signaux contradictoires: ${Math.round(governanceContradiction * 100)}%`,
      reason: isVeryHigh ?
        'Strong contradiction detected - cautious approach recommended' :
        'Moderate contradiction detected between sources',
      icon: isVeryHigh ? '🚨' : '⚡',
      source: 'governance-contradiction'
    });
  } else if (onchainContradictions > 0 && governanceContradiction < 0.25) {
    // Fallback vers contradictions on-chain seulement si governance très faible
    recos.push({
      key: 'reco:onchain:contradiction',
      priority: 'medium',
      title: 'Contradictory on-chain signals detected',
      reason: `${onchainContradictions} divergence(s) entre indicateurs`,
      icon: '⚡',
      source: 'onchain-contradiction'
    });
  }

  // 6. RISK BUDGET RECOMMENDATIONS avec hysteresis (up=0.45, down=0.37)
  const stablesAlloc = u.risk?.budget?.stables_allocation || 0;
  flags.stables_high = flip(flags.stables_high, stablesAlloc, 0.45, 0.37);

  if (flags.stables_high) {
    const targetStables = u.risk.budget.percentages?.stables || 0;
    // Generate tactical action based on risk profile
    let tacticalAction = '';
    if (targetStables >= 40) {
      tacticalAction = `Target: ${targetStables}% stables - Secure progressively`;
    } else if (targetStables >= 25) {
      tacticalAction = `Target: ${targetStables}% stables - Reduce risk exposure`;
    }

    recos.push({
      key: 'reco:risk:stables_high',
      topic: 'stables_allocation',
      value: targetStables,
      priority: 'medium',
      title: `High risk budget detected`,
      reason: `Recommended stables allocation: ${targetStables}%`,
      tacticalAction: tacticalAction,
      icon: '💡',
      source: 'risk-budget'
    });
  }

  // CONSOLIDATION DES RECOMMENDATIONS STABLES (même allocation = 1 seule carte)
  function consolidateStablesRecommendations(recos) {
    const stablesRecs = recos.filter(r => r.topic === 'stables_allocation');

    // DEBUG: Log all stables recommendations detected
    debugLogger.debug('[CONSOLIDATION DEBUG] Stables recommendations found:', {
      count: stablesRecs.length,
      recos: stablesRecs.map(r => ({
        key: r.key,
        topic: r.topic,
        value: r.value,
        title: r.title,
        reason: r.reason,
        source: r.source
      }))
    });

    if (stablesRecs.length <= 1) return recos; // Pas de duplication

    // ✅ VALIDATION: Vérifier si les valeurs sont cohérentes (±5%)
    const values = stablesRecs.map(r => r.value).filter(v => v != null);
    const avgValue = values.reduce((a, b) => a + b, 0) / values.length;
    const maxDelta = Math.max(...values.map(v => Math.abs(v - avgValue)));
    const areValuesConsistent = maxDelta <= 5;

    debugLogger.debug('[CONSOLIDATION DEBUG] Values consistency check:', {
      values,
      avgValue,
      maxDelta,
      areValuesConsistent
    });

    // Si incohérent (>5% delta), PRIORISER Risk Budget (source canonique)
    if (!areValuesConsistent) {
      const riskBudgetRec = stablesRecs.find(r => r.source === 'risk-budget');
      if (riskBudgetRec) {
        debugLogger.info('[CONSOLIDATION] Inconsistent stables recommendations - using Risk Budget as canonical source');
        // Garder seulement Risk Budget et Regime (si proche)
        const keptRecs = stablesRecs.filter(r => {
          if (r.source === 'risk-budget') return true;
          if (r.source === 'regime-intelligence' && Math.abs(r.value - riskBudgetRec.value) <= 3) return true;
          return false;
        });

        // Si 1 seule source reste, retourner tel quel
        if (keptRecs.length <= 1) {
          return [...recos.filter(r => r.topic !== 'stables_allocation'), ...keptRecs];
        }

        // Sinon merger les sources cohérentes
        stablesRecs.length = 0;
        stablesRecs.push(...keptRecs);
      } else {
        // Pas de Risk Budget, garder la valeur la plus conservatrice (la plus haute)
        const maxStables = Math.max(...values);
        const conservativeRec = stablesRecs.find(r => r.value === maxStables);
        debugLogger.info('[CONSOLIDATION] No Risk Budget found - using most conservative recommendation');
        return [...recos.filter(r => r.topic !== 'stables_allocation'), conservativeRec];
      }
    }

    // Consolidation normale si valeurs cohérentes
    const order = { critical: 0, high: 1, medium: 2, low: 3 };
    const value = Math.round(avgValue); // Utiliser moyenne si cohérent
    const sources = [...new Set(stablesRecs.map(r => r.source))];
    const topPriority = stablesRecs.reduce((p, r) =>
      order[p] <= order[r.priority] ? p : r.priority, 'medium'
    );

    // Mapper les sources à des labels lisibles
    const sourceLabels = {
      'strategy-api': 'Strategy',
      'regime-intelligence': 'Regime',
      'risk-budget': 'Risk'
    };

    const merged = {
      key: `reco:stables:consensus:${value}`,
      topic: 'stables_allocation',
      value: value,
      priority: topPriority,
      title: `Stables allocation: ${value}%`,
      subtitle: `Consensus confirmed by ${sources.length} sources`,
      reason: stablesRecs.map(r => {
        const sourceLabel = sourceLabels[r.source] || r.source;
        return `• ${sourceLabel}: ${r.reason || r.title}`;
      }).join('\n'),
      icon: '🎯',
      source: sources.join(' + '),
      consolidated: true,
      sourceCount: sources.length
    };

    debugLogger.debug('[CONSOLIDATION DEBUG] Merged recommendation:', merged);

    // Remplacer les N cartes par 1
    return [merged, ...recos.filter(r => r.topic !== 'stables_allocation')];
  }

  // Appliquer consolidation
  recos = consolidateStablesRecommendations(recos);

  // DÉDUPLICATION + TRI STABLE par clé canonique
  const prio = { critical: 0, high: 1, medium: 2, low: 3 };
  const uniqueRecos = Array.from(new Map(recos.map(r => [r.key, r])).values())
    .sort((a,b) =>
      (prio[a.priority] - prio[b.priority]) ||
      (a.source||'').localeCompare(b.source||'') ||
      (a.key||'').localeCompare(b.key||'')
    );

  // Sauvegarder dans cache snapshot
  _recoCache = {
    snapshotId: currentSnapshotId,
    recos: uniqueRecos,
    timestamp: now
  };

  console.debug('🎯 Recommendations derived:', uniqueRecos.length, 'unique from', [...new Set(uniqueRecos.map(r => r.source))].join(', '));
  console.debug('🔑 Snapshot ID:', currentSnapshotId.substring(0, 80) + '...');

  return uniqueRecos;
}

// Exports pour compatibilité
export { calculateIntelligentDecisionIndexAPI as calculateIntelligentDecisionIndex };
export { clamp01, pct, colorForScore };  // Utilitaires conservés

// Export de la fonction critique pour simulation-engine.js
export { computeMacroTargetsDynamic };

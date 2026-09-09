// Allocation Engine V2 - Descente hiérarchique avec Feature Flag
// Macro → Secteurs → Coins avec contraintes explicites uniquement

import { getAssetGroup, UNIFIED_ASSET_GROUPS, GROUP_ORDER, loadTaxonomyDataSync } from '../shared-asset-groups.js';
// ✅ MODIFIÉ (Phase 1.2): Utiliser selectEffectiveCap pour cohérence staleness/alert/policy
import { selectEffectiveCap } from '../selectors/governance.js';

/**
 * 🆕 STRUCTURE MODULATION V2 (Oct 2025)
 * Applique deltaCap depuis structure modulation au cap effectif de gouvernance
 *
 * @param {object} state - Store state (pour selectEffectiveCap)
 * @param {number} deltaCap - Ajustement de cap (±0.5 max)
 * @returns {number} Cap effectif ajusté (%)
 */
function getEffectiveCapWithStructure(state, deltaCap = 0) {
  const capEff = selectEffectiveCap(state); // Source gouvernance (staleness, alerts, policy)
  const adjusted = Math.max(0, capEff + (deltaCap || 0)); // Jamais négatif
  const maxDelta = 0.5; // Garde-fou: +0.5% max vis-à-vis de la gouvernance
  return Math.min(adjusted, capEff + maxDelta);
}

// Feature flag pour activation
const ALLOCATION_ENGINE_V2 = true; // Will be controlled by config later

// Configuration des floors contextuels
const FLOORS_CONFIG = {
  // Floors de base par secteur
  base: {
    'BTC': 0.15,      // 15% minimum
    'ETH': 0.12,      // 12% minimum
    'Stablecoins': 0.10, // 10% minimum sécurité
    'SOL': 0.03,      // 3% minimum
    'L1/L0 majors': 0.08, // 8% minimum pour diversification
    'L2/Scaling': 0.03,
    'DeFi': 0.04,
    'Memecoins': 0.02,
    'Gaming/NFT': 0.01,
    'AI/Data': 0.01,
    'Others': 0.01
  },

  // Floors renforcés quand Cycle ≥ 90 (bull market)
  bullish: {
    'SOL': 0.06,
    'L1/L0 majors': 0.12,
    'L2/Scaling': 0.06,
    'DeFi': 0.08,
    'Memecoins': 0.05, // Permis en bull
    'Gaming/NFT': 0.02,
    'AI/Data': 0.02
  },

  // Incumbency: positions détenues ne peuvent pas aller à 0%
  incumbency: 0.03 // 3% minimum pour assets détenus
};

/**
 * Moteur d'allocation hiérarchique V2
 * @param {Object} context - Contexte unifié (scores, adaptive weights, etc.)
 * @param {Array} currentPositions - Positions actuelles du portefeuille
 * @param {Object} options - Options (feature flags, constraints)
 */
export async function calculateHierarchicalAllocation(context, currentPositions = [], options = {}) {
  const enableV2 = options.enableV2 ?? ALLOCATION_ENGINE_V2;
  const enableFloors = options.enableFloors === true;
  const respectIncumbency = options.respectIncumbency === true;

  console.debug('🏗️ Allocation Engine called:', { enableV2, contextualScores: !!context.adaptiveWeights });

  if (!enableV2) {
    console.debug('⚠️ Allocation Engine V2 disabled, using fallback');
    return null; // Fallback vers V1
  }

  try {
    // Ensure taxonomy data is loaded before proceeding
    try {
      loadTaxonomyDataSync(); // Fonction synchrone, pas d'await
      console.debug('✅ Taxonomy data loaded for allocation engine');
    } catch (taxonomyError) {
      (window.debugLogger?.warn || console.warn)('⚠️ Taxonomy loading failed, continuing with fallback:', taxonomyError.message);
      // Continue quand même, getAssetGroup aura ses fallbacks
    }
    // 1. EXTRACTION DU CONTEXTE
    const {
      cycleScore,
      onchainScore,
      riskScore,
      adaptiveWeights = {},
      risk_budget = {},
      contradiction = 0,
      // ✅ NOUVEAU (Phase 1.3): Récupérer meme_cap depuis regime.allocation_bias
      regime = {},
      // 🆕 NOUVEAU (Oct 2025): Structure Modulation V2 pour deltaCap
      structure_modulation = {}
    } = context;

    if (![cycleScore, onchainScore, riskScore].every(Number.isFinite)) {
      throw new Error('Cycle, on-chain and Risk Score are required for allocation');
    }

    // Extraire meme_cap depuis le régime de marché
    const meme_cap = regime?.allocation_bias?.meme_cap ?? null;

    // 🆕 Extraire deltaCap depuis structure modulation
    const deltaCap = structure_modulation?.delta_cap ?? 0;

    // 2. DÉTECTION PHASE MARCHÉ
    const isBullishPhase = cycleScore >= 90;
    const isModeratePhase = cycleScore >= 70 && cycleScore < 90;
    const selectedFloors = enableFloors
      ? (isBullishPhase ? { ...FLOORS_CONFIG.base, ...FLOORS_CONFIG.bullish } : FLOORS_CONFIG.base)
      : {};

    console.debug('📊 Market phase detection:', { cycleScore, isBullishPhase, isModeratePhase });

    // 3. ALLOCATION NIVEAU 1 - MACRO
    const macroAllocation = calculateMacroAllocation(context, selectedFloors);
    console.debug('🌍 Macro allocation:', macroAllocation);

    // 4. ALLOCATION NIVEAU 2 - SECTEURS
    const sectorAllocation = calculateSectorAllocation(macroAllocation, selectedFloors, isBullishPhase);
    console.debug('🏭 Sector allocation:', sectorAllocation);

    // 5. ALLOCATION NIVEAU 3 - COINS. Incumbency is opt-in only.
    const positionsForAllocation = respectIncumbency ? currentPositions : [];
    const coinAllocation = calculateCoinAllocation(sectorAllocation, positionsForAllocation, selectedFloors, meme_cap);
    console.debug('🪙 Coin allocation:', coinAllocation);

    // 6. CALCUL ITERATIONS ESTIMÉES
    const executionPlan = calculateExecutionPlan(coinAllocation, currentPositions, context.execution);

    // 7. VALIDATION FINALE + CHECKSUM + CONTRÔLES
    const totalCheck = validateTotalAllocation(coinAllocation);
    const allocationEntries = Object.entries(coinAllocation);

    // CONTRÔLES HIÉRARCHIQUES
    const hierarchyCheck = validateHierarchy(coinAllocation, currentPositions);
    console.debug('🔍 Hierarchy validation:', hierarchyCheck);

    // GUARD: target_sum_mismatch
    const targetSum = Object.values(coinAllocation).reduce((sum, val) =>
      sum + (typeof val === 'number' && !isNaN(val) ? val : 0), 0
    );
    if (Math.abs(targetSum - 1.0) > 0.01) {
      (window.debugLogger?.warn || console.warn)(`⚠️ target_sum_mismatch: somme secteurs = ${(targetSum * 100).toFixed(1)}% (≠ 100%)`);
    }

    // CHECKSUM DÉTAILLÉ
    console.debug('💯 CHECKSUM:', {
      total_allocation: totalCheck.total,
      entries_count: allocationEntries.length,
      valid_entries: allocationEntries.filter(([k, v]) => v > 0.001).length,
      is_normalized: totalCheck.isValid,
      hierarchy_ok: hierarchyCheck.valid,
      target_sum_ok: Math.abs(targetSum - 1.0) <= 0.01,
      allocation_breakdown: Object.fromEntries(
        allocationEntries
          .filter(([k, v]) => v > 0.001)
          .map(([k, v]) => [k, `${(v * 100).toFixed(1)}%`])
      )
    });

    if (!totalCheck.isValid) {
      throw new Error(`Invalid allocation total: ${totalCheck.total}`);
    }

    // 8. LOGS POUR DEBUG
    logAllocationDecisions({
      context,
      macro: macroAllocation,
      sectors: sectorAllocation,
      coins: coinAllocation,
      execution: executionPlan
    });

    // Convert allocation object to array with proper rounding (sum = 100%)
    const entries = Object.entries(coinAllocation);
    const percentages = entries.map(([group, weight]) => ({
      group,
      exact: weight * 100,
      rounded: Math.floor(weight * 100)
    }));

    // Calculate remainder and distribute to largest remainders
    let totalRounded = percentages.reduce((sum, p) => sum + p.rounded, 0);
    const remainders = percentages.map(p => ({ ...p, remainder: p.exact - p.rounded }));
    remainders.sort((a, b) => b.remainder - a.remainder);

    // Distribute remaining points to largest remainders
    const pointsToDistribute = 100 - totalRounded;
    for (let i = 0; i < pointsToDistribute && i < remainders.length; i++) {
      remainders[i].rounded += 1;
    }

    const allocations = remainders.map(p => ({
      group: p.group,
      target_allocation: p.rounded
    }));

    const result = {
      version: 'v2',
      allocation: coinAllocation,
      allocations,
      execution: executionPlan,
      metadata: {
        phase: isBullishPhase ? 'bullish' : isModeratePhase ? 'moderate' : 'bearish',
        floors_applied: selectedFloors,
        adaptive_weights: adaptiveWeights,
        total_check: validateTotalAllocation(coinAllocation),
        // ✅ NOUVEAU (Phase 1.3): Métadonnées meme_cap
        meme_cap: typeof window !== 'undefined' ? window._allocationMetadata?.meme_cap : {
          defined: meme_cap !== null,
          value: meme_cap,
          applied: false
        },
        // 🆕 NOUVEAU (Oct 2025): Structure Modulation V2
        structure_modulation: structure_modulation?.enabled ? {
          ...structure_modulation,
          cap_after: executionPlan.cap_pct_per_iter // Cap effectif APRÈS deltaCap
        } : null
      }
    };

    console.debug('🎯 Final V2 allocation result:', result);
    return result;

  } catch (error) {
    (window.debugLogger?.error || console.error)('❌ Allocation Engine V2 failed:', error);
    return null; // Fallback vers V1
  }
}

/**
 * Niveau 1: Allocation Macro (BTC, ETH, Stables, Alts total)
 */
function calculateMacroAllocation(context, floors) {
  const { cycleScore, adaptiveWeights = {}, risk_budget = {} } = context;

  // SOURCE UNIQUE: le budget stablecoins doit être fourni et vérifiable.
  const stablesTarget = Number.isFinite(risk_budget.target_stables_pct)
    ? risk_budget.target_stables_pct / 100
    : (Number.isFinite(risk_budget.stable_allocation) ? risk_budget.stable_allocation : null);
  if (!Number.isFinite(stablesTarget) || stablesTarget < 0 || stablesTarget > 1) {
    throw new Error('Verified stablecoin risk budget is unavailable');
  }

  // RENORMALISATION PROPORTIONNELLE des non-stables
  const nonStablesSpace = 1 - stablesTarget;

  // Ratios de base selon cycle (avant renormalisation)
  let baseBtcRatio, baseEthRatio, baseAltsRatio;
  if (cycleScore >= 90) {
    // Bull market: plus d'alts
    baseBtcRatio = 0.25;
    baseEthRatio = 0.20;
    baseAltsRatio = 0.55; // Le reste
  } else if (cycleScore >= 70) {
    // Modéré: équilibré
    baseBtcRatio = 0.30;
    baseEthRatio = 0.22;
    baseAltsRatio = 0.48;
  } else {
    // Bearish: défensif
    baseBtcRatio = 0.35;
    baseEthRatio = 0.25;
    baseAltsRatio = 0.40;
  }

  // Renormalisation proportionnelle pour respecter l'espace non-stables
  const baseTotal = baseBtcRatio + baseEthRatio + baseAltsRatio;
  let btcTarget = (baseBtcRatio / baseTotal) * nonStablesSpace;
  let ethTarget = (baseEthRatio / baseTotal) * nonStablesSpace;
  let altsTarget = (baseAltsRatio / baseTotal) * nonStablesSpace;

  // Appliquer floors (après renormalisation)
  btcTarget = Math.max(btcTarget, floors.BTC || 0);
  ethTarget = Math.max(ethTarget, floors.ETH || 0);
  const finalStablesTarget = Math.max(stablesTarget, floors.Stablecoins || 0);

  // Renormalisation finale si floors causent dépassement
  const preNormTotal = btcTarget + ethTarget + finalStablesTarget + altsTarget;
  if (preNormTotal > 1) {
    const excess = preNormTotal - 1;
    // Réduire alts en priorité, puis BTC/ETH proportionnellement
    if (altsTarget >= excess) {
      altsTarget = Math.max(0.05, altsTarget - excess);
    } else {
      altsTarget = 0.05;
      const remainingExcess = excess - (altsTarget - 0.05);
      const btcEthTotal = btcTarget + ethTarget;
      if (btcEthTotal > 0) {
        btcTarget *= (1 - remainingExcess / btcEthTotal);
        ethTarget *= (1 - remainingExcess / btcEthTotal);
      }
    }
  }

  return {
    BTC: btcTarget,
    ETH: ethTarget,
    Stablecoins: finalStablesTarget,
    Alts: altsTarget
  };
}

/**
 * Niveau 2: Redistribution des Alts vers secteurs
 */
function calculateSectorAllocation(macroAllocation, floors, isBullishPhase) {
  const altsTotal = macroAllocation.Alts;

  // Debug: log the floors being used
  console.debug('🏗️ Sector allocation floors:', floors);
  console.debug('📊 Bullish phase:', isBullishPhase, 'Alts total:', altsTotal);

  // Secteurs alts à distribuer
  const altSectors = ['SOL', 'L1/L0 majors', 'L2/Scaling', 'DeFi', 'Memecoins', 'Gaming/NFT', 'AI/Data', 'Others'];

  let allocation = {
    BTC: macroAllocation.BTC,
    ETH: macroAllocation.ETH,
    Stablecoins: macroAllocation.Stablecoins
  };

  // Ratios souhaités par secteur alts
  const sectorRatios = isBullishPhase ? {
    'SOL': 0.25,
    'L1/L0 majors': 0.30,
    'L2/Scaling': 0.15,
    'DeFi': 0.20,
    'Memecoins': 0.05,
    'Gaming/NFT': 0.03,
    'AI/Data': 0.02
  } : {
    'SOL': 0.20,
    'L1/L0 majors': 0.40,
    'L2/Scaling': 0.10,
    'DeFi': 0.15,
    'Memecoins': 0.02,
    'Gaming/NFT': 0.01,
    'AI/Data': 0.01
  };

  // Calcul initial avec floors
  let sectorWeights = {};
  Object.entries(sectorRatios).forEach(([sector, ratio]) => {
    const desiredWeight = altsTotal * ratio;
    const floorWeight = floors[sector] || 0;
    sectorWeights[sector] = Math.max(floorWeight, desiredWeight);
  });

  // NORMALISATION: si la somme des floors > budget alts, réduire proportionnellement
  const totalSectorWeights = Object.values(sectorWeights).reduce((sum, w) => sum + w, 0);
  const othersFloor = floors.Others ?? 0;
  const availableForSectors = Math.max(0, altsTotal - othersFloor);

  if (totalSectorWeights > availableForSectors) {
    const scale = availableForSectors / totalSectorWeights;
    console.debug(`🔧 Sector floors exceed budget alts (${(totalSectorWeights * 100).toFixed(1)}% > ${(availableForSectors * 100).toFixed(1)}%), scaling by ${scale.toFixed(3)}`);
    Object.keys(sectorWeights).forEach(sector => {
      sectorWeights[sector] *= scale;
    });
  }

  // Appliquer les poids normalisés
  Object.entries(sectorWeights).forEach(([sector, weight]) => {
    allocation[sector] = weight;
  });

  // Others = reste disponible
  const finalAllocated = Object.values(sectorWeights).reduce((sum, w) => sum + w, 0);
  allocation.Others = Math.max(othersFloor, altsTotal - finalAllocated);

  return allocation;
}

/**
 * Niveau 3: Distribution intra-secteur avec protection incumbency + meme_cap
 * @param {number|null} meme_cap - Cap maximal pour Memecoins en % (0-100), depuis market-regimes
 */
function calculateCoinAllocation(sectorAllocation, currentPositions, floors, meme_cap = null) {
  const coinAllocation = {};
  const heldAssets = new Set(currentPositions.map(pos => pos.symbol?.toUpperCase()).filter(Boolean));

  console.debug('🔒 Incumbency protection for held assets:', Array.from(heldAssets));
  console.debug('🎭 Meme cap from regime:', meme_cap !== null ? `${meme_cap}%` : 'none');

  // Debug: check how assets are classified
  currentPositions.forEach(pos => {
    const symbol = pos.symbol?.toUpperCase();
    const group = getAssetGroup(symbol);
    console.debug(`🏷️ Asset ${symbol} → Group: ${group}`);
  });

  // Debug: show UNIFIED_ASSET_GROUPS structure
  console.debug('🏗️ UNIFIED_ASSET_GROUPS:', UNIFIED_ASSET_GROUPS);

  // Calculer total portfolio pour seuil minimum incumbency (1%)
  const totalPortfolioValue = currentPositions.reduce((sum, pos) => sum + (parseFloat(pos.value_usd) || 0), 0);
  const INCUMBENCY_MIN_THRESHOLD = 0.01; // 1% du portfolio total

  console.debug('💼 Total portfolio value:', totalPortfolioValue.toFixed(2), 'USD');

  // Pour chaque secteur, distribuer vers les coins
  Object.entries(sectorAllocation).forEach(([sector, sectorWeight]) => {
    // Ensure sectorWeight is a valid number
    const validSectorWeight = isNaN(sectorWeight) || sectorWeight == null ? 0 : sectorWeight;

    // Tous les secteurs peuvent avoir des subdivisions si des coins sont détenus
    const sectorAssets = UNIFIED_ASSET_GROUPS[sector] || [];
    const heldInSector = sectorAssets.filter(asset => heldAssets.has(asset));

    if (heldInSector.length === 0) {
      // Pas d'assets détenus dans ce secteur → allocation au groupe
      coinAllocation[sector] = validSectorWeight;
    } else {
      // INCUMBENCY BORNÉ: si n*3% > secteur, répartir secteur/n, reste = 0
      const desiredIncumbencyFloor = floors.incumbency || 0.03;
      const desiredIncumbencyTotal = heldInSector.length * desiredIncumbencyFloor;

      let actualIncumbencyFloor, remainingWeight;
      if (desiredIncumbencyTotal > validSectorWeight) {
        // Cas: incumbency dépasserait le secteur → répartir équitablement
        actualIncumbencyFloor = validSectorWeight / heldInSector.length;
        remainingWeight = 0;
        console.debug(`⚠️ Incumbency capped for ${sector}: ${heldInSector.length} × ${desiredIncumbencyFloor.toFixed(3)} → ${actualIncumbencyFloor.toFixed(3)} each`);
      } else {
        // Cas normal: incumbency + reste
        actualIncumbencyFloor = desiredIncumbencyFloor;
        remainingWeight = validSectorWeight - desiredIncumbencyTotal;
      }

      // HIERARCHIE STRICTE: soit secteur global, soit coins individuels, jamais les deux
      if (heldInSector.length === 1) {
        // 🆕 SEUIL MINIMUM: Ne pas exposer individuellement si < 1% du portfolio
        const assetPosition = currentPositions.find(pos => pos.symbol?.toUpperCase() === heldInSector[0]);
        const assetValue = assetPosition ? parseFloat(assetPosition.value_usd) || 0 : 0;
        const assetPercentage = totalPortfolioValue > 0 ? assetValue / totalPortfolioValue : 0;

        if (assetPercentage >= INCUMBENCY_MIN_THRESHOLD) {
          // Asset >= 1% du portfolio → l'exposer directement
          coinAllocation[heldInSector[0]] = validSectorWeight;
          console.debug(`✅ Individual exposure for ${heldInSector[0]} (${(assetPercentage * 100).toFixed(2)}% of portfolio)`);
        } else {
          // Asset < 1% du portfolio → utiliser le groupe
          coinAllocation[sector] = validSectorWeight;
          console.debug(`⚠️ Using group ${sector} instead of ${heldInSector[0]} (only ${(assetPercentage * 100).toFixed(2)}% of portfolio)`);
        }
        // NE PAS ajouter le secteur global pour éviter double-comptage
      } else if (heldInSector.length > 1) {
        // Plusieurs coins détenus → distribution avec incumbency borné
        heldInSector.forEach(asset => {
          const assetWeight = actualIncumbencyFloor + (remainingWeight / heldInSector.length);
          coinAllocation[asset] = isNaN(assetWeight) ? actualIncumbencyFloor : assetWeight;
        });
        // NE PAS ajouter le secteur global
      }
    }
  });

  // ✅ NOUVEAU (Phase 1.3): Appliquer meme_cap APRÈS calcul initial, AVANT normalisation
  let memeCapApplied = false;
  if (meme_cap !== null && typeof meme_cap === 'number') {
    const memeCapDecimal = meme_cap / 100; // Convertir % en décimal (0-1)

    // Calculer allocation actuelle Memecoins (groupe + coins individuels)
    let totalMemecoins = coinAllocation['Memecoins'] || 0;

    // Ajouter coins individuels classés comme memecoins
    const memecoinsGroup = UNIFIED_ASSET_GROUPS['Memecoins'] || [];
    memecoinsGroup.forEach(asset => {
      if (coinAllocation[asset]) {
        totalMemecoins += coinAllocation[asset];
      }
    });

    // Appliquer le cap si dépassement
    if (totalMemecoins > memeCapDecimal) {
      const excess = totalMemecoins - memeCapDecimal;
      const reductionFactor = memeCapDecimal / totalMemecoins;

      // Réduire proportionnellement groupe + coins individuels
      if (coinAllocation['Memecoins']) {
        coinAllocation['Memecoins'] *= reductionFactor;
      }
      memecoinsGroup.forEach(asset => {
        if (coinAllocation[asset]) {
          coinAllocation[asset] *= reductionFactor;
        }
      });

      // Redistribuer l'excédent vers BTC/ETH COINS (pas les groupes!)
      const btcShare = 0.6;
      const ethShare = 0.4;

      // Distribuer aux coins détenus dans BTC group
      const btcGroup = UNIFIED_ASSET_GROUPS['BTC'] || [];
      const heldBtc = btcGroup.filter(asset => heldAssets.has(asset));
      if (heldBtc.length > 0) {
        const btcExcessPerCoin = (excess * btcShare) / heldBtc.length;
        heldBtc.forEach(asset => {
          coinAllocation[asset] = (coinAllocation[asset] || 0) + btcExcessPerCoin;
        });
      } else {
        // Si aucun BTC coin détenu, allouer au groupe
        coinAllocation['BTC'] = (coinAllocation['BTC'] || 0) + excess * btcShare;
      }

      // Distribuer aux coins détenus dans ETH group
      const ethGroup = UNIFIED_ASSET_GROUPS['ETH'] || [];
      const heldEth = ethGroup.filter(asset => heldAssets.has(asset));
      if (heldEth.length > 0) {
        const ethExcessPerCoin = (excess * ethShare) / heldEth.length;
        heldEth.forEach(asset => {
          coinAllocation[asset] = (coinAllocation[asset] || 0) + ethExcessPerCoin;
        });
      } else {
        // Si aucun ETH coin détenu, allouer au groupe
        coinAllocation['ETH'] = (coinAllocation['ETH'] || 0) + excess * ethShare;
      }

      memeCapApplied = true;
      console.debug(`🎭 Meme cap applied: ${(totalMemecoins * 100).toFixed(1)}% → ${meme_cap}% (excess ${(excess * 100).toFixed(2)}% → BTC/ETH coins: ${[...heldBtc, ...heldEth].join(', ')})`);
    }
  }

  // Log métadonnée pour validation
  if (typeof window !== 'undefined') {
    if (!window._allocationMetadata) window._allocationMetadata = {};
    window._allocationMetadata.meme_cap = {
      defined: meme_cap !== null,
      value: meme_cap,
      applied: memeCapApplied
    };
  }

  // 🔧 NORMALISATION PRÉVENTIVE: corriger les erreurs d'arrondi accumulées
  const allocSum = Object.values(coinAllocation).reduce((sum, val) =>
    sum + (typeof val === 'number' && !isNaN(val) ? val : 0), 0
  );

  if (Math.abs(allocSum - 1.0) > 0.001) {
    console.debug(`🔧 Normalizing coin allocation: ${(allocSum * 100).toFixed(2)}% → 100%`);
    const scale = 1.0 / allocSum;
    Object.keys(coinAllocation).forEach(key => {
      if (typeof coinAllocation[key] === 'number' && !isNaN(coinAllocation[key])) {
        coinAllocation[key] *= scale;
      }
    });
  }

  return coinAllocation;
}

/**
 * Calcul du plan d'exécution (iterations estimées)
 * 🆕 MODIFIÉ (Oct 2025): Support Structure Modulation V2 deltaCap
 */
function calculateExecutionPlan(targetAllocation, currentPositions, executionContext = {}) {
  let capPct = executionContext.cap_pct_per_iter;

  // 🆕 Extraire deltaCap depuis structure_modulation
  const deltaCap = executionContext.structure_modulation?.delta_cap ?? 0;

  // ✅ MODIFIÉ (Phase 1.2): Utiliser selectEffectiveCap au lieu de selectCapPercent
  // 🆕 MODIFIÉ (Oct 2025): Appliquer deltaCap depuis structure modulation
  // Gère automatiquement: backend error (5%), staleness (8%), alert override, policy, engine + structure deltaCap
  if (capPct == null) {
    const contextState = executionContext.state || executionContext.unified_state || null;
    if (contextState) {
      capPct = getEffectiveCapWithStructure(contextState, deltaCap);
    }
  }

  if (capPct == null && typeof executionContext.cap_daily === 'number') {
    const raw = executionContext.cap_daily;
    capPct = raw <= 1 ? Math.round(raw * 100) : Math.round(raw);
  }

  if (capPct == null && typeof window !== 'undefined') {
    try {
      const fallbackState = (typeof window.store?.snapshot === 'function' ? window.store.snapshot() : null) || window.realDataStore || {};
      capPct = getEffectiveCapWithStructure(fallbackState, deltaCap);
    } catch (error) {
      console.debug('calculateExecutionPlan cap fallback failed', error?.message || error);
    }
  }

  // PARE-FEU (Oct 2025): Ne jamais fallbacker silencieusement à 0%
  // Si cap indisponible, retourner plan vide (skip ce tick)
  if (capPct == null) {
    debugLogger.warn('[Exec] Cap indisponible → skip iteration (pas de fallback 0%)');
    return {
      iterations: 0,
      estimated_days: 0,
      cap_unavailable: true,
      message: 'Cap not available, execution skipped this tick'
    };
  }

  const capPerIter = capPct / 100;

  const currentAlloc = calculateCurrentAllocation(currentPositions);
  let maxDelta = 0;
  let maxDeltaGroup = '';

  const deltas = [];
  Object.entries(targetAllocation).forEach(([asset, target]) => {
    const current = currentAlloc[asset] || 0;
    const delta = Math.abs(target - current);
    deltas.push({ asset, current: current * 100, target: target * 100, delta: delta * 100 });

    if (delta > maxDelta) {
      maxDelta = delta;
      maxDeltaGroup = asset;
    }
  });

  const estimatedIters = capPerIter > 0 ? Math.ceil(maxDelta / capPerIter) : Infinity;

  console.debug('🔄 Convergence calculation:', {
    maxDeltaPct: (maxDelta * 100).toFixed(1),
    maxDeltaGroup,
    capPerIter: capPct,
    estimatedIters: Number.isFinite(estimatedIters) ? estimatedIters : '∞',
    formula: capPerIter > 0 ? `ceil(${(maxDelta * 100).toFixed(1)}% / ${capPct}%) = ${estimatedIters}` : 'cap=0 -> ∞',
    allDeltas: deltas.filter(d => d.delta > 0.1).map(d => `${d.asset}: ${d.current.toFixed(1)}% → ${d.target.toFixed(1)}% (Δ${d.delta.toFixed(1)}%)`)
  });

  return {
    estimated_iters_to_target: Number.isFinite(estimatedIters) ? estimatedIters : Infinity,
    max_delta_pct: maxDelta * 100,
    cap_per_iter: capPct,
    convergence_time_estimate: capPerIter > 0 ? `${estimatedIters} rebalances` : 'Cap unavailable'
  };
}

/**
 * Utilitaires
 */
function calculateCurrentAllocation(positions) {
  const total = positions.reduce((sum, pos) => sum + (parseFloat(pos.value_usd) || 0), 0);
  const allocation = {};

  positions.forEach(pos => {
    const group = getAssetGroup(pos.symbol);
    const weight = (parseFloat(pos.value_usd) || 0) / total;
    allocation[group] = (allocation[group] || 0) + weight;
  });

  return allocation;
}

function validateTotalAllocation(allocation) {
  // Filter out null, undefined, and NaN values before summing
  const validValues = Object.values(allocation).filter(val =>
    val !== null && val !== undefined && !isNaN(val) && typeof val === 'number'
  );
  const total = validValues.reduce((sum, val) => sum + val, 0);
  const isValid = Math.abs(total - 1) < 0.001; // Tolérance 0.1%

  if (!isValid) {
    (window.debugLogger?.warn || console.warn)('⚠️ Total allocation mismatch:', total, 'from values:', validValues);
  }

  return { total, isValid };
}

/**
 * Validation hiérarchique - détecte double-comptage et incohérences
 */
function validateHierarchy(allocation, currentPositions) {
  const issues = [];
  const allocationKeys = Object.keys(allocation);

  // 🔍 ÉTAPE 0: Identifier en amont les groupes avec coins éponymes (BTC, ETH, SOL, Stablecoins)
  const topLevelGroups = ['BTC', 'ETH', 'Stablecoins', 'SOL', 'L1/L0 majors', 'L2/Scaling', 'DeFi', 'AI/Data', 'Gaming/NFT', 'Memecoins', 'Others'];
  const eponymousGroups = new Set();
  topLevelGroups.forEach(group => {
    const groupAssets = UNIFIED_ASSET_GROUPS[group] || [];
    if (groupAssets.includes(group)) {
      eponymousGroups.add(group);
    }
  });

  // Vérifier double-comptage: un coin ne doit pas coexister avec son groupe parent
  // ✅ FIX: Skip validation si le groupe parent est éponyme
  currentPositions.forEach(pos => {
    const symbol = pos.symbol?.toUpperCase();
    const group = getAssetGroup(symbol);

    // Skip si le groupe parent est éponyme (BTC, ETH, SOL, Stablecoins)
    // car allocation[group] représente alors le coin éponyme, pas un groupe parent
    if (eponymousGroups.has(group)) {
      return; // Skip - pas de double-comptage dans ce cas
    }

    if (allocation[symbol] && allocation[group] && symbol !== group) {
      issues.push(`Double-comptage: ${symbol} (${allocation[symbol].toFixed(3)}) + ${group} (${allocation[group].toFixed(3)})`);
    }
  });

  // Vérifier cohérence des groupes vs sous-éléments avec GROUP_ORDER (synchrone)
  // (eponymousGroups déjà calculé ci-dessus)

  topLevelGroups.forEach(group => {
    const groupWeight = allocation[group] || 0;
    const groupAssets = UNIFIED_ASSET_GROUPS[group] || [];

    // SPECIAL CASE: Si le groupe contient un enfant avec le même nom (ex: coin "BTC" dans groupe "BTC"),
    // alors allocation[group] représente le COIN éponyme, pas le groupe parent.
    // On doit skip la validation hiérarchique pour ce groupe ET tous ses enfants.
    const hasEponymousChild = eponymousGroups.has(group);
    if (hasEponymousChild && groupWeight > 0) {
      console.debug(`🔍 group_eponymous: ${group} contains child with same name, allocation[${group}]=${(groupWeight * 100).toFixed(1)}% is treated as coin, not parent group`);
      return; // Skip validation for this group
    }

    // IMPORTANT: Exclure les coins qui ont le même nom que le groupe (déjà géré ci-dessus)
    const childrenWeights = groupAssets
      .filter(asset => asset !== group)  // Exclure le coin éponyme
      .filter(asset => allocation[asset])
      .reduce((sum, asset) => sum + (allocation[asset] || 0), 0);

    if (groupWeight > 0 && childrenWeights > 0) {
      const childrenList = groupAssets.filter(asset => asset !== group && allocation[asset]).map(asset => `${asset}=${(allocation[asset] * 100).toFixed(2)}%`).join(', ');
      const issue = `Groupe ${group} (${(groupWeight * 100).toFixed(2)}%) coexiste avec enfants (${(childrenWeights * 100).toFixed(2)}%: ${childrenList})`;
      (window.debugLogger?.warn || console.warn)(`⚠️ HIERARCHY: ${issue}`);
      issues.push(issue);
    }

    // GUARD: group_without_descent - affiné selon la demande
    const isTerminal = ['BTC', 'ETH', 'Stablecoins', 'Others'].includes(group);
    if (groupWeight > 0.001 && childrenWeights === 0 && groupAssets.length > 0) {
      if (isTerminal) {
        console.debug(`🔍 group_without_descent (terminal): ${group} (${groupWeight.toFixed(3)}) - OK pour terminal`);
      } else {
        console.debug(`🔍 group_without_descent (secteur): ${group} (${groupWeight.toFixed(3)}) - drill-down vide autorisé`);
      }
    }
  });

  // GUARD: child_at_top_level - WARN seulement si parent a poids > 0 (vrai double-comptage)
  // ✅ FIX: Skip validation pour les enfants de groupes éponymes (BTC, ETH)
  allocationKeys.forEach(key => {
    if (!topLevelGroups.includes(key) && allocation[key] > 0.001) {
      const parentGroup = getAssetGroup(key);
      const parentWeight = allocation[parentGroup] || 0;

      // 🔍 NOUVEAU: Si le parent est un groupe éponyme, skip la validation
      // car tous les coins du groupe (BTC, TBTC, WBTC / ETH, STETH, RETH, WSTETH)
      // sont des allocations individuelles, pas des double-comptages
      if (eponymousGroups.has(parentGroup)) {
        console.debug(`🔍 child_in_eponymous_group: ${key} belongs to eponymous group ${parentGroup} - skip validation`);
        return;
      }

      if (parentGroup !== key && parentGroup !== 'Others' && parentWeight > 0.001) {
        (window.debugLogger?.warn || console.warn)(`⚠️ child_at_top_level: ${key} (${allocation[key].toFixed(3)}) + parent ${parentGroup} (${parentWeight.toFixed(3)}) = vrai double-comptage`);
        issues.push(`child_at_top_level: ${key} → ${parentGroup}`);
      } else if (parentGroup !== key) {
        console.debug(`🔍 child_at_top_level: ${key} (${allocation[key].toFixed(3)}) mais parent ${parentGroup} = 0 - OK`);
      }
    }
  });

  return {
    valid: issues.length === 0,
    issues: issues
  };
}

function logAllocationDecisions(data) {
  console.group('🏗️ Allocation Engine Decisions');
  console.debug('📊 Input context:', {
    cycle: data.context.cycleScore,
    onchain: data.context.onchainScore,
    contradiction: data.context.contradiction,
    adaptive_weights: data.context.adaptiveWeights
  });
  console.debug('🌍 Macro allocation:', data.macro);
  console.debug('🏭 Sector allocation:', data.sectors);
  console.debug('🪙 Final coin allocation:', data.coins);
  console.debug('⏱️ Execution plan:', data.execution);
  console.groupEnd();
}

// Export par défaut pour compatibilité
export default { calculateHierarchicalAllocation };

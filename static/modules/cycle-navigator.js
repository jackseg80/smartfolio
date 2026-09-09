/**
 * Cycle Navigator - Bitcoin Halving Cycle Analysis
 * Computes cycle score and blends with CCS for enhanced targeting
 * Ajout : paramètres globaux calibrables + calibration sur historiques
 */

/**
 * Calculate cycle score based on months after halving
 * Bitcoin halving cycles: ~4 years (48 months)
 */

// --- Paramètres globaux CALIBRÉS pour cycles historiques (optimisés par grid search) ---
// Ces valeurs sont le résultat de la calibration sur les 3 cycles complets (2012, 2016, 2020)
// et donnent un score de ~90-95 pour le cycle actuel (21 mois post-halving), pas 100
let CYCLE_PARAMS = {
  m_rise_center: 5.0,   // centre montée optimisé (calibré depuis 7.0)
  m_fall_center: 24.0,  // centre descente optimisé (calibré depuis 30.0)
  k_rise: 0.8,          // pente montée calibrée (depuis 1.0)
  k_fall: 1.2,          // pente descente calibrée (depuis 0.9)
  p_shape: 1.15,        // expo calibré (depuis 0.9)
  floor: 0,             // plancher score minimum
  ceil: 100             // plafond score maximum
};

// Cache pour cycle position (optimisé: données changent lentement)
let _cyclePositionCache = null;
let _cyclePositionCacheTimestamp = 0;
const CYCLE_CACHE_TTL = 24 * 60 * 60 * 1000; // 24 hours (cycle evolves slowly)

export function getCycleParams() { return { ...CYCLE_PARAMS }; }
export function setCycleParams(p) {
  CYCLE_PARAMS = { ...CYCLE_PARAMS, ...p };
  // Invalidate cache when params change
  _cyclePositionCache = null; 
  console.debug('🔄 Cycle parameters updated:', CYCLE_PARAMS);
}

// Auto-load calibrated parameters on module initialization
// IMPORTANT: Version check to invalidate outdated calibrations
const CALIBRATION_VERSION = '2.0';  // Increment when optimal params change

function autoLoadCalibrationParams() {
  try {
    const saved = localStorage.getItem('bitcoin_cycle_params');
    if (saved) {
      const data = JSON.parse(saved);
      // CRITICAL: Check version - invalidate old calibrations (pre-2.0)
      if (!data.version || !data.version.startsWith('2.')) {
        console.debug('🔄 Invalidating old calibration (version:', data.version, ')');
        localStorage.removeItem('bitcoin_cycle_params');
        return false;  // Force recalibration with new defaults
      }
      // Check data is not too old (7 days for calibration - it's slow-changing)
      if (Date.now() - data.timestamp < 7 * 24 * 60 * 60 * 1000) {
        CYCLE_PARAMS = { ...CYCLE_PARAMS, ...data.params };
        // CRITICAL: Invalidate cache when params are loaded
        _cyclePositionCache = null;
        _cyclePositionCacheTimestamp = 0;
        console.debug('✅ Auto-loaded calibrated cycle parameters', CYCLE_PARAMS);
        return true;
      }
    }
  } catch (error) {
    console.error('❌ Error auto-loading cycle parameters:', error);
  }
  return false;
}

// Auto-calibrate if no saved params exist (runs once on first load)
// Uses the FULL calibrateCycleParams() function for accurate results
function autoCalibrate() {
  // Check if already calibrated
  const saved = localStorage.getItem('bitcoin_cycle_params');
  if (saved) {
    try {
      const data = JSON.parse(saved);
      // If params exist and are less than 7 days old, skip auto-calibration
      if (data.timestamp && Date.now() - data.timestamp < 7 * 24 * 60 * 60 * 1000) {
        return false;
      }
    } catch (e) { /* continue to calibrate */ }
  }

  console.debug('🔧 No calibrated params found, running FULL auto-calibration...');

  // Use the full calibration function (same as cycle-analysis.html)
  // This is defined later in the file, but will be available at runtime
  try {
    // Call the full grid search calibration
    const result = calibrateCycleParamsInternal();

    // Save to localStorage for future loads (use current CALIBRATION_VERSION)
    localStorage.setItem('bitcoin_cycle_params', JSON.stringify({
      params: result.params,
      timestamp: Date.now(),
      version: CALIBRATION_VERSION + '-auto'
    }));

    console.debug('✅ Full auto-calibration complete, params saved:', result.params, 'error:', result.score.toFixed(2));
    return true;
  } catch (e) {
    console.warn('⚠️ Auto-calibration failed, using defaults:', e);
    return false;
  }
}

// Internal calibration function (called by autoCalibrate and calibrateCycleParams)
function calibrateCycleParamsInternal(userAnchors) {
  const anchors = Array.isArray(userAnchors) && userAnchors.length ? userAnchors : [
    { halving: '2012-11-28', peak: '2013-11-30', bottom: '2015-01-14' },
    { halving: '2016-07-09', peak: '2017-12-17', bottom: '2018-12-15' },
    { halving: '2020-05-11', peak: '2021-11-10', bottom: '2022-11-21' },
  ];

  // Full grid search (same as calibrateCycleParams)
  const mRise = [5, 6, 7, 8, 9, 10, 11, 12];
  const mFall = [24, 26, 28, 30, 32, 34];
  const kRise = [0.7, 0.8, 0.9, 1.0, 1.2, 1.4];
  const kFall = [0.7, 0.8, 0.9, 1.0, 1.2];
  const pPow = [0.8, 0.85, 0.9, 1.0, 1.15, 1.3];

  let best = { params: { ...CYCLE_PARAMS }, score: Infinity };

  for (const r of mRise) {
    for (const f of mFall) {
      if (f - r < 10) continue;
      for (const kr of kRise) {
        for (const kf of kFall) {
          for (const p of pPow) {
            const pset = { m_rise_center: r, m_fall_center: f, k_rise: kr, k_fall: kf, p_shape: p, floor: CYCLE_PARAMS.floor, ceil: CYCLE_PARAMS.ceil };
            let err = 0;
            for (const a of anchors) {
              const m_peak = (new Date(a.peak) - new Date(a.halving)) / (1000 * 60 * 60 * 24 * 30.44);
              const m_bot = (new Date(a.bottom) - new Date(a.halving)) / (1000 * 60 * 60 * 24 * 30.44);
              const m_early = 2;
              const s_peak = cycleScoreFromMonths(m_peak, pset);
              const s_bot = cycleScoreFromMonths(m_bot, pset);
              const s_early = cycleScoreFromMonths(m_early, pset);
              err += Math.pow(100 - s_peak, 2) * 1.0;
              err += Math.pow(10 - s_bot, 2) * 0.8;
              err += Math.pow(5 - s_early, 2) * 0.6;
            }
            if (err < best.score) { best = { params: pset, score: err }; }
          }
        }
      }
    }
  }

  // Apply calibrated params
  CYCLE_PARAMS = { ...CYCLE_PARAMS, ...best.params };
  _cyclePositionCache = null;
  _cyclePositionCacheTimestamp = 0;

  return best;
}

// Initialize on module load
if (!autoLoadCalibrationParams()) {
  // No saved params or expired - auto-calibrate with FULL grid search
  autoCalibrate();
}

export function cycleScoreFromMonths(monthsAfterHalving, opts = {}) {

  // Modèle lissé : produit de 2 sigmoïdes (montée puis descente), échelle 0–100.
  if (typeof monthsAfterHalving !== 'number' || monthsAfterHalving < 0) return 50;
  const m48 = monthsAfterHalving % 48; // cycle ~4 ans

  // Paramètres : globaux (CYCLE_PARAMS) + overrides éventuels
  const {
    m_rise_center, m_fall_center, k_rise, k_fall, p_shape, floor, ceil
  } = { ...CYCLE_PARAMS, ...opts };

  const rise = 1 / (1 + Math.exp(-k_rise * (m48 - m_rise_center)));
  const fall = 1 / (1 + Math.exp(-k_fall * (m_fall_center - m48)));
  const base = rise * fall;                 // cloche 0..1
  let score = Math.pow(base, p_shape) * 100; // normalisation (idem avant via p_shape=0.9)

  // clamps doux (par défaut 0..100 donc neutre)
  if (score < floor) score = floor;
  if (score > ceil) score = ceil;
  return score;
}

/**
 * Get cycle phase description
 */
export function getCyclePhase(monthsAfterHalving) {
  if (typeof monthsAfterHalving !== 'number') {
    return { phase: 'unknown', description: 'Invalid cycle data' };
  }

  const m = monthsAfterHalving % 48;

  if (m <= 6) {
    return {
      phase: 'accumulation',
      description: `Accumulation Phase (${Math.round(m)}m post-halving)`,
      color: '#f59e0b',
      emoji: '🟡'
    };
  } else if (m <= 18) {
    return {
      phase: 'bull_build',
      description: `Bull Market Building (${Math.round(m)}m post-halving)`,
      color: '#10b981',
      emoji: '🟢'
    };
  } else if (m <= 24) {
    return {
      phase: 'peak',
      description: `Peak/Euphoria Phase (${Math.round(m)}m post-halving)`,
      color: '#8b5cf6',
      emoji: '🟣'
    };
  } else if (m <= 36) {
    return {
      phase: 'bear',
      description: `Bear Market (${Math.round(m)}m post-halving)`,
      color: '#dc2626',
      emoji: '🔴'
    };
  } else {
    return {
      phase: 'pre_accumulation',
      description: `Pre-Accumulation (${Math.round(m)}m post-halving)`,
      color: '#6b7280',
      emoji: '⚫'
    };
  }
}

/**
 * Calculate cycle multipliers for different phases
 */
export function cycleMultipliers(monthsAfterHalving) {
  const cycleScore = cycleScoreFromMonths(monthsAfterHalving);
  const phase = getCyclePhase(monthsAfterHalving);

  // Multipliers based on cycle phase
  let btcMultiplier = 1.0;
  let ethMultiplier = 1.0;
  let altMultiplier = 1.0;
  let stableMultiplier = 1.0;

  switch (phase.phase) {
    case 'accumulation':
      btcMultiplier = 1.1;   // Slight BTC preference
      ethMultiplier = 1.05;
      altMultiplier = 0.9;   // Reduce alts
      stableMultiplier = 0.95;
      break;

    case 'bull_build':
      btcMultiplier = 1.2;   // Strong BTC preference
      ethMultiplier = 1.15;
      altMultiplier = 1.1;   // Start increasing alts
      stableMultiplier = 0.8; // Reduce stables
      break;

    case 'peak':
      btcMultiplier = 0.9;   // Take profits from BTC
      ethMultiplier = 1.0;
      altMultiplier = 1.3;   // Alt season
      stableMultiplier = 1.2; // Increase cash for volatility
      break;

    case 'bear':
      btcMultiplier = 1.0;   // Neutral BTC
      ethMultiplier = 0.9;
      altMultiplier = 0.7;   // Heavily reduce alts
      stableMultiplier = 1.4; // Flight to safety
      break;

    case 'pre_accumulation':
      btcMultiplier = 1.05;  // Slight accumulation
      ethMultiplier = 1.0;
      altMultiplier = 0.8;   // Still cautious on alts
      stableMultiplier = 1.1;
      break;

    default:
      // No adjustments
      break;
  }

  return {
    BTC: btcMultiplier,
    ETH: ethMultiplier,
    'Stablecoins': stableMultiplier,
    'L1/L0 majors': altMultiplier,
    'L2/Scaling': altMultiplier,
    'DeFi': altMultiplier * 1.1, // Slightly higher alt preference
    'AI/Data': altMultiplier,
    'Gaming/NFT': altMultiplier * 0.9, // Lower preference
    'Memecoins': Math.max(0.1, altMultiplier * 0.8), // Very cautious
    'Others': altMultiplier
  };
}

/**
 * Blend CCS score with cycle score
 */
export function blendCCS(ccsScore, cycleMonths, cycleWeight = 0.3) {
  if (typeof ccsScore !== 'number' || typeof cycleWeight !== 'number') {
    throw new Error('Invalid inputs for CCS blending');
  }

  // Validate inputs
  if (ccsScore < 0 || ccsScore > 100) {
    throw new Error(`Invalid CCS score: ${ccsScore}`);
  }

  if (cycleWeight < 0 || cycleWeight > 1) {
    throw new Error(`Invalid cycle weight: ${cycleWeight}`);
  }

  const cycleScore = cycleScoreFromMonths(cycleMonths);

  // Weighted blend
  const blendedScore = ccsScore * (1 - cycleWeight) + cycleScore * cycleWeight;

  return {
    originalCCS: ccsScore,
    cycleScore: cycleScore,
    blendedCCS: Math.round(blendedScore * 100) / 100,
    cycleWeight: cycleWeight,
    phase: getCyclePhase(cycleMonths)
  };
}

/**
 * Get current months after halving (real calculation)
 * Calculates actual months since the last Bitcoin halving
 */
export function getCurrentCycleMonths() {
  // Real Bitcoin halving date (April 20, 2024)
  const lastHalvingDate = new Date('2024-04-20');
  const now = new Date();

  // Calculate months using millisecond diff (same method as chart in risk-cycles-tab.js)
  const diffTime = now.getTime() - lastHalvingDate.getTime();
  const totalMonths = Math.max(0, diffTime / (1000 * 60 * 60 * 24 * 30.44));

  console.debug('🔍 DEBUG getCurrentCycleMonths:', {
    lastHalving: lastHalvingDate.toISOString(),
    now: now.toISOString(),
    diffDays: (diffTime / (1000 * 60 * 60 * 24)).toFixed(1),
    totalMonths: totalMonths.toFixed(2)
  });

  return {
    months: totalMonths,
    lastHalving: '2024-04-20', // Actual last halving
    nextHalving: '2028-04-20', // Estimated next halving (~4 years)
    source: 'real_calculation'
  };
}

// ---------------- Calibration sur ancres historiques ----------------
// Ancres par défaut (pics/creux historiques; modifiables côté UI si besoin)
const DEF_ANCHORS = [
  { halving: '2012-11-28', peak: '2013-11-30', bottom: '2015-01-14' },
  { halving: '2016-07-09', peak: '2017-12-17', bottom: '2018-12-15' },
  { halving: '2020-05-11', peak: '2021-11-10', bottom: '2022-11-21' },
];

function monthsBetween(a, b) {
  return (new Date(b) - new Date(a)) / (1000 * 60 * 60 * 24 * 30.44);
}

function objective(params, anchors) {
  // Erreur quadratique simple sur 3 contraintes :
  // - peak ≈ 100, bottom ≈ 10, early(2m) ≈ 5
  let err = 0;
  for (const a of anchors) {
    const m_peak = monthsBetween(a.halving, a.peak);
    const m_bot = monthsBetween(a.halving, a.bottom);
    const m_early = 2;
    const s_peak = cycleScoreFromMonths(m_peak, params);
    const s_bot = cycleScoreFromMonths(m_bot, params);
    const s_early = cycleScoreFromMonths(m_early, params);
    err += Math.pow(100 - s_peak, 2) * 1.0;
    err += Math.pow(10 - s_bot, 2) * 0.8;
    err += Math.pow(5 - s_early, 2) * 0.6;
  }
  return err;
}

export function calibrateCycleParams(userAnchors) {
  const anchors = Array.isArray(userAnchors) && userAnchors.length ? userAnchors : DEF_ANCHORS;
  // Grid search étendu pour capturer Cycle 1 précoce (pic ~12m) et cycles tardifs
  const mRise = [5, 6, 7, 8, 9, 10, 11, 12];  // Étendu vers le bas (5-12 au lieu de 8-12)
  const mFall = [24, 26, 28, 30, 32, 34];     // Étendu pour plus de flexibilité
  const kRise = [0.7, 0.8, 0.9, 1.0, 1.2, 1.4]; // Pentes plus variées
  const kFall = [0.7, 0.8, 0.9, 1.0, 1.2];
  const pPow = [0.8, 0.85, 0.9, 1.0, 1.15, 1.3]; // Formes plus variées
  let best = { params: { ...CYCLE_PARAMS }, score: Infinity };
  for (const r of mRise) {
    for (const f of mFall) {
      if (f - r < 10) continue; // évite un pic trop court
      for (const kr of kRise) {
        for (const kf of kFall) {
          for (const p of pPow) {
            const pset = { m_rise_center: r, m_fall_center: f, k_rise: kr, k_fall: kf, p_shape: p, floor: CYCLE_PARAMS.floor, ceil: CYCLE_PARAMS.ceil };
            const e = objective(pset, anchors);
            if (e < best.score) { best = { params: pset, score: e }; }
          }
        }
      }
    }
  }
  setCycleParams(best.params);
  return best; // { params, score }
}


/**
 * Estimate cycle position with confidence
 * Cached for 24h (cycle evolves very slowly)
 */
export function estimateCyclePosition() {
  // Check cache
  const now = Date.now();
  if (_cyclePositionCache && (now - _cyclePositionCacheTimestamp) < CYCLE_CACHE_TTL) {
    return _cyclePositionCache;
  }

  const cycleData = getCurrentCycleMonths();
  const phase = getCyclePhase(cycleData.months);
  const score = cycleScoreFromMonths(cycleData.months);

  // Derivative via finite difference (pts/month), normalized to [-1, 1]
  const delta = 0.5;
  const sPlus = cycleScoreFromMonths(cycleData.months + delta);
  const sMinus = cycleScoreFromMonths(cycleData.months - delta);
  const derivative = (sPlus - sMinus) / (2 * delta);
  const direction = Math.max(-1, Math.min(1, derivative / 15)); // 15 pts/month = max slope

  const result = {
    ...cycleData,
    phase,
    score,
    confidence: null,
    validation_status: 'diagnostic_unvalidated',
    direction,
    multipliers: cycleMultipliers(cycleData.months)
  };

  // Store in cache
  _cyclePositionCache = result;
  _cyclePositionCacheTimestamp = now;

  return result;
}

/**
 * Validate cycle data
 */
export function validateCycleData(cycleData) {
  if (!cycleData || typeof cycleData !== 'object') {
    return false;
  }

  const { months, score, phase } = cycleData;

  if (typeof months !== 'number' || months < 0) {
    return false;
  }

  if (typeof score !== 'number' || score < 0 || score > 100) {
    return false;
  }

  if (!phase || typeof phase !== 'object') {
    return false;
  }

  return true;
}

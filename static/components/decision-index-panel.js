/**
 * Decision Index Panel v8.0 - Mini Cycle Chart + Fused Pillars
 *
 * Layout 2 colonnes équilibré avec recommandations contextuelles
 * - Colonne gauche: Score DI + Barre + Contributions annotées + Metadata + Recommendation
 * - Colonne droite: Context Bar + Mini Cycle Chart + Allocation texte + Pillars fusionnés + Key Metrics
 * - Design gaming compact et moderne
 *
 * Changements v8.0:
 * - ✅ Mini Cycle Chart (Chart.js sigmoïde avec phases colorées + marqueur position)
 * - ✅ Allocation en texte compact (une ligne, plus de graphique)
 * - ✅ 3 piliers fusionnés en une seule rangée horizontale
 * - ✅ Context Bar remonté en haut de la colonne droite
 * - ❌ Supprimé: Allocation bar graphique (trop d'espace)
 * - ❌ Supprimé: 3 pillar bars séparés (fusionnés)
 *
 * @version 8.0.0
 * @date 2026-02-09
 */

// Debounce timeout
let refreshTimeout = null;

// État du popover d'aide
let helpPopoverState = {
  isOpen: false,
  lastFocusedElement: null
};

// AbortController pour nettoyer les event listeners (prévenir memory leaks)
let helpSystemController = null;

/**
 * Détermine le niveau de couleur d'un score (sémantique positive: plus haut = meilleur)
 */
function getScoreLevel(score) {
  if (score == null || isNaN(score)) return 'medium';
  const s = Number(score);
  if (s >= 75) return 'excellent';
  if (s >= 60) return 'good';
  if (s >= 45) return 'medium';
  if (s >= 30) return 'warning';
  return 'danger';
}

/**
 * Détermine la couleur en format CSS
 */
function getScoreColor(score) {
  const level = getScoreLevel(score);
  switch(level) {
    case 'excellent': return '#10b981'; // green
    case 'good': return '#3b82f6'; // blue
    case 'medium': return '#f59e0b'; // amber
    case 'warning': return '#ef4444'; // red
    case 'danger': return '#991b1b'; // dark red
    default: return '#6b7280'; // gray
  }
}

/**
 * Génère un gradient progressif rouge → vert pour la barre principale
 * Utilise un dégradé continu basé sur le score actuel
 */
function getGradientForScore(score) {
  // Dégradé global rouge → orange → jaune → vert (fond de la track)
  const baseGradient = 'linear-gradient(90deg, ' +
    '#991b1b 0%, ' +      // 0%: Rouge foncé
    '#dc2626 15%, ' +     // 15%: Rouge
    '#ef4444 30%, ' +     // 30%: Rouge vif
    '#f97316 40%, ' +     // 40%: Orange
    '#f59e0b 50%, ' +     // 50%: Jaune-orange
    '#fbbf24 60%, ' +     // 60%: Jaune
    '#84cc16 70%, ' +     // 70%: Vert-jaune
    '#22c55e 80%, ' +     // 80%: Vert clair
    '#10b981 90%, ' +     // 90%: Vert
    '#059669 100%)';      // 100%: Vert vif

  // Calculer la couleur de fin basée sur le score
  let endColor;
  if (score <= 30) {
    endColor = '#dc2626'; // Rouge
  } else if (score <= 45) {
    endColor = '#f97316'; // Orange
  } else if (score <= 60) {
    endColor = '#fbbf24'; // Jaune
  } else if (score <= 75) {
    endColor = '#22c55e'; // Vert clair
  } else {
    endColor = '#10b981'; // Vert vif
  }

  // Gradient de la barre remplie (du rouge au score actuel)
  const fillGradient = 'linear-gradient(90deg, ' +
    '#991b1b 0%, ' +
    '#dc2626 15%, ' +
    '#ef4444 30%, ' +
    '#f97316 40%, ' +
    '#f59e0b 50%, ' +
    '#fbbf24 60%, ' +
    '#84cc16 70%, ' +
    '#22c55e 80%, ' +
    '#10b981 90%, ' +
    `${endColor} 100%)`;

  return fillGradient;
}

/**
 * Helper: valeur sûre (fallback 0)
 */
function _safe(val) {
  return (typeof val === 'number' && Number.isFinite(val)) ? val : 0;
}

/**
 * Helper: arrondi à N décimales
 */
function _round(val, decimals = 1) {
  const v = _safe(val);
  return Number(v.toFixed(decimals));
}

/**
 * Calcule le texte du niveau
 */
function getLevelText(score) {
  const level = getScoreLevel(score);
  switch(level) {
    case 'excellent': return 'Excellent';
    case 'good': return 'Good';
    case 'medium': return 'Average';
    case 'warning': return 'Low';
    case 'danger': return 'Critical';
    default: return 'N/A';
  }
}

/**
 * Calcule la trend pour affichage avec sigma
 */
function computeTrendInfo(history) {
  const arr = Array.isArray(history) ? history.map(h => (h?.di ?? h) ?? 0) : [];
  if (arr.length < 2) return { delta: 0, trend: '→', color: 'neutral', sigma: 0, state: 'N/A' };

  const recent = arr.slice(-7);
  const delta = recent[recent.length - 1] - recent[0];

  // Calcul sigma (volatilité)
  const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
  const variance = recent.reduce((a, b) => a + (b - avg) ** 2, 0) / recent.length;
  const sigma = Math.sqrt(variance);

  // État
  let state = 'Stable';
  if (delta > 1) state = 'Haussier';
  else if (delta < -1) state = 'Baissier';

  return {
    delta: _round(delta, 1),
    trend: delta > 0 ? '↗' : delta < 0 ? '↘' : '→',
    color: delta > 1 ? 'positive' : delta < -1 ? 'negative' : 'neutral',
    sigma: _round(sigma, 1),
    state
  };
}

/**
 * Calcule les contributions relatives (w×s)/Σ
 */
function calculateRelativeContributions(weights, scores) {
  const epsilon = 1e-6;

  const raw = {
    cycle: (weights.cycle || 0) * (scores.cycle || 0),
    onchain: (weights.onchain || 0) * (scores.onchain || 0),
    risk: (weights.risk || 0) * (scores.risk || 0)
  };

  const sum = Object.values(raw).reduce((a, b) => a + b, 0) || epsilon;

  return {
    cycle: (raw.cycle / sum) * 100,
    onchain: (raw.onchain / sum) * 100,
    risk: (raw.risk / sum) * 100,
    raw: raw
  };
}

/**
 * Génère mini sparkline SVG compact
 */
function renderMiniSparkline(series, width = 60, height = 16) {
  if (!Array.isArray(series) || series.length < 2) {
    return '<span class="no-data">—</span>';
  }

  const validSeries = series.filter(v => typeof v === 'number' && Number.isFinite(v));
  if (validSeries.length < 2) {
    return '<span class="no-data">—</span>';
  }

  const min = Math.min(...validSeries);
  const max = Math.max(...validSeries);
  const span = (max - min) || 1;
  const px = (i) => (i / (validSeries.length - 1)) * (width - 2) + 1;
  const py = (v) => height - ((v - min) / span) * (height - 2) - 1;
  const d = validSeries.map((v, i) => `${i === 0 ? 'M' : 'L'} ${px(i).toFixed(1)} ${py(v).toFixed(1)}`).join(' ');

  return `
    <svg class="mini-spark" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
      <path d="${d}" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.6"/>
    </svg>
  `;
}

/**
 * Génère la section complète Scores + Contributions (Option 2: Barre Annotée)
 */
function renderScoresAndContributions(scores, contributions) {
  const items = [
    { key: 'cycle', label: 'Cycle', pct: contributions.cycle },
    { key: 'onchain', label: 'OnCh', pct: contributions.onchain },
    { key: 'risk', label: 'Risk', pct: contributions.risk }
  ];

  return `
    <div class="scores-contrib-annotated">
      <div class="contrib-title">WEIGHTS</div>

      <!-- Labels au-dessus de la barre -->
      <div class="contrib-labels-row">
        ${items.map(item => `
          <div class="contrib-label" style="width: ${item.pct}%;">
            <span class="label-name">${item.label}</span>
            <span class="label-pct">${item.pct.toFixed(0)}%</span>
          </div>
        `).join('')}
      </div>

      <!-- Barre empilée -->
      <div class="contrib-bar-stacked">
        <div class="contrib-seg cycle" style="width: ${contributions.cycle}%;"></div>
        <div class="contrib-seg onchain" style="width: ${contributions.onchain}%;"></div>
        <div class="contrib-seg risk" style="width: ${contributions.risk}%;"></div>
      </div>
    </div>
  `;
}

/**
 * Génère la recommandation actionnable basée sur le DI et les piliers
 */
function renderRecommendation(score, meta, scores = {}) {
  let icon = '💡';
  let title = 'Position neutre';
  let action = 'Monitoring recommended';
  let details = '';
  let colorClass = 'neutral';

  // Extraire les scores des piliers
  const cycle = scores.cycle || 0;
  const onchain = scores.onchain || 0;
  const risk = scores.risk || 0;

  // Détection d'alertes spécifiques
  const criticalOnchain = onchain < 30;
  const lowRisk = risk < 40;
  const strongCycle = cycle >= 70;

  // Logique de recommandation basée sur le DI global
  if (score >= 75) {
    icon = '🚀';
    title = 'Excellent timing';
    colorClass = 'bullish';

    if (strongCycle && onchain >= 50) {
      action = 'Allocate 15-20% to risky assets';
      details = 'Cycle expansion + On-Chain favorable → Accumulation opportunity';
    } else if (criticalOnchain) {
      action = 'Allocate with caution (10-15%)';
      details = 'Despite high DI, weak on-chain signals → Vigilance required';
    } else {
      action = 'Gradually increase risk exposure';
      details = 'Favorable conditions → Reduce stables to 10-15%';
    }

  } else if (score >= 60) {
    icon = '✅';
    title = 'Favorable position';
    colorClass = 'positive';

    if (cycle >= 60 && risk >= 50) {
      action = 'Maintain current allocation';
      details = `Cycle ${Math.round(cycle)} + Risk ${Math.round(risk)} → Stable balance`;
    } else if (criticalOnchain) {
      action = 'Hold but monitor on-chain';
      details = 'Degraded on-chain signals → Prepare adjustments if needed';
    } else {
      action = 'Maintain allocation, minor adjustments OK';
      details = 'Solid position → Opportunistic rebalancing possible';
    }

  } else if (score >= 45) {
    icon = '⚠️';
    title = 'Mixed position';
    colorClass = 'warning';

    if (lowRisk) {
      action = 'Reduce exposure, secure gains';
      details = `Low risk (${Math.round(risk)}) → Increase stables to 25-30%`;
    } else if (criticalOnchain) {
      action = 'Prioritize absolute caution';
      details = 'Critical on-chain signals → Avoid new risky positions';
    } else {
      action = 'Wait and enhanced monitoring';
      details = 'Uncertain context → Avoid major changes';
    }

  } else if (score >= 30) {
    icon = '🛡️';
    title = 'Unfavorable position';
    colorClass = 'defensive';

    if (lowRisk && criticalOnchain) {
      action = 'Reduce exposure immediately';
      details = 'Risk + On-Chain weak → Secure 40-50% in stables';
    } else {
      action = 'Reduce risky assets to 30-40%';
      details = 'Degraded conditions → Protect capital';
    }

  } else {
    icon = '🚨';
    title = 'ALERT - Critical position';
    colorClass = 'critical';

    action = 'Secure the portfolio immediately';
    details = `DI ${score} → Move 60-70% to stables, reduce leverage`;
  }

  return `
    <div class="di-recommendation ${colorClass}">
      <div class="reco-content">
        <div class="reco-header">
          <span class="reco-icon">${icon}</span>
          <span class="reco-title">${title}</span>
        </div>
        <div class="reco-action">${action}</div>
        ${details ? `<div class="reco-details">${details}</div>` : ''}
      </div>
    </div>
  `;
}

/**
 * Génère les métadonnées utiles
 */
function renderMetadata(meta) {
  const confidence = meta.confidence ? `${Math.round(meta.confidence * 100)}%` : 'N/A';
  const mode = meta.mode || 'Standard';
  const source = meta.source || 'N/A';
  const timestamp = meta.timestamp || meta.last_update;

  // Blended Score (régime) - affiché pour clarifier les recommandations
  const blendedScore = meta.blended_score ?? meta.regime_score ?? null;
  const blendedDisplay = blendedScore != null ? Math.round(blendedScore) : '--';

  let freshness = 'N/A';
  if (timestamp) {
    try {
      const diff = Date.now() - new Date(timestamp).getTime();
      const minutes = Math.floor(diff / 60000);
      if (minutes < 1) freshness = 'À l\'instant';
      else if (minutes < 60) freshness = `Il y a ${minutes}min`;
      else if (minutes < 1440) freshness = `Il y a ${Math.floor(minutes / 60)}h`;
      else freshness = `Il y a ${Math.floor(minutes / 1440)}j`;
    } catch (e) {
      freshness = 'Inconnu';
    }
  }

  // Détection des overrides actifs
  const overrides = [];
  const fearGreed = meta.sentiment_fg != null ? meta.sentiment_fg : null;
  const contradiction = meta.contradiction != null ? meta.contradiction : null;

  if (fearGreed != null && fearGreed < 25) {
    overrides.push(`🚨 ML Sentiment Extrême (${fearGreed})`);
  }
  if (contradiction != null && contradiction > 0.5) {
    overrides.push(`⚠️ Contradiction (${Math.round(contradiction * 100)}%)`);
  }
  // Override #4: Macro Stress (VIX/DXY) - Feb 2026
  if (meta.macro_stress) {
    const macroDetails = [];
    if (meta.vix_stress) macroDetails.push(`VIX ${meta.vix_value?.toFixed(1)}`);
    if (meta.dxy_stress) macroDetails.push(`DXY +${meta.dxy_change_30d?.toFixed(1)}%`);
    overrides.push(`🌍 Macro Stress (${macroDetails.join(', ') || 'VIX/DXY'})`);
  }

  const overrideBadge = overrides.length > 0
    ? `<div class="meta-row meta-override">
        <span class="meta-label">⚡ Override</span>
        <span class="meta-value meta-override-value">${overrides.join(', ')}</span>
      </div>`
    : '';

  return `
    <div class="di-metadata">
      <div class="meta-row">
        <span class="meta-label">Confiance</span>
        <span class="meta-value">${confidence}</span>
      </div>
      <div class="meta-row">
        <span class="meta-label">Mode</span>
        <span class="meta-value">${mode}</span>
      </div>
      <div class="meta-row">
        <span class="meta-label">Blended</span>
        <span class="meta-value" title="Regime score used in recommendations">${blendedDisplay}</span>
      </div>
      <div class="meta-row">
        <span class="meta-label">Updated</span>
        <span class="meta-value">${freshness}</span>
      </div>
      ${overrideBadge}
    </div>
  `;
}

/**
 * Génère la colonne gauche avec score principal
 */
function getLeftParts(data) {
  const score = Math.round(data.di);
  const gradient = getGradientForScore(score);
  const levelText = getLevelText(score);
  const m = data.meta || {};
  const s = data.scores || {};

  const contributions = calculateRelativeContributions(data.weights || {}, s);

  return {
    scoreBlock: `
      <div class="di-score-block">
        <div class="di-title-row">
          <span class="di-title">DECISION INDEX</span>
          <button class="di-help-btn" aria-label="Aide" type="button">?</button>
        </div>
        <div class="di-score-section">
          <div class="di-score-big">${score}</div>
          <div class="di-score-label">${levelText}</div>
        </div>
        <div class="di-main-bar-compact">
          <div class="di-bar-track">
            <div class="di-bar-fill" style="width: ${score}%; background: ${gradient};">
              <div class="di-bar-glow"></div>
            </div>
            <div class="di-bar-segments">
              ${Array(10).fill(0).map((_, i) =>
                `<div class="seg ${(i+1)*10 <= score ? 'on' : ''}"></div>`
              ).join('')}
            </div>
          </div>
          <div class="di-bar-labels">
            <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
          </div>
        </div>
      </div>`,
    weights: renderScoresAndContributions(s, contributions),
    metadata: renderMetadata(m),
    recommendation: renderRecommendation(score, m, s)
  };
}

/**
 * Génère le footer stats global (à placer en bas du panneau complet)
 */
// renderGlobalFooterStats removed in v8.0 — merged into renderMetricsBar

/**
 * Génère une barre de pilier compacte avec phases visuelles pour le Cycle
 */
function renderCompactPillarBar(label, icon, value, subtext, confidence, color, meta = {}) {
  const percentage = Math.min(100, Math.max(0, value));
  const barColor = color || getScoreColor(value);

  // Si c'est le pilier Cycle, afficher les phases visuelles
  if (label === 'Cycle' && meta.cycle_months != null) {
    return renderCyclePillarWithPhases(value, subtext, confidence, meta);
  }

  // Barre standard pour les autres piliers
  return `
    <div class="pillar-bar-compact">
      <div class="pillar-header">
        <div class="pillar-label">
          <span class="pillar-icon">${icon}</span>
          <span class="pillar-name">${label}</span>
          ${confidence ? `<span class="conf-chip">${confidence}%</span>` : ''}
        </div>
        <div class="pillar-score">${Math.round(value)}</div>
      </div>
      <div class="pillar-track">
        <div class="pillar-fill" style="width: ${percentage}%; background: ${barColor};"></div>
      </div>
      ${subtext ? `<div class="pillar-sub">${subtext}</div>` : ''}
    </div>
  `;
}

/**
 * Génère une barre de cycle avec phases visuelles
 */
function renderCyclePillarWithPhases(value, subtext, confidence, meta) {
  const months = meta.cycle_months || 0;
  const totalMonths = 48;
  const positionPercent = Math.min(100, (months / totalMonths) * 100);

  // Phases du cycle (48 mois)
  const phases = [
    { name: 'Acc', start: 0, end: 6, color: '#f59e0b', emoji: '🟡' },
    { name: 'Bull', start: 6, end: 18, color: '#10b981', emoji: '🟢' },
    { name: 'Peak', start: 18, end: 24, color: '#8b5cf6', emoji: '🟣' },
    { name: 'Bear', start: 24, end: 36, color: '#dc2626', emoji: '🔴' },
    { name: 'Pre', start: 36, end: 48, color: '#6b7280', emoji: '⚫' }
  ];

  // Déterminer la phase actuelle
  const currentPhase = phases.find(p => months >= p.start && months < p.end) || phases[0];

  // Générer les segments de phases
  const phaseSegments = phases.map(p => {
    const widthPct = ((p.end - p.start) / totalMonths) * 100;
    const isActive = p === currentPhase;
    return `
      <div class="cycle-phase-mini"
           style="width: ${widthPct}%; background: ${p.color}; opacity: ${isActive ? 1 : 0.4};"
           title="${p.name}: ${p.start}-${p.end}m"></div>
    `;
  }).join('');

  return `
    <div class="pillar-bar-compact pillar-cycle-visual">
      <div class="pillar-header">
        <div class="pillar-label">
          <span class="pillar-icon">🔄</span>
          <span class="pillar-name">Cycle</span>
          ${confidence ? `<span class="conf-chip">${confidence}%</span>` : ''}
        </div>
        <div class="pillar-score">${Math.round(value)}</div>
      </div>
      <div class="pillar-track pillar-track-cycle">
        <!-- Phases en arrière-plan -->
        <div class="cycle-phases-bg">
          ${phaseSegments}
        </div>
        <!-- Marqueur de position -->
        <div class="cycle-position-mini" style="left: ${positionPercent}%;">
          <div class="cycle-marker-mini"></div>
        </div>
      </div>
      ${subtext ? `<div class="pillar-sub">${currentPhase.emoji} ${subtext}</div>` : ''}
    </div>
  `;
}

/**
 * Génère une barre horizontale compacte pour l'allocation (remplace le donut)
 * @param {Object} allocation - { btc: %, eth: %, stables: %, alts: % }
 */
function renderAllocationBar(allocation) {
  if (!allocation || typeof allocation !== 'object') {
    return '<div class="alloc-bar-placeholder">Allocation non disponible</div>';
  }

  const btc = allocation.btc || 0;
  const eth = allocation.eth || 0;
  const stables = allocation.stables || 0;
  const alts = allocation.alts || 0;
  const total = btc + eth + stables + alts;

  if (total === 0) {
    return '<div class="alloc-bar-placeholder">No data</div>';
  }

  // Normaliser à 100%
  const normalize = (v) => (v / total) * 100;
  const segments = [
    { name: 'BTC', pct: normalize(btc), color: '#f7931a' },
    { name: 'ETH', pct: normalize(eth), color: '#627eea' },
    { name: 'Stables', pct: normalize(stables), color: '#26a17b' },
    { name: 'Alts', pct: normalize(alts), color: '#8b5cf6' }
  ].filter(s => s.pct > 0);

  // Barre empilée horizontale
  const barSegments = segments.map(seg => `
    <div class="alloc-seg" style="width: ${seg.pct}%; background: ${seg.color};" title="${seg.name}: ${seg.pct.toFixed(0)}%">
      <span class="alloc-seg-label">${seg.pct >= 12 ? `${seg.pct.toFixed(0)}%` : ''}</span>
    </div>
  `).join('');

  // Légende inline compacte
  const legend = segments.map(seg =>
    `<span class="alloc-leg-inline" style="--c:${seg.color}">
      <span class="alloc-dot-sm"></span>${seg.name} ${seg.pct.toFixed(0)}%
    </span>`
  ).join('');

  return `
    <div class="alloc-bar-container">
      <div class="alloc-bar-title">ALLOCATION</div>
      <div class="alloc-bar-track">
        ${barSegments}
      </div>
      <div class="alloc-legend-inline">${legend}</div>
    </div>
  `;
}

// ═══════════════════════════════════════════════════════════
// MINI CYCLE CHART (v8.0)
// ═══════════════════════════════════════════════════════════

// Chart.js instance reference for cleanup
let _diMiniCycleChart = null;

/**
 * Compute cycle score inline (same sigmoid as cycle-navigator.js)
 * Reads calibrated params from localStorage, falls back to defaults
 */
function _computeCycleScore(monthsAfterHalving, params) {
  if (typeof monthsAfterHalving !== 'number' || monthsAfterHalving < 0) return 50;
  const m48 = monthsAfterHalving % 48;
  const { m_rise_center, m_fall_center, k_rise, k_fall, p_shape } = params;
  const rise = 1 / (1 + Math.exp(-k_rise * (m48 - m_rise_center)));
  const fall = 1 / (1 + Math.exp(-k_fall * (m_fall_center - m48)));
  return Math.max(0, Math.min(100, Math.pow(rise * fall, p_shape) * 100));
}

/**
 * Load calibrated cycle params from localStorage or use defaults
 */
function _getLocalCycleParams() {
  const defaults = { m_rise_center: 5.0, m_fall_center: 24.0, k_rise: 0.8, k_fall: 1.2, p_shape: 1.15 };
  try {
    const saved = localStorage.getItem('bitcoin_cycle_params');
    if (saved) {
      const data = JSON.parse(saved);
      if (data?.params && data.version && data.version.startsWith('2.')) {
        return { ...defaults, ...data.params };
      }
    }
  } catch (_) { /* ignore */ }
  return defaults;
}

/**
 * Render canvas container for the mini cycle chart
 */
function renderMiniCycleChartContainer() {
  return `
    <div class="di-mini-cycle-container">
      <div class="di-mini-cycle-title">CYCLE POSITION</div>
      <div class="di-mini-cycle-wrapper">
        <canvas id="di-mini-cycle-canvas"></canvas>
      </div>
    </div>
  `;
}

/**
 * Mount the mini cycle chart using Chart.js (call AFTER DOM insertion)
 */
function mountMiniCycleChart(cycleMonths) {
  const canvas = document.getElementById('di-mini-cycle-canvas');
  if (!canvas) return;

  // Guard: Chart.js not loaded
  if (typeof Chart === 'undefined') {
    canvas.parentElement.innerHTML = '<div style="text-align:center;color:rgba(148,163,184,0.5);font-size:0.7rem;padding:1rem;">Chart.js not available</div>';
    return;
  }

  // Destroy previous instance
  if (_diMiniCycleChart) {
    _diMiniCycleChart.destroy();
    _diMiniCycleChart = null;
  }
  const existing = Chart.getChart(canvas);
  if (existing) existing.destroy();

  const params = _getLocalCycleParams();
  const currentMonths = typeof cycleMonths === 'number' && cycleMonths > 0 ? cycleMonths : 22;
  const currentScore = _computeCycleScore(currentMonths, params);

  // Generate curve data (0-48 months, step 0.5)
  const curveData = [];
  for (let m = 0; m <= 48; m += 0.5) {
    curveData.push({ x: m, y: _computeCycleScore(m, params) });
  }

  // Phase definitions
  const phases = [
    { name: 'Acc',  start: 0,  end: 6,  color: 'rgba(245,158,11,0.12)', border: '#f59e0b' },
    { name: 'Bull', start: 6,  end: 18, color: 'rgba(16,185,129,0.12)',  border: '#10b981' },
    { name: 'Peak', start: 18, end: 24, color: 'rgba(139,92,246,0.12)',  border: '#8b5cf6' },
    { name: 'Bear', start: 24, end: 36, color: 'rgba(220,38,38,0.10)',   border: '#dc2626' },
    { name: 'Pre',  start: 36, end: 48, color: 'rgba(107,114,128,0.08)', border: '#6b7280' }
  ];

  // Check if annotation plugin is available
  const hasAnnotationPlugin = !!(window.ChartAnnotation || Chart.registry?.plugins?.get('annotation'));

  // Build annotation config (only if plugin available)
  const annotationConfig = {};
  if (hasAnnotationPlugin) {
    const annotations = {};
    phases.forEach((p, i) => {
      annotations[`phase${i}`] = {
        type: 'box',
        xMin: p.start,
        xMax: p.end,
        yMin: 0,
        yMax: 100,
        backgroundColor: p.color,
        borderWidth: 0
      };
      annotations[`phaseLabel${i}`] = {
        type: 'label',
        xValue: (p.start + p.end) / 2,
        yValue: 96,
        content: p.name,
        color: p.border,
        font: { size: 9, weight: '600' },
        backgroundColor: 'transparent'
      };
    });

    // Current position vertical line
    annotations.currentLine = {
      type: 'line',
      xMin: currentMonths,
      xMax: currentMonths,
      borderColor: '#ef4444',
      borderWidth: 2,
      borderDash: [4, 3],
      label: {
        display: true,
        content: 'NOW',
        position: 'start',
        backgroundColor: 'rgba(239,68,68,0.9)',
        color: '#fff',
        font: { size: 8, weight: '700' },
        padding: { x: 4, y: 2 },
        borderRadius: 3
      }
    };

    // Current position point
    annotations.currentPoint = {
      type: 'point',
      xValue: currentMonths,
      yValue: currentScore,
      backgroundColor: '#ef4444',
      borderColor: '#fff',
      borderWidth: 2,
      radius: 5
    };

    annotationConfig.annotation = { annotations };
  }

  // Add a visible current-position point on the curve (works without annotation plugin)
  const currentPointData = [{ x: currentMonths, y: currentScore }];

  _diMiniCycleChart = new Chart(canvas, {
    type: 'line',
    data: {
      datasets: [
        {
          data: curveData,
          borderColor: '#10b981',
          borderWidth: 2.5,
          fill: {
            target: 'origin',
            above: 'rgba(16,185,129,0.08)'
          },
          pointRadius: 0,
          pointHoverRadius: 0,
          tension: 0.4
        },
        // Current position dot (always visible even without annotation plugin)
        {
          data: currentPointData,
          borderColor: '#ef4444',
          backgroundColor: '#ef4444',
          pointRadius: 6,
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointHoverRadius: 8,
          showLine: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600, easing: 'easeOutQuart' },
      layout: { padding: { top: 4, right: 8, bottom: 0, left: 4 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: true,
          filter: (item) => item.datasetIndex === 1, // Only on current position dot
          callbacks: {
            title: () => 'Current Position',
            label: (ctx) => `Score: ${Math.round(ctx.parsed.y)} (${Math.round(ctx.parsed.x)}m)`
          }
        },
        datalabels: { display: false },
        ...annotationConfig
      },
      scales: {
        x: {
          type: 'linear',
          min: 0,
          max: 48,
          grid: { color: 'rgba(148,163,184,0.06)', drawTicks: false },
          border: { display: false },
          ticks: {
            stepSize: 6,
            color: 'rgba(148,163,184,0.4)',
            font: { size: 8 },
            callback: v => v === 0 ? 'H' : `${v}m`,
            padding: 2
          }
        },
        y: {
          min: 0,
          max: 100,
          grid: { color: 'rgba(148,163,184,0.06)', drawTicks: false },
          border: { display: false },
          ticks: {
            stepSize: 50,
            color: 'rgba(148,163,184,0.4)',
            font: { size: 8 },
            padding: 2
          }
        }
      }
    }
  });
}

// ═══════════════════════════════════════════════════════════
// ALLOCATION TEXT COMPACT (v8.0)
// ═══════════════════════════════════════════════════════════

/**
 * Render allocation as a compact text line (replaces the allocation bar)
 */
function renderAllocationText(allocation) {
  if (!allocation || typeof allocation !== 'object') {
    return '';
  }

  const btc = allocation.btc || 0;
  const eth = allocation.eth || 0;
  const stables = allocation.stables || 0;
  const alts = allocation.alts || 0;
  const total = btc + eth + stables + alts;
  if (total === 0) return '';

  const norm = (v) => Math.round((v / total) * 100);
  const items = [
    { name: 'BTC', pct: norm(btc), color: '#f7931a' },
    { name: 'ETH', pct: norm(eth), color: '#627eea' },
    { name: 'Stables', pct: norm(stables), color: '#26a17b' },
    { name: 'Alts', pct: norm(alts), color: '#8b5cf6' }
  ].filter(s => s.pct > 0);

  const text = items.map(s =>
    `<span class="alloc-text-item"><span class="alloc-text-dot" style="background:${s.color};"></span>${s.name} ${s.pct}%</span>`
  ).join('<span class="alloc-text-sep">·</span>');

  return `
    <div class="di-alloc-text">
      <span class="alloc-text-label">ALLOCATION</span>
      <div class="alloc-text-values">${text}</div>
    </div>
  `;
}

// ═══════════════════════════════════════════════════════════
// FUSED PILLARS (v8.0)
// ═══════════════════════════════════════════════════════════

/**
 * Render the 3 pillars (Cycle, On-Chain, Risk) as a single compact row
 */
function renderFusedPillars(scores, meta) {
  const cycleVal = Math.round(scores.cycle || 0);
  const onchainVal = Math.round(scores.onchain || 0);
  const riskVal = Math.round(scores.risk || 0);

  const items = [
    { label: 'CYCLE', value: cycleVal, color: '#3b82f6' },
    { label: 'ONCHAIN', value: onchainVal, color: '#8b5cf6' },
    { label: 'RISK', value: riskVal, color: '#ef4444' }
  ];

  return `
    <div class="di-pillars-fused">
      ${items.map(item => `
        <div class="pillar-fused-col">
          <span class="pillar-fused-label">${item.label}</span>
          <div class="pillar-fused-score" style="color: ${getScoreColor(item.value)};">${item.value}</div>
          <div class="pillar-fused-track">
            <div class="pillar-fused-fill" style="width:${Math.min(100, item.value)}%;background:${item.color};"></div>
          </div>
        </div>
      `).join('<div class="pillar-fused-divider"></div>')}
    </div>
  `;
}

/**
 * Génère les Key Metrics (VaR + Sharpe + Risk Budget)
 */
/**
 * Render unified metrics bar (Active Factors + Key Metrics merged)
 * Full-width bar below the 2-column grid
 */
function renderMetricsBar(meta, scores = {}) {
  const leftItems = [];

  // ML Sentiment
  const sentimentFG = meta.sentiment_fg || '—';
  const sentimentColor = typeof sentimentFG === 'number'
    ? (sentimentFG >= 70 ? '#ef4444' : sentimentFG >= 30 ? '#f59e0b' : '#10b981') : '#6b7280';
  leftItems.push({ label: 'Sentiment', value: `${sentimentFG}`, color: sentimentColor });

  // Contradiction
  const contradiction = meta.contradiction ?? null;
  if (contradiction != null) {
    const cPct = Math.round(contradiction * 100);
    const cColor = cPct > 50 ? '#ef4444' : cPct > 25 ? '#f59e0b' : '#10b981';
    leftItems.push({ label: 'Contradiction', value: `${cPct}%`, color: cColor });
  }

  // Macro Stress
  const macroStress = meta.macro_stress ?? false;
  const macroVal = meta.macro_penalty ?? null;
  if (macroStress || macroVal) {
    const details = [];
    if (meta.vix_value) details.push(`VIX ${meta.vix_value.toFixed(0)}`);
    if (meta.dxy_change_30d) details.push(`DXY ${meta.dxy_change_30d > 0 ? '+' : ''}${meta.dxy_change_30d.toFixed(1)}%`);
    const mColor = macroStress ? '#ef4444' : '#10b981';
    leftItems.push({
      label: 'Macro',
      value: macroStress ? (details.length ? details.join(' ') : 'Stress') : 'Normal',
      color: mColor
    });
  } else {
    leftItems.push({ label: 'Macro', value: 'Normal', color: '#10b981' });
  }

  // VaR 95%
  const var95 = meta.risk_var95 ?? meta.var95 ?? null;
  const var95Display = var95 != null ? `${(Math.abs(var95) * 100).toFixed(2)}%` : '--';
  const varColor = var95 != null
    ? (Math.abs(var95) > 0.05 ? '#ef4444' : Math.abs(var95) > 0.03 ? '#f59e0b' : '#10b981')
    : '#6b7280';
  leftItems.push({ label: 'VaR 95%', value: var95Display, color: varColor });

  // Sharpe Ratio
  const sharpe = meta.sharpe ?? meta.sharpe_ratio ?? meta.risk_sharpe ?? null;
  const sharpeDisplay = sharpe != null ? sharpe.toFixed(2) : '--';
  const sharpeColor = sharpe != null
    ? (sharpe >= 1.5 ? '#10b981' : sharpe >= 0.5 ? '#f59e0b' : '#ef4444')
    : '#6b7280';
  leftItems.push({ label: 'Sharpe', value: sharpeDisplay, color: sharpeColor });

  // Risk Budget
  const riskBudget = meta.risk_budget ?? null;
  const riskyPct = riskBudget?.risky ?? null;
  const riskBudgetDisplay = riskyPct != null ? `${Math.round(riskyPct)}%` : '--';
  const riskBudgetColor = riskyPct != null
    ? (riskyPct >= 60 ? '#ef4444' : riskyPct >= 40 ? '#f59e0b' : '#10b981')
    : '#6b7280';
  leftItems.push({ label: 'Risk %', value: riskBudgetDisplay, color: riskBudgetColor });

  // Right side: Status + Source
  const statusClass = meta.live ? 'live' : 'offline';
  const statusText = meta.live ? 'Live' : 'Off';
  const source = meta.source || 'N/A';

  return `
    <div class="di-metrics-bar">
      <div class="metrics-bar-left">
        ${leftItems.map(item => `
          <div class="metrics-bar-item">
            <span class="metrics-bar-label">${item.label}</span>
            <span class="metrics-bar-value" style="color:${item.color};">${item.value}</span>
          </div>
        `).join('')}
      </div>
      <div class="metrics-bar-right">
        <div class="metrics-bar-item">
          <span class="metrics-bar-label">Status</span>
          <span class="metrics-bar-value ${statusClass}">${statusText}</span>
        </div>
        <div class="metrics-bar-item">
          <span class="metrics-bar-label">Source</span>
          <span class="metrics-bar-value">${source}</span>
        </div>
      </div>
    </div>
  `;
}

/**
 * Génère la Quick Context Bar (Régime, Phase, Vol, Cycle position)
 */
function renderQuickContextBar(meta) {
  const regime = meta.phase || meta.regime || 'Neutral';
  const regimeEmoji = meta.regime_emoji || '📊';

  // Phase d'allocation (depuis cycle score)
  const cyclePhase = meta.cycle_phase || 'Unknown';

  // Volatilité annualisée (Feb 2026: corrigé pour afficher vraie volatilité, pas VaR)
  const vol = meta.volatility_annualized != null
    ? `${(meta.volatility_annualized * 100).toFixed(1)}%`
    : (meta.volatility ? `${(meta.volatility * 100).toFixed(1)}%` : '--');

  // Position dans le cycle - format amélioré
  const months = meta.cycle_months;
  const cyclePos = months
    ? (months > 18 ? `${Math.round(months)}m+` : `${Math.round(months)}m`)
    : '--';
  const cycleTooltip = months
    ? `${Math.round(months)} mois depuis halving (cycle typique: 18 mois)`
    : 'Position dans le cycle Bitcoin';

  // Couleur du régime
  const regimeColor = {
    'bear market': '#dc2626',
    'correction': '#ea580c',
    'bull market': '#22c55e',
    'expansion': '#3b82f6',
    'neutral': '#6b7280',
    'bearish': '#ef4444',
    'risk_off': '#ef4444'
  }[regime.toLowerCase()] || '#6b7280';

  return `
    <div class="di-context-bar">
      <div class="ctx-item">
        <span class="ctx-label">Conditions</span>
        <span class="ctx-value" style="color: ${regimeColor};">${regimeEmoji} ${regime}</span>
      </div>
      <div class="ctx-divider"></div>
      <div class="ctx-item" data-ctx="phase">
        <span class="ctx-label">Phase</span>
        <span class="ctx-value" title="${cyclePhase}">${cyclePhase}</span>
      </div>
      <div class="ctx-divider"></div>
      <div class="ctx-item">
        <span class="ctx-label">Vol</span>
        <span class="ctx-value">${vol}</span>
      </div>
      <div class="ctx-divider"></div>
      <div class="ctx-item" data-ctx="cycle">
        <span class="ctx-label">Cycle</span>
        <span class="ctx-value" title="${cycleTooltip}">${cyclePos}</span>
      </div>
    </div>
  `;
}

/**
 * Génère la colonne droite v8.0:
 * Context Bar → Mini Cycle Chart → Allocation texte + Pillars fusionnés → Active Factors + Key Metrics
 */
function getRightParts(data) {
  const s = data.scores || {};
  const m = data.meta || {};

  return {
    contextBar: renderQuickContextBar(m),
    cycleChart: renderMiniCycleChartContainer(),
    allocation: data.allocation ? renderAllocationText(data.allocation) : '',
    fusedPillars: renderFusedPillars(s, m)
  };
}

/**
 * Génère le contenu d'aide
 */
function renderHelpContent() {
  return `
    <div class="di-help-popup" style="display: none;" role="dialog" aria-labelledby="di-help-title" aria-modal="true">
      <div class="di-help-content">
        <div class="di-help-header">
          <h3 id="di-help-title">📊 Decision Index</h3>
          <button class="di-help-close" aria-label="Close" type="button">×</button>
        </div>
        <div class="di-help-body">
          <p><strong>Decision Index (DI) - Strategic Score</strong><br>
          Continuous score <strong>0-100</strong> computed by weighted pillars:<br>
          <code>DI = (Cycle × w₁ + OnChain × w₂ + Risk × w₃) × phase_factor</code><br>
          <br>
          ⚠️ <strong>Important:</strong> DI IS a weighted sum!<br>
          • Adaptive weights based on context (strong cycle → boost wCycle)<br>
          • Adjustment by market phase (bullish/bearish)</p>

          <p><strong>Scale</strong><br>
          75+ = Favorable conditions (aggressive allocation OK)<br>
          60-74 = Neutral (hold position)<br>
          45-59 = Cautious (enhanced monitoring)<br>
          30-44 = Defensive (reduce exposure)<br>
          &lt;30 = Secure (max stables)</p>

          <p><strong>Pillars (right column)</strong><br>
          🔄 Cycle: Blended CCS (CCS blended with cycle position)<br>
          🔗 On-Chain: Fundamental blockchain metrics<br>
          🛡️ Risk: Portfolio robustness (higher = better)</p>

          <p><strong>Conditions vs Phase vs Regime</strong><br>
          • <strong>Conditions</strong> = Composite market outlook (CCS + On-Chain, without Risk)<br>
          • <strong>Phase</strong> = Applied strategy based on cycle score<br>
          • <strong>Regime</strong> (on Market Regimes page) = ML detection per asset (BTC/ETH/Stock)<br>
          Conditions and Regime may differ: Conditions reflects composite scores, Regime reflects actual drawdown.</p>

          <p><strong>Contributions</strong><br>
          Percentages = weights used to compute DI AND allocation.<br>
          Adaptive weights based on market context:<br>
          • Cycle ≥70 → boost cycle (55-65%)<br>
          • Cycle ≥90 → strong boost (65% cycle, 25% onchain, 10% risk)<br>
          • Contradiction >50% → penalizes OnChain/Risk<br>
          • Phase bullish/bearish → adjusts final score (±5%)</p>

          <p><strong>Smart Recommendation</strong><br>
          Contextual advice based on DI + 3-pillar analysis:<br>
          • 75+ : Allocate towards risk (15-20% stables)<br>
          • 60-74 : Hold allocation, minor adjustments OK<br>
          • 45-59 : Wait and enhanced monitoring<br>
          • 30-44 : Reduce exposure (30-40% risky assets)<br>
          • <30 : Secure immediately (60-70% stables)</p>

          <p><strong>Contextual Adaptations</strong><br>
          • Critical On-Chain → Specific alerts<br>
          • Low Risk → Increased stables recommended<br>
          • Strong Cycle → Accumulation opportunities<br>
          • <strong>ML Extreme Sentiment (<25)</strong> → Defensive override applied</p>

          <p><strong>Active Overrides</strong><br>
          External factors can modify the allocation:<br>
          • ML Sentiment <25 → Forces defensive allocation<br>
          • Contradiction >50% → Penalizes On-Chain/Risk<br>
          • 🌍 Macro Stress (VIX>30 or DXY+5%) → -15 pts penalty on DI<br>
          • Structure Score <50 → +10pts stables</p>

          <p><strong>Metadata</strong><br>
          Confidence: Model certainty level<br>
          Mode: Calculation method (Manual/Standard/Priority)<br>
          Last Update: Data freshness</p>
        </div>
      </div>
    </div>
  `;
}

/**
 * Monte le système d'aide avec cleanup automatique (prévenir memory leaks)
 */
function mountHelpSystem(container) {
  const trigger = container.querySelector('.di-help-btn');
  const popup = container.querySelector('.di-help-popup');
  const closeBtn = container.querySelector('.di-help-close');

  if (!trigger || !popup) return;

  // Cleanup des event listeners précédents
  if (helpSystemController) {
    helpSystemController.abort();
  }

  // Nouveau controller pour gérer tous les listeners de ce panel
  helpSystemController = new AbortController();
  const signal = helpSystemController.signal;

  const toggleHelp = (show) => {
    if (show) {
      helpPopoverState.isOpen = true;
      helpPopoverState.lastFocusedElement = document.activeElement;
      popup.style.display = 'block';
      trigger.setAttribute('aria-expanded', 'true');
      setTimeout(() => popup.classList.add('show'), 10);
    } else {
      helpPopoverState.isOpen = false;
      popup.classList.remove('show');
      trigger.setAttribute('aria-expanded', 'false');
      setTimeout(() => popup.style.display = 'none', 300);
    }
  };

  // Tous les listeners utilisent le même signal pour cleanup automatique
  trigger.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleHelp(!helpPopoverState.isOpen);
  }, { signal });

  if (closeBtn) {
    closeBtn.addEventListener('click', () => toggleHelp(false), { signal });
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && helpPopoverState.isOpen) {
      toggleHelp(false);
    }
  }, { signal });

  document.addEventListener('click', (e) => {
    if (helpPopoverState.isOpen && !popup.contains(e.target) && e.target !== trigger) {
      toggleHelp(false);
    }
  }, { signal });
}

/**
 * Ajoute les styles CSS
 */
function injectStyles() {
  const styleId = 'di-gaming-styles';
  if (document.getElementById(styleId)) return;

  const styles = document.createElement('style');
  styles.id = styleId;
  styles.textContent = `
    /* Container principal 2 colonnes */
    .di-panel-gaming {
      background: linear-gradient(135deg,
        rgba(15, 23, 42, 0.95) 0%,
        rgba(30, 41, 59, 0.95) 100%);
      border: 1px solid rgba(148, 163, 184, 0.1);
      border-radius: 12px;
      padding: 1.25rem;
      backdrop-filter: blur(10px);
      box-shadow:
        0 10px 25px -5px rgba(0, 0, 0, 0.1),
        0 8px 10px -6px rgba(0, 0, 0, 0.1);
      position: relative;
      overflow: hidden;
    }

    /* Layout 3 rangées alignées */
    .di-row {
      display: grid;
      grid-template-columns: 1fr 1.2fr;
      gap: 0.75rem;
      align-items: stretch;
      position: relative;
      z-index: 1;
    }

    .di-row + .di-row {
      margin-top: 0.5rem;
    }

    .di-top-left, .di-top-right {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .di-score-block {
      background: rgba(30, 41, 59, 0.3);
      border: 1px solid rgba(148, 163, 184, 0.08);
      border-radius: 8px;
      padding: 0.5rem 0.625rem;
    }

    .di-title-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .di-title {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: rgba(148, 163, 184, 0.8);
    }

    .di-help-btn {
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: rgba(59, 130, 246, 0.2);
      border: 1px solid rgba(59, 130, 246, 0.3);
      color: #60a5fa;
      font-size: 0.75rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .di-help-btn:hover {
      background: rgba(59, 130, 246, 0.3);
      transform: scale(1.1);
    }

    .di-score-section {
      display: flex;
      align-items: baseline;
      gap: 0.75rem;
      margin: 0.25rem 0;
    }

    .di-score-big {
      font-size: 2.75rem;
      font-weight: 800;
      line-height: 1;
      background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .di-score-label {
      font-size: 0.875rem;
      font-weight: 600;
      color: rgba(148, 163, 184, 0.8);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    /* Barre principale compacte */
    .di-main-bar-compact {
      margin: 0.25rem 0;
    }

    .di-bar-track {
      position: relative;
      height: 24px;
      background: rgba(15, 23, 42, 0.5);
      border-radius: 999px;
      overflow: hidden;
      box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    .di-bar-fill {
      height: 100%;
      position: relative;
      border-radius: 999px;
      transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
      box-shadow:
        0 0 20px rgba(0, 0, 0, 0.3),
        0 0 30px currentColor,
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
      overflow: hidden;
    }

    .di-bar-glow {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 50%;
      background: linear-gradient(180deg,
        rgba(255, 255, 255, 0.4) 0%,
        rgba(255, 255, 255, 0.1) 50%,
        transparent 100%);
      border-radius: 999px 999px 0 0;
    }

    /* Animation de brillance subtile */
    @keyframes shine {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(100%); }
    }

    .di-bar-fill::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 30%;
      height: 100%;
      background: linear-gradient(90deg,
        transparent 0%,
        rgba(255, 255, 255, 0.3) 50%,
        transparent 100%);
      animation: shine 3s ease-in-out infinite;
      pointer-events: none;
    }

    .di-bar-segments {
      position: absolute;
      top: 2px;
      left: 2px;
      right: 2px;
      bottom: 2px;
      display: flex;
      gap: 2px;
      pointer-events: none;
    }

    .di-bar-segments .seg {
      flex: 1;
      background: rgba(0, 0, 0, 0.15);
      border-radius: 2px;
      transition: all 0.3s;
      border-right: 1px solid rgba(0, 0, 0, 0.1);
    }

    .di-bar-segments .seg:last-child {
      border-right: none;
    }

    .di-bar-segments .seg.on {
      background: rgba(255, 255, 255, 0.05);
      border-right-color: rgba(255, 255, 255, 0.1);
    }

    .di-bar-labels {
      display: flex;
      justify-content: space-between;
      margin-top: 0.25rem;
      font-size: 0.625rem;
      color: rgba(148, 163, 184, 0.5);
    }

    /* Scores + Contributions Annotées (Option 2) */
    .scores-contrib-annotated {
      background: rgba(30, 41, 59, 0.3);
      border-radius: 8px;
      padding: 0.5rem 0.625rem;
      border: 1px solid rgba(148, 163, 184, 0.08);
    }

    .contrib-title {
      font-size: 0.625rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: rgba(148, 163, 184, 0.6);
      margin-bottom: 0.5rem;
      text-align: center;
    }

    .contrib-subtitle {
      font-size: 0.55rem;
      text-transform: none;
      letter-spacing: 0;
      color: rgba(148, 163, 184, 0.4);
      cursor: help;
    }

    /* Ligne 1: Labels (icons + scores) */
    .contrib-labels-row {
      display: flex;
      margin-bottom: 0.375rem;
    }

    .contrib-label {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.125rem;
      font-size: 0.75rem;
    }

    .label-name {
      font-size: 0.6875rem;
      font-weight: 600;
      color: rgba(148, 163, 184, 0.9);
    }

    .label-pct {
      font-weight: 700;
      color: rgba(226, 232, 240, 1);
      font-size: 0.75rem;
    }

    /* Ligne 2: Barre empilée */
    .contrib-bar-stacked {
      display: flex;
      height: 20px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(15, 23, 42, 0.5);
      box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
      margin-bottom: 0.375rem;
    }

    .contrib-seg {
      height: 100%;
      transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
    }

    .contrib-seg.cycle {
      background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
    }

    .contrib-seg.onchain {
      background: linear-gradient(90deg, #7c3aed 0%, #8b5cf6 100%);
    }

    .contrib-seg.risk {
      background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
    }

    .contrib-seg::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 40%;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.3) 0%, transparent 100%);
    }

    /* Recommandation actionnable - compact v8 */
    .di-recommendation {
      background: rgba(30, 41, 59, 0.3);
      border-radius: 8px;
      padding: 0.5rem 0.625rem;
      border: 1px solid rgba(148, 163, 184, 0.08);
    }

    .reco-content {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }

    .reco-header {
      display: flex;
      align-items: center;
      gap: 0.375rem;
      padding-bottom: 0.25rem;
      border-bottom: 1px solid rgba(148, 163, 184, 0.08);
    }

    .reco-icon {
      font-size: 1rem;
      flex-shrink: 0;
    }

    .reco-title {
      font-size: 0.6875rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: rgba(226, 232, 240, 1);
    }

    .reco-action {
      font-size: 0.75rem;
      line-height: 1.3;
      color: rgba(226, 232, 240, 0.95);
      font-weight: 600;
    }

    .reco-details {
      font-size: 0.6875rem;
      line-height: 1.4;
      color: rgba(148, 163, 184, 0.7);
      font-style: italic;
    }

    /* Variantes de couleur pour recommandation */
    .di-recommendation.bullish {
      border-color: rgba(16, 185, 129, 0.3);
      background: rgba(16, 185, 129, 0.05);
    }

    .di-recommendation.positive {
      border-color: rgba(59, 130, 246, 0.3);
      background: rgba(59, 130, 246, 0.05);
    }

    .di-recommendation.warning {
      border-color: rgba(245, 158, 11, 0.3);
      background: rgba(245, 158, 11, 0.05);
    }

    .di-recommendation.defensive {
      border-color: rgba(239, 68, 68, 0.3);
      background: rgba(239, 68, 68, 0.05);
    }

    .di-recommendation.critical {
      border-color: rgba(153, 27, 27, 0.4);
      background: rgba(153, 27, 27, 0.1);
    }

    /* Metadata - compact inline v8 */
    .di-metadata {
      background: rgba(30, 41, 59, 0.3);
      border-radius: 8px;
      padding: 0.5rem 0.625rem;
      border: 1px solid rgba(148, 163, 184, 0.08);
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 0.25rem 0.75rem;
    }

    .meta-row {
      display: flex;
      align-items: center;
      gap: 0.375rem;
    }

    .meta-label {
      font-size: 0.5625rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: rgba(148, 163, 184, 0.5);
    }

    .meta-value {
      font-size: 0.6875rem;
      font-weight: 600;
      color: rgba(226, 232, 240, 0.85);
    }

    /* Badge Override */
    .meta-override {
      grid-column: 1 / -1;
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.3);
      border-radius: 4px;
      padding: 0.5rem;
    }

    .meta-override-value {
      color: #fca5a5 !important;
      font-weight: 700;
      font-size: 0.7rem;
    }

    /* ═══════════════════════════════════════════════════════════
       LEGACY COLUMN STYLES (kept for simulations.html compat)
       ═══════════════════════════════════════════════════════════ */

    .pillars-container {
      display: flex;
      flex-direction: column;
      gap: 0.875rem;
    }

    /* Barres de piliers compactes */
    .pillar-bar-compact {
      background: rgba(30, 41, 59, 0.3);
      border-radius: 8px;
      padding: 0.5rem 0.625rem;
      border: 1px solid rgba(148, 163, 184, 0.08);
      transition: all 0.3s;
    }

    .pillar-bar-compact:hover {
      background: rgba(30, 41, 59, 0.5);
      transform: translateX(2px);
    }

    .pillar-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.375rem;
    }

    .pillar-label {
      display: flex;
      align-items: center;
      gap: 0.375rem;
    }

    .pillar-icon {
      font-size: 0.875rem;
    }

    .pillar-name {
      font-size: 0.75rem;
      font-weight: 600;
      color: rgba(226, 232, 240, 0.9);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .conf-chip {
      background: rgba(59, 130, 246, 0.2);
      color: #60a5fa;
      padding: 0.125rem 0.25rem;
      border-radius: 999px;
      font-size: 0.5rem;
      font-weight: 600;
    }

    .pillar-score {
      font-size: 1.25rem;
      font-weight: 700;
      color: rgba(226, 232, 240, 1);
    }

    .pillar-track {
      position: relative;
      height: 12px;
      background: rgba(15, 23, 42, 0.5);
      border-radius: 999px;
      overflow: hidden;
      box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.3);
    }

    .pillar-fill {
      height: 100%;
      position: relative;
      border-radius: 999px;
      transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
      box-shadow: 0 0 8px currentColor;
    }

    .pillar-sub {
      margin-top: 0.25rem;
      font-size: 0.625rem;
      color: rgba(148, 163, 184, 0.7);
    }

    /* ═══════════════════════════════════════════════════════════
       CYCLE PILLAR - Visual Phases
       ═══════════════════════════════════════════════════════════ */

    .pillar-cycle-visual .pillar-track {
      position: relative;
      overflow: visible;
    }

    .pillar-track-cycle {
      background: rgba(15, 23, 42, 0.8) !important;
    }

    .cycle-phases-bg {
      position: absolute;
      inset: 0;
      display: flex;
      height: 100%;
      overflow: hidden;
      border-radius: 999px;
    }

    .cycle-phase-mini {
      height: 100%;
      position: relative;
      transition: all 0.3s ease;
    }

    .cycle-phase-mini::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 35%;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.25) 0%, transparent 100%);
    }

    .cycle-position-mini {
      position: absolute;
      top: -4px;
      bottom: -4px;
      width: 3px;
      transform: translateX(-50%);
      z-index: 10;
      pointer-events: none;
    }

    .cycle-marker-mini {
      width: 100%;
      height: 100%;
      background: linear-gradient(180deg, #3b82f6 0%, #1d4ed8 100%);
      border-radius: 999px;
      box-shadow:
        0 0 8px rgba(59, 130, 246, 0.8),
        0 0 12px rgba(59, 130, 246, 0.4),
        0 2px 4px rgba(0, 0, 0, 0.3);
      animation: pulse-marker 2s ease-in-out infinite;
    }

    @keyframes pulse-marker {
      0%, 100% {
        box-shadow:
          0 0 8px rgba(59, 130, 246, 0.8),
          0 0 12px rgba(59, 130, 246, 0.4),
          0 2px 4px rgba(0, 0, 0, 0.3);
      }
      50% {
        box-shadow:
          0 0 12px rgba(59, 130, 246, 1),
          0 0 20px rgba(59, 130, 246, 0.6),
          0 2px 6px rgba(0, 0, 0, 0.4);
      }
    }

    /* ═══════════════════════════════════════════════════════════
       LEGACY (removed sections)
       ═══════════════════════════════════════════════════════════ */

    /* Full-width metrics bar (replaces footer) */
    .di-metrics-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.5rem 1rem;
      margin-top: 0.5rem;
      background: rgba(30, 41, 59, 0.2);
      border-top: 1px solid rgba(148, 163, 184, 0.08);
      border-radius: 0 0 12px 12px;
    }

    .metrics-bar-left,
    .metrics-bar-right {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .metrics-bar-item {
      display: flex;
      align-items: center;
      gap: 0.375rem;
    }

    .metrics-bar-label {
      font-size: 0.5625rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: rgba(148, 163, 184, 0.5);
    }

    .metrics-bar-value {
      font-size: 0.75rem;
      font-weight: 700;
      color: rgba(226, 232, 240, 0.85);
    }

    .metrics-bar-value.live {
      color: #10b981;
    }

    .metrics-bar-value.offline {
      color: #ef4444;
    }

    /* ═══════════════════════════════════════════════════════════
       MINI CYCLE CHART (v8.0)
       ═══════════════════════════════════════════════════════════ */

    .di-mini-cycle-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      background: rgba(30, 41, 59, 0.3);
      border-radius: 8px;
      padding: 0.5rem 0.625rem;
      border: 1px solid rgba(148, 163, 184, 0.08);
      min-height: 0;
    }

    .di-mini-cycle-title {
      font-size: 0.5rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: rgba(148, 163, 184, 0.5);
      margin-bottom: 0.25rem;
      text-align: center;
      flex-shrink: 0;
    }

    .di-mini-cycle-wrapper {
      position: relative;
      flex: 1;
      min-height: 80px;
    }

    .di-mini-cycle-wrapper canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100% !important;
      height: 100% !important;
    }

    /* ═══════════════════════════════════════════════════════════
       ALLOCATION TEXT COMPACT (v8.0)
       ═══════════════════════════════════════════════════════════ */

    .di-alloc-text {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 0.625rem;
      background: rgba(30, 41, 59, 0.3);
      border: 1px solid rgba(148, 163, 184, 0.08);
      border-radius: 8px;
    }

    .alloc-text-label {
      font-size: 0.5rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: rgba(148, 163, 184, 0.5);
      flex-shrink: 0;
    }

    .alloc-text-values {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.125rem;
      font-size: 0.6875rem;
      color: rgba(226, 232, 240, 0.85);
    }

    .alloc-text-item {
      display: inline-flex;
      align-items: center;
      gap: 0.2rem;
      font-weight: 600;
    }

    .alloc-text-dot {
      width: 5px;
      height: 5px;
      border-radius: 50%;
      flex-shrink: 0;
    }

    .alloc-text-sep {
      color: rgba(148, 163, 184, 0.3);
      margin: 0 0.125rem;
    }

    /* ═══════════════════════════════════════════════════════════
       FUSED PILLARS (v8.0)
       ═══════════════════════════════════════════════════════════ */

    .di-pillars-fused {
      display: flex;
      align-items: stretch;
      gap: 0;
      background: rgba(30, 41, 59, 0.3);
      border: 1px solid rgba(148, 163, 184, 0.08);
      border-radius: 8px;
      padding: 0.5rem 0.625rem;
    }

    .pillar-fused-col {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 0.25rem;
      padding: 0.25rem 0.375rem;
    }

    .pillar-fused-divider {
      width: 1px;
      background: rgba(148, 163, 184, 0.12);
      align-self: stretch;
      flex-shrink: 0;
    }

    .pillar-fused-label {
      font-size: 0.5625rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: rgba(148, 163, 184, 0.6);
    }

    .pillar-fused-score {
      font-size: 1.5rem;
      font-weight: 700;
      line-height: 1;
    }

    .pillar-fused-track {
      width: 100%;
      height: 6px;
      background: rgba(15, 23, 42, 0.5);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 0.25rem;
    }

    .pillar-fused-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Legacy allocation bar (kept for backward compat with simulations.html) */
    .alloc-bar-container {
      background: rgba(30, 41, 59, 0.3);
      border-radius: 8px;
      padding: 0.625rem 0.75rem;
      border: 1px solid rgba(148, 163, 184, 0.08);
      margin-bottom: 0.75rem;
    }

    .alloc-bar-title {
      font-size: 0.625rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: rgba(148, 163, 184, 0.6);
      margin-bottom: 0.5rem;
      text-align: center;
    }

    .alloc-bar-track {
      display: flex;
      height: 24px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(15, 23, 42, 0.5);
      box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    .alloc-seg {
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
      min-width: 4px;
    }

    .alloc-seg::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 40%;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.3) 0%, transparent 100%);
    }

    .alloc-seg-label {
      font-size: 0.6rem;
      font-weight: 700;
      color: rgba(255, 255, 255, 0.95);
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
      z-index: 1;
    }

    .alloc-legend-inline {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 0.375rem 0.625rem;
      margin-top: 0.5rem;
    }

    .alloc-leg-inline {
      display: flex;
      align-items: center;
      gap: 0.2rem;
      font-size: 0.6rem;
      color: rgba(226, 232, 240, 0.85);
    }

    .alloc-dot-sm {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--c, #6b7280);
    }

    .alloc-bar-placeholder {
      text-align: center;
      padding: 0.75rem;
      color: rgba(148, 163, 184, 0.5);
      font-size: 0.75rem;
    }

    /* ═══════════════════════════════════════════════════════════
       QUICK CONTEXT BAR
       ═══════════════════════════════════════════════════════════ */

    .di-context-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      background: rgba(30, 41, 59, 0.3);
      border: 1px solid rgba(148, 163, 184, 0.08);
      border-radius: 8px;
      padding: 0.5rem 0.625rem;
    }

    .ctx-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.125rem;
      flex: 1;
      min-width: 0;
    }

    .ctx-label {
      font-size: 0.5rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: rgba(148, 163, 184, 0.6);
    }

    .ctx-value {
      font-size: 0.7rem;
      font-weight: 600;
      color: rgba(226, 232, 240, 0.95);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 100%;
    }

    /* Phase with proper overflow handling */
    .ctx-item[data-ctx="phase"] .ctx-value {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 120px;
    }

    .ctx-divider {
      width: 1px;
      height: 24px;
      background: rgba(148, 163, 184, 0.15);
      flex-shrink: 0;
    }

    /* Popup d'aide compact */
    .di-help-popup {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%) scale(0.95);
      width: min(400px, 90vw);
      max-height: 70vh;
      background: linear-gradient(135deg,
        rgba(15, 23, 42, 0.98) 0%,
        rgba(30, 41, 59, 0.98) 100%);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 12px;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
      z-index: 1000;
      opacity: 0;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      backdrop-filter: blur(20px);
    }

    .di-help-popup.show {
      opacity: 1;
      transform: translate(-50%, -50%) scale(1);
    }

    .di-help-content {
      padding: 1.25rem;
      overflow-y: auto;
      max-height: 70vh;
    }

    .di-help-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid rgba(148, 163, 184, 0.1);
    }

    .di-help-header h3 {
      margin: 0;
      font-size: 1rem;
      font-weight: 700;
      color: rgba(226, 232, 240, 1);
    }

    .di-help-close {
      background: none;
      border: none;
      font-size: 1.25rem;
      color: rgba(148, 163, 184, 0.5);
      cursor: pointer;
      padding: 0.25rem;
      transition: all 0.2s;
      border-radius: 4px;
    }

    .di-help-close:hover {
      color: rgba(226, 232, 240, 1);
      background: rgba(239, 68, 68, 0.1);
    }

    .di-help-body {
      color: rgba(203, 213, 225, 0.9);
      line-height: 1.5;
      font-size: 0.75rem;
    }

    .di-help-body p {
      margin: 0 0 0.75rem 0;
    }

    .di-help-body p:last-child {
      margin-bottom: 0;
    }

    /* Responsive */
    @media (max-width: 768px) {
      .di-row {
        grid-template-columns: 1fr;
      }

      .di-score-big {
        font-size: 3rem;
      }

      .di-metrics-bar {
        flex-wrap: wrap;
        gap: 0.5rem;
      }

      .metrics-bar-left,
      .metrics-bar-right {
        gap: 0.5rem;
        flex-wrap: wrap;
      }

      .di-metadata {
        gap: 0.125rem 0.5rem;
      }

      .di-context-bar {
        padding: 0.5rem;
        gap: 0.25rem;
      }

      .ctx-label {
        font-size: 0.45rem;
      }

      .ctx-value {
        font-size: 0.65rem;
      }

      .ctx-item[data-ctx="phase"] .ctx-value {
        max-width: 80px;
      }

      .di-mini-cycle-wrapper {
        min-height: 120px;
      }

      .di-pillars-fused {
        padding: 0.375rem 0.5rem;
      }

      .pillar-fused-score {
        font-size: 1.125rem;
      }
    }

    @media (max-width: 480px) {
      .di-panel-gaming {
        padding: 1rem;
      }

      .di-score-big {
        font-size: 2.5rem;
      }

      .contrib-labels-row {
        font-size: 0.65rem;
      }

      .di-metadata {
        justify-content: center;
      }

      .ctx-item[data-ctx="phase"] .ctx-value {
        max-width: 60px;
      }

      .di-context-bar {
        flex-wrap: wrap;
        justify-content: center;
      }

      .ctx-item {
        flex: 0 0 calc(50% - 0.5rem);
      }

      .ctx-divider {
        display: none;
      }

      .di-mini-cycle-wrapper {
        min-height: 100px;
      }

      .pillar-fused-score {
        font-size: 1rem;
      }

      .pillar-fused-label {
        font-size: 0.5rem;
      }
    }

    /* Light mode */
    @media (prefers-color-scheme: light) {
      .di-panel-gaming {
        background: linear-gradient(135deg,
          rgba(255, 255, 255, 0.95) 0%,
          rgba(248, 250, 252, 0.95) 100%);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
      }

      .di-bar-track,
      .pillar-track,
      .contrib-bar-stacked {
        background: rgba(226, 232, 240, 0.5);
      }

      .pillar-bar-compact,
      .scores-contrib-annotated,
      .di-recommendation,
      .di-metadata,
      .di-score-block,
      .di-mini-cycle-container,
      .di-alloc-text,
      .di-pillars-fused,
      .di-context-bar {
        background: rgba(248, 250, 252, 0.5);
      }

      .di-score-big {
        background: linear-gradient(135deg, #1e293b 0%, #475569 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
      }

      .pillar-name,
      .pillar-score,
      .meta-value,
      .label-pct,
      .reco-title,
      .reco-action {
        color: #1e293b;
      }

      .di-title,
      .contrib-title,
      .meta-label,
      .metrics-bar-label,
      .reco-details {
        color: #64748b;
      }
    }
  `;

  document.head.appendChild(styles);
}

/**
 * Render principal du panneau
 */
function _renderDIPanelInternal(container, data, opts = {}) {
  if (!container) {
    console.error('❌ DI Panel: container element not found');
    return;
  }

  // Cleanup previous mini cycle chart before re-render
  if (_diMiniCycleChart) {
    _diMiniCycleChart.destroy();
    _diMiniCycleChart = null;
  }

  // Injecter les styles si nécessaire
  injectStyles();

  // Générer les parties
  const left = getLeftParts(data);
  const right = getRightParts(data);
  const metricsBar = renderMetricsBar(data.meta || {}, data.scores || {});

  // Layout 3 rangées alignées: chaque rangée est une grille 2 colonnes
  container.innerHTML = `
    <div class="di-panel-gaming">
      <div class="di-row di-row-top">
        <div class="di-top-left">
          ${left.scoreBlock}
          ${left.weights}
        </div>
        <div class="di-top-right">
          ${right.contextBar}
          ${right.cycleChart}
        </div>
      </div>
      <div class="di-row di-row-mid">
        ${left.metadata}
        ${right.allocation || '<div></div>'}
      </div>
      <div class="di-row di-row-bot">
        ${left.recommendation}
        ${right.fusedPillars}
      </div>
      ${metricsBar}
      ${renderHelpContent()}
    </div>
  `;

  // Monter le système d'aide
  mountHelpSystem(container);

  // Mount mini cycle chart after DOM is ready
  const cycleMonths = data.meta?.cycle_months;
  requestAnimationFrame(() => {
    mountMiniCycleChart(cycleMonths);
  });
}

/**
 * Fonction publique principale avec debounce
 */
export function renderDecisionIndexPanel(container, data, opts = {}) {
  clearTimeout(refreshTimeout);
  refreshTimeout = setTimeout(() => {
    _renderDIPanelInternal(container, data, opts);
  }, 100);
}

/**
 * Cleanup (détruit les event listeners pour prévenir memory leaks)
 */
export function destroyDIPanelCharts() {
  // Cleanup mini cycle chart
  if (_diMiniCycleChart) {
    _diMiniCycleChart.destroy();
    _diMiniCycleChart = null;
  }

  // Cleanup event listeners
  if (helpSystemController) {
    helpSystemController.abort();
    helpSystemController = null;
  }

  // Reset state
  helpPopoverState.isOpen = false;
  helpPopoverState.lastFocusedElement = null;
}

/**
 * Helper pour s'assurer que les dépendances sont chargées
 */
export async function ensureChartJSLoaded() {
  // Chart.js requis pour le mini cycle chart (v8.0)
  return typeof Chart !== 'undefined';
}

// Logger minimal
const debugLogger = {
  debug: (...args) => console.debug('[DI Panel]', ...args),
  error: (...args) => console.error('[DI Panel]', ...args),
  warn: (...args) => console.warn('[DI Panel]', ...args)
};

if (typeof window !== 'undefined') {
  window.debugLogger = debugLogger;
}
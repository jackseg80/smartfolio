/**
 * Unified Market Regime Constants - Single Source of Truth.
 *
 * All regime-related frontend code MUST import from here.
 * Scores: 0-100 where 100 = best market conditions.
 */

export const MarketRegime = Object.freeze({
    BEAR_MARKET: 0,
    CORRECTION: 1,
    BULL_MARKET: 2,
    EXPANSION: 3,
});

export const REGIME_NAMES = Object.freeze([
    'Bear Market', 'Correction', 'Bull Market', 'Expansion'
]);

export const REGIME_IDS = Object.freeze({
    'Bear Market': 0,
    'Correction': 1,
    'Bull Market': 2,
    'Expansion': 3,
});

export const REGIME_SCORE_RANGES = Object.freeze([
    { min: 0,  max: 25,  id: 0, name: 'Bear Market' },
    { min: 26, max: 50,  id: 1, name: 'Correction' },
    { min: 51, max: 75,  id: 2, name: 'Bull Market' },
    { min: 76, max: 100, id: 3, name: 'Expansion' },
]);

export const REGIME_COLORS = Object.freeze({
    0: '#dc2626',            // Bear Market - red
    1: '#ea580c',            // Correction - orange
    2: '#22c55e',            // Bull Market - green
    3: '#3b82f6',            // Expansion - blue
    'Bear Market': '#dc2626',
    'Correction': '#ea580c',
    'Bull Market': '#22c55e',
    'Expansion': '#3b82f6',
    // snake_case keys for convenience
    'bear_market': '#dc2626',
    'correction': '#ea580c',
    'bull_market': '#22c55e',
    'expansion': '#3b82f6',
});

export const REGIME_EMOJIS = Object.freeze({
    'Bear Market': '\u{1F534}',   // red circle
    'Correction': '\u{1F7E0}',    // orange circle
    'Bull Market': '\u{1F7E2}',   // green circle
    'Expansion': '\u{1F535}',     // blue circle
    'bear_market': '\u{1F534}',
    'correction': '\u{1F7E0}',
    'bull_market': '\u{1F7E2}',
    'expansion': '\u{1F535}',
});

/** Legacy name -> canonical name mapping */
export const LEGACY_TO_CANONICAL = Object.freeze({
    // Convention C (Legacy crypto cycle names)
    'accumulation': 'Bear Market',
    'Accumulation': 'Bear Market',
    'euphoria': 'Bull Market',
    'Euphoria': 'Bull Market',
    'distribution': 'Expansion',
    'Distribution': 'Expansion',
    // Convention B (Training script short names)
    'bull': 'Bull Market',
    'Bull': 'Bull Market',
    'bear': 'Bear Market',
    'Bear': 'Bear Market',
    'sideways': 'Correction',
    'Sideways': 'Correction',
    // Snake-case variants
    'bull_market': 'Bull Market',
    'bear_market': 'Bear Market',
    'expansion': 'Expansion',
    'correction': 'Correction',
    // French variants
    'euphorie': 'Bull Market',
    'Euphorie': 'Bull Market',
    // Neutral / Unknown
    'neutral': 'Correction',
    'unknown': 'Correction',
    'Consolidation': 'Correction',
    'consolidation': 'Correction',
    'early_expansion': 'Correction',
    // Canonical (pass-through)
    'Bear Market': 'Bear Market',
    'Correction': 'Correction',
    'Bull Market': 'Bull Market',
    'Expansion': 'Expansion',
});

/**
 * Convert score (0-100, 100=best) to regime object.
 * @param {number} score
 * @returns {{ id: number, name: string, key: string }}
 */
export function scoreToRegime(score) {
    if (score <= 25) return { id: 0, name: 'Bear Market', key: 'bear_market' };
    if (score <= 50) return { id: 1, name: 'Correction', key: 'correction' };
    if (score <= 75) return { id: 2, name: 'Bull Market', key: 'bull_market' };
    return { id: 3, name: 'Expansion', key: 'expansion' };
}

/**
 * Get regime name from ID.
 * @param {number} regimeId
 * @returns {string}
 */
export function regimeName(regimeId) {
    return REGIME_NAMES[Math.max(0, Math.min(3, regimeId))];
}

/**
 * Convert any legacy/variant regime name to canonical form.
 * @param {string} name
 * @returns {string}
 */
export function normalizeRegimeName(name) {
    if (!name) return 'Correction';
    return LEGACY_TO_CANONICAL[name] || LEGACY_TO_CANONICAL[name.toLowerCase()] || name;
}

/**
 * Convert regime name to snake_case key.
 * 'Bear Market' -> 'bear_market', etc.
 * @param {string} name
 * @returns {string}
 */
export function regimeToKey(name) {
    const canonical = normalizeRegimeName(name);
    return canonical.toLowerCase().replace(' ', '_');
}

/**
 * Get regime color from name or ID.
 * @param {string|number} regime
 * @returns {string}
 */
export function regimeColor(regime) {
    return REGIME_COLORS[regime] || REGIME_COLORS[normalizeRegimeName(String(regime))] || '#6b7280';
}

/**
 * Get regime emoji from name.
 * @param {string} regime
 * @returns {string}
 */
export function regimeEmoji(regime) {
    const canonical = normalizeRegimeName(regime);
    return REGIME_EMOJIS[canonical] || '\u{26AA}';
}

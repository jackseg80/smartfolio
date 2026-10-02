/**
 * Fonctions ML partagées - Crypto Portfolio
 * Centralise les appels API, utilitaires UI et fonctions communes
 */

// Configuration API - uses centralized window.getApiBase() from global-config.js
import { getAuthHeaders } from './core/auth-guard.js';

// Utilitaires UI communes
export function showLoading(elementId, message = 'Loading...') {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<span class="loading-spinner"></span> ${message}`;
    }
}

export function showError(message, container = null) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.style.cssText = `
        background: var(--error-bg, #fee);
        color: var(--error-text, #c53030);
        border: 1px solid var(--error-border, #fed7d7);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    `;
    errorDiv.textContent = message;

    if (container) {
        container.appendChild(errorDiv);
    } else {
        document.body.appendChild(errorDiv);
    }

    setTimeout(() => errorDiv.remove(), 5000);
}

export function showSuccess(message, container = null) {
    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.style.cssText = `
        background: var(--success-bg, #f0fff4);
        color: var(--success-text, #2d7d32);
        border: 1px solid var(--success-border, #c6f6d5);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    `;
    successDiv.textContent = message;

    if (container) {
        container.appendChild(successDiv);
    } else {
        document.body.appendChild(successDiv);
    }

    setTimeout(() => successDiv.remove(), 5000);
}

// API Calls communes
export async function fetchMLStatus(endpoint) {
    try {
        const apiBase = window.getApiBase();
        const response = await fetch(`${apiBase}/api/ml/${endpoint}`, {
            headers: getAuthHeaders()
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        (window.debugLogger?.warn || console.warn)(`ML API ${endpoint} unavailable:`, error.message);
        return null;
    }
}

export async function postMLAction(endpoint, data = {}) {
    try {
        const apiBase = window.getApiBase();
        const response = await fetch(`${apiBase}/api/ml/${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
            body: JSON.stringify(data)
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        (window.debugLogger?.error || console.error)(`ML API ${endpoint} failed:`, error);
        throw error;
    }
}

// Fonctions ML spécifiques
export async function getVolatilityStatus() {
    return await fetchMLStatus('volatility/models/status');
}

export async function getRegimeStatus() {
    return await fetchMLStatus('regime/status');
}

export async function getCorrelationStatus() {
    return await fetchMLStatus('correlation/status');
}

export async function getSentimentStatus() {
    return await fetchMLStatus('sentiment/status');
}

export async function getRebalanceStatus() {
    return await fetchMLStatus('rebalance/status');
}

// Actions ML
export async function trainVolatilityModel(symbols = ['BTC', 'ETH']) {
    return await postMLAction('volatility/train-portfolio', { symbols });
}

export async function getCurrentRegime() {
    return await fetchMLStatus('regime/current');
}

export async function trainRegimeModel() {
    return await postMLAction('regime/train', {});
}

export async function analyzeCorrelations(symbols = ['BTC', 'ETH'], windowDays = 30) {
    const source = window.globalConfig?.get('data_source');
    if (!source) {
        throw new Error('No portfolio source selected');
    }
    return await fetchMLStatus(
        `correlation/matrix/current?window_days=${windowDays}&source=${encodeURIComponent(source)}`
    );
}

export async function analyzeSentiment(symbols = ['BTC', 'ETH'], days = 7) {
    return await fetchMLStatus(`sentiment/analyze?symbols=${symbols.join(',')}&days=${days}`);
}

export async function getFearGreedIndex(days = 7) {
    return await fetchMLStatus(`sentiment/fear-greed?days=${days}`);
}

// Utilitaires de formatage - réexportés depuis core/formatters.js
export { formatPercentage, formatCurrency, formatDate } from './core/formatters.js';

// Gestion des boutons d'action
export function setupActionButton(buttonId, actionFn, loadingText = 'Traitement...') {
    const btn = document.getElementById(buttonId);
    if (!btn) return;

    btn.addEventListener('click', async (e) => {
        const originalText = btn.textContent;
        btn.disabled = true;
        btn.textContent = loadingText;

        try {
            await actionFn(e);
            showSuccess('Operation completed successfully');
        } catch (error) {
            showError(`Error: ${error.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = originalText;
        }
    });
}

// Mise à jour des status cards
export function updateStatusCard(cardId, data) {
    const card = document.getElementById(cardId);
    if (!card || !data) return;

    const statusElement = card.querySelector('.status-indicator');
    const valueElements = card.querySelectorAll('[data-value]');

    // Mise à jour du statut
    if (statusElement) {
        const isActive = data.active || data.loaded || data.status === 'active';
        statusElement.className = `status-indicator ${isActive ? 'active' : 'inactive'}`;
    }

    // Mise à jour des valeurs
    valueElements.forEach(el => {
        const key = el.getAttribute('data-value');
        if (data[key] !== undefined) {
            el.textContent = data[key];
        }
    });
}

// Chargement de tous les status ML
export async function loadAllMLStatus() {
    const [volatility, regime, correlation, sentiment, rebalance] = await Promise.allSettled([
        getVolatilityStatus(),
        getRegimeStatus(),
        getCorrelationStatus(),
        getSentimentStatus(),
        getRebalanceStatus()
    ]);

    return {
        volatility: volatility.status === 'fulfilled' ? volatility.value : null,
        regime: regime.status === 'fulfilled' ? regime.value : null,
        correlation: correlation.status === 'fulfilled' ? correlation.value : null,
        sentiment: sentiment.status === 'fulfilled' ? sentiment.value : null,
        rebalance: rebalance.status === 'fulfilled' ? rebalance.value : null
    };
}

// SOURCE UNIQUE DE VÉRITÉ - Status ML unifié (comme AI Dashboard)
// Cache pour éviter les appels répétés
let mlUnifiedCache = { data: null, timestamp: 0 };
const ML_CACHE_TTL = 15 * 60 * 1000; // 15 minutes (optimized: ML orchestrator runs hourly)

/**
 * Fonction centralisée qui utilise la MÊME logique prioritaire que AI Dashboard
 * Priority 1: Governance Engine -> Priority 2: ML Status API -> Priority 3: Stable fallback
 */
export async function getUnifiedMLStatus() {
    // No stale result cache: identity/source changes must take effect immediately.
    const source = window.globalConfig?.get?.('data_source') || 'cointracking';
    const data = await fetchMLStatus('overview?source=' + encodeURIComponent(source));
    return {
        totalLoaded: data?.counts?.models_loaded ?? null,
        totalModels: data?.counts?.files_present ?? null,
        confidence: null,
        source: data ? 'verified_overview' : 'unavailable',
        timestamp: data?.observed_at ?? null,
        available: Boolean(data),
        reason: data ? 'Confidence requires an evaluation record' : 'ML overview unavailable',
        individual: {
            volatility: { loaded: null, available: false },
            regime: { loaded: null, available: false },
            correlation: { loaded: null, available: false },
            sentiment: { loaded: null, available: false }
        }
    };
}

/**
 * Clear ML cache (for testing)
 */
export function clearMLUnifiedCache() {
    mlUnifiedCache = { data: null, timestamp: 0 };
    (window.debugLogger?.debug || console.log)("ML unified cache cleared");
}

// Initialisation globale UNIFIED
export function initializeMLDashboard() {
    (window.debugLogger?.debug || console.log)("ML Dashboard initialized with unified status");

    // Utiliser le status unifié au lieu de loadAllMLStatus
    getUnifiedMLStatus().then(status => {
        (window.debugLogger?.info || console.log)("Unified ML Status loaded:", status);

        // Mettre à jour les cards avec les données unifiées
        if (status.individual.volatility) updateStatusCard('volatility-card', status.individual.volatility);
        if (status.individual.regime) updateStatusCard('regime-card', status.individual.regime);
        if (status.individual.correlation) updateStatusCard('correlation-card', status.individual.correlation);
        if (status.individual.sentiment) updateStatusCard('sentiment-card', status.individual.sentiment);
    });
}

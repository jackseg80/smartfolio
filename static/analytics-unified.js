/**
 * Analytics Unified - Dynamic Data Loading
 * Récupère les vraies données depuis les APIs backend
 */

console.debug('🔄 Analytics Unified - Initialisation');

// Import risk alerts loader
import { startRiskAlertsPolling } from './modules/risk-alerts-loader.js';
// PERFORMANCE FIX (Dec 2025): Throttle utilities to prevent event spam
import { throttle } from './utils/debounce.js';

// PERFORMANCE FIX (Dec 2025): DOM selector cache to prevent repeated traversals
const domCache = new Map();

/**
 * Get a DOM element from cache or query and cache it
 * @param {string} selector - CSS selector
 * @param {Element} parent - Parent element (optional, defaults to document)
 * @returns {Element|null}
 */
function getCachedElement(selector, parent = document) {
    const cacheKey = parent === document ? selector : `${parent.id || 'parent'}_${selector}`;

    if (!domCache.has(cacheKey)) {
        const element = parent.querySelector(selector);
        if (element) {
            domCache.set(cacheKey, element);
        }
        return element;
    }

    return domCache.get(cacheKey);
}

/**
 * Clear DOM cache (call when DOM structure changes significantly)
 */
function clearDomCache() {
    domCache.clear();
}

// Configuration
const API_BASE = window.getApiBase();

// 🆕 Cache TTL adaptatif selon CLAUDE.md (Oct 2025 optimization)
const CACHE_TTL = {
    'risk-dashboard': 30 * 60 * 1000,     // 30 min (Risk Metrics VaR - historique daily)
    'cache-stats': 15 * 60 * 1000,        // 15 min (Performance cache stats)
    'memory-stats': 15 * 60 * 1000,       // 15 min (Memory usage)
    'cycle-analysis': 24 * 60 * 60 * 1000 // 24h (Cycle Score - évolution 0.1%/jour)
};

// Cache intelligent avec TTL adaptatif
const cache = new Map();

async function fetchWithCache(key, fetchFn) {
    const now = Date.now();
    const cached = cache.get(key);
    const ttl = CACHE_TTL[key] || 60000; // Fallback 1 min si clé inconnue

    if (cached && (now - cached.timestamp) < ttl) {
        console.debug(`✅ Cache hit: ${key} (age: ${Math.round((now - cached.timestamp) / 1000)}s / TTL: ${ttl / 1000}s)`);
        return cached.data;
    }

    try {
        console.debug(`🔄 Cache miss: ${key} - fetching fresh data...`);
        const data = await fetchFn();
        cache.set(key, { data, timestamp: now });
        return data;
    } catch (error) {
        (window.debugLogger?.warn || console.warn)(`Failed to fetch ${key}:`, error);
        // Retourner données cachées même expirées si erreur réseau (stale-while-revalidate)
        if (cached) {
            console.debug(`⚠️ Using stale cache for ${key} due to fetch error`);
            return cached.data;
        }
        return null;
    }
}

// Tab switching functionality
document.addEventListener('DOMContentLoaded', function () {
    setupTabSwitching();
    loadInitialData();

    // PERFORMANCE FIX (Dec 2025): Throttle storage events to prevent spam
    // Storage events can fire rapidly during batch updates - throttle to 500ms
    const throttledStorageHandler = throttle((e) => {
        if (e.key && e.key.startsWith('risk_score_')) {
            try { refreshScoresFromLocalStorage(); } catch (_) { }
        }
    }, 500);

    window.addEventListener('storage', throttledStorageHandler);

    // Also listen to unified riskStore (populated by analytics-unified.html)
    attachRiskStoreListener();
});

function getScoresFromLocalStorage() {
    try {
        // Primary: values saved by risk-dashboard
        const onchainLS = parseFloat(localStorage.getItem('risk_score_onchain'));
        const riskLS = parseFloat(localStorage.getItem('risk_score_risk'));
        const blendedLS = parseFloat(localStorage.getItem('risk_score_blended'));
        const ccsLS = parseFloat(localStorage.getItem('risk_score_ccs'));

        let onchain = Number.isFinite(onchainLS) ? onchainLS : null;
        let risk = Number.isFinite(riskLS) ? riskLS : null;
        let blended = Number.isFinite(blendedLS) ? blendedLS : null;
        let ccs = Number.isFinite(ccsLS) ? ccsLS : null;

        // Fallback: unified store (set by analytics-unified inline loader)
        try {
            const s = window.riskStore ? window.riskStore.snapshot() : null;
            if (onchain == null && typeof s?.scores?.onchain === 'number') onchain = s.scores.onchain;
            if (risk == null && typeof s?.scores?.risk === 'number') risk = s.scores.risk;
            if (blended == null && typeof s?.scores?.blended === 'number') blended = s.scores.blended;
            if (ccs == null) {
                if (typeof s?.scores?.ccs === 'number') ccs = s.scores.ccs;
                else if (typeof s?.ccs?.score === 'number') ccs = s.ccs.score;
            }
        } catch (_) { }

        const timestamp = localStorage.getItem('risk_score_timestamp');
        return {
            onchain: Number.isFinite(onchain) ? onchain : null,
            risk: Number.isFinite(risk) ? risk : null,
            blended: Number.isFinite(blended) ? blended : null,
            ccs: Number.isFinite(ccs) ? ccs : null,
            timestamp
        };
    } catch (_) {
        return { onchain: null, risk: null, blended: null, ccs: null, timestamp: null };
    }
}

function refreshScoresFromLocalStorage() {
    const scores = getScoresFromLocalStorage();
    if (scores.onchain != null) {
        updateMetric('risk-kpi-onchain', Math.round(scores.onchain), 'Fondamentaux on-chain');
    }
    if (scores.blended != null) {
        updateMetric('risk-kpi-blended', Math.round(scores.blended), 'CCS × Cycle (synthèse)');
    }
}

// Subscribe to window.riskStore when it becomes available to keep KPIs synced
function attachRiskStoreListener() {
    const tryAttach = () => {
        if (window.riskStore && typeof window.riskStore.subscribe === 'function') {
            // Initial pull
            try { refreshScoresFromLocalStorage(); } catch (_) { }
            try {
                window.riskStore.subscribe(() => {
                    try { refreshScoresFromLocalStorage(); } catch (_) { }
                });
                console.debug('Analytics Unified: subscribed to riskStore updates');
            } catch (_) { }
            return true;
        }
        return false;
    };
    if (!tryAttach()) {
        let attempts = 0;
        const id = setInterval(() => {
            attempts += 1;
            if (tryAttach() || attempts > 20) {
                clearInterval(id);
            }
        }, 250);
    }
}

function setupTabSwitching() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetId = button.dataset.target;

            // Update active states
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanels.forEach(panel => panel.classList.remove('active'));

            button.classList.add('active');
            document.querySelector(targetId).classList.add('active');

            // Load data for active tab
            loadTabData(targetId);
        });
    });
}

async function loadInitialData() {
    // Load Risk tab data by default
    await loadTabData('#tab-risk');
}

async function loadTabData(tabId) {
    const tab = tabId.replace('#tab-', '');

    try {
        switch (tab) {
            case 'risk':
                await loadRiskData();
                break;
            case 'performance':
                await loadPerformanceData();
                break;
            case 'cycles':
                await loadCycleData();
                break;
            case 'monitoring':
                await loadMonitoringData();
                break;
            case 'intelligence-ml':
                // ML Tab handled by its own initialization system
                (window.debugLogger?.debug || console.log)('🤖 Intelligence ML tab activated - components should auto-initialize');
                break;
            default:
                (window.debugLogger?.warn || console.warn)(`Unknown tab: ${tab}`);
        }
    } catch (error) {
        debugLogger.error(`Error loading ${tab} data:`, error);
        showErrorState(tabId);
    }
}

async function loadRiskData() {
    console.debug("Analytics Unified: Loading Risk Dashboard data...");

    // Start real-time risk alerts polling (unified alert system)
    startRiskAlertsPolling();

    // 🆕 FIX Nov 2025: Multi-tenant support avec X-User header
    const activeUser = localStorage.getItem('activeUser') || 'demo';

    const riskData = await fetchWithCache('risk-dashboard', async () => {
        const minUsd = globalConfig?.get('min_usd_threshold') || 10;
        const url = `${API_BASE}/api/risk/dashboard?min_usd=${minUsd}&price_history_days=365&lookback_days=90`;
        const response = await fetch(url, {
            headers: { 'X-User': activeUser }
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    });

    if (!(riskData?.success && riskData?.risk_metrics)) {
        showRiskError();
        return;
    }

    const metrics = riskData.risk_metrics;

    // Core risk metrics
    updateMetric('risk-var', formatPercent(Math.abs(metrics.var_95_1d)), '95% confidence level');
    updateMetric('risk-drawdown', formatPercent(Math.abs(metrics.max_drawdown)), 'Current cycle');
    updateMetric('risk-volatility', formatPercent(metrics.volatility_annualized), '30-day annualized');
    updateMetric('risk-score', `${metrics.risk_score || '--'}/100`, getRiskLevel(metrics.risk_score));

    // Note: Risk alerts now loaded dynamically via startRiskAlertsPolling()

    // Diversification metrics
    const corr = riskData.correlation_metrics || {};
    if (typeof corr.diversification_ratio === 'number') {
        updateMetric('risk-kpi-diversification', (corr.diversification_ratio).toFixed(2), 'Corrélation de portefeuille');
    } else {
        updateMetric('risk-kpi-diversification', '--', 'Indisponible');
    }
    if (typeof corr.effective_assets === 'number') {
        updateMetric('risk-kpi-effective-assets', Math.round(corr.effective_assets), 'Actifs non-redondants');
    } else {
        updateMetric('risk-kpi-effective-assets', '--', 'Indisponible');
    }

    // Scores depuis le Risk Dashboard (source de vérité)
    const ls = getScoresFromLocalStorage();
    if (ls.onchain != null) {
        updateMetric('risk-kpi-onchain', Math.round(ls.onchain), 'Fondamentaux on-chain');
    } else {
        updateMetric('risk-kpi-onchain', '--', 'Fondamentaux on-chain (bientôt)');
    }
    if (ls.blended != null) {
        updateMetric('risk-kpi-blended', Math.round(ls.blended), 'CCS × Cycle (synthèse)');
    } else {
        updateMetric('risk-kpi-blended', '--', 'Synthèse indisponible (ouvrez le Risk Dashboard)');
    }

    // Timestamp (si dispo) au bas du panneau
    try {
        const ts = ls.timestamp ? new Date(Number(ls.timestamp)).toLocaleTimeString() : null;
        const panel = document.querySelector('#tab-risk .panel-card');
        if (ts && panel) {
            let info = panel.querySelector('.scores-updated-at');
            if (!info) {
                info = document.createElement('div');
                info.className = 'scores-updated-at';
                info.style.cssText = 'text-align:center; font-size:12px; color: var(--theme-text-muted); margin-top:.25rem;';
                panel.appendChild(info);
            }
            info.textContent = `Mis à jour: ${ts}`;
        }
    } catch { }
} async function loadPerformanceData() {
    console.debug('💾 Loading Performance Monitor data...');

    // 🆕 FIX Nov 2025: Multi-tenant support
    const activeUser = localStorage.getItem('activeUser') || 'demo';

    // Performance Monitor is about SYSTEM performance, not financial performance
    let cacheStats = null, memoryStats = null;
    try {
        cacheStats = await fetchWithCache('cache-stats', async () => {
            const response = await fetch(`${API_BASE}/api/performance/cache/stats`, {
                headers: { 'X-User': activeUser }
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        });
    } catch (e) { (window.debugLogger?.warn || console.warn)('cache-stats failed', e); }
    try {
        memoryStats = await fetchWithCache('memory-stats', async () => {
            const response = await fetch(`${API_BASE}/api/performance/system/memory`, {
                headers: { 'X-User': activeUser }
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        });
    } catch (e) { (window.debugLogger?.warn || console.warn)('memory-stats failed', e); }

    if (cacheStats?.success) {
        const cache = cacheStats.cache_stats;
        const memory = memoryStats?.memory_usage || {};

        // Update Performance metrics with real system data
        updateMetric('perf-cache-size', cache?.memory_cache_size ?? '--', 'Memory cache entries');
        updateMetric('perf-disk-cache', `${(cache?.disk_cache_size_mb ?? '--')} MB`, 'Disk cache usage');
        if (memoryStats?.success) {
            updateMetric('perf-memory', `${(memory.rss_mb || 0).toFixed(0)} MB`, 'Process memory');
            const sysUsedPct = (memory.total_system_mb && memory.available_system_mb)
                ? (((memory.total_system_mb - memory.available_system_mb) / memory.total_system_mb) * 100)
                : null;
            updateMetric('perf-system-memory', sysUsedPct != null ? `${sysUsedPct.toFixed(1)}%` : 'N/A', 'System memory usage');
        } else {
            updateMetric('perf-memory', 'N/A', 'psutil non dispo');
            updateMetric('perf-system-memory', 'N/A', 'psutil non dispo');
        }

        // Update performance breakdown
        updatePerformanceBreakdown(cache, memory);

    } else {
        showPerformanceError();
    }
}

async function loadCycleData() {
    console.debug('🔄 Loading Cycle Analysis data...');

    // Import cycle analysis functions (they should be available globally or imported)
    try {
        const cycleModule = await import('./modules/cycle-navigator.js');
        const cycleData = await cycleModule.estimateCyclePosition();

        if (cycleData && cycleData.phase) {
            const phase = cycleData.phase;
            const months = Math.round(cycleData.months || 0);
            const confidence = Math.round((cycleData.confidence || 0) * 100);

            // Update Cycle metrics with real data
            updateMetric('cycle-phase', phase.phase.replace('_', ' ').toUpperCase(), `${phase.emoji} Current phase`);
            updateMetric('cycle-progress', `${months} months`, 'Post-halving progress');
            updateMetric('cycle-score', Math.round(cycleData.score || 50), 'Cycle position score');
            updateMetric('cycle-confidence', `${confidence}%`, 'Model certainty');

            // Update cycle indicators
            updateCycleIndicators(cycleData, phase);

        } else {
            showCycleError();
        }
    } catch (error) {
        debugLogger.error('Cycle data loading failed:', error);
        showCycleError();
    }
}

async function loadMonitoringData() {
    console.debug('📈 Loading Advanced Analytics data...');

    // 🆕 FIX Nov 2025: Multi-tenant support
    const activeUser = localStorage.getItem('activeUser') || 'demo';

    try {
        const url = `${API_BASE}/analytics/advanced/metrics?days=365`;
        const response = await fetch(url, {
            headers: { 'X-User': activeUser }
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        // Update 4 KPIs
        updateMetric('monitor-total-return', `${(data.total_return_pct).toFixed(1)}%`, 'Sur la période');
        updateMetric('monitor-sharpe', (data.sharpe_ratio).toFixed(2), 'Risque ajusté');
        updateMetric('monitor-volatility', `${(data.volatility_pct).toFixed(1)}%`, 'Risque de marché');
        updateMetric('monitor-drawdown', `${Math.abs(data.max_drawdown_pct).toFixed(1)}%`, 'Pire baisse');

        // Breakdown panel
        const breakdown = document.getElementById('advanced-metrics-breakdown');
        if (breakdown) {
            breakdown.innerHTML = `
                  <div style="display:flex; justify-content:space-between;"><span>Volatility:</span><span>${data.volatility_pct.toFixed(1)}%</span></div>
                  <div style="display:flex; justify-content:space-between;"><span>Sortino:</span><span>${data.sortino_ratio.toFixed(2)}</span></div>
                  <div style="display:flex; justify-content:space-between;"><span>Omega:</span><span>${data.omega_ratio.toFixed(2)}</span></div>
                  <div style="display:flex; justify-content:space-between;"><span>Positive Months:</span><span>${data.positive_months_pct.toFixed(1)}%</span></div>
              `;
        }
    } catch (error) {
        debugLogger.error('Advanced analytics loading failed:', error);
        showMonitoringError();
    }
}

// PERFORMANCE FIX (Dec 2025): Preload metric containers at initialization
const metricContainersCache = new Map();

/**
 * Initialize metric containers cache
 * Call this once after DOM is loaded to cache all metric containers
 */
function initMetricContainersCache() {
    const tabs = document.querySelectorAll('[id^="tab-"]');
    tabs.forEach(panel => {
        // Cache containers with data-metric attribute
        const metricsWithAttr = panel.querySelectorAll('[data-metric]');
        metricsWithAttr.forEach(container => {
            const metricId = container.getAttribute('data-metric');
            metricContainersCache.set(metricId, {
                container,
                valueEl: container.querySelector('.metric-value'),
                subtitleEl: container.querySelector('small')
            });
        });

        // Cache positional metric cards as fallback
        const cards = panel.querySelectorAll('.metric-card');
        cards.forEach((card, idx) => {
            const fallbackKey = `${panel.id}_card_${idx}`;
            if (!card.hasAttribute('data-metric')) { // Don't override explicit mappings
                metricContainersCache.set(fallbackKey, {
                    container: card,
                    valueEl: card.querySelector('.metric-value'),
                    subtitleEl: card.querySelector('small')
                });
            }
        });
    });
    console.debug(`✅ Cached ${metricContainersCache.size} metric containers`);
}

// Utility functions
function updateMetric(id, value, subtitle) {
    // PERFORMANCE FIX (Dec 2025): Use cached metric containers instead of querySelector
    let cached = metricContainersCache.get(id);

    // Fallback: try positional mapping if not found
    if (!cached) {
        const tabPrefix = id.split('-')[0];
        const tabMap = { risk: 'risk', perf: 'performance', cycle: 'cycles', monitor: 'monitoring' };
        const panelId = tabMap[tabPrefix] || tabPrefix;
        const idx = getMetricIndex(id) - 1;
        const fallbackKey = `tab-${panelId}_card_${idx}`;
        cached = metricContainersCache.get(fallbackKey);
    }

    if (!cached) return;

    const { valueEl, subtitleEl } = cached;

    if (valueEl) {
        // 🆕 Retirer skeleton loader et aria-busy quand données arrivent
        valueEl.classList.remove('skeleton');
        valueEl.removeAttribute('aria-busy');
        valueEl.textContent = value;
    }
    if (subtitleEl) subtitleEl.textContent = subtitle;
}

function getMetricIndex(id) {
    const indices = {
        'risk-var': 1,
        'risk-drawdown': 2,
        'risk-volatility': 3,
        'risk-score': 4,
        'risk-kpi-diversification': 1,
        'risk-kpi-effective-assets': 2,
        'risk-kpi-blended': 3,
        'risk-kpi-onchain': 4,
        'perf-cache-size': 1,
        'perf-disk-cache': 2,
        'perf-memory': 3,
        'perf-system-memory': 4,
        'cycle-phase': 1,
        'cycle-progress': 2,
        'cycle-score': 3,
        'cycle-confidence': 4,
        'monitor-total-return': 1,
        'monitor-sharpe': 2,
        'monitor-volatility': 3,
        'monitor-drawdown': 4
    };
    return indices[id] || 1;
}

function formatPercent(value) {
    if (value == null || isNaN(value)) return 'N/A';
    return `${(value * 100).toFixed(2)}%`;
}

function getRiskLevel(score) {
    if (!score) return 'Unknown';
    if (score < 30) return 'Low risk';
    if (score < 70) return 'Moderate risk';
    return 'High risk';
}

// Note: updateRiskAlerts() removed - now handled by risk-alerts-loader.js (unified alert system)

function updatePerformanceBreakdown(cache, memory) {
    const breakdownContainer = document.querySelector('#tab-performance .panel-card div:nth-child(3)');
    if (!breakdownContainer) return;

    const memEntries = Number(cache?.memory_cache_size);
    const diskFiles = Number(cache?.disk_cache_files);
    const hitRate = (Number.isFinite(memEntries) && Number.isFinite(diskFiles) && (memEntries + diskFiles) > 0)
        ? ((memEntries / (memEntries + diskFiles)) * 100).toFixed(1) + '%'
        : 'N/A';
    const availGb = Number.isFinite(Number(memory?.available_system_mb)) ? (memory.available_system_mb / 1024).toFixed(1) + ' GB' : 'N/A';
    const procEff = (typeof memory?.percent === 'number')
        ? (memory.percent < 5 ? 'Excellent' : memory.percent < 10 ? 'Good' : 'Average')
        : 'N/A';

    breakdownContainer.innerHTML = `
        <h4>System Performance</h4>
        <div style="display: grid; gap: 0.5rem;">
            <div style="display: flex; justify-content: space-between;"><span>Memory Cache Hit Rate:</span><span style="color: var(--success);">${hitRate}</span></div>
            <div style="display: flex; justify-content: space-between;"><span>Available Memory:</span><span>${availGb}</span></div>
            <div style="display: flex; justify-content: space-between;"><span>Process Efficiency:</span><span>${procEff}</span></div>
        </div>
    `;
}

function updateCycleIndicators(cycleData, phase) {
    const indicatorsContainer = document.querySelector('#tab-cycles .panel-card div:nth-child(3)');
    if (!indicatorsContainer) return;

    indicatorsContainer.innerHTML = `
        <h4>Market Cycle Indicators</h4>
        <div style="display: grid; gap: 0.5rem;">
            <div style="display: flex; justify-content: space-between;"><span>Current Phase:</span><span>${phase.phase.replace('_', ' ')}</span></div>
            <div style="display: flex; justify-content: space-between;"><span>Phase Color:</span><span style="color: ${phase.color};">●</span></div>
            <div style="display: flex; justify-content: space-between;"><span>Score Range:</span><span>${Math.round(cycleData.score)}/100</span></div>
        </div>
    `;
}

// No longer needed; advanced metrics are filled inline above

// Error state functions
function showErrorState(tabId) {
    const panel = document.querySelector(`${tabId} .panel-card`);
    if (panel) {
        panel.innerHTML = `
            <h3>⚠️ Data Loading Error</h3>
            <p style="color: var(--theme-text-muted);">Unable to load data. Please check if the backend server is running.</p>
            <button onclick="loadTabData('${tabId}')" class="btn btn-primary" style="background: var(--brand-primary); color: white; padding: 0.5rem 1rem; border: none; border-radius: 0.25rem; cursor: pointer;">Retry</button>
        `;
    }
}

function showRiskError() {
    showErrorState('#tab-risk');
}

function showPerformanceError() {
    showErrorState('#tab-performance');
}

function showCycleError() {
    showErrorState('#tab-cycles');
}

function showMonitoringError() {
    showErrorState('#tab-monitoring');
}

// 🆕 Smart polling avec Page Visibility API - Nov 2025 optimization
let pollInterval = null;

function startSmartPolling() {
    // Ne pas démarrer si page cachée
    if (document.hidden) {
        console.debug('⏸️ Page hidden - polling paused');
        return;
    }

    // Clear existing interval si présent
    if (pollInterval) {
        clearInterval(pollInterval);
    }

    // Auto-refresh intelligent toutes les 5 minutes
    pollInterval = setInterval(() => {
        // Double-check que la page est toujours visible
        if (document.hidden) {
            console.debug('⏸️ Skip refresh - page hidden');
            return;
        }

        const activeTab = document.querySelector('.tab-panel.active');
        if (activeTab) {
            console.debug(`🔄 Auto-refresh: ${activeTab.id}`);
            // Note: on ne clear PAS le cache - on laisse fetchWithCache gérer le TTL
            loadTabData(`#${activeTab.id}`);
        }
    }, 5 * 60 * 1000);

    console.debug('▶️ Smart polling started (5 min interval)');
}

// Pause/Resume polling selon visibilité de la page
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.debug('👁️ Page hidden - pausing polling');
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
    } else {
        console.debug('👁️ Page visible - resuming polling + immediate refresh');
        // Refresh immédiat au retour sur la page
        const activeTab = document.querySelector('.tab-panel.active');
        if (activeTab) {
            loadTabData(`#${activeTab.id}`);
        }
        // Redémarrer le polling
        startSmartPolling();
    }
});

// Démarrer le polling au chargement
startSmartPolling();

console.debug('✅ Analytics Unified - Initialization complete');





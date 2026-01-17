/**
 * Store utilitaire pour résumer les données Saxo (Bourse)
 * Utilisé par dashboard.html et settings.html pour éviter la duplication
 */
import { safeFetch } from './http.js';

let _cachedSummary = null;
let _cacheTimestamp = 0;
let _cachedForUser = null; // Track which user the cache is for
let _cachedForSource = null; // Track which source the cache is for (CRITICAL!)
const CACHE_TTL = 300000; // 5 minutes (optimized for cross-page sharing)

// LocalStorage keys for cross-page caching
const CACHE_KEY_PREFIX = 'saxo_summary_';

/**
 * Load cache from localStorage (cross-page persistent cache)
 */
function loadCacheFromStorage(activeUser, bourseSource) {
    try {
        const cacheKey = `${CACHE_KEY_PREFIX}${activeUser}_${bourseSource}`;
        const cached = localStorage.getItem(cacheKey);
        if (!cached) return null;

        const { summary, timestamp } = JSON.parse(cached);
        const age = Date.now() - timestamp;

        if (age < CACHE_TTL) {
            (window.debugLogger?.debug || console.log)(`[Saxo Summary] ✅ Loaded from localStorage (age: ${Math.round(age/1000)}s)`);
            return { summary, timestamp };
        } else {
            (window.debugLogger?.debug || console.log)(`[Saxo Summary] ❌ localStorage cache expired (age: ${Math.round(age/1000)}s)`);
            localStorage.removeItem(cacheKey);
            return null;
        }
    } catch (err) {
        (window.debugLogger?.warn || console.warn)('[Saxo Summary] Failed to load from localStorage:', err);
        return null;
    }
}

/**
 * Save cache to localStorage (cross-page persistent cache)
 */
function saveCacheToStorage(activeUser, bourseSource, summary, timestamp) {
    try {
        const cacheKey = `${CACHE_KEY_PREFIX}${activeUser}_${bourseSource}`;
        localStorage.setItem(cacheKey, JSON.stringify({ summary, timestamp }));
        (window.debugLogger?.debug || console.log)(`[Saxo Summary] 💾 Saved to localStorage: ${cacheKey}`);
    } catch (err) {
        (window.debugLogger?.warn || console.warn)('[Saxo Summary] Failed to save to localStorage:', err);
    }
}

/**
 * Récupère un résumé des positions Saxo depuis l'API legacy
 * @returns {Promise<{total_value: number, positions_count: number, asof: string, error?: string}>}
 */
/**
 * Convert Sources V2 balance items to Saxo summary format
 */
function convertV2BalancesToSaxoSummary(items, userId) {
    if (!items || items.length === 0) {
        return {
            total_value: 0,
            positions_count: 0,
            asof: 'Manuel (vide)',
            isEmpty: true,
            source: 'manual'
        };
    }

    // Calculate total value from balance items
    const totalValue = items.reduce((sum, item) => {
        const value = Number(item.value_usd || item.value || 0);
        return sum + value;
    }, 0);

    // Find the most recent timestamp
    const timestamps = items
        .map(item => item.timestamp || item.updated_at)
        .filter(Boolean);
    const latestTimestamp = timestamps.length > 0
        ? Math.max(...timestamps.map(t => new Date(t).getTime()))
        : Date.now();

    const asof = new Date(latestTimestamp).toLocaleString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });

    return {
        total_value: totalValue,
        positions_count: items.length,
        asof: asof || 'Manuel',
        isEmpty: items.length === 0,
        source: 'manual',
        cash_balance: 0, // Manual entries don't track cash separately
        positions: items // Include raw items for detailed views
    };
}

export async function fetchSaxoSummary() {
    const now = Date.now();
    const activeUser = localStorage.getItem('activeUser') || 'demo';

    // Get current source FIRST (before checking cache)
    // ✅ FALLBACK: If wealthContextBar not ready, use localStorage directly
    let bourseSource = window.wealthContextBar?.getContext()?.bourse;

    if (!bourseSource) {
        // Fallback to localStorage (wealthContextBar may not be ready yet)
        bourseSource = localStorage.getItem('bourseSource') || 'api:saxobank_api';
        (window.debugLogger?.warn || console.warn)(`[Saxo Summary] ⚠️ wealthContextBar not ready, using localStorage fallback: ${bourseSource}`);
    }

    (window.debugLogger?.debug || console.log)(`[Saxo Summary] Fetching for user: ${activeUser}, source: ${bourseSource}`);

    // Invalider le cache si l'utilisateur OU la source a changé
    if (_cachedForUser && (_cachedForUser !== activeUser || _cachedForSource !== bourseSource)) {
        (window.debugLogger?.debug || console.log)(`[Saxo Summary] User or source changed, invalidating all caches`);
        _cachedSummary = null;
        _cacheTimestamp = 0;
        _cachedForUser = null;
        _cachedForSource = null;
        // ✅ CRITICAL: Also invalidate localStorage and availableSources cache
        try {
            const oldCacheKey = `${CACHE_KEY_PREFIX}${_cachedForUser}_${_cachedForSource}`;
            localStorage.removeItem(oldCacheKey);
        } catch (e) { /* ignore */ }
        window.availableSources = null;
        window._availableSourcesUser = null;
    }

    // 1️⃣ Check memory cache first (fastest)
    if (_cachedSummary && _cachedForUser === activeUser && _cachedForSource === bourseSource && (now - _cacheTimestamp) < CACHE_TTL) {
        (window.debugLogger?.debug || console.log)(`[Saxo Summary] ⚡ Returning memory cache (age: ${Math.round((now - _cacheTimestamp)/1000)}s)`);
        return _cachedSummary;
    }

    // 2️⃣ Try localStorage cache (cross-page sharing)
    const cachedFromStorage = loadCacheFromStorage(activeUser, bourseSource);
    if (cachedFromStorage) {
        _cachedSummary = cachedFromStorage.summary;
        _cacheTimestamp = cachedFromStorage.timestamp;
        _cachedForUser = activeUser;
        _cachedForSource = bourseSource;
        return _cachedSummary;
    }

    // 3️⃣ No cache available, fetch from API
    (window.debugLogger?.debug || console.log)(`[Saxo Summary] 🌐 No cache, fetching from API...`);

    try {
        // Check if Sources V2 mode (manual_bourse or any V2 source)
        if (bourseSource === 'manual_bourse') {
            (window.debugLogger?.debug || console.log)(`[Saxo Summary] Using Sources V2 mode: ${bourseSource}`);

            try {
                // Use fetch directly (safeFetch may not be loaded yet due to race condition)
                const response = await fetch('/api/sources/v2/bourse/balances', {
                    headers: { 'X-User': activeUser }
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const apiData = await response.json();

                // Debug: log full response
                console.log('🔍 [Saxo Summary] V2 API Response:', {
                    ok: apiData.ok,
                    hasData: !!apiData.data,
                    dataKeys: apiData.data ? Object.keys(apiData.data) : [],
                    items_at_data: apiData.data?.items,
                    count_at_data: apiData.data?.count
                });

                if (!apiData.ok) {
                    const errorMsg = apiData.data?.message || apiData.data?.error || 'API request failed';
                    throw new Error(`Sources V2 API failed: ${errorMsg}`);
                }

                const data = apiData.data;
                if (!data) {
                    throw new Error('No data returned from Sources V2 API');
                }

                // Backend returns success_response: { ok, data: { items: [...], count: N } }
                // safeFetch returns this as-is, so items are at data.data.items
                const items = data.data?.items || data.items || [];
                const summary = convertV2BalancesToSaxoSummary(items, activeUser);

                // Cache and return
                _cachedSummary = summary;
                _cacheTimestamp = now;
                _cachedForUser = activeUser;
                _cachedForSource = bourseSource;
                saveCacheToStorage(activeUser, bourseSource, summary);

                (window.debugLogger?.debug || console.log)(`[Saxo Summary] ✅ V2 manual source loaded: ${items.length} items`);
                return summary;
            } catch (error) {
                (window.debugLogger?.error || console.error)(`[Saxo Summary] V2 API error:`, error);

                // Return empty state with error
                const emptySummary = {
                    total_value: 0,
                    positions_count: 0,
                    asof: 'Erreur V2',
                    isEmpty: true,
                    error: error.message
                };
                _cachedSummary = emptySummary;
                _cacheTimestamp = now;
                _cachedForUser = activeUser;
                _cachedForSource = bourseSource;
                return emptySummary;
            }
        }

        // bourseSource already defined at top of function (line 22)
        let apiUrl = '/api/saxo/positions';

        // Check if API mode (api:saxobank_api)
        if (bourseSource && bourseSource.startsWith('api:')) {
            // API mode: use api-positions endpoint (returns positions array)
            apiUrl = '/api/saxo/api-positions';
            (window.debugLogger?.debug || console.log)(`[Saxo Summary] Using API mode: ${bourseSource}`);
        }
        // Check if CSV mode (saxo:file_key)
        else if (bourseSource && bourseSource !== 'all' && bourseSource.startsWith('saxo:')) {
            // Extract file_key from source (same logic as saxo-dashboard.html)
            const key = bourseSource.substring(5); // Remove 'saxo:' prefix

            try {
                // Load available sources to resolve file_key
                // ✅ CRITICAL: Reload sources if user changed
                if (!window.availableSources || window._availableSourcesUser !== activeUser) {
                    const response = await fetch('/api/users/sources', {
                        headers: { 'X-User': activeUser }
                    });
                    if (response.ok) {
                        const data = await response.json();
                        window.availableSources = data.sources || [];
                        window._availableSourcesUser = activeUser; // Track user
                    }
                }

                // Find matching source and extract filename
                const source = window.availableSources?.find(s => s.key === key);
                if (source?.file_path) {
                    const fileKey = source.file_path.split(/[/\\]/).pop();
                    apiUrl += `?file_key=${encodeURIComponent(fileKey)}`;
                    (window.debugLogger?.debug || console.log)(`[Saxo Summary] Using source-specific file: ${fileKey}`);
                }
            } catch (err) {
                (window.debugLogger?.warn || console.warn)('[Saxo Summary] Failed to resolve file_key:', err);
            }
        }

        (window.debugLogger?.debug || console.log)(`[Saxo Summary] Fetching from: ${apiUrl} for user: ${activeUser}`);

        // ⚠️ IMPORTANT: Don't use globalConfig.getApiUrl() for Saxo API endpoints
        // It adds incompatible params (source, pricing, min_usd) that cause 400 errors
        const finalUrl = apiUrl.includes('/api/saxo/api-')
            ? apiUrl  // Direct URL for Saxo API (no globalConfig params)
            : (window.globalConfig?.getApiUrl(apiUrl) || apiUrl);  // Use globalConfig for CSV endpoints

        const { ok, data } = await safeFetch(
            finalUrl,
            {
                timeout: 30000,  // 30s for API calls (Live mode can be slow)
                headers: {
                    'X-User': activeUser  // ✅ CRITICAL: Always pass user
                }
            }
        );

        // ✅ CRITICAL DEBUG: Log the exact structure of the API response
        (window.debugLogger?.debug || console.log)(`[Saxo Summary] API response:`, {
            ok,
            hasData: !!data,
            hasDataData: !!data?.data,
            dataDataKeys: data?.data ? Object.keys(data.data) : [],
            positionsPath1: Array.isArray(data?.data?.positions) ? `data.data.positions[${data.data.positions.length}]` : 'not found',
            positionsPath2: Array.isArray(data?.positions) ? `data.positions[${data.positions.length}]` : 'not found',
            totalValuePath: typeof data?.data?.total_value === 'number' ? data.data.total_value : 'not found',
            cashBalancePath: typeof data?.data?.cash_balance === 'number' ? data.data.cash_balance : 'not found'
        });

        if (!ok || !data) {
            // Différencier les types d'erreur pour un meilleur message utilisateur
            let errorMessage = 'API Saxo temporairement indisponible';
            let errorDetail = data?.error || 'unknown error';

            // Cas spécifique: utilisateur non connecté (401)
            if (data?.error && (data.error.includes('Not connected') || data.error.includes('tokens expired'))) {
                errorMessage = 'Non connecté à Saxo API';
                errorDetail = 'Veuillez vous connecter dans Paramètres > Sources > SaxoBank API';
                (window.debugLogger?.warn || console.warn)(`[Saxo Summary] User not connected to Saxo API: ${data.error}`);
            } else {
                (window.debugLogger?.warn || console.warn)(`[Saxo Summary] API returned error: ${errorDetail}`);
            }

            const emptySummary = {
                total_value: 0,
                positions_count: 0,
                asof: errorMessage,
                isEmpty: true,
                error: errorDetail,
                needsConnection: data?.error && (data.error.includes('Not connected') || data.error.includes('tokens expired'))
            };
            _cachedSummary = emptySummary;
            _cacheTimestamp = now;
            _cachedForUser = activeUser;
            _cachedForSource = bourseSource;
            saveCacheToStorage(activeUser, bourseSource, emptySummary, now);
            return emptySummary;
        }

        // ✅ FIXED: safeFetch wraps backend response, so we need data.data.positions
        const positions = Array.isArray(data.data?.positions) ? data.data.positions :
                         Array.isArray(data.positions) ? data.positions :
                         Array.isArray(data) ? data : [];

        (window.debugLogger?.debug || console.log)(`[Saxo Summary] ✅ Positions extracted:`, {
            count: positions.length,
            source: Array.isArray(data.data?.positions) ? 'data.data.positions' :
                    Array.isArray(data.positions) ? 'data.positions' :
                    Array.isArray(data) ? 'data (array)' : 'unknown',
            firstPosition: positions[0] ? {symbol: positions[0].symbol || positions[0].asset_name, value: positions[0].market_value_usd || positions[0].value} : null
        });

        if (positions.length === 0) {
            const summary = {
                total_value: 0,
                positions_count: 0,
                asof: 'Aucune donnée',
                isEmpty: true
            };
            _cachedSummary = summary;
            _cacheTimestamp = now;
            _cachedForUser = activeUser;
            _cachedForSource = bourseSource;
            saveCacheToStorage(activeUser, bourseSource, summary, now);
            return summary;
        }

        // ✅ CRITICAL: Always use backend total_value if available (includes cash + positions)
        let totalValue = 0;
        let cashBalance = 0;

        // Check if backend already calculated total_value (API mode)
        if (typeof data.data?.total_value === 'number') {
            // Backend provides total_value - use it directly (includes positions + cash)
            totalValue = data.data.total_value;
            cashBalance = data.data?.cash_balance || 0;
            (window.debugLogger?.debug || console.log)(`[Saxo Summary] ✅ Using backend values: total=$${totalValue.toFixed(2)}, cash=$${cashBalance.toFixed(2)}`);
        } else {
            // CSV mode: calculate manually from positions
            totalValue = positions.reduce((sum, pos) => {
                const value = Number(pos.market_value_usd || pos.market_value || pos.value || 0);
                return sum + value;
            }, 0);

            (window.debugLogger?.debug || console.log)(`[Saxo Summary] Manual calculation - positions total: $${totalValue.toFixed(2)}`);

            // Add cash/liquidities
            try {
                // Construct URL with user_id (required) + file_key (optional)
                let cashUrl = `/api/saxo/cash?user_id=${encodeURIComponent(activeUser)}`;

                if (bourseSource && bourseSource !== 'all' && bourseSource.startsWith('saxo:')) {
                    const key = bourseSource.substring(5);
                    const source = window.availableSources?.find(s => s.key === key);
                    if (source?.file_path) {
                        const fileKey = source.file_path.split(/[/\\]/).pop();
                        cashUrl += `&file_key=${encodeURIComponent(fileKey)}`;
                    }
                }

                const cashResponse = await safeFetch(cashUrl, {
                    timeout: 3000,
                    headers: { 'X-User': activeUser }
                });

                if (cashResponse?.ok && cashResponse.data?.cash_amount) {
                    cashBalance = Number(cashResponse.data.cash_amount || 0);
                    totalValue += cashBalance;
                    (window.debugLogger?.debug || console.log)(`[Saxo Summary] ✅ Added cash: $${cashBalance}, new total: $${totalValue}`);
                }
            } catch (cashError) {
                // Non-blocking: continue without cash if endpoint fails
                (window.debugLogger?.debug || console.log)('[Saxo Summary] Cash fetch failed (non-blocking):', cashError.message);
            }
        }

        // Trouver la date la plus récente (asof)
        let latestDate = 'Date inconnue';
        try {
            const dates = positions
                .map(pos => pos.asof || pos.date || pos.timestamp)
                .filter(d => d)
                .sort((a, b) => new Date(b) - new Date(a));

            if (dates.length > 0) {
                const date = new Date(dates[0]);
                latestDate = date.toLocaleDateString('fr-FR', {
                    day: '2-digit',
                    month: '2-digit',
                    year: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                });
            }
        } catch (e) {
            // Fallback: extraire de portfolio_id si disponible
            const portfolioId = data.portfolio_id || data.portfolios?.[0]?.portfolio_id || '';
            if (portfolioId.includes('25-09-2025')) {
                latestDate = '25/09/2025 18:40';
            }
        }

        const summary = {
            total_value: totalValue,
            positions_count: positions.length,
            asof: latestDate,
            isEmpty: false
        };

        _cachedSummary = summary;
        _cacheTimestamp = now;
        _cachedForUser = activeUser;
        _cachedForSource = bourseSource;
        saveCacheToStorage(activeUser, bourseSource, summary, now);

        return summary;

    } catch (error) {
        (window.debugLogger?.warn || console.warn)('[Saxo Summary] Erreur fetch:', error.message);

        const errorSummary = {
            total_value: 0,
            positions_count: 0,
            asof: 'Erreur',
            error: error?.message || String(error) || 'unknown error',
            isEmpty: true
        };

        // Cache l'erreur pour éviter les appels répétés
        _cachedSummary = errorSummary;
        _cacheTimestamp = now;
        _cachedForUser = activeUser;
        _cachedForSource = bourseSource;
        saveCacheToStorage(activeUser, bourseSource, errorSummary, now);
        return errorSummary;
    }
}

/**
 * Force le rechargement du cache (utilisé après upload ou changement de source)
 * @param {boolean} clearAll - Si true, efface TOUS les caches Saxo (tous users)
 */
export function invalidateSaxoCache(clearAll = false) {
    _cachedSummary = null;
    _cacheTimestamp = 0;
    _cachedForUser = null;
    _cachedForSource = null;

    // ✅ CRITICAL: Also clear localStorage caches for saxo
    try {
        const keysToRemove = [];
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && key.startsWith(CACHE_KEY_PREFIX)) {
                if (clearAll) {
                    // Clear ALL Saxo caches (all users/sources)
                    keysToRemove.push(key);
                } else {
                    // Clear only current user's cache
                    const activeUser = localStorage.getItem('activeUser') || 'demo';
                    if (key.includes(activeUser)) {
                        keysToRemove.push(key);
                    }
                }
            }
        }
        keysToRemove.forEach(key => localStorage.removeItem(key));
        (window.debugLogger?.debug || console.log)(`[Saxo Summary] Cleared ${keysToRemove.length} localStorage cache entries (clearAll=${clearAll})`);
    } catch (err) {
        (window.debugLogger?.warn || console.warn)('[Saxo Summary] Failed to clear localStorage:', err);
    }

    // ✅ CRITICAL: Also invalidate availableSources cache
    window.availableSources = null;
    window._availableSourcesUser = null;
    (window.debugLogger?.debug || console.log)('[Saxo Summary] All caches invalidated');
}

// ✅ EXPOSE globally for debug console access
window.invalidateSaxoCache = invalidateSaxoCache;

/**
 * Formate une valeur monétaire pour l'affichage (wrapper simple pour USD)
 * Note: Kept local for backward compatibility with dashboard-main-controller.js
 * @param {number} value - Valeur en USD
 * @returns {string} - Valeur formatée (ex: "$1,234.56")
 */
export function formatCurrency(value) {
    if (!Number.isFinite(value) || value === 0) {
        return '$0';
    }

    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: value >= 1000 ? 0 : 2
    }).format(value);
}

/**
 * Détermine la couleur pour les métriques selon la valeur
 * @param {number} value - Valeur numérique
 * @returns {string} - Classe CSS ou couleur
 */
export function getMetricColor(value) {
    if (!Number.isFinite(value) || value === 0) {
        return 'var(--theme-text-muted)';
    }
    return value > 0 ? 'var(--success)' : 'var(--theme-text)';
}
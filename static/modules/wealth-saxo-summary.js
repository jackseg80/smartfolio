/**
 * Store utilitaire pour résumer les données Saxo (Bourse)
 * Utilisé par dashboard.html et settings.html pour éviter la duplication
 */
import { safeFetch } from './http.js';

let _cachedSummary = null;
let _cacheTimestamp = 0;
let _cachedForUser = null; // Track which user the cache is for
let _cachedForSource = null; // Track which source the cache is for (CRITICAL!)
const CACHE_TTL = 30000; // 30 secondes

/**
 * Récupère un résumé des positions Saxo depuis l'API legacy
 * @returns {Promise<{total_value: number, positions_count: number, asof: string, error?: string}>}
 */
export async function fetchSaxoSummary() {
    const now = Date.now();
    const activeUser = localStorage.getItem('activeUser') || 'demo';

    // Get current source FIRST (before checking cache)
    const bourseSource = window.wealthContextBar?.getContext()?.bourse;

    (window.debugLogger?.debug || console.log)(`[Saxo Summary] Fetching for user: ${activeUser}, source: ${bourseSource}, cached for: ${_cachedForUser || 'none'}/${_cachedForSource || 'none'}`);

    // Invalider le cache si l'utilisateur OU la source a changé
    if (_cachedForUser && (_cachedForUser !== activeUser || _cachedForSource !== bourseSource)) {
        (window.debugLogger?.debug || console.log)(`[Saxo Summary] User or source changed (${_cachedForUser}/${_cachedForSource} → ${activeUser}/${bourseSource}), invalidating cache`);
        _cachedSummary = null;
        _cacheTimestamp = 0;
        _cachedForUser = null;
        _cachedForSource = null;
        // ✅ CRITICAL: Also invalidate availableSources cache
        window.availableSources = null;
        window._availableSourcesUser = null;
    }

    // Retourner le cache si valide pour cet utilisateur ET cette source
    if (_cachedSummary && _cachedForUser === activeUser && _cachedForSource === bourseSource && (now - _cacheTimestamp) < CACHE_TTL) {
        (window.debugLogger?.debug || console.log)(`[Saxo Summary] Returning cached data for ${activeUser}/${bourseSource}`);
        return _cachedSummary;
    }

    try {
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

        if (!ok || !data) {
            (window.debugLogger?.warn || console.warn)('[Saxo Summary] API returned error, using empty state');
            const emptySummary = {
                total_value: 0,
                positions_count: 0,
                asof: 'API indisponible',
                isEmpty: true,
                error: 'API Saxo temporairement indisponible'
            };
            _cachedSummary = emptySummary;
            _cacheTimestamp = now;
            _cachedForUser = activeUser;
        _cachedForSource = bourseSource;
            _cachedForSource = bourseSource;
            return emptySummary;
        }

        const positions = Array.isArray(data.positions) ? data.positions :
                         Array.isArray(data) ? data : [];

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
            _cachedForSource = bourseSource;
            return summary;
        }

        // Calculer la valeur totale des positions
        let totalValue = positions.reduce((sum, pos) => {
            const value = Number(pos.market_value_usd || pos.market_value || pos.value || 0);
            return sum + value;
        }, 0);

        // Récupérer et ajouter le cash/liquidités
        try {
            // Construire l'URL avec user_id (obligatoire) + file_key (optionnel)
            let cashUrl = `/api/saxo/cash?user_id=${encodeURIComponent(activeUser)}`;
            const bourseSource = window.wealthContextBar?.getContext()?.bourse;

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
                const cashAmount = Number(cashResponse.data.cash_amount || 0);
                totalValue += cashAmount;
                (window.debugLogger?.debug || console.log)(`[Saxo Summary] Added cash: $${cashAmount}, new total: $${totalValue}`);
            }
        } catch (cashError) {
            // Non-bloquant : on continue sans le cash si l'endpoint n'existe pas ou échoue
            (window.debugLogger?.debug || console.log)('[Saxo Summary] Cash fetch failed (non-blocking):', cashError.message);
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
        return summary;

    } catch (error) {
        (window.debugLogger?.warn || console.warn)('[Saxo Summary] Erreur fetch:', error.message);

        const errorSummary = {
            total_value: 0,
            positions_count: 0,
            asof: 'Erreur',
            error: error.message,
            isEmpty: true
        };

        // Cache l'erreur pour éviter les appels répétés
        _cachedSummary = errorSummary;
        _cacheTimestamp = now;
        _cachedForUser = activeUser;
        _cachedForSource = bourseSource;
        return errorSummary;
    }
}

/**
 * Force le rechargement du cache (utilisé après upload)
 */
export function invalidateSaxoCache() {
    _cachedSummary = null;
    _cacheTimestamp = 0;
    _cachedForUser = null;
}

/**
 * Formate une valeur monétaire pour l'affichage
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
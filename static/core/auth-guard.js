/**
 * Auth Guard - Protection d'authentification JWT pour les pages
 *
 * Usage dans chaque page HTML:
 *
 * <script type="module">
 *   import { checkAuth, logout, getAuthHeaders } from './core/auth-guard.js';
 *
 *   // Vérifier authentification au chargement
 *   await checkAuth();
 *
 *   // Utiliser dans les fetch
 *   const response = await fetch('/api/endpoint', {
 *     headers: getAuthHeaders()
 *   });
 * </script>
 */

import { StorageService } from './storage-service.js';

const API_BASE = window.location.origin;

// Pages publiques (ne nécessitent pas d'authentification)
const PUBLIC_PAGES = ['/static/login.html', '/login.html'];
const AUTH_RETRY_EXCLUDED_PATHS = new Set([
    '/auth/login',
    '/auth/logout',
    '/auth/refresh',
    '/auth/session',
    '/auth/verify'
]);
let sessionRefreshInFlight = null;
let lastSessionRefreshAt = 0;

function redirectExpiredSession() {
    if (isPublicPage() || window.__smartfolioAuthRedirecting) return;

    window.__smartfolioAuthRedirecting = true;
    StorageService.clearAuth();
    StorageService.remove('userInfo');

    const loginUrl = '/static/login.html?message=session_expired';
    try {
        window.location.replace(loginUrl);
    } catch (_) {
        window.location.href = loginUrl;
    }
}

async function ensureFreshCookieSession() {
    if (Date.now() - lastSessionRefreshAt < 5000) return true;
    if (!sessionRefreshInFlight) {
        sessionRefreshInFlight = refreshCookieSession()
            .then(refreshed => {
                if (refreshed) lastSessionRefreshAt = Date.now();
                return refreshed;
            })
            .catch(() => false)
            .finally(() => { sessionRefreshInFlight = null; });
    }
    return sessionRefreshInFlight;
}

/**
 * Vérifie si la page actuelle est publique
 */
function isPublicPage() {
    const currentPath = window.location.pathname;
    return PUBLIC_PAGES.some(page => currentPath.endsWith(page));
}

/**
 * Récupère le token JWT stocké
 */
export function getAuthToken() {
    return StorageService.getAuthToken();
}

/**
 * Récupère l'utilisateur actuel
 */
export function getCurrentUser() {
    return StorageService.getActiveUser();
}

/**
 * Récupère les informations de l'utilisateur
 */
export function getUserInfo() {
    try {
        const userInfoStr = localStorage.getItem('userInfo');
        return userInfoStr ? JSON.parse(userInfoStr) : null;
    } catch (err) {
        console.error('Error parsing userInfo:', err);
        return null;
    }
}

/**
 * Génère les headers d'authentification pour fetch
 *
 * Supporte à la fois JWT (nouveau) et X-User (legacy)
 *
 * @param {boolean} includeXUser - Inclure X-User header pour compatibilité (default: true)
 * @returns {Object} Headers object
 */
export function getAuthHeaders(includeXUser = true) {
    const headers = {};

    const token = getAuthToken();
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    const currentUser = getCurrentUser();
    if (includeXUser && currentUser) {
        headers['X-User'] = currentUser;
    }
    const csrfCookie = document.cookie
        .split('; ')
        .find(value => value.startsWith('smartfolio_csrf='));
    if (csrfCookie) {
        headers['X-CSRF-Token'] = decodeURIComponent(csrfCookie.split('=').slice(1).join('='));
    }

    return headers;
}

/**
 * Wrap fetch so legacy same-origin calls inherit the current authentication
 * and CSRF headers during the dual-to-cookie migration. Cross-origin requests
 * are passed through unchanged to prevent credential leakage.
 */
export function createAuthenticatedFetch(fetchImplementation) {
    return async function authenticatedFetch(input, init = {}) {
        const request = (
            typeof Request !== 'undefined' && input instanceof Request
        ) ? input : null;
        const target = new URL(request?.url || String(input), window.location.href);

        if (target.origin !== window.location.origin) {
            return fetchImplementation(input, init);
        }

        const headers = new Headers(request?.headers);
        new Headers(init.headers).forEach((value, name) => headers.set(name, value));
        const authHeaders = getAuthHeaders();
        Object.entries(authHeaders).forEach(([name, value]) => {
            if (!headers.has(name)) headers.set(name, value);
        });

        const requestOptions = {
            ...init,
            credentials: init.credentials || request?.credentials || 'same-origin',
            headers
        };
        const retryInput = request ? request.clone() : input;
        const response = await fetchImplementation(input, requestOptions);

        if (response.status !== 401 || AUTH_RETRY_EXCLUDED_PATHS.has(target.pathname)) {
            return response;
        }

        const refreshed = await ensureFreshCookieSession();
        if (!refreshed) {
            redirectExpiredSession();
            return response;
        }

        // The first request was rejected before executing its operation. Retry
        // it once with the renewed cookie, bearer and CSRF credentials.
        const renewedAuthHeaders = getAuthHeaders();
        Object.entries(renewedAuthHeaders).forEach(([name, value]) => {
            requestOptions.headers.set(name, value);
        });
        const retryResponse = await fetchImplementation(retryInput, requestOptions);
        if (retryResponse.status === 401) redirectExpiredSession();
        return retryResponse;
    };
}

export function installAuthenticatedFetch() {
    if (
        typeof window === 'undefined'
        || typeof window.fetch !== 'function'
        || window.__smartfolioAuthenticatedFetchInstalled
    ) {
        return;
    }
    const nativeFetch = window.fetch.bind(window);
    window.fetch = createAuthenticatedFetch(nativeFetch);
    window.__smartfolioAuthenticatedFetchInstalled = true;
}

installAuthenticatedFetch();

async function refreshCookieSession() {
    const csrfCookie = document.cookie
        .split('; ')
        .find(value => value.startsWith('smartfolio_csrf='));
    if (!csrfCookie) return false;

    const csrfToken = decodeURIComponent(csrfCookie.split('=').slice(1).join('='));
    const response = await fetch(`${API_BASE}/auth/refresh`, {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'X-CSRF-Token': csrfToken }
    });
    if (!response.ok) return false;

    const data = await response.json();
    const user = data.data?.user;
    const token = data.data?.token;
    if (token) StorageService.setAuthToken(token);
    if (user) {
        StorageService.setActiveUser(user.id);
        localStorage.setItem('userInfo', JSON.stringify(user));
    }
    return true;
}

/**
 * Vérifie si le token JWT est valide
 *
 * @returns {Promise<boolean>} True si authentifié, false sinon
 */
export async function verifyToken() {
    const token = getAuthToken();

    try {
        const sessionResponse = await fetch(`${API_BASE}/auth/session`, {
            credentials: 'same-origin',
            headers: getAuthHeaders()
        });
        if (sessionResponse.ok) {
            const sessionData = await sessionResponse.json();
            const user = sessionData.data?.user;
            if (user) {
                StorageService.setActiveUser(user.id);
                localStorage.setItem('userInfo', JSON.stringify(user));
            }
            return true;
        }
        if (sessionResponse.status === 401 && await refreshCookieSession()) {
            return true;
        }
        if (!token) return false;

        const response = await fetch(`${API_BASE}/auth/verify`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (!response.ok) {
            return false;
        }

        const data = await response.json();
        return data.ok && data.data?.valid;
    } catch (err) {
        console.error('Token verification error:', err);
        return false;
    }
}

/**
 * Déconnecte l'utilisateur et redirige vers login
 *
 * @param {boolean} showMessage - Afficher un message de déconnexion (default: false)
 */
export async function logout(showMessage = false) {
    try {
        await fetch(`${API_BASE}/auth/logout`, {
            method: 'POST',
            credentials: 'same-origin',
            headers: getAuthHeaders(false)
        });
    } catch (err) {
        console.debug('Logout endpoint error:', err);
    }

    // 🔒 FIX: Capture currentUser avant de vider localStorage
    const currentUser = getCurrentUser();

    // Clear auth data via StorageService
    StorageService.clearAuth();
    StorageService.remove('userInfo');

    // Clear caches
    if (window.clearCache) {
        window.clearCache();
    }

    // Clear data caches (NOT user settings - they're isolated by user)
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        // 🔒 FIX: Ne PAS supprimer smartfolio_settings_* (isolées par user)
        // Chaque user garde ses propres settings
        if (key && (key.startsWith('risk_score') || key.startsWith('cache:') || key.startsWith('portfolio_'))) {
            keysToRemove.push(key);
        }
    }
    keysToRemove.forEach(key => StorageService.remove(key));

    console.debug(`✅ Logged out user: ${currentUser} (settings preserved for future login)`);

    console.debug('User logged out');

    // Redirect to login
    const redirectUrl = '/static/login.html';
    if (showMessage) {
        window.location.href = `${redirectUrl}?message=logged_out`;
    } else {
        window.location.href = redirectUrl;
    }
}

/**
 * Vérifie l'authentification et redirige si nécessaire
 *
 * À appeler au chargement de chaque page protégée
 *
 * @param {Object} options - Options de vérification
 * @param {boolean} options.skipDevMode - Ne pas bypass en mode DEV (default: false)
 * @returns {Promise<Object>} User info si authentifié
 */
export async function checkAuth(options = {}) {
    const { skipDevMode = false } = options;

    // Skip si page publique
    if (isPublicPage()) {
        return null;
    }

    // NOTE: DEV_SKIP_AUTH est géré uniquement backend (api/deps.py)
    // Le frontend ne doit PAS vérifier ce mode pour des raisons de sécurité

    // Cookie sessions are attempted first; legacy JWT remains a transition fallback.
    const isValid = await verifyToken();
    if (!isValid) {
        console.warn('Invalid or expired token, redirecting to login');
        redirectExpiredSession();
        return null;
    }

    // Token valide, retourner user info
    const userInfo = getUserInfo();
    console.debug('Authenticated as:', userInfo?.label || getCurrentUser());
    return userInfo;
}

/**
 * Vérifie si l'utilisateur a un rôle spécifique
 *
 * @param {string} role - Rôle à vérifier (e.g., 'admin', 'ml_admin')
 * @returns {boolean} True si l'utilisateur a le rôle
 */
export function hasRole(role) {
    const userInfo = getUserInfo();
    if (!userInfo || !userInfo.roles) {
        return false;
    }
    return userInfo.roles.includes(role);
}

/**
 * Vérifie si l'utilisateur est admin
 *
 * @returns {boolean} True si admin
 */
export function isAdmin() {
    return hasRole('admin');
}

/**
 * Redirige vers login si l'utilisateur n'a pas le rôle requis
 *
 * @param {string} requiredRole - Rôle requis (e.g., 'admin')
 * @param {string} message - Message d'erreur personnalisé
 */
export function requireRole(requiredRole, message = 'Insufficient permissions') {
    if (!hasRole(requiredRole)) {
        console.error(`Access denied: ${message}`);
        alert(`Access denied: ${message}`);
        window.location.href = '/static/dashboard.html';
    }
}

// Export global pour compatibilité legacy
if (typeof window !== 'undefined') {
    window.authGuard = {
        checkAuth,
        logout,
        getAuthHeaders,
        getAuthToken,
        getCurrentUser,
        getUserInfo,
        verifyToken,
        hasRole,
        isAdmin,
        requireRole
    };
}

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

const API_BASE = window.location.origin;

// Pages publiques (ne nécessitent pas d'authentification)
const PUBLIC_PAGES = ['/static/login.html', '/login.html'];

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
    return localStorage.getItem('authToken');
}

/**
 * Récupère l'utilisateur actuel
 */
export function getCurrentUser() {
    return localStorage.getItem('activeUser') || 'demo';
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

    if (includeXUser) {
        headers['X-User'] = getCurrentUser();
    }

    return headers;
}

/**
 * Vérifie si le token JWT est valide
 *
 * @returns {Promise<boolean>} True si authentifié, false sinon
 */
export async function verifyToken() {
    const token = getAuthToken();

    if (!token) {
        return false;
    }

    try {
        const response = await fetch(`${API_BASE}/auth/verify?token=${encodeURIComponent(token)}`);

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
    const token = getAuthToken();

    // Appeler l'endpoint logout (optionnel, pour logs serveur)
    if (token) {
        try {
            await fetch(`${API_BASE}/auth/logout`, {
                method: 'POST',
                headers: getAuthHeaders(false)
            });
        } catch (err) {
            console.debug('Logout endpoint error:', err);
        }
    }

    // Clear localStorage
    localStorage.removeItem('authToken');
    localStorage.removeItem('activeUser');
    localStorage.removeItem('userInfo');

    // Clear caches
    if (window.clearCache) {
        window.clearCache();
    }

    // Clear data caches
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && (key.startsWith('risk_score') || key.startsWith('cache:') || key.startsWith('portfolio_'))) {
            keysToRemove.push(key);
        }
    }
    keysToRemove.forEach(key => localStorage.removeItem(key));

    console.log('User logged out');

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

    // Vérifier présence du token
    const token = getAuthToken();
    if (!token) {
        console.warn('No auth token found, redirecting to login');
        window.location.href = '/static/login.html';
        return null;
    }

    // Vérifier validité du token
    const isValid = await verifyToken();
    if (!isValid) {
        console.warn('Invalid or expired token, redirecting to login');
        localStorage.removeItem('authToken');
        localStorage.removeItem('activeUser');
        localStorage.removeItem('userInfo');
        window.location.href = '/static/login.html?message=session_expired';
        return null;
    }

    // Token valide, retourner user info
    const userInfo = getUserInfo();
    console.log('Authenticated as:', userInfo?.label || getCurrentUser());
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

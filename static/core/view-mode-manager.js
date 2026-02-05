/**
 * ViewModeManager - Gestion du mode de visualisation Simple/Pro
 *
 * Permet de basculer entre deux niveaux d'affichage :
 * - Simple : Vue executive summary (métriques clés, graphiques simples)
 * - Pro : Vue complète Bloomberg-style (toutes données, tables, détails techniques)
 *
 * Usage:
 *   import { ViewModeManager } from './core/view-mode-manager.js';
 *
 *   // Initialiser (applique le mode sauvegardé)
 *   ViewModeManager.init();
 *
 *   // Changer de mode
 *   ViewModeManager.setMode('simple');
 *   ViewModeManager.toggle();
 *
 *   // Écouter les changements
 *   ViewModeManager.on('change', (mode) => console.log('Mode:', mode));
 *
 * @module ViewModeManager
 * @version 1.0.0
 * @since Feb 2026
 */

const STORAGE_KEY = 'smartfolio_view_mode';
const DEFAULT_MODE = 'pro';

export const ViewModes = Object.freeze({
    SIMPLE: 'simple',
    PRO: 'pro'
});

class ViewModeManagerClass {
    constructor() {
        this._listeners = new Set();
        this._initialized = false;
    }

    /**
     * Initialise le ViewModeManager et applique le mode sauvegardé
     * @returns {string} Le mode actuel
     */
    init() {
        if (this._initialized) {
            return this.getMode();
        }

        const mode = this.getMode();
        this._applyMode(mode);
        this._initialized = true;

        // Écouter les changements depuis d'autres onglets
        window.addEventListener('storage', (e) => {
            if (e.key === STORAGE_KEY && e.newValue) {
                this._applyMode(e.newValue);
                this._notifyListeners(e.newValue);
            }
        });

        console.debug('[ViewModeManager] Initialized with mode:', mode);
        return mode;
    }

    /**
     * Récupère le mode actuel depuis localStorage
     * @returns {string} 'simple' ou 'pro'
     */
    getMode() {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored && Object.values(ViewModes).includes(stored)) {
                return stored;
            }
        } catch (e) {
            console.warn('[ViewModeManager] localStorage unavailable:', e);
        }
        return DEFAULT_MODE;
    }

    /**
     * Définit le mode de visualisation
     * @param {string} mode - 'simple' ou 'pro'
     * @returns {boolean} true si le mode a changé
     */
    setMode(mode) {
        if (!Object.values(ViewModes).includes(mode)) {
            console.error('[ViewModeManager] Invalid mode:', mode);
            return false;
        }

        const currentMode = this.getMode();
        if (currentMode === mode) {
            return false;
        }

        try {
            localStorage.setItem(STORAGE_KEY, mode);
        } catch (e) {
            console.warn('[ViewModeManager] Could not save to localStorage:', e);
        }

        this._applyMode(mode);
        this._notifyListeners(mode);

        console.debug('[ViewModeManager] Mode changed to:', mode);
        return true;
    }

    /**
     * Bascule entre les modes Simple et Pro
     * @returns {string} Le nouveau mode
     */
    toggle() {
        const current = this.getMode();
        const newMode = current === ViewModes.PRO ? ViewModes.SIMPLE : ViewModes.PRO;
        this.setMode(newMode);
        return newMode;
    }

    /**
     * Vérifie si le mode actuel est Simple
     * @returns {boolean}
     */
    isSimple() {
        return this.getMode() === ViewModes.SIMPLE;
    }

    /**
     * Vérifie si le mode actuel est Pro
     * @returns {boolean}
     */
    isPro() {
        return this.getMode() === ViewModes.PRO;
    }

    /**
     * Ajoute un listener pour les changements de mode
     * @param {string} event - 'change'
     * @param {Function} callback - Fonction appelée avec le nouveau mode
     * @returns {Function} Fonction pour retirer le listener
     */
    on(event, callback) {
        if (event !== 'change') {
            console.warn('[ViewModeManager] Unknown event:', event);
            return () => {};
        }

        this._listeners.add(callback);
        return () => this._listeners.delete(callback);
    }

    /**
     * Retire un listener
     * @param {Function} callback
     */
    off(callback) {
        this._listeners.delete(callback);
    }

    /**
     * Applique le mode au DOM (data-view-mode sur body)
     * @private
     */
    _applyMode(mode) {
        document.body.setAttribute('data-view-mode', mode);

        // Dispatch custom event pour les composants qui écoutent
        window.dispatchEvent(new CustomEvent('viewmode:change', {
            detail: { mode, isSimple: mode === ViewModes.SIMPLE, isPro: mode === ViewModes.PRO }
        }));
    }

    /**
     * Notifie tous les listeners du changement
     * @private
     */
    _notifyListeners(mode) {
        this._listeners.forEach(callback => {
            try {
                callback(mode);
            } catch (e) {
                console.error('[ViewModeManager] Listener error:', e);
            }
        });
    }
}

// Singleton export
export const ViewModeManager = new ViewModeManagerClass();

// Export global pour usage sans import
if (typeof window !== 'undefined') {
    window.ViewModeManager = ViewModeManager;
    window.ViewModes = ViewModes;
}

/**
 * Système de logging centralisé pour Crypto Rebalancer
 * 
 * Permet d'activer/désactiver les logs de debug en production
 * tout en gardant les erreurs critiques visibles.
 */

class DebugLogger {
    constructor() {
        // Vérifier si on est en mode debug via localStorage ou variable globale
        this.debugEnabled = this.isDebugEnabled();
        this._consolePatched = false;
        this._fetchPatched = false;

        // Niveaux de log
        this.LEVELS = {
            ERROR: 0,   // Toujours affiché
            WARN: 1,    // Toujours affiché
            INFO: 2,    // Affiché si debug activé
            DEBUG: 3    // Affiché si debug activé
        };

        // Toast integration (lazy loaded)
        this._toastEnabled = true; // Afficher les toasts par défaut
        this._toastInstance = null;
        this._loadToast();

        // Note: Can't use debugLogger.debug here since debugLogger isn't created yet
        if (this.debugEnabled) {
            console.log(`DebugLogger initialized - Debug mode: ${this.debugEnabled ? 'ON' : 'OFF'}, Toasts: ${this._toastEnabled ? 'ON' : 'OFF'}`);
        }

        // Synchroniser avec globalConfig si présent
        try {
            window.addEventListener('debugModeChanged', (e) => {
                const enabled = !!e?.detail?.enabled;
                this.setDebugMode(enabled);
            });
        } catch (_) {}

        // Appliquer les hooks (console.debug, fetch tracer)
        this.applyConsoleOverride();
        this.applyFetchTracer();
    }

    /**
     * Charge le système Toast de manière asynchrone
     */
    async _loadToast() {
        try {
            // Attendre que Toast soit disponible (chargé par module ou global)
            if (window.Toast) {
                this._toastInstance = window.Toast;
            } else {
                // Essayer d'importer dynamiquement
                const module = await import('./components/toast.js');
                this._toastInstance = module.Toast;
            }
        } catch (error) {
            // Toast pas disponible, on continue sans
            this._toastEnabled = false;
            if (this.debugEnabled) {
                console.log("Toast system not available, using console only");
            }
        }
    }

    /**
     * Active/désactive les toasts visuels
     */
    setToastEnabled(enabled) {
        this._toastEnabled = enabled;
        localStorage.setItem('debug_toast_enabled', enabled.toString());
    }

    /**
     * Affiche un toast si disponible
     */
    _showToast(type, message) {
        if (!this._toastEnabled || !this._toastInstance) return;

        try {
            // Retirer le préfixe de niveau, déjà représenté par le toast
            const cleanMessage = message.replace(/^\[(?:Warning|Error|Info)\]\s*/, '');

            // Limiter la longueur du message
            const shortMessage = cleanMessage.length > 150
                ? cleanMessage.substring(0, 147) + '...'
                : cleanMessage;

            // Afficher selon le type
            if (type === 'error') {
                this._toastInstance.error(shortMessage, { duration: 8000 });
            } else if (type === 'warn') {
                this._toastInstance.warning(shortMessage, { duration: 6000 });
            }
        } catch (e) {
            // Toast failed, silent fallback
            if (this.debugEnabled) {
                console.log('Toast display failed:', e);
            }
        }
    }
    
    /**
     * Détecte si le mode debug est activé
     */
    isDebugEnabled() {
        // Priorité 1: localStorage (pour toggle runtime)
        const localStorageDebug = localStorage.getItem('crypto_debug_mode');
        if (localStorageDebug !== null) {
            return localStorageDebug === 'true';
        }
        
        // Priorité 2: Variable globale de configuration
        if (window.globalConfig) {
            const debugFromConfig = window.globalConfig.get('debug_mode');
            if (debugFromConfig !== undefined) {
                return debugFromConfig === true;
            }
        }
        
        // Priorité 3: URL parameter pour debug temporaire
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.has('debug')) {
            return urlParams.get('debug') === 'true';
        }
        
        // Priorité 4: Détection environment (localhost = debug par défaut)
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return true;
        }
        
        // Production par défaut
        return false;
    }
    
    /**
     * Active/désactive le mode debug
     */
    setDebugMode(enabled) {
        this.debugEnabled = enabled;
        localStorage.setItem('crypto_debug_mode', enabled.toString());
        debugLogger.debug(`Debug mode ${enabled ? 'ENABLED' : 'DISABLED'}`);
        // Mettre à jour les hooks
        this.applyConsoleOverride();
    }
    
    /**
     * Log de niveau ERROR (toujours affiché + toast)
     */
    error(message, ...args) {
        console.error(`Error ${message}`, ...args);
        // Vérifier que this existe et que _showToast est défini avant d'appeler
        if (this && typeof this._showToast === 'function') {
            this._showToast('error', message);
        }
    }

    /**
     * Log de niveau WARN (toujours affiché + toast)
     */
    warn(message, ...args) {
        console.warn(`Warning ${message}`, ...args);
        // Vérifier que this existe et que _showToast est défini avant d'appeler
        if (this && typeof this._showToast === 'function') {
            this._showToast('warn', message);
        }
    }

    /**
     * Log de niveau INFO (affiché uniquement si debug activé)
     */
    info(message, ...args) {
        if (this && this.debugEnabled) {
            console.log(`Info ${message}`, ...args);
        }
    }

    /**
     * Log de niveau DEBUG (affiché uniquement si debug activé)
     */
    debug(message, ...args) {
        if (this && this.debugEnabled) {
            console.log(`DEBUG ${message}`, ...args);
        }
    }
    
    /**
     * Log conditionnel pour les performances critiques
     */
    perf(message, ...args) {
        if (this.debugEnabled) {
            console.time(message);
            debugLogger.debug(`PERF ${message}`, ...args);
        }
    }
    
    /**
     * Fin de mesure de performance
     */
    perfEnd(message) {
        if (this.debugEnabled) {
            console.timeEnd(message);
        }
    }
    
    /**
     * Log pour les interactions API
     */
    api(endpoint, data = null) {
        if (this.debugEnabled) {
            console.group(`API ${endpoint}`);
            if (data) debugLogger.debug('Data:', data);
            console.groupEnd();
        }
    }
    
    /**
     * Log pour les changements d'état UI
     */
    ui(action, details = null) {
        if (this.debugEnabled) {
            debugLogger.debug(`UI ${action}`, details || '');
        }
    }
    
    /**
     * Affiche les statistiques de debug
     */
    stats() {
        if (this.debugEnabled) {
            console.group("Debug Statistics");
            debugLogger.debug('Debug mode:', this.debugEnabled);
            debugLogger.debug('Environment:', window.location.hostname);
            debugLogger.debug('Config available:', !!window.globalConfig);
            console.groupEnd();
        }
    }

    /**
     * Rend console.debug/log/info silencieux hors debug, non-destructif
     * En production (debug OFF): masque console.log, console.info, console.debug
     * Garde toujours: console.error, console.warn
     */
    applyConsoleOverride() {
        try {
            if (!this._consolePatched) {
                // Sauvegarder les méthodes originales
                console.__origDebug = console.__origDebug || console.debug?.bind(console) || console.log.bind(console);
                console.__origLog = console.__origLog || console.log?.bind(console);
                console.__origInfo = console.__origInfo || console.info?.bind(console);
                this._consolePatched = true;
            }

            // console.debug - toujours silencieux hors debug
            console.debug = (...args) => {
                if (!this.debugEnabled) return; // no-op
                try { console.__origDebug(`[debug]`, ...args); } catch { /* ignore */ }
            };

            // console.log - silencieux hors debug (nouveau)
            console.log = (...args) => {
                if (!this.debugEnabled) return; // no-op
                try { console.__origLog(...args); } catch { /* ignore */ }
            };

            // console.info - silencieux hors debug (nouveau)
            console.info = (...args) => {
                if (!this.debugEnabled) return; // no-op
                try { console.__origInfo(`[info]`, ...args); } catch { /* ignore */ }
            };
        } catch (_) {}
    }

    /**
     * Trace léger des appels fetch quand activé
     * Activé si debugEnabled && (localStorage.debug_trace_api === 'true')
     */
    applyFetchTracer() {
        try {
            if (this._fetchPatched) return;
            const originalFetch = window.fetch?.bind(window);
            if (!originalFetch) return;
            window.__origFetch = originalFetch;
            window.fetch = async (input, init = {}) => {
                const trace = this.debugEnabled && (localStorage.getItem('debug_trace_api') === 'true');
                const start = trace ? performance.now() : 0;
                let ok = false, status = 'n/a', urlStr = (typeof input === 'string') ? input : (input?.url || '[Request]');
                try {
                    const resp = await originalFetch(input, init);
                    ok = resp.ok; status = resp.status;
                    return resp;
                } catch (err) {
                    if (trace) debugLogger.warn("fetch error", { url: urlStr, err: err?.message });
                    throw err;
                } finally {
                    if (trace) {
                        const dur = (performance.now() - start).toFixed(0);
                        console.debug("fetch", { url: urlStr, ok, status, ms: Number(dur) });
                    }
                }
            };
            this._fetchPatched = true;
        } catch (_) {}
    }
}

// Instance globale
const debugLogger = new DebugLogger();

// Export pour utilisation dans d'autres scripts
window.debugLogger = debugLogger;

// Raccourcis pour compatibilité
window.log = {
    error: debugLogger.error.bind(debugLogger),
    warn: debugLogger.warn.bind(debugLogger),
    info: debugLogger.info.bind(debugLogger),
    debug: debugLogger.debug.bind(debugLogger),
    perf: debugLogger.perf.bind(debugLogger),
    perfEnd: debugLogger.perfEnd.bind(debugLogger),
    api: debugLogger.api.bind(debugLogger),
    ui: debugLogger.ui.bind(debugLogger)
};

// Interface pour toggle debug depuis la console
window.toggleDebug = () => {
    debugLogger.setDebugMode(!debugLogger.debugEnabled);
    return `Debug mode is now ${debugLogger.debugEnabled ? 'ON' : 'OFF'}`;
};

// Raccourcis pratiques pour dev
window.debugOn = () => { debugLogger.setDebugMode(true); window.globalConfig?.setDebugMode?.(true); return 'Debug ON'; };
window.debugOff = () => { debugLogger.setDebugMode(false); window.globalConfig?.setDebugMode?.(false); return 'Debug OFF'; };

debugLogger.debug("Debug Logger loaded - Type toggleDebug() to switch modes");

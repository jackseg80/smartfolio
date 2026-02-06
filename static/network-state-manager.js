/**
 * Network State Manager - Gestion robuste de l'état réseau
 *
 * Fonctionnalités:
 * - Détection online/offline
 * - Gestion visibilité onglet (pause polling when inactive)
 * - Exponential backoff pour retry
 * - Notifications utilisateur
 * - Pause/resume automatique des intervalles
 *
 * Usage:
 *   import { networkStateManager } from './network-state-manager.js';
 *
 *   // Enregistrer un intervalle à gérer
 *   const interval = networkStateManager.createManagedInterval(callback, 30000);
 *
 *   // Exécuter une requête avec retry
 *   const data = await networkStateManager.fetchWithRetry('/api/endpoint');
 */

class NetworkStateManager {
  constructor() {
    this.isOnline = navigator.onLine;
    this.isVisible = !document.hidden;
    this.consecutiveFailures = 0;
    this.managedIntervals = new Map(); // Map<id, {callback, delay, intervalId}>
    this.listeners = new Set();
    this.isRecovering = false;

    this._init();
  }

  _init() {
    // Écouter les événements réseau
    window.addEventListener('online', () => this._handleOnline());
    window.addEventListener('offline', () => this._handleOffline());

    // Écouter la visibilité de l'onglet
    document.addEventListener('visibilitychange', () => this._handleVisibilityChange());

    // Écouter le focus de la fenêtre (complément de visibilitychange)
    window.addEventListener('focus', () => this._handleFocus());
    window.addEventListener('blur', () => this._handleBlur());
  }

  _handleOnline() {
    console.debug('🌐 Network connection restored');
    this.isOnline = true;
    this.consecutiveFailures = 0;
    this.isRecovering = true;

    this._notifyListeners('online');
    this._showNotification('✅ Network connection restored', 'success');

    // Attendre 1s avant de reprendre (laisser la connexion se stabiliser)
    setTimeout(() => {
      this.isRecovering = false;
      this._resumeIntervals();
    }, 1000);
  }

  _handleOffline() {
    console.warn('⚠️ Network connection lost');
    this.isOnline = false;

    this._notifyListeners('offline');
    this._showNotification('⚠️ Network connection lost', 'warning');
    this._pauseIntervals();
  }

  _handleVisibilityChange() {
    const wasVisible = this.isVisible;
    this.isVisible = !document.hidden;

    if (!wasVisible && this.isVisible) {
      // Onglet redevenu visible
      console.debug('👁️ Tab became visible, resuming polling');
      this._notifyListeners('visible');

      // Reset failures si l'utilisateur revient après longtemps
      if (this.consecutiveFailures > 0) {
        console.debug('🔄 Resetting failure count on tab visibility');
        this.consecutiveFailures = 0;
      }

      this._resumeIntervals();
    } else if (wasVisible && !this.isVisible) {
      // Onglet masqué
      console.debug('👁️ Tab hidden, pausing polling');
      this._notifyListeners('hidden');
      this._pauseIntervals();
    }
  }

  _handleFocus() {
    // Fenêtre récupère le focus - vérifier la connexion
    if (this.isOnline && this.isVisible) {
      this._resumeIntervals();
    }
  }

  _handleBlur() {
    // Fenêtre perd le focus - optionnel: ralentir les polling
    // Pour l'instant on ne fait rien, le visibilitychange gère déjà
  }

  _pauseIntervals() {
    console.debug(`⏸️ Pausing ${this.managedIntervals.size} managed intervals`);
    for (const [id, info] of this.managedIntervals) {
      if (info.intervalId) {
        clearInterval(info.intervalId);
        info.intervalId = null;
      }
    }
  }

  _resumeIntervals() {
    if (!this.isOnline || !this.isVisible) {
      console.debug('⏸️ Not resuming intervals: offline or hidden');
      return;
    }

    console.debug(`▶️ Resuming ${this.managedIntervals.size} managed intervals`);
    for (const [id, info] of this.managedIntervals) {
      if (!info.intervalId) {
        // Exécuter immédiatement puis reprendre l'intervalle
        // Wrapper dans Promise.resolve pour gérer callbacks sync ET async
        Promise.resolve(info.callback()).catch(err => {
          console.warn(`Interval ${id} (${info.name || 'unknown'}) failed on resume:`, err);
        });

        info.intervalId = setInterval(info.callback, info.delay);
      }
    }
  }

  /**
   * Crée un intervalle géré qui se pause/resume automatiquement
   * @param {Function} callback - Fonction à exécuter (peut être async)
   * @param {number} delay - Délai en ms
   * @param {string} [name] - Nom pour debugging
   * @returns {string} ID de l'intervalle
   */
  createManagedInterval(callback, delay, name = '') {
    const id = `interval_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;

    const info = {
      callback,
      delay,
      name,
      intervalId: null
    };

    this.managedIntervals.set(id, info);

    // Démarrer uniquement si online et visible
    if (this.isOnline && this.isVisible) {
      info.intervalId = setInterval(callback, delay);
    }

    console.debug(`📌 Created managed interval: ${name || id} (${delay}ms)`);
    return id;
  }

  /**
   * Supprime un intervalle géré
   * @param {string} id - ID de l'intervalle
   */
  clearManagedInterval(id) {
    const info = this.managedIntervals.get(id);
    if (info) {
      if (info.intervalId) {
        clearInterval(info.intervalId);
      }
      this.managedIntervals.delete(id);
      console.debug(`🗑️ Cleared managed interval: ${info.name || id}`);
    }
  }

  /**
   * Exécute une requête fetch avec retry et backoff exponentiel
   * @param {string} url - URL à fetcher
   * @param {object} [options] - Options fetch
   * @param {number} [maxRetries=3] - Nombre max de tentatives
   * @returns {Promise<Response>}
   */
  async fetchWithRetry(url, options = {}, maxRetries = 3) {
    let lastError;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      // Ne pas retry si offline
      if (!this.isOnline) {
        throw new Error('Network offline');
      }

      try {
        const response = await fetch(url, options);

        // Succès - reset failures
        this.consecutiveFailures = 0;
        return response;

      } catch (error) {
        lastError = error;
        this.consecutiveFailures++;

        // Vérifier si c'est une erreur réseau spécifique
        const isNetworkError =
          error.message.includes('Failed to fetch') ||
          error.message.includes('ERR_NETWORK_IO_SUSPENDED') ||
          error.message.includes('NetworkError');

        if (isNetworkError && attempt < maxRetries) {
          // Backoff exponentiel: 1s, 2s, 4s
          const backoffMs = Math.min(1000 * Math.pow(2, attempt), 10000);
          console.debug(`🔄 Network error, retrying in ${backoffMs}ms (attempt ${attempt + 1}/${maxRetries})`);

          await this._sleep(backoffMs);
          continue;
        }

        // Pas de retry ou max atteint
        break;
      }
    }

    // Toutes les tentatives ont échoué
    if (this.consecutiveFailures >= 3) {
      this._showNotification('⚠️ Connection issues detected', 'warning');
    }

    throw lastError;
  }

  /**
   * Wrapper pour apiRequest avec gestion réseau
   * @param {object} globalConfig - Instance GlobalConfig
   * @param {string} endpoint - Endpoint API
   * @param {object} [options] - Options
   * @returns {Promise<any>}
   */
  async apiRequestWithRetry(globalConfig, endpoint, options = {}) {
    if (!this.isOnline) {
      throw new Error('Network offline - request skipped');
    }

    try {
      const result = await globalConfig.apiRequest(endpoint, options);
      this.consecutiveFailures = 0;
      return result;
    } catch (error) {
      this.consecutiveFailures++;

      // Si erreur réseau et pas trop de failures, retry silencieusement
      const isNetworkError =
        error.message.includes('Failed to fetch') ||
        error.message.includes('ERR_NETWORK_IO_SUSPENDED');

      if (isNetworkError && this.consecutiveFailures < 5) {
        console.debug(`⚠️ Network error (${this.consecutiveFailures} consecutive), will retry on next poll`);
        // Ne pas afficher d'erreur dans la console pour éviter le spam
        return null; // Retourner null au lieu de throw
      }

      throw error;
    }
  }

  /**
   * Enregistre un listener pour les événements réseau
   * @param {Function} callback - Callback (event: 'online'|'offline'|'visible'|'hidden')
   */
  addListener(callback) {
    this.listeners.add(callback);
  }

  removeListener(callback) {
    this.listeners.delete(callback);
  }

  _notifyListeners(event) {
    for (const callback of this.listeners) {
      try {
        callback(event);
      } catch (err) {
        console.error('Listener error:', err);
      }
    }
  }

  _showNotification(message, type = 'info') {
    // Utiliser le système de toast si disponible
    if (window.showToast) {
      window.showToast(message, type);
    } else {
      console.debug(`[${type.toUpperCase()}] ${message}`);
    }
  }

  _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Getter pour savoir si on peut faire des requêtes
   */
  get canMakeRequests() {
    return this.isOnline && this.isVisible && !this.isRecovering;
  }

  /**
   * Affiche l'état actuel
   */
  getStatus() {
    return {
      online: this.isOnline,
      visible: this.isVisible,
      recovering: this.isRecovering,
      consecutiveFailures: this.consecutiveFailures,
      managedIntervals: this.managedIntervals.size,
      canMakeRequests: this.canMakeRequests
    };
  }
}

// Instance singleton
export const networkStateManager = new NetworkStateManager();

// Exposer globalement pour debugging
window.networkStateManager = networkStateManager;

console.debug('🌐 Network State Manager initialized');

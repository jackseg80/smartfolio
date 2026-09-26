/**
 * Cycle Parameters Loader
 * Charge automatiquement les paramètres calibrés depuis localStorage
 * et les applique au module cycle-navigator.js
 *
 * Utilisé par: risk-dashboard.html, analytics-unified.html
 * Source de calibration: cycle-analysis.html
 */

// Version must match CALIBRATION_VERSION in cycle-navigator.js
const CALIBRATION_VERSION_PREFIX = '2.';

/**
 * Charge les paramètres calibrés depuis localStorage
 * @returns {Object|null} Paramètres calibrés ou null si non disponibles/expirés
 */
export function loadCalibrationParams() {
  try {
    const saved = localStorage.getItem('bitcoin_cycle_params');
    if (!saved) {
      console.debug("Aucun paramètre calibré trouvé dans localStorage");
      return null;
    }

    const data = JSON.parse(saved);

    // CRITICAL: Check version - reject old calibrations (pre-2.0)
    if (!data.version || !data.version.startsWith(CALIBRATION_VERSION_PREFIX)) {
      console.debug("Anciens paramètres calibrés rejetés (version:", data.version, ')');
      localStorage.removeItem('bitcoin_cycle_params');
      return null;
    }

    // Vérifier que les données ne sont pas trop anciennes (24h)
    const MAX_AGE_MS = 24 * 60 * 60 * 1000; // 24 heures
    const age = Date.now() - data.timestamp;

    if (age > MAX_AGE_MS) {
      console.debug("[Warning] Paramètres calibrés expirés (>24h), utilisation des paramètres par défaut");
      return null;
    }

    console.debug("[OK] Paramètres calibrés chargés depuis localStorage", {
      params: data.params,
      age_hours: (age / (60 * 60 * 1000)).toFixed(1),
      version: data.version
    });

    return data.params;

  } catch (error) {
    console.error("[Error] Erreur chargement paramètres calibrés:", error);
    return null;
  }
}

/**
 * Applique les paramètres calibrés au module cycle-navigator
 * @param {Object} cycleNavigatorModule - Module importé de cycle-navigator.js
 * @returns {boolean} True si les paramètres ont été appliqués, false sinon
 */
export async function applyCalibratedParams(cycleNavigatorModule) {
  try {
    // Charger les paramètres sauvegardés
    const savedParams = loadCalibrationParams();

    if (!savedParams) {
      console.debug("Utilisation des paramètres par défaut du modèle cycle");
      return false;
    }

    // Vérifier que le module a la fonction setCycleParams
    if (typeof cycleNavigatorModule.setCycleParams !== 'function') {
      console.warn("[Warning] Module cycle-navigator ne supporte pas setCycleParams");
      return false;
    }

    // Appliquer les paramètres
    cycleNavigatorModule.setCycleParams(savedParams);

    console.debug("[OK] Paramètres calibrés appliqués au cycle-navigator", savedParams);

    return true;

  } catch (error) {
    console.error("[Error] Erreur application paramètres calibrés:", error);
    return false;
  }
}

/**
 * Invalide les caches cycle obsolètes dans localStorage
 * Appelé après chargement des paramètres calibrés pour forcer un recalcul
 * @param {number} paramsTimestamp - Timestamp des paramètres calibrés
 */
function invalidateObsoleteCycleCaches(paramsTimestamp) {
  const CYCLE_CACHE_KEYS = [
    'analytics_unified_cycle',
    'risk_scores_cache_',  // Partial match for multi-tenant
    'cycle_content_cache',  // risk-dashboard CYCLE_CONTENT
    'cycle_data_cache',     // risk-dashboard CYCLE_DATA
    'cycle_chart_cache',    // risk-dashboard CYCLE_CHART
    'ccs_data_cache',       // CCS data may include cycle-blended scores
  ];

  let invalidatedCount = 0;

  try {
    // Parcourir toutes les clés localStorage
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (!key) continue;

      // Vérifier si c'est un cache cycle
      const isCycleCache = CYCLE_CACHE_KEYS.some(pattern => key.includes(pattern));
      if (!isCycleCache) continue;

      try {
        const cached = JSON.parse(localStorage.getItem(key));
        const cacheTimestamp = cached?.timestamp || 0;

        // Si le cache est plus ancien que les paramètres calibrés, l'invalider
        if (cacheTimestamp < paramsTimestamp) {
          localStorage.removeItem(key);
          invalidatedCount++;
          console.debug(`Cache cycle obsolète invalidé: ${key}`);
        }
      } catch (e) {
        // Ignorer les erreurs de parsing
      }
    }

    if (invalidatedCount > 0) {
      console.debug(`OK ${invalidatedCount} cache(s) cycle obsolète(s) invalidé(s)`);
    }
  } catch (error) {
    console.warn("[Warning] Erreur lors de l'invalidation des caches cycle:", error);
  }
}

/**
 * Auto-chargement des paramètres calibrés au démarrage
 * À utiliser dans les pages qui importent cycle-navigator.js
 *
 * @example
 * import { autoLoadCalibratedParams } from './modules/cycle-params-loader.js';
 * autoLoadCalibratedParams();
 */
export async function autoLoadCalibratedParams() {
  try {
    // Import dynamique du module cycle-navigator
    const cycleModule = await import('./cycle-navigator.js');

    // Charger les paramètres pour obtenir le timestamp
    const saved = localStorage.getItem('bitcoin_cycle_params');
    const paramsData = saved ? JSON.parse(saved) : null;
    const paramsTimestamp = paramsData?.timestamp || 0;

    // Invalider les caches obsolètes AVANT d'appliquer les paramètres
    if (paramsTimestamp > 0) {
      invalidateObsoleteCycleCaches(paramsTimestamp);
    }

    // Appliquer les paramètres calibrés
    const applied = await applyCalibratedParams(cycleModule);

    if (applied) {
      console.debug("Cycle calibré activé automatiquement");

      // Dispatch event pour notifier les autres composants
      window.dispatchEvent(new CustomEvent('cycle-params-loaded', {
        detail: { source: 'localStorage', calibrated: true, timestamp: paramsTimestamp }
      }));
    } else {
      console.debug("Cycle non calibré - paramètres par défaut utilisés");

      window.dispatchEvent(new CustomEvent('cycle-params-loaded', {
        detail: { source: 'default', calibrated: false }
      }));
    }

    return applied;

  } catch (error) {
    console.error("[Error] Erreur auto-chargement paramètres calibrés:", error);
    return false;
  }
}

/**
 * Listener pour les mises à jour de calibration depuis cycle-analysis.html
 * Recharge automatiquement les paramètres quand ils sont mis à jour
 */
export function listenForCalibrationUpdates() {
  window.addEventListener('message', async (event) => {
    // Vérifier que c'est une mise à jour de paramètres cycle
    if (event.data?.type === 'CYCLE_PARAMS_UPDATED') {
      console.debug("Mise à jour des paramètres cycle détectée", event.data);

      // Invalider les caches obsolètes avec le timestamp de la mise à jour
      const updateTimestamp = event.data.timestamp || Date.now();
      invalidateObsoleteCycleCaches(updateTimestamp);

      // Recharger les paramètres
      await autoLoadCalibratedParams();

      // Notifier les composants que les paramètres ont changé
      window.dispatchEvent(new CustomEvent('cycle-params-updated', {
        detail: event.data
      }));
    }
  });

  // Écouter aussi les événements storage (quand une autre page modifie localStorage)
  window.addEventListener('storage', async (event) => {
    if (event.key === 'bitcoin_cycle_params' && event.newValue) {
      console.debug("Paramètres cycle modifiés depuis une autre page");
      try {
        const data = JSON.parse(event.newValue);
        invalidateObsoleteCycleCaches(data.timestamp || Date.now());
        await autoLoadCalibratedParams();
      } catch (e) {
        console.warn('Erreur parsing mise à jour cycle params:', e);
      }
    }
  });

  console.debug("Écoute des mises à jour de calibration cycle activée");
}

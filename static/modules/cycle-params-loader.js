/**
 * Cycle Parameters Loader
 * Charge automatiquement les paramètres calibrés depuis localStorage
 * et les applique au module cycle-navigator.js
 *
 * Utilisé par: risk-dashboard.html, analytics-unified.html
 * Source de calibration: cycle-analysis.html
 */

/**
 * Charge les paramètres calibrés depuis localStorage
 * @returns {Object|null} Paramètres calibrés ou null si non disponibles/expirés
 */
export function loadCalibrationParams() {
  try {
    const saved = localStorage.getItem('bitcoin_cycle_params');
    if (!saved) {
      console.debug('ℹ️ Aucun paramètre calibré trouvé dans localStorage');
      return null;
    }

    const data = JSON.parse(saved);

    // Vérifier que les données ne sont pas trop anciennes (24h)
    const MAX_AGE_MS = 24 * 60 * 60 * 1000; // 24 heures
    const age = Date.now() - data.timestamp;

    if (age > MAX_AGE_MS) {
      console.debug('⚠️ Paramètres calibrés expirés (>24h), utilisation des paramètres par défaut');
      return null;
    }

    console.debug('✅ Paramètres calibrés chargés depuis localStorage', {
      params: data.params,
      age_hours: (age / (60 * 60 * 1000)).toFixed(1),
      version: data.version
    });

    return data.params;

  } catch (error) {
    console.error('❌ Erreur chargement paramètres calibrés:', error);
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
      console.debug('📊 Utilisation des paramètres par défaut du modèle cycle');
      return false;
    }

    // Vérifier que le module a la fonction setCycleParams
    if (typeof cycleNavigatorModule.setCycleParams !== 'function') {
      console.warn('⚠️ Module cycle-navigator ne supporte pas setCycleParams');
      return false;
    }

    // Appliquer les paramètres
    cycleNavigatorModule.setCycleParams(savedParams);

    console.debug('✅ Paramètres calibrés appliqués au cycle-navigator', savedParams);

    return true;

  } catch (error) {
    console.error('❌ Erreur application paramètres calibrés:', error);
    return false;
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

    // Appliquer les paramètres calibrés
    const applied = await applyCalibratedParams(cycleModule);

    if (applied) {
      console.debug('🎯 Cycle calibré activé automatiquement');

      // Dispatch event pour notifier les autres composants
      window.dispatchEvent(new CustomEvent('cycle-params-loaded', {
        detail: { source: 'localStorage', calibrated: true }
      }));
    } else {
      console.debug('📊 Cycle non calibré - paramètres par défaut utilisés');

      window.dispatchEvent(new CustomEvent('cycle-params-loaded', {
        detail: { source: 'default', calibrated: false }
      }));
    }

    return applied;

  } catch (error) {
    console.error('❌ Erreur auto-chargement paramètres calibrés:', error);
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
      console.debug('🔄 Mise à jour des paramètres cycle détectée', event.data);

      // Recharger les paramètres
      await autoLoadCalibratedParams();

      // Notifier les composants que les paramètres ont changé
      window.dispatchEvent(new CustomEvent('cycle-params-updated', {
        detail: event.data
      }));
    }
  });

  console.debug('👂 Écoute des mises à jour de calibration cycle activée');
}

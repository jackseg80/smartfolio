/**
 * Risk Dashboard Toast System
 *
 * Provides toast notifications for the risk dashboard.
 * Extracted from risk-dashboard-main-controller.js (Feb 2026)
 *
 * Exports:
 * - showToast: Display a toast notification
 * - showS3AlertToast: Display critical alert toast
 * - loadDismissedAlerts: Load dismissed alerts from localStorage
 * - shouldShowAlert: Check if alert should be shown
 *
 * Global exports (window.*):
 * - hideToast: Hide a specific toast
 * - hideAllToasts: Hide all toasts
 */

// Toast ID counter
let toastIdCounter = 0;

// Registry for secure toast action execution (no eval)
const toastActionsRegistry = new Map();

// Track dismissed toasts to prevent re-showing
const dismissedToasts = new Set();

/**
 * Execute toast action safely without eval()
 * Parses action string and calls appropriate functions
 */
function executeToastAction(actionString, toastId) {
  // Whitelist of allowed functions
  const allowedFunctions = {
    hideToast: window.hideToast,
    openAlertModal: window.openAlertModal
  };

  // Parse pattern: window.openAlertModal('id').then(() => hideToast('toastId'))
  const chainPattern = /window\.(\w+)\('([^']+)'\)\.then\(\(\)\s*=>\s*(\w+)\('([^']+)'\)\)/;
  const chainMatch = actionString.match(chainPattern);

  if (chainMatch) {
    const [_, func1Name, arg1, func2Name, arg2] = chainMatch;

    if (allowedFunctions[func1Name] && allowedFunctions[func2Name]) {
      const result = allowedFunctions[func1Name](arg1);

      if (result && typeof result.then === 'function') {
        result.then(() => allowedFunctions[func2Name](arg2));
      } else {
        allowedFunctions[func2Name](arg2);
      }
      return;
    }
  }

  // Parse simple pattern: hideToast('toastId')
  const simplePattern = /(\w+)\('([^']+)'\)/;
  const simpleMatch = actionString.match(simplePattern);

  if (simpleMatch) {
    const [_, funcName, arg] = simpleMatch;

    if (allowedFunctions[funcName]) {
      allowedFunctions[funcName](arg);
      return;
    }
  }

  console.warn('[Toasts] Action not recognized or not allowed:', actionString);
}

/**
 * Show a toast notification
 * @param {string} message - Toast message
 * @param {string} type - Toast type: 'info', 'warning', 'error', 's1', 's2', 's3'
 * @param {Object} options - Options
 * @param {number} options.duration - Auto-hide duration in ms
 * @param {Array} options.actions - Action buttons
 * @param {string} options.title - Toast title
 * @param {Object} options.alertData - Associated alert data
 * @returns {string} Toast ID
 */
export function showToast(message, type = 'info', options = {}) {
  const {
    duration = type === 'error' ? 8000 : 5000,
    actions = [],
    title = null,
    alertData = null
  } = options;

  const toastId = `toast-${++toastIdCounter}`;
  let toastContainer = document.getElementById('toast-container');

  if (!toastContainer) {
    // Try to get container from Toast.js module (may not be loaded yet)
    if (window.Toast && typeof window.Toast.getContainer === 'function') {
      toastContainer = window.Toast.getContainer();
    }
  }

  if (!toastContainer) {
    // Container still not available - fail silently to avoid spam
    return toastId;
  }

  // Map type to severity for styling
  let severityClass = 'toast-s1';
  let severityText = 'INFO';

  if (type === 'error' || type === 'critical' || type === 's3') {
    severityClass = 'toast-s3';
    severityText = 'CRITICAL';
  } else if (type === 'warning' || type === 's2') {
    severityClass = 'toast-s2';
    severityText = 'WARNING';
  }

  const toast = document.createElement('div');
  toast.id = toastId;
  toast.className = `toast ${severityClass}`;

  let actionsHtml = '';
  if (actions.length > 0) {
    actionsHtml = `
      <div class="toast-actions">
        ${actions.map((action, idx) => `
          <button class="toast-action ${action.secondary ? 'toast-action-secondary' : ''}"
                  data-action-index="${idx}">${action.label}</button>
        `).join('')}
      </div>`;
  }

  toast.innerHTML = `
    <div class="toast-header">
      <div class="toast-severity toast-severity-${type === 'error' || type === 's3' ? 's3' : type === 'warning' || type === 's2' ? 's2' : 's1'}">
        <div class="toast-severity-icon"></div>
        ${severityText}
      </div>
      <button class="toast-close" onclick="hideToast('${toastId}')">&times;</button>
    </div>
    <div class="toast-body">
      ${title ? `<div class="toast-title">${title}</div>` : ''}
      <div class="toast-description">${message}</div>
      ${actionsHtml}
    </div>
  `;

  toastContainer.appendChild(toast);

  // Store actions in registry for secure execution
  if (actions.length > 0) {
    toastActionsRegistry.set(toastId, actions);
  }

  // Animate in
  setTimeout(() => toast.classList.add('show'), 10);

  // Auto-hide (except for critical alerts)
  if (type !== 'error' && type !== 'critical' && type !== 's3') {
    setTimeout(() => window.hideToast(toastId), duration);
  }

  return toastId;
}

/**
 * Load previously dismissed alerts from localStorage
 */
export function loadDismissedAlerts() {
  try {
    const stored = localStorage.getItem('dismissedAlerts');
    if (stored) {
      const alertIds = JSON.parse(stored);
      alertIds.forEach(id => dismissedToasts.add(id));
      console.debug('[Toasts] Loaded dismissed alerts:', alertIds.length);
    }
  } catch (e) {
    console.warn('[Toasts] Failed to load dismissed alerts:', e);
  }
}

/**
 * Check if alert should be shown (not dismissed)
 * @param {string} alertId - Alert ID
 * @returns {boolean}
 */
export function shouldShowAlert(alertId) {
  return !dismissedToasts.has(alertId);
}

/**
 * Hide a specific toast
 * @param {string} toastId - Toast ID to hide
 */
function hideToast(toastId) {
  console.debug('[Toasts] Hiding toast:', toastId);
  const toast = document.getElementById(toastId);
  if (!toast) {
    console.warn('[Toasts] Toast not found:', toastId);
    return;
  }

  // Extract alert ID from toast ID (format: toast-{alertId})
  const alertId = toastId.replace('toast-', '');
  dismissedToasts.add(alertId);
  localStorage.setItem('dismissedAlerts', JSON.stringify([...dismissedToasts]));

  toast.classList.add('hide');

  setTimeout(() => {
    if (toast && toast.parentNode) {
      toast.remove();
    }
  }, 300);
}

/**
 * Hide all toasts
 */
function hideAllToasts() {
  console.debug('[Toasts] Hiding all toasts...');
  const toastContainer = document.getElementById('toast-container');
  if (!toastContainer) {
    return;
  }

  const toasts = toastContainer.querySelectorAll('.toast');

  if (toasts.length === 0) {
    // Fallback - look for toasts anywhere in document
    const allToasts = document.querySelectorAll('.toast');
    allToasts.forEach(toast => toast.remove());
  } else {
    toasts.forEach(toast => {
      toast.classList.add('hide');
      setTimeout(() => {
        if (toast && toast.parentNode) {
          toast.remove();
        }
      }, 300);
    });
  }

  // Force clear container if needed
  setTimeout(() => {
    const remainingToasts = toastContainer.querySelectorAll('.toast');
    if (remainingToasts.length > 0) {
      toastContainer.innerHTML = '';
    }
  }, 500);
}

/**
 * Show toast for S3 critical alerts
 * @param {Object} alert - Alert data
 * @param {Function} [getAlertTypeFn] - Optional function to get display name (uses window.getAlertTypeDisplayName or fallback)
 * @returns {string|null} Toast ID or null if dismissed
 */
export function showS3AlertToast(alert, getAlertTypeFn = null) {
  // Check if this alert has already been dismissed
  if (!shouldShowAlert(alert.id)) {
    console.debug('[Toasts] Skipping already dismissed alert:', alert.id);
    return null;
  }

  // Use provided function, window global, or fallback
  const getDisplayName = getAlertTypeFn || window.getAlertTypeDisplayName || ((type) => type || 'Alert');
  const title = getDisplayName(alert.alert_type);
  const message = `${alert.data.current_value?.toFixed(2) || 'N/A'} > ${alert.data.adaptive_threshold?.toFixed(2) || 'N/A'}`;

  // Use alert ID for consistent tracking
  const nextToastId = `toast-${alert.id}`;

  const actions = [
    {
      label: 'View Details',
      onclick: `window.openAlertModal('${alert.id}').then(() => hideToast('${nextToastId}'))`
    },
    {
      label: 'Dismiss',
      secondary: true,
      onclick: `hideToast('${nextToastId}')`
    }
  ];

  const toastId = showToast(message, 's3', {
    title: `🚨 ${title}`,
    actions,
    duration: 0, // Don't auto-hide critical alerts
    alertData: alert
  });

  // Update the toast ID to match alert ID for proper tracking
  const toastElement = document.getElementById(toastId);
  if (toastElement) {
    toastElement.id = nextToastId;
  }

  return nextToastId;
}

// Expose functions globally for onclick handlers
window.hideToast = hideToast;
window.hideAllToasts = hideAllToasts;

// Also expose showToast globally for backward compatibility
window.showToast = showToast;

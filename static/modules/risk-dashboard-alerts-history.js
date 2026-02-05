/**
 * Risk Dashboard Alerts History System
 *
 * Provides alerts history management for the risk dashboard.
 * Extracted from risk-dashboard-main-controller.js (Feb 2026)
 *
 * Exports:
 * - loadAlertsHistory: Load and display alerts history
 * - refreshAlertsHistory: Refresh alerts data
 * - initializeAlertsTab: Initialize alerts tab
 * - formatAlertType: Format alert type for display
 * - formatRelativeTime: Format timestamp as relative time
 * - getAlertTypeDisplayName: Get display name for alert type
 *
 * Global exports (window.*):
 * - filterAlertsHistory: Filter alerts (for onclick handlers)
 * - refreshAlertsHistory: Refresh alerts
 * - getAlertTypeDisplayName: Get display name
 */

// ====== State ======
let alertsHistoryData = [];
let currentAlertsPage = 1;
let totalAlertsPages = 1;
const alertsPerPage = 10;

// ====== Alert Type Formatting ======

/**
 * Get display name for alert type
 * @param {string} alertType - Alert type code
 * @returns {string} Human-readable display name
 */
export function getAlertTypeDisplayName(alertType) {
  const typeMap = {
    'VOL_Q90_CROSS': 'Volatility Q90 Cross',
    'REGIME_FLIP': 'Regime Flip',
    'CORR_HIGH': 'High Correlation',
    'CONTRADICTION_SPIKE': 'Contradiction Spike',
    'DECISION_DROP': 'Decision Drop',
    'EXEC_COST_SPIKE': 'Execution Cost Spike'
  };
  return typeMap[alertType] || alertType.replace(/_/g, ' ');
}

/**
 * Format alert type with icon for table display
 * @param {string} alertType - Alert type code
 * @returns {string} HTML string with icon and label
 */
export function formatAlertType(alertType) {
  const typeMap = {
    'VOL_Q90_CROSS': { icon: '📊', label: 'High Volatility' },
    'REGIME_FLIP': { icon: '🔄', label: 'Regime Change' },
    'CORR_HIGH': { icon: '🔗', label: 'High Correlation' },
    'CONTRADICTION_SPIKE': { icon: '⚠️', label: 'ML Contradiction' },
    'DECISION_DROP': { icon: '📉', label: 'Low Confidence' },
    'EXEC_COST_SPIKE': { icon: '💸', label: 'High Exec Cost' }
  };

  const mapped = typeMap[alertType] || { icon: '🔔', label: alertType };
  return `<span style="display: inline-flex; align-items: center; gap: 0.5rem;"><span style="font-size: 1.1rem;">${mapped.icon}</span><span>${mapped.label}</span></span>`;
}

/**
 * Format timestamp as relative time
 * @param {string} timestamp - ISO timestamp
 * @returns {string} Relative time string (e.g., "5m ago")
 */
export function formatRelativeTime(timestamp) {
  const now = new Date();
  const alertTime = new Date(timestamp);
  const diffMs = now - alertTime;
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;

  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
}

// ====== Data Fetching ======

/**
 * Fetch alerts history from API
 * @param {Object} filters - Filter options
 * @returns {Promise<Array>} Array of alerts
 */
async function fetchAlertsHistory(filters = {}) {
  try {
    console.debug('Loading alerts history...', filters);

    // Build query parameters
    const params = new URLSearchParams();
    params.append('limit', '100'); // Fetch more for client-side filtering
    params.append('offset', '0');
    params.append('include_snoozed', 'true');

    // Only add filters if they have values (don't filter if empty)
    if (filters.severity && filters.severity.trim() !== '') {
      params.append('severity_filter', filters.severity);
    }
    if (filters.type && filters.type.trim() !== '') {
      params.append('type_filter', filters.type);
    }

    // Use active endpoint with X-User header via globalConfig
    const alerts = await window.globalConfig.apiRequest('/api/alerts/active', {
      params: Object.fromEntries(params)
    });
    console.debug(`History loaded: ${alerts?.length || 0} alerts`, alerts.slice(0, 2));

    return alerts || [];

  } catch (error) {
    (window.debugLogger?.error || console.error)('Failed to load alerts history:', error);
    throw error;
  }
}

/**
 * Filter alerts by time period
 * @param {Array} alerts - Array of alerts
 * @param {number} days - Number of days to filter
 * @returns {Array} Filtered alerts
 */
function filterAlertsHistoryByPeriod(alerts, days) {
  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - days);

  return alerts.filter(alert => {
    const alertDate = new Date(alert.created_at);
    return alertDate >= cutoffDate;
  });
}

// ====== Main Functions ======

/**
 * Load alerts history with filters
 */
export async function loadAlertsHistory() {
  try {
    const severityFilterEl = document.getElementById('alerts-severity-filter');
    const typeFilterEl = document.getElementById('alerts-type-filter');
    const periodFilterEl = document.getElementById('alerts-period-filter');

    const severityFilter = severityFilterEl ? severityFilterEl.value : '';
    const typeFilter = typeFilterEl ? typeFilterEl.value : '';
    const periodDays = periodFilterEl ? parseInt(periodFilterEl.value) : 7;

    console.debug('Loading alerts with filters:', { severityFilter, typeFilter, periodDays });

    // Fetch all alerts with basic filters
    let alerts = await fetchAlertsHistory({
      severity: severityFilter,
      type: typeFilter
    });

    // Filter by period client-side
    alerts = filterAlertsHistoryByPeriod(alerts, periodDays);

    alertsHistoryData = alerts;

    // Store globally for modal system
    window.currentAlertsData = alerts;

    currentAlertsPage = 1;
    totalAlertsPages = Math.ceil(alerts.length / alertsPerPage);

    renderAlertsHistoryPage();
    updateAlertsStats(alerts);

  } catch (error) {
    (window.debugLogger?.error || console.error)('Error loading alerts history:', error);
    const errorContainer = document.getElementById('alerts-history-content');
    if (errorContainer) {
      errorContainer.innerHTML =
        '<div class="error">Failed to load alerts history. Please try again.</div>';
    }
  }
}

/**
 * Refresh alerts history (alias for loadAlertsHistory)
 */
export function refreshAlertsHistory() {
  loadAlertsHistory();
}

/**
 * Initialize alerts tab
 */
export function initializeAlertsTab() {
  loadAlertsHistory();
}

// ====== Rendering ======

/**
 * Render current page of alerts history
 */
function renderAlertsHistoryPage() {
  const container = document.getElementById('alerts-history-content');
  const paginationContainer = document.getElementById('alerts-pagination');

  // Guard: Check if elements exist
  if (!container || !paginationContainer) {
    (window.debugLogger?.warn || console.warn)('⚠️ Alerts DOM elements not found, skipping render');
    return;
  }

  if (!alertsHistoryData.length) {
    container.innerHTML = '<div class="no-data">No alerts found for the selected criteria.</div>';
    paginationContainer.style.display = 'none';
    return;
  }

  // Calculate pagination
  const start = (currentAlertsPage - 1) * alertsPerPage;
  const end = start + alertsPerPage;
  const pageAlerts = alertsHistoryData.slice(start, end);

  // Render alerts table
  const table = document.createElement('table');
  table.className = 'alerts-table';

  // Table header
  table.innerHTML = `
        <thead>
          <tr>
            <th>Severity</th>
            <th>Type</th>
            <th>Message</th>
            <th>Created</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          ${pageAlerts.map(alert => createAlertsHistoryRow(alert)).join('')}
        </tbody>
      `;

  container.innerHTML = '';
  container.appendChild(table);

  // Update pagination
  updateAlertsPagination();
}

/**
 * Create HTML row for an alert
 * @param {Object} alert - Alert data
 * @returns {string} HTML string for table row
 */
function createAlertsHistoryRow(alert) {
  const formatTime = (timestamp) => {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const getStatusText = (alert) => {
    if (alert.acknowledged_at) {
      return `<span style="color: var(--success)">✅ Acknowledged</span>`;
    }
    if (alert.snooze_until && new Date(alert.snooze_until) > new Date()) {
      return `<span style="color: var(--warning)">⏸️ Snoozed</span>`;
    }
    return `<span style="color: var(--error)">🚨 Active</span>`;
  };

  // Format unifié : Action → Impact € → 2 raisons → Détails
  const formatUnifiedAlert = (alert) => {
    // Simuler le formatage unifié côté client
    const formatted = formatAlertClientSide(alert);

    return `
          <div class="alert-unified-format">
            <div class="alert-action">
              <strong>🎯 ${formatted.action}</strong>
            </div>
            <div class="alert-impact">
              💰 Impact: <span class="alert-impact-value">${formatted.impact}</span>
            </div>
            <div class="alert-reasons">
              📋 Raisons:
              <ul class="alert-reasons-list">
                ${formatted.reasons.map(reason => `<li>${reason}</li>`).join('')}
              </ul>
            </div>
            <div class="alert-details">
              ℹ️ ${formatted.details}
            </div>
          </div>
        `;
  };

  // Formatage unifié côté client
  const formatAlertClientSide = (alert) => {
    const alertType = alert.alert_type;
    const severity = alert.severity;
    const currentValue = alert.data?.current_value || 0;
    const portfolioValue = 100000; // €100k par défaut

    // Templates simplifiés
    const templates = {
      'VOL_Q90_CROSS': {
        'S1': { action: 'Surveillance volatilité', impact_base: 0.5, reasons: ['Volatilité Q90 dépassée', 'Conditions de marché agitées'] },
        'S2': { action: 'Réduction exposition (mode Slow)', impact_base: 2.0, reasons: ['Volatilité critique détectée', 'Risque de drawdown majoré'] },
        'S3': { action: 'Arrêt immédiat trading (Freeze)', impact_base: 8.0, reasons: ['Volatilité extrême mesurée', 'Protection capital prioritaire'] }
      },
      'EXEC_COST_SPIKE': {
        'S1': { action: 'Surveillance coûts exécution', impact_base: 0.2, reasons: ['Coûts trading légèrement élevés', 'Conditions liquidité moyennes'] },
        'S2': { action: 'Ralentissement trading (mode Slow)', impact_base: 1.5, reasons: ['Coûts exécution anormalement hauts', 'Liquidité marché dégradée'] },
        'S3': { action: 'Arrêt trading (mode Freeze)', impact_base: 4.0, reasons: ['Coûts exécution prohibitifs', 'Liquidité marché très dégradée'] }
      },
      'DECISION_DROP': {
        'S1': { action: 'Monitoring confiance décision', impact_base: 0.4, reasons: ['Score décision en baisse', 'Confiance allocation réduite'] },
        'S2': { action: 'Mode prudent allocation', impact_base: 2.2, reasons: ['Chute confiance décision significative', 'Qualité allocation dégradée'] },
        'S3': { action: 'Mode ultra-conservateur (Freeze)', impact_base: 9.0, reasons: ['Effondrement confiance décision', 'Allocations potentiellement erronées'] }
      }
    };

    const template = templates[alertType]?.[severity] ||
      { action: `Alerte ${alertType}`, impact_base: 1.0, reasons: ['Situation détectée', 'Action recommandée'] };

    const impact_euro = portfolioValue * template.impact_base / 100;

    return {
      action: template.action,
      impact: impact_euro >= 1 ? `€${Math.round(impact_euro).toLocaleString()}` : `€${impact_euro.toFixed(2)}`,
      reasons: template.reasons,
      details: `Valeur ${currentValue} détectée - ${severity} à ${new Date(alert.created_at).toLocaleTimeString()}`
    };
  };

  const getSeverityBadge = (severity) => {
    const severityConfig = {
      'S1': { icon: 'ℹ️', label: 'Info', color: '#3b82f6', bg: 'rgba(59, 130, 246, 0.1)' },
      'S2': { icon: '⚠️', label: 'Warning', color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.1)' },
      'S3': { icon: '🚨', label: 'Critical', color: '#ef4444', bg: 'rgba(239, 68, 68, 0.1)' }
    };
    const config = severityConfig[severity] || severityConfig['S1'];
    return `<span style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 0.75rem; border-radius: 6px; background: ${config.bg}; color: ${config.color}; font-weight: 600; font-size: 0.9rem;"><span style="font-size: 1.1rem;">${config.icon}</span><span>${severity}</span></span>`;
  };

  return `
        <tr>
          <td>
            ${getSeverityBadge(alert.severity)}
          </td>
          <td>
            <div class="alert-type-label">${formatAlertType(alert.alert_type)}</div>
          </td>
          <td class="alert-message-cell">
            ${formatUnifiedAlert(alert)}
          </td>
          <td>
            <div class="alert-timestamp">${formatTime(alert.created_at)}</div>
            <div class="alert-timestamp" style="font-size: 0.7rem; color: var(--text-muted);">${formatRelativeTime(alert.created_at)}</div>
          </td>
          <td>
            ${getStatusText(alert)}
          </td>
          <td>
            <button
              class="alert-action-btn"
              onclick="openAlertModal('${alert.id}')"
              title="View alert details">
              📋 Details
            </button>
          </td>
        </tr>
      `;
}

// ====== Stats & Pagination ======

/**
 * Update alerts statistics display
 * @param {Array} alerts - Array of alerts
 */
function updateAlertsStats(alerts) {
  console.debug('Updating alerts stats for', alerts?.length || 0, 'alerts');

  const stats = {
    total: alerts?.length || 0,
    S1: alerts?.filter(a => a.severity === 'S1').length || 0,
    S2: alerts?.filter(a => a.severity === 'S2').length || 0,
    S3: alerts?.filter(a => a.severity === 'S3').length || 0,
    acknowledged: alerts?.filter(a => a.acknowledged_at).length || 0
  };

  console.debug('Computed stats:', stats);

  const statsHtml = `
        <div class="alerts-stats" id="alerts-stats-display">
          <div class="alerts-stat">
            <div class="alerts-stat-number">${stats.total}</div>
            <div class="alerts-stat-label">Total</div>
          </div>
          <div class="alerts-stat">
            <div class="alerts-stat-number">${stats.S3}</div>
            <div class="alerts-stat-label">Critical</div>
          </div>
          <div class="alerts-stat">
            <div class="alerts-stat-number">${stats.S2}</div>
            <div class="alerts-stat-label">Warnings</div>
          </div>
          <div class="alerts-stat">
            <div class="alerts-stat-number">${stats.S1}</div>
            <div class="alerts-stat-label">Info</div>
          </div>
          <div class="alerts-stat">
            <div class="alerts-stat-number">${stats.acknowledged}</div>
            <div class="alerts-stat-label">Acknowledged</div>
          </div>
        </div>
      `;

  const container = document.getElementById('alerts-history-content');
  if (container) {
    // Remove existing stats if present
    const existingStats = document.getElementById('alerts-stats-display');
    if (existingStats) {
      existingStats.remove();
    }

    container.insertAdjacentHTML('afterbegin', statsHtml);
    console.debug('Stats HTML inserted into container');
  } else {
    (window.debugLogger?.warn || console.warn)('alerts-history-content container not found');
  }
}

/**
 * Update pagination controls
 */
function updateAlertsPagination() {
  const paginationContainer = document.getElementById('alerts-pagination');
  const prevBtn = document.getElementById('alerts-prev-btn');
  const nextBtn = document.getElementById('alerts-next-btn');
  const pageInfo = document.getElementById('alerts-page-info');

  // Guard: Check if pagination elements exist
  if (!paginationContainer || !prevBtn || !nextBtn || !pageInfo) {
    (window.debugLogger?.warn || console.warn)('⚠️ Pagination DOM elements not found, skipping update');
    return;
  }

  if (totalAlertsPages <= 1) {
    paginationContainer.style.display = 'none';
    return;
  }

  paginationContainer.style.display = 'flex';
  prevBtn.disabled = currentAlertsPage <= 1;
  nextBtn.disabled = currentAlertsPage >= totalAlertsPages;
  pageInfo.textContent = `Page ${currentAlertsPage} of ${totalAlertsPages}`;
}

/**
 * Load previous page of alerts
 */
export function loadPreviousAlertsPage() {
  if (currentAlertsPage > 1) {
    currentAlertsPage--;
    renderAlertsHistoryPage();
  }
}

/**
 * Load next page of alerts
 */
export function loadNextAlertsPage() {
  if (currentAlertsPage < totalAlertsPages) {
    currentAlertsPage++;
    renderAlertsHistoryPage();
  }
}

// ====== Global Exports ======
// Expose functions globally for onclick handlers and backward compatibility

window.filterAlertsHistory = function () {
  loadAlertsHistory();
};

window.refreshAlertsHistory = refreshAlertsHistory;
window.getAlertTypeDisplayName = getAlertTypeDisplayName;
window.loadPreviousAlertsPage = loadPreviousAlertsPage;
window.loadNextAlertsPage = loadNextAlertsPage;

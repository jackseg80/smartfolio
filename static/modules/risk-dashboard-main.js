/**
 * Risk Dashboard - Main Orchestrator (Simplified)
 * Single-view dashboard - tabs removed, content moved to dedicated pages:
 * - Cycle Analysis → cycle-analysis.html
 * - Strategic Targets → rebalance.html
 * - Alerts History → monitoring.html
 */

// Global state
let autoRefreshInterval = null;
let isRefreshing = false;

/**
 * Refresh dashboard data
 * @param {boolean} forceRefresh - Force refresh, bypass cache
 */
export async function refreshDashboard(forceRefresh = false) {
  if (isRefreshing) {
    debugLogger.debug('⏸️ Refresh already in progress, skipping...');
    return;
  }

  isRefreshing = true;
  const refreshBtn = document.getElementById('refresh-btn');
  if (refreshBtn) {
    refreshBtn.disabled = true;
  }

  try {
    debugLogger.debug(`🔄 Refreshing dashboard (force: ${forceRefresh})`);

    // Risk content is handled by legacy controller (risk-dashboard-main-controller.js)
    // Dispatch event to trigger refresh
    window.dispatchEvent(new CustomEvent('riskDashboardRefresh', { detail: { force: forceRefresh } }));

    // Update timestamp
    const timestamp = document.getElementById('last-update');
    if (timestamp) {
      timestamp.textContent = `Last update: ${new Date().toLocaleTimeString('fr-FR')}`;
    }

    debugLogger.debug('✅ Dashboard refreshed successfully');
  } catch (error) {
    debugLogger.error('❌ Failed to refresh dashboard:', error);
  } finally {
    isRefreshing = false;
    if (refreshBtn) {
      refreshBtn.disabled = false;
    }
  }
}

/**
 * Toggle auto-refresh
 */
export function toggleAutoRefresh() {
  const btn = document.getElementById('auto-refresh-btn');
  if (!btn) return;

  if (autoRefreshInterval) {
    // Disable auto-refresh
    clearInterval(autoRefreshInterval);
    autoRefreshInterval = null;
    btn.textContent = '⏱️ Enable Auto-Refresh (30s)';
    btn.style.background = 'var(--brand-primary)';
    debugLogger.debug('⏸️ Auto-refresh disabled');
  } else {
    // Enable auto-refresh
    autoRefreshInterval = setInterval(() => refreshDashboard(false), 30000);
    btn.textContent = '⏸️ Disable Auto-Refresh';
    btn.style.background = 'var(--success)';
    debugLogger.debug('▶️ Auto-refresh enabled (30s)');
  }
}

/**
 * Initialize the dashboard (simplified - single view)
 */
export async function initDashboard() {
  debugLogger.debug('🚀 Initializing Risk Dashboard (simplified view)...');

  // Auto-calculate scores if auto_calc=true (for iframe refresh from dashboard.html)
  const urlParams = new URLSearchParams(window.location.search);
  const autoCalc = urlParams.get('auto_calc') === 'true';
  if (autoCalc) {
    debugLogger.debug('🤖 Auto-calc mode detected, will trigger refresh after init...');
  }

  try {
    // NOTE: All risk content is handled by risk-dashboard-main-controller.js
    // This orchestrator only handles initialization and refresh coordination

    // Listen for data source changes
    window.addEventListener('dataSourceChanged', (event) => {
      debugLogger.debug(`🔄 Data source changed: ${event.detail.oldSource} → ${event.detail.newSource}`);
      // Let the legacy controller handle it to preserve advanced sections (GRI, Phase 3A, VaR, etc.)
      debugLogger.debug('⏸️ Defers to legacy controller for source changes');
    });

    debugLogger.debug('✅ Risk Dashboard initialized successfully');

    // Trigger auto-refresh if auto_calc=true
    if (autoCalc) {
      debugLogger.debug('🔄 Triggering automatic scores calculation...');
      setTimeout(async () => {
        try {
          await refreshDashboard(true);
          debugLogger.debug('✅ Auto-calc completed, scores persisted to localStorage');
        } catch (error) {
          debugLogger.error('❌ Auto-calc failed:', error);
        }
      }, 1000);
    }
  } catch (error) {
    debugLogger.error('❌ Failed to initialize dashboard:', error);
  }
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initDashboard);
} else {
  initDashboard();
}

// Export main functions
export default {
  refreshDashboard,
  toggleAutoRefresh,
  initDashboard
};

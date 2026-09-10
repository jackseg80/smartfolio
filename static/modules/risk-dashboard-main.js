/**
 * Risk Dashboard - Main Orchestrator
 * Manages tab switching, global refresh, and initialization
 * - Risk Overview: Risk metrics and scores
 * - Market Cycles: Cycle analysis and on-chain indicators (NO chart - chart is in cycle-analysis.html)
 * - Strategic Targets → rebalance.html (external link)
 * - Alerts History → monitoring.html (external link)
 */

import { showLoading, showError } from './risk-utils.js';

// Global state
let currentTab = 'risk';
let autoRefreshInterval = null;
let isRefreshing = false;

/**
 * Switch between dashboard tabs with lazy loading
 * @param {string} tabName - Tab to switch to ('risk', 'cycles')
 * @param {boolean} forceReload - Force reload even if content exists
 */
export async function switchTab(tabName, forceReload = false) {
  debugLogger.debug(`🔄 Switching to tab: ${tabName} (forceReload: ${forceReload})`);

  // Update tab buttons (classes + ARIA attributes)
  document.querySelectorAll('.tab-button').forEach(btn => {
    const isActive = btn.dataset.tab === tabName;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
  });

  // Update tab panes
  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === `${tabName}-tab`);
  });

  currentTab = tabName;

  // Lazy-load tab content
  try {
    const container = document.getElementById(`${tabName}-tab`);
    if (!container) {
      debugLogger.error(`Container for tab ${tabName} not found`);
      return;
    }

    // ✅ SPECIAL: Skip 'risk' tab - handled by legacy controller (risk-dashboard-main-controller.js)
    if (tabName === 'risk') {
      debugLogger.debug('⏸️ Risk tab handled by legacy controller');
      return;
    }

    // Check if tab already has content (skip check if forceReload)
    const hasContent = !forceReload && container.children.length > 1;

    if (!hasContent) {
      showLoading(container, `Loading ${tabName} data...`);

      switch (tabName) {
        case 'cycles':
          const { renderCyclesContent } = await import('./risk-cycles-tab.js');
          await renderCyclesContent();
          break;

        default:
          showError(container, `Unknown tab: ${tabName}`);
      }
    }
  } catch (error) {
    debugLogger.error(`Failed to load tab ${tabName}:`, error);
    const container = document.getElementById(`${tabName}-tab`);
    if (container) {
      showError(container, `Failed to load ${tabName} content`, error.message);
    }
  }
}

/**
 * Refresh all dashboard data
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
    refreshBtn.textContent = '🔄 Refreshing...';
  }

  try {
    debugLogger.debug(`🔄 Refreshing dashboard (force: ${forceRefresh})`);

    // Refresh current tab with forceReload flag
    await switchTab(currentTab, forceRefresh);

    // Update timestamp
    const timestamp = document.getElementById('last-update');
    if (timestamp) {
      timestamp.textContent = `Last update: ${new Date().toLocaleTimeString('en-US')}`;
    }

    debugLogger.debug('✅ Dashboard refreshed successfully');
  } catch (error) {
    debugLogger.error('❌ Failed to refresh dashboard:', error);
  } finally {
    isRefreshing = false;
    if (refreshBtn) {
      refreshBtn.disabled = false;
      refreshBtn.textContent = '🔄 Refresh';
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
 * Initialize the dashboard
 */
export async function initDashboard() {
  if (window.__smartfolioAuthReady && !await window.__smartfolioAuthReady) return;

  debugLogger.debug('🚀 Initializing Risk Dashboard...');

  // Auto-calculate scores if auto_calc=true (for iframe refresh from dashboard.html)
  const urlParams = new URLSearchParams(window.location.search);
  const autoCalc = urlParams.get('auto_calc') === 'true';
  if (autoCalc) {
    debugLogger.debug('🤖 Auto-calc mode detected, will trigger refresh after init...');
  }

  try {
    // Setup tab switching
    document.querySelectorAll('.tab-button').forEach(btn => {
      btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        if (tabName) switchTab(tabName);
      });
    });

    // Listen for data source changes
    window.addEventListener('dataSourceChanged', (event) => {
      debugLogger.debug(`🔄 Data source changed: ${event.detail.oldSource} → ${event.detail.newSource}`);
      // Let the legacy controller handle it to preserve advanced sections (GRI, Phase 3A, VaR, etc.)
      debugLogger.debug('⏸️ Modular system defers to legacy controller for source changes');
    });

    // Initialize first tab
    await switchTab('risk');

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

// Make switchTab available globally for onclick handlers
window.switchTab = switchTab;

// Export main functions
export default {
  switchTab,
  refreshDashboard,
  toggleAutoRefresh,
  initDashboard
};

/**
 * GovernancePanel.js - Composant UI pour la gouvernance des décisions
 * 
 * Fonctionnalités :
 * - Affichage du statut de gouvernance en temps réel
 * - Modal d'approbation pour les décisions proposées
 * - Contrôles freeze/unfreeze du système
 * - Indicateurs ML (contradiction index, confidence, etc.)
 */

import { store } from '../core/risk-dashboard-store.js';

class GovernancePanel {
  constructor(container) {
    this.container = container;
    this.isVisible = false;
    this.refreshInterval = null;
    this.unsubscribeStore = null;

    this.init();

    if (store && typeof store.subscribe === 'function') {
      let rafId = null;
      const scheduleUpdate = () => {
        if (rafId) return;
        rafId = requestAnimationFrame(() => {
          rafId = null;
          try {
            this.updateDisplay();
          } catch (error) {
            console.debug('GovernancePanel auto-update skipped:', error?.message || error);
          }
        });
      };
      this.unsubscribeStore = store.subscribe(scheduleUpdate);
    }
  }

  init() {
    this.render();
    this.bindEvents();
    this.startAutoRefresh();
  }
  
  render() {
    const html = `
      <div class="governance-panel" id="governance-panel">
        <div class="governance-header">
          <h2>🏛️ Decision Engine</h2>
          <div class="header-badges">
            <div class="transparency-badge" id="transparency-badge" title="Transparency Level">
              <span class="badge-icon">🔍</span>
              <span class="badge-text" id="transparency-level">Normal</span>
            </div>
            <div class="alerts-badge" id="alerts-badge" title="Active Alerts" style="display: none;">
              <span class="badge-icon">🚨</span>
              <span class="badge-text" id="alerts-count">0</span>
            </div>
          </div>
          <button class="governance-toggle" id="governance-toggle" aria-label="Collapse/Expand governance panel" aria-expanded="true">
            <span class="toggle-icon" aria-hidden="true">▲</span>
          </button>
        </div>
        
        <div class="governance-content" id="governance-content">
          <!-- Status Section with Enhanced Transparency -->
          <div class="governance-section">
            <h4>📊 System Status</h4>
            <div class="status-grid-compact">
              <div class="status-card primary">
                <div class="status-header">
                  <span class="status-icon">🏛️</span>
                  <span class="status-title">State</span>
                </div>
                <div class="status-content">
                  <span class="status-badge" id="gov-state">LOADING...</span>
                  <span class="status-timestamp" id="state-timestamp" title="Last Update"></span>
                </div>
              </div>
              
              <div class="status-card secondary">
                <div class="status-header">
                  <span class="status-icon">🎛️</span>
                  <span class="status-title">Mode</span>
                </div>
                <div class="status-content">
                  <span class="mode-badge" id="gov-mode">manual</span>
                  <div class="mode-explanation" id="mode-explanation" style="display: none;">
                    <small class="mode-help-text"></small>
                  </div>
                </div>
              </div>
              
              <div class="status-card">
                <div class="status-header">
                  <span class="status-icon">📋</span>
                  <span class="status-title">Policy</span>
                </div>
                <div class="status-content">
                  <span class="policy-info" id="gov-policy">--</span>
                </div>
              </div>
              
              <div class="status-card">
                <div class="status-header">
                  <span class="status-icon">⚡</span>
                  <span class="status-title">Cooldown</span>
                </div>
                <div class="status-content">
                  <span class="cooldown-status" id="gov-cooldown">--</span>
                </div>
              </div>
              
              <div class="status-card" id="freeze-status-card" style="display: none;">
                <div class="status-header">
                  <span class="status-icon">❄️</span>
                  <span class="status-title">Auto-Unfreeze</span>
                </div>
                <div class="status-content">
                  <span class="freeze-timer" id="freeze-timer"></span>
                  <div class="freeze-progress">
                    <div class="freeze-progress-bar" id="freeze-progress-bar"></div>
                  </div>
                </div>
              </div>
              
              <div class="status-card warning">
                <div class="status-header">
                  <span class="status-icon">⚠️</span>
                  <span class="status-title">Contradiction</span>
                </div>
                <div class="status-content">
                  <span class="contradiction-meter" id="gov-contradiction">--</span>
                </div>
              </div>
            </div>
          </div>
          
          <!-- ML Signals Section -->
          <div class="governance-section">
            <h4>🧠 ML Signals</h4>
            <div class="signals-cards" id="ml-signals">
              <div class="signal-card">
                <div class="signal-header">
                  <span class="signal-icon">🎯</span>
                  <span class="signal-title">Confidence</span>
                </div>
                <div class="signal-metrics">
                  <div class="signal-value-large" id="confidence-value">--</div>
                  <div class="signal-bar-compact">
                    <div class="signal-fill-compact" id="confidence-fill"></div>
                  </div>
                  <div class="signal-label" id="confidence-label">Loading...</div>
                </div>
              </div>
              
              <div class="signal-card">
                <div class="signal-header">
                  <span class="signal-icon">📊</span>
                  <span class="signal-title">Decision Score</span>
                </div>
                <div class="signal-metrics">
                  <div class="signal-value-large" id="decision-value">--</div>
                  <div class="signal-bar-compact">
                    <div class="signal-fill-compact" id="decision-fill"></div>
                  </div>
                  <div class="signal-label" id="decision-label">Loading...</div>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Mode Selection Section -->
          <div class="governance-section">
            <h4>🎛️ Governance Mode</h4>
            <div class="mode-selector" id="mode-selector">
              <select class="gov-select" id="governance-mode-select" aria-label="Governance mode">
                <option value="manual">🤝 Manual</option>
                <option value="ai_assisted">🤖 AI Assisted</option>
                <option value="full_ai">🚀 Full AI</option>
              </select>
              <button class="gov-btn secondary" id="btn-change-mode">
                Change Mode
              </button>
            </div>
          </div>

          <!-- Actions Section -->
          <div class="governance-section">
            <h4>⚡ Actions</h4>
            <div class="actions-grid">
              <button class="gov-btn primary" id="btn-refresh" title="Refresh governance state">
                🔄 Refresh
              </button>
              <button class="gov-btn secondary" id="btn-propose" title="Propose test decision">
                📋 Propose
              </button>
              <button class="gov-btn info" id="btn-cooldown-check" title="Check cooldown status">
                ⏱️ Cooldown
              </button>
              <button class="gov-btn warning" id="btn-freeze" title="Freeze system (emergency stop)">
                ❄️ Freeze
              </button>
              <button class="gov-btn success" id="btn-unfreeze" title="Unfreeze system" disabled>
                🔥 Unfreeze
              </button>
            </div>
          </div>
          
          <!-- State Transitions Section (shown for DRAFT/REVIEWED/APPROVED plans) -->
          <div class="governance-section state-transitions" id="state-transitions" style="display: none;">
            <h4>🔄 Plan Transitions</h4>
            <div class="transition-flow">
              <div class="transition-step">
                <button class="transition-btn" id="btn-review" disabled>
                  📝 Review
                </button>
              </div>
              <div class="transition-arrow">→</div>
              <div class="transition-step">
                <button class="transition-btn" id="btn-approve-transition" disabled>
                  ✅ Approve
                </button>
              </div>
              <div class="transition-arrow">→</div>
              <div class="transition-step">
                <button class="transition-btn" id="btn-activate" disabled>
                  🚀 Activate
                </button>
              </div>
              <div class="transition-arrow">→</div>
              <div class="transition-step">
                <button class="transition-btn" id="btn-execute" disabled>
                  ⚡ Execute
                </button>
              </div>
            </div>
            <div class="transition-actions">
              <button class="gov-btn danger" id="btn-cancel-plan" title="Cancel current plan">
                ❌ Cancel Plan
              </button>
            </div>
          </div>
          
          <!-- Pending Decision Section (shown when DRAFT state) -->
          <div class="governance-section pending-decision" id="pending-decision" style="display: none;">
            <h4>📋 Pending Decision</h4>
            <div class="decision-summary" id="decision-summary">
              <p>No pending decisions</p>
            </div>
            <div class="decision-actions">
              <button class="gov-btn success" id="btn-approve">
                ✅ Approve
              </button>
              <button class="gov-btn danger" id="btn-reject">
                ❌ Reject
              </button>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Approval Modal -->
      <div class="governance-modal" id="approval-modal" style="display: none;">
        <div class="modal-content">
          <div class="modal-header">
            <h3>📋 Decision Approval</h3>
            <button class="modal-close" id="modal-close" aria-label="Close">&times;</button>
          </div>
          <div class="modal-body" id="modal-body">
            <!-- Dynamic content -->
          </div>
          <div class="modal-footer">
            <button class="gov-btn secondary" id="modal-cancel">Cancel</button>
            <button class="gov-btn danger" id="modal-reject">Reject</button>
            <button class="gov-btn success" id="modal-approve">Approve</button>
          </div>
        </div>
      </div>
      
      <!-- Confirmation Modal for Critical Actions -->
      <div class="governance-modal confirmation-modal" id="confirmation-modal" style="display: none;">
        <div class="modal-content">
          <div class="modal-header">
            <h3><span id="confirmation-icon">⚠️</span> <span id="confirmation-title">Confirm Action</span></h3>
            <button class="modal-close" id="confirmation-close" aria-label="Close">&times;</button>
          </div>
          <div class="modal-body">
            <div class="confirmation-message" id="confirmation-message">
              Are you sure you want to perform this action?
            </div>
            <div class="confirmation-details" id="confirmation-details" style="display: none;">
              <!-- Dynamic details -->
            </div>
            <div class="idempotency-section" style="margin-top: 15px;">
              <label>
                <input type="checkbox" id="confirmation-understood"> 
                I understand the consequences of this action
              </label>
              <div class="idempotency-info" style="margin-top: 8px; font-size: 0.85em; color: #666;">
                <span class="idempotency-key" id="idempotency-display" style="font-family: monospace;"></span>
              </div>
            </div>
          </div>
          <div class="modal-footer">
            <button class="gov-btn secondary" id="confirmation-cancel">Cancel</button>
            <button class="gov-btn danger" id="confirmation-confirm" disabled>
              <span id="confirmation-action-text">Confirm</span>
            </button>
          </div>
        </div>
      </div>
    `;
    
    this.container.innerHTML = html;
  }
  
  bindEvents() {
    // Toggle panel
    document.getElementById('governance-toggle').addEventListener('click', () => {
      this.togglePanel();
    });
    
    // Mode selector
    document.getElementById('btn-change-mode').addEventListener('click', () => {
      this.changeGovernanceMode();
    });

    // Action buttons
    document.getElementById('btn-refresh').addEventListener('click', () => {
      this.refreshState();
    });

    document.getElementById('btn-propose').addEventListener('click', () => {
      this.proposeDecision();
    });
    
    document.getElementById('btn-freeze').addEventListener('click', () => {
      this.showConfirmation({
        title: 'Freeze System',
        icon: '❄️',
        message: 'This will immediately stop all automated trading and put the system in emergency mode.',
        details: `
          <div class="confirmation-warning">
            <strong>Warning:</strong> This action will:
            <ul>
              <li>Stop all active executions immediately</li>
              <li>Prevent new decisions from being generated</li>
              <li>Require manual intervention to resume</li>
            </ul>
          </div>
        `,
        actionText: 'Freeze System',
        onConfirm: () => this.freezeSystem()
      });
    });
    
    document.getElementById('btn-unfreeze').addEventListener('click', () => {
      this.showConfirmation({
        title: 'Unfreeze System',
        icon: '🔥', 
        message: 'This will restore the system to normal operation.',
        details: `
          <div class="confirmation-info">
            <strong>Note:</strong> This will:
            <ul>
              <li>Resume automated decision making</li>
              <li>Allow new executions to proceed</li>
              <li>Return to the previous governance mode</li>
            </ul>
          </div>
        `,
        actionText: 'Unfreeze System',
        onConfirm: () => this.unfreezeSystem()
      });
    });
    
    document.getElementById('btn-cooldown-check').addEventListener('click', () => {
      this.checkCooldownStatus();
    });
    
    // State transition buttons
    document.getElementById('btn-review').addEventListener('click', () => {
      this.reviewPlan();
    });
    
    document.getElementById('btn-approve-transition').addEventListener('click', () => {
      this.approvePlan();
    });
    
    document.getElementById('btn-activate').addEventListener('click', () => {
      this.activatePlan();
    });
    
    document.getElementById('btn-execute').addEventListener('click', () => {
      this.executePlan();
    });
    
    document.getElementById('btn-cancel-plan').addEventListener('click', () => {
      this.cancelPlan();
    });
    
    document.getElementById('btn-approve').addEventListener('click', () => {
      this.showApprovalModal();
    });
    
    document.getElementById('btn-reject').addEventListener('click', () => {
      this.rejectDecision();
    });
    
    // Modal events
    document.getElementById('modal-close').addEventListener('click', () => {
      this.hideApprovalModal();
    });
    
    document.getElementById('modal-cancel').addEventListener('click', () => {
      this.hideApprovalModal();
    });
    
    document.getElementById('modal-approve').addEventListener('click', () => {
      this.approveDecision();
    });
    
    document.getElementById('modal-reject').addEventListener('click', () => {
      this.rejectDecision();
      this.hideApprovalModal();
    });
    
    // Confirmation modal events
    document.getElementById('confirmation-close').addEventListener('click', () => {
      this.hideConfirmation();
    });
    
    document.getElementById('confirmation-cancel').addEventListener('click', () => {
      this.hideConfirmation();
    });
    
    document.getElementById('confirmation-understood').addEventListener('change', (e) => {
      const confirmBtn = document.getElementById('confirmation-confirm');
      confirmBtn.disabled = !e.target.checked;
    });
    
    document.getElementById('confirmation-confirm').addEventListener('click', () => {
      if (this.currentConfirmationAction) {
        this.hideConfirmation();
        this.currentConfirmationAction();
      }
    });
  }
  
  togglePanel() {
    const content = document.getElementById('governance-content');
    const icon = document.querySelector('.toggle-icon');
    
    this.isVisible = !this.isVisible;
    content.style.display = this.isVisible ? 'block' : 'none';
    icon.textContent = this.isVisible ? '▼' : '▲';
    
    // Save state
    localStorage.setItem('governance_panel_visible', this.isVisible.toString());
  }
  
  async refreshState(options = {}) {
    const { silent = false } = options;
    try {
      const refreshBtn = document.getElementById('btn-refresh');
      if (refreshBtn) {
        refreshBtn.disabled = true;
        refreshBtn.innerHTML = '⏳ Refreshing...';
      }

      // Try to sync governance state with graceful error handling
      try {
        if (store && typeof store.syncGovernanceState === 'function') {
          await store.syncGovernanceState();
        }
      } catch (govError) {
        (window.debugLogger?.warn || console.warn)('Governance state sync failed (non-critical):', govError.message);
        store.set('ui.apiStatus.backend', 'unavailable');
      }

      // Try to sync ML signals with graceful error handling
      try {
        if (store && typeof store.syncMLSignals === 'function') {
          await store.syncMLSignals();
        }
      } catch (mlError) {
        (window.debugLogger?.warn || console.warn)('ML signals sync failed (non-critical):', mlError.message);
        store.set('governance.ml_signals', null);
        store.set('ui.apiStatus.signals', 'unavailable');
      }

      // Try to sync alerts with graceful error handling
      try {
        await this.syncAlerts();
      } catch (alertError) {
        (window.debugLogger?.warn || console.warn)('Alerts sync failed (non-critical):', alertError.message);
      }

      // Update UI
      this.updateDisplay();

      if (refreshBtn) {
        refreshBtn.disabled = false;
        refreshBtn.innerHTML = '🔄 Refresh';
      }

      if (!silent) {
        this.showNotification('State refreshed successfully', 'success');
      }

    } catch (error) {
      debugLogger.error('Critical error in refresh state:', error);
      if (!silent) {
        this.showNotification('Refresh completed with warnings - check console', 'warning');
      }
      
      // Reset button state
      const refreshBtn = document.getElementById('btn-refresh');
      if (refreshBtn) {
        refreshBtn.disabled = false;
        refreshBtn.innerHTML = '🔄 Refresh';
      }
    }
  }
  
  async syncAlerts() {
    try {
      // Fetch active alerts from the alerts system
      const response = await fetch(`${window.location.origin}/api/alerts/active?include_snoozed=false`);
      if (response.ok) {
        const alerts = await response.json();
        
        // Update alerts badge
        this.updateAlertsBadge(alerts.length);
        
        // Update transparency level based on alert severity
        if (alerts.length > 0) {
          const hasCritical = alerts.some(alert => alert.severity === 'S3');
          const hasMajor = alerts.some(alert => alert.severity === 'S2');
          
          if (hasCritical) {
            this.updateTransparencyBadge('Critical', 'Critical alerts active - enhanced transparency');
          } else if (hasMajor) {
            this.updateTransparencyBadge('High', 'Major alerts active - increased monitoring');
          } else {
            this.updateTransparencyBadge('Normal', 'Minor alerts active');
          }
        } else {
          this.updateTransparencyBadge('Normal', 'No active alerts');
        }
        
        // Store alerts for potential use
        store.set('alerts.active', alerts);
        
      } else if (response.status === 503) {
        // Alert engine not initialized - not an error in development
        (window.debugLogger?.warn || console.warn)('Alert engine not yet initialized');
        this.updateAlertsBadge(0);
        this.updateTransparencyBadge('Normal', 'Alert system initializing...');
      }
    } catch (error) {
      (window.debugLogger?.warn || console.warn)('Could not sync alerts:', error.message);
      // Non-blocking error - alerts system might not be running
      this.updateAlertsBadge(0);
      this.updateTransparencyBadge('Normal', 'Alert system unavailable');
    }
  }
  
  updateDisplay() {
    // Get data with safe fallbacks
    const governanceStatus = store && typeof store.getGovernanceStatus === 'function' 
      ? store.getGovernanceStatus() 
      : { state: 'UNKNOWN', mode: null, confidence: 'Unknown', contradiction_level: 'Unknown' };
      
    const governance = store && typeof store.get === 'function' 
      ? store.get('governance') 
      : null;
      
    const mlSignals = store && typeof store.get === 'function' 
      ? store.get('governance.ml_signals') 
      : null;
    
    // Update status badges
    const stateEl = document.getElementById('gov-state');
    stateEl.textContent = governanceStatus.state;
    stateEl.className = `status-badge ${this.getStateClass(governanceStatus.state)}`;
    
    // Update state timestamp
    const timestampEl = document.getElementById('state-timestamp');
    if (governance?.last_update) {
      const updateTime = new Date(governance.last_update);
      timestampEl.textContent = ` (${updateTime.toLocaleTimeString()})`;
      timestampEl.style.display = 'inline';
    } else {
      timestampEl.style.display = 'none';
    }
    
    document.getElementById('gov-mode').textContent = governanceStatus.mode || '--';
    
    // Update auto-unfreeze timer if system is frozen
    if (governanceStatus.mode === 'freeze' && governance?.auto_unfreeze_at) {
      this.updateFreezeTimer(governance.auto_unfreeze_at);
    } else {
      const freezeStatusCard = document.getElementById('freeze-status-card');
      if (freezeStatusCard) freezeStatusCard.style.display = 'none';
    }
    
    // Update mode selector
    const modeSelect = document.getElementById('governance-mode-select');
    if (modeSelect) {
      modeSelect.value = governanceStatus.mode === 'freeze' ? 'manual' : governanceStatus.mode;
    }
    
    // Update policy info from active_policy
    const activePolicy = governance?.active_policy;
    if (activePolicy) {
      const capPercent = Math.round(activePolicy.cap_daily * 100);
      document.getElementById('gov-policy').textContent = `${activePolicy.mode} ${capPercent}%`;
    } else {
      document.getElementById('gov-policy').textContent = '--';
    }
    
    // Update contradiction index
    const contradiction = governance?.contradiction_index;
    const contradictionEl = document.getElementById('gov-contradiction');
    contradictionEl.textContent = Number.isFinite(contradiction) ? `${Math.round(contradiction * 100)}%` : '--';
    contradictionEl.className = Number.isFinite(contradiction)
      ? `contradiction-meter ${this.getContradictionClass(contradiction)}`
      : 'contradiction-meter';
    
    // Cooldown stays unavailable until the backend supplies a concrete value.
    const cooldownEl = document.getElementById('gov-cooldown');
    cooldownEl.textContent = '--';
    cooldownEl.className = 'cooldown-status';
    
    // Update ETag info (optional - only if element exists)
    const etagEl = document.getElementById('gov-etag');
    if (etagEl) {
      const currentEtag = governance?.etag || '--';
      etagEl.textContent = currentEtag.slice(-8); // Show last 8 chars
      etagEl.title = currentEtag; // Full ETag on hover
    }
    
    this.updateMLSignals(mlSignals);
    
    // Update freeze/unfreeze buttons
    const freezeBtn = document.getElementById('btn-freeze');
    const unfreezeBtn = document.getElementById('btn-unfreeze');
    
    // Check if system is frozen based on governance mode
    const isFrozen = governanceStatus.mode === 'freeze';
    if (isFrozen) {
      freezeBtn.disabled = true;
      unfreezeBtn.disabled = false;
    } else {
      freezeBtn.disabled = false;
      unfreezeBtn.disabled = true;
    }
    
    // Show/hide pending decision section
    this.updatePendingDecision(governanceStatus);
    
    // Show/hide state transitions section and update button states
    this.updateStateTransitions(governanceStatus);
  }
  
  updateMLSignals(signals) {
    // Update confidence card
    const confidence = signals?.confidence;
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceLabel = document.getElementById('confidence-label');
    
    if (confidenceFill && confidenceValue && confidenceLabel) {
      confidenceFill.style.width = Number.isFinite(confidence) ? `${confidence * 100}%` : '0%';
      confidenceFill.className = Number.isFinite(confidence)
        ? `signal-fill-compact ${this.getConfidenceClass(confidence)}`
        : 'signal-fill-compact';
      confidenceValue.textContent = Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : '--';
      confidenceLabel.textContent = Number.isFinite(confidence) ? this.getConfidenceLabel(confidence) : 'Unavailable';
    }
    
    // Update decision score card
    const decisionScore = signals?.decision_score;
    const decisionFill = document.getElementById('decision-fill');
    const decisionValue = document.getElementById('decision-value');
    const decisionLabel = document.getElementById('decision-label');
    
    if (decisionFill && decisionValue && decisionLabel) {
      decisionFill.style.width = Number.isFinite(decisionScore) ? `${decisionScore * 100}%` : '0%';
      decisionFill.className = Number.isFinite(decisionScore)
        ? `signal-fill-compact ${this.getDecisionClass(decisionScore)}`
        : 'signal-fill-compact';
      decisionValue.textContent = Number.isFinite(decisionScore) ? `${Math.round(decisionScore * 100)}%` : '--';
      decisionLabel.textContent = Number.isFinite(decisionScore) ? this.getDecisionLabel(decisionScore) : 'Unavailable';
    }
  }
  
  updatePendingDecision(governanceStatus) {
    const pendingSection = document.getElementById('pending-decision');
    
    if (governanceStatus.state === 'DRAFT') {
      pendingSection.style.display = 'block';
      const summary = document.getElementById('decision-summary');
      summary.innerHTML = `
        <div class="decision-preview">
          <p><strong>Proposed Decision:</strong> New allocation targets available</p>
          <p><strong>Confidence:</strong> ${governanceStatus.confidence || 'N/A'}</p>
          <p><strong>Contradiction:</strong> ${governanceStatus.contradiction_level || 'Low'}</p>
          <p><strong>Recommended Action:</strong> Review and approve if confident</p>
        </div>
      `;
    } else {
      pendingSection.style.display = 'none';
    }
  }
  
  async freezeSystem() {
    try {
      const idempotencyKey = this.currentIdempotencyKey || this.generateIdempotencyKey();
      const result = await store.freezeSystem('Manual freeze from UI', {
        idempotencyKey,
        ttl_minutes: 360, // 6 hours default
        source: 'governance_panel'
      });
      
      if (result) {
        this.showNotification(`System frozen successfully (Key: ${idempotencyKey.slice(0, 8)})`, 'success');
        this.refreshState();
        // Update transparency badge
        this.updateTransparencyBadge('High', 'System frozen - enhanced monitoring active');
      } else {
        this.showNotification('Failed to freeze system', 'error');
      }
    } catch (error) {
      debugLogger.error('Freeze system error:', error);
      if (error.message && error.message.includes('idempotent')) {
        this.showNotification('Action already processed (duplicate request detected)', 'warning');
      } else {
        this.showNotification('Error freezing system', 'error');
      }
    }
  }
  
  async unfreezeSystem() {
    try {
      const idempotencyKey = this.currentIdempotencyKey || this.generateIdempotencyKey();
      const result = await store.unfreezeSystem({
        idempotencyKey,
        source: 'governance_panel'
      });
      
      if (result) {
        this.showNotification(`System unfrozen successfully (Key: ${idempotencyKey.slice(0, 8)})`, 'success');
        this.refreshState();
        // Reset transparency badge
        this.updateTransparencyBadge('Normal', 'System operational');
      } else {
        this.showNotification('Failed to unfreeze system', 'error');
      }
    } catch (error) {
      debugLogger.error('Unfreeze system error:', error);
      if (error.message && error.message.includes('idempotent')) {
        this.showNotification('Action already processed (duplicate request detected)', 'warning');
      } else {
        this.showNotification('Error unfreezing system', 'error');
      }
    }
  }

  async changeGovernanceMode() {
    try {
      const modeSelect = document.getElementById('governance-mode-select');
      const selectedMode = modeSelect.value;
      const currentMode = store.getGovernanceStatus().mode;
      
      if (selectedMode === currentMode) {
        this.showNotification('Mode already set to ' + selectedMode, 'warning');
        return;
      }

      const btn = document.getElementById('btn-change-mode');
      btn.disabled = true;
      btn.textContent = 'Changing...';

      const result = await store.setGovernanceMode(selectedMode, `Mode change: ${currentMode} → ${selectedMode}`);
      if (result) {
        this.showNotification(`Mode changed to ${selectedMode}`, 'success');
        this.refreshState();
      } else {
        this.showNotification('Failed to change mode', 'error');
      }

      btn.disabled = false;
      btn.textContent = 'Change Mode';
      
    } catch (error) {
      debugLogger.error('Change mode error:', error);
      this.showNotification('Error changing mode', 'error');
    }
  }

  async proposeDecision() {
    try {
      const btn = document.getElementById('btn-propose');
      btn.disabled = true;
      btn.textContent = 'Proposing...';

      const currentMode = store.getGovernanceStatus().mode || 'unknown';
      const proposed = store.get('targets.proposed');
      const targets = proposed && typeof proposed === 'object'
        ? Object.entries(proposed)
          .filter(([symbol, weight]) => symbol !== 'model_version' && Number.isFinite(weight) && weight > 0)
          .map(([symbol, weight]) => ({ symbol, weight: weight / 100 }))
        : [];
      const totalWeight = targets.reduce((sum, target) => sum + target.weight, 0);
      if (targets.length === 0 || Math.abs(totalWeight - 1) > 0.001) {
        this.showNotification('Select a valid target allocation before proposing a decision', 'warning');
        btn.disabled = false;
        btn.textContent = '📋 Propose';
        return;
      }
      const reason = `Selected target allocation in ${currentMode} mode`;

      const result = await store.proposeDecision(targets, reason);
      if (result) {
        this.showNotification('Decision proposed successfully', 'success');
        this.refreshState();
      } else {
        this.showNotification('Failed to propose decision', 'error');
      }

      btn.disabled = false;
      btn.textContent = '📋 Propose';
      
    } catch (error) {
      debugLogger.error('Propose decision error:', error);
      this.showNotification('Error proposing decision', 'error');
    }
  }
  
  showApprovalModal() {
    const modal = document.getElementById('approval-modal');
    const modalBody = document.getElementById('modal-body');
    
    const governance = store.get('governance');
    const mlSignals = store.get('governance.ml_signals');
    
    modalBody.innerHTML = `
      <div class="approval-details">
        <div class="detail-section">
          <h4>📊 Decision Summary</h4>
          <p>The system has proposed a new allocation strategy based on current ML signals.</p>
        </div>
        
        <div class="detail-section">
          <h4>🧠 ML Analysis</h4>
          <table class="details-table">
            <tr>
              <td>Confidence Level:</td>
              <td><strong>${Number.isFinite(mlSignals?.confidence) ? `${Math.round(mlSignals.confidence * 100)}%` : '--'}</strong></td>
            </tr>
            <tr>
              <td>Decision Score:</td>
              <td><strong>${Number.isFinite(mlSignals?.decision_score) ? `${Math.round(mlSignals.decision_score * 100)}%` : '--'}</strong></td>
            </tr>
            <tr>
              <td>Contradiction Index:</td>
              <td><strong>${Number.isFinite(governance?.contradiction_index) ? `${Math.round(governance.contradiction_index * 100)}%` : '--'}</strong></td>
            </tr>
            <tr>
              <td>Market Regime:</td>
              <td><strong>${this.getRegimeSummary(mlSignals)}</strong></td>
            </tr>
          </table>
        </div>
        
        <div class="detail-section">
          <h4>⚙️ Execution Policy</h4>
          <table class="details-table">
            <tr>
              <td>Mode:</td>
              <td><strong>${governance?.mode || '--'}</strong></td>
            </tr>
            <tr>
              <td>State:</td>
              <td><strong>${governance?.current_state || 'UNKNOWN'}</strong></td>
            </tr>
            <tr>
              <td>Contradiction:</td>
              <td><strong>${Number.isFinite(governance?.contradiction_index) ? `${Math.round(governance.contradiction_index * 100)}%` : '--'}</strong></td>
            </tr>
          </table>
        </div>
        
        <div class="detail-section warning">
          <h4>⚠️ Important Notes</h4>
          <ul>
            <li>This decision will override current allocation targets</li>
            <li>Execution will follow the policy constraints above</li>
            <li>You can freeze the system anytime to stop execution</li>
          </ul>
        </div>
      </div>
    `;
    
    modal.style.display = 'flex';
  }
  
  hideApprovalModal() {
    document.getElementById('approval-modal').style.display = 'none';
  }
  
  showConfirmation(options) {
    this.currentConfirmationAction = options.onConfirm;
    this.currentIdempotencyKey = this.generateIdempotencyKey();
    
    // Set modal content
    document.getElementById('confirmation-icon').textContent = options.icon || '⚠️';
    document.getElementById('confirmation-title').textContent = options.title || 'Confirm Action';
    document.getElementById('confirmation-message').textContent = options.message || 'Are you sure?';
    document.getElementById('confirmation-action-text').textContent = options.actionText || 'Confirm';
    
    // Set details if provided
    const detailsEl = document.getElementById('confirmation-details');
    if (options.details) {
      detailsEl.innerHTML = options.details;
      detailsEl.style.display = 'block';
    } else {
      detailsEl.style.display = 'none';
    }
    
    // Show idempotency key
    document.getElementById('idempotency-display').textContent = 
      `Idempotency Key: ${this.currentIdempotencyKey}`;
    
    // Reset form
    document.getElementById('confirmation-understood').checked = false;
    document.getElementById('confirmation-confirm').disabled = true;
    
    // Show modal
    document.getElementById('confirmation-modal').style.display = 'flex';
  }
  
  hideConfirmation() {
    document.getElementById('confirmation-modal').style.display = 'none';
    this.currentConfirmationAction = null;
    this.currentIdempotencyKey = null;
  }
  
  generateIdempotencyKey() {
    // Generate UUID v4-like key
    return 'xxxx-xxxx-4xxx-yxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c == 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }
  
  updateTransparencyBadge(level = 'Normal', tooltip = '') {
    const badge = document.getElementById('transparency-badge');
    const levelEl = document.getElementById('transparency-level');
    
    if (!badge || !levelEl) return;
    
    levelEl.textContent = level;
    badge.title = tooltip || `Transparency Level: ${level}`;
    
    // Update badge styling based on level
    badge.className = 'transparency-badge';
    switch(level.toLowerCase()) {
      case 'high':
        badge.classList.add('transparency-high');
        break;
      case 'critical':
        badge.classList.add('transparency-critical');
        break;
      default:
        badge.classList.add('transparency-normal');
    }
  }
  
  updateAlertsBadge(alertCount = 0) {
    const badge = document.getElementById('alerts-badge');
    const countEl = document.getElementById('alerts-count');
    
    if (!badge || !countEl) return;
    
    countEl.textContent = alertCount;
    
    if (alertCount > 0) {
      badge.style.display = 'flex';
      badge.title = `${alertCount} active alert${alertCount > 1 ? 's' : ''}`;
      badge.className = 'alerts-badge';
      
      // Severity-based styling
      if (alertCount >= 3) {
        badge.classList.add('alerts-critical');
      } else if (alertCount >= 2) {
        badge.classList.add('alerts-warning');  
      } else {
        badge.classList.add('alerts-info');
      }
    } else {
      badge.style.display = 'none';
    }
  }
  
  updateFreezeTimer(autoUnfreezeAt) {
    const freezeStatusCard = document.getElementById('freeze-status-card');
    const timerEl = document.getElementById('freeze-timer');
    const progressBar = document.getElementById('freeze-progress-bar');
    
    if (!autoUnfreezeAt) {
      if (freezeStatusCard) freezeStatusCard.style.display = 'none';
      return;
    }
    
    if (freezeStatusCard) freezeStatusCard.style.display = 'block';
    
    const now = new Date();
    const unfreezeTime = new Date(autoUnfreezeAt);
    const totalTime = 6 * 60 * 60 * 1000; // 6 hours in ms (default)
    const remainingTime = unfreezeTime - now;
    
    if (remainingTime <= 0) {
      timerEl.textContent = 'Auto-unfreeze due';
      progressBar.style.width = '100%';
      return;
    }
    
    // Format remaining time
    const hours = Math.floor(remainingTime / (60 * 60 * 1000));
    const minutes = Math.floor((remainingTime % (60 * 60 * 1000)) / (60 * 1000));
    timerEl.textContent = `${hours}h ${minutes}m remaining`;
    
    // Update progress bar
    const progress = ((totalTime - remainingTime) / totalTime) * 100;
    progressBar.style.width = `${Math.min(progress, 100)}%`;
  }
  
  async approveDecision() {
    try {
      const decisionId = store.get('governance.last_decision_id') || 'current';
      const result = await store.approveDecision(decisionId, true, 'Approved from UI');
      if (result) {
        this.showNotification('Decision approved successfully', 'success');
        this.hideApprovalModal();
        this.refreshState();
      } else {
        this.showNotification('Failed to approve decision', 'error');
      }
    } catch (error) {
      debugLogger.error('Approve decision error:', error);
      this.showNotification('Error approving decision', 'error');
    }
  }
  
  async rejectDecision() {
    try {
      // For now, we'll just reset to manual mode
      // In a full implementation, this would call a reject endpoint
      this.showNotification('Decision rejected (feature in development)', 'warning');
    } catch (error) {
      debugLogger.error('Reject decision error:', error);
      this.showNotification('Error rejecting decision', 'error');
    }
  }
  
  startAutoRefresh() {
    // Auto-refresh every 30 seconds
    this.refreshInterval = setInterval(() => {
      if (this.isVisible) {
        this.refreshState();
      }
    }, 30000);
  }
  
  destroy() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
    if (typeof this.unsubscribeStore === 'function') {
      this.unsubscribeStore();
      this.unsubscribeStore = null;
    }
  }
  
  // Helper methods for styling
  getStateClass(state) {
    switch (state) {
      case 'DRAFT': return 'warning';
      case 'REVIEWED': return 'info';
      case 'APPROVED': return 'primary';
      case 'ACTIVE': return 'success';
      case 'EXECUTED': return 'completed';
      case 'CANCELLED': return 'danger';
      case 'FROZEN': return 'danger';
      default: return 'secondary';
    }
  }
  
  getContradictionClass(level) {
    if (level > 0.7) return 'high';
    if (level > 0.4) return 'medium';
    return 'low';
  }
  
  getConfidenceClass(confidence) {
    if (confidence > 0.8) return 'excellent';
    if (confidence > 0.6) return 'good';
    if (confidence > 0.4) return 'fair';
    return 'poor';
  }
  
  getDecisionClass(score) {
    if (score > 0.7) return 'strong';
    if (score > 0.5) return 'moderate';
    return 'weak';
  }
  
  getConfidenceLabel(confidence) {
    if (confidence > 0.8) return 'Excellent';
    if (confidence > 0.6) return 'Good';
    if (confidence > 0.4) return 'Fair';
    return 'Poor';
  }
  
  getDecisionLabel(score) {
    if (score > 0.7) return 'Strong Signal';
    if (score > 0.5) return 'Moderate Signal';
    return 'Weak Signal';
  }
  
  getRegimeSummary(signals) {
    if (!signals?.regime) return 'Unknown';
    
    const regime = signals.regime;
    const maxRegime = Object.keys(regime).reduce((a, b) => 
      regime[a] > regime[b] ? a : b
    );
    
    return maxRegime.charAt(0).toUpperCase() + maxRegime.slice(1);
  }
  
  // NEW METHODS FOR PHASE 0 FUNCTIONALITY

  updateStateTransitions(governanceStatus) {
    const transitionsSection = document.getElementById('state-transitions');
    const state = governanceStatus.state;
    
    // Show transitions section for plans that can be transitioned
    if (['DRAFT', 'REVIEWED', 'APPROVED', 'ACTIVE'].includes(state)) {
      transitionsSection.style.display = 'block';
      this.updateTransitionButtons(state);
    } else {
      transitionsSection.style.display = 'none';
    }
  }
  
  updateTransitionButtons(currentState) {
    // Reset all buttons
    const buttons = ['btn-review', 'btn-approve-transition', 'btn-activate', 'btn-execute'];
    buttons.forEach(id => {
      const btn = document.getElementById(id);
      btn.disabled = true;
      btn.classList.remove('active', 'completed');
    });
    
    // Enable appropriate button based on current state
    switch (currentState) {
      case 'DRAFT':
        document.getElementById('btn-review').disabled = false;
        break;
      case 'REVIEWED':
        document.getElementById('btn-review').classList.add('completed');
        document.getElementById('btn-approve-transition').disabled = false;
        break;
      case 'APPROVED':
        document.getElementById('btn-review').classList.add('completed');
        document.getElementById('btn-approve-transition').classList.add('completed');
        document.getElementById('btn-activate').disabled = false;
        break;
      case 'ACTIVE':
        ['btn-review', 'btn-approve-transition', 'btn-activate'].forEach(id => {
          document.getElementById(id).classList.add('completed');
        });
        document.getElementById('btn-execute').disabled = false;
        break;
    }
  }
  
  async checkCooldownStatus() {
    try {
      const apiUrl = window.globalConfig?.getApiUrl('/execution/governance/cooldown-status') || 
                   '/execution/governance/cooldown-status';
      
      const response = await fetch(apiUrl);
      const data = await response.json();
      
      const cooldownEl = document.getElementById('gov-cooldown');
      if (data.can_publish) {
        cooldownEl.textContent = "Ready";
        cooldownEl.className = "cooldown-status ready";
      } else {
        cooldownEl.textContent = "Cooling down";
        cooldownEl.className = "cooldown-status cooling";
      }
      
      this.showNotification(`Cooldown: ${data.reason}`, data.can_publish ? 'success' : 'warning');
      
    } catch (error) {
      debugLogger.error('Error checking cooldown:', error);
      this.showNotification('Error checking cooldown status', 'error');
    }
  }
  
  async reviewPlan() {
    try {
      const planId = store.get('governance.last_decision_id') || 'current';
      const etag = store.get('governance.etag');
      
      const url = window.globalConfig?.getApiUrl(`/execution/governance/review/${planId}`) || 
                 `/execution/governance/review/${planId}`;
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(etag && { 'If-Match': etag })
        },
        body: JSON.stringify({
          reviewed_by: 'UI_User',
          notes: 'Reviewed via GovernancePanel UI'
        })
      });
      
      if (response.ok) {
        this.showNotification('Plan reviewed successfully', 'success');
        this.refreshState();
      } else {
        this.showNotification('Failed to review plan', 'error');
      }
    } catch (error) {
      debugLogger.error('Review plan error:', error);
      this.showNotification('Error reviewing plan', 'error');
    }
  }
  
  async approvePlan() {
    try {
      const planId = store.get('governance.last_decision_id') || 'current';
      const etag = store.get('governance.etag');
      
      const url = window.globalConfig?.getApiUrl(`/execution/governance/approve/${planId}`) || 
                 `/execution/governance/approve/${planId}`;
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(etag && { 'If-Match': etag })
        },
        body: JSON.stringify({
          resource_type: 'plan',
          approved: true,
          approved_by: 'UI_User',
          notes: 'Approved via GovernancePanel UI'
        })
      });
      
      if (response.ok) {
        this.showNotification('Plan approved successfully', 'success');
        this.refreshState();
      } else {
        this.showNotification('Failed to approve plan', 'error');
      }
    } catch (error) {
      debugLogger.error('Approve plan error:', error);
      this.showNotification('Error approving plan', 'error');
    }
  }
  
  async activatePlan() {
    try {
      const planId = store.get('governance.last_decision_id') || 'current';
      const etag = store.get('governance.etag');
      
      const url = window.globalConfig?.getApiUrl(`/execution/governance/activate/${planId}`) || 
                 `/execution/governance/activate/${planId}`;
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(etag && { 'If-Match': etag })
        }
      });
      
      if (response.ok) {
        this.showNotification('Plan activated successfully', 'success');
        this.refreshState();
      } else {
        this.showNotification('Failed to activate plan', 'error');
      }
    } catch (error) {
      debugLogger.error('Activate plan error:', error);
      this.showNotification('Error activating plan', 'error');
    }
  }
  
  async executePlan() {
    try {
      const planId = store.get('governance.last_decision_id') || 'current';
      const etag = store.get('governance.etag');
      
      const url = window.globalConfig?.getApiUrl(`/execution/governance/execute/${planId}`) || 
                 `/execution/governance/execute/${planId}`;
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(etag && { 'If-Match': etag })
        }
      });
      
      if (response.ok) {
        this.showNotification('Plan marked as executed', 'success');
        this.refreshState();
      } else {
        this.showNotification('Failed to execute plan', 'error');
      }
    } catch (error) {
      debugLogger.error('Execute plan error:', error);
      this.showNotification('Error executing plan', 'error');
    }
  }
  
  async cancelPlan() {
    try {
      const planId = store.get('governance.last_decision_id') || 'current';
      const etag = store.get('governance.etag');
      
      const url = window.globalConfig?.getApiUrl(`/execution/governance/cancel/${planId}`) || 
                 `/execution/governance/cancel/${planId}`;
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(etag && { 'If-Match': etag })
        },
        body: JSON.stringify({
          cancelled_by: 'UI_User',
          reason: 'Cancelled via GovernancePanel UI'
        })
      });
      
      if (response.ok) {
        this.showNotification('Plan cancelled successfully', 'success');
        this.refreshState();
      } else {
        this.showNotification('Failed to cancel plan', 'error');
      }
    } catch (error) {
      debugLogger.error('Cancel plan error:', error);
      this.showNotification('Error cancelling plan', 'error');
    }
  }

  showNotification(message, type = 'info') {
    // Simple notification system
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 12px 16px;
      border-radius: 4px;
      z-index: 10000;
      max-width: 300px;
      max-height: 150px;
      height: auto;
      width: auto;
      min-height: auto;
      overflow: auto;
      white-space: pre-wrap;
      word-wrap: break-word;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    `;
    
    // Type-specific styling
    switch (type) {
      case 'success':
        notification.style.backgroundColor = '#d4edda';
        notification.style.color = '#155724';
        break;
      case 'error':
        notification.style.backgroundColor = '#f8d7da';
        notification.style.color = '#721c24';
        break;
      case 'warning':
        notification.style.backgroundColor = '#fff3cd';
        notification.style.color = '#856404';
        break;
      default:
        notification.style.backgroundColor = '#d1ecf1';
        notification.style.color = '#0c5460';
    }
    
    document.body.appendChild(notification);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 3000);
  }
}

export { GovernancePanel };

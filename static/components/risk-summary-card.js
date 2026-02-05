/**
 * RiskSummaryCard - Web Component réutilisable pour afficher les métriques de risque
 *
 * Trois niveaux de détail :
 * - compact  : Score principal + status badge (pour dashboard)
 * - detailed : Score + breakdown des sous-scores (pour analytics)
 * - full     : Tout + alerts + VaR + graphiques (pour risk page)
 *
 * Usage:
 *   <risk-summary-card level="compact"></risk-summary-card>
 *   <risk-summary-card level="detailed" poll-ms="30000"></risk-summary-card>
 *   <risk-summary-card level="full" show-alerts="true"></risk-summary-card>
 *
 * Attributs:
 *   - level: "compact" | "detailed" | "full" (default: "detailed")
 *   - poll-ms: Intervalle de polling en ms (0 = désactivé, default: 30000)
 *   - show-alerts: "true" | "false" (default: true pour full, false sinon)
 *   - link-to: URL vers la page détaillée (default: "risk-dashboard.html")
 *
 * @customElement risk-summary-card
 * @version 1.0.0
 * @since Feb 2026
 */

const LEVELS = {
    COMPACT: 'compact',
    DETAILED: 'detailed',
    FULL: 'full'
};

const template = document.createElement('template');
template.innerHTML = `
<style>
    :host {
        display: block;
        font-family: system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif;
    }

    .risk-card {
        background: var(--theme-surface, #ffffff);
        border: 1px solid var(--theme-border, #e2e8f0);
        border-radius: var(--radius-card, 8px);
        padding: var(--space-lg, 16px);
        transition: all 0.2s ease;
    }

    .risk-card:hover {
        border-color: var(--brand-primary, #3b82f6);
        box-shadow: var(--shadow-md, 0 4px 6px -1px rgba(0,0,0,0.1));
    }

    /* Header */
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-md, 12px);
        padding-bottom: var(--space-sm, 8px);
        border-bottom: 1px solid var(--theme-border, #e2e8f0);
    }

    .card-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 1rem;
        font-weight: 600;
        color: var(--theme-text, #1e293b);
        text-decoration: none;
    }

    .card-title:hover {
        color: var(--brand-primary, #3b82f6);
    }

    .card-icon {
        font-size: 1.25rem;
    }

    /* Status Badge */
    .status-badge {
        padding: 4px 10px;
        border-radius: var(--radius-badge, 4px);
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .status-healthy {
        background: var(--success-bg, #d1fae5);
        color: var(--success, #059669);
        border: 1px solid var(--success, #059669);
    }

    .status-warning {
        background: var(--warning-bg, #fef3c7);
        color: var(--warning, #f59e0b);
        border: 1px solid var(--warning, #f59e0b);
    }

    .status-danger {
        background: var(--danger-bg, #fee2e2);
        color: var(--danger, #ef4444);
        border: 1px solid var(--danger, #ef4444);
    }

    /* Main Score */
    .main-score {
        text-align: center;
        padding: var(--space-md, 12px) 0;
    }

    .score-label {
        font-size: 0.75rem;
        color: var(--theme-text-muted, #64748b);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }

    .score-value {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .score-value.healthy { color: var(--success, #059669); }
    .score-value.warning { color: var(--warning, #f59e0b); }
    .score-value.danger { color: var(--danger, #ef4444); }

    /* Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: var(--space-sm, 8px);
        margin-top: var(--space-md, 12px);
    }

    .metric-item {
        text-align: center;
        padding: var(--space-sm, 8px);
        background: var(--theme-surface-elevated, #f8fafc);
        border-radius: var(--radius-sm, 4px);
        border: 1px solid var(--theme-border, #e2e8f0);
    }

    .metric-label {
        font-size: 0.7rem;
        color: var(--theme-text-muted, #64748b);
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-bottom: 2px;
    }

    .metric-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--theme-text, #1e293b);
    }

    /* Alerts Section */
    .alerts-section {
        margin-top: var(--space-md, 12px);
        padding-top: var(--space-md, 12px);
        border-top: 1px solid var(--theme-border, #e2e8f0);
    }

    .alerts-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-sm, 8px);
    }

    .alerts-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--theme-text, #1e293b);
    }

    .alerts-count {
        font-size: 0.75rem;
        padding: 2px 8px;
        background: var(--danger-bg, #fee2e2);
        color: var(--danger, #ef4444);
        border-radius: 10px;
        font-weight: 600;
    }

    .alerts-list {
        max-height: 150px;
        overflow-y: auto;
    }

    .alert-item {
        padding: 8px 10px;
        margin-bottom: 6px;
        border-radius: var(--radius-sm, 4px);
        font-size: 0.8rem;
        line-height: 1.4;
        border-left: 3px solid var(--danger, #ef4444);
        background: var(--danger-bg, #fee2e2);
    }

    .alert-item.warning {
        border-left-color: var(--warning, #f59e0b);
        background: var(--warning-bg, #fef3c7);
    }

    .alert-item.info {
        border-left-color: var(--info, #3b82f6);
        background: var(--info-bg, #dbeafe);
    }

    /* Footer Stats */
    .footer-stats {
        margin-top: var(--space-md, 12px);
        padding-top: var(--space-sm, 8px);
        border-top: 1px solid var(--theme-border, #e2e8f0);
        font-size: 0.8rem;
        color: var(--theme-text-muted, #64748b);
    }

    .footer-stats strong {
        color: var(--theme-text, #1e293b);
    }

    /* Loading State */
    .loading {
        text-align: center;
        padding: var(--space-xl, 24px);
        color: var(--theme-text-muted, #64748b);
    }

    .spinner {
        width: 24px;
        height: 24px;
        border: 2px solid var(--theme-border, #e2e8f0);
        border-top-color: var(--brand-primary, #3b82f6);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        margin: 0 auto 8px;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Error State */
    .error {
        text-align: center;
        padding: var(--space-lg, 16px);
        background: var(--danger-bg, #fee2e2);
        color: var(--danger, #ef4444);
        border-radius: var(--radius-sm, 4px);
        font-size: 0.85rem;
    }

    /* Compact Mode Overrides */
    :host([level="compact"]) .metrics-grid,
    :host([level="compact"]) .alerts-section,
    :host([level="compact"]) .footer-stats {
        display: none;
    }

    :host([level="compact"]) .main-score {
        padding: var(--space-sm, 8px) 0;
    }

    :host([level="compact"]) .score-value {
        font-size: 2rem;
    }

    /* Detailed Mode - Hide Alerts */
    :host([level="detailed"]) .alerts-section {
        display: none;
    }
</style>

<div class="risk-card">
    <div class="loading">
        <div class="spinner"></div>
        Loading risk data...
    </div>
</div>
`;

class RiskSummaryCard extends HTMLElement {
    static get observedAttributes() {
        return ['level', 'poll-ms', 'show-alerts', 'link-to'];
    }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.appendChild(template.content.cloneNode(true));

        this._card = this.shadowRoot.querySelector('.risk-card');
        this._pollInterval = null;
        this._abortController = null;
    }

    get level() {
        return this.getAttribute('level') || LEVELS.DETAILED;
    }

    get pollMs() {
        return parseInt(this.getAttribute('poll-ms') || '30000', 10);
    }

    get showAlerts() {
        const attr = this.getAttribute('show-alerts');
        if (attr === 'true') return true;
        if (attr === 'false') return false;
        return this.level === LEVELS.FULL;
    }

    get linkTo() {
        return this.getAttribute('link-to') || 'risk-dashboard.html';
    }

    connectedCallback() {
        this._fetchData();

        if (this.pollMs > 0) {
            this._pollInterval = setInterval(() => this._fetchData(), this.pollMs);
        }
    }

    disconnectedCallback() {
        if (this._pollInterval) {
            clearInterval(this._pollInterval);
            this._pollInterval = null;
        }
        if (this._abortController) {
            this._abortController.abort();
        }
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue !== newValue && this.isConnected) {
            this._fetchData();
        }
    }

    async _fetchData() {
        if (this._abortController) {
            this._abortController.abort();
        }
        this._abortController = new AbortController();

        try {
            const activeUser = localStorage.getItem('activeUser') || 'demo';
            const response = await fetch('/api/risk/dashboard', {
                headers: { 'X-User': activeUser },
                signal: this._abortController.signal
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            this._render(this._normalizeData(data));
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error('[RiskSummaryCard] Fetch error:', error);
                this._renderError(error.message);
            }
        }
    }

    _normalizeData(apiJson) {
        if (!apiJson || typeof apiJson !== 'object') return {};

        const root = apiJson.risk || apiJson.data || apiJson;
        const scores = root.scores || {};
        const governance = root.governance || {};
        const alerts = Array.isArray(root.alerts) ? root.alerts :
                       (Array.isArray(root.active_alerts) ? root.active_alerts : []);

        // Risk score: plus haut = plus robuste (0-100)
        const riskScore = scores.risk ?? scores.riskScore ?? 50;

        // Blended Decision Index
        const blended = scores.blended ?? scores.blendedDecision ?? scores.decision_index ?? 50;

        // VaR
        const var95 = root.var_95 ?? root.portfolio_var ?? governance.var_95;

        return {
            riskScore,
            blended,
            onchain: scores.onchain ?? scores.onChain ?? 50,
            cycle: scores.cycle ?? root.ccs?.score ?? 50,
            var95,
            alerts,
            alertsCount: alerts.length,
            governance: {
                status: governance.status || 'unknown',
                cap: governance.cap_daily ?? governance.effective_cap,
                contradiction: governance.contradiction_index
            }
        };
    }

    _render(data) {
        const level = this.level;
        const statusClass = this._getStatusClass(data.riskScore);
        const statusLabel = this._getStatusLabel(data.riskScore);

        let html = `
            <div class="card-header">
                <a href="${this.linkTo}" class="card-title">
                    <span class="card-icon">🛡️</span>
                    Risk Summary
                </a>
                <span class="status-badge status-${statusClass}">${statusLabel}</span>
            </div>

            <div class="main-score">
                <div class="score-label">Risk Score</div>
                <div class="score-value ${statusClass}">${Math.round(data.riskScore)}</div>
            </div>
        `;

        // Metrics Grid (detailed & full)
        if (level !== LEVELS.COMPACT) {
            html += `
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">Cycle</div>
                        <div class="metric-value">${Math.round(data.cycle)}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">On-Chain</div>
                        <div class="metric-value">${Math.round(data.onchain)}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Decision</div>
                        <div class="metric-value">${Math.round(data.blended)}</div>
                    </div>
                </div>
            `;
        }

        // Alerts Section (full only or if show-alerts="true")
        if (this.showAlerts && data.alerts.length > 0) {
            html += `
                <div class="alerts-section">
                    <div class="alerts-header">
                        <span class="alerts-title">Active Alerts</span>
                        <span class="alerts-count">${data.alertsCount}</span>
                    </div>
                    <div class="alerts-list">
                        ${data.alerts.slice(0, 5).map(alert => `
                            <div class="alert-item ${alert.severity || 'warning'}">
                                ${alert.message || alert.title || 'Alert'}
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        // Footer Stats (full only)
        if (level === LEVELS.FULL) {
            html += `
                <div class="footer-stats">
                    <strong>Active Alerts:</strong> ${data.alertsCount}<br>
                    <strong>VaR (95%):</strong> ${data.var95 ? `${(data.var95 * 100).toFixed(1)}%` : '--'}
                </div>
            `;
        }

        this._card.innerHTML = html;
    }

    _renderError(message) {
        this._card.innerHTML = `
            <div class="error">
                ⚠️ Error loading risk data<br>
                <small>${message}</small>
            </div>
        `;
    }

    _getStatusClass(score) {
        if (score >= 70) return 'healthy';
        if (score >= 40) return 'warning';
        return 'danger';
    }

    _getStatusLabel(score) {
        if (score >= 70) return 'Low Risk';
        if (score >= 40) return 'Moderate';
        return 'High Risk';
    }

    // Public API
    refresh() {
        this._fetchData();
    }
}

if (!customElements.get('risk-summary-card')) {
    customElements.define('risk-summary-card', RiskSummaryCard);
}

export { RiskSummaryCard, LEVELS };

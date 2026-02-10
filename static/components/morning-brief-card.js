/**
 * MorningBriefCard — Web Component for the daily morning brief.
 *
 * Displays P&L summary, Decision Index, alerts count, top movers
 * with expandable sections and auto-refresh.
 *
 * Usage:
 *   <morning-brief-card auto-refresh="900"></morning-brief-card>
 *
 * Attributes:
 *   - auto-refresh: seconds between auto-refresh (default: 900 = 15min)
 *   - collapsed:    if present, start collapsed
 *
 * @customElement morning-brief-card
 */

const briefTemplate = document.createElement('template');
briefTemplate.innerHTML = `
<style>
  :host {
    display: block;
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif;
    color: var(--theme-text, #1e293b);
  }

  .brief-container {
    background: var(--theme-surface, #ffffff);
    border: 1px solid var(--theme-border, #e2e8f0);
    border-radius: 12px;
    overflow: hidden;
  }

  .brief-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: var(--theme-surface-elevated, #f8fafc);
    border-bottom: 1px solid var(--theme-border, #e2e8f0);
    cursor: pointer;
    user-select: none;
  }

  .brief-header:hover {
    background: var(--theme-surface-hover, #f1f5f9);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .header-left h3 {
    margin: 0;
    font-size: 0.95rem;
    font-weight: 600;
  }

  .brief-time {
    font-size: 0.75rem;
    color: var(--theme-text-muted, #64748b);
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .icon-btn {
    background: none;
    border: none;
    cursor: pointer;
    padding: 4px;
    font-size: 0.85rem;
    color: var(--theme-text-muted, #64748b);
    border-radius: 4px;
    transition: background 0.2s;
    line-height: 1;
  }
  .icon-btn:hover { background: var(--theme-border, #e2e8f0); }
  .icon-btn.spinning { animation: spin 1s linear infinite; }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .chevron {
    transition: transform 0.2s;
    font-size: 0.8rem;
  }
  .chevron.collapsed { transform: rotate(-90deg); }

  .brief-body {
    padding: 12px 16px;
    display: grid;
    gap: 12px;
  }

  .brief-body.hidden { display: none; }

  /* Sections */
  .section {
    display: grid;
    gap: 4px;
  }

  .section-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--theme-text-muted, #64748b);
    margin: 0;
  }

  /* P&L grid */
  .pnl-grid {
    display: grid;
    grid-template-columns: auto 1fr 1fr;
    gap: 4px 12px;
    font-size: 0.85rem;
  }

  .pnl-label {
    color: var(--theme-text-muted, #64748b);
    font-weight: 500;
  }

  .pnl-value {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  .positive { color: var(--color-success-600, #047857); }
  .negative { color: var(--color-danger-600, #b91c1c); }
  .neutral { color: var(--theme-text-muted, #64748b); }

  /* DI score */
  .di-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.9rem;
  }

  .di-score {
    font-size: 1.3rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }

  .di-meta {
    font-size: 0.75rem;
    color: var(--theme-text-muted, #64748b);
  }

  /* Alerts badge */
  .alert-row {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.85rem;
  }

  .alert-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 20px;
    height: 20px;
    padding: 0 6px;
    border-radius: 10px;
    font-size: 0.7rem;
    font-weight: 600;
    color: white;
  }

  .badge-s3 { background: var(--color-danger-600, #b91c1c); }
  .badge-s2 { background: var(--color-warning-600, #b45309); }
  .badge-s1 { background: var(--color-info-600, #1d4ed8); }

  /* Movers */
  .movers-list {
    display: grid;
    gap: 2px;
    font-size: 0.85rem;
  }

  .mover-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .mover-symbol {
    font-weight: 500;
  }

  /* Loading / Error */
  .loading, .error-msg {
    text-align: center;
    padding: 16px;
    font-size: 0.85rem;
    color: var(--theme-text-muted, #64748b);
  }

  .error-msg { color: var(--color-danger-600, #b91c1c); }
</style>

<div class="brief-container">
  <div class="brief-header">
    <div class="header-left">
      <span>&#x2600;&#xFE0F;</span>
      <h3>Morning Brief</h3>
      <span class="brief-time"></span>
    </div>
    <div class="header-actions">
      <button class="icon-btn refresh-btn" title="Refresh">&#x21bb;</button>
      <span class="chevron">&#x25BC;</span>
    </div>
  </div>
  <div class="brief-body">
    <div class="loading">Loading morning brief...</div>
  </div>
</div>
`;

class MorningBriefCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.appendChild(briefTemplate.content.cloneNode(true));

    this._body = this.shadowRoot.querySelector('.brief-body');
    this._header = this.shadowRoot.querySelector('.brief-header');
    this._chevron = this.shadowRoot.querySelector('.chevron');
    this._refreshBtn = this.shadowRoot.querySelector('.refresh-btn');
    this._timeEl = this.shadowRoot.querySelector('.brief-time');
    this._intervalId = null;
    this._collapsed = false;
  }

  connectedCallback() {
    // Toggle collapse
    this._header.addEventListener('click', (e) => {
      if (e.target.closest('.refresh-btn')) return;
      this._collapsed = !this._collapsed;
      this._body.classList.toggle('hidden', this._collapsed);
      this._chevron.classList.toggle('collapsed', this._collapsed);
    });

    // Refresh button
    this._refreshBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this._load(true);
    });

    // Initial collapsed state
    if (this.hasAttribute('collapsed')) {
      this._collapsed = true;
      this._body.classList.add('hidden');
      this._chevron.classList.add('collapsed');
    }

    // Load data
    this._load(false);

    // Auto-refresh
    const interval = parseInt(this.getAttribute('auto-refresh') || '900', 10);
    this._intervalId = setInterval(() => this._load(false), interval * 1000);
  }

  disconnectedCallback() {
    if (this._intervalId) {
      clearInterval(this._intervalId);
      this._intervalId = null;
    }
  }

  async _load(force = false) {
    this._refreshBtn.classList.add('spinning');

    try {
      const user = localStorage.getItem('activeUser') || 'demo';
      const forceParam = force ? '&force=true' : '';

      // Use auth headers if available
      const headers = { 'X-User': user };
      const token = localStorage.getItem('authToken');
      if (token) headers['Authorization'] = `Bearer ${token}`;

      const resp = await fetch(`/api/morning-brief?source=cointracking${forceParam}`, { headers });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const json = await resp.json();
      const brief = json.data || json;
      this._render(brief);
    } catch (err) {
      console.warn('Morning brief load failed:', err);
      this._body.innerHTML = `<div class="error-msg">Failed to load morning brief</div>`;
    } finally {
      this._refreshBtn.classList.remove('spinning');
    }
  }

  _render(brief) {
    // Update time
    const genAt = brief.generated_at;
    if (genAt) {
      try {
        const d = new Date(genAt);
        this._timeEl.textContent = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      } catch { this._timeEl.textContent = ''; }
    }

    let html = '';

    // P&L Section
    const pnl = brief.pnl;
    if (pnl) {
      html += `<div class="section">
        <p class="section-title">Portfolio: $${(pnl.total_value_usd || 0).toLocaleString()}</p>
        <div class="pnl-grid">`;

      for (const window of ['24h', '7d', '30d']) {
        const w = pnl[window];
        if (w && w.available) {
          const cls = w.absolute_change >= 0 ? 'positive' : 'negative';
          const sign = w.absolute_change >= 0 ? '+' : '';
          html += `
            <span class="pnl-label">${window}</span>
            <span class="pnl-value ${cls}">${sign}$${Math.abs(w.absolute_change).toLocaleString()}</span>
            <span class="pnl-value ${cls}">${sign}${w.percentage_change.toFixed(1)}%</span>`;
        } else {
          html += `
            <span class="pnl-label">${window}</span>
            <span class="pnl-value neutral">--</span>
            <span class="pnl-value neutral">--</span>`;
        }
      }
      html += `</div></div>`;
    }

    // Decision Index
    const di = brief.decision_index;
    if (di) {
      const score = di.blended_score ?? di.decision_score ?? 0;
      const scoreColor = score > 70 ? 'positive' : score >= 40 ? 'neutral' : 'negative';
      html += `<div class="section">
        <p class="section-title">Decision Index</p>
        <div class="di-row">
          <span class="di-score ${scoreColor}">${Math.round(score)}</span>
          <span class="di-meta">/ 100 &middot; Conf: ${Math.round(di.confidence || 0)}%</span>
        </div>
      </div>`;
    }

    // Alerts
    const alerts = brief.alerts;
    if (alerts && alerts.total > 0) {
      html += `<div class="section">
        <p class="section-title">Alerts (24h)</p>
        <div class="alert-row">`;
      const sev = alerts.by_severity || {};
      if (sev['S3']) html += `<span class="alert-badge badge-s3">${sev['S3']} critical</span>`;
      if (sev['S2']) html += `<span class="alert-badge badge-s2">${sev['S2']} warning</span>`;
      if (sev['S1']) html += `<span class="alert-badge badge-s1">${sev['S1']} info</span>`;
      html += `</div></div>`;
    } else if (alerts) {
      html += `<div class="section">
        <p class="section-title">Alerts (24h)</p>
        <span style="font-size:0.85rem;color:var(--color-success-600,#047857)">No active alerts</span>
      </div>`;
    }

    // Top Movers
    const movers = brief.top_movers;
    if (movers && movers.movers && movers.movers.length > 0) {
      html += `<div class="section">
        <p class="section-title">Top Movers (${movers.period || '24h'})</p>
        <div class="movers-list">`;
      for (const m of movers.movers.slice(0, 3)) {
        if (m.change_pct != null) {
          const cls = m.change_pct >= 0 ? 'positive' : 'negative';
          const sign = m.change_pct >= 0 ? '+' : '';
          html += `<div class="mover-row">
            <span class="mover-symbol">${m.symbol}</span>
            <span class="${cls}">${sign}${m.change_pct.toFixed(1)}%</span>
          </div>`;
        }
      }
      html += `</div></div>`;
    }

    // Warnings
    if (brief.warnings && brief.warnings.length > 0) {
      html += `<div style="font-size:0.7rem;color:var(--theme-text-tertiary,#94a3b8);text-align:center">
        ${brief.warnings.length} section(s) unavailable
      </div>`;
    }

    this._body.innerHTML = html || '<div class="loading">No data available</div>';
  }
}

customElements.define('morning-brief-card', MorningBriefCard);

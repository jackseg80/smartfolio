/**
 * DataFreshness - Web Component showing data freshness indicator
 *
 * Displays a colored dot + relative time ("Updated 3m ago") with auto-refresh.
 * Reads timestamps from fetcher.js cache or custom sources.
 *
 * Usage:
 *   <data-freshness cache-key="scores" auto-refresh="30"></data-freshness>
 *   <data-freshness cache-key="portfolio" refresh-url="/api/portfolio/refresh"></data-freshness>
 *   <data-freshness timestamp="1707580800000"></data-freshness>
 *
 * Attributes:
 *   - cache-key:    fetcher.js cache key to monitor
 *   - timestamp:    manual timestamp (ms) — overrides cache-key
 *   - auto-refresh: seconds between UI updates (default: 30)
 *   - refresh-url:  optional URL to call when user clicks refresh button
 *   - stale-warn:   minutes before yellow warning (default: 5)
 *   - stale-error:  minutes before red error (default: 30)
 *   - compact:      if present, show compact "3m" instead of "Updated 3m ago"
 *
 * @customElement data-freshness
 */

const template = document.createElement('template');
template.innerHTML = `
<style>
  :host {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif;
    font-size: 0.8rem;
    color: var(--theme-text-secondary, #64748b);
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
    transition: background-color 0.3s ease;
  }

  .dot.fresh { background-color: var(--color-success-600, #047857); }
  .dot.warn  { background-color: var(--color-warning-600, #b45309); }
  .dot.stale { background-color: var(--color-danger-600, #b91c1c); }
  .dot.unknown { background-color: var(--theme-text-tertiary, #94a3b8); }

  .label {
    white-space: nowrap;
  }

  .refresh-btn {
    background: none;
    border: none;
    cursor: pointer;
    padding: 2px;
    font-size: 0.75rem;
    color: var(--theme-text-secondary, #64748b);
    opacity: 0.7;
    transition: opacity 0.2s;
    line-height: 1;
  }
  .refresh-btn:hover { opacity: 1; }
  .refresh-btn.spinning { animation: spin 1s linear infinite; }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
<span class="dot unknown"></span>
<span class="label">--</span>
<button class="refresh-btn" style="display:none" title="Refresh data">&#x21bb;</button>
`;

class DataFreshness extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.appendChild(template.content.cloneNode(true));

    this._dot = this.shadowRoot.querySelector('.dot');
    this._label = this.shadowRoot.querySelector('.label');
    this._refreshBtn = this.shadowRoot.querySelector('.refresh-btn');
    this._intervalId = null;
    this._manualTimestamp = null;
  }

  static get observedAttributes() {
    return ['cache-key', 'timestamp', 'auto-refresh', 'refresh-url', 'stale-warn', 'stale-error', 'compact'];
  }

  connectedCallback() {
    this._startAutoRefresh();
    this._update();

    if (this.getAttribute('refresh-url')) {
      this._refreshBtn.style.display = '';
      this._refreshBtn.addEventListener('click', () => this._onRefreshClick());
    }
  }

  disconnectedCallback() {
    this._stopAutoRefresh();
  }

  attributeChangedCallback(name, oldVal, newVal) {
    if (name === 'timestamp' && newVal) {
      this._manualTimestamp = parseInt(newVal, 10);
    }
    if (name === 'auto-refresh') {
      this._stopAutoRefresh();
      this._startAutoRefresh();
    }
    this._update();
  }

  /** Set timestamp programmatically */
  setTimestamp(ms) {
    this._manualTimestamp = ms;
    this._update();
  }

  _getTimestamp() {
    // Manual timestamp takes priority
    if (this._manualTimestamp) return this._manualTimestamp;

    // Try fetcher.js cache
    const cacheKey = this.getAttribute('cache-key');
    if (cacheKey) {
      try {
        // Import dynamically — works with module scripts
        const { getLastUpdateTime } = window.__fetcherExports || {};
        if (getLastUpdateTime) return getLastUpdateTime(cacheKey);

        // Fallback: read localStorage directly
        const raw = localStorage.getItem(`cache:${cacheKey}`);
        if (raw) {
          const parsed = JSON.parse(raw);
          return parsed?.timestamp || null;
        }
      } catch { /* ignore */ }
    }
    return null;
  }

  _update() {
    const ts = this._getTimestamp();
    const staleWarn = parseInt(this.getAttribute('stale-warn') || '5', 10);
    const staleError = parseInt(this.getAttribute('stale-error') || '30', 10);
    const compact = this.hasAttribute('compact');

    if (!ts) {
      this._dot.className = 'dot unknown';
      this._label.textContent = compact ? '--' : 'No data';
      return;
    }

    const ageMs = Date.now() - ts;
    const ageMin = ageMs / 60000;

    // Determine status
    let status, text;
    if (ageMin < staleWarn) {
      status = 'fresh';
      text = this._formatAge(ageMs, compact);
    } else if (ageMin < staleError) {
      status = 'warn';
      text = this._formatAge(ageMs, compact);
    } else {
      status = 'stale';
      text = this._formatAge(ageMs, compact);
    }

    this._dot.className = `dot ${status}`;
    this._label.textContent = text;
  }

  _formatAge(ageMs, compact) {
    const sec = Math.floor(ageMs / 1000);
    if (sec < 60) {
      return compact ? `${sec}s` : `Updated ${sec}s ago`;
    }
    const min = Math.floor(sec / 60);
    if (min < 60) {
      return compact ? `${min}m` : `Updated ${min}m ago`;
    }
    const hours = Math.floor(min / 60);
    if (hours < 24) {
      return compact ? `${hours}h` : `Updated ${hours}h ago`;
    }
    const days = Math.floor(hours / 24);
    return compact ? `${days}d` : `Updated ${days}d ago`;
  }

  _startAutoRefresh() {
    const interval = parseInt(this.getAttribute('auto-refresh') || '30', 10);
    this._intervalId = setInterval(() => this._update(), interval * 1000);
  }

  _stopAutoRefresh() {
    if (this._intervalId) {
      clearInterval(this._intervalId);
      this._intervalId = null;
    }
  }

  async _onRefreshClick() {
    const url = this.getAttribute('refresh-url');
    if (!url) return;

    this._refreshBtn.classList.add('spinning');
    try {
      const user = localStorage.getItem('activeUser') || 'demo';
      const response = await fetch(url, {
        headers: { 'X-User': user },
      });
      if (response.ok) {
        // Update timestamp to now
        this._manualTimestamp = Date.now();
        this._update();
      }
    } catch (err) {
      console.warn('Data refresh failed:', err);
    } finally {
      this._refreshBtn.classList.remove('spinning');
    }
  }
}

customElements.define('data-freshness', DataFreshness);

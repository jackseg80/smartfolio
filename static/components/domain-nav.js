/**
 * DomainNav - Web Component pour navigation contextuelle entre pages liées
 *
 * Affiche une barre de navigation horizontale avec les pages du même domaine.
 * La page actuelle est automatiquement détectée et mise en surbrillance.
 *
 * Usage:
 *   <domain-nav domain="risk"></domain-nav>
 *   <domain-nav domain="bourse"></domain-nav>
 *   <domain-nav domain="custom" items='[{"href":"page1.html","label":"Page 1"}]'></domain-nav>
 *
 * Domaines prédéfinis:
 *   - risk: risk-dashboard, market-regimes, advanced-risk, cycle-analysis
 *   - bourse: saxo-dashboard, bourse-analytics, bourse-recommendations
 *   - analytics: analytics-unified, simulations, di-backtest
 *
 * Attributs:
 *   - domain: "risk" | "bourse" | "analytics" | "custom"
 *   - items: JSON array pour domain="custom"
 *   - variant: "tabs" | "pills" | "breadcrumb" (default: "tabs")
 *
 * @customElement domain-nav
 * @version 1.0.0
 * @since Feb 2026
 */

const DOMAINS = {
    risk: [
        { href: 'risk-dashboard.html', label: 'Overview', icon: '🛡️' },
        { href: 'market-regimes.html', label: 'Market Regimes', icon: '📈' },
        { href: 'advanced-risk.html', label: 'Advanced', icon: '🔬' },
        { href: 'cycle-analysis.html', label: 'Bitcoin Cycle', icon: '🔄' }
    ],
    bourse: [
        { href: 'saxo-dashboard.html', label: 'Dashboard', icon: '📊' },
        { href: 'bourse-analytics.html', label: 'Analytics', icon: '📉' },
        { href: 'bourse-recommendations.html', label: 'Opportunities', icon: '💡' }
    ],
    analytics: [
        { href: 'analytics-unified.html', label: 'ML Intelligence', icon: '🧠' },
        { href: 'simulations.html', label: 'Simulations', icon: '🎮' },
        { href: 'di-backtest.html', label: 'DI Backtest', icon: '📜' }
    ]
};

const template = document.createElement('template');
template.innerHTML = `
<style>
    :host {
        display: block;
        font-family: system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif;
    }

    .domain-nav {
        display: flex;
        gap: 4px;
        padding: 4px;
        background: var(--theme-surface-elevated, #f8fafc);
        border: 1px solid var(--theme-border, #e2e8f0);
        border-radius: var(--radius-lg, 8px);
        margin-bottom: var(--space-lg, 16px);
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }

    /* Hide scrollbar but keep functionality */
    .domain-nav::-webkit-scrollbar {
        height: 4px;
    }

    .domain-nav::-webkit-scrollbar-thumb {
        background: var(--theme-border, #e2e8f0);
        border-radius: 2px;
    }

    .nav-item {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 10px 16px;
        border: none;
        background: transparent;
        color: var(--theme-text-muted, #64748b);
        font-size: 0.875rem;
        font-weight: 500;
        text-decoration: none;
        border-radius: var(--radius-md, 6px);
        cursor: pointer;
        transition: all 0.2s ease;
        white-space: nowrap;
    }

    .nav-item:hover {
        color: var(--theme-text, #1e293b);
        background: var(--theme-surface-hover, rgba(0, 0, 0, 0.04));
    }

    .nav-item.active {
        color: var(--brand-primary, #3b82f6);
        background: var(--theme-surface, #ffffff);
        box-shadow: var(--shadow-sm, 0 1px 2px rgba(0, 0, 0, 0.05));
    }

    .nav-icon {
        font-size: 1rem;
        line-height: 1;
    }

    /* Pills variant */
    :host([variant="pills"]) .domain-nav {
        background: transparent;
        border: none;
        padding: 0;
        gap: 8px;
    }

    :host([variant="pills"]) .nav-item {
        background: var(--theme-surface-elevated, #f8fafc);
        border: 1px solid var(--theme-border, #e2e8f0);
    }

    :host([variant="pills"]) .nav-item.active {
        background: var(--brand-primary, #3b82f6);
        color: white;
        border-color: var(--brand-primary, #3b82f6);
    }

    /* Breadcrumb variant */
    :host([variant="breadcrumb"]) .domain-nav {
        background: transparent;
        border: none;
        padding: 0;
        gap: 0;
    }

    :host([variant="breadcrumb"]) .nav-item {
        padding: 6px 12px;
        border-radius: 0;
    }

    :host([variant="breadcrumb"]) .nav-item:not(:last-child)::after {
        content: "›";
        margin-left: 12px;
        color: var(--theme-text-muted, #64748b);
    }

    :host([variant="breadcrumb"]) .nav-item.active {
        background: transparent;
        box-shadow: none;
        font-weight: 600;
    }

    /* Compact mode */
    :host([compact]) .nav-item {
        padding: 8px 12px;
        font-size: 0.8125rem;
    }

    :host([compact]) .nav-icon {
        display: none;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .nav-item {
            padding: 8px 12px;
            font-size: 0.8125rem;
        }

        .nav-icon {
            display: none;
        }
    }

    /* Dark mode */
    :host-context([data-theme="dark"]) .domain-nav {
        background: var(--theme-surface-elevated, #1e293b);
        border-color: var(--theme-border, #334155);
    }

    :host-context([data-theme="dark"]) .nav-item.active {
        background: var(--theme-surface, #0f172a);
    }
</style>

<nav class="domain-nav" role="navigation" aria-label="Domain navigation">
</nav>
`;

class DomainNav extends HTMLElement {
    static get observedAttributes() {
        return ['domain', 'items', 'variant'];
    }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.appendChild(template.content.cloneNode(true));
        this._nav = this.shadowRoot.querySelector('.domain-nav');
    }

    connectedCallback() {
        this._render();
    }

    attributeChangedCallback() {
        if (this.isConnected) {
            this._render();
        }
    }

    get domain() {
        return this.getAttribute('domain') || 'custom';
    }

    get items() {
        const itemsAttr = this.getAttribute('items');
        if (itemsAttr) {
            try {
                return JSON.parse(itemsAttr);
            } catch {
                console.warn('[DomainNav] Invalid items JSON');
                return [];
            }
        }
        return DOMAINS[this.domain] || [];
    }

    _render() {
        const items = this.items;
        const currentPath = window.location.pathname;
        const currentPage = currentPath.split('/').pop() || 'index.html';

        this._nav.innerHTML = items.map(item => {
            const isActive = currentPage === item.href ||
                             currentPath.endsWith(item.href) ||
                             (item.aliases && item.aliases.includes(currentPage));

            return `
                <a href="${item.href}"
                   class="nav-item ${isActive ? 'active' : ''}"
                   ${isActive ? 'aria-current="page"' : ''}>
                    ${item.icon ? `<span class="nav-icon">${item.icon}</span>` : ''}
                    <span class="nav-label">${item.label}</span>
                </a>
            `;
        }).join('');
    }

    // Public API

    /**
     * Actualise la navigation (utile après navigation SPA)
     */
    refresh() {
        this._render();
    }

    /**
     * Définit les items programmatiquement
     * @param {Array} items - Array d'objets {href, label, icon?}
     */
    setItems(items) {
        this._customItems = items;
        this.setAttribute('domain', 'custom');
        this.setAttribute('items', JSON.stringify(items));
    }
}

if (!customElements.get('domain-nav')) {
    customElements.define('domain-nav', DomainNav);
}

export { DomainNav, DOMAINS };

/**
 * ViewToggle - Web Component pour basculer entre les modes Simple et Pro
 *
 * Usage:
 *   <view-toggle></view-toggle>
 *
 * Le composant se synchronise automatiquement avec ViewModeManager.
 * Style intégré, compatible dark/light mode via CSS variables.
 *
 * @customElement view-toggle
 * @version 1.0.0
 * @since Feb 2026
 */

import { ViewModeManager, ViewModes } from '../core/view-mode-manager.js';

const template = document.createElement('template');
template.innerHTML = `
<style>
    :host {
        display: inline-flex;
        align-items: center;
        font-family: system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif;
    }

    .view-toggle {
        display: inline-flex;
        align-items: center;
        gap: 2px;
        padding: 3px;
        background: var(--theme-surface-elevated, #f1f5f9);
        border: 1px solid var(--theme-border, #e2e8f0);
        border-radius: var(--radius-lg, 8px);
        position: relative;
    }

    .view-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border: none;
        background: transparent;
        color: var(--theme-text-muted, #64748b);
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        border-radius: var(--radius-md, 6px);
        transition: all 0.2s ease;
        position: relative;
        z-index: 1;
        white-space: nowrap;
    }

    .view-btn:hover:not(.active) {
        color: var(--theme-text, #1e293b);
        background: var(--theme-surface-hover, rgba(0, 0, 0, 0.04));
    }

    .view-btn.active {
        color: var(--brand-primary, #3b82f6);
        background: var(--theme-surface, #ffffff);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .view-btn:focus-visible {
        outline: 2px solid var(--brand-primary, #3b82f6);
        outline-offset: 2px;
    }

    .icon {
        font-size: 14px;
        line-height: 1;
    }

    .label {
        display: inline;
    }

    /* Compact mode for narrow spaces */
    :host([compact]) .label {
        display: none;
    }

    :host([compact]) .view-btn {
        padding: 6px 10px;
    }

    /* Dark mode adjustments */
    :host-context([data-theme="dark"]) .view-toggle {
        background: var(--theme-surface-elevated, #1e293b);
        border-color: var(--theme-border, #334155);
    }

    :host-context([data-theme="dark"]) .view-btn.active {
        background: var(--theme-surface, #0f172a);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }

    :host-context([data-theme="dark"]) .view-btn:hover:not(.active) {
        background: var(--theme-surface-hover, rgba(255, 255, 255, 0.05));
    }

    /* Tooltip */
    .view-btn[data-tooltip]::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: calc(100% + 8px);
        left: 50%;
        transform: translateX(-50%);
        padding: 6px 10px;
        background: var(--theme-text, #1e293b);
        color: var(--theme-bg, #ffffff);
        font-size: 12px;
        font-weight: 400;
        border-radius: 6px;
        white-space: nowrap;
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.2s, visibility 0.2s;
        pointer-events: none;
        z-index: 1000;
    }

    .view-btn[data-tooltip]:hover::after {
        opacity: 1;
        visibility: visible;
    }
</style>

<div class="view-toggle" role="radiogroup" aria-label="Mode d'affichage">
    <button class="view-btn" data-mode="simple" role="radio" aria-checked="false"
            data-tooltip="Vue simplifi\u00e9e : m\u00e9triques cl\u00e9s uniquement">
        <span class="icon" aria-hidden="true">\ud83d\udcca</span>
        <span class="label">Simple</span>
    </button>
    <button class="view-btn" data-mode="pro" role="radio" aria-checked="true"
            data-tooltip="Vue pro : toutes les donn\u00e9es et graphiques">
        <span class="icon" aria-hidden="true">\ud83d\udd2c</span>
        <span class="label">Pro</span>
    </button>
</div>
`;

class ViewToggle extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.appendChild(template.content.cloneNode(true));

        this._buttons = this.shadowRoot.querySelectorAll('.view-btn');
        this._unsubscribe = null;
    }

    connectedCallback() {
        // Init ViewModeManager if not already
        ViewModeManager.init();

        // Set initial state
        this._updateUI(ViewModeManager.getMode());

        // Listen for clicks
        this._buttons.forEach(btn => {
            btn.addEventListener('click', this._handleClick.bind(this));
        });

        // Listen for mode changes (from other tabs or programmatic changes)
        this._unsubscribe = ViewModeManager.on('change', (mode) => {
            this._updateUI(mode);
        });

        // Also listen to the global event
        window.addEventListener('viewmode:change', this._handleGlobalChange);
    }

    disconnectedCallback() {
        if (this._unsubscribe) {
            this._unsubscribe();
        }
        window.removeEventListener('viewmode:change', this._handleGlobalChange);
    }

    _handleClick(e) {
        const btn = e.currentTarget;
        const mode = btn.dataset.mode;

        if (mode && mode !== ViewModeManager.getMode()) {
            ViewModeManager.setMode(mode);
        }
    }

    _handleGlobalChange = (e) => {
        this._updateUI(e.detail.mode);
    }

    _updateUI(mode) {
        this._buttons.forEach(btn => {
            const isActive = btn.dataset.mode === mode;
            btn.classList.toggle('active', isActive);
            btn.setAttribute('aria-checked', isActive ? 'true' : 'false');
        });
    }

    // Public API
    get mode() {
        return ViewModeManager.getMode();
    }

    set mode(value) {
        ViewModeManager.setMode(value);
    }

    toggle() {
        return ViewModeManager.toggle();
    }
}

// Register custom element
if (!customElements.get('view-toggle')) {
    customElements.define('view-toggle', ViewToggle);
}

export { ViewToggle };

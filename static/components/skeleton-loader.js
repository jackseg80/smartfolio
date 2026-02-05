/**
 * SkeletonLoader - Web Component pour afficher des loading states animés
 *
 * Usage:
 *   <!-- Ligne de texte -->
 *   <skeleton-loader type="text" width="200px"></skeleton-loader>
 *
 *   <!-- Titre -->
 *   <skeleton-loader type="title"></skeleton-loader>
 *
 *   <!-- Avatar/Cercle -->
 *   <skeleton-loader type="avatar" size="48"></skeleton-loader>
 *
 *   <!-- Rectangle (image, card) -->
 *   <skeleton-loader type="rect" width="100%" height="200px"></skeleton-loader>
 *
 *   <!-- Card complète -->
 *   <skeleton-loader type="card"></skeleton-loader>
 *
 *   <!-- Table rows -->
 *   <skeleton-loader type="table" rows="5"></skeleton-loader>
 *
 * Attributs:
 *   - type: "text" | "title" | "avatar" | "rect" | "card" | "table" | "metric"
 *   - width: Largeur (default: 100%)
 *   - height: Hauteur (pour rect)
 *   - size: Taille (pour avatar, en px)
 *   - rows: Nombre de lignes (pour table)
 *   - animated: "true" | "false" (default: true)
 *
 * @customElement skeleton-loader
 * @version 1.0.0
 * @since Feb 2026
 */

const template = document.createElement('template');
template.innerHTML = `
<style>
    :host {
        display: block;
        font-family: system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif;
    }

    .skeleton {
        background: linear-gradient(
            90deg,
            var(--skeleton-base, #e2e8f0) 0%,
            var(--skeleton-shine, #f1f5f9) 50%,
            var(--skeleton-base, #e2e8f0) 100%
        );
        background-size: 200% 100%;
        border-radius: var(--radius-sm, 4px);
    }

    :host([animated="true"]) .skeleton,
    :host(:not([animated])) .skeleton {
        animation: shimmer 1.5s infinite;
    }

    :host([animated="false"]) .skeleton {
        animation: none;
    }

    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* Text skeleton */
    .skeleton-text {
        height: 16px;
        margin-bottom: 8px;
    }

    .skeleton-text:last-child {
        width: 60%;
    }

    /* Title skeleton */
    .skeleton-title {
        height: 24px;
        width: 40%;
        margin-bottom: 12px;
    }

    /* Avatar skeleton */
    .skeleton-avatar {
        border-radius: 50%;
    }

    /* Rect skeleton */
    .skeleton-rect {
        width: 100%;
    }

    /* Card skeleton */
    .skeleton-card {
        padding: var(--space-lg, 16px);
        border: 1px solid var(--theme-border, #e2e8f0);
        border-radius: var(--radius-card, 8px);
        background: var(--theme-surface, #fff);
    }

    .skeleton-card .skeleton-title {
        margin-bottom: 16px;
    }

    .skeleton-card .skeleton-text {
        margin-bottom: 10px;
    }

    /* Table skeleton */
    .skeleton-table {
        width: 100%;
    }

    .skeleton-table-row {
        display: flex;
        gap: 16px;
        padding: 12px 0;
        border-bottom: 1px solid var(--theme-border, #e2e8f0);
    }

    .skeleton-table-row:last-child {
        border-bottom: none;
    }

    .skeleton-table-cell {
        flex: 1;
        height: 16px;
    }

    .skeleton-table-cell:first-child {
        flex: 0.5;
    }

    .skeleton-table-cell:last-child {
        flex: 0.7;
    }

    /* Metric skeleton */
    .skeleton-metric {
        text-align: center;
        padding: var(--space-md, 12px);
    }

    .skeleton-metric-label {
        height: 12px;
        width: 60%;
        margin: 0 auto 8px;
    }

    .skeleton-metric-value {
        height: 32px;
        width: 80%;
        margin: 0 auto;
    }

    /* Dark mode */
    :host-context([data-theme="dark"]) .skeleton {
        --skeleton-base: #334155;
        --skeleton-shine: #475569;
    }

    :host-context([data-theme="dark"]) .skeleton-card {
        background: var(--theme-surface, #1e293b);
        border-color: var(--theme-border, #334155);
    }
</style>

<div class="skeleton-container"></div>
`;

class SkeletonLoader extends HTMLElement {
    static get observedAttributes() {
        return ['type', 'width', 'height', 'size', 'rows', 'animated'];
    }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.appendChild(template.content.cloneNode(true));
        this._container = this.shadowRoot.querySelector('.skeleton-container');
    }

    connectedCallback() {
        this._render();
    }

    attributeChangedCallback() {
        if (this.isConnected) {
            this._render();
        }
    }

    get type() {
        return this.getAttribute('type') || 'text';
    }

    get width() {
        return this.getAttribute('width') || '100%';
    }

    get height() {
        return this.getAttribute('height') || '100px';
    }

    get size() {
        return parseInt(this.getAttribute('size') || '48', 10);
    }

    get rows() {
        return parseInt(this.getAttribute('rows') || '3', 10);
    }

    _render() {
        switch (this.type) {
            case 'text':
                this._renderText();
                break;
            case 'title':
                this._renderTitle();
                break;
            case 'avatar':
                this._renderAvatar();
                break;
            case 'rect':
                this._renderRect();
                break;
            case 'card':
                this._renderCard();
                break;
            case 'table':
                this._renderTable();
                break;
            case 'metric':
                this._renderMetric();
                break;
            default:
                this._renderText();
        }
    }

    _renderText() {
        this._container.innerHTML = `
            <div class="skeleton skeleton-text" style="width: ${this.width}"></div>
        `;
    }

    _renderTitle() {
        this._container.innerHTML = `
            <div class="skeleton skeleton-title" style="width: ${this.width}"></div>
        `;
    }

    _renderAvatar() {
        this._container.innerHTML = `
            <div class="skeleton skeleton-avatar" style="width: ${this.size}px; height: ${this.size}px;"></div>
        `;
    }

    _renderRect() {
        this._container.innerHTML = `
            <div class="skeleton skeleton-rect" style="width: ${this.width}; height: ${this.height};"></div>
        `;
    }

    _renderCard() {
        this._container.innerHTML = `
            <div class="skeleton-card">
                <div class="skeleton skeleton-title"></div>
                <div class="skeleton skeleton-text"></div>
                <div class="skeleton skeleton-text"></div>
                <div class="skeleton skeleton-text" style="width: 70%;"></div>
            </div>
        `;
    }

    _renderTable() {
        const rows = [];
        for (let i = 0; i < this.rows; i++) {
            rows.push(`
                <div class="skeleton-table-row">
                    <div class="skeleton skeleton-table-cell"></div>
                    <div class="skeleton skeleton-table-cell"></div>
                    <div class="skeleton skeleton-table-cell"></div>
                    <div class="skeleton skeleton-table-cell"></div>
                </div>
            `);
        }
        this._container.innerHTML = `
            <div class="skeleton-table">
                ${rows.join('')}
            </div>
        `;
    }

    _renderMetric() {
        this._container.innerHTML = `
            <div class="skeleton-metric">
                <div class="skeleton skeleton-metric-label"></div>
                <div class="skeleton skeleton-metric-value"></div>
            </div>
        `;
    }
}

if (!customElements.get('skeleton-loader')) {
    customElements.define('skeleton-loader', SkeletonLoader);
}

export { SkeletonLoader };

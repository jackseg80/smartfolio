/**
 * EmptyState - Web Component pour afficher des états vides standardisés
 *
 * Usage:
 *   <empty-state
 *       icon="📭"
 *       title="No data available"
 *       description="Try adjusting your filters or add some data."
 *       action-text="Add Data"
 *       action-href="/settings.html">
 *   </empty-state>
 *
 * Attributs:
 *   - icon: Emoji ou icône (default: "📭")
 *   - title: Titre principal (default: "No data")
 *   - description: Description optionnelle
 *   - action-text: Texte du bouton d'action
 *   - action-href: Lien du bouton
 *   - variant: "default" | "compact" | "inline" (default: "default")
 *
 * @customElement empty-state
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

    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: var(--space-2xl, 32px) var(--space-lg, 16px);
        color: var(--theme-text-muted, #64748b);
    }

    /* Variants */
    :host([variant="compact"]) .empty-state {
        padding: var(--space-lg, 16px) var(--space-md, 12px);
    }

    :host([variant="inline"]) .empty-state {
        flex-direction: row;
        gap: 12px;
        padding: var(--space-md, 12px);
        text-align: left;
    }

    /* Icon */
    .icon {
        font-size: 3rem;
        line-height: 1;
        margin-bottom: var(--space-md, 12px);
        opacity: 0.7;
    }

    :host([variant="compact"]) .icon {
        font-size: 2rem;
        margin-bottom: var(--space-sm, 8px);
    }

    :host([variant="inline"]) .icon {
        font-size: 1.5rem;
        margin-bottom: 0;
    }

    /* Title */
    .title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--theme-text, #1e293b);
        margin-bottom: var(--space-xs, 4px);
    }

    :host([variant="compact"]) .title {
        font-size: 1rem;
    }

    :host([variant="inline"]) .title {
        font-size: 0.9rem;
    }

    /* Description */
    .description {
        font-size: 0.875rem;
        max-width: 300px;
        line-height: 1.5;
    }

    :host([variant="compact"]) .description {
        font-size: 0.8125rem;
        max-width: 250px;
    }

    :host([variant="inline"]) .description {
        font-size: 0.8125rem;
        max-width: none;
    }

    /* Content wrapper for inline */
    .content {
        display: flex;
        flex-direction: column;
    }

    :host([variant="inline"]) .content {
        flex: 1;
    }

    /* Action */
    .action {
        margin-top: var(--space-lg, 16px);
    }

    :host([variant="compact"]) .action {
        margin-top: var(--space-md, 12px);
    }

    :host([variant="inline"]) .action {
        margin-top: var(--space-sm, 8px);
    }

    .action-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        background: var(--brand-primary, #3b82f6);
        color: white;
        border: none;
        border-radius: var(--radius-md, 6px);
        font-size: 0.875rem;
        font-weight: 500;
        text-decoration: none;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .action-btn:hover {
        background: var(--brand-primary-hover, #2563eb);
        transform: translateY(-1px);
    }

    /* Secondary variant */
    :host([action-variant="secondary"]) .action-btn {
        background: var(--theme-surface, #fff);
        color: var(--brand-primary, #3b82f6);
        border: 1px solid var(--brand-primary, #3b82f6);
    }

    :host([action-variant="secondary"]) .action-btn:hover {
        background: var(--brand-primary, #3b82f6);
        color: white;
    }

    /* Illustration slot */
    ::slotted([slot="illustration"]) {
        margin-bottom: var(--space-md, 12px);
    }
</style>

<div class="empty-state">
    <slot name="illustration"></slot>
    <span class="icon"></span>
    <div class="content">
        <div class="title"></div>
        <div class="description"></div>
        <div class="action"></div>
    </div>
</div>
`;

class EmptyState extends HTMLElement {
    static get observedAttributes() {
        return ['icon', 'title', 'description', 'action-text', 'action-href', 'variant'];
    }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.appendChild(template.content.cloneNode(true));

        this._iconEl = this.shadowRoot.querySelector('.icon');
        this._titleEl = this.shadowRoot.querySelector('.title');
        this._descriptionEl = this.shadowRoot.querySelector('.description');
        this._actionEl = this.shadowRoot.querySelector('.action');
    }

    connectedCallback() {
        this._render();
    }

    attributeChangedCallback() {
        if (this.isConnected) {
            this._render();
        }
    }

    _render() {
        // Icon
        const icon = this.getAttribute('icon') || '📭';
        this._iconEl.textContent = icon;

        // Title
        const title = this.getAttribute('title') || 'No data';
        this._titleEl.textContent = title;

        // Description
        const description = this.getAttribute('description');
        if (description) {
            this._descriptionEl.textContent = description;
            this._descriptionEl.style.display = 'block';
        } else {
            this._descriptionEl.style.display = 'none';
        }

        // Action
        const actionText = this.getAttribute('action-text');
        const actionHref = this.getAttribute('action-href');

        if (actionText) {
            if (actionHref) {
                this._actionEl.innerHTML = `<a href="${actionHref}" class="action-btn">${actionText}</a>`;
            } else {
                this._actionEl.innerHTML = `<button class="action-btn">${actionText}</button>`;
                this._actionEl.querySelector('button').addEventListener('click', () => {
                    this.dispatchEvent(new CustomEvent('action', { bubbles: true }));
                });
            }
            this._actionEl.style.display = 'block';
        } else {
            this._actionEl.style.display = 'none';
        }
    }
}

if (!customElements.get('empty-state')) {
    customElements.define('empty-state', EmptyState);
}

export { EmptyState };

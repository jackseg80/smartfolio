/**
 * UIModal - Composant modal accessible et unifié
 *
 * Features:
 * - Full ARIA support (role, aria-modal, aria-labelledby)
 * - Focus trap avec gestion clavier
 * - Escape pour fermer
 * - Backdrop click
 * - Animations fluides
 * - Responsive (full-screen sur mobile)
 * - Theme-aware (suit dark/light mode)
 *
 * Usage:
 * ```javascript
 * import { UIModal } from './components/ui-modal.js';
 *
 * // Simple
 * UIModal.show({
 *   title: 'Export Data',
 *   content: '<p>Choose format:</p><select>...</select>',
 *   onConfirm: () => { console.debug('Confirmed'); }
 * });
 *
 * // Confirmation
 * const confirmed = await UIModal.confirm('Delete?', 'This action cannot be undone.');
 * if (confirmed) { // delete }
 *
 * // Alert
 * await UIModal.alert('Success', 'Data exported successfully.');
 * ```
 */

export class UIModal {
  static instances = [];
  static styleInjected = false;

  constructor(options = {}) {
    this.id = `modal-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    this.title = options.title || '';
    this.content = options.content || '';
    this.size = options.size || 'medium'; // small, medium, large, fullscreen
    this.closable = options.closable !== false;
    this.onClose = options.onClose || null;
    this.onConfirm = options.onConfirm || null;
    this.confirmText = options.confirmText || 'Confirmer';
    this.cancelText = options.cancelText || 'Annuler';
    this.showFooter = options.showFooter !== false;

    this.previousActiveElement = null;
    this.backdrop = null;
    this.modal = null;
    this.focusableElements = [];
    this._escapeHandler = null;
    this._tabHandler = null;

    UIModal.injectStyles();
  }

  static injectStyles() {
    if (UIModal.styleInjected) return;

    const style = document.createElement('style');
    style.textContent = `
      /* UIModal Styles - Using design tokens */
      .ui-modal-backdrop {
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(4px);
        z-index: var(--z-modal-backdrop, 900);
        opacity: 0;
        transition: opacity var(--transition-normal, 200ms);
      }

      .ui-modal-backdrop.visible {
        opacity: 1;
      }

      .ui-modal {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%) scale(0.95);
        background: var(--theme-surface);
        border: 1px solid var(--theme-border);
        border-radius: var(--radius-xl, 12px);
        box-shadow: var(--shadow-xl);
        z-index: var(--z-modal, 1000);
        max-height: 90vh;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        opacity: 0;
        transition: opacity var(--transition-normal),
                    transform var(--transition-normal);
      }

      .ui-modal.visible {
        opacity: 1;
        transform: translate(-50%, -50%) scale(1);
      }

      .ui-modal--small { width: min(400px, 90vw); }
      .ui-modal--medium { width: min(560px, 90vw); }
      .ui-modal--large { width: min(800px, 90vw); }
      .ui-modal--fullscreen {
        width: 95vw;
        height: 90vh;
        max-height: 90vh;
      }

      .ui-modal__header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: var(--space-4, 1rem) var(--space-6, 1.5rem);
        border-bottom: 1px solid var(--theme-border);
        flex-shrink: 0;
      }

      .ui-modal__title {
        margin: 0;
        font-size: var(--font-size-lg, 1.125rem);
        font-weight: var(--font-weight-semibold, 600);
        color: var(--theme-text);
      }

      .ui-modal__close {
        background: transparent;
        border: none;
        padding: var(--space-2, 0.5rem);
        cursor: pointer;
        color: var(--theme-text-muted);
        border-radius: var(--radius-md, 6px);
        transition: background var(--transition-fast),
                    color var(--transition-fast);
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .ui-modal__close:hover {
        background: var(--theme-surface-hover, rgba(0, 0, 0, 0.04));
        color: var(--theme-text);
      }

      .ui-modal__close:focus-visible {
        outline: none;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25);
      }

      .ui-modal__body {
        flex: 1;
        overflow-y: auto;
        padding: var(--space-6, 1.5rem);
      }

      .ui-modal__footer {
        display: flex;
        justify-content: flex-end;
        gap: var(--space-3, 0.75rem);
        padding: var(--space-4, 1rem) var(--space-6, 1.5rem);
        border-top: 1px solid var(--theme-border);
        flex-shrink: 0;
      }

      /* Mobile - Full screen */
      @media (max-width: 640px) {
        .ui-modal {
          width: 100vw !important;
          height: 100vh;
          max-height: 100vh;
          border-radius: 0;
          top: 0;
          left: 0;
          transform: translate(0, 0) scale(0.95);
        }

        .ui-modal.visible {
          transform: translate(0, 0) scale(1);
        }
      }
    `;
    document.head.appendChild(style);
    UIModal.styleInjected = true;
  }

  open() {
    this.previousActiveElement = document.activeElement;

    // Créer backdrop
    this.backdrop = document.createElement('div');
    this.backdrop.className = 'ui-modal-backdrop';
    this.backdrop.setAttribute('aria-hidden', 'true');

    // Créer modal
    this.modal = document.createElement('div');
    this.modal.className = `ui-modal ui-modal--${this.size}`;
    this.modal.setAttribute('role', 'dialog');
    this.modal.setAttribute('aria-modal', 'true');
    this.modal.setAttribute('aria-labelledby', `${this.id}-title`);

    this.modal.innerHTML = `
      <header class="ui-modal__header">
        <h2 class="ui-modal__title" id="${this.id}-title">${this.title}</h2>
        ${this.closable ? `
          <button class="ui-modal__close" aria-label="Fermer" type="button">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
              <path d="M18 6L6 18M6 6l12 12"/>
            </svg>
          </button>
        ` : ''}
      </header>
      <div class="ui-modal__body">
        ${typeof this.content === 'string' ? this.content : ''}
      </div>
      ${this.showFooter ? `
        <footer class="ui-modal__footer">
          ${this.closable ? `<button class="btn btn-secondary ui-modal__cancel" type="button">${this.cancelText}</button>` : ''}
          ${this.onConfirm ? `<button class="btn btn-primary ui-modal__confirm" type="button">${this.confirmText}</button>` : ''}
        </footer>
      ` : ''}
    `;

    // Injecter contenu DOM si nécessaire
    if (typeof this.content !== 'string') {
      const bodyEl = this.modal.querySelector('.ui-modal__body');
      bodyEl.innerHTML = '';
      bodyEl.appendChild(this.content);
    }

    // Ajouter au DOM
    document.body.appendChild(this.backdrop);
    document.body.appendChild(this.modal);
    document.body.style.overflow = 'hidden';

    // Animer l'ouverture
    requestAnimationFrame(() => {
      this.backdrop.classList.add('visible');
      this.modal.classList.add('visible');
    });

    // Event listeners
    this._setupEventListeners();

    // Focus trap
    this._setupFocusTrap();

    UIModal.instances.push(this);
    return this;
  }

  close() {
    if (!this.modal) return;

    this.backdrop.classList.remove('visible');
    this.modal.classList.remove('visible');

    setTimeout(() => {
      this.backdrop?.remove();
      this.modal?.remove();
      document.body.style.overflow = '';

      // Cleanup event listeners
      if (this._escapeHandler) {
        document.removeEventListener('keydown', this._escapeHandler);
      }
      if (this._tabHandler) {
        document.removeEventListener('keydown', this._tabHandler);
      }

      if (this.previousActiveElement) {
        this.previousActiveElement.focus();
      }

      if (this.onClose) this.onClose();

      UIModal.instances = UIModal.instances.filter(m => m !== this);
    }, 200); // Match transition duration
  }

  _setupEventListeners() {
    // Close button
    const closeBtn = this.modal.querySelector('.ui-modal__close');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => this.close());
    }

    // Cancel button
    const cancelBtn = this.modal.querySelector('.ui-modal__cancel');
    if (cancelBtn) {
      cancelBtn.addEventListener('click', () => this.close());
    }

    // Confirm button
    const confirmBtn = this.modal.querySelector('.ui-modal__confirm');
    if (confirmBtn && this.onConfirm) {
      confirmBtn.addEventListener('click', () => {
        this.onConfirm();
        this.close();
      });
    }

    // Backdrop click
    this.backdrop.addEventListener('click', () => {
      if (this.closable) this.close();
    });

    // Escape key
    this._escapeHandler = (e) => {
      if (e.key === 'Escape' && this.closable) {
        // Only close if this is the topmost modal
        const topModal = UIModal.instances[UIModal.instances.length - 1];
        if (topModal === this) {
          this.close();
        }
      }
    };
    document.addEventListener('keydown', this._escapeHandler);
  }

  _setupFocusTrap() {
    this.focusableElements = this.modal.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    if (this.focusableElements.length > 0) {
      this.focusableElements[0].focus();
    }

    this._tabHandler = (e) => {
      if (e.key !== 'Tab') return;

      // Only trap if this is the topmost modal
      const topModal = UIModal.instances[UIModal.instances.length - 1];
      if (topModal !== this) return;

      const first = this.focusableElements[0];
      const last = this.focusableElements[this.focusableElements.length - 1];

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', this._tabHandler);
  }

  // ═══════════════════════════════════════════════════════════
  // API STATIQUE pour usage simplifié
  // ═══════════════════════════════════════════════════════════

  /**
   * Afficher un modal
   * @param {Object} options - Configuration du modal
   * @returns {UIModal} Instance du modal
   */
  static show(options) {
    return new UIModal(options).open();
  }

  /**
   * Modal de confirmation avec promesse
   * @param {string} title - Titre
   * @param {string} message - Message
   * @returns {Promise<boolean>} true si confirmé, false si annulé
   */
  static confirm(title, message) {
    return new Promise((resolve) => {
      UIModal.show({
        title,
        content: `<p style="margin: 0;">${message}</p>`,
        size: 'small',
        onConfirm: () => resolve(true),
        onClose: () => resolve(false),
        confirmText: 'Confirmer',
        cancelText: 'Annuler'
      });
    });
  }

  /**
   * Modal d'alerte
   * @param {string} title - Titre
   * @param {string} message - Message
   * @returns {Promise<void>}
   */
  static alert(title, message) {
    return new Promise((resolve) => {
      UIModal.show({
        title,
        content: `<p style="margin: 0;">${message}</p>`,
        size: 'small',
        showFooter: true,
        confirmText: 'OK',
        onConfirm: () => resolve(),
        onClose: () => resolve(),
        closable: true
      });
    });
  }

  /**
   * Fermer tous les modals ouverts
   */
  static closeAll() {
    [...UIModal.instances].forEach(modal => modal.close());
  }
}

// Export global pour usage sans import
if (typeof window !== 'undefined') {
  window.UIModal = UIModal;
}

/**
 * Toast - Système de notifications unifié
 *
 * Features:
 * - 5 types: success, error, warning, info, loading
 * - Auto-dismiss configurable
 * - Animations fluides
 * - Responsive (mobile-friendly)
 * - Theme-aware
 * - ARIA live regions pour accessibilité
 * - Empilable (jusqu'à 5 toasts simultanés)
 *
 * Usage:
 * ```javascript
 * import { Toast } from './components/toast.js';
 *
 * // Simple
 * Toast.success('Données exportées avec succès');
 * Toast.error('Erreur de connexion');
 * Toast.warning('Attention: données non sauvegardées');
 * Toast.info('Nouvelle version disponible');
 *
 * // Loading avec dismiss manuel
 * const dismiss = Toast.loading('Traitement en cours...');
 * await someAsyncOperation();
 * dismiss();
 * Toast.success('Terminé!');
 *
 * // Options avancées
 * Toast.success('Saved!', { duration: 3000, title: 'Success' });
 * ```
 */

export class Toast {
  static container = null;
  static styleInjected = false;
  static maxToasts = 5;

  static injectStyles() {
    if (Toast.styleInjected) return;

    const style = document.createElement('style');
    style.textContent = `
      /* Toast Container */
      .toast-container {
        position: fixed;
        bottom: var(--space-6, 1.5rem);
        right: var(--space-6, 1.5rem);
        z-index: var(--z-toast, 1300);
        display: flex;
        flex-direction: column;
        gap: var(--space-3, 0.75rem);
        pointer-events: none;
        max-width: 450px;
      }

      /* Individual Toast */
      .toast {
        display: flex;
        align-items: flex-start;
        gap: var(--space-3, 0.75rem);
        padding: var(--space-4, 1rem);
        background: var(--theme-surface);
        border: 1px solid var(--theme-border);
        border-radius: var(--radius-lg, 0.5rem);
        box-shadow: var(--shadow-lg);
        min-width: 300px;
        max-width: 450px;
        pointer-events: auto;
        opacity: 0;
        transform: translateX(calc(100% + var(--space-6, 1.5rem)));
        transition: opacity var(--transition-normal),
                    transform var(--transition-normal);
      }

      .toast.visible {
        opacity: 1;
        transform: translateX(0);
      }

      .toast.removing {
        opacity: 0;
        transform: translateX(calc(100% + var(--space-6, 1.5rem)));
      }

      /* Toast Icon */
      .toast__icon {
        flex-shrink: 0;
        width: 20px;
        height: 20px;
        margin-top: 2px;
      }

      /* Toast Content */
      .toast__content {
        flex: 1;
        min-width: 0;
      }

      .toast__title {
        font-weight: var(--font-weight-semibold, 600);
        font-size: var(--font-size-sm, 0.875rem);
        margin-bottom: var(--space-1, 0.25rem);
        color: var(--theme-text);
      }

      .toast__message {
        color: var(--theme-text-muted);
        font-size: var(--font-size-sm, 0.875rem);
        line-height: var(--line-height-normal, 1.5);
        word-wrap: break-word;
      }

      /* Close Button */
      .toast__close {
        flex-shrink: 0;
        background: none;
        border: none;
        padding: var(--space-1, 0.25rem);
        cursor: pointer;
        color: var(--theme-text-muted);
        border-radius: var(--radius-sm, 0.25rem);
        transition: background var(--transition-fast),
                    color var(--transition-fast);
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .toast__close:hover {
        background: var(--theme-surface-hover, rgba(0, 0, 0, 0.04));
        color: var(--theme-text);
      }

      /* Toast Variants */
      .toast--success {
        border-left: 4px solid var(--success, #059669);
      }
      .toast--success .toast__icon {
        color: var(--success, #059669);
      }

      .toast--error {
        border-left: 4px solid var(--danger, #dc2626);
      }
      .toast--error .toast__icon {
        color: var(--danger, #dc2626);
      }

      .toast--warning {
        border-left: 4px solid var(--warning, #d97706);
      }
      .toast--warning .toast__icon {
        color: var(--warning, #d97706);
      }

      .toast--info {
        border-left: 4px solid var(--info, #2563eb);
      }
      .toast--info .toast__icon {
        color: var(--info, #2563eb);
      }

      .toast--loading .toast__icon {
        color: var(--brand-primary, #3b82f6);
        animation: toast-spin 1s linear infinite;
      }

      /* Loading Animation */
      @keyframes toast-spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }

      /* Mobile Responsive */
      @media (max-width: 640px) {
        .toast-container {
          left: var(--space-4, 1rem);
          right: var(--space-4, 1rem);
          bottom: var(--space-4, 1rem);
          max-width: none;
        }
        .toast {
          min-width: 0;
          max-width: none;
        }
      }
    `;
    document.head.appendChild(style);
    Toast.styleInjected = true;
  }

  static getContainer() {
    if (!Toast.container) {
      Toast.injectStyles();
      Toast.container = document.createElement('div');
      Toast.container.id = 'toast-container'; // Backward compatibility with legacy code
      Toast.container.className = 'toast-container';
      Toast.container.setAttribute('aria-live', 'polite');
      Toast.container.setAttribute('aria-label', 'Notifications');
      Toast.container.setAttribute('role', 'region');
      document.body.appendChild(Toast.container);
    }
    return Toast.container;
  }

  static icons = {
    success: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
      <polyline points="22 4 12 14.01 9 11.01"/>
    </svg>`,

    error: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <line x1="15" y1="9" x2="9" y2="15"/>
      <line x1="9" y1="9" x2="15" y2="15"/>
    </svg>`,

    warning: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/>
      <line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>`,

    info: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="16" x2="12" y2="12"/>
      <line x1="12" y1="8" x2="12.01" y2="8"/>
    </svg>`,

    loading: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
    </svg>`
  };

  static defaultDurations = {
    success: 5000,
    error: 8000,
    warning: 6000,
    info: 5000,
    loading: 0 // No auto-dismiss
  };

  /**
   * Afficher un toast
   * @param {string} type - Type de toast (success, error, warning, info, loading)
   * @param {string} message - Message principal
   * @param {Object} options - Options { title, duration }
   * @returns {Function} Fonction pour dismiss manuel
   */
  static show(type, message, options = {}) {
    const container = Toast.getContainer();

    // Limit toast count
    const existingToasts = container.querySelectorAll('.toast');
    if (existingToasts.length >= Toast.maxToasts) {
      // Remove oldest toast
      existingToasts[0]?.remove();
    }

    const title = options.title || null;
    const duration = options.duration !== undefined
      ? options.duration
      : Toast.defaultDurations[type];

    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.setAttribute('role', 'status');
    toast.setAttribute('aria-live', type === 'error' ? 'assertive' : 'polite');

    toast.innerHTML = `
      <div class="toast__icon">${Toast.icons[type]}</div>
      <div class="toast__content">
        ${title ? `<div class="toast__title">${title}</div>` : ''}
        ${message ? `<div class="toast__message">${message}</div>` : ''}
      </div>
      <button class="toast__close" aria-label="Fermer la notification" type="button">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
          <path d="M18 6L6 18M6 6l12 12"/>
        </svg>
      </button>
    `;

    container.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => {
      toast.classList.add('visible');
    });

    const dismiss = () => {
      if (!toast.parentElement) return; // Already removed

      toast.classList.add('removing');
      setTimeout(() => {
        toast.remove();
      }, 200); // Match transition duration
    };

    // Close button
    toast.querySelector('.toast__close').addEventListener('click', dismiss);

    // Auto-dismiss
    if (duration > 0) {
      setTimeout(dismiss, duration);
    }

    return dismiss;
  }

  /**
   * Toast de succès
   * @param {string} message - Message
   * @param {Object} options - Options { title, duration }
   * @returns {Function} Fonction dismiss
   */
  static success(message, options = {}) {
    return Toast.show('success', message, {
      title: options.title || 'Succès',
      ...options
    });
  }

  /**
   * Toast d'erreur
   * @param {string} message - Message
   * @param {Object} options - Options { title, duration }
   * @returns {Function} Fonction dismiss
   */
  static error(message, options = {}) {
    return Toast.show('error', message, {
      title: options.title || 'Erreur',
      ...options
    });
  }

  /**
   * Toast d'avertissement
   * @param {string} message - Message
   * @param {Object} options - Options { title, duration }
   * @returns {Function} Fonction dismiss
   */
  static warning(message, options = {}) {
    return Toast.show('warning', message, {
      title: options.title || 'Attention',
      ...options
    });
  }

  /**
   * Toast d'information
   * @param {string} message - Message
   * @param {Object} options - Options { title, duration }
   * @returns {Function} Fonction dismiss
   */
  static info(message, options = {}) {
    return Toast.show('info', message, {
      title: options.title || 'Info',
      ...options
    });
  }

  /**
   * Toast de chargement (pas d'auto-dismiss)
   * @param {string} message - Message
   * @param {Object} options - Options { title }
   * @returns {Function} Fonction dismiss
   */
  static loading(message, options = {}) {
    return Toast.show('loading', message, {
      title: options.title || 'Chargement',
      duration: 0, // No auto-dismiss
      ...options
    });
  }

  /**
   * Fermer tous les toasts
   */
  static closeAll() {
    const container = Toast.container;
    if (!container) return;

    const toasts = container.querySelectorAll('.toast');
    toasts.forEach(toast => {
      toast.classList.add('removing');
      setTimeout(() => toast.remove(), 200);
    });
  }
}

// Export global pour usage sans import
if (typeof window !== 'undefined') {
  window.Toast = Toast;
}

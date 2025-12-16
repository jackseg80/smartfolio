# Plan d'Implémentation UI - SmartFolio

> Guide pratique pour uniformiser l'interface
> Priorités: P0 (critique) → P3 (nice-to-have)

---

## P0 - Corrections Critiques (1-2 jours)

### 1. Créer fichier de tokens CSS

**Fichier**: `static/css/tokens.css`

```css
/**
 * SmartFolio Design Tokens
 * Source unique de vérité pour toutes les valeurs de design
 */

:root {
  /* ══════════════════════════════════════════════════════════
     COULEURS - Palette complète avec variations
     ══════════════════════════════════════════════════════════ */

  /* Primary (Bleu) */
  --color-primary-50: #eff6ff;
  --color-primary-100: #dbeafe;
  --color-primary-200: #bfdbfe;
  --color-primary-300: #93c5fd;
  --color-primary-400: #60a5fa;
  --color-primary-500: #3b82f6;  /* Valeur principale */
  --color-primary-600: #2563eb;
  --color-primary-700: #1d4ed8;

  /* Success (Vert) */
  --color-success-50: #ecfdf5;
  --color-success-100: #d1fae5;
  --color-success-500: #059669;  /* Valeur principale */
  --color-success-600: #047857;
  --color-success-light: #8bd17c;  /* Pour indicateurs */

  /* Warning (Ambre) */
  --color-warning-50: #fffbeb;
  --color-warning-100: #fef3c7;
  --color-warning-500: #d97706;  /* Valeur principale */
  --color-warning-600: #b45309;
  --color-warning-light: #f0c96b;

  /* Danger (Rouge) */
  --color-danger-50: #fef2f2;
  --color-danger-100: #fee2e2;
  --color-danger-500: #dc2626;  /* Valeur principale */
  --color-danger-600: #b91c1c;
  --color-danger-light: #ff9aa4;

  /* Info (Bleu clair) */
  --color-info-50: #eff6ff;
  --color-info-100: #dbeafe;
  --color-info-500: #2563eb;
  --color-info-light: #9bbcff;

  /* ══════════════════════════════════════════════════════════
     OPACITÉS STANDARDISÉES
     ══════════════════════════════════════════════════════════ */
  --opacity-subtle: 0.05;
  --opacity-light: 0.1;
  --opacity-medium: 0.2;
  --opacity-strong: 0.3;
  --opacity-heavy: 0.5;

  /* ══════════════════════════════════════════════════════════
     Z-INDEX SCALE
     ══════════════════════════════════════════════════════════ */
  --z-base: 0;
  --z-dropdown: 100;
  --z-sticky: 200;
  --z-fixed: 300;
  --z-modal-backdrop: 900;
  --z-modal: 1000;
  --z-popover: 1100;
  --z-tooltip: 1200;
  --z-toast: 1300;

  /* ══════════════════════════════════════════════════════════
     TYPOGRAPHIE
     ══════════════════════════════════════════════════════════ */
  --font-family-base: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  --font-family-mono: 'Monaco', 'Consolas', 'Courier New', monospace;

  --font-size-xs: 0.75rem;    /* 12px */
  --font-size-sm: 0.875rem;   /* 14px */
  --font-size-base: 1rem;     /* 16px */
  --font-size-lg: 1.125rem;   /* 18px */
  --font-size-xl: 1.25rem;    /* 20px */
  --font-size-2xl: 1.5rem;    /* 24px */
  --font-size-3xl: 1.875rem;  /* 30px */

  --line-height-tight: 1.25;
  --line-height-normal: 1.5;
  --line-height-relaxed: 1.75;

  /* ══════════════════════════════════════════════════════════
     ESPACEMENT
     ══════════════════════════════════════════════════════════ */
  --space-0: 0;
  --space-1: 0.25rem;   /* 4px */
  --space-2: 0.5rem;    /* 8px */
  --space-3: 0.75rem;   /* 12px */
  --space-4: 1rem;      /* 16px */
  --space-5: 1.25rem;   /* 20px */
  --space-6: 1.5rem;    /* 24px */
  --space-8: 2rem;      /* 32px */
  --space-10: 2.5rem;   /* 40px */
  --space-12: 3rem;     /* 48px */

  /* ══════════════════════════════════════════════════════════
     BORDURES & RADIUS
     ══════════════════════════════════════════════════════════ */
  --radius-sm: 0.25rem;   /* 4px */
  --radius-md: 0.375rem;  /* 6px */
  --radius-lg: 0.5rem;    /* 8px */
  --radius-xl: 0.75rem;   /* 12px */
  --radius-2xl: 1rem;     /* 16px */
  --radius-full: 9999px;

  /* ══════════════════════════════════════════════════════════
     TRANSITIONS
     ══════════════════════════════════════════════════════════ */
  --duration-instant: 50ms;
  --duration-fast: 150ms;
  --duration-normal: 200ms;
  --duration-slow: 300ms;
  --duration-slower: 500ms;

  --ease-default: ease;
  --ease-in: ease-in;
  --ease-out: ease-out;
  --ease-in-out: ease-in-out;
  --ease-smooth: cubic-bezier(0.4, 0, 0.2, 1);
  --ease-bounce: cubic-bezier(0.68, -0.55, 0.265, 1.55);

  /* ══════════════════════════════════════════════════════════
     BREAKPOINTS (pour référence JS)
     ══════════════════════════════════════════════════════════ */
  --breakpoint-sm: 640px;
  --breakpoint-md: 768px;
  --breakpoint-lg: 1024px;
  --breakpoint-xl: 1280px;
  --breakpoint-2xl: 1536px;
}
```

**Action**: Ajouter `<link rel="stylesheet" href="css/tokens.css">` AVANT shared-theme.css dans toutes les pages.

---

### 2. Corriger les couleurs hardcodées des tooltips

**Fichier**: `static/css/risk-dashboard.css`

**Avant** (lignes ~1124-1150):
```css
.tooltip {
  background: #0e1528;
  color: #e9f0ff;
  border: 1px solid #243355;
}
```

**Après**:
```css
.tooltip {
  background: var(--theme-surface-elevated);
  color: var(--theme-text);
  border: 1px solid var(--theme-border);
  box-shadow: var(--shadow-lg);
}

[data-theme="dark"] .tooltip {
  background: var(--theme-surface);
  border-color: var(--theme-border);
}
```

---

### 3. Ajouter variable manquante dans shared-theme.css

**Fichier**: `static/css/shared-theme.css`

**Ajouter** dans `:root`:
```css
/* Surface hover - manquant */
--theme-surface-hover: rgba(0, 0, 0, 0.04);

/* Variants légers standardisés */
--success-light: var(--color-success-light, #8bd17c);
--warning-light: var(--color-warning-light, #f0c96b);
--danger-light: var(--color-danger-light, #ff9aa4);
--info-light: var(--color-info-light, #9bbcff);
```

**Ajouter** dans `[data-theme="dark"]`:
```css
--theme-surface-hover: rgba(255, 255, 255, 0.06);
```

---

## P1 - Refactoring Structurel (3-5 jours)

### 4. Extraire CSS de saxo-dashboard.html

**Créer**: `static/css/saxo-dashboard.css`

```bash
# Étapes:
1. Copier tout le contenu <style>...</style> de saxo-dashboard.html
2. Coller dans static/css/saxo-dashboard.css
3. Remplacer couleurs hardcodées par variables CSS
4. Supprimer <style> de saxo-dashboard.html
5. Ajouter: <link rel="stylesheet" href="css/saxo-dashboard.css">
```

**Vérifications après extraction**:
- [ ] Dark mode fonctionne
- [ ] Responsive fonctionne
- [ ] Graphiques affichés correctement

---

### 5. Créer composant Modal unifié

**Fichier**: `static/components/ui-modal.js`

```javascript
/**
 * UIModal - Composant modal accessible et unifié
 * Usage: UIModal.show({ title: '...', content: '...' })
 */

export class UIModal {
  static instances = [];
  static styleInjected = false;

  constructor(options = {}) {
    this.id = `modal-${Date.now()}`;
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

    UIModal.injectStyles();
  }

  static injectStyles() {
    if (UIModal.styleInjected) return;

    const style = document.createElement('style');
    style.textContent = `
      .ui-modal-backdrop {
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(4px);
        z-index: var(--z-modal-backdrop, 900);
        opacity: 0;
        transition: opacity var(--duration-normal, 200ms) var(--ease-smooth);
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
        transition: opacity var(--duration-normal) var(--ease-smooth),
                    transform var(--duration-normal) var(--ease-smooth);
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
        padding: var(--space-4) var(--space-6);
        border-bottom: 1px solid var(--theme-border);
      }

      .ui-modal__title {
        margin: 0;
        font-size: var(--font-size-lg);
        font-weight: 600;
        color: var(--theme-text);
      }

      .ui-modal__close {
        background: transparent;
        border: none;
        padding: var(--space-2);
        cursor: pointer;
        color: var(--theme-text-muted);
        border-radius: var(--radius-md);
        transition: background var(--duration-fast), color var(--duration-fast);
      }

      .ui-modal__close:hover {
        background: var(--theme-surface-hover);
        color: var(--theme-text);
      }

      .ui-modal__body {
        flex: 1;
        overflow-y: auto;
        padding: var(--space-6);
      }

      .ui-modal__footer {
        display: flex;
        justify-content: flex-end;
        gap: var(--space-3);
        padding: var(--space-4) var(--space-6);
        border-top: 1px solid var(--theme-border);
      }

      @media (max-width: 640px) {
        .ui-modal {
          width: 100vw !important;
          height: 100vh;
          max-height: 100vh;
          border-radius: 0;
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
          <button class="ui-modal__close" aria-label="Fermer">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
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
          ${this.closable ? `<button class="btn btn-secondary ui-modal__cancel">${this.cancelText}</button>` : ''}
          ${this.onConfirm ? `<button class="btn btn-primary ui-modal__confirm">${this.confirmText}</button>` : ''}
        </footer>
      ` : ''}
    `;

    // Injecter contenu DOM si nécessaire
    if (typeof this.content !== 'string') {
      this.modal.querySelector('.ui-modal__body').appendChild(this.content);
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

      if (this.previousActiveElement) {
        this.previousActiveElement.focus();
      }

      if (this.onClose) this.onClose();

      UIModal.instances = UIModal.instances.filter(m => m !== this);
    }, 200);
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
        this.close();
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

  // API statique pour usage simplifié
  static show(options) {
    return new UIModal(options).open();
  }

  static confirm(title, message) {
    return new Promise((resolve) => {
      UIModal.show({
        title,
        content: `<p>${message}</p>`,
        size: 'small',
        onConfirm: () => resolve(true),
        onClose: () => resolve(false)
      });
    });
  }

  static alert(title, message) {
    return new Promise((resolve) => {
      UIModal.show({
        title,
        content: `<p>${message}</p>`,
        size: 'small',
        showFooter: true,
        confirmText: 'OK',
        onConfirm: () => resolve(),
        closable: true
      });
    });
  }
}

// Export global pour usage sans import
window.UIModal = UIModal;
```

**Usage**:
```javascript
// Simple
UIModal.show({
  title: 'Export Data',
  content: '<p>Choose format:</p><select>...</select>',
  onConfirm: () => { /* export logic */ }
});

// Confirmation
const confirmed = await UIModal.confirm('Supprimer?', 'Cette action est irréversible.');
if (confirmed) { /* delete */ }
```

---

### 6. Créer système de notifications unifié

**Fichier**: `static/components/toast.js`

```javascript
/**
 * Toast - Système de notifications unifié
 */

export class Toast {
  static container = null;
  static styleInjected = false;

  static injectStyles() {
    if (Toast.styleInjected) return;

    const style = document.createElement('style');
    style.textContent = `
      .toast-container {
        position: fixed;
        bottom: var(--space-6);
        right: var(--space-6);
        z-index: var(--z-toast, 1300);
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        pointer-events: none;
      }

      .toast {
        display: flex;
        align-items: flex-start;
        gap: var(--space-3);
        padding: var(--space-4);
        background: var(--theme-surface);
        border: 1px solid var(--theme-border);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-lg);
        min-width: 300px;
        max-width: 450px;
        pointer-events: auto;
        opacity: 0;
        transform: translateX(100%);
        transition: opacity var(--duration-normal) var(--ease-smooth),
                    transform var(--duration-normal) var(--ease-smooth);
      }

      .toast.visible {
        opacity: 1;
        transform: translateX(0);
      }

      .toast.removing {
        opacity: 0;
        transform: translateX(100%);
      }

      .toast__icon {
        flex-shrink: 0;
        width: 20px;
        height: 20px;
      }

      .toast__content {
        flex: 1;
      }

      .toast__title {
        font-weight: 600;
        margin-bottom: var(--space-1);
      }

      .toast__message {
        color: var(--theme-text-muted);
        font-size: var(--font-size-sm);
      }

      .toast__close {
        flex-shrink: 0;
        background: none;
        border: none;
        padding: var(--space-1);
        cursor: pointer;
        color: var(--theme-text-muted);
        border-radius: var(--radius-sm);
      }

      .toast__close:hover {
        background: var(--theme-surface-hover);
      }

      .toast--success { border-left: 4px solid var(--success); }
      .toast--success .toast__icon { color: var(--success); }

      .toast--error { border-left: 4px solid var(--danger); }
      .toast--error .toast__icon { color: var(--danger); }

      .toast--warning { border-left: 4px solid var(--warning); }
      .toast--warning .toast__icon { color: var(--warning); }

      .toast--info { border-left: 4px solid var(--info); }
      .toast--info .toast__icon { color: var(--info); }

      .toast--loading .toast__icon {
        animation: toast-spin 1s linear infinite;
      }

      @keyframes toast-spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }

      @media (max-width: 640px) {
        .toast-container {
          left: var(--space-4);
          right: var(--space-4);
          bottom: var(--space-4);
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
      Toast.container.className = 'toast-container';
      Toast.container.setAttribute('aria-live', 'polite');
      Toast.container.setAttribute('aria-label', 'Notifications');
      document.body.appendChild(Toast.container);
    }
    return Toast.container;
  }

  static icons = {
    success: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
      <polyline points="22 4 12 14.01 9 11.01"/>
    </svg>`,
    error: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="10"/>
      <line x1="15" y1="9" x2="9" y2="15"/>
      <line x1="9" y1="9" x2="15" y2="15"/>
    </svg>`,
    warning: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/>
      <line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>`,
    info: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="16" x2="12" y2="12"/>
      <line x1="12" y1="8" x2="12.01" y2="8"/>
    </svg>`,
    loading: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
    </svg>`
  };

  static show(type, title, message, duration = 5000) {
    const container = Toast.getContainer();

    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.innerHTML = `
      <div class="toast__icon">${Toast.icons[type]}</div>
      <div class="toast__content">
        ${title ? `<div class="toast__title">${title}</div>` : ''}
        ${message ? `<div class="toast__message">${message}</div>` : ''}
      </div>
      <button class="toast__close" aria-label="Fermer">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
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
      toast.classList.add('removing');
      setTimeout(() => toast.remove(), 200);
    };

    toast.querySelector('.toast__close').addEventListener('click', dismiss);

    if (duration > 0) {
      setTimeout(dismiss, duration);
    }

    return dismiss;
  }

  static success(message, title = 'Succès') {
    return Toast.show('success', title, message, 5000);
  }

  static error(message, title = 'Erreur') {
    return Toast.show('error', title, message, 8000);
  }

  static warning(message, title = 'Attention') {
    return Toast.show('warning', title, message, 6000);
  }

  static info(message, title = 'Info') {
    return Toast.show('info', title, message, 5000);
  }

  static loading(message, title = 'Chargement') {
    return Toast.show('loading', title, message, 0); // No auto-dismiss
  }
}

// Export global
window.Toast = Toast;
```

**Usage**:
```javascript
Toast.success('Données exportées avec succès');
Toast.error('Erreur de connexion', 'Impossible de joindre le serveur');
Toast.warning('Attention: données non sauvegardées');

// Loading avec dismiss manuel
const dismiss = Toast.loading('Traitement en cours...');
await someAsyncOperation();
dismiss();
Toast.success('Terminé!');
```

---

## P2 - Consolidation (1 semaine)

### 7. Unifier les styles de boutons

**Fichier**: `static/css/shared-theme.css`

**Remplacer/Consolider**:
```css
/* ══════════════════════════════════════════════════════════
   BOUTONS - Styles unifiés
   ══════════════════════════════════════════════════════════ */

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  padding: var(--space-3) var(--space-4);
  font-family: inherit;
  font-size: var(--font-size-sm);
  font-weight: 500;
  line-height: 1;
  text-decoration: none;
  border: 1px solid transparent;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-smooth);
  white-space: nowrap;
}

.btn:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25);
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Tailles */
.btn-xs {
  padding: var(--space-1) var(--space-2);
  font-size: var(--font-size-xs);
}

.btn-sm {
  padding: var(--space-2) var(--space-3);
  font-size: var(--font-size-xs);
}

.btn-lg {
  padding: var(--space-4) var(--space-6);
  font-size: var(--font-size-base);
}

/* Variantes */
.btn-primary {
  background: var(--brand-primary);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: var(--brand-primary-hover);
}

.btn-secondary {
  background: var(--theme-surface-elevated);
  color: var(--theme-text);
  border-color: var(--theme-border);
}

.btn-secondary:hover:not(:disabled) {
  background: var(--theme-surface-hover);
}

.btn-ghost {
  background: transparent;
  color: var(--theme-text);
}

.btn-ghost:hover:not(:disabled) {
  background: var(--theme-surface-hover);
}

.btn-danger {
  background: var(--danger);
  color: white;
}

.btn-danger:hover:not(:disabled) {
  background: var(--danger-hover, #b91c1c);
}
```

**Puis supprimer** les duplications dans `rebalance.css`, `risk-dashboard.css`, etc.

---

### 8. Standardiser les animations

**Fichier**: `static/css/shared-theme.css`

**Ajouter une seule définition**:
```css
/* ══════════════════════════════════════════════════════════
   ANIMATIONS - Définitions uniques
   ══════════════════════════════════════════════════════════ */

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

@keyframes pulse-subtle {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.8; transform: scale(0.98); }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideUp {
  from { transform: translateY(10px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

@keyframes slideDown {
  from { transform: translateY(-10px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

/* Classes utilitaires */
.animate-spin { animation: spin 1s linear infinite; }
.animate-pulse { animation: pulse 2s ease-in-out infinite; }
.animate-fade-in { animation: fadeIn var(--duration-normal) var(--ease-smooth); }
.animate-slide-up { animation: slideUp var(--duration-normal) var(--ease-smooth); }
```

**Supprimer** les duplications de `@keyframes spin` dans:
- `ai-components.css`
- `analytics-unified-theme.css`
- `shared-ml-styles.css`

---

### 9. Créer abstraction Chart unifiée

**Fichier**: `static/core/chart-config.js`

```javascript
/**
 * Configuration Chart.js unifiée
 */

export const chartColors = {
  primary: getComputedStyle(document.documentElement)
    .getPropertyValue('--brand-primary').trim() || '#3b82f6',
  accent: getComputedStyle(document.documentElement)
    .getPropertyValue('--brand-accent').trim() || '#2dd4bf',
  success: '#059669',
  warning: '#d97706',
  danger: '#dc2626',
  text: 'var(--theme-text)',
  textMuted: 'var(--theme-text-muted)',
  border: 'var(--theme-border)',
  surface: 'var(--theme-surface)'
};

export const chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,

  plugins: {
    legend: {
      labels: {
        color: chartColors.text,
        font: { size: 12 }
      }
    },
    tooltip: {
      backgroundColor: 'var(--theme-surface-elevated)',
      titleColor: chartColors.text,
      bodyColor: chartColors.textMuted,
      borderColor: chartColors.border,
      borderWidth: 1,
      cornerRadius: 8,
      padding: 12,
      displayColors: true,
      boxPadding: 4
    }
  },

  scales: {
    x: {
      grid: { color: 'var(--theme-border-subtle)' },
      ticks: { color: chartColors.textMuted }
    },
    y: {
      grid: { color: 'var(--theme-border-subtle)' },
      ticks: { color: chartColors.textMuted }
    }
  }
};

/**
 * Créer un chart avec config par défaut
 */
export function createChart(ctx, type, data, customOptions = {}) {
  const options = deepMerge(chartDefaults, customOptions);

  return new Chart(ctx, {
    type,
    data,
    options
  });
}

/**
 * Palette de couleurs pour séries multiples
 */
export function getSeriesColors(count) {
  const palette = [
    '#3b82f6', '#2dd4bf', '#8b5cf6', '#f59e0b',
    '#ef4444', '#10b981', '#6366f1', '#ec4899',
    '#14b8a6', '#f97316', '#84cc16', '#06b6d4'
  ];
  return palette.slice(0, count);
}

// Deep merge helper
function deepMerge(target, source) {
  const output = { ...target };
  for (const key of Object.keys(source)) {
    if (source[key] instanceof Object && key in target) {
      output[key] = deepMerge(target[key], source[key]);
    } else {
      output[key] = source[key];
    }
  }
  return output;
}
```

---

## P3 - Documentation & Cleanup (ongoing)

### 10. Créer Design System doc

**Fichier**: `docs/DESIGN_SYSTEM.md`

```markdown
# SmartFolio Design System

## Couleurs

### Usage
- Primary: Actions principales, liens, focus
- Success: Confirmations, gains positifs
- Warning: Alertes non-critiques
- Danger: Erreurs, pertes, suppressions

### Tokens
Toujours utiliser les variables CSS:
- `var(--brand-primary)` au lieu de `#3b82f6`
- `var(--success)` au lieu de `#059669`

## Typographie

| Élément | Classe/Variable | Taille |
|---------|-----------------|--------|
| Titre page | `--font-size-2xl` | 24px |
| Titre card | `--font-size-lg` | 18px |
| Body | `--font-size-sm` | 14px |
| Label | `--font-size-xs` | 12px |

## Espacement

Utiliser l'échelle de 4px:
- `--space-1`: 4px
- `--space-2`: 8px
- `--space-3`: 12px
- `--space-4`: 16px
- `--space-6`: 24px
- `--space-8`: 32px

## Composants

### Boutons
```html
<button class="btn btn-primary">Primary</button>
<button class="btn btn-secondary">Secondary</button>
<button class="btn btn-ghost">Ghost</button>
<button class="btn btn-sm">Small</button>
```

### Modals
```javascript
import { UIModal } from './components/ui-modal.js';

UIModal.show({
  title: 'Titre',
  content: '<p>Contenu HTML</p>',
  onConfirm: () => { /* action */ }
});
```

### Notifications
```javascript
import { Toast } from './components/toast.js';

Toast.success('Message de succès');
Toast.error('Message d\'erreur');
Toast.loading('Chargement...');
```

## Breakpoints

| Nom | Valeur | Usage |
|-----|--------|-------|
| sm | 640px | Mobile landscape |
| md | 768px | Tablette |
| lg | 1024px | Desktop |
| xl | 1280px | Large desktop |
| 2xl | 1536px | 4K displays |

## Accessibilité

### Modals
- Toujours: `role="dialog"`, `aria-modal="true"`, `aria-labelledby`
- Focus trap obligatoire
- Escape pour fermer

### Interactive
- Focus visible sur tous les éléments interactifs
- Contrast ratio minimum 4.5:1
```

---

## Checklist de Migration

### Phase 1 (P0)
- [ ] Créer `static/css/tokens.css`
- [ ] Ajouter import tokens dans toutes les pages
- [ ] Corriger tooltips hardcodés dans risk-dashboard.css
- [ ] Ajouter `--theme-surface-hover` dans shared-theme.css

### Phase 2 (P1)
- [ ] Extraire CSS de saxo-dashboard.html
- [ ] Extraire CSS de simulations.html
- [ ] Créer `ui-modal.js`
- [ ] Créer `toast.js`
- [ ] Migrer export-button.js vers UIModal

### Phase 3 (P2)
- [ ] Unifier styles boutons
- [ ] Supprimer duplications @keyframes
- [ ] Créer chart-config.js
- [ ] Migrer tous les charts vers config unifiée

### Phase 4 (P3)
- [ ] Documentation DESIGN_SYSTEM.md
- [ ] Audit accessibilité (Lighthouse)
- [ ] Supprimer tous les inline styles
- [ ] Tests visuels (si applicable)

---

## Ressources

- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [MDN ARIA Practices](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA)
- [Chart.js Documentation](https://www.chartjs.org/docs/)
```

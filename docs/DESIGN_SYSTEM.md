# SmartFolio Design System

> Documentation du système de design frontend - Février 2026

## Vue d'ensemble

Le design system SmartFolio est organisé en couches :

```text
tokens.css          → Variables de base (couleurs, espacement, typographie)
    ↓
shared-theme.css    → Thème global (light/dark) + composants de base
    ↓
view-modes.css      → Système Simple/Pro (nouveau Fév 2026)
    ↓
[page].css          → Styles spécifiques par page
```

## Fichiers Principaux

| Fichier | Rôle |
|---------|------|
| `static/css/tokens.css` | Source de vérité - Variables CSS |
| `static/shared-theme.css` | Thème et composants globaux |
| `static/css/view-modes.css` | Classes Simple/Pro |
| `static/theme-compat.css` | Aliases de compatibilité |

## Système de Vues Simple/Pro

### Concept

Deux niveaux de visualisation pour chaque page :

- **Simple** : Vue executive summary (3-5 métriques clés)
- **Pro** : Vue Bloomberg Terminal (toutes les données)

### Fichiers

```text
static/core/view-mode-manager.js   → Gestionnaire d'état
static/components/view-toggle.js   → Web Component UI
static/css/view-modes.css          → Classes utilitaires
```

### Usage HTML

```html
<!-- Dans <head> -->
<link rel="stylesheet" href="css/view-modes.css">
<script type="module" src="components/view-toggle.js"></script>

<!-- Initialisation -->
<script type="module">
import { ViewModeManager } from './core/view-mode-manager.js';
ViewModeManager.init();
</script>

<!-- Toggle UI -->
<view-toggle></view-toggle>

<!-- Éléments conditionnels -->
<div class="pro-only">Visible en mode Pro uniquement</div>
<div class="simple-only">Visible en mode Simple uniquement</div>
```

### Classes Disponibles

| Classe | Description |
|--------|-------------|
| `.pro-only` | Caché en mode Simple |
| `.simple-only` | Caché en mode Pro |
| `.simple-compact` | Padding réduit en mode Simple |
| `.simple-large-value .metric-value` | Valeurs agrandies en mode Simple |
| `.simple-hide-label .metric-label` | Labels cachés en mode Simple |

### API JavaScript

```javascript
import { ViewModeManager, ViewModes } from './core/view-mode-manager.js';

// Initialiser
ViewModeManager.init();

// Lire le mode
ViewModeManager.getMode();     // 'simple' ou 'pro'
ViewModeManager.isSimple();    // boolean
ViewModeManager.isPro();       // boolean

// Changer le mode
ViewModeManager.setMode('simple');
ViewModeManager.toggle();

// Écouter les changements
const unsubscribe = ViewModeManager.on('change', (mode) => {
    console.log('Mode changé:', mode);
});

// Se désabonner
unsubscribe();
```

## Tokens CSS

### Couleurs Principales (WCAG AA Compliant)

```css
/* Success - WCAG AA (5.4:1 sur success-bg) */
--success: #047857;      /* WCAG FIX: Was #059669 (4.5:1) */
--success-bg: #d1fae5;

/* Danger - WCAG AA (5.0:1 sur danger-bg) */
--danger: #b91c1c;       /* WCAG FIX: Was #dc2626 (4.1:1) */
--danger-bg: #fee2e2;

/* Warning - WCAG AA (5.2:1 sur warning-bg) */
--warning: #92400e;      /* WCAG FIX: Was #d97706 (3.2:1) */
--warning-bg: #fef3c7;

/* Info - WCAG AA (5.2:1 sur info-bg) */
--info: #1d4ed8;         /* WCAG FIX: Was #2563eb (4.4:1) */
--info-bg: #dbeafe;

/* Brand */
--brand-primary: #3b82f6;
--brand-accent: #2dd4bf;
```

### Border Radius (Standardisé)

```css
/* Valeurs de base */
--radius-sm: 0.25rem;   /* 4px */
--radius-md: 0.375rem;  /* 6px */
--radius-lg: 0.5rem;    /* 8px */
--radius-xl: 0.75rem;   /* 12px */

/* Aliases sémantiques */
--radius-card: var(--radius-lg);      /* 8px - Cards */
--radius-button: var(--radius-md);    /* 6px - Boutons */
--radius-input: var(--radius-md);     /* 6px - Inputs */
--radius-badge: var(--radius-sm);     /* 4px - Badges */
--radius-modal: var(--radius-xl);     /* 12px - Modals */
```

### Espacement

```css
--space-xs: 0.25rem;   /* 4px */
--space-sm: 0.5rem;    /* 8px */
--space-md: 0.75rem;   /* 12px */
--space-lg: 1rem;      /* 16px */
--space-xl: 1.5rem;    /* 24px */
--space-2xl: 2rem;     /* 32px */
```

## Composants Existants

### Web Components

| Composant | Tag | Description |
|-----------|-----|-------------|
| ViewToggle | `<view-toggle>` | Toggle Simple/Pro |
| FlyoutPanel | `<flyout-panel>` | Panel coulissant |
| RiskSidebar | `<risk-sidebar-full>` | Sidebar risk metrics |
| RiskSnapshot | `<risk-snapshot>` | Compact risk view |
| RiskSummaryCard | `<risk-summary-card>` | Card risk 3 niveaux |
| EmptyState | `<empty-state>` | États vides standardisés |
| SkeletonLoader | `<skeleton-loader>` | Loading states animés |
| DomainNav | `<domain-nav>` | Navigation contextuelle |

### Composants JavaScript

| Composant | Import | Description |
|-----------|--------|-------------|
| Toast | `window.Toast` | Notifications |
| UIModal | `window.UIModal` | Modals/Dialogs |
| Badges | `Badges.js` | Badges gouvernance |
| DataTable | `DataTable` | Table tri/filtre/export |

## Nouveaux Composants (Phase 2)

### RiskSummaryCard

Card de résumé des métriques de risque avec 3 niveaux de détail.

```html
<!-- Compact (dashboard) -->
<risk-summary-card level="compact"></risk-summary-card>

<!-- Detailed (analytics) -->
<risk-summary-card level="detailed" poll-ms="30000"></risk-summary-card>

<!-- Full (risk page) -->
<risk-summary-card level="full" show-alerts="true"></risk-summary-card>
```

### DataTable

Table réutilisable avec tri, filtrage, pagination et export.

```javascript
import { DataTable } from './components/data-table.js';

const table = new DataTable('#container', {
    columns: [
        { key: 'symbol', label: 'Symbol', sortable: true },
        { key: 'value', label: 'Value', format: 'currency' },
        { key: 'change', label: 'Change', format: 'percent', colorCode: true }
    ],
    pagination: { enabled: true, pageSize: 25 },
    filterable: true,
    exportable: true
});
table.setData(myData);
```

### EmptyState

Affichage standardisé pour les états vides.

```html
<empty-state
    icon="Empty"
    title="No data available"
    description="Try adjusting your filters."
    action-text="Add Data"
    action-href="/settings.html">
</empty-state>
```

### SkeletonLoader

Loading states animés (shimmer effect).

```html
<skeleton-loader type="text" width="200px"></skeleton-loader>
<skeleton-loader type="card"></skeleton-loader>
<skeleton-loader type="table" rows="5"></skeleton-loader>
<skeleton-loader type="metric"></skeleton-loader>
```

### DomainNav

Navigation contextuelle entre pages liées.

```html
<!-- Domaines prédéfinis -->
<domain-nav domain="risk"></domain-nav>
<domain-nav domain="bourse"></domain-nav>
<domain-nav domain="analytics"></domain-nav>

<!-- Variantes visuelles -->
<domain-nav domain="risk" variant="pills"></domain-nav>
<domain-nav domain="risk" variant="breadcrumb"></domain-nav>
```

## Bonnes Pratiques

### Couleurs

```css
/* BON - Utiliser les variables */
color: var(--success);
background: var(--success-bg);

/* MAUVAIS - Couleurs hardcodées */
color: #22c55e;
color: #10b981;
```

### Border Radius

```css
/* BON - Utiliser les aliases sémantiques */
border-radius: var(--radius-card);

/* ACCEPTABLE - Utiliser les tokens */
border-radius: var(--radius-lg);

/* MAUVAIS - Valeurs hardcodées */
border-radius: 8px;
border-radius: 12px;
```

### Modes Simple/Pro

```html
<!-- BON - Classes sur les conteneurs -->
<section class="pro-only">
    <h3>Détails techniques</h3>
    <!-- Contenu pro -->
</section>

<!-- BON - Classes sur éléments individuels -->
<div class="metric">
    <span class="metric-label">Total</span>
    <span class="metric-value">$10,000</span>
    <span class="metric-detail pro-only">+2.5% depuis hier</span>
</div>
```

## Responsive Design

### Breakpoints

```css
--breakpoint-xs: 480px;
--breakpoint-sm: 640px;
--breakpoint-md: 768px;    /* Mobile */
--breakpoint-lg: 1024px;   /* Tablet */
--breakpoint-xl: 1280px;
--breakpoint-2xl: 1536px;  /* Desktop XL */
```

### Media Queries Standard

```css
/* Mobile first */
@media (max-width: 768px) { }

/* Tablet */
@media (max-width: 1024px) { }

/* Desktop large */
@media (min-width: 1400px) { }

/* Ultra-wide */
@media (min-width: 2000px) { }
```

## Dark Mode

Le thème dark est géré via `data-theme="dark"` sur `<body>`.

```css
[data-theme="dark"] {
    --theme-bg: #0a0f14;
    --theme-text: #e7eef7;
    --theme-surface: #0f172a;
    --success-bg: rgba(5, 150, 105, 0.1);
}
```

## Migration Guide

### Depuis les anciennes couleurs success

```css
/* AVANT */
--status-active: #22c55e;
--ai-success: #10b981;

/* APRÈS */
--status-active: var(--success, #059669);
--ai-success: var(--success, #059669);
```

### Depuis les radius hardcodés

```css
/* AVANT */
border-radius: 12px;

/* APRÈS */
border-radius: var(--radius-xl);
/* ou pour les cards */
border-radius: var(--radius-card);
```

### Depuis les styles inline

Pour les pages avec beaucoup de styles inline (>50), créer un fichier CSS externe :

```bash
# Étapes
1. Créer static/css/{page-name}.css
2. Extraire les styles inline
3. Ajouter <link rel="stylesheet" href="css/{page-name}.css">
4. Garder uniquement les styles "Critical CSS" inline (skeleton, layout initial)
```

**Pages prioritaires à migrer :**
- `bourse-analytics.html` (271 styles inline)
- `bourse-recommendations.html` (223 styles inline)
- `settings.html` (120 styles inline)
- `admin-dashboard.html` (120 styles inline)

## Accessibilité (WCAG 2.1 AA)

> **Audit Lighthouse : Février 2026** - Toutes les pages principales atteignent ≥90 en accessibilité

### Corrections Effectuées (Fév 2026)

| Problème | Pages Affectées | Solution |
|----------|-----------------|----------|
| Color contrast insuffisant | Toutes | Couleurs sémantiques assombries (voir Tokens CSS) |
| Heading order incorrect | analytics-unified, risk-dashboard, saxo-dashboard | Restructuration h1→h2→h3→h4 |
| Missing main landmark | settings, saxo-dashboard | Ajout `<main role="main">` |
| Form labels manquants | settings, ai-chat-modal, WealthContextBar | Ajout `<label for="id">` et `aria-label` |
| aria-required-children | analytics-unified | Bouton refresh sorti du tablist |
| Label-content mismatch | dashboard (export buttons) | `aria-label` aligné avec contenu visuel |

### Focus Visible

```css
/* Tous les éléments interactifs doivent avoir un focus visible */
*:focus-visible {
  outline: 2px solid var(--brand-primary, #3b82f6);
  outline-offset: 2px;
}
```

### Contraste Minimum

| Élément | Ratio minimum | Couleurs SmartFolio |
|---------|---------------|---------------------|
| Texte normal | 4.5:1 | success=#047857, danger=#b91c1c, warning=#92400e, info=#1d4ed8 |
| Grand texte (18px+ ou 14px bold) | 3:1 | Conforme |
| Composants UI | 3:1 | Conforme |

### Structure Sémantique

```html
<!-- Ordre des headings -->
<h1>Page Title</h1>
  <h2>Section</h2>
    <h3>Subsection</h3>
      <h4>Detail</h4>

<!-- Main landmark obligatoire -->
<body>
  <nav>...</nav>
  <main role="main">
    <!-- Contenu principal -->
  </main>
</body>

<!-- Tablist correct -->
<div class="tabs-wrapper">
  <div role="tablist" aria-label="Section tabs">
    <button role="tab">Tab 1</button>
    <button role="tab">Tab 2</button>
  </div>
  <button class="btn-refresh">Refresh</button> <!-- Hors du tablist -->
</div>
```

### Reduced Motion

```css
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

### Attributs ARIA

```html
<!-- Loading states -->
<div aria-busy="true" aria-live="polite">Loading...</div>

<!-- Navigation -->
<nav role="navigation" aria-label="Main navigation">

<!-- Alertes -->
<div role="alert" aria-live="assertive">Error message</div>

<!-- Form labels -->
<label for="input-id">Label text</label>
<input id="input-id" aria-label="Description complète">

<!-- Selects -->
<select aria-label="Sélection de source">
```

## Tests Visuels

### Checklist Avant Déploiement

- [ ] Test en mode Light et Dark
- [ ] Test en mode Simple et Pro
- [ ] Test responsive (768px, 1024px, 1400px, 2000px)
- [ ] Test accessibilité clavier (Tab navigation)
- [ ] Test avec `prefers-reduced-motion`
- [ ] Vérifier contrastes avec DevTools

### Lighthouse Targets

| Métrique | Target |
|----------|--------|
| Performance | ≥85 |
| Accessibility | ≥90 |
| Best Practices | ≥90 |
| SEO | ≥80 |

## Changelog

### Février 2026

#### Phase 4.2 - Accessibilité WCAG AA (5 Fév)

- **Couleurs sémantiques WCAG AA** : success=#047857, danger=#b91c1c, warning=#92400e, info=#1d4ed8
- **Heading order** : Corrigé sur analytics-unified (19 headings), risk-dashboard, saxo-dashboard
- **Main landmarks** : Ajoutés sur settings.html, saxo-dashboard.html
- **Form labels** : Corrigés sur settings.html, ai-chat-modal.html, WealthContextBar.js, GovernancePanel.js
- **ARIA fixes** : analytics-unified tablist, dashboard export buttons
- **Lighthouse scores** : Toutes pages ≥92 en accessibilité (target: 90)

#### Phase 4.1 - Finitions (4 Fév)

- Unifié `--theme-accent` sur `var(--brand-accent)` (était `#00ff88`)
- Corrigé couleurs Decision Index (`.status-badge--live` → `var(--success)`)
- Documentation accessibilité ajoutée

#### Phase 3 - Intégration

- Toggle Simple/Pro intégré sur 12 pages
- Navigation contextuelle (domain-nav) sur 10 pages
- CSS extrait de simulations.html (468 lignes)

#### Phase 2 - Composants

- Créé `risk-summary-card.js`, `data-table.js`, `empty-state.js`, `skeleton-loader.js`, `domain-nav.js`

#### Phase 1 - Foundation

- Créé système de vues Simple/Pro
- Unifié couleur success sur `#059669`
- Standardisé border-radius avec aliases sémantiques

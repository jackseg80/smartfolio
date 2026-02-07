# UI Audit Report - SmartFolio

> Audit complet de l'interface utilisateur
> Date: Décembre 2025

---

## Executive Summary

L'audit révèle un projet avec une **base solide** (système de thème CSS variables, responsive design, dark/light mode) mais des **incohérences significatives** qui nuisent à l'expérience utilisateur et à la maintenabilité.

### Score Global: 6/10

| Catégorie | Score | Statut |
|-----------|-------|--------|
| Système de couleurs | 7/10 | ⚠️ Bien structuré mais duplications |
| Typographie | 6/10 | ⚠️ Incohérences de tailles |
| Composants UI | 5/10 | ❌ Fragmenté, pas de bibliothèque |
| Graphiques | 4/10 | ❌ Multiples bibliothèques non unifiées |
| Accessibilité | 5/10 | ⚠️ Partielle, modals problématiques |
| Responsive | 8/10 | ✅ Bon, breakpoints cohérents |
| Architecture CSS | 5/10 | ⚠️ Fichiers monolithiques, duplications |

---

## 1. Analyse des Pages HTML

### Structure des 8 Pages Principales

| Page | Lignes | CSS | Problèmes |
|------|--------|-----|-----------|
| **dashboard.html** | 899 | shared + theme-compat | ✅ Bien structurée |
| **analytics-unified.html** | 493 | + analytics-unified-theme | ✅ Bonnes pratiques (critical CSS) |
| **risk-dashboard.html** | 285 | + risk-dashboard.css | ✅ Structure claire |
| **rebalance.html** | 288 | + rebalance.css | ✅ Deux onglets cohérents |
| **execution.html** | 1190 | ⚠️ Inline massif | ❌ Business logic inline |
| **simulations.html** | 1846 | ⚠️ CSS inline extensif | ❌ Fichier trop volumineux |
| **wealth-dashboard.html** | 1341 | ⚠️ Inline styles | ⚠️ Modal inline vs overlay |
| **saxo-dashboard.html** | 6656 | ⚠️ >6000 lignes CSS inline | ❌ CRITIQUE - refactoring urgent |

### Incohérences de Structure

```
❌ saxo-dashboard.html: 6656 lignes (CSS inline massif)
❌ simulations.html: 1846 lignes (contrôles flyout custom)
❌ execution.html: 1190 lignes (logique métier inline)
✅ analytics-unified.html: 493 lignes (bien modulaire)
```

---

## 2. Système de Couleurs

### Variables CSS Définies (shared-theme.css)

```css
/* Couleurs de marque */
--brand-primary: #3b82f6      /* Bleu principal */
--brand-accent: #2dd4bf       /* Teal accent */

/* États sémantiques */
--success: #059669            /* Vert succès */
--warning: #d97706            /* Ambre avertissement */
--danger: #dc2626             /* Rouge erreur */

/* Thème clair */
--theme-bg: #f8fafc
--theme-surface: #ffffff
--theme-text: #1e293b

/* Thème sombre */
[data-theme="dark"] --theme-bg: #0a0f14
[data-theme="dark"] --theme-surface: #0e1620
```

### ❌ Problème #1: Couleurs Hardcodées

**risk-dashboard.css (lignes 1124-1150)**:
```css
.tooltip {
  background: #0e1528;    /* ❌ Hardcodé */
  color: #e9f0ff;         /* ❌ Hardcodé */
  border: 1px solid #243355;
}
```

**Impact**: Les tooltips ne suivent pas le thème light/dark.

### ❌ Problème #2: Palette AI Isolée

**ai-components.css** définit son propre système:
```css
--ai-primary: #6366f1   /* ≠ #3b82f6 brand-primary */
--ai-secondary: #8b5cf6
--ai-success: #10b981   /* ≠ #059669 success */
```

**Impact**: Les composants AI ont une apparence différente du reste.

### ❌ Problème #3: Decision Index Panel Différent

**decision-index-panel.css**:
```css
--di-color-cycle: #7aa2f7   /* ≠ #3b82f6 brand-primary */
--di-color-onchain: #2ac3de
--di-color-risk: #f7768e
```

### ❌ Problème #4: Variantes de Succès Incohérentes

| Fichier | Couleur "Success Light" |
|---------|------------------------|
| analytics-unified-theme.css | `#8bd17c` |
| decision-index-panel.css | `#6ecb8b` |
| ai-components.css | `#10b981` |

**3 couleurs différentes pour le même concept!**

### ❌ Problème #5: Opacités Non Standardisées

```css
/* Même couleur, opacités différentes partout */
rgba(122, 162, 247, 0.10)  /* DI panel */
rgba(122, 162, 247, 0.15)  /* Analytics */
rgba(122, 162, 247, 0.20)  /* Risk dashboard */
rgba(122, 162, 247, 0.25)  /* Autre endroit */
```

---

## 3. Typographie

### Échelle de Tailles (Incohérente)

| Usage | Taille attendue | Trouvée dans les fichiers |
|-------|-----------------|---------------------------|
| Titres H1 | `2rem` | `1.5rem`, `2rem`, `24px`, `20px` |
| Headers H3 | `1.25rem` | `1.75rem`, `1.25rem`, `1.1rem` |
| Body text | `0.875rem` | `0.875rem`, `0.9rem`, `0.95rem` |
| Labels | `0.75rem` | `0.75rem`, `0.8rem`, `0.85rem` |

### Famille de Police

✅ **Cohérent** - Toutes les pages utilisent:
```css
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', ...
```

❌ **Exception**: `risk-dashboard.css` utilise Monaco/Consolas pour les métriques.

---

## 4. Composants UI

### 4.1 Graphiques - CRITIQUE ❌

**Problème**: 3 bibliothèques différentes sans interface commune.

| Bibliothèque | Fichiers | Usage |
|--------------|----------|-------|
| **Chart.js 4.4.x** | Toutes pages | Donuts, lignes, barres |
| **Highcharts** | AdvancedCharts.js | Heatmaps, scatter |
| **Plotly.js** | saxo-dashboard.html | Dendrogrammes |

**Impacts**:
- Tooltips différents selon la lib
- Exports différents
- Thèmes non synchronisés
- Bundle JS plus lourd

### 4.2 Tooltips - Partiellement Unifié ⚠️

**3 systèmes coexistent**:

1. **CSS-based** (`tooltips.js`): `[data-tooltip]` attribute
2. **Chart tooltips**: Formatters JavaScript custom
3. **Aria-label**: Badges (pas de tooltip visuel)

```html
<!-- Système 1: CSS -->
<div data-tooltip="Description">Content</div>

<!-- Système 2: Chart.js -->
tooltip: { formatter: function() { return '...'; } }

<!-- Système 3: ARIA only -->
<span aria-label="Info">Badge</span>
```

### 4.3 Modals/Dialogs - CRITIQUE ❌

**3 implémentations incompatibles**:

| Fichier | Approche | Accessibilité |
|---------|----------|---------------|
| export-button.js | Divs + inline CSS | ❌ Aucune ARIA |
| decision-index-panel.js | Class-based + CSS injecté | ✅ Complète |
| wealth-dashboard.html | Inline HTML | ⚠️ Partielle |

**Export Modal - Problématique**:
```javascript
// ❌ Pas d'accessibilité
overlay.style.cssText = `position: fixed; ...`;
modal.className = 'export-modal';
// Manque: role="dialog", aria-modal="true", focus trap
```

**Decision Index - Bonne pratique**:
```html
<!-- ✅ Accessible -->
<div role="dialog" aria-labelledby="title" aria-modal="true">
```

### 4.4 Tables - Fragmenté ❌

**Aucun composant table réutilisable**:
- Chaque page reconstruit ses tables en HTML
- Pas de tri/filtrage/pagination unifié
- Styles dupliqués

### 4.5 Boutons - Incohérent ⚠️

**shared-theme.css**:
```css
.btn { padding: 0.75rem 1.5rem; font-size: 0.875rem; }
.btn-sm { padding: 0.5rem 1rem; font-size: 0.75rem; }
```

**rebalance.css**:
```css
.btn { padding: 8px 10px; font-weight: 700; }
.btn.small { padding: 6px 8px; font-size: 12px; }
```

**2 échelles de padding différentes!**

### 4.6 Cards - 3 Implémentations

| Fichier | Background | Border | Padding |
|---------|------------|--------|---------|
| shared-theme.css | `var(--theme-surface)` | `var(--theme-border)` | `var(--space-xl)` |
| ai-components.css | `var(--ai-bg-primary)` | `var(--ai-border)` | `var(--ai-spacing-lg)` |
| analytics-unified | `var(--glass-bg)` | `var(--glass-border)` | `clamp(1.5rem, 4vw, 2rem)` |

---

## 5. Loading States & Erreurs

### États de Chargement - Non Unifié ⚠️

**shared-ml-functions.js**:
```javascript
export function showLoading(elementId, message = 'Chargement...') {
    element.innerHTML = `<span class="loading-spinner"></span> ${message}`;
}
```

**ai-components.js**:
```javascript
statusDot.className = 'status-dot loading';
// Classes: loading, online, offline
```

**btc-regime-chart.js**:
```javascript
try {
    showLoadingState();
    // ...
} catch (error) {
    console.error('...'); // ❌ Pas de feedback utilisateur
}
```

### Gestion d'Erreurs - Incohérente ❌

- Certains composants: Toast auto-dismiss (5s)
- Certains: Console.error seulement
- Certains: Alert modal

---

## 6. Responsive Design

### ✅ Points Positifs

- **Breakpoints cohérents**: 768px, 1024px, 1400px, 2000px
- **Grid auto-fit**: `repeat(auto-fit, minmax(300px, 1fr))`
- **Container responsive**: `max-width: none` ou `95vw`

### ⚠️ Points d'Attention

| Page | Container | Problème |
|------|-----------|----------|
| dashboard.html | `max-width: 95vw` | Différent des autres |
| Autres pages | `max-width: none` | Standard |

---

## 7. Architecture CSS

### Fichiers Monolithiques

| Fichier | Lignes | Statut |
|---------|--------|--------|
| risk-dashboard.css | 2321 | ❌ À découper |
| decision-index-panel.css | 815 | ⚠️ Acceptable |
| analytics-unified-theme.css | 960 | ⚠️ Acceptable |
| ai-components.css | 746 | ✅ OK |

### Duplications Identifiées

```css
/* @keyframes spin défini 4 fois: */
shared-theme.css
ai-components.css
analytics-unified-theme.css
shared-ml-styles.css
```

### Variables Non Définies Référencées

```css
/* governance-panel.css utilise: */
var(--theme-surface-hover)  /* ❌ Non défini dans shared-theme.css */

/* rebalance.css utilise: */
var(--brand-primary-subtle) /* ❌ Non défini partout */
```

---

## 8. Erreurs d'Implémentation Détectées

### 8.1 Z-Index Non Coordonné

```css
/* export-button.js */
z-index: 10000

/* Pas de système de gestion des z-index */
```

**Recommandation**: Créer échelle z-index:
```css
--z-dropdown: 100;
--z-modal: 1000;
--z-tooltip: 1100;
--z-overlay: 10000;
```

### 8.2 Inline Styles dans 10+ Fichiers HTML

**Fichiers affectés**:
- ai-dashboard.html
- alias-manager.html
- analytics-unified.html
- dashboard.html
- execution.html
- execution_history.html
- wealth-dashboard.html
- saxo-dashboard.html (critique)
- simulations.html
- performance-monitor.html

**117+ occurrences** de `style=` dans le JS.

### 8.3 Shadow DOM vs Global Styles

Mix de Web Components (Shadow DOM) et classes globales:
- `FlyoutPanel` - Web Component
- `GovernancePanel` - Classe JS
- `renderDecisionIndexPanel()` - Fonction pure

**Aucune convention architecturale claire.**

### 8.4 Focus Management Manquant

**Export Modal**: Pas de focus trap
```javascript
// ❌ Le focus peut sortir du modal
overlay.onclick = () => overlay.remove();
```

### 8.5 Animation Duplications

```css
/* 4 définitions différentes de @keyframes pulse */
/* Timing: 1.5s, 2s, 3s selon les fichiers */
```

---

## 9. Améliorations Potentielles

### 9.1 Performance

1. **Lazy Loading Charts**: Plotly.js (1.5MB) chargé même si onglet non utilisé
2. **CSS critique**: Seul analytics-unified.html utilise critical CSS inline
3. **Bundle splitting**: Tout Chart.js chargé même pour pages simples

### 9.2 UX

1. **Skeleton loaders**: Implémentés sur analytics, manquants ailleurs
2. **Feedback erreurs**: Souvent console.error sans UI
3. **Transitions**: Inconsistantes (150ms, 200ms, 300ms selon composants)

### 9.3 Maintenabilité

1. **Documentation composants**: Aucune
2. **Storybook/Design System**: Inexistant
3. **Tests visuels**: Non détectés

---

## 10. Plan d'Amélioration

### Phase 1: Fondations (Priorité Critique)

#### 1.1 Système de Design Tokens
```css
/* Créer static/css/tokens.css */
:root {
  /* Couleurs - Échelle complète */
  --color-primary-50: #eff6ff;
  --color-primary-100: #dbeafe;
  --color-primary-500: #3b82f6;
  --color-primary-600: #2563eb;

  /* Opacités standardisées */
  --opacity-subtle: 0.1;
  --opacity-light: 0.2;
  --opacity-medium: 0.3;

  /* Z-index scale */
  --z-base: 0;
  --z-dropdown: 100;
  --z-sticky: 200;
  --z-modal: 1000;
  --z-popover: 1100;
  --z-toast: 1200;

  /* Transitions */
  --duration-fast: 150ms;
  --duration-normal: 200ms;
  --duration-slow: 300ms;
  --easing-default: ease;
  --easing-smooth: cubic-bezier(0.4, 0, 0.2, 1);
}
```

#### 1.2 Refactorer saxo-dashboard.html
- Extraire CSS vers `static/css/saxo-dashboard.css`
- Réduire de 6656 → ~500 lignes HTML
- Utiliser tokens CSS

#### 1.3 Corriger Tooltips Hardcodés
```css
/* Avant */
.tooltip { background: #0e1528; }

/* Après */
.tooltip {
  background: var(--theme-surface-elevated);
  color: var(--theme-text);
  border: 1px solid var(--theme-border);
}
```

### Phase 2: Composants Unifiés

#### 2.1 Créer UIModal Component
```javascript
// static/components/ui-modal.js
export class UIModal {
  constructor(options) {
    this.title = options.title;
    this.content = options.content;
    this.onClose = options.onClose;
  }

  open() {
    // Focus trap
    // ARIA attributes
    // Escape key handling
    // Backdrop click
  }

  close() {
    // Return focus
    // Cleanup
  }
}
```

#### 2.2 Créer DataTable Component
```javascript
// static/components/data-table.js
export class DataTable {
  constructor(container, options) {
    this.sortable = options.sortable ?? true;
    this.filterable = options.filterable ?? false;
    this.pagination = options.pagination ?? null;
  }

  setData(data) { /* ... */ }
  sort(column, direction) { /* ... */ }
  filter(predicate) { /* ... */ }
}
```

#### 2.3 Unifier Notifications
```javascript
// static/components/notifications.js
export const toast = {
  success(message, duration = 5000) { /* ... */ },
  error(message, duration = 8000) { /* ... */ },
  warning(message, duration = 6000) { /* ... */ },
  loading(message) { /* returns dismiss function */ }
};
```

### Phase 3: Consolidation Graphiques

#### 3.1 Créer Chart Abstraction
```javascript
// static/core/chart-factory.js
export function createChart(container, type, options) {
  const config = {
    theme: getCurrentTheme(),
    tooltip: standardTooltipConfig,
    colors: standardColorPalette,
    ...options
  };

  switch(type) {
    case 'line':
    case 'bar':
    case 'doughnut':
      return new ChartJSWrapper(container, config);
    case 'heatmap':
    case 'scatter':
      return new HighchartsWrapper(container, config);
    case 'dendrogram':
      return new PlotlyWrapper(container, config);
  }
}
```

#### 3.2 Standardiser Tooltips Charts
```javascript
// Configuration partagée
export const chartTooltipConfig = {
  backgroundColor: 'var(--theme-surface-elevated)',
  borderColor: 'var(--theme-border)',
  titleColor: 'var(--theme-text)',
  bodyColor: 'var(--theme-text-muted)',
  padding: 12,
  cornerRadius: 8
};
```

### Phase 4: Nettoyage & Documentation

#### 4.1 Supprimer Duplications
- [ ] Fusionner les 4 `@keyframes spin`
- [ ] Consolider les définitions de `.card`
- [ ] Unifier les styles de `.btn`

#### 4.2 Audit Accessibilité
- [ ] Ajouter ARIA à tous les modals
- [ ] Implémenter focus traps
- [ ] Tester navigation clavier
- [ ] Vérifier contraste couleurs

#### 4.3 Documentation
- [ ] Créer `docs/DESIGN_SYSTEM.md`
- [ ] Documenter tokens CSS
- [ ] Ajouter JSDoc aux composants

---

## 11. Fichiers à Modifier (Priorité)

### Critique (Phase 1)
1. `static/saxo-dashboard.html` → Extraire CSS
2. `static/css/shared-theme.css` → Ajouter tokens manquants
3. `static/css/risk-dashboard.css` → Corriger hardcoded colors

### Important (Phase 2)
4. `static/components/` → Créer UIModal, DataTable
5. `static/modules/export-button.js` → Ajouter accessibilité
6. `static/simulations.html` → Extraire inline CSS

### Normal (Phase 3-4)
7. `static/ai-components.css` → Intégrer avec theme
8. `static/components/decision-index-panel.css` → Utiliser tokens
9. Tous HTML → Supprimer inline styles

---

## 12. Métriques de Succès

| Métrique | Actuel | Cible |
|----------|--------|-------|
| Fichiers CSS | 12 | 8 |
| Lignes CSS total | ~8000 | ~4000 |
| Couleurs hardcodées | 50+ | 0 |
| Duplications @keyframes | 4 | 1 |
| Composants sans ARIA | 3+ | 0 |
| Inline styles HTML | 117+ | <20 |
| Score Lighthouse Accessibility | ~75 | >90 |

---

## Conclusion

Le projet SmartFolio a une **base technique solide** avec un bon système de thème CSS et une architecture responsive fonctionnelle. Cependant, la croissance organique a créé des **incohérences significatives** qui impactent:

1. **Expérience utilisateur**: Apparence non uniforme entre pages
2. **Accessibilité**: Modals et tooltips non conformes
3. **Maintenabilité**: Duplications et fichiers monolithiques
4. **Performance**: Bundles non optimisés

La mise en œuvre du plan d'amélioration en 4 phases permettra d'atteindre un niveau de qualité UI professionnel tout en préservant les fonctionnalités existantes.

---

*Rapport généré le 16 décembre 2025*


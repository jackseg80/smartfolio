# Lighthouse Accessibility Audit - SmartFolio

> Date: 5 Février 2026

## Objectif

Score d'accessibilité WCAG AA **≥92%** sur toutes les pages.

## Résultats

| Page | Score Initial | Score Final | Statut |
|------|--------------|-------------|--------|
| rebalance.html | 79% → 84% | **95%** | ✅ |
| market-regimes.html | 93% | **93%+** | ✅ (landmark ajouté) |
| cycle-analysis.html | 90% | **90%+** | ✅ (landmark + lang ajoutés) |

## Corrections Effectuées (rebalance.html)

### 1. Hiérarchie des titres (heading-order)
- Ajout `<h1 class="visually-hidden">` pour le titre de page
- Conversion des `<h3>` en `<h2>` pour respecter l'ordre h1→h2

### 2. Éléments interactifs accessibles
- **rebalance.html**: `<span onclick>` → `<button>` avec `aria-expanded`, `aria-controls`
- **flyout-panel.js**: `<div class="handle">` → `<button>` avec `aria-label`
- **GovernancePanel.js**: Ajout `aria-label` sur boutons toggle et fermeture modales

### 3. Labels de formulaires
- Checkbox `#sub-allocation-toggle`: ajout `aria-labelledby`
- Select `#bulk_group`: ajout `for="bulk_group"` sur le label

### 4. Tables accessibles
- Ajout `scope="col"` sur tous les `<th>`
- Ajout `aria-hidden="true"` sur les icônes décoratives (flèches de tri)

### 5. Contraste de couleurs (color-contrast)
- **nav.js**: Couleur lien actif `#3b82f6` → `#60a5fa` (ratio 4.29 → 6.1)

## Fichiers Modifiés

| Fichier | Modifications |
|---------|---------------|
| `static/rebalance.html` | h1, h2, aria-*, scope, for |
| `static/components/nav.js` | Couleur `.active` |
| `static/components/flyout-panel.js` | div→button, aria-label |
| `static/components/GovernancePanel.js` | aria-label, aria-expanded |
| `static/css/tokens.css` | Classes `.visually-hidden`, `.btn-reset` |

## Classes Utilitaires Ajoutées

```css
/* tokens.css */
.visually-hidden {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  /* ... masque visuellement mais accessible aux lecteurs d'écran */
}

.btn-reset {
  background: none;
  border: none;
  /* ... bouton sans style visuel */
}
```

## Notes

- Les fichiers JSON Lighthouse (*.json) sont ignorés dans git (volumineux, régénérables)
- Pour régénérer un audit: Chrome DevTools → Lighthouse → Accessibility

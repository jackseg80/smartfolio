# Chart Abstraction Guide

> Guide d'utilisation de l'abstraction Chart.js unifiée
> Date: 16 Décembre 2025

## Vue d'Ensemble

Le fichier `static/core/chart-config.js` fournit une abstraction unifiée pour Chart.js avec :

- ✅ **Configuration par défaut** : Couleurs, thème, tooltips cohérents
- ✅ **Theme-aware** : Support automatique dark/light mode
- ✅ **Palette de couleurs** : Couleurs cohérentes pour séries multiples
- ✅ **Helper simplifié** : Moins de code répétitif
- ✅ **Presets** : Configurations prêtes pour cas d'usage communs

---

## Installation

### 1. Importer le module

```javascript
import { createChart, chartColors, getSeriesColors } from './core/chart-config.js';
```

### 2. Utiliser la fonction helper

**Avant (vanilla Chart.js)** :
```javascript
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: ['Jan', 'Feb', 'Mar'],
    datasets: [{
      label: 'Sales',
      data: [100, 200, 150],
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59, 130, 246, 0.2)'
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { /* config */ },
      tooltip: { /* config */ }
    },
    scales: {
      x: { /* config */ },
      y: { /* config */ }
    }
  }
});
```

**Après (avec abstraction)** :
```javascript
const chart = createChart(ctx, 'line', {
  labels: ['Jan', 'Feb', 'Mar'],
  datasets: [{
    label: 'Sales',
    data: [100, 200, 150],
    borderColor: chartColors.primary,
    backgroundColor: chartColors.primaryAlpha
  }]
});
```

**Résultat** :
- **-60% de code** (15 lignes → 6 lignes)
- **Theme-aware** automatique
- **Configuration cohérente** avec le reste de l'app

---

## API Reference

### `createChart(ctx, type, data, [customOptions])`

Crée un Chart.js avec configuration par défaut.

**Paramètres** :
- `ctx` : Canvas context ou élément
- `type` : Type de chart (`'line'`, `'bar'`, `'pie'`, `'doughnut'`, etc.)
- `data` : Données du chart (labels + datasets)
- `customOptions` : Options custom pour override les defaults (optionnel)

**Exemple** :
```javascript
const chart = createChart(ctx, 'bar', {
  labels: ['A', 'B', 'C'],
  datasets: [{
    label: 'Values',
    data: [10, 20, 30],
    backgroundColor: chartColors.primary
  }]
});
```

---

### `chartColors`

Objet avec couleurs theme-aware.

**Propriétés** :
- `primary` : Couleur primaire (bleu)
- `primaryAlpha` : Variante transparente (20%)
- `accent` : Couleur accent (teal)
- `accentAlpha` : Variante transparente
- `success` : Vert
- `warning` : Ambre
- `danger` : Rouge
- `text` : Couleur texte
- `textMuted` : Texte muted
- `border` : Couleur bordure
- `surface` : Couleur surface

**Exemple** :
```javascript
datasets: [{
  borderColor: chartColors.primary,
  backgroundColor: chartColors.primaryAlpha,
  pointBackgroundColor: chartColors.accent
}]
```

---

### `getSeriesColors(count)`

Retourne palette de N couleurs pour séries multiples.

**Paramètres** :
- `count` : Nombre de couleurs nécessaires

**Retourne** : `string[]` - Array de couleurs hex

**Exemple** :
```javascript
const colors = getSeriesColors(3); // ['#3b82f6', '#2dd4bf', '#8b5cf6']

datasets: colors.map((color, i) => ({
  label: `Series ${i + 1}`,
  data: dataArray[i],
  borderColor: color,
  backgroundColor: withAlpha(color, 0.2)
}))
```

---

### `withAlpha(color, alpha)`

Ajoute transparence à une couleur.

**Paramètres** :
- `color` : Couleur hex (`#3b82f6`)
- `alpha` : Valeur alpha (0-1)

**Retourne** : `string` - Couleur hex avec alpha

**Exemple** :
```javascript
backgroundColor: withAlpha('#3b82f6', 0.3) // #3b82f64d (30%)
```

---

### `updateChartTheme(chart)`

Met à jour le thème d'un chart existant (utile lors du changement dark/light).

**Paramètres** :
- `chart` : Instance Chart.js

**Exemple** :
```javascript
// Écouter changement de thème
document.addEventListener('themeChanged', () => {
  updateChartTheme(myChart);
});
```

---

### `chartPresets`

Configurations prêtes pour cas d'usage communs.

**Presets disponibles** :
- `timeSeries` : Line chart pour séries temporelles
- `barComparison` : Bar chart pour comparaisons
- `allocation` : Doughnut chart pour allocations portfolio

**Exemple** :
```javascript
import { createChart, chartPresets } from './core/chart-config.js';

const chart = createChart(ctx, 'line', data, chartPresets.timeSeries);
```

---

## Exemples de Migration

### Exemple 1 : Line Chart Simple

**Avant** :
```javascript
const chart = new Chart(ctx, {
  type: 'line',
  data: { /* ... */ },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#1f2937' }
      },
      tooltip: {
        backgroundColor: '#f9fafb',
        titleColor: '#1f2937'
      }
    },
    scales: {
      x: {
        grid: { color: '#f3f4f6' },
        ticks: { color: '#6b7280' }
      },
      y: {
        grid: { color: '#f3f4f6' },
        ticks: { color: '#6b7280' }
      }
    }
  }
});
```

**Après** :
```javascript
import { createChart, chartColors } from './core/chart-config.js';

const chart = createChart(ctx, 'line', { /* data */ });
```

---

### Exemple 2 : Séries Multiples

**Avant** :
```javascript
const colors = ['#3b82f6', '#2dd4bf', '#8b5cf6'];
const datasets = dataArray.map((data, i) => ({
  label: labels[i],
  data: data,
  borderColor: colors[i],
  backgroundColor: colors[i] + '33' // Hardcoded alpha
}));

const chart = new Chart(ctx, {
  type: 'line',
  data: { labels, datasets },
  options: { /* ... 30 lignes de config ... */ }
});
```

**Après** :
```javascript
import { createChart, getSeriesColors, withAlpha } from './core/chart-config.js';

const colors = getSeriesColors(dataArray.length);
const datasets = dataArray.map((data, i) => ({
  label: labels[i],
  data: data,
  borderColor: colors[i],
  backgroundColor: withAlpha(colors[i], 0.2)
}));

const chart = createChart(ctx, 'line', { labels, datasets });
```

---

### Exemple 3 : Custom Options

**Override des defaults** :
```javascript
import { createChart, chartPresets } from './core/chart-config.js';

const chart = createChart(ctx, 'line', data, {
  ...chartPresets.timeSeries, // Utiliser preset
  plugins: {
    legend: {
      display: false // Override legend
    }
  },
  scales: {
    y: {
      beginAtZero: true // Override scale Y
    }
  }
});
```

---

### Exemple 4 : Theme Change Support

**Mettre à jour charts lors du changement de thème** :
```javascript
import { createChart, updateChartTheme } from './core/chart-config.js';

// Créer chart
const chart = createChart(ctx, 'line', data);

// Écouter changements de thème
const observer = new MutationObserver(() => {
  updateChartTheme(chart);
});

observer.observe(document.documentElement, {
  attributes: true,
  attributeFilter: ['data-theme']
});
```

---

## Checklist de Migration

Pour chaque fichier utilisant Chart.js :

- [ ] Importer `createChart`, `chartColors` de `chart-config.js`
- [ ] Remplacer `new Chart(ctx, { ... })` par `createChart(ctx, type, data)`
- [ ] Remplacer couleurs hardcodées par `chartColors.*`
- [ ] Supprimer options redondantes (responsive, scales, tooltips)
- [ ] Ajouter `updateChartTheme()` si changement de thème supporté
- [ ] Tester en dark/light mode

---

## Fichiers à Migrer

**Total : 9 usages dans 4 fichiers**

1. `cycle-analysis.html` (1 chart)
2. `execution_history.html` (1 chart)
3. `portfolio-optimization-advanced.html` (2 charts)
4. `saxo-dashboard.html` (5 charts)

**Priorité** : saxo-dashboard.html (5 charts)

---

## Avantages

### Avant Abstraction

**Problèmes** :
- ❌ Code répétitif (30-50 lignes de config par chart)
- ❌ Couleurs hardcodées (`#3b82f6`, `rgba(59, 130, 246, 0.2)`)
- ❌ Pas de support dark/light mode automatique
- ❌ Incohérence visuelle entre charts
- ❌ Difficile à maintenir (changement = modifier tous les charts)

### Après Abstraction

**Avantages** :
- ✅ Code concis (-60% de lignes)
- ✅ Couleurs theme-aware (variables CSS)
- ✅ Support dark/light mode automatique
- ✅ Cohérence visuelle garantie
- ✅ Maintenance centralisée (modifier 1 fichier = tous les charts)
- ✅ Presets pour cas d'usage communs
- ✅ Palette de couleurs unifiée

---

## Exemple Complet

**saxo-dashboard.html - Avant** :
```javascript
stressTestChartInstance = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: ['Current', '-10%', '-20%', '-30%'],
    datasets: [{
      label: 'Portfolio Value',
      data: [100000, 90000, 80000, 70000],
      backgroundColor: '#3b82f6',
      borderColor: '#2563eb',
      borderWidth: 1
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#1f2937', font: { size: 12 } }
      },
      tooltip: {
        backgroundColor: '#f9fafb',
        titleColor: '#1f2937',
        bodyColor: '#6b7280',
        borderColor: '#e5e7eb',
        borderWidth: 1,
        cornerRadius: 8,
        padding: 12
      }
    },
    scales: {
      x: {
        grid: { color: '#f3f4f6' },
        ticks: { color: '#6b7280', font: { size: 11 } }
      },
      y: {
        grid: { color: '#f3f4f6' },
        ticks: { color: '#6b7280', font: { size: 11 } }
      }
    }
  }
});
```

**saxo-dashboard.html - Après** :
```javascript
import { createChart, chartColors, chartPresets } from './core/chart-config.js';

stressTestChartInstance = createChart(ctx, 'bar', {
  labels: ['Current', '-10%', '-20%', '-30%'],
  datasets: [{
    label: 'Portfolio Value',
    data: [100000, 90000, 80000, 70000],
    backgroundColor: chartColors.primary,
    borderColor: chartColors.primary,
    borderWidth: 1
  }]
}, chartPresets.barComparison);
```

**Résultat** :
- **-70% de code** (35 lignes → 11 lignes)
- **Theme-aware** automatique
- **Maintenance facilitée**

---

## Support

Pour questions ou problèmes, voir :
- `static/core/chart-config.js` - Code source
- `docs/UI_IMPROVEMENT_PLAN.md` - Plan d'amélioration complet
- Chart.js docs: https://www.chartjs.org/docs/

---

**Créé le 16 Décembre 2025**
**Version 1.0**

# Cycle Phase Presets - Integration dans SimControls

**Date**: 2025-09-30
**Status**: [OK] Complété
**Objectif**: Intégrer les presets de phases de cycle dans le panneau de contrôle existant pour créer un Analytics Playground.

---

## Résumé

Transformation du simulateur en **Analytics Playground** permettant de faire du What-If Analysis sur **données réelles de wallet** avec des scénarios de phases de marché prédéfinis.

**Principe**: "Et si on entrait en bear market demain avec MON wallet actuel ?"

---

## Architecture Finale

### Composants
1. **`static/presets/cycle_phase_presets.json`** (190 lignes)
   - 10 scénarios de phases de marché (bull, bear, euphorie, capitulation, etc.)
   - Overrides pour cycle_score, onchain_score, risk_score, contradiction_pct
   - Résultats attendus (DI range, stables target)

2. **`static/components/SimControls.js`** (modifié)
   - Chargement dual: presets simulation + presets cycle
   - Dropdown avec 2 optgroups:
     - Presets Simulation (existants)
     - What-If Scenarios (nouveaux)
   - Méthodes: `loadCyclePreset()` + `loadSimPreset()`

3. **`static/simulations.html`** (nettoyé)
   - Suppression du dropdown séparé créé par erreur
   - Suppression des fonctions `loadCyclePresets()` et `applyPreset()` standalone
   - Store initialization intégré (lignes 1187-1250)

---

## Modifications Détaillées

### 1. SimControls.js

#### Chargement Dual des Presets (lignes 579-633)
```javascript
async loadPresets() {
  // Charger presets de simulation (existants)
  const simResponse = await fetch('./presets/sim_presets.json');
  const simPresets = await simResponse.json();

  // Charger presets de phases de cycle (nouveaux)
  let cyclePresets = { presets: [] };
  try {
    const cycleResponse = await fetch('./presets/cycle_phase_presets.json');
    cyclePresets = await cycleResponse.json();
  } catch (error) {
    console.warn(' SIM: Cycle phase presets not available:', error);
  }

  const select = document.getElementById('sim-preset-select');
  if (select) {
    // Groupe 1: Presets Simulation
    if (simPresets.presets && simPresets.presets.length > 0) {
      const simGroup = document.createElement('optgroup');
      simGroup.label = ' Presets Simulation';
      simPresets.presets.forEach((preset, index) => {
        const option = document.createElement('option');
        option.value = `sim:${index}`;
        option.textContent = `${preset.name} - ${preset.desc}`;
        simGroup.appendChild(option);
      });
      select.appendChild(simGroup);
    }

    // Groupe 2: What-If Scenarios
    if (cyclePresets.presets && cyclePresets.presets.length > 0) {
      const cycleGroup = document.createElement('optgroup');
      cycleGroup.label = ' What-If Scenarios (Phases Marché)';
      cyclePresets.presets.forEach((preset, index) => {
        if (preset.id !== 'current') {
          const option = document.createElement('option');
          option.value = `cycle:${index}`;
          option.textContent = preset.name;
          option.title = preset.description;
          cycleGroup.appendChild(option);
        }
      });
      select.appendChild(cycleGroup);
    }
  }

  this.presets = simPresets.presets || [];
  this.cyclePresets = cyclePresets.presets || [];
}
```

#### Dispatcher de Presets (lignes 635-644)
```javascript
loadPreset(presetValue) {
  // Parser le type de preset (sim:X ou cycle:X)
  const [presetType, presetIndex] = presetValue.split(':');

  if (presetType === 'sim') {
    this.loadSimPreset(parseInt(presetIndex, 10));
  } else if (presetType === 'cycle') {
    this.loadCyclePreset(parseInt(presetIndex, 10));
  }
}
```

#### Chargement Preset Cycle (lignes 674-716)
```javascript
loadCyclePreset(presetIndex) {
  const preset = this.cyclePresets[presetIndex];
  if (!preset) return;

  this.isLoadingPreset = true;
  this.activePresetIndex = `cycle:${presetIndex}`;

  // Appliquer les overrides sur les données actuelles
  const overrides = preset.overrides || {};

  // Mapper les overrides vers le state du simulateur
  if (overrides.cycle_score !== undefined) {
    this.state.cycleScore = overrides.cycle_score;
  }
  if (overrides.onchain_score !== undefined) {
    this.state.onChainScore = overrides.onchain_score;
  }
  if (overrides.risk_score !== undefined) {
    this.state.riskScore = overrides.risk_score;
  }
  if (overrides.contradiction_pct !== undefined) {
    this.state.contradictionPenalty = overrides.contradiction_pct / 100;
  }

  // Stocker les infos du preset pour affichage
  this.state.presetInfo = {
    name: preset.name,
    desc: preset.description || '',
    type: 'cycle',
    expected: preset.expected || {}
  };

  this.updateUI();
  this.isLoadingPreset = false;
  this.debouncedUpdate();

  console.log(' SIM: cyclePresetLoaded -', {
    name: preset.name,
    id: preset.id,
    overrides
  });
}
```

### 2. simulations.html - Nettoyage

#### Supprimé (lignes 545-551 - ancien code)
```html
<!-- [Error] SUPPRIMÉ: Dropdown séparé créé par erreur -->
<div style="display: flex; align-items: center; gap: 8px; margin-top: 8px;">
  <label for="cycle-preset-select"> Scénario:</label>
  <select id="cycle-preset-select">
    <option value="">Charger les presets...</option>
  </select>
  <button id="apply-preset-btn"> Appliquer</button>
</div>
```

#### Supprimé (lignes 905-973 - ancien code)
```javascript
// [Error] SUPPRIMÉ: Fonctions standalone redondantes
let cyclePresets = null;
async function loadCyclePresets() { /* ... */ }
async function applyPreset() { /* ... */ }
```

#### Supprimé (ligne 912 - ancien code)
```javascript
// [Error] SUPPRIMÉ: Appel redondant
await loadCyclePresets();
```

#### Supprimé (lignes 979-980 - ancien code)
```javascript
// [Error] SUPPRIMÉ: Event listener redondant
document.getElementById('apply-preset-btn')?.addEventListener('click', applyPreset);
```

---

## Utilisation

### 1. Ouvrir le Simulateur
```
http://localhost:8080/static/simulations.html
```

### 2. Ouvrir le Panneau de Contrôle
Cliquer sur " Contrôles" (flyout à droite)

### 3. Sélectionner un Preset
Dans le dropdown en haut du panneau:
- ** Presets Simulation**: Scénarios complets avec tous paramètres
- ** What-If Scenarios**: Phases de marché appliquées aux données réelles

### 4. Exemples de Presets Cycle
- Bull Run - Début (cycle 70, onchain 55, risk 45)
- Bull Run - Euphorie (cycle 95, onchain 85, contradiction 60%)
- Bear Market - Début (cycle 30, onchain 35, risk 65)
- Capitulation (cycle 5, onchain 10, risk 90)
- Marché Latéral (cycle 50, onchain 45, risk 50)

### 5. Résultat
- Les sliders se mettent à jour automatiquement
- Le simulateur recalcule avec les overrides
- Les KPI affichent les nouveaux résultats
- Comparaison AVANT ↔ APRÈS (à implémenter)

---

## Roadmap

### Phase 1: Infrastructure  (Complété)
- [x] Créer `cycle_phase_presets.json` avec 10 scénarios
- [x] Intégrer chargement dans `SimControls.loadPresets()`
- [x] Dispatcher `sim:X` vs `cycle:X`
- [x] Mapper overrides vers state
- [x] Nettoyer UI redondante dans simulations.html

### Phase 2: Application Complète (À faire)
- [ ] Implémenter `applyCyclePresetToStore()` dans simulations.html
- [ ] Injecter overrides dans `window.store` avant `getUnifiedState()`
- [ ] Capturer AVANT (baseline) vs APRÈS (preset)
- [ ] Afficher deltas et color coding

### Phase 3: UX Avancée (À faire)
- [ ] Split-screen AVANT ↔ APRÈS
- [ ] Indicateurs de changement (▲▼ +X% / -X%)
- [ ] Recommandations basées sur résultats
- [ ] Export des comparaisons

---

## Notes Techniques

### Isolation Multi-Tenant
Les presets cycle s'appliquent sur les **données réelles de l'utilisateur actif** (`localStorage.getItem('activeUser')`).

Le store est initialisé avec:
```javascript
window.store.set('wallet.balances', balances); // Données user réelles
window.store.set('cycle.score', cycleScore);   // Override preset
```

### Fallback Gracieux
Si `cycle_phase_presets.json` est absent, le système continue avec les presets de simulation uniquement (pas d'erreur bloquante).

### Performance
- Les presets sont chargés **une seule fois** au démarrage
- Le changement de preset est instantané (pas de fetch réseau)
- Debounce 200ms pour éviter calculs multiples

---

## Références

- **Spec complète**: [docs/ANALYTICS_PLAYGROUND_SPEC.md](./ANALYTICS_PLAYGROUND_SPEC.md)
- **Presets JSON**: [static/presets/cycle_phase_presets.json](../static/presets/cycle_phase_presets.json)
- **Composant**: [static/components/SimControls.js](../static/components/SimControls.js)
- **Page**: [static/simulations.html](../static/simulations.html)

---

## Lessons Learned

### Erreur Initiale
Création d'un nouveau dropdown séparé au lieu de réutiliser le composant existant.

**User feedback**: "attends, il faut réutiliser le panneau contrôle pour les presets qui sont déjà dedans. Ne refait pas tout, inspires-toi de ce qui existe !"

### Solution
Intégration propre dans `SimControls` avec:
- Chargement centralisé
- Optgroups pour séparation visuelle
- Réutilisation du système de state existant
- Zero duplication de code

### Principe
**Toujours auditer le code existant avant d'ajouter des fonctionnalités.**

Si un composant existe déjà pour une tâche similaire → l'étendre au lieu de recréer.


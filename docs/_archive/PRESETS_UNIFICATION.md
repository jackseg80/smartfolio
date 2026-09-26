# Presets Unification - Version 2.0

**Date**: 2025-09-30
**Status**: [OK] Complété
**Migration**: cycle_phase_presets.json → sim_presets.json (unifié)

---

## Résumé

Unification des presets de simulation et des presets de phases de cycle dans un seul fichier avec:
- **Noms courts** et **emojis** pour UX rapide
- **Tooltips détaillés** au survol pour contexte additionnel
- **Paramètres complets** (governance, risk_budget, execution, etc.)

---

## Structure Unifiée

### Fichier: `static/presets/sim_presets.json`

**Format**:
```json
{
  "version": "2.0",
  "presets": [
    {
      "name": " Bull Run - Début",
      "desc": "Optimisme modéré, risque contrôlé",
      "tooltip": "Début de bull market: rotation progressive vers alts quality, stables réduits prudemment",
      "inputs": { /* cycle, onchain, risk scores */ },
      "regime_phase": { /* phase engine config */ },
      "risk_budget": { /* min/max stables, circuit breakers */ },
      "governance": { /* caps par groupe */ },
      "execution": { /* thresholds, slippage */ }
    }
  ]
}
```

---

## 12 Presets Unifiés

### 1.  Bull Run - Début
- **Cycle**: 70, **Onchain**: 55, **Risk**: 45
- **Tooltip**: Début de bull market: rotation progressive vers alts quality, stables réduits prudemment
- **Phase Engine**: Shadow mode
- **Stables**: 15-55%

### 2.  Bull Run - Euphorie
- **Cycle**: 75, **Onchain**: 82, **Risk**: 65
- **Tooltip**: Transition euphorie → distribution: rotation défensive prudente, premiers signaux de top
- **Phase Engine**: Shadow mode
- **Stables**: 15-65%

### 3.  ETH Expansion
- **Cycle**: 68, **Onchain**: 72, **Risk**: 38
- **Tooltip**: Phase d'expansion ETH: L2/Scaling surperforment, rotation vers écosystème Ethereum
- **Phase Engine**: Apply mode (eth_expansion forcé)
- **Stables**: 8-45%
- **Governance**: L2 cap 20%, ETH max 40%

### 4.  Altseason - Large Caps
- **Cycle**: 78, **Onchain**: 74, **Risk**: 35
- **Tooltip**: Altseason concentré sur L1 majors (SOL, AVAX) et top 50, BTC/ETH stagnent
- **Phase Engine**: Apply mode (largecap_alt forcé)
- **Stables**: 6-40%

### 5.  Altseason - Complet
- **Cycle**: 88, **Onchain**: 85, **Risk**: 25
- **Tooltip**: Altseason maximal: memecoins et Others explosent, euphorie irrationnelle, danger élevé
- **Phase Engine**: Apply mode (full_altseason forcé)
- **Stables**: 5-35%
- **Governance**: Memes cap 15%, Others cap 8%

### 6.  Crash Imminent
- **Cycle**: 92, **Onchain**: 88, **Risk**: 90
- **Tooltip**: Signaux de top: cycle très haut mais contradictions explosent, sortie d'urgence recommandée
- **Phase Engine**: Apply mode (risk_off forcé)
- **Stables**: 35-85%
- **Contradiction Penalty**: 0.35 (très élevé)

### 7.  Bear Market - Début
- **Cycle**: 35, **Onchain**: 28, **Risk**: 75
- **Tooltip**: Début de bear market: réduction progressive des risques, rotation vers qualité et stables
- **Phase Engine**: Apply mode (risk_off forcé)
- **Stables**: 20-70%

### 8.  Capitulation
- **Cycle**: 15, **Onchain**: 12, **Risk**: 88
- **Tooltip**: Marché en capitulation: maximum stables, circuit-breakers actifs, focus préservation capital
- **Phase Engine**: Apply mode (risk_off forcé)
- **Stables**: 30-80%
- **Market Overlays**: Vol Z 3.8, DD -45%

### 9.  Dead Cat Bounce
- **Cycle**: 42, **Onchain**: 35, **Risk**: 70
- **Tooltip**: Rebond technique dans un bear: prudence maintenue, faible confiance, pas de FOMO
- **Phase Engine**: Off
- **Stables**: 25-65%
- **Confiances**: Très faibles (0.3-0.5)

### 10.  Reprise Post-Bear
- **Cycle**: 45, **Onchain**: 55, **Risk**: 48
- **Tooltip**: Sortie de bear market: accumulation progressive, onchain repart, allocation prudente
- **Phase Engine**: Shadow mode
- **Stables**: 18-60%

### 11. ₿ BTC Season
- **Cycle**: 65, **Onchain**: 58, **Risk**: 45
- **Tooltip**: BTC surperforme, alts stagnent: rotation défensive vers BTC en préparation altseason
- **Phase Engine**: Off
- **Governance**: BTC max 65%, ETH max 25%

### 12. ↔️ Marché Latéral
- **Cycle**: 52, **Onchain**: 48, **Risk**: 55
- **Tooltip**: Marché en range: hystérésis active pour éviter over-trading, pas de tendance claire
- **Phase Engine**: Off
- **Stables**: 15-55%
- **Execution**: Thresholds élevés (3.5% / 2.0%) pour réduire over-trading

---

## Changements Techniques

### SimControls.js

**Avant (v1)**: Deux sources de presets
```javascript
// Charger sim_presets.json
// Charger cycle_phase_presets.json
// Créer 2 optgroups dans le dropdown
// Router vers loadSimPreset() ou loadCyclePreset()
```

**Après (v2)**: Une seule source unifiée
```javascript
async loadPresets() {
  const response = await fetch('./presets/sim_presets.json');
  const data = await response.json();

  data.presets.forEach((preset, index) => {
    const option = document.createElement('option');
    option.value = index;
    option.textContent = `${preset.name} - ${preset.desc}`;
    option.title = preset.tooltip; // Tooltip au survol
    select.appendChild(option);
  });
}

loadPreset(presetIndex) {
  const preset = this.presets[presetIndex];
  // Charger tous les paramètres
  this.state = {
    ...preset.inputs,
    phaseEngine: preset.regime_phase,
    riskBudget: preset.risk_budget,
    governance: preset.governance,
    execution: preset.execution,
    presetInfo: {
      name: preset.name,
      desc: preset.desc,
      tooltip: preset.tooltip
    }
  };
}
```

### Fichiers Supprimés

- [Error] `static/presets/cycle_phase_presets.json` (190 lignes)
  - Contenu fusionné dans `sim_presets.json`
  - Pas de perte d'information

---

## UX Améliorée

### Dropdown des Presets

**Avant**:
```
┌─ Presets ─────────────────────────────┐
│  Presets Simulation                 │
│   ├─ Fin de Bull Run - Transition...  │
│   ├─ Early Bear - Début de bear...    │
│   └─ ...                               │
│                                        │
│  What-If Scenarios (Phases Marché)  │
│   ├─  Bull Run - Début              │
│   ├─  Bull Run - Euphorie           │
│   └─ ...                               │
└────────────────────────────────────────┘
```

**Après**:
```
┌─ Presets ─────────────────────────────┐
│  Bull Run - Début - Optimisme modéré │
│  Bull Run - Euphorie - Pic du bull   │
│  ETH Expansion - L2 en feu           │
│  Altseason - Large Caps              │
│  Altseason - Complet                 │
│ [Warning] Crash Imminent - Danger extrême     │
│  Bear Market - Début                 │
│  Capitulation - Mode survie          │
│  Dead Cat Bounce - Rebond temporaire │
│  Reprise Post-Bear - Premiers signes │
│ ₿ BTC Season - Dominance BTC forte     │
│ ↔️ Marché Latéral - Range              │
└────────────────────────────────────────┘
```

### Tooltips au Survol

Quand on passe la souris sur un preset:
```
 Bull Run - Début - Optimisme modéré
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Début de bull market: rotation progressive
vers alts quality, stables réduits prudemment
```

---

## Migration

### Pour les Développeurs

Si vous avez du code qui référence `cycle_phase_presets.json`:

**Avant**:
```javascript
const response = await fetch('./presets/cycle_phase_presets.json');
const data = await response.json();
const preset = data.presets.find(p => p.id === 'bull_early');
```

**Après**:
```javascript
const response = await fetch('./presets/sim_presets.json');
const data = await response.json();
const preset = data.presets.find(p => p.name.includes('Bull Run - Début'));
// Ou par index si connu
const preset = data.presets[0]; // Premier preset
```

### Structure de Données

**Avant (cycle_phase_presets.json)**:
```json
{
  "id": "bull_early",
  "name": " Bull Run - Début",
  "description": "Début de bull market: optimisme modéré",
  "overrides": {
    "cycle_score": 70,
    "onchain_score": 55
  }
}
```

**Après (sim_presets.json v2)**:
```json
{
  "name": " Bull Run - Début",
  "desc": "Optimisme modéré, risque contrôlé",
  "tooltip": "Début de bull market: rotation progressive vers alts quality",
  "inputs": {
    "cycleScore": 70,
    "onChainScore": 55,
    "riskScore": 45
  },
  "risk_budget": { /* complet */ },
  "governance": { /* complet */ }
}
```

---

## Avantages de l'Unification

### Avantages

1. **Single Source of Truth**: Un seul fichier à maintenir
2. **UX Simplifiée**: Pas de séparation artificielle entre "simulation" et "what-if"
3. **Tooltips Informatifs**: Contexte détaillé sans surcharger l'UI
4. **Paramètres Complets**: Tous les presets ont governance/execution/risk_budget
5. **Noms Courts**: Liste plus compacte dans le dropdown
6. **Emojis Visuels**: Identification rapide par symbole

### Métriques

- **Fichiers**: 2 → 1 (-50%)
- **Lignes de code SimControls**: 75 → 30 (-60%)
- **Presets totaux**: 10 anciens + 9 nouveaux = 12 unifiés (dédupliqués)
- **Champs par preset**: 5 → 8 (+60% d'informations)

---

## Roadmap Future

### Phase 1  (Complété)
- [x] Unifier les fichiers JSON
- [x] Simplifier le code de chargement
- [x] Ajouter tooltips
- [x] Supprimer l'ancien fichier

### Phase 2 (Optionnel)
- [ ] Ajouter des tags filtres (bull/bear/sideways)
- [ ] Preset search/filter dans le dropdown
- [ ] Preset favoris (localStorage)
- [ ] Comparaison entre 2 presets côte à côte

### Phase 3 (Optionnel)
- [ ] Preset import/export utilisateur
- [ ] Preset sharing via URL
- [ ] Historique des presets utilisés

---

## Références

- **Fichier unifié**: [static/presets/sim_presets.json](../static/presets/sim_presets.json)
- **Composant**: [static/components/SimControls.js:579-631](../static/components/SimControls.js#L579-L631)
- **Doc intégration**: [CYCLE_PRESETS_INTEGRATION.md](./CYCLE_PRESETS_INTEGRATION.md)
- **Spec Analytics**: [ANALYTICS_PLAYGROUND_SPEC.md](./ANALYTICS_PLAYGROUND_SPEC.md)
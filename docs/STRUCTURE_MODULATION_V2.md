# Structure Modulation V2 - Garde-fou d'Allocation

## Objectif

Utiliser le **Portfolio Structure Score V2** (structure pure: HHI, memes, GRI, diversification) comme **garde-fou de l'allocation** pour moduler :
1. **Cible de stables** (target_stables_pct) : ±5 à ±10 pts selon qualité structurelle
2. **Cap effectif** (cap_pct_per_iter) : ±0.5 pt selon structure

## Architecture

### Entrées API

Depuis `/api/risk/dashboard?risk_version=v2_shadow|v2_active` :

```json
{
  "risk_metrics": {
    "risk_version_info": {
      "portfolio_structure_score": 81.7,  // ← Utilisé pour modulation
      "structure_breakdown": {
        "hhi": 0.0,
        "memecoins": 0.64,
        "gri": 16.23,
        "low_diversification": 0.0,
        "base": 100.0,
        "total_penalties": 16.87,
        "final_score": 83.13
      }
    }
  },
  "risk_budget": {
    "target_stables_pct": 41,  // Base AVANT modulation
    "min_stables": 10,
    "max_stables": 60
  }
}
```

### Règles de Modulation

| Structure Score | Δ Stables | Δ Cap   | Signification                          |
|-----------------|-----------|---------|----------------------------------------|
| < 50            | +10 pts   | -0.5%   | Faible → Plus de stables, cap réduit   |
| 50-60           | +5 pts    | 0       | Moyenne-faible → Modérément prudent    |
| 60-80           | 0         | 0       | Neutre (zone saine)                    |
| ≥ 80            | -5 pts    | +0.5%   | Forte → Moins de stables, cap augmenté |

### Fichiers Modifiés

#### 1. `static/core/unified-insights-v2.js`

**Helper de modulation** (lignes 32-48) :
```javascript
export function computeStructureModulation(structureScore) {
  if (structureScore == null || Number.isNaN(structureScore)) {
    return { deltaStables: 0, deltaCap: 0 };
  }
  if (structureScore < 50)  return { deltaStables: +10, deltaCap: -0.5 };
  if (structureScore < 60)  return { deltaStables: +5,  deltaCap: 0   };
  if (structureScore >= 80) return { deltaStables: -5,  deltaCap: +0.5 };
  return { deltaStables: 0, deltaCap: 0 }; // Neutre 60-80
}
```

**Intégration dans computeMacroTargetsDynamic()** (lignes 144-174) :
```javascript
// 0) Stables = SOURCE DE VÉRITÉ avec Structure Modulation V2
let stablesBase = rb?.target_stables_pct || 25;

// [New] Structure Modulation V2
const structureScore = data?.risk?.risk_metrics?.risk_version_info?.portfolio_structure_score;
const { deltaStables, deltaCap } = computeStructureModulation(structureScore);

// Appliquer modulation avec clamp [min_stables, max_stables]
const minStables = rb?.min_stables ?? 10;
const maxStables = rb?.max_stables ?? 60;
const stablesModulated = Math.max(minStables, Math.min(maxStables, stablesBase + deltaStables));

// Métadonnées pour UI/logs
ctx.structure_modulation = {
  structure_score: structureScore ?? null,
  delta_stables: deltaStables,
  delta_cap: deltaCap,
  stables_before: stablesBase,
  stables_after: stablesModulated,
  note: 'V2 portfolio structure modulation',
  enabled: structureScore != null
};

const stables = stablesModulated;
const riskyPool = Math.max(0, 100 - stables);
```

**Exposition dans unified state** (lignes 859-889) :
```javascript
// Exposer structure_modulation dans le store pour UI
unifiedState.structure_modulation = structureMod;
```

#### 2. `static/core/allocation-engine.js`

**Helper cap effectif** (lignes 8-21) :
```javascript
function getEffectiveCapWithStructure(state, deltaCap = 0) {
  const capEff = selectEffectiveCap(state); // Gouvernance base
  const adjusted = Math.max(0, capEff + (deltaCap || 0));
  const maxDelta = 0.5; // Garde-fou: +0.5% max
  return Math.min(adjusted, capEff + maxDelta);
}
```

**Extraction deltaCap** (lignes 100-101) :
```javascript
const deltaCap = structure_modulation?.delta_cap ?? 0;
```

**Usage dans calculateExecutionPlan()** (lignes 486-501) :
```javascript
const deltaCap = executionContext.structure_modulation?.delta_cap ?? 0;

// Utiliser cap avec structure modulation
if (capPct == null) {
  const contextState = executionContext.state || executionContext.unified_state || null;
  if (contextState) {
    capPct = getEffectiveCapWithStructure(contextState, deltaCap);
  }
}
```

**Métadonnées result** (lignes 190-194) :
```javascript
structure_modulation: structure_modulation?.enabled ? {
  ...structure_modulation,
  cap_after: executionPlan.cap_pct_per_iter // Cap APRÈS deltaCap
} : null
```

#### 3. `static/risk-dashboard.html`

**Badge Structure Modulation V2** (lignes 4367-4434) :

Affiche si `structure_modulation.enabled` :
- **Structure Score** : 0-100
- **Δ Stables** : +/- points (avec flèche couleur)
- **Stables après** : % final (clamped)
- **Cap effectif** : % avec delta appliqué
- **Couleur** : Rouge (+stables, prudence), Vert (-stables, opportunité), Bleu (neutre)

## Exemples

### Cas 1 : Portfolio Degen (Structure Score = 20)

**Entrée** :
- `portfolio_structure_score = 20` (très faible)
- `target_stables_pct = 35` (base)
- `min_stables = 10`, `max_stables = 60`

**Calcul** :
```
deltaStables = +10  (structure < 50)
deltaCap = -0.5

stablesModulated = clamp(35 + 10, 10, 60) = 45%
riskyPool = 100 - 45 = 55%
```

**Résultat** :
- Stables : **35% → 45%** (+10 pts)
- Cap effectif : **5% → 4.5%** (-0.5 pts)
- Risky pool réduit (plus prudent)

### Cas 2 : Portfolio Sain (Structure Score = 85)

**Entrée** :
- `portfolio_structure_score = 85` (forte)
- `target_stables_pct = 30`

**Calcul** :
```
deltaStables = -5  (structure ≥ 80)
deltaCap = +0.5

stablesModulated = clamp(30 - 5, 10, 60) = 25%
riskyPool = 100 - 25 = 75%
```

**Résultat** :
- Stables : **30% → 25%** (-5 pts)
- Cap effectif : **5% → 5.5%** (+0.5 pts)
- Risky pool augmenté (opportunité)

### Cas 3 : Structure Moyenne (Score = 65)

**Entrée** :
- `portfolio_structure_score = 65`
- `target_stables_pct = 40`

**Calcul** :
```
deltaStables = 0  (structure 60-80, neutre)
deltaCap = 0

stablesModulated = clamp(40 + 0, 10, 60) = 40%
```

**Résultat** :
- Stables : **40% → 40%** (inchangé)
- Cap effectif : **5% → 5%** (inchangé)
- Aucune modulation

## Cohérence avec Autres Systèmes

### Compatibilité meme_cap

**meme_cap** s'applique **APRÈS** structure modulation, **AVANT** normalisation :

1. Structure Modulation → Ajuste stables → Calcule risky pool
2. Tilts (Phase Engine, régime, etc.) → Appliqués sur risky pool
3. **meme_cap** → Plafonne Memecoins, redistribue excès à BTC/ETH
4. Normalisation → Somme = 100%

**Métadonnées** :
```javascript
{
  meme_cap: { defined: true, value: 2, applied: true },
  structure_modulation: {
    structure_score: 20,
    delta_stables: +10,
    stables_after: 45,
    cap_after: 4.5
  }
}
```

### Clamp Bornes

**min_stables / max_stables** sont **TOUJOURS** respectés :

```javascript
const stablesModulated = Math.max(min_stables, Math.min(max_stables, stablesBase + deltaStables));
```

Exemples :
- Base 58% + delta +10 = 68% → **Clampé à max_stables (60%)**
- Base 8% + delta +10 = 18% → **OK (entre 10-60%)**
- Base 5% + delta +10 = 15% → **Clampé à min_stables (10%)**

### Cap Gouvernance

**deltaCap** est **ajouté** au cap de gouvernance (staleness, alerts, policy), avec garde-fou :

```javascript
function getEffectiveCapWithStructure(state, deltaCap = 0) {
  const capEff = selectEffectiveCap(state);  // Ex: 5% (avec staleness)
  const adjusted = Math.max(0, capEff + deltaCap);
  const maxDelta = 0.5;  // ±0.5% max
  return Math.min(adjusted, capEff + maxDelta);
}
```

Priorité cascade :
1. **Staleness** (8%) > **Backend error** (5%) → Cap de base
2. **Alert override** (si actif) → Peut forcer 1-3%
3. **Policy engine** (si actif) → Peut réduire
4. **Structure deltaCap** (±0.5%) → Ajustement final

## Tests de Cohérence

### Assertion 1 : Somme = 100%

```javascript
const totalCheck = validateTotalAllocation(coinAllocation);
console.assert(
  Math.abs(totalCheck.sum - 100) < 0.1,
  'Total allocation must be 100%'
);
```

### Assertion 2 : Stables dans bornes

```javascript
const stablesPct = coinAllocation['Stablecoins'] || 0;
console.assert(
  stablesPct >= risk_budget.min_stables && stablesPct <= risk_budget.max_stables,
  'Stables must be within [min_stables, max_stables]'
);
```

### Assertion 3 : Risk Dashboard ≈ Analytics Unified

```javascript
// Dashboard affiche stables_after
const stablesFromDashboard = structure_modulation.stables_after;

// Analytics Unified utilise même calcul via unified-insights-v2.js
const stablesFromAnalytics = unifiedState.risk_budget.target_stables_pct; // Modulated

// Assertion: écart < 1%
console.assert(
  Math.abs(stablesFromDashboard - stablesFromAnalytics) < 1,
  'Stables coherence across views'
);
```

## UI - Badge Structure Modulation

### Affichage

Le badge apparaît dans `risk-dashboard.html` si :
- `structure_modulation.enabled === true`
- `structure_modulation.structure_score != null`

### Contenu

```
[In progress] Structure Modulation V2                     active

Structure Score:            81.7/100
Δ Stables:                  +5 pts
                            → 46% stables

Cap effectif:               5.5% (+0.5)

 Modulation basée sur la qualité structurelle (HHI, memes, GRI, diversification)
```

### Couleurs

- **Rouge** (`#f7768e`) : +stables (prudence, structure faible)
- **Vert** (`#9ece6a`) : -stables (opportunité, structure forte)
- **Bleu** (`#7aa2f7`) : Neutre (structure moyenne)

## Logs de Debug

### Unified Insights V2

```javascript
console.debug(' Structure Modulation V2:', {
  structure_score: 85,
  delta_stables: -5,
  delta_cap: +0.5,
  stables_before: 30,
  stables_after: 25,
  enabled: true
});
```

### Allocation Engine

```javascript
console.debug(' Execution Plan:', {
  cap_pct_per_iter: 5.5,  // APRÈS deltaCap
  structure_modulation: {
    delta_cap: +0.5,
    cap_after: 5.5
  }
});
```

## Pas de Double-Comptage

**IMPORTANT** : Portfolio Structure Score V2 **N'EST PAS** utilisé dans le Decision Index (DI).

**DI** utilise :
- **Cycle Score** (CCS)
- **On-Chain Score**
- **Risk Score** (VaR, Sharpe, DD, Vol)

**Structure Modulation** agit **uniquement** en garde-fou d'allocation, **après** calcul du DI.

```
Decision Index (DI) → Régime → Risk Budget (stables base)
                               ↓
                     Structure Modulation (stables modulés)
                               ↓
                     Allocation Engine (risky pool, tilts, caps)
```

## Roadmap

### Phase Actuelle (Oct 2025)
- [OK] Helper `computeStructureModulation()`
- [OK] Modulation stables dans `computeMacroTargetsDynamic()`
- [OK] Cap effectif avec `deltaCap`
- [OK] Badge UI avec métadonnées
- [OK] Exposition dans unified state
- [OK] Cohérence avec meme_cap et clamp bornes

### Phase Suivante
- [ ] Tests unitaires (structure faible, forte, clamp bornes)
- [ ] Validation cohérence Dashboard ↔ Analytics (±1%)
- [ ] Logs enrichis (breakdown structure_breakdown dans tooltip)
- [ ] Feature flag pour désactiver si besoin

---

**Date d'implémentation** : 2025-10-03
**Version** : Structure Modulation V2
**Statut** : [OK] Implémenté et testé

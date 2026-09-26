# Unified Insights v2 - Architecture & Conventions

## Date
2025-09-30

## Vue d'ensemble

`static/core/unified-insights-v2.js` est le module **production** de calcul du Decision Index (DI) et de gestion des poids adaptatifs. Il est utilisé par Analytics Unified et synchronisé avec le Simulateur (`simulation-engine.js`).

> [Warning] **Note Importante**: Le Decision Index (DI) est **différent** du "Score Décisionnel" (formule canonique 50/30/20).
> Voir [DECISION_INDEX_V2.md](DECISION_INDEX_V2.md) pour comprendre le système dual et les overrides.

---

## Architecture

### 1. Calcul du Decision Index

**Formule** :
```
DI = wCycle × scoreCycle + wOnchain × scoreOnchain + wRisk × scoreRisk
```

**[Warning] IMPORTANT — Sémantique Risk** :

> **[Warning] Règle Canonique — Sémantique Risk**
>
> Le **Risk Score** est un indicateur **positif** de robustesse, borné **[0..100]**.
>
> **Convention** : Plus haut = plus robuste (risque perçu plus faible).
>
> **Conséquence** : Dans le Decision Index (DI), Risk contribue **positivement** :
> ```
> DI = wCycle·scoreCycle + wOnchain·scoreOnchain + wRisk·scoreRisk
> ```
>
> ** Interdit** : Ne jamais inverser avec `100 - scoreRisk`.
>
> **Visualisation** : Contribution = `(poids × score) / Σ(poids × score)`
>
>  Source : [RISK_SEMANTICS.md](RISK_SEMANTICS.md)

### 2. Poids Adaptatifs

**Fonction** : `calculateAdaptiveWeights(scores, governance, options)`

**Poids de base** :
```javascript
const BASE_WEIGHTS = { wCycle: 0.5, wOnchain: 0.3, wRisk: 0.2 };
```

**Règles d'adaptation** :
1. **Boost Cycle ≥ 90** : `wCycle=0.65, wOnchain=0.25, wRisk=0.1`
2. **Boost Cycle ≥ 70** : `wCycle=0.55, wOnchain=0.28, wRisk=0.17`
3. **Pénalité Contradiction** (si `contradiction_index ≥ 0.5`) :
   - `wOnchain × 0.9`
   - `wRisk × 0.9`
   - Redistribution vers `wCycle`
4. **Normalisation finale** : Σ(poids) = 1.0

**Source Contradiction** : `governance.contradiction_index` (source primaire)

### 3. Scaling par Confidences

Après les poids adaptatifs, un **scaling par confidences** est appliqué :
```javascript
wCycle *= (0.8 + 0.4 * confidences.cycle);
wOnchain *= (0.8 + 0.4 * confidences.onchain);
wRisk *= (0.8 + 0.4 * confidences.risk);
```

Ceci est **intentionnel** : les poids adaptatifs forment la base, le scaling par confidence affine ensuite.

---

## Propagation des Poids à l'UI

**[Warning] CRITIQUE — Ne pas transformer les poids** :
- Les poids post-adaptatifs (wCycle, wOnchain, wRisk) doivent être **propagés tels quels** à l'UI
- **Erreur fréquente** : Inverser Risk avec `100 - scoreRisk` ou `1 - wRisk`
- **Correct** : Passer `{ cycle: wCycle, onchain: wOnchain, risk: wRisk }` directement

**Visualisation Contributions** :
```javascript
// Formule correcte
const total = wCycle * scoreCycle + wOnchain * scoreOnchain + wRisk * scoreRisk;
const contribCycle = (wCycle * scoreCycle) / total;
const contribOnchain = (wOnchain * scoreOnchain) / total;
const contribRisk = (wRisk * scoreRisk) / total; // PAS de 100 - scoreRisk
```

---

## Badges & Métadonnées

**Badges UI** (Confiance, Contradiction, Cap, Mode) :
- **Influencent les poids** via la politique d'adaptation
- **N'influencent PAS les scores bruts** (cycle, onchain, risk)

**Exemple** :
- Contradiction élevée → pénalise `wOnchain` et `wRisk` (réduction 10%)
- Cycle ≥ 90 → booste `wCycle` à 0.65
- Cap dynamique → appliqué par Execution Layer (pas dans le calcul DI)

---

## Modules Liés

- **`static/modules/simulation-engine.js`** : Réplique aligned avec unified-insights-v2 (voir [SIMULATION_ENGINE_ALIGNMENT.md](SIMULATION_ENGINE_ALIGNMENT.md))
- **`static/components/decision-index-panel.js`** : Visualisation contributions (barre empilée)
- **`static/core/allocation-engine.js`** : Utilise DI pour calculer les targets
- **`static/core/strategy-api-adapter.js`** : Adaptateur Strategy API v3

---

## Check-list QA

**Avant déploiement, vérifier** :
1. [OK] Aucune occurrence de `100 - risk` ou `100 - scoreRisk` dans le code
2. [OK] `calculateAdaptiveWeights` retourne poids normalisés (Σ = 1.0)
3. [OK] Poids post-adaptatifs propagés tels quels à l'UI (pas de transformation)
4. [OK] Contributions relatives calculées avec `(w × s) / Σ(w × s)` sans inversion Risk
5. [OK] Écart DI Analytics vs Simulateur < 0.1 avec mêmes inputs

**Validation** :
```javascript
// Test manuelle dans console
const scores = { cycle: 85, onchain: 70, risk: 60 };
const governance = { contradiction_index: 0.4 };
const weights = calculateAdaptiveWeights(scores, governance, {});
console.log('Weights sum:', weights.wCycle + weights.wOnchain + weights.wRisk); // Doit être 1.0

const DI = weights.wCycle * scores.cycle + weights.wOnchain * scores.onchain + weights.wRisk * scores.risk;
console.log('DI:', DI); // Comparer avec simulateur
```

---

## Références

- [docs/index.md — Sémantique de Risk](index.md#sémantique-de-risk-pilier-du-decision-index)
- [docs/architecture.md — Pilier Risk](architecture.md#pilier-risk-sémantique-et-propagation)
- [docs/SIMULATION_ENGINE_ALIGNMENT.md](SIMULATION_ENGINE_ALIGNMENT.md)
- [CLAUDE.md — Règles obligatoires](../CLAUDE.md)

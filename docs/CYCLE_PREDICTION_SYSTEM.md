# 📊 Système de Prédiction des Cycles Bitcoin

> Documentation technique du modèle de scoring et validation des cycles Bitcoin
> Dernière mise à jour: Janvier 2026

---

## 🎯 Vue d'Ensemble

Le système de prédiction des cycles Bitcoin utilise un **modèle double-sigmoïde** pour calculer un score de cycle (0-100) basé sur le nombre de mois écoulés depuis le dernier halving. Ce score est utilisé pour:

1. **Détection de phase** - Identifier la phase actuelle du cycle (accumulation, bull, peak, bear, pré-accumulation)
2. **Allocation dynamique** - Ajuster les allocations d'actifs selon la phase
3. **Blending avec CCS** - Combiner avec le Crypto Composite Score pour une vision hybride
4. **Prédiction de timing** - Estimer la position dans le cycle actuel

---

## 📐 Modèle Mathématique

### Formule Double-Sigmoïde

Le score de cycle est calculé par le produit de deux fonctions sigmoïdes:

```javascript
rise = 1 / (1 + exp(-k_rise × (months - m_rise_center)))
fall = 1 / (1 + exp(-k_fall × (m_fall_center - months)))
base = rise × fall
score = (base ^ p_shape) × 100
```

**Paramètres par défaut optimisés (Jan 2026):**

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `m_rise_center` | 7.0 | Centre de la montée (mois) |
| `m_fall_center` | 30.0 | Centre de la descente (mois) |
| `k_rise` | 1.0 | Pente de montée (vitesse) |
| `k_fall` | 0.9 | Pente de descente (vitesse) |
| `p_shape` | 0.9 | Exposant de forme (douceur) |
| `floor` | 0 | Score minimum |
| `ceil` | 100 | Score maximum |

### Caractéristiques du Modèle

- **Forme en cloche** - Le score monte puis descend sur un cycle de ~48 mois
- **Peak théorique** - Maximum atteint entre 15-18 mois post-halving
- **Modularité 48 mois** - `months % 48` pour gérer les cycles successifs
- **Symétrie adaptable** - Les pentes rise/fall peuvent être ajustées indépendamment

---

## 🔄 Calibration Automatique

Le système implémente une **calibration automatique** basée sur les cycles historiques pour optimiser les paramètres.

### Ancres Historiques

```javascript
Cycle 1: Halving 2012-11-28 → Peak 2013-11-30 (~12 mois) → Bottom 2015-01-14 (~26 mois)
Cycle 2: Halving 2016-07-09 → Peak 2017-12-17 (~17 mois) → Bottom 2018-12-15 (~29 mois)
Cycle 3: Halving 2020-05-11 → Peak 2021-11-10 (~18 mois) → Bottom 2022-11-21 (~30 mois)
Cycle 4: Halving 2024-04-20 → En cours
```

### Objectifs de Calibration

Le modèle cherche à minimiser l'erreur quadratique sur:
1. **Pics = 100** - Score devrait être ~100 aux dates de pics historiques
2. **Bottoms = 10** - Score devrait être ~10 aux dates de creux
3. **Early phase = 5** - Score devrait être ~5 à 2 mois post-halving

### Grid Search

**Plages de recherche étendues (Jan 2026):**

```javascript
m_rise:  [5, 6, 7, 8, 9, 10, 11, 12]  // Étendu pour Cycle 1 précoce
m_fall:  [24, 26, 28, 30, 32, 34]     // Étendu pour flexibilité
k_rise:  [0.7, 0.8, 0.9, 1.0, 1.2, 1.4]
k_fall:  [0.7, 0.8, 0.9, 1.0, 1.2]
p_shape: [0.8, 0.85, 0.9, 1.0, 1.15, 1.3]
```

**Total combinaisons:** 8 × 6 × 6 × 5 × 6 = **8,640 configurations testées**

### Persistance

Les paramètres calibrés sont sauvegardés dans `localStorage`:
```javascript
{
  params: { m_rise_center, m_fall_center, k_rise, k_fall, p_shape },
  timestamp: Date.now(),
  version: '1.0'
}
```

**TTL:** 24 heures (recalibration automatique si plus ancien)

---

## 📈 Phases du Cycle

Le modèle définit 5 phases basées sur le score et les mois post-halving:

| Phase | Mois | Score Typique | Emoji | Stratégie |
|-------|------|---------------|-------|-----------|
| **Accumulation** | 0-6 | 0-30 | 🟡 | BTC/ETH focus, alts réduits |
| **Bull Build** | 7-18 | 30-90 | 🟢 | Montée progressive, alts augmentent |
| **Peak/Euphoria** | 19-24 | 80-100 | 🟣 | Alt season, attention au top |
| **Bear Market** | 25-36 | 10-40 | 🔴 | Stables max, alts réduits fortement |
| **Pré-Accumulation** | 37-48 | 5-20 | ⚫ | Retour progressif BTC/ETH |

---

## 🎲 Multiplicateurs d'Allocation

Chaque phase applique des multiplicateurs aux classes d'actifs:

### Exemple: Bull Build Phase

```javascript
{
  BTC: 1.2,           // +20% vs allocation de base
  ETH: 1.15,          // +15%
  'L1/L0 majors': 1.1,
  'L2/Scaling': 1.1,
  'DeFi': 1.21,       // Alts × 1.1 (bonus DeFi)
  'Stablecoins': 0.8, // -20% (réduction cash)
  'Memecoins': 0.88   // Alts × 0.8 (prudence)
}
```

---

## 🔬 Métriques de Validation

### Précision du Modèle

Le système calcule 3 métriques clés:

1. **Erreur Pics** - `|100 - score_at_peak|` pour chaque cycle
2. **Erreur Creux** - `|10 - score_at_bottom|` pour chaque cycle
3. **Précision Globale** - `100 - (total_error / num_cycles / 2)`

**Seuils de qualité:**
- ✅ **Excellent:** Erreur pics < 15, Erreur creux < 20, Précision > 80%
- ⚠️ **Moyen:** Erreur pics < 30, Erreur creux < 35, Précision > 60%
- ❌ **Faible:** Au-delà de ces seuils → Recalibration recommandée

### Confidence Score

Le système calcule une confiance (0-1) basée sur:

```javascript
confidence = base_confidence + calibration_bonus
```

**Base confidence:**
- Distance au centre de phase (plus proche = plus confiant)
- Phase typicality (phases stables vs transitions)

**Calibration bonus:** +5% si calibration < 24h

**Cap par précision modèle:** Limité par la précision globale validée

---

## 🔗 Intégration avec CCS (Crypto Composite Score)

### Blending Formula

```javascript
CCS* (blended) = CCS × (1 - weight) + CycleScore × weight
```

**Poids par défaut:** `weight = 0.3` (30% cycle, 70% CCS)

### Cas d'Usage

| Divergence CCS vs Cycle | Interprétation | Action |
|-------------------------|----------------|--------|
| CCS haut, Cycle bas | Sentiment bullish mais cycle bearish | Prudence, réduire exposure |
| CCS bas, Cycle haut | Sentiment bearish mais cycle bullish | Opportunité accumulation |
| Les deux hauts | Forte confluence bullish | Allocation agressive |
| Les deux bas | Forte confluence bearish | Protection maximale |

---

## 📊 Graphiques et Visualisation

### Bitcoin Cycle Chart

Le graphique principal ([cycle-analysis.html](../static/cycle-analysis.html)) affiche:

1. **Prix Bitcoin historique** (échelle log, depuis 2014 via FRED)
2. **Score de cycle** (0-100, ligne verte épaisse)
3. **Lignes de halving** (verticales violettes)
4. **Timeline des cycles** (barres en bas avec durées en jours)
5. **Position actuelle** (ligne rouge verticale)

**Adaptation contextuelle:**
- Si `enable_dynamic_weighting = true`, la ligne de cycle change de couleur selon la phase
- Couleurs phases: Accumulation (🟡), Bull (🟢), Peak (🟣), Bear (🔴), Pré-Acc (⚫)

### Cycle Position Indicator (Feb 2026)

Timeline visuelle montrant la position actuelle dans le cycle 4 (7 phases):
Pre-halving Rally → Post-halving Consolidation → Bull Run → Distribution → Bear Capitulation → Bottom → Re-accumulation

La phase active est détectée dynamiquement à partir du drawdown, trend 30d, cycle score et mois depuis le halving.

### Cycle-over-Cycle Comparison (Feb 2026)

Deux onglets de comparaison inter-cycles (cycles 2, 3, 4):
- **Normalized**: Performance base 100 au halving (rendements décroissants: x30 → x8 → x2)
- **Drawdown**: Chute depuis le peak de chaque cycle

**Données dynamiques**: Le cycle 4 est alimenté en temps réel via l'API CoinGecko (cache localStorage 1h). Les cycles 2-3 sont des données historiques fixes.

### Cycle Anatomy Table (Feb 2026)

Tableau comparatif des métriques-clé par cycle (halving date/price, peak date/price, return, drawdown, bottom). Le cycle 4 est mis à jour automatiquement avec les données live.

---

## ⚙️ Configuration et Réglages

### Variables Globales

```javascript
// Dans cycle-navigator.js
CYCLE_PARAMS = {
  m_rise_center: 7.0,
  m_fall_center: 30.0,
  k_rise: 1.0,
  k_fall: 0.9,
  p_shape: 0.9,
  floor: 0,
  ceil: 100
}
```

### Fonctions Principales

| Fonction | Module | Description |
|----------|--------|-------------|
| `cycleScoreFromMonths(months)` | cycle-navigator.js | Calcule score pour N mois |
| `calibrateCycleParams(anchors)` | cycle-navigator.js | Grid search optimisation |
| `getCurrentCycleMonths()` | cycle-navigator.js | Mois depuis dernier halving |
| `getCyclePhase(months)` | cycle-navigator.js | Détermine phase actuelle |
| `blendCCS(ccs, cycleMonths, weight)` | cycle-navigator.js | Blend CCS + Cycle |
| `runFullAnalysis()` | cycle-analysis.html | Validation complète |

---

## 🚀 Utilisation

### 1. Analyse Manuelle

```javascript
// Page: cycle-analysis.html
runFullAnalysis()      // Analyse complète avec métriques
calibrateModel()       // Recalibration forcée
testAlternatives()     // Test modèles alternatifs
generateReport()       // Export rapport Markdown
```

### 2. Intégration Programmatique

```javascript
import { cycleScoreFromMonths, getCurrentCycleMonths, getCyclePhase } from './modules/cycle-navigator.js';

// Position actuelle
const { months } = getCurrentCycleMonths();
const score = cycleScoreFromMonths(months);
const phase = getCyclePhase(months);

console.log(`Cycle: ${Math.round(months)}m post-halving`);
console.log(`Score: ${Math.round(score)}/100`);
console.log(`Phase: ${phase.phase} ${phase.emoji}`);
```

### 3. Blending avec CCS

```javascript
import { blendCCS } from './modules/cycle-navigator.js';

const ccs = 65;           // CCS actuel
const cycleMonths = 9;    // 9 mois post-halving
const weight = 0.3;       // 30% cycle

const result = blendCCS(ccs, cycleMonths, weight);
console.log(`CCS*: ${result.blendedCCS}`);
console.log(`Cycle Score: ${result.cycleScore}`);
console.log(`Phase: ${result.phase.description}`);
```

---

## 🔍 Diagnostic et Debug

### Console Logs

Le système émet des logs structurés:

```javascript
✅ Cycle navigator module loaded successfully
🎯 Calibration historique automatique (fresh): { params, score }
🔍 DEBUG getCurrentCycleMonths: { lastHalving, now, totalMonths }
💾 Paramètres calibrés sauvegardés: { m_rise_center: 7.2, ... }
```

### LocalStorage Inspection

```javascript
// Voir paramètres calibrés
JSON.parse(localStorage.getItem('bitcoin_cycle_params'))

// Voir précision modèle
localStorage.getItem('cycle_model_precision')  // 0.0 - 1.0
```

### Forcer Recalibration

```javascript
// Supprimer cache
localStorage.removeItem('bitcoin_cycle_params');

// Recharger page → recalibration auto
location.reload();
```

---

## 📚 Références et Contexte

### Halvings Bitcoin

| Date | Block | Reward | Statut |
|------|-------|--------|--------|
| 2012-11-28 | 210,000 | 25 BTC | ✅ Confirmé |
| 2016-07-09 | 420,000 | 12.5 BTC | ✅ Confirmé |
| 2020-05-11 | 630,000 | 6.25 BTC | ✅ Confirmé |
| 2024-04-20 | 840,000 | 3.125 BTC | ✅ Confirmé |
| **2028-04-01** | 1,050,000 | 1.5625 BTC | 🔮 Estimé |

### Théorie des Cycles

Le modèle repose sur l'observation empirique que:
1. Les halvings réduisent l'offre de nouveaux BTC → Pression haussière
2. Les pics surviennent ~12-18 mois après halving (moyenne ~15m)
3. Les creux surviennent ~28-30 mois après halving
4. Chaque cycle montre un **lengthening** (allongement) progressif

### Limitations

- **Données limitées:** Seulement 3 cycles complets (4 en cours)
- **Marchés changeants:** L'adoption institutionnelle peut altérer les patterns
- **Events exogènes:** Crises macro, régulations, peuvent perturber les cycles
- **Overfitting risk:** Trop d'optimisation sur peu de données historiques

---

## 🛠️ Maintenance

### Checklist Annuelle

- [ ] Valider les dates de halvings passés et futurs
- [ ] Mettre à jour `HISTORICAL_CYCLES` avec données complètes
- [ ] Recalibrer le modèle avec le nouveau cycle complet
- [ ] Revoir les seuils de phases si patterns changent
- [ ] Tester la précision sur données out-of-sample

### Évolutions Futures

1. **Machine Learning:** Remplacer grid search par gradient descent ou Bayesian optimization
2. **Indicateurs On-Chain:** Intégrer MVRV, NVT, Puell Multiple dans le score
3. **Multi-Asset:** Étendre le modèle aux cycles Ethereum (The Merge, upgrades)
4. **Régression Adaptative:** Ajuster automatiquement les paramètres chaque trimestre

---

## 📞 Support

**Fichiers concernés:**
- [`static/cycle-analysis.html`](../static/cycle-analysis.html) - Page d'analyse
- [`static/modules/cycle-navigator.js`](../static/modules/cycle-navigator.js) - Moteur de calcul
- [`static/modules/risk-cycles-tab.js`](../static/modules/risk-cycles-tab.js) - Visualisation

**Commandes utiles:**
```javascript
// Console DevTools
window.runFullAnalysis()     // Analyse + validation
window.calibrateModel()      // Recalibration
window.forceCycleRefresh()   // Clear cache + refresh
```

**Contact:** Voir `CLAUDE.md` pour règles du système global

---

**Version:** 2.0 (Janvier 2026)
**Auteur:** SmartFolio Team
**Licence:** Propriétaire

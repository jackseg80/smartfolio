# Fix: Score Color Semantics Inversion

**Date:** 8 octobre 2025
**Statut:** [OK] Complété

## Contexte

Plusieurs scores affichaient des couleurs **inversées** par rapport à leur sémantique positive canonique définie dans `RISK_SEMANTICS.md`.

### Problème Identifié

**Sémantique canonique** : Tous les scores (Cycle, On-Chain, Risk) sont **positifs** → plus haut = meilleur signal.

**Formule Decision Index** :
```
DI = wCycle × scoreCycle + wOnchain × scoreOnchain + wRisk × scoreRisk
```

**Bug observé** :
- Score Risk 90/100 → affiché en **rouge** (danger) [Error]
- Score On-Chain 33/100 → devrait être **rouge** mais affiché en **vert** [Error]

## Solution Implémentée

### A. Correction des Couleurs dans UnifiedInsights.js

**Fichier** : `static/components/UnifiedInsights.js`

**Ligne 463** : Fonction `colorRisk()` inversée

**Avant** :
```javascript
// - Risk scale: high = risky (red)
const colorRisk = (s) => s > 70 ? 'var(--danger)' : s >= 40 ? 'var(--warning)' : 'var(--success)';
```

**Après** :
```javascript
// - Risk Score scale: high = robust/low risk (green) - See RISK_SEMANTICS.md
const colorRisk = (s) => s > 70 ? 'var(--success)' : s >= 40 ? 'var(--warning)' : 'var(--danger)';
```

**Impact** : `colorRisk()` maintenant identique à `colorPositive()` - tous les scores suivent la même sémantique.

**Lignes affectées** :
- L658 : Score Cycle (utilisé dans carte " Cycle")
- L667 : Score On-Chain (utilisé dans carte " On-Chain")
- L674 : Score Risk (utilisé dans carte " Risque & Budget")

### B. Correction des Couleurs dans analytics-unified.html

**Fichier** : `static/analytics-unified.html`

**Fonction** : `updateRiskMetrics()` lignes 520-577

**Lignes 557-558** : Risk Score inversé

**Avant** :
```javascript
scoreElement.style.color = riskScore > 70 ? 'var(--danger)' :
  riskScore > 40 ? 'var(--warning)' : 'var(--success)';
```

**Après** :
```javascript
scoreElement.style.color = riskScore > 70 ? 'var(--success)' :
  riskScore > 40 ? 'var(--warning)' : 'var(--danger)';
```

**Lignes 568-569** : On-Chain Score inversé (même correction)

### C. Correction des Recommandations On-Chain

**Fichier** : `static/modules/onchain-indicators.js`

**Fonction** : `generateRecommendations()` lignes 1588-1643

#### C.1 Ajout des Paliers Manquants

**Avant** : Seulement 2 cas (< 30 et > 80), **aucune recommandation pour 30-80** [Error]

**Après** : 5 paliers complets

```javascript
if (enhanced_score > 80) {
  // Zone de Distribution Probable (euphorie)
} else if (enhanced_score >= 60) {
  // Marché Bull Confirmé
} else if (enhanced_score >= 40) {
  // Zone de Transition
} else if (enhanced_score >= 30) {
  // Momentum Faible Détecté ← Nouveau !
} else {
  // Zone d'Accumulation Probable (capitulation)
}
```

#### C.2 Utilisation du Bon Score

**Fichier** : `static/risk-dashboard.html` ligne 5272

**Problème** : `generateRecommendations(enhanced)` utilisait le **blend** cycle+onchain au lieu du score pur on-chain.

**Avant** :
```javascript
const enhanced = await enhanceCycleScore(sigmoidScore, 0.25); // 75% cycle + 25% onchain
const recos = generateRecommendations(enhanced); // [Error] Score élevé même si onchain faible
```

**Après** :
```javascript
const composite = calculateCompositeScoreV2(indicators, true); // Score pur on-chain
const recosData = { enhanced_score: composite.score, contributors: composite.contributors, confidence: composite.confidence };
const recos = generateRecommendations(recosData); // [OK] Score on-chain pur (33)
```

**Pourquoi** :
- `enhanced_score` = blend (75% cycle élevé + 25% onchain faible) = score élevé → mauvaise recommandation
- `composite.score` = score on-chain pur (33) → recommandation correcte

### D. Cache-Busting pour Modules ES6

**Fichier** : `static/risk-dashboard.html`

**Lignes 5227-5229 et 4672-4674** : Ajout cache-buster dynamique

```javascript
const cacheBuster = `?v=${Date.now()}`;
const onchainModule = await import(`./modules/onchain-indicators.js${cacheBuster}`);
const cycleModule = await import(`./modules/cycle-navigator.js${cacheBuster}`);
```

**Pourquoi** : Les imports ES6 sont mis en cache par le navigateur, empêchant les mises à jour de se propager.

## Échelle de Couleurs Finale

### Pour tous les scores (Cycle, On-Chain, Risk)

| Score | Couleur | Interprétation | Sémantique |
|-------|---------|----------------|------------|
| 80-100 | [Positive] Vert (success) | Excellent signal | Euphorie/Bull fort |
| 40-79 | [Pending] Orange (warning) | Signal moyen | Transition/Modéré |
| 0-39 | [Negative] Rouge (danger) | Signal faible | Bearish/Faible momentum |

### Échelle de Recommandations On-Chain

| Score | Type | Titre | Action |
|-------|------|-------|--------|
| > 80 | warning | Zone de Distribution Probable | Réduire exposition altcoins |
| 60-80 | info | Marché Bull Confirmé | Maintenir allocation |
| 40-60 | neutral | Zone de Transition | Prudence, attendre confirmation |
| 30-40 | caution | Momentum Faible Détecté | Réduire progressivement risque |
| < 30 | opportunity | Zone d'Accumulation Probable | Augmenter BTC/ETH progressivement |

## Validation

### Exemple Concret (User Report)

**Score On-Chain** : 33/100

**Avant** :
- [Error] Couleur : Vert (success) → incohérent
- [Error] Recommandation : "Zone de Distribution Probable - Score élevé" → incohérent

**Après** :
- [OK] Couleur : Rouge (danger) → cohérent avec score faible
- [OK] Recommandation : "Momentum Faible Détecté - Score faible - Indicateurs on-chain pessimistes" → cohérent

### Tests de Régression

```bash
# Vérifier qu'aucune inversion n'existe
grep -r "100 - risk" static/ docs/ # Doit retourner 0 résultats
grep -r "100 - scoreRisk" static/ docs/ # Doit retourner 0 résultats

# Vérifier les couleurs
grep -A2 "colorRisk.*=" static/components/UnifiedInsights.js
# Doit afficher: s > 70 ? 'var(--success)'

grep -A2 "riskScore > 70" static/analytics-unified.html
# Doit afficher: 'var(--success)'
```

## Fichiers Modifiés

1. `static/components/UnifiedInsights.js` (ligne 461-463)
2. `static/analytics-unified.html` (lignes 557-558, 568-569)
3. `static/modules/onchain-indicators.js` (lignes 1592-1628)
4. `static/risk-dashboard.html` (lignes 4672-4674, 5227-5229, 5272)

## Références

- **Source de vérité** : [docs/RISK_SEMANTICS.md](RISK_SEMANTICS.md)
- **Formule Decision Index** : [docs/UNIFIED_INSIGHTS_V2.md](UNIFIED_INSIGHTS_V2.md)
- **UX Guidelines** : [docs/UX_GUIDE.md](UX_GUIDE.md)

## Notes

- Toutes les inversions `100 - scoreRisk` ont été éliminées
- La sémantique positive est maintenant **cohérente** partout
- Les couleurs suivent la **même échelle** pour tous les scores
- Les recommandations couvrent **tous les paliers** de score

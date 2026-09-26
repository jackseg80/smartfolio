# Portfolio Optimization - Guide Utilisateur

## Vue d'ensemble

L'onglet **Optimization** dans [rebalance.html](../static/rebalance.html) fournit des algorithmes mathématiques sophistiqués pour optimiser l'allocation de portefeuille.

### Différence Rebalancing vs Optimization

| Aspect | Rebalancing (Onglet 1) | Optimization (Onglet 2) |
|--------|------------------------|-------------------------|
| **Approche** | Tactique & règles métier | Mathématique & quantitative |
| **Input** | Stratégies prédéfinies (CCS, Conservative, etc.) | Données historiques de prix |
| **Méthode** | Allocation Engine V2 (floors, incumbency, phase engine) | Algorithmes d'optimisation (Markowitz, Black-Litterman, etc.) |
| **Output** | Plan d'actions (BUY/SELL) basé sur targets | Allocation optimale selon objectif (Max Sharpe, Risk Parity, etc.) |
| **Usage** | Exécution quotidienne, ajustements tactiques | Revue périodique (mensuelle/trimestrielle), décisions stratégiques |

**Workflow recommandé :**
1. **Mensuel/Trimestriel** : Utiliser **Optimization** pour définir allocation stratégique optimale
2. **Hebdomadaire/Quotidien** : Utiliser **Rebalancing** pour ajustements tactiques selon conditions marché

---

## Interface & Accès

### Accès
- URL : `http://localhost:8080/static/rebalance.html`
- Cliquer sur l'onglet **"Optimization"** (en haut)
- L'interface se charge en iframe (`portfolio-optimization-advanced.html?nav=off`)

### Bouton "Open in new tab"
Ouvre l'interface d'optimisation dans une fenêtre séparée pour meilleure visualisation.

---

## Paramètres de Base

Communs à tous les algorithmes :

| Paramètre | Description | Valeur par défaut | Recommandations |
|-----------|-------------|-------------------|-----------------|
| **Source des données** | CoinTracking CSV/API, ou Demo | `cointracking` | Utiliser API pour données temps réel |
| **Historique (jours)** | Fenêtre de calcul | `365` | **Court terme** : 90-180j (plus réactif)<br>**Long terme** : 365-730j (plus stable) |
| **Montant min par asset ($)** | Filtre dust assets | `100` | Adapter selon taille portfolio |
| **Taux sans risque (%)** | Pour calcul Sharpe ratio | `2.0` | 2-4% selon environnement macro |

---

## Les 6 Algorithmes d'Optimisation

### 1.  Max Sharpe (Recommandé pour la plupart)

**Théorie :** Maximise le ratio de Sharpe `(Rendement - Taux sans risque) / Volatilité`

**Objectif :** Meilleur rendement ajusté du risque

**Paramètres :**
- **Poids max par asset (%)** : `35%` (évite concentration excessive)
- **Poids max par secteur (%)** : `60%` (diversification sectorielle)

**Quand utiliser :**
- [OK] Profil **équilibré** (rendement ET risque)
- [OK] Horizon moyen-long terme (6-24 mois)
- [OK] Marchés **normaux** (pas extrêmes)

**Limites :**
- [Error] Peut sous-performer en marchés **très volatils** (bull run extrême)
- [Error] Sensible à l'historique (garbage in, garbage out)

**Exemple de résultat :**
```
BTC: 25%, ETH: 20%, SOL: 8%, Stablecoins: 15%, L1/L0: 12%, DeFi: 10%, Others: 10%
Sharpe attendu: 1.8, Volatilité: 42%, Rendement annualisé: 78%
```

---

### 2.  Black-Litterman

**Théorie :** Combine équilibre du marché (historique) avec **vues personnelles** sur rendements futurs

**Objectif :** Intégrer opinions/analyses dans l'optimisation

**Paramètres :**
- **Vues de marché (JSON)** : Rendements annuels attendus par asset
  ```json
  {"BTC": 0.15, "ETH": 0.12, "SOL": 0.20}
  ```
  *0.15 = 15% par an*

- **Confiance dans les vues (JSON)** : 0-1 (0 = aucune confiance, 1 = certitude absolue)
  ```json
  {"BTC": 0.8, "ETH": 0.6, "SOL": 0.7}
  ```

**Quand utiliser :**
- [OK] Vous avez des **convictions fortes** sur certains assets (analyse fondamentale, catalyseurs)
- [OK] Vous voulez **override historique** avec vision prospective
- [OK] Marchés en **transition** (régime changeant)

**Limites :**
- [Error] Requiert **expertise** pour définir vues réalistes
- [Error] Confiance excessive → biais confirmation

**Exemple de résultat :**
```
# Si vue bullish sur SOL (0.20 / confiance 0.9)
SOL surpondéré vs Max Sharpe: 8% → 15%
```

---

### 3.  Risk Parity

**Théorie :** Égalise la **contribution au risque** de chaque asset (pas le poids !)

**Objectif :** Diversification maximale du risque

**Paramètres :**
- **Volatilité cible (%)** : `15%` (optionnel, sinon optimisation libre)

**Quand utiliser :**
- [OK] Profil **défensif** (priorité = gestion risque)
- [OK] Marchés **incertains** ou **baissiers**
- [OK] Portefeuille avec assets très **hétérogènes** (mix stables + altcoins)

**Limites :**
- [Error] Peut **sous-performer** en bull run (alloue moins aux high-performers volatils)
- [Error] Favorise **stablecoins** (faible volatilité)

**Exemple de résultat :**
```
# BTC et SOL très volatils → poids réduits
# Stablecoins peu volatils → poids augmentés
Stablecoins: 30%, BTC: 18%, ETH: 15%, Alts: 37%
Volatilité portfolio: 15% (vs 42% Max Sharpe)
```

---

### 4.  Max Diversification

**Théorie :** Maximise `Ratio Diversification = Σ(poids × volatilité individuelle) / Volatilité portfolio`

**Objectif :** Minimiser corrélations, maximiser bénéfices diversification

**Paramètres :**
- **Ratio diversification min** : `1.5` (1 = aucun bénéfice, 3+ = excellent)
- **Exposition corrélation max** : `0.7` (limite assets fortement corrélés)

**Quand utiliser :**
- [OK] Portefeuille **concentré** actuellement (peu d'assets)
- [OK] Assets avec **faibles corrélations** disponibles
- [OK] Objectif = **résilience** multi-scénarios

**Limites :**
- [Error] Peut diluer **alpha** (surpondère assets décorrélés mais sous-performants)
- [Error] Difficile en crypto (corrélations élevées BTC-alts)

**Exemple de résultat :**
```
# Favorise assets décorrélés (ex: stables, certains L1 exotiques)
Ratio diversification: 2.1
```

---

### 5.  CVaR Optimization (Conditional Value at Risk)

**Théorie :** Minimise les **pertes extrêmes** (queue de distribution gauche)

**Objectif :** Protection contre scénarios catastrophes (bear market sévère)

**Paramètres :**
- **Niveau de confiance (%)** : `95%` (optimise pour les 5% pires scénarios)
- **Poids CVaR vs Sharpe** : `0.7` (1.0 = 100% focus CVaR, 0.0 = 100% Sharpe)

**Quand utiliser :**
- [OK] Profil **très défensif** (capital preservation > rendement)
- [OK] Anticipation **krach** ou haute volatilité
- [OK] Patrimoine critique (retraite, etc.)

**Limites :**
- [Error] **Sous-performe** en bull run (sacrifie upside pour protection downside)
- [Error] Très **conservateur** (favorise stablecoins massivement)

**Exemple de résultat :**
```
# Niveau confiance 95%, poids CVaR 0.8
Stablecoins: 45%, BTC: 20%, ETH: 15%, Alts: 20%
CVaR à 95%: -12% (vs -28% portfolio actuel)
```

---

### 6.  Frontière Efficiente

**Théorie :** Calcule **tous les portfolios optimaux** pour différents niveaux de risque (volatilité)

**Objectif :** Visualiser trade-off risque/rendement, choisir point optimal selon profil

**Paramètres :**
- **Nombre de points** : `30` (précision de la courbe)
- **Afficher portfolio actuel** : `Oui` (comparaison visuelle)

**Quand utiliser :**
- [OK] **Découverte** : explorer espace des possibles
- [OK] **Comparaison** : évaluer si portfolio actuel est efficient
- [OK] **Éducation** : comprendre trade-offs risque/rendement

**Limites :**
- [Error] Ne donne **pas une allocation unique** (courbe complète)
- [Error] Requiert **choix manuel** du point sur la frontière

**Exemple de résultat :**
```
Graphique : Frontière risque (X) vs rendement (Y)
30 points de Min Variance (5% vol, 8% rdt) à Max Return (60% vol, 120% rdt)
Point actuel : hors frontière → sous-optimal !
```

---

## Interpréter les Résultats

### Section 1 : Allocation Optimale

**Graphique circulaire** : Poids % par asset

**Table détaillée :**
| Colonne | Description |
|---------|-------------|
| **Asset** | Symbole |
| **Weight** | Poids % optimal |
| **Current** | Poids % actuel |
| **Delta** | Différence (optimal - actuel) |

**Interprétation :**
- Delta > 0 → **Acheter** (sous-pondéré)
- Delta < 0 → **Vendre** (sur-pondéré)

---

### Section 2 : Métriques de Performance

**KPIs principaux :**
- **Sharpe Ratio** : >1.5 = bon, >2.0 = excellent
- **Volatilité annualisée** : Comparer vs profil de risque
- **Rendement annualisé** : Historique (pas prédiction !)
- **Max Drawdown** : Perte max historique

**Métriques additionnelles** (selon algorithme) :
- **Ratio Diversification** (Max Diversification)
- **CVaR** (CVaR Optimization)
- **Sortino Ratio** : Sharpe ne pénalisant que downside volatility

**[Warning] ATTENTION :** Rendements historiques ≠ performances futures !

---

### Section 3 : Plan de Rééquilibrage

Table des **trades nécessaires** pour atteindre allocation optimale :

| Colonne | Description |
|---------|-------------|
| **Asset** | Symbole |
| **Action** | BUY / SELL |
| **Amount ($)** | Montant en USD |
| **Current %** | Allocation actuelle |
| **Target %** | Allocation optimale |

**Usage :**
1. Exporter plan (CSV/JSON)
2. Importer dans **onglet Rebalancing** pour exécution
3. Ou exécuter directement via [execution.html](../static/execution.html)

---

### Section 4 : Comparaison d'Algorithmes

Bouton **" Comparer Algorithmes"** → Table comparative :

| Algorithme | Sharpe | Volatilité | Rendement | Max Drawdown |
|------------|--------|------------|-----------|--------------|
| Max Sharpe | 1.82 | 42% | 78% | -35% |
| Risk Parity | 1.21 | 15% | 22% | -12% |
| Black-Litterman | 1.95 | 38% | 76% | -32% |
| ... | ... | ... | ... | ... |

**Usage :**
- Comparer **métriques** entre stratégies
- Identifier **compromis** (ex: Risk Parity = -56% rendement mais -23 pts drawdown)
- Valider **robustesse** (si Max Sharpe >> autres → signal fort)

---

## Workflows Recommandés

### Workflow 1 : Revue Stratégique Mensuelle

**Objectif :** Redéfinir allocation stratégique long-terme

**Étapes :**
1. **Max Sharpe** (historique 365j) → Allocation baseline
2. **Black-Litterman** (si convictions fortes sur certains assets) → Ajustements prospectifs
3. **Comparaison** → Valider cohérence entre algorithmes
4. **Export plan** → Implémenter dans onglet Rebalancing

**Fréquence :** 1x/mois ou après événements majeurs (halving BTC, régulation, etc.)

---

### Workflow 2 : Évaluation Profil Risque

**Objectif :** Trouver allocation optimale selon tolérance au risque

**Étapes :**
1. **Frontière Efficiente** (30 points, 365j historique)
2. Identifier **point actuel** sur graphique
3. Si **hors frontière** → Portfolio sous-optimal !
4. Choisir **point cible** sur frontière selon profil :
   - Défensif → Gauche (faible vol, rendement modéré)
   - Équilibré → Milieu
   - Agressif → Droite (haute vol, rendement élevé)
5. Utiliser **Max Sharpe** ou **Black-Litterman** pour allocation concrète

**Fréquence :** 1x/trimestre ou changement situation personnelle

---

### Workflow 3 : Gestion de Crise (Bear Market)

**Objectif :** Minimiser pertes, préserver capital

**Étapes :**
1. **CVaR Optimization** (confiance 95%, poids CVaR 0.8)
2. Comparer avec **Risk Parity** (vol cible 12%)
3. Choisir allocation la plus **défensive**
4. Implémenter **immédiatement** (rotation vers stables)

**Fréquence :** Réactif (détection régime baissier, krach)

---

### Workflow 4 : Optimisation Multi-Contraintes

**Objectif :** Respecter contraintes métier/fiscales tout en optimisant

**Étapes :**
1. Définir **contraintes custom** :
   - Floors : BTC ≥ 15%, Stablecoins ≥ 10%
   - Caps : Memecoins ≤ 10%
   - Lock : Assets fiscaux (ne pas vendre avant 1 an)
2. **Max Sharpe** avec contraintes → Allocation optimale contrainte
3. Comparer avec **optimisation libre** → Évaluer coût des contraintes

**Fréquence :** Ad-hoc (selon besoins)

---

## Best Practices

### Do's

1. **Historique adapté au contexte :**
   - Bull run actif → 90-180j (réactif)
   - Marché stable → 365j (équilibré)
   - Bear market → 180-365j (éviter biais récent)

2. **Valider cohérence multi-algorithmes :**
   - Si Max Sharpe ≈ Black-Litterman → Signal robuste
   - Si forte divergence → Revoir hypothèses

3. **Backtesting :**
   - Utiliser endpoint `/api/portfolio/optimization/backtest`
   - Tester allocation sur périodes historiques (2022 bear, 2021 bull)

4. **Combiner avec Decision Index :**
   - DI ≥ 65 (score régime bullish) → Accepter allocations agressives
   - DI < 50 (bear) → Forcer défensif (CVaR, Risk Parity)

5. **Rebalancer progressivement :**
   - Grandes rotations (>30% delta) → Échelonner sur 2-4 semaines
   - Éviter market impact + slippage

---

### Don'ts

1. **Ne pas over-optimize :**
   - Optimisation ≠ prédiction magique
   - Garbage in, garbage out (données pourries → résultats pourris)

2. **Ne pas ignorer transaction costs :**
   - Algorithmes ne tiennent **pas compte** des frais/slippage
   - Ajuster manuellement trades < 50$ (dust)

3. **Ne pas changer allocation trop fréquemment :**
   - Optimisation = **stratégique** (mensuel/trimestriel)
   - Pas **tactique** quotidien (utiliser Rebalancing pour ça)

4. **Ne pas suivre aveuglément :**
   - Résultats = **suggestions**, pas ordres
   - Valider cohérence avec analyse macro/fondamentale

5. **Ne pas négliger contraintes réelles :**
   - Liquidité assets (certains non tradables facilement)
   - Fiscalité (ventes génèrent taxes)
   - Minimums exchanges (ne pas acheter <10$ BTC)

---

## Dépannage

### Problème : "Optimization failed" / Erreur API

**Causes possibles :**
- Données historiques insuffisantes (< 30j)
- Trop peu d'assets (< 3)
- Contraintes incompatibles (ex: tous les poids max < 100%)

**Solutions :**
1. Réduire historique (365j → 180j)
2. Augmenter `minusd` pour filtrer plus d'assets
3. Relâcher contraintes (poids max 35% → 50%)

---

### Problème : Résultats incohérents (allocations extrêmes)

**Causes possibles :**
- Données de prix corrompues (outliers)
- Période historique non représentative (ex: uniquement bull run)

**Solutions :**
1. Vérifier données source (endpoint `/api/portfolio/optimization/analyze`)
2. Tester différentes fenêtres historiques (90j, 180j, 365j, 730j)
3. Utiliser contraintes (poids max 30%) pour limiter extrêmes

---

### Problème : Allocation 100% stablecoins

**Cause :** Algorithme détecte rendement ajusté du risque négatif sur crypto

**Interprétation :** Signal baissier fort (bear market sévère dans historique)

**Solutions :**
1. Normal si période historique = 2022 bear → Réduire fenêtre (exclure bear)
2. Utiliser **Black-Litterman** avec vues bullish si vous anticipez reprise
3. Accepter allocation défensive si contexte macro justifie

---

## API Endpoints

### GET `/api/portfolio/optimization/analyze`

**Description :** Analyse portfolio actuel + suggestions de paramètres

**Réponse exemple :**
```json
{
  "recommended_lookback_days": 365,
  "suggested_algorithm": "max_sharpe",
  "current_metrics": {
    "sharpe": 1.2,
    "volatility": 0.48,
    "max_drawdown": -0.42
  },
  "optimization_readiness": "ready"
}
```

---

### POST `/api/portfolio/optimization/optimize`

**Description :** Optimisation standard (Max Sharpe)

**Body exemple :**
```json
{
  "source": "cointracking",
  "lookback_days": 365,
  "min_usd": 100,
  "risk_free_rate": 0.02,
  "constraints": {
    "max_weight": 0.35,
    "max_sector_weight": 0.60
  }
}
```

**Réponse :** Allocation optimale + métriques + plan trades

---

### POST `/api/portfolio/optimization/optimize-advanced`

**Description :** Optimisation avec algorithme personnalisé

**Body exemple :**
```json
{
  "algorithm": "black_litterman",
  "source": "cointracking_api",
  "lookback_days": 365,
  "parameters": {
    "market_views": {"BTC": 0.15, "ETH": 0.12},
    "view_confidence": {"BTC": 0.8, "ETH": 0.6}
  }
}
```

---

### POST `/api/portfolio/optimization/backtest`

**Description :** Backtest allocation sur périodes historiques

**Body exemple :**
```json
{
  "allocation": {"BTC": 0.30, "ETH": 0.25, "Stables": 0.20, "Alts": 0.25},
  "start_date": "2023-01-01",
  "end_date": "2024-01-01",
  "rebalance_frequency": "monthly"
}
```

**Réponse :** Performance historique (rendement, Sharpe, drawdown, etc.)

---

## Références Théoriques

### Markowitz (1952)
- **Paper :** "Portfolio Selection", Journal of Finance
- **Concept :** Optimisation moyenne-variance (Max Sharpe)

### Black & Litterman (1992)
- **Paper :** "Global Portfolio Optimization", Financial Analysts Journal
- **Concept :** Intégration vues subjectives avec équilibre marché

### Rockafellar & Uryasev (2000)
- **Paper :** "Optimization of Conditional Value-at-Risk", Journal of Risk
- **Concept :** CVaR (tail risk minimization)

### Maillard et al. (2010)
- **Paper :** "The Properties of Equally Weighted Risk Contribution Portfolios"
- **Concept :** Risk Parity (equal risk contribution)

---

## Intégration avec Autres Modules

### Analytics Unified
- Importer métriques ML (sentiment, cycle score) → Black-Litterman views
- Comparer allocation optimale vs allocation actuelle (dashboard)

### Risk Dashboard
- VaR optimisation target → CVaR Optimization (alignment)
- Stress tests allocation proposée (scénarios extrêmes)

### Rebalance (Onglet 1)
1. **Optimization** génère allocation stratégique → Export JSON
2. **Rebalancing** importe comme "Custom Strategy"
3. Apply Strategy → Génère plan d'actions

### Execution
- Plan de rééquilibrage Optimization → Exécution directe
- Fragmentation, timing, slippage management

---

## Changelog

### v2.0 (Oct 2025)
- [OK] 6 algorithmes (Max Sharpe, Black-Litterman, Risk Parity, Max Div, CVaR, Frontière)
- [OK] Comparaison multi-algorithmes
- [OK] Support contraintes custom
- [OK] Intégration iframe dans rebalance.html

### v1.0 (Archivé)
- Basic Markowitz optimization uniquement

---

## Support & Ressources

### Documentation Complémentaire
- [ARCHITECTURE.md](ARCHITECTURE.md) - Vue d'ensemble système
- [API_REFERENCE.md](API_REFERENCE.md) - Endpoints détaillés
- [ALLOCATION_ENGINE_V2.md](ALLOCATION_ENGINE_V2.md) - Logique Rebalancing

### Exemples Code
- [portfolio-optimization-advanced.html](../static/portfolio-optimization-advanced.html) - Interface complète
- [services/portfolio_optimization.py](../services/portfolio_optimization.py) - Backend

### Contact
- GitHub Issues : https://github.com/anthropics/claude-code/issues
- Docs Claude Code : `/help`

---

*Dernière mise à jour : Décembre 2025*

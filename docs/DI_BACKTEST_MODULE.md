# DI Backtest Module

> Documentation du module de backtest du Decision Index
> Création: Février 2026

## Vue d'ensemble

Le module DI Backtest permet de valider rétrospectivement les performances du Decision Index sur des périodes historiques (2017+). Il reconstruit le DI historique et simule des stratégies de trading pour établir la confiance dans le système.

## Architecture

```
services/di_backtest/
├── __init__.py                  # Exports du module
├── data_sources.py              # Agrégation données historiques
├── historical_di_calculator.py  # Reconstruction DI historique
├── di_backtest_engine.py        # Moteur de simulation
└── trading_strategies.py        # Stratégies de trading DI

api/di_backtest_endpoints.py     # 6 endpoints REST
static/di-backtest.html          # Interface utilisateur
```

## Reconstruction du Decision Index

### Formule

Le DI est reconstruit avec la même formule que le système de production:

```python
raw_score = (cycle × 0.30 + onchain × 0.35 + risk × 0.25 + sentiment × 0.10)
adjusted_score = raw_score × phase_factor + macro_penalty
final_score = clamp(adjusted_score, 0, 100)
```

**Phase factors:**
- Bearish (cycle < 70): 0.85
- Moderate (70 ≤ cycle < 90): 1.0
- Bullish (cycle ≥ 90): 1.05

### Sources de Données

| Composant | Source | Profondeur | Méthode |
|-----------|--------|------------|---------|
| **Cycle Score** | Halvings BTC | Illimité | Double-sigmoïde (port de cycle-navigator.js) |
| **OnChain Score** | Proxy calculé | 3000+ jours | Distance 200 DMA + RSI + Momentum |
| **Risk Score** | Prix historiques | 3000+ jours | Volatilité inversée + Drawdown inversé |
| **Sentiment** | Fear & Greed API | 365 jours | alternative.me + fallback proxy |
| **Macro Penalty** | FRED API (VIX/DXY) | 20+ ans | VIX>30 ou DXY±5% → -15 pts |

### Buffer de Calcul

Un buffer de **250 jours** est automatiquement ajouté avant la date de début demandée pour permettre le calcul des indicateurs rolling (200 DMA, volatilité 90j, etc.).

## Stratégies de Trading

### S1: DIThresholdStrategy

Allocation basée sur des seuils de DI:
- DI < 40 → 30-50% risky (mode défensif)
- DI 40-60 → 60% risky (mode neutre)
- DI > 70 → 75-85% risky (mode agressif)

### S2: DIMomentumStrategy

Suit la tendance du DI:
- DI hausse >5pts/7j → augmenter exposition +10%
- DI baisse >5pts/7j → réduire exposition -10%

### S3: DIContrarianStrategy

Stratégie contrariante:
- DI < 20 (extreme fear) → 85% risky (accumulation)
- DI > 80 (extreme greed) → 30% risky (prise de profits)

### S4: DIRiskParityStrategy

Risk parity classique ajusté par le DI:
```
allocation = risk_parity_weights × (DI / 50)
```

### S5: DISignalStrategy

Signaux discrets:
- **BUY**: DI croise 40 à la hausse (confirmation 3 jours)
- **SELL**: DI croise 60 à la baisse (holding minimum 14 jours)

### S6: DISmartfolioReplicaStrategy (Recommandé)

**Réplique la logique réelle de l'Allocation Engine V2 de SmartFolio.**

Utilise le cycle score (pas le DI directement) pour déterminer la phase de marché:
- Phase bullish (cycle ≥90): 15% stables, 85% risky
- Phase moderate (70≤cycle<90): 20% stables, 80% risky
- Phase bearish (cycle <70): 30% stables, 70% risky

Floors appliqués:
- BTC minimum: 15%
- ETH minimum: 12%
- Stables minimum: 10%

**Idéal pour**: Valider que le DI Backtest est cohérent avec le comportement réel du projet SmartFolio

## API Endpoints

### POST /api/di-backtest/run

Exécute un backtest complet.

**Request:**
```json
{
  "start_date": "2020-01-01",
  "end_date": "2021-12-31",
  "strategy": "threshold",
  "initial_capital": 10000,
  "transaction_cost": 0.001
}
```

**Response:**
```json
{
  "ok": true,
  "result": {
    "total_return": 3.136,
    "annualized_return": 1.85,
    "benchmark_return": 4.99,
    "excess_return": -1.86,
    "max_drawdown": -0.456,
    "sharpe_ratio": 2.35,
    "sortino_ratio": 6.40,
    "calmar_ratio": 4.06,
    "num_trades": 24,
    "win_rate": 0.65,
    "strategy_name": "DI Threshold"
  }
}
```

### POST /api/di-backtest/historical-di

Reconstruit le DI historique sans backtest.

**Request:**
```json
{
  "start_date": "2020-01-01",
  "end_date": "2020-12-31"
}
```

### GET /api/di-backtest/strategies

Liste les stratégies disponibles.

### POST /api/di-backtest/compare

Compare plusieurs stratégies sur la même période.

### GET /api/di-backtest/events

Retourne les événements de marché majeurs (halvings, crashes, etc.).

### GET /api/di-backtest/period-analysis

Analyse détaillée par période prédéfinie.

## Périodes Prédéfinies

| Nom | Période | Description |
|-----|---------|-------------|
| Bull Run 2017 | 2017-01 → 2017-12 | Premier grand bull run BTC |
| Crash 2018 | 2018-01 → 2018-12 | Bear market post-2017 |
| COVID Crash | 2020-02 → 2020-05 | Crash COVID-19 |
| Bull Run 2020-2021 | 2020-10 → 2021-11 | Bull run post-halving |
| Bear Market 2022 | 2021-11 → 2022-11 | Bear market post-ATH |
| Recovery 2023-2024 | 2022-11 → 2024-04 | Phase de recovery |
| Full History | 2017-01 → aujourd'hui | Historique complet |

## Métriques de Performance

### KPIs Principaux

| Métrique | Description |
|----------|-------------|
| **Total Return** | Rendement total sur la période |
| **Annualized Return** | Rendement annualisé |
| **Max Drawdown** | Perte maximale depuis un pic |
| **Sharpe Ratio** | Rendement ajusté au risque (vs risk-free) |
| **Sortino Ratio** | Sharpe avec downside deviation uniquement |
| **Calmar Ratio** | Rendement annualisé / Max Drawdown |
| **Win Rate** | % de trades gagnants |
| **Volatility** | Écart-type annualisé des rendements |

### Métriques DI Spécifiques

| Métrique | Description |
|----------|-------------|
| **di_btc_correlation_30d** | Corrélation DI vs returns futurs 30j |
| **di_accuracy_pct** | Précision des signaux DI |
| **avg_di_during_gains** | DI moyen pendant les gains |
| **avg_di_during_losses** | DI moyen pendant les pertes |

## Interface Utilisateur

### Accès

Menu Analytics → DI Backtest (`/di-backtest.html`)

### Contrôles

- **Période**: Date début/fin avec presets
- **Stratégie**: Sélecteur des 5 stratégies
- **Capital initial**: Montant de départ
- **Coûts de transaction**: % par trade (défaut 0.1%)

### Visualisations

1. **Equity Curve**: Portfolio vs Benchmark (BTC Buy & Hold)
2. **DI History Chart**: Évolution du DI avec composants
3. **Drawdown Chart**: Underwater curve
4. **Monthly Returns**: Calendrier des performances

**Zoom**: Cliquer sur n'importe quel graphique pour l'afficher en plein écran. Fermer avec Échap ou clic en dehors.

## Utilisation Programmatique

```python
from services.di_backtest import (
    historical_di_calculator,
    di_backtest_engine,
    DIThresholdStrategy
)

# Reconstruire le DI historique
di_data = await historical_di_calculator.calculate_historical_di(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2021, 12, 31)
)

# Exécuter un backtest
strategy = DIThresholdStrategy()
result = di_backtest_engine.run_backtest(
    di_history=di_data.di_history,
    strategy=strategy,
    initial_capital=10000
)

print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.1%}")
```

## Limitations Connues

1. **OnChain Score**: Proxy basé sur les prix (pas de données on-chain réelles)
2. **Sentiment**: Historique Fear & Greed limité à ~365 jours
3. **Granularité**: Données journalières uniquement
4. **Benchmark**: Comparaison vs BTC uniquement (pas ETH/SPY)

## Performance Typique

Exemple: Bull Run 2020-2021 avec DI Threshold Strategy

```
Total Return: 313.6%
Benchmark (BTC): 499.0%
Max Drawdown: -45.6%
Sharpe Ratio: 2.35
Sortino Ratio: 6.40
```

La stratégie DI est plus conservative que 100% BTC mais offre un meilleur ratio risque/rendement.

## Voir Aussi

- [DECISION_INDEX_V2.md](DECISION_INDEX_V2.md) - Formule complète du DI
- [RISK_SEMANTICS.md](RISK_SEMANTICS.md) - Sémantique du Risk Score
- [architecture.md](architecture.md) - Architecture globale SmartFolio

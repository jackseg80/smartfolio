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

**Réplique le pipeline complet de l'Allocation Engine V2 de SmartFolio (4 couches).**

**Layer 1 — Risk Budget** (formule production):

```python
blended     = 0.5×CycleScore + 0.3×OnChainScore + 0.2×RiskScore
risk_factor = 0.5 + 0.5 × (RiskScore / 100)
base_risky  = clamp((blended - 35) / 45, 0, 1)
risky       = clamp(base_risky × risk_factor, 0.20, 0.85)
```

Source: `static/modules/market-regimes.js` → `calculateRiskBudget()`

**Layer 2 — Market Overrides**:

- On-chain divergence: |cycle - onchain| ≥ 30 → +10% stables
- Low risk score: risk ≤ 30 → stables ≥ 50%
- Macro penalty: VIX > 30 OR DXY ±5% → -15 pts sur le DI

**Layer 3 — Exposure Cap**: risky clampé entre 20% et 85%

**Layer 4 — Governance Penalty** (contradiction-based):

- Contradiction index reconstruit depuis données historiques (vol+cycle, DI vs cycle, score divergence)
- Penalty proportionnelle: 0-25% de réduction selon le niveau de contradiction
- Impact historique faible (~0.1-0.3%) car les conditions se déclenchent rarement simultanément

**Idéal pour**: Valider que le DI Backtest est cohérent avec le comportement réel du projet SmartFolio

### S7: DITrendGateStrategy

SMA200 trend gate + DI allocation:
- BTC > SMA200 → risk-on allocation (80% risky par défaut)
- BTC < SMA200 → risk-off allocation (20% risky par défaut)
- Whipsaw filter: N jours de confirmation avant changement
- Options: DI modulation en risk-on, drawdown circuit breaker

### S8: DICycleRotationStrategy (Multi-Asset)

**Rotation 3 actifs (BTC/ETH/Stables) basée sur 5 phases du cycle Bitcoin.**

Utilise `cycle_score` + `cycle_direction` pour détecter la phase:

| Phase | Cycle Score | Direction | BTC | ETH | Stables |
|-------|-------------|-----------|-----|-----|---------|
| Accumulation | < 70 | ≥ 0 | 50% | 15% | 35% |
| Bull Building | 70-89 | ≥ 0 | 35% | 35% | 30% |
| Peak | ≥ 90 | any | 20% | 20% | 60% |
| Distribution | 70-89 | < 0 | 20% | 10% | 70% |
| Bear | < 70 | < 0 | 15% | 5% | 80% |

**Features**:
- EMA smoothing (alpha=0.15) pour éviter les transitions abruptes
- Floor constraints: BTC≥10%, ETH≥5%, Stables≥10%
- DI modulation optionnelle (désactivée par défaut)

**Résultats backtest (Full History 2017-2025)**:
- Rot_conservative: **Sharpe 1.042**, MaxDD -32.6%, Return 612.6%
- Rot_default: **Sharpe 0.965**, MaxDD -41.3%, Return 692.0%
- vs Replica V2.1: Sharpe 0.759, MaxDD -57.3%, Return 537.8%

**Idéal pour**: Valider la rotation multi-asset par phase de cycle avant application aux 11 groupes en production

### S9: DIAdaptiveContinuousStrategy (Multi-Asset)

**Allocation continue basée sur le DI + trend confirmation (golden/death cross).**

Contrairement à S8 (5 phases discrètes), S9 utilise un mapping continu piecewise linéaire du DI vers l'allocation risky, sans seuils binaires.

**Pipeline d'allocation:**

1. **DI → Allocation continue** (piecewise linéaire):

| DI Range | Allocation Risky | Description |
|----------|-----------------|-------------|
| 0-25 | floor (10%) | Bear protection |
| 25-50 | 10% → 40% | Rampe graduelle |
| 50-75 | 40% → 70% | Rampe intermédiaire |
| 75-100 | 70% → ceiling (85%) | Rampe bull agressive |

2. **Trend overlay** (golden/death cross — structurellement différent de l'onchain proxy):
   - Golden cross (SMA50 > SMA200 ET price > SMA200) → +10% boost
   - Death cross (SMA50 < SMA200 ET price < SMA200) → -10% réduction
   - Zone mixte → 0% (neutre)
   - Le onchain proxy mesure la *magnitude* (distance à SMA200), le trend overlay mesure la *direction* (SMA50 vs SMA200)

3. **Risk score adjustment**: `(risk_score - 50) / 500` → ±10% max

4. **Smoothing asymétrique continu** (via `cycle_direction`, pas de seuil binaire):
   ```
   direction_factor = (cycle_direction + 1) / 2    # [0, 1]
   alpha = alpha_bear + direction_factor × (alpha_bull - alpha_bear)
   # direction=-1 → alpha_bear (fast exit)
   # direction=+1 → alpha_bull (slow entry)
   ```

5. **BTC/ETH split continu**: ETH share de 20% (bear) à 40% (bull) du risky, interpolé sur DI 30-80

6. **Floor enforcement**: BTC≥10%, ETH≥3%, Stables≥10%

**Paramètres (`ContinuousParams`):**

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `alloc_floor` | 0.10 | Min risky (bear floor) |
| `alloc_ceiling` | 0.85 | Max risky (bull ceiling) |
| `enable_trend_overlay` | True | Golden/death cross |
| `sma_fast` / `sma_slow` | 50 / 200 | Périodes SMA pour trend |
| `trend_boost_pct` | 0.10 | Boost ±allocation |
| `enable_risk_adjustment` | True | Ajustement risk score |
| `smoothing_alpha_bull` | 0.12 | Slow entry |
| `smoothing_alpha_bear` | 0.50 | Fast exit |
| `eth_share_bull` / `eth_share_bear` | 0.40 / 0.20 | ETH share du risky |

**Différences clés vs S8 (Cycle Rotation):**

| Aspect | S8 (Cycle Rotation) | S9 (Adaptive Continuous) |
|--------|---------------------|--------------------------|
| Allocation | 5 phases discrètes | Mapping continu piecewise |
| Signal de base | `cycle_score` | `di_value` (composite complet) |
| Ceiling | 70% (bull) | 85% (configurable) |
| Trend | Aucun | Golden/death cross overlay |
| Smoothing | EMA uniforme ou asym binaire | Direction-continu (pas de seuil) |
| Avg risky OOS | ~28% | ~41-51% |

**Idéal pour**: Exploration paramétrique de l'allocation continue. Outil de recherche, pas de production.

## Multi-Asset Engine

Le moteur de backtest supporte 2 modes:

- **2-asset** (défaut): BTC + Stablecoins — utilisé par toutes les stratégies S1-S7
- **3-asset** (`multi_asset=True`): BTC + ETH + Stablecoins — utilisé par S8

Le mode 3-asset est activé automatiquement pour `di_cycle_rotation` et `di_adaptive_continuous` via l'API.

**Backward compatibility**: Le mode 2-asset utilise le même code (dict de poids) que le mode 3-asset. Les stratégies existantes produisent des résultats identiques.

**Source ETH**: Binance API via `data_sources.get_multi_asset_prices(["ETH"])`, disponible depuis 2017.

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
    "rebalance_count": 88,
    "avg_risky_allocation": 0.43,
    "portfolio_turnover": 0.12,
    "upside_capture": 0.75,
    "downside_capture": 0.42,
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
| **Rebalance Count** | Nombre de rebalancements effectués |
| **Avg Risky Allocation** | Allocation moyenne en actifs risqués |
| **Portfolio Turnover** | Rotation du portefeuille (volume rebalancé) |
| **Upside Capture** | Capture des hausses vs benchmark |
| **Downside Capture** | Capture des baisses vs benchmark |
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
- **Stratégie**: Sélecteur des 9 stratégies
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

## DI Calculator V2 — Proxies Adaptatifs

> Ajout: 8 Fev 2026

### Contexte : DI Production vs DI Backtest

Le DI backtest utilise des **proxies** car les donnees de production ne sont pas historiquement disponibles :

| Composant | **Production** | **Backtest** |
|-----------|----------------|-------------|
| OnChain | ML orchestrator, APIs live | Proxy prix : 200 DMA + RSI + momentum 90j |
| Risk | Metriques live multi-sources | Proxy prix : vol 90j + DD 90j |
| Sentiment | APIs sentiment temps reel | F&G API (365j max) + proxy V2 (vol 30j, mom 14j, RSI 14j, 52w high) |
| Cycle | `phase_engine` (ML-based) | Sigmoide deterministe (halvings) |
| Macro | `macro_stress_service` (-15 binaire) | FRED VIX/DXY |
| Phase Factor | ML per-template | Base sur cycle_score (V1) ou raw_score (V2) |

Le V2 calculator rend ces proxies plus realistes sans changer la formule DI.

### DICalculatorConfig

```python
from services.di_backtest.historical_di_calculator import DICalculatorConfig

config = DICalculatorConfig(
    normalization="adaptive",   # "fixed" (V1 compat) | "adaptive" (expanding percentile)
    macro_mode="graduated",     # "binary" (-15 ou 0) | "graduated" (additive, lineaire)
    phase_mode="blended",       # "cycle" (V1, base sur cycle_score) | "blended" (continu, base sur raw_score)
    cycle_mode="corrected",     # "deterministic" (V1 sigmoide pure) | "corrected" (correction conditionnelle)
    cycle_correction_pts=10,    # Points de correction quand prix contredit le modele
)
```

### Composants V2 modifies

**1. OnChain proxy — Expanding percentile** (`normalization="adaptive"`)

- DMA distance et momentum 90j normalises via `expanding(min_periods=N).rank(pct=True) * 100`
- RSI inchange (deja auto-normalise 0-100 par construction)
- Warm-up fallback a normalisation fixe pour les premiers jours
- Aucun look-ahead : chaque jour ne voit que l'historique passe

**2. Risk score — Expanding percentile inverse** (`normalization="adaptive"`)

- Volatilite : `(-vol).expanding().rank(pct=True) * 100` (haut rank = basse vol = robuste)
- Drawdown : `drawdown.expanding().rank(pct=True) * 100` (haut rank = petit DD = robuste)
- Semantique preservee : high score = robust (pas d'inversion 100-x)

**3. Macro penalty V2 — Graduee additive** (`macro_mode="graduated"`)

```text
VIX penalty : 0 (VIX <= 20) -> -10 (VIX >= 45), lineaire
DXY penalty : 0 (change <= 2%) -> -8 (change >= 10%), lineaire
Total = VIX + DXY, cappe a -15
```

Exemple : VIX=35 -> ~-6pts (au lieu de -15 en V1). VIX=35 + DXY+5% -> ~-10pts.

**4. Cycle score — Correcteur conditionnel** (`cycle_mode="corrected"`)

Corrige la sigmoide uniquement quand le prix contredit fortement le modele :

- Cycle > 85 ET momentum 6m < -20% -> correction -10pts (sigmoide trop optimiste)
- Cycle < 40 ET momentum 6m > +30% -> correction +10pts (sigmoide trop pessimiste)

Pas de blend permanent (evite double-comptage du momentum deja dans l'onchain proxy).

**5. Sentiment proxy V2** (`normalization="adaptive"`)

- 4 composants (vs 2 en V1): vol 30j inversee (25%), momentum 14j (25%), RSI 14j (25%), distance 52-week high (25%)
- Vol, momentum et distance normalises via expanding percentile. RSI inchange (deja 0-100)
- Automatiquement utilise quand `normalization="adaptive"` (V2 default)
- **KS results**: degradation sur 5/6 expanding windows (+0.038 a +0.250 KS shift). Le proxy 4-composants est plus regime-dependant que le V1 simple. Impact negligeable sur le DI (sentiment = 10% poids)

**6. Phase factor V2 — Continu** (`phase_mode="blended"`)

```text
factor = 0.85 + 0.20 * (raw_score / 100)
```

Remplace les 3 paliers discrets (0.85 / 1.0 / 1.05) par un facteur continu de 0.85 a 1.05, base sur le raw_score blended (plus proche du phase_engine ML en production).

### Utilisation V2

```python
from services.di_backtest.historical_di_calculator import (
    HistoricalDICalculator, DICalculatorConfig,
)

calculator = HistoricalDICalculator()

# V2 (defaults adaptatifs)
di_data = await calculator.calculate_historical_di_v2(
    user_id="jack",
    start_date="2017-01-01",
    end_date="2025-12-31",
    config=DICalculatorConfig(),  # Tous les defaults V2
)

# V1 (backward compat, inchange)
di_data_v1 = await calculator.calculate_historical_di(
    user_id="jack",
    start_date="2017-01-01",
    end_date="2025-12-31",
)
```

## Walk-Forward Validation

### Scripts d'analyse

| Script | Description |
|--------|-------------|
| `scripts/analysis/diagnose_di_components.py` | Diagnostic IS vs OOS (KS tests, correlations, distributions) |
| `scripts/analysis/walk_forward_rotation.py` | Walk-forward V1 : split fixe IS/OOS |
| `scripts/analysis/walk_forward_rotation_v2.py` | Walk-forward V2 : expanding multi-window, V1 vs V2 |
| `scripts/analysis/compare_rotation_v3.py` | Comparaison detaillee des configs rotation |

### Expanding Walk-Forward (V2)

Le split fixe (IS 2017-2021, OOS 2021-2025) est biaise : le bear 2022 pese ~50% du OOS, ecrasant la retention. L'expanding walk-forward dilue ce biais :

| Train | Test | Window |
|-------|------|--------|
| 2017-2019 | 2020 | OOS-1 |
| 2017-2020 | 2021 | OOS-2 |
| 2017-2021 | 2022 | OOS-3 |
| 2017-2022 | 2023 | OOS-4 |
| 2017-2023 | 2024-2025 | OOS-5 |

### Resultats Walk-Forward (8 Fevrier 2026)

**Sharpe Retention moyenne (expanding, 5 windows) — 12 configs, 288 backtests:**

| Config | Type | V1 Calculator | V2 Calculator |
|--------|------|--------------|--------------|
| **Cons+AsymA** | rotation | 155% | 153% |
| **Cons+Fast** | rotation | 154% | 153% |
| **Replica V2.1** | replica | 154% | 111% |
| **Cons+SMA150+AsymA** | rotation | 141% | 140% |
| AC-C75 | continuous | 122% | 93% |
| AC-C70-FB80-T15 | continuous | 121% | 88% |
| AC-C70 | continuous | 120% | 94% |
| AC-Default | continuous | 117% | 92% |
| AC-C70-FB70 | continuous | 117% | 89% |
| AC-C75-FB65-T15 | continuous | 114% | 87% |
| AC-C80-FB60 | continuous | 111% | 85% |
| AC-FB70 | continuous | 109% | 88% |

Toutes les strategies sont **ROBUST** (retention > 80%). Les rotations retiennent ~150% du Sharpe IS→OOS, les strategies continues ~85-95%.

**KS Distribution Shift (V2 vs V1, amelioration):**

| Composant | Fenetres ameliorees | Gain moyen |
|-----------|--------------------|-----------:|
| risk_score | 4/6 | +0.15 |
| onchain_score | 3/6 | +0.12 |
| decision_index | 4/6 | +0.15 |
| cycle_score | 2/6 | +0.11 (mixte) |
| sentiment_score | 1/6 | -0.10 (V2 plus regime-dependant) |

**Rank stability:** Le champion IS reste top-3 OOS dans 3/5 windows (V1 et V2).

### Diagnostic DI — Findings cles

Resultats de `diagnose_di_components.py` sur IS (2017-2021) vs OOS (2021-2025) :

- **risk_score** : Plus gros distribution shift (KS=0.420), mais correlation amelioree OOS
- **cycle_score** : Correlation INVERSEE (IS: +0.144, OOS: -0.178) — le smoking gun. La sigmoide deterministe donne des signaux contre-productifs en OOS
- **decision_index** : Correlation passe de +0.148 a -0.096 (entraine par le cycle)
- **Macro penalty** : MOINS frequente OOS (6.0%) que IS (8.3%), contrairement a l'intuition
- **Phase** : Bearish 61.4% OOS (pas 80% comme suppose initialement)

### Exploration Parametrique S9 — Adaptive Continuous (8 Fev 2026)

8 variantes de `ContinuousParams` testees pour ameliorer l'asymetrie de capture :

| Config | Ceiling | Alpha Bear | Trend Boost | Hypothese |
|--------|---------|------------|-------------|-----------|
| AC-Default | 0.85 | 0.50 | 0.10 | Baseline S9 |
| AC-C70 | 0.70 | 0.50 | 0.10 | Cap exposition max |
| AC-C75 | 0.75 | 0.50 | 0.10 | Cap modere |
| AC-FB70 | 0.85 | 0.70 | 0.10 | Sortie bear rapide |
| AC-C70-FB70 | 0.70 | 0.70 | 0.10 | Cap + sortie rapide |
| AC-C75-FB65-T15 | 0.75 | 0.65 | 0.15 | 3 leviers equilibres |
| AC-C70-FB80-T15 | 0.70 | 0.80 | 0.15 | Protection agressive |
| AC-C80-FB60 | 0.80 | 0.60 | 0.10 | Ajustement leger |

**Capture Asymmetry OOS moyenne (V2, 5 fenetres expanding):**

| Config | 2020 | 2021 | 2022 | 2023 | 2024-25 | Moyenne |
|--------|------|------|------|------|---------|---------|
| **Cons+AsymA** | +0.049 | +0.060 | +0.006 | -0.004 | -0.003 | **+0.022** |
| Cons+Fast | +0.043 | +0.064 | +0.007 | -0.004 | -0.007 | +0.021 |
| Cons+SMA150+AsymA | +0.049 | +0.061 | +0.006 | -0.004 | -0.015 | +0.019 |
| AC-Default | +0.029 | +0.038 | -0.014 | -0.029 | -0.032 | -0.002 |
| AC-FB70 | +0.029 | +0.034 | -0.021 | -0.028 | -0.026 | -0.002 |
| AC-C70-FB80-T15 | +0.026 | +0.033 | -0.015 | -0.030 | -0.031 | -0.003 |
| AC-C75-FB65-T15 | +0.025 | +0.034 | -0.012 | -0.033 | -0.032 | -0.004 |
| AC-C70 | +0.019 | +0.035 | -0.014 | -0.030 | -0.032 | -0.004 |
| AC-C70-FB70 | +0.021 | +0.031 | -0.021 | -0.028 | -0.028 | -0.005 |

**Sharpe OOS moyen et MaxDD worst-case (V2):**

| Config | Avg OOS Sharpe | Worst MaxDD OOS |
|--------|----------------|-----------------|
| **Cons+AsymA** | **1.222** | **-21.7%** |
| Cons+Fast | 1.203 | -21.6% |
| Cons+SMA150+AsymA | 1.158 | -17.6% |
| AC-C70-FB80-T15 | 0.889 | -28.8% |
| AC-Default | 0.888 | -29.9% |
| AC-FB70 | 0.878 | -30.3% |
| AC-C70-FB70 | 0.850 | -30.2% |

**Rendement absolu IS (V2, train 2017-2021):**

| Config | Return IS | Sharpe IS | Avg Risky IS |
|--------|-----------|-----------|--------------|
| AC-FB70 | 968% | 1.507 | 42.7% |
| AC-Default | 906% | 1.468 | 42.8% |
| AC-C70-FB80-T15 | 829% | 1.487 | 40.9% |
| Cons+AsymA | 436% | 1.631 | 28.0% |

**Conclusion de l'exploration:**

Le probleme est **structurel, pas parametrique**. Les 8 variantes AC-* sont clustered entre -0.002 et -0.005 d'asymetrie moyenne OOS malgre des parametres tres differents (ceiling 0.70-0.85, alpha_bear 0.50-0.80, trend 0.10-0.15).

La cause : le mapping continu DI→allocation produit une allocation risky moyenne de ~41-51% (vs ~28% pour Cons+AsymA). Le DI passe la majorite du temps entre 40 et 70, zone ou l'allocation est deja 30-60% risky. Baisser le ceiling ne change presque rien car le DI depasse rarement 75.

Les **5 phases discretes** de S8 agissent comme un filtre naturel de double confirmation (cycle_score ET direction doivent s'aligner), ce que le mapping continu n'a pas.

### Strategie recommandee

**Cons+AsymA** (asymetrique alpha bull=0.15, bear=0.50) est la config par defaut recommandee :

- Meilleure capture asymmetry OOS (+0.022 vs -0.002 pour le meilleur AC-*)
- Meilleur Sharpe OOS moyen (1.222 vs 0.889)
- Meilleure retention (153% vs 92%)
- MaxDD controle (-21.7% vs -28.8%)
- L'asymetrie d'alpha (entree lente, sortie rapide) est structurellement robuste

SMA150+AsymA reste disponible pour les profils prudents (meilleur MaxDD: -17.6%) mais perd de la rank stability sur les windows recentes (2023-2025).

S9 (Adaptive Continuous) reste disponible comme outil d'exploration parametrique via l'UI, et genere les rendements absolus IS les plus eleves (906-968%). Il n'est pas recommande comme strategie de reference pour le walk-forward.

## Limitations Connues

1. **OnChain Score**: Proxy base sur les prix (pas de donnees on-chain reelles)
2. **Sentiment**: Historique Fear & Greed limite a ~365 jours — proxy V2 (4 composants) au-dela. Le proxy V2 a un KS shift plus eleve que V1 (regime-dependant) mais impact negligeable (10% poids)
3. **Granularite**: Donnees journalieres uniquement
4. **ETH avant 2017**: Donnees ETH limitees (Binance listing 2017), backtests multi-asset demarrent a 2017
5. **Bear 2022**: Sharpe -1.3 pour toutes les configs. Tail risks exogenes (Luna/FTX) non-predictibles par proxy — normal, le DI production a ML signals + governance manuelle
6. **Cycle correction V2**: Legere degradation sur 2024-25 (-0.07 KS) due au changement structurel du cycle BTC (ETF spot, adoption institutionnelle)

## Performance Typique (Full History 2017-2025)

Comparaison des meilleures strategies sur Full History (weekly rebalance, $10k initial):

| Strategie | Type | Sharpe | MaxDD | Return | Avg Risky |
|-----------|------|--------|-------|--------|-----------|
| Cons+AsymA (3-asset) | rotation | **1.042** | **-29.8%** | 542% | 28% |
| Rot_default (3-asset) | rotation | 0.965 | -41.3% | 692% | 35.5% |
| AC-Default (3-asset) | continuous | 0.906 | -50.8% | **906%** | 43% |
| TG no whipsaw (2-asset) | trend gate | 0.955 | -54.4% | 1586% | 55% |
| Replica V2.1 (2-asset) | replica | 0.759 | -57.3% | 538% | 40% |
| BTC Buy & Hold (benchmark) | — | — | -83% | ~9500% | 100% |

**Key findings:**

- **Cons+AsymA** est la strategie recommandee : meilleur Sharpe (1.042), meilleure protection bear (-29.8% MaxDD), meilleure capture asymmetry OOS (+0.022)
- Adaptive Continuous (S9) genere les rendements absolus les plus eleves (906% IS) mais avec un profil de risque plus agressif (MaxDD -51%, asymetrie negative OOS)
- L'exploration parametrique de S9 (8 variantes, 288 backtests) confirme que la limitation est structurelle (allocation moyenne trop haute), pas parametrique
- TrendGate no whipsaw offre le meilleur return absolu mais avec plus de risque
- DI modulation n'apporte pas de valeur ajoutee pour l'allocation (desactivee par defaut)
- Smoothing rapide (alpha 0.30) ou instant (1.0) performent mieux que le smoothing lent
- Les phases discretes de S8 agissent comme un filtre naturel de double confirmation que le mapping continu de S9 ne reproduit pas

## Voir Aussi

- [DECISION_INDEX_V2.md](DECISION_INDEX_V2.md) - Formule complete du DI
- [RISK_SEMANTICS.md](RISK_SEMANTICS.md) - Semantique du Risk Score
- [architecture.md](architecture.md) - Architecture globale SmartFolio

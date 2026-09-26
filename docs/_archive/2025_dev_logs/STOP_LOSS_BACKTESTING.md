# Stop Loss Backtesting - Guide Complet

> **Date:** Octobre 2025
> **Status:** Ready to Test
> **Module:** ML Bourse - Stop Loss Validation

## Objectif

Valider empiriquement que **ATR 2x > Fixed %** sur données historiques réelles avant d'investir dans des améliorations plus complexes (Support Detection, Personnalisation).

## Fichiers Créés

### Backend (Python)
```
services/ml/bourse/
├── stop_loss_backtest.py          [NOUVEAU - 470 lignes]
│   └── StopLossBacktest          Main backtesting class
├── test_backtest.py               [NOUVEAU - 180 lignes]
│   └── Quick test script (3 assets)
└── generate_backtest_report.py    [NOUVEAU - 280 lignes]
    └── HTML report generator

run_backtest_standalone.py         [NOUVEAU - Wrapper script]
```

### Documentation
```
docs/STOP_LOSS_BACKTESTING.md      [CE FICHIER]
```

---

## Comment Utiliser

### Option 1 : Installation Propre (Recommandé)

Si vous n'avez pas `torch` installé et que vous rencontrez des erreurs d'import :

```bash
# Activer venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/Mac

# Installer dépendances manquantes (optionnel, seulement si erreurs)
pip install torch  # OU commentez les imports dans services/ml/__init__.py

# Exécuter backtest
python run_backtest_standalone.py

# Générer rapport HTML
python services/ml/bourse/generate_backtest_report.py
```

### Option 2 : Test Manuel (Si Import Issues)

Si les imports posent problème, voici un script minimal standalone :

```python
# test_minimal.py
import sys
sys.path.insert(0, 'd:\\Python\\smartfolio')

# Import direct (bypass __init__.py)
import importlib.util

spec = importlib.util.spec_from_file_location(
    "stop_loss_backtest",
    "d:\\Python\\smartfolio\\services\\ml\\bourse\\stop_loss_backtest.py"
)
stop_loss_backtest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stop_loss_backtest)

# Initialize
backtester = stop_loss_backtest.StopLossBacktest(
    cache_dir="data/cache/bourse",
    market_regime="Bull Market",
    timeframe="short"
)

# Run test
results = backtester.run_multi_asset_backtest(
    symbols=["AAPL", "NVDA", "SPY"],
    lookback_days=180,
    entry_interval_days=7
)

# Print aggregate
print(results['aggregate'])
```

---

## Fonctionnement du Backtest

### 1. Stratégie de Simulation

**Logique :**
```python
# Entrées : Tous les 7 jours (entry_interval_days)
for entry_date in dates[::7]:
    entry_price = close[entry_date]

    # Calcul stop loss selon méthode
    if method == "atr_2x":
        atr = calculate_atr(historical_14d)
        stop_loss = entry_price - (atr × 2.5)  # Bull Market multiplier
    elif method == "fixed_pct":
        stop_loss = entry_price × (1 - 0.05)  # 5% for short timeframe

    # Target profit
    target = entry_price × 1.08  # 8% target

    # Suivi pendant 30 jours (holding_period_days)
    for day in next_30_days:
        if low[day] <= stop_loss:
            exit_reason = "stop_loss"
            exit_price = stop_loss
            break
        if high[day] >= target:
            exit_reason = "target_reached"
            exit_price = target
            break

    # Calcul P&L
    pnl_pct = (exit_price - entry_price) / entry_price
    pnl_usd = pnl_pct × 100  # Assume 100 shares
```

### 2. Métriques Calculées

**Par Asset :**
- `total_trades` : Nombre de trades simulés
- `win_rate` : % de trades gagnants
- `avg_pnl_pct` : P&L moyen par trade
- `total_pnl_usd` : P&L cumulé (assume 100 shares)
- `stops_hit_pct` : % de sorties via stop loss
- `targets_reached_pct` : % de sorties via target
- `avg_holding_days` : Durée moyenne de détention

**Aggregate (tous assets) :**
- `total_pnl_usd` : Somme de tous les P&L
- `avg_win_rate` : Moyenne des win rates
- `avg_stops_hit_pct` : Moyenne des stops touchés
- `assets_won` : Nombre d'assets où la méthode a gagné
- `pnl_improvement_pct` : % d'amélioration ATR vs Fixed

### 3. Assets Testés

**3 profils de volatilité :**

| Asset | Type | Vol Annuelle | Profil |
|-------|------|--------------|--------|
| **AAPL** | Stock | ~25-30% | Modéré (blue chip) |
| **NVDA** | Stock | ~40-50% | Élevé (tech volatile) |
| **SPY** | ETF | ~15-20% | Faible (market index) |

**Période testée :** 180 jours (6 mois) de données historiques

**Fréquence d'entrée :** Tous les 7 jours (~26 trades par asset)

---

## Résultats Attendus

### Scénario 1 : ATR Supérieur ( Objectif)

```
 AGGREGATE RESULTS (3 assets):

  ┌─────────────────────┬──────────────┬──────────────┐
  │ Metric              │ ATR 2x       │ Fixed %      │
  ├─────────────────────┼──────────────┼──────────────┤
  │ Total P&L (all)     │     $+3,420  │     $+2,150  │
  │ Avg Win Rate        │        62.0% │        58.0% │
  │ Avg Stops Hit %     │        18.0% │        28.0% │ ← Moins de sorties prématurées
  │ Assets Won          │            3 │            0 │
  └─────────────────────┴──────────────┴──────────────┘

   Overall Winner: ATR 2x
   P&L Difference: $+1,270 (+59.1%)
```

**Verdict :** [OK] ATR 2x validé → Procéder à Phase 2 (Support Detection)

---

### Scénario 2 : Résultats Mixtes ( Investigation)

```
 AGGREGATE RESULTS (3 assets):

  ┌─────────────────────┬──────────────┬──────────────┐
  │ Total P&L (all)     │     $+2,800  │     $+2,650  │
  │ Avg Win Rate        │        60.0% │        59.5% │
  │ Avg Stops Hit %     │        20.0% │        23.0% │
  │ Assets Won          │            2 │            1 │
  └─────────────────────┴──────────────┴──────────────┘

   Overall Winner: ATR 2x
   P&L Difference: $+150 (+5.7%)
```

**Verdict :** [Warning] Amélioration marginale (<10%) → Analyser asset par asset pour comprendre

**Actions :**
- Si NVDA (high vol) : ATR gagne beaucoup → ATR utile pour assets volatils
- Si SPY (low vol) : Fixed gagne → ATR trop large pour assets stables
- **Solution** : Utiliser Fixed % pour ETFs, ATR pour stocks volatils

---

### Scénario 3 : Fixed Supérieur ( Problème)

```
 AGGREGATE RESULTS (3 assets):

  ┌─────────────────────┬──────────────┬──────────────┐
  │ Total P&L (all)     │     $+1,900  │     $+2,800  │
  │ Avg Win Rate        │        55.0% │        61.0% │
  │ Avg Stops Hit %     │        30.0% │        22.0% │ ← ATR a PLUS de sorties
  │ Assets Won          │            0 │            3 │
  └─────────────────────┴──────────────┴──────────────┘

   Overall Winner: Fixed %
   P&L Difference: $-900 (-32%)
```

**Verdict :** [Error] ATR underperforms → Investigation requise

**Causes possibles :**
1. **Multiplier trop élevé** (2.5x) → Tester 2.0x ou 1.5x
2. **Market regime incorrect** → Vérifier si période test = Bear market
3. **Assets inadaptés** → Tester sur plus d'assets (10-15)
4. **Bugs de calcul** → Vérifier ATR calculation
5. **Target trop ambitieux** (8%) → Tester 5% target

---

## Analyse Post-Backtest

### 1. Fichiers Générés

**JSON Results :**
```bash
data/backtest_results.json  # Raw results (can be analyzed programmatically)
```

**HTML Report :**
```bash
static/backtest_report.html  # Visual report (open in browser)
```

### 2. Analyse des Trades Individuels

Si résultats mixtes, analyser les trades détaillés :

```python
import json

with open('data/backtest_results.json', 'r') as f:
    results = json.load(f)

# Analyser NVDA (high vol asset)
nvda = [r for r in results['individual_results'] if r['symbol'] == 'NVDA'][0]

print(f"NVDA - ATR vs Fixed:")
print(f"  ATR: {nvda['atr_2x']['total_pnl_usd']} USD")
print(f"  Fixed: {nvda['fixed_pct']['total_pnl_usd']} USD")
print(f"  Winner: {nvda['comparison']['winner']}")
print(f"  Verdict: {nvda['comparison']['verdict']}")

# Examiner stops prématurés
print(f"\nStops Hit:")
print(f"  ATR: {nvda['atr_2x']['stops_hit_pct']*100:.1f}%")
print(f"  Fixed: {nvda['fixed_pct']['stops_hit_pct']*100:.1f}%")
```

### 3. Statistiques Avancées (Optionnel)

```python
# Sharpe ratio par méthode
import numpy as np

def calculate_sharpe(trades):
    returns = [t['pnl_pct'] for t in trades]
    return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

# Max drawdown par méthode
def calculate_max_drawdown(trades):
    cumulative = np.cumsum([t['pnl_usd'] for t in trades])
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return drawdown.min()
```

---

## Prochaines Étapes

### Si ATR > Fixed (+10% ou plus) :

[OK] **Phase 2 : ATR-Anchored Support Detection**
- Implémenter détection MA50 + niveaux psychologiques
- Ajustement limité ±2% du stop ATR
- Backtest à nouveau pour mesurer amélioration

 **Temps estimé :** 2-3h

---

### Si Résultats Mixtes (+5% à +10%) :

[Warning] **Investigation Granulaire**
- Séparer assets par volatilité (high/medium/low)
- Tester multipliers ATR adaptatifs (1.5x, 2.0x, 2.5x)
- Comparer par régime de marché (Bull vs Bear)

 **Temps estimé :** 1 jour

---

### Si Fixed > ATR :

[Error] **Debug & Réajustement**
1. Vérifier calcul ATR (période 14 jours correct ?)
2. Tester sur période plus longue (365 jours au lieu de 180)
3. Tester multipliers plus conservateurs (1.5x au lieu de 2.5x)
4. Vérifier market regime (peut-être Bear au lieu de Bull)
5. Augmenter nombre d'assets testés (10-15 au lieu de 3)

 **Temps estimé :** 1-2 jours

---

## Troubleshooting

### Erreur : "No cache file found for {SYMBOL}"

**Cause :** Parquet file manquant dans `data/cache/bourse/`

**Solution :**
```python
# Télécharger manuellement via BourseDataFetcher
from services.risk.bourse.data_fetcher import BourseDataFetcher
from datetime import datetime, timedelta

fetcher = BourseDataFetcher()
end = datetime.now()
start = end - timedelta(days=365)

import asyncio
df = asyncio.run(fetcher.fetch_historical_prices("AAPL", start, end, source="yahoo"))
print(f"Downloaded {len(df)} days of AAPL data")
```

---

### Erreur : "Insufficient data for {SYMBOL}"

**Cause :** Cache file existe mais contient <30 jours de données

**Solution :** Réduire `lookback_days` dans le test :
```python
results = backtester.run_multi_asset_backtest(
    symbols=["AAPL"],
    lookback_days=90,  # Au lieu de 180
    entry_interval_days=7
)
```

---

### Warning : "Failed to simulate trade on {DATE}"

**Cause :** Données manquantes à cette date (gaps, weekends)

**Impact :** Non bloquant, ce trade est sauté

**Action :** Ignorer si < 10% des trades, investiguer si > 20%

---

## Références

### Code Source
- **Main Module** : [`services/ml/bourse/stop_loss_backtest.py`](../services/ml/bourse/stop_loss_backtest.py)
- **Test Script** : [`run_backtest_standalone.py`](../run_backtest_standalone.py)
- **Report Generator** : [`services/ml/bourse/generate_backtest_report.py`](../services/ml/bourse/generate_backtest_report.py)

### Documentation Liée
- **Stop Loss System** : [`docs/STOP_LOSS_SYSTEM.md`](STOP_LOSS_SYSTEM.md)
- **Bourse Risk Analytics** : [`docs/BOURSE_RISK_ANALYTICS_SPEC.md`](BOURSE_RISK_ANALYTICS_SPEC.md)

### Méthodologie
- **ATR (Average True Range)** : Wilder, J. Welles (1978). *New Concepts in Technical Trading Systems*
- **Backtesting Best Practices** : *Advances in Financial Machine Learning* by Marcos López de Prado

---

## Support

**Logs :** `logs/app.log` (5 MB rotatifs, 3 backups)

**Debug Mode :**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Questions / Bugs :** Voir `CLAUDE.md` pour contact

---

**[OK] Module prêt à tester - Temps total de développement : ~2h**

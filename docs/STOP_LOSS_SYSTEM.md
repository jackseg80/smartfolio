# Système de Stop Loss Intelligent

> **Date:** Octobre 2025
> **Status:** Production
> **Module:** ML Bourse Recommendations
> **Update:** Trailing Stop ajouté (Oct 2025)

## Vue d'ensemble

Le système de stop loss intelligent calcule des niveaux de stop loss optimaux en utilisant **6 méthodes différentes** au lieu d'un simple pourcentage fixe. Cela permet d'adapter le stop loss à la volatilité de chaque asset, au régime de marché actuel, et aux gains latents des positions.

**Méthodes principales :**
- **Trailing Stop** (prioritaire pour positions legacy >20% gain)
- **Fixed Variable** (recommandée pour positions standard - validée par backtest +8% vs Fixed 5%)

## Architecture

### Backend

**Fichiers principaux :**
- `services/stop_loss/trailing_stop_calculator.py` - Calculateur trailing stop (générique, réutilisable)
- `services/ml/bourse/stop_loss_calculator.py` - Calculateur multi-méthodes (6 méthodes)
- `services/ml/bourse/price_targets.py` - Intégration dans les price targets
- `services/ml/bourse/recommendations_orchestrator.py` - Orchestration
- `connectors/saxo_import.py` - Extraction avg_price depuis CSV
- `adapters/saxo_adapter.py` - Propagation avg_price

### Frontend

**Fichiers modifiés :**
- `static/saxo-dashboard.html` - Tableau comparatif + badge  legacy + modal highlight

---

## Méthodes de calcul

### 0. Trailing Stop (NEW - Oct 2025)

**Applicable uniquement aux positions legacy avec gains significatifs (>20%).**

**Concept :** Protéger les gains latents en utilisant un stop loss plus large basé sur l'All-Time High (ATH) au lieu du prix actuel.

**Gain Tiers :**
- **0-20%** : Non applicable (utilise Fixed Variable)
- **20-50%** : Trailing -15% from ATH
- **50-100%** : Trailing -20% from ATH
- **100-500%** : Trailing -25% from ATH
- **>500%** : Trailing -30% from ATH (legacy)

**Exemple concret :**
```python
Position AAPL:
  Prix entrée: $91.90 (il y a plusieurs années)
  Prix actuel: $262.82
  Gain latent: +186% (+$170.92 par action)
  ATH (estimé 365j): $265.29

  Tier: 100-500% → Trailing -25%
  Stop loss: $265.29 × 0.75 = $198.97

  Vs Fixed Variable (6%): $247.05
  → Économie: $48.08 par action de "breathing room"
  → Gain minimum protégé: +117% au lieu de +169%
```

**Estimation ATH :**
- Utilise `max(price_history['high'].tail(365))` - pas de tracking temps réel
- Lookback par défaut: 365 jours (configurable)
- Conservatif: `ATH = max(historical_high, current_price)`

**Priorité :** Méthode #1 (prioritaire sur Fixed Variable pour positions legacy)

**Documentation complète :** [TRAILING_STOP_IMPLEMENTATION.md](TRAILING_STOP_IMPLEMENTATION.md)

---

### 1. Fixed Variable (Recommandé )

**Formule :**
```python
# Bucket basé sur volatilité annuelle
if annual_volatility > 0.40:
    stop_pct = 0.08  # High vol
elif annual_volatility > 0.25:
    stop_pct = 0.06  # Moderate vol
else:
    stop_pct = 0.04  # Low vol

stop_loss = current_price × (1 - stop_pct)
```

**Buckets de volatilité :**
- **Low vol (<25%)** : Stop 4% - Assets stables (KO, SPY, ETFs bonds)
- **Moderate vol (25-40%)** : Stop 6% - Majorité des stocks (AAPL, NVDA, MSFT)
- **High vol (>40%)** : Stop 8% - Assets erratiques (TSLA, PLTR, crypto)

**Avantages :**
- [OK] **Simplicité** : 3 règles simples (4-6-8%)
- [OK] **S'adapte à la volatilité** de l'asset
- [OK] **Validé empiriquement** : Backtest sur 372 trades, 6 assets, 1-5 ans
- [OK] **Performance supérieure** : +8% vs Fixed 5%, +156% vs ATR 2x
- [OK] **Évite over-optimization** : Pas de paramètres complexes

**Résultats backtest (Oct 2025) :**
```
Fixed Variable:  $105,232  [OK] WINNER (+8.0% vs Fixed 5%)
Fixed 5%:        $ 97,642  (-7.2% vs Fixed Var)
ATR 2x:          $ 41,176  (-60.9% vs Fixed Var)
```

**Exemple :**
```
NVDA (vol 30% moderate):
- Annual volatility = 30%
- Bucket = "moderate" (25-40%)
- Stop Loss = current × (1 - 0.06) = -6%
- Prix $182.16 → Stop $171.23
```

**Détails complets :** Voir `docs/STOP_LOSS_BACKTEST_RESULTS.md`

---

### 2. ATR-based (Adaptive)

**Formule :**
```python
stop_loss = current_price - (ATR_14d × multiplier)
```

**Multiplicateurs adaptatifs :**
- Bull Market: 2.5× (plus de room)
- Correction: 2.0× (neutre)
- Bear Market: 1.5× (plus serré)

**Avantages :**
- [OK] S'adapte à la volatilité de l'asset
- [OK] Asset volatile (TSLA) = stop plus large
- [OK] Asset stable (KO) = stop plus serré
- [OK] Méthode professionnelle standard

**Exemple :**
```
NVDA (vol 40%):
- ATR 14d = $3.43
- Multiplier = 2.5 (Bull Market)
- Stop Loss = $182.16 - ($3.43 × 2.5) = $175.30 (-3.8%)
```

---

### 2. Technical Support

**Formule :**
```python
support_levels = [MA20, MA50]
stop_loss = closest_support_below_current_price
```

**Avantages :**
- [OK] Basé sur niveaux techniques réels
- [OK] Respecte les supports clés
- [Warning] Peut être trop proche ou trop loin

**Exemple :**
```
AAPL:
- Current: $175.50
- MA20: $178.50 (au-dessus, pas utilisable)
- MA50: $172.30 (en dessous)
- Stop Loss = $172.30 (-1.8%)
```

---

### 3. Volatility 2σ (Statistical)

**Formule :**
```python
daily_volatility = std_dev(returns) / sqrt(252)
stop_loss = current_price × (1 - 2 × daily_volatility)
```

**Avantages :**
- [OK] Approche statistique pure
- [OK] 95% de couverture (2 écarts-types)
- [Warning] Peut être trop large pour assets très volatils

**Exemple :**
```
TSLA (vol 50%):
- Daily vol = 50% / sqrt(252) = 3.15%
- Stop Loss = current × (1 - 2 × 3.15%) = -6.3%
```

---

### 4. Fixed Percentage (Legacy)

**Formule :**
```python
stop_loss = current_price × (1 - fixed_pct)
# short: 5%, medium: 8%, long: 12%
```

**Avantages :**
- [OK] Simple et prévisible
- [Error] Ne s'adapte PAS à la volatilité
- [Error] Peut être trop serré ou trop large

**Utilisation :**
- Fallback si pas assez de données historiques
- Méthode legacy (ancienne version)

---

## Sélection de la méthode recommandée

**Priorité (Mise à jour Oct 2025) :**
```
1. Fixed Variable (TOUJOURS - gagnant backtest)
2. ATR-based (si ≥15 jours de données)
3. Technical Support (si ≥50 jours de données)
4. Volatility 2σ (si ≥30 jours de données)
5. Fixed % (fallback legacy)
```

**Code :**
```python
def _determine_best_method(self, stop_loss_levels):
    # NEW: Fixed Variable always wins (validated by backtest)
    if "fixed_variable" in stop_loss_levels:
        return "fixed_variable"  # Priorité 1
    elif "atr_2x" in stop_loss_levels:
        return "atr_2x"  # Priorité 2
    elif "technical_support" in stop_loss_levels:
        return "technical_support"  # Priorité 3
    elif "volatility_2std" in stop_loss_levels:
        return "volatility_2std"  # Priorité 4
    else:
        return "fixed_pct"  # Fallback
```

---

## Badge de qualité

Chaque méthode a un badge de qualité :

| Méthode | Qualité | Raison |
|---------|---------|--------|
| **Fixed Variable** | **HIGH** | [OK] Gagnant backtest, simple, adaptatif (4-6-8%) |
| ATR 2x | **MEDIUM** | S'adapte mais complexe, perdu backtest -60% |
| Technical Support | **MEDIUM** | Basé sur TA réel mais peut être imprécis |
| Volatility 2σ | **MEDIUM** | Statistiquement valide mais générique |
| Fixed % | **LOW** | Ne s'adapte pas, méthode simpliste legacy |

---

## Take Profits Adaptatifs (Option C)

**Implémenté :** Octobre 2025
**Validation :** Aligné avec système Fixed Variable

### Principe

Au lieu de TP fixes (+8% / +15%), les TP sont calculés comme **multiples du risque** pour garantir des R/R minimums.

**Formule :**
```python
risk = current_price - stop_loss
tp1 = current_price + (risk × tp1_multiplier)
tp2 = current_price + (risk × tp2_multiplier)
```

### Multiples adaptatifs par volatilité

```python
TP_MULTIPLIERS = {
    "low":      {"tp1": 2.0, "tp2": 3.0},   # Assets stables
    "moderate": {"tp1": 1.5, "tp2": 2.5},   # Majorité stocks
    "high":     {"tp1": 1.2, "tp2": 2.0}    # Assets erratiques
}
```

### Rationale par bucket

| Volatilité | Stop | TP1 Multiple | TP2 Multiple | Logique |
|------------|------|--------------|--------------|---------|
| **Low (<25%)** | -4% | 2.0x | 3.0x | Mouvements prévisibles → viser plus loin |
| **Moderate (25-40%)** | -6% | 1.5x | 2.5x | Équilibré |
| **High (>40%)** | -8% | 1.2x | 2.0x | Erratique → prendre profits vite |

### Exemples concrets

**SPY (Low vol 18%) :**
```
Prix : $575.00
Stop : $552.00 (-4%)
Risk : $23.00

TP1 = $575 + ($23 × 2.0) = $621.00 (+8%)
TP2 = $575 + ($23 × 3.0) = $644.00 (+12%)

R/R TP1 = 2.00 [OK]
R/R TP2 = 3.00 [OK]
```

**NVDA (Moderate vol 30%) :**
```
Prix : $182.16
Stop : $171.23 (-6%)
Risk : $10.93

TP1 = $182.16 + ($10.93 × 1.5) = $198.56 (+9%)
TP2 = $182.16 + ($10.93 × 2.5) = $209.49 (+15%)

R/R TP1 = 1.50 [OK]
R/R TP2 = 2.50 [OK]
```

**TSLA (High vol 44%) :**
```
Prix : $448.98
Stop : $413.06 (-8%)
Risk : $35.92

TP1 = $448.98 + ($35.92 × 1.2) = $492.08 (+9.6%)
TP2 = $448.98 + ($35.92 × 2.0) = $520.82 (+16%)

R/R TP1 = 1.20 [Warning] (limite acceptable)
R/R TP2 = 2.00 [OK]
```

### Avantages vs TP fixes

**Avant (système ancien) :**
```python
# Timeframe "medium"
tp1 = current_price × 1.08  # +8% pour TOUS
tp2 = current_price × 1.15  # +15% pour TOUS

# Résultat : R/R uniformes
# - Low vol + stop 4% → R/R = 8/4 = 2.00
# - Moderate vol + stop 6% → R/R = 8/6 = 1.33
# - High vol + stop 8% → R/R = 8/8 = 1.00
```

**Après (Option C) :**
```python
# Adaptatif selon volatilité
risk = current_price - stop_loss
tp1 = current_price + (risk × multipliers[vol_bucket]["tp1"])
tp2 = current_price + (risk × multipliers[vol_bucket]["tp2"])

# Résultat : R/R garantis minimums
# - Low vol → R/R ≥ 2.00
# - Moderate vol → R/R ≥ 1.50
# - High vol → R/R ≥ 1.20
```

### Impact sur le portfolio

**Distribution R/R observée (après implémentation) :**
```
R/R 2.00 : 9 positions  (32%) - Low vol assets
R/R 1.50 : 11 positions (39%) - Moderate vol assets
R/R 1.20 : 4 positions  (14%) - High vol assets
N/A      : 4 positions  (14%)

→ 70% du portfolio avec R/R ≥ 1.50 [OK]
```

### Fichiers modifiés

**Backend :**
- `services/ml/bourse/price_targets.py` (lignes 134-164)
  - Méthode `_calculate_buy_targets()` : TP adaptatifs
  - Méthode `_calculate_hold_targets()` : Même logique

**Code exemple :**
```python
# Get volatility bucket from Fixed Variable stop loss
vol_bucket = stop_loss_analysis["stop_loss_levels"]["fixed_variable"]["volatility_bucket"]

# TP multipliers
TP_MULTIPLIERS = {
    "low": {"tp1": 2.0, "tp2": 3.0},
    "moderate": {"tp1": 1.5, "tp2": 2.5},
    "high": {"tp1": 1.2, "tp2": 2.0}
}

multipliers = TP_MULTIPLIERS[vol_bucket]

# Calculate TP based on risk multiples
risk = current_price - stop_loss
tp1 = current_price + (risk × multipliers["tp1"])
tp2 = current_price + (risk × multipliers["tp2"])

# Override with technical resistance if better
if sr_levels and "resistance1" in sr_levels:
    tp1 = max(tp1, sr_levels["resistance1"])
```

### Bénéfices

1. [OK] **R/R minimums garantis** pour toutes les positions
2. [OK] **Plus de R/R uniformes** (1.33 partout)
3. [OK] **Cohérence avec stop loss** : Système complet basé volatilité
4. [OK] **Logique de trading réaliste** : Prendre profits plus vite sur high vol
5. [OK] **Simplicité** : Mêmes 3 buckets (low/moderate/high)

---

## Frontend - Affichage

### Tableau comparatif dans le modal

```
 Stop Loss Analysis (4 Methods Compared)

┌─────────────────────────────────────────────────────┐
│ Method            │ Price   │ Distance │ Max Loss │ Quality │
├─────────────────────────────────────────────────────┤
│ [OK] ATR 2x         │ $175.30 │  -3.8%   │  -€318  │ HIGH   │
│ (Recommended)     │         │          │         │        │
│ Technical Support │ $178.50 │  -2.0%   │  -€168  │ MEDIUM │
│ Volatility 2σ     │ $172.80 │  -5.1%   │  -€427  │ MEDIUM │
│ Fixed %           │ $171.64 │  -5.8%   │  -€493  │ LOW    │
└─────────────────────────────────────────────────────┘

 Why ATR 2x?
2.5× ATR below current. Adapts to asset volatility.
```

### Badge R/R dans le tableau principal

**Colonne R/R avec icônes :**
- [OK] Vert : R/R ≥ 2.0 (excellent)
- [Warning] Orange : R/R ≥ 1.5 (acceptable)
- [Error] Rouge : R/R < 1.5 (mauvais)

**Tri par défaut :** Descendant sur R/R (meilleurs trades en premier)

---

## Badge d'alerte R/R

Si R/R < 1.5, un badge d'alerte apparaît dans le modal :

```
[Warning] Poor Risk/Reward Ratio
Current R/R: 1:0.87 (minimum recommended: 1:1.5)
[Warning] Risk: 5.8% downside for only 5% upside
 Suggestion: Wait for better entry point or consider tighter stop loss
```

---

## API Response Structure

```json
{
  "price_targets": {
    "stop_loss": 175.30,
    "stop_loss_pct": -3.8,
    "take_profit_1": 191.27,
    "risk_reward_tp1": 1.89,
    "stop_loss_analysis": {
      "current_price": 182.16,
      "timeframe": "short",
      "market_regime": "Bull Market",
      "recommended_method": "atr_2x",
      "stop_loss_levels": {
        "atr_2x": {
          "price": 175.30,
          "distance_pct": -3.8,
          "atr_value": 3.43,
          "multiplier": 2.5,
          "reasoning": "2.5× ATR below current. Adapts to asset volatility.",
          "quality": "high"
        },
        "technical_support": {
          "price": 178.50,
          "distance_pct": -2.0,
          "level": "MA20",
          "reasoning": "MA20 support at $178.50",
          "quality": "medium"
        },
        "volatility_2std": {
          "price": 172.80,
          "distance_pct": -5.1,
          "volatility": 0.41,
          "reasoning": "2 std deviations for 41% annual volatility",
          "quality": "medium"
        },
        "fixed_pct": {
          "price": 171.64,
          "distance_pct": -5.8,
          "percentage": 0.05,
          "reasoning": "Simple 5% stop for short timeframe",
          "quality": "low"
        }
      }
    }
  }
}
```

---

## Cas d'usage

### 1. Asset volatile (NVDA, TSLA)

**Problème avec Fixed % :**
```
TSLA:
- Vol annuelle: 50%
- Fixed 5% stop = trop serré
- Sortie prématurée sur "noise" normal
```

**Solution avec ATR :**
```
TSLA:
- ATR 14d = $8.50
- Stop = current - (8.50 × 2.5) = -6.8%
- Plus large, s'adapte à la volatilité
```

### 2. Asset stable (KO, PG)

**Problème avec Fixed % :**
```
KO:
- Vol annuelle: 15%
- Fixed 5% stop = trop large
- Perte excessive avant sortie
```

**Solution avec ATR :**
```
KO:
- ATR 14d = $0.80
- Stop = current - (0.80 × 2.5) = -3.2%
- Plus serré, adapté à la faible volatilité
```

### 3. Position existante (HOLD)

**Utilité :**
Même pour les HOLD, le système calcule les stop loss :
- Monitoring : "Si prix passe sous X, réévaluer"
- Trailing stop : Protéger les gains
- Risk management : Connaître le risque actuel

---

## Configuration

### Multipliers ATR par régime

**Fichier :** `stop_loss_calculator.py:17-23`

```python
ATR_MULTIPLIERS = {
    "Bull Market": 2.5,      # Plus de room
    "Expansion": 2.5,
    "Correction": 2.0,       # Neutre
    "Bear Market": 1.5,      # Plus serré
    "default": 2.0
}
```

### Fixed percentages par timeframe

**Fichier :** `stop_loss_calculator.py:25-29`

```python
FIXED_STOPS = {
    "short": 0.05,   # 5% pour 1-2 semaines
    "medium": 0.08,  # 8% pour 1 mois
    "long": 0.12     # 12% pour 3-6 mois
}
```

---

## Tests

### Validation des 4 méthodes

```python
# Test avec NVDA (volatile)
calculator = StopLossCalculator(timeframe="short", market_regime="Bull Market")
result = calculator.calculate_all_methods(
    current_price=182.16,
    price_data=nvda_ohlc,
    volatility=0.40
)

assert result["recommended_method"] == "atr_2x"
assert result["stop_loss_levels"]["atr_2x"]["price"] < 182.16
assert result["stop_loss_levels"]["atr_2x"]["quality"] == "high"
```

### Validation du fallback

```python
# Test avec données insuffisantes
result = calculator.calculate_all_methods(
    current_price=100.0,
    price_data=None,  # Pas de données historiques
    volatility=None
)

# Doit fallback sur fixed_pct
assert result["recommended_method"] == "fixed_pct"
assert "atr_2x" not in result["stop_loss_levels"]
```

---

## Migration

### Avant (v1)

```python
# Stop loss fixe, pas d'adaptation
stop_loss = current_price * (1 - 0.05)  # Toujours 5%
```

### Après (v2)

```python
# Stop loss intelligent, adaptatif
calculator = StopLossCalculator(timeframe, market_regime)
analysis = calculator.calculate_all_methods(current_price, price_data, volatility)
stop_loss = analysis["stop_loss_levels"][analysis["recommended_method"]]["price"]
```

**Compatibilité :** Les anciens endpoints retournent toujours un stop loss simple, mais incluent maintenant `stop_loss_analysis` en bonus.

---

## Performance

### Calcul ATR

- **Temps moyen :** 2-5ms par asset
- **Données requises :** 15 jours minimum (idéal: 90 jours)
- **Cache :** Les données OHLC sont déjà cachées par `data_source`

### Impact frontend

- **Taille response :** +1-2 KB par recommendation
- **Rendu modal :** < 10ms (HTML statique)
- **Pas d'impact** sur le chargement initial du tableau

---

## Roadmap

### Phase 1 ( Complété)
- [x] Backend: StopLossCalculator avec 4 méthodes
- [x] Integration dans PriceTargets
- [x] Frontend: Tableau comparatif dans modal
- [x] Badge R/R + alerte si < 1.5

### Phase 2 (Future)
- [ ] Trailing stop automation
- [ ] Alertes SMS/Email quand prix approche SL
- [ ] Backtesting: "Si j'avais utilisé ATR vs Fixed, quelle différence ?"
- [ ] Portfolio-level stop: "Fermer tout si -15%"

### Phase 3 (Advanced)
- [ ] Machine learning pour optimiser les multipliers ATR
- [ ] Détection automatique des supports Fibonacci
- [ ] Stop loss adaptatif basé sur le sentiment de marché

---

## Références

- **ATR (Average True Range)** : Wilder, J. Welles (1978). *New Concepts in Technical Trading Systems*
- **2σ Rule** : Distribution normale, 95% de couverture
- **R/R minimum recommandé** : 1:1.5 (source: Van Tharp Institute)

---

## Changelog

**v1.0 (Octobre 2025)**
- Initial release
- 4 méthodes: ATR, Technical Support, Volatility 2σ, Fixed %
- Frontend: Tableau comparatif + badges R/R
- Adaptation au régime de marché

---

## Auteur

Système développé par Claude Code (Anthropic) en collaboration avec l'équipe.

**Contact :** Voir `CLAUDE.md` pour questions/bugs.

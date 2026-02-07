# Market Regime System

> **Version:** 2.0 - February 2026
> **Status:** Production
> **Scope:** Unified regime detection across all assets (Crypto + Stocks)

## Overview

SmartFolio uses a **unified 4-regime system** to classify market conditions. Every component
(backend ML, frontend allocation, recommendations, UI) uses the same canonical names and IDs.

### Canonical Regimes

| ID | Name | Score Range (0-100) | Color | CSS Key | Emoji |
| -- | ---- | ------------------- | ----- | ------- | ----- |
| 0 | Bear Market | 0-25 | `#dc2626` (red) | `bear-market` | Red circle |
| 1 | Correction | 26-50 | `#ea580c` (orange) | `correction` | Orange circle |
| 2 | Bull Market | 51-75 | `#22c55e` (green) | `bull-market` | Green circle |
| 3 | Expansion | 76-100 | `#3b82f6` (blue) | `expansion` | Blue circle |

Score = 0-100, where **100 = best market conditions**.

---

## Source of Truth

All regime-related code **must** import from these modules:

| Module | Language | Path |
| ------ | -------- | ---- |
| `regime_constants` | Python | `services/regime_constants.py` |
| `regime-constants` | JavaScript | `static/core/regime-constants.js` |

### Available Exports

**Python** (`from services.regime_constants import ...`):

- `MarketRegime(IntEnum)` - BEAR_MARKET=0, CORRECTION=1, BULL_MARKET=2, EXPANSION=3
- `REGIME_NAMES` - `['Bear Market', 'Correction', 'Bull Market', 'Expansion']`
- `REGIME_IDS` - Name-to-ID dict
- `REGIME_SCORE_RANGES` - `[(0, 25, 0), (26, 50, 1), (51, 75, 2), (76, 100, 3)]`
- `REGIME_COLORS` - ID-to-hex dict
- `REGIME_DESCRIPTIONS` - ID-to-dict with name, description, strategy, risk_level
- `LEGACY_TO_CANONICAL` - Legacy name mapping (see below)
- `score_to_regime(score)` - Score to `MarketRegime` enum
- `regime_name(regime_id)` - ID to display name
- `normalize_regime_name(name)` - Any legacy name to canonical
- `regime_to_key(name)` - Name to snake_case key (`'Bear Market'` -> `'bear_market'`)

**JavaScript** (`import { ... } from './core/regime-constants.js'`):

- `MarketRegime` - Frozen enum object
- `REGIME_NAMES`, `REGIME_IDS`, `REGIME_SCORE_RANGES`, `REGIME_COLORS`, `REGIME_EMOJIS`
- `LEGACY_TO_CANONICAL`
- `scoreToRegime(score)` - Returns `{ id, name, key }`
- `regimeName(id)` - ID to display name
- `normalizeRegimeName(name)` - Any legacy name to canonical
- `regimeToKey(name)` - Name to snake_case key

---

## Detection Architecture

Regimes are detected using a **Hybrid Rule-Based + ML** approach. Rules have priority for clear
cases; ML (HMM + Neural Network) handles nuanced situations.

```
Market Data (OHLCV)
    │
    ├─► Rule-Based Detection (high confidence, 85-95%)
    │   └─► Clear patterns: deep drawdowns, strong recoveries, clear trends
    │
    ├─► HMM (Hidden Markov Model)
    │   └─► Statistical regime clustering from returns + volatility
    │
    └─► Neural Network (RegimeClassifier)
        └─► Trained on labeled data, 4-class classification
    │
    └─► Final Regime = Rules if confident, else ML consensus
```

### Detection Priority

1. **Rule-based** (confidence 85-95%) - Fires first for clear patterns
2. **HMM + NN ensemble** - Used when no rule matches with sufficient confidence

### Per-Asset Thresholds

| Rule | Crypto (BTC/ETH) | Stocks (SPY) |
| ---- | ----------------- | ------------ |
| **Bear Market** | DD ≤ -30% + trend ≤ -10%, sustained 20d | DD ≤ -15%, sustained 30d |
| **Expansion** | Lookback DD ≤ -30% + trend ≥ +15%/month + DD < -15% | Recovery from -15%+ at +10%/month |
| **Bull Market** | DD > -20%, vol < 60%, trend > +10% | DD > -8%, vol < 18%, trend > +2% |
| **Bull (recovery)** | trend > +5%, vol < 65%, DD > -50% | N/A |
| **Correction** | (DD < -10% AND vol > 65%) OR (DD < -20% AND \|trend\| < 10%) | Fallback (no clear rule match) |

**Important**: Bear Market requires a **negative trend** (price actively declining), not just deep
drawdown from ATH. Without this, recovery rallies (e.g., BTC $16k→$70k in 2023-2024) were
misclassified as Bear Market because the drawdown from ATH ($69k) was still >30%.

### Regime History Smoothing

Both crypto and stock regime-history endpoints apply **minimum duration smoothing** after detection:

- **Crypto**: min 7 days (higher volatility allows faster regime changes)
- **Stocks**: min 14 days (regimes are inherently more stable)

Short-lived transitions (< min_duration) are absorbed by the longer neighboring regime.
This is implemented by `smooth_regime_sequence()` in `services/regime_constants.py`.

See also:

- [BTC_HYBRID_REGIME_DETECTOR.md](BTC_HYBRID_REGIME_DETECTOR.md) - Crypto details
- [HYBRID_REGIME_DETECTOR.md](HYBRID_REGIME_DETECTOR.md) - Stocks details

---

## ML Training Pipeline

Four models, three training entry points:

### Models

| Model Name | Type | Asset | Saved To |
| ---------- | ---- | ----- | -------- |
| `btc_regime_detector` | Neural Network | BTC/crypto | `models/regime/regime_neural_best.pth` |
| `btc_regime_hmm` | HMM | BTC/crypto | `models/regime/btc_regime_hmm.pkl` |
| `stock_regime_detector` | Neural Network | SPY/stocks | `models/stocks/regime/regime_neural_best.pth` |
| `volatility_forecaster` | LSTM | BTC/ETH/SOL | `models/volatility/{SYM}_volatility_best.pth` |

### Training Entry Points

1. **CLI Script**: `python scripts/train_models.py --real-data --days 3650 --epochs-regime 200`
   - Trains crypto models (btc_regime_detector, btc_regime_hmm, volatility_forecaster)
   - Default: 3650 days (10 years) to capture multiple full market cycles

2. **Admin Dashboard**: `POST /admin/ml/train/{model_name}`
   - Trains any model via background job (same underlying code)

3. **Auto-Trainer**: Scheduled via APScheduler
   - Regime models: daily at 3:00 AM UTC
   - Volatility models: daily at midnight UTC

All three entry points use the same training functions. No duplication.

### Training Labeling Heuristic

The Neural Network training uses a heuristic to label historical data points. The labeling
is aligned with the production rule-based thresholds and uses **drawdown as the primary signal**:

| Priority | Condition | Label |
| -------- | --------- | ----- |
| 1 | 30d drawdown ≤ -25% AND (downtrend OR negative return) | Bear Market |
| 2 | 30d return < -1.5×threshold AND strong downtrend (corr < -0.3) | Bear Market |
| 3 | 30d return > 1.5×threshold AND strong uptrend (corr > 0.4) | Expansion |
| 4 | Positive trend + positive return + DD > -15% | Bull Market |
| 5 | Clear uptrend (corr > 0.2) + DD > -20% | Bull Market |
| 6 | Fallback (everything else) | Correction |

Order matters: Bear (clearest signal) → Expansion (strong positive) → Bull → Correction (fallback).
Thresholds are crypto-appropriate: 15-20% drawdowns in 30d are normal even in bull markets.

---

## Three Concepts: Conditions vs Phase vs Regime

The UI displays three related but distinct concepts:

| Concept | Where Displayed | Source | What It Measures |
| ------- | --------------- | ------ | ---------------- |
| **Conditions** | DI Panel (analytics-unified) | Blended score WITHOUT Risk | Composite market outlook (CCS + OnChain) |
| **Phase** | DI Panel (analytics-unified) | Cycle score alone | Applied strategy (bearish/moderate/bullish) |
| **Regime** | Market Regimes page | ML detection per asset | Actual asset drawdown + volatility |

### Why Conditions ≠ Regime

- **Conditions** reflects a composite score from multiple signals (CCS, OnChain), excluding Risk
  Score (which measures portfolio robustness, not market direction)
- **Regime** detects actual drawdown per asset (e.g., BTC at -43% = Bear Market)
- It is normal to see Conditions = "Correction" while Regime BTC = "Bear Market"

### Regime Score Calculation

The Decision Index panel calculates a "regime score" that excludes Risk:

```javascript
// blendedScore = CCS×0.50 + OnChain×0.30 + Risk×0.20
// regimeScore removes Risk influence:
regimeScore = Math.floor((blendedScore - riskScore * 0.20) / 0.80)
```

The full blendedScore (with Risk) is still used for allocation calculations
(`calculateRiskBudget()`).

### Market Overrides (applyMarketOverrides)

After computing the base regime, `applyMarketOverrides()` applies protective adjustments:

1. **On-Chain Divergence**: `|cycleScore - onchainScore| >= 30` → +10% stables (with Schmitt trigger hysteresis: up=30, down=20 to prevent flip-flop)
2. **Low Risk Score**: `riskScore <= 30` → stables >= 50%

The divergence check uses `cycleScore` directly (not the blended score) to detect when the cycle model is bullish but on-chain indicators don't confirm. With blended, the 0.3×onchain weight dilutes the gap.

### Cycle Direction Penalty (V2.1)

`calculateRiskBudget()` applies a direction penalty when the cycle score is high (>80) but descending:

```javascript
direction_penalty = Math.max(0, -cycleDirection) * confidence * 0.15
risk_factor *= (1 - direction_penalty)
```

This distinguishes ascending phases (Month ~9, score=94, no penalty) from descending phases (Month ~21.5, score=94, ~9% risk_factor reduction). `cycleDirection` is the normalized sigmoid derivative ([-1, 1]) from `estimateCyclePosition()`.

See also: [DECISION_INDEX_V2.md](DECISION_INDEX_V2.md)

---

## Related Concepts (Not Regime)

### Bitcoin Cycle Phases

The cycle system (cycle-navigator, phase-engine) uses different terminology:

- Accumulation (0-6 months post-halving)
- Bull Build (7-18 months)
- Peak/Euphoria (19-24 months)
- Bear Market (25-36 months)
- Pre-Accumulation (37-48 months)

These are **cycle position** labels based on months since halving, NOT regime detection.
"Accumulation" here is a valid Bitcoin cycle term, distinct from the deprecated regime name.

### Phase Engine Tilts

The phase engine (`static/core/phase-engine.js`) produces allocation tilts:

- Risk-off, ETH Expansion, Large-cap Altseason, Full Altseason, Neutral

These are **allocation adjustment phases**, NOT market regimes.

---

## Legacy Name Migration (Feb 2026)

Before the unification, 5 different naming conventions existed:

| Convention | Names | Origin |
| ---------- | ----- | ------ |
| A (canonical) | Bear Market, Correction, Bull Market, Expansion | Current standard |
| B (training) | Bull, Bear, Sideways, Distribution | Old train_models.py |
| C (crypto cycle) | Accumulation, Expansion, Euphoria, Distribution | Old frontend |
| D (snake_case) | bull_market, bear_market, etc. | Orchestrator |
| E (French) | Euphorie, Contraction | Knowledge base |

All legacy names are mapped via `LEGACY_TO_CANONICAL` in both Python and JS modules.
The mapping is used by `normalize_regime_name()` / `normalizeRegimeName()` for backward
compatibility with cached data and external integrations.

### Key Legacy Mappings

| Legacy Name | Canonical Name | Reason |
| ----------- | -------------- | ------ |
| Accumulation | Bear Market | Was crypto cycle term, now replaced |
| Euphoria | Bull Market | Was crypto cycle term, now replaced |
| Sideways | Correction | Was HMM training label |
| Distribution | Expansion | Was crypto cycle term (inverted semantics) |
| Consolidation | Correction | Was used in alerts |
| neutral / unknown | Correction | Safe fallback |

### Critical Bugs Fixed

1. **Rebalancing Engine** (`rebalancing_engine.py`): regime==0 was treated as "Accumulation"
   (buy more) instead of "Bear Market" (reduce exposure)
2. **Training Script** (`train_models.py`): IDs were inverted (Bull=0 vs production Bear=0)
3. **Orchestrator** (`orchestrator.py`): Hardcoded `'bull_market'` and missing `'correction'`
4. **DI Regime Inflation**: Risk Score (portfolio robustness) was inflating the blended score,
   causing "Bull Market" display when CCS was bearish

---

## Frontend Integration

### CSS Classes

Use hyphenated class names for regime styling:

```css
.regime-chip.bear-market { color: #dc2626; }
.regime-chip.correction  { color: #ea580c; }
.regime-chip.bull-market  { color: #22c55e; }
.regime-chip.expansion    { color: #3b82f6; }
```

Defined in `static/css/risk-dashboard.css`.

### Allocation by Regime

Defined in `static/modules/market-regimes.js`:

| Regime | Stables Target | BTC | ETH | Alts | Meme Cap |
| ------ | -------------- | --- | --- | ---- | -------- |
| Bear Market | 40% | 35% | 15% | 10% | 0% |
| Correction | 25% | 35% | 20% | 20% | 0% |
| Bull Market | 15% | 30% | 25% | 25% | 5% |
| Expansion | 10% | 25% | 25% | 30% | 10% |

### Exposure Caps

Defined in `static/modules/targets-coordinator.js`:

| Regime | Min Exposure | Max Exposure |
| ------ | ------------ | ------------ |
| Bear Market | 20% | 40% |
| Correction | 40% | 70% |
| Bull Market | 60% | 85% |
| Expansion | 75% | 95% |

---

## API Endpoints

| Endpoint | Returns | Asset |
| -------- | ------- | ----- |
| `GET /api/ml/crypto/regime?benchmark=BTC` | Current crypto regime | BTC/ETH |
| `GET /api/ml/bourse/regime?benchmark=SPY` | Current stock regime | SPY |
| `GET /api/ml/crypto/regime-history?lookback_days=365` | Historical crypto regimes | BTC |
| `GET /api/ml/bourse/regime-history?lookback_days=7300` | Historical stock regimes | SPY |
| `GET /api/ml/predictions/live` | Live predictions incl. regime | All |

All endpoints return canonical names (`Bear Market`, `Correction`, `Bull Market`, `Expansion`).

---

## Testing

### Automated Tests

- `tests/integration/test_regime_consistency.py` - 28 tests verifying constants, mappings,
  score boundaries, normalization, colors
- `tests/unit/test_bourse_alerts.py` - 22 tests for alerts with canonical regime names
- `tests/integration/test_stop_loss_integration.py` - 30 tests across all 4 regimes

### Verification Commands

```bash
# Run regime tests
pytest -q tests/integration/test_regime_consistency.py

# Grep for legacy names (should return 0 outside LEGACY_TO_CANONICAL)
grep -rn "Accumulation\|Euphoria\|Sideways" --include="*.py" --include="*.js" services/ api/ static/ scripts/
```

---

## File Reference

### Source of Truth

| File | Role |
| ---- | ---- |
| `services/regime_constants.py` | Python constants, enums, utilities |
| `static/core/regime-constants.js` | JavaScript mirror of Python module |

### Key Backend Files

| File | Role |
| ---- | ---- |
| `services/ml/models/btc_regime_detector.py` | BTC hybrid rule+HMM detection |
| `services/ml/models/regime_detector.py` | Stock hybrid rule+HMM detection |
| `services/ml/models/rebalancing_engine.py` | Regime-aware portfolio rebalancing |
| `services/execution/signals.py` | Trading signal regime mapping |
| `services/ml/orchestrator.py` | ML prediction aggregation |
| `scripts/train_models.py` | Model training script (crypto) |

### Key Frontend Files

| File | Role |
| ---- | ---- |
| `static/modules/market-regimes.js` | Regime allocation + display data |
| `static/modules/dynamic-weighting.js` | Phase-based weight adjustment |
| `static/modules/targets-coordinator.js` | Exposure caps by regime |
| `static/core/unified-insights-v2.js` | Unified state + recommendations |
| `static/components/decision-index-panel.js` | DI display (Conditions label) |
| `static/css/risk-dashboard.css` | Regime chip styling |

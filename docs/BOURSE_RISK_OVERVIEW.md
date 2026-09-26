# Bourse Risk & Analytics - Vue d'ensemble

> **Status:** [OK] Production Ready (Phase 2.9 Complete)
> **Module:** Analyse de risque pour portefeuille actions Saxo Bank

>  **Documentation complète** : [BOURSE_RISK_ANALYTICS_SPEC.md](BOURSE_RISK_ANALYTICS_SPEC.md) — Spécification technique détaillée avec historique d'implémentation

---

## Objectif

Module **Risk & Analytics** pour portefeuille bourse (Saxo Bank) combinant :
- **Métriques classiques** : VaR, Sharpe, volatilité, drawdown
- **Intelligence prédictive** : ML pour signaux, régimes, volatilité forecast
- **Analytics avancés** : secteurs, FX exposure, concentration

---

## Architecture en 3 Piliers

### 1. Risk Classique
- VaR 95% à 1 jour (3 méthodes)
- Volatilité rolling (30j, 90j, 252j)
- Sharpe/Sortino ratio
- Max drawdown, Beta portfolio
- Liquidity score (0-100)

### 2. ML Prédictif
- Trend signal (-1 à +1)
- Volatility forecast (1d, 7d, 30d)
- Régime marché (bull/bear/sideways/high_vol)
- Sector rotation signals

### 3. Analytics Avancés
- Position VaR contribution
- Correlation matrix & clusters
- FX exposure (USD, EUR, CHF)
- Stress scenarios
- Concentration metrics

---

## Endpoints API Principaux

| Endpoint | Description |
|----------|-------------|
| `GET /api/risk/bourse/dashboard` | Dashboard complet |
| `GET /api/risk/bourse/var` | VaR avec breakdown par position |
| `GET /api/risk/bourse/correlation` | Matrice corrélations |
| `GET /api/risk/bourse/stress` | Stress testing |
| `GET /api/ml/bourse/regime` | Régime ML (SPY-based) |
| `GET /saxo/recommendations` | Recommandations BUY/HOLD/SELL |

---

## Fichiers Clés

**Backend:**
- `services/ml/bourse/` - ML modules (regime, volatility, recommendations)
- `api/risk_bourse_endpoints.py` - API router
- `services/ml/bourse/opportunity_scanner.py` - Market opportunities

**Frontend:**
- `static/saxo-dashboard.html` - Dashboard principal
- Onglets: Overview, Recommendations, Opportunities, Advanced Risk

---

## Intégration avec autres modules

- **Stop Loss System** : [STOP_LOSS_SYSTEM.md](STOP_LOSS_SYSTEM.md)
- **Market Opportunities** : [MARKET_OPPORTUNITIES_SYSTEM.md](MARKET_OPPORTUNITIES_SYSTEM.md)
- **Export System** : [EXPORT_SYSTEM.md](EXPORT_SYSTEM.md)

---

## Tests

```bash
pytest tests/unit/test_risk_bourse.py -v
pytest tests/integration/test_saxo_risk_flow.py -v
```

---

Pour les détails d'implémentation, phases, changelog, et spécifications API complètes, consulter [BOURSE_RISK_ANALYTICS_SPEC.md](BOURSE_RISK_ANALYTICS_SPEC.md).

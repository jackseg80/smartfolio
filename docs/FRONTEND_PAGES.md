# Frontend Pages — Inventaire

**Total** : 25 pages HTML production dans `static/`

---

## Pages Production (Principales)

| Page                  | Title                | Endpoints                          | Modules JS                                   | Statut |
|-----------------------|----------------------|------------------------------------|----------------------------------------------|--------|
| dashboard.html        | Portfolio Overview   | /balances/current, /portfolio/*    | wealth-saxo-summary.js                       | Prod   |
| analytics-unified.html| Analytics            | /api/ml/*, /balances/current       | unified-insights-v2.js, decision-index-panel | Prod   |
| risk-dashboard.html   | Risk Management      | /api/risk/*                        | risk-dashboard-store.js, GovernancePanel     | Prod   |
| simulations.html      | Simulation Engine    | local (riskStore, presets)         | simulation-engine.js, SimControls.js         | Prod   |
| rebalance.html        | Rebalancing          | /rebalance/plan                    | allocation-engine.js                         | Prod   |
| execution.html        | Execution            | /execution/*                       | execution-manager.js                         | Prod   |
| execution_history.html| Execution History    | /execution/history/*               | -                                            | Prod   |
| monitoring.html       | Monitoring           | /api/alerts/*                      | -                                            | Prod   |
| admin-dashboard.html  | Admin Dashboard      | /admin/*                           | -                                            | Prod   |
| login.html            | Login                | /auth/*                            | auth-guard.js                                | Prod   |

---

## Pages Risk

| Page | Title | Endpoints | Modules JS | Statut |
|------|-------|-----------|------------|--------|
| advanced-risk.html    | Advanced Risk        | /api/risk/advanced/*               | plotly.js                                    | Prod   |
| market-regimes.html   | Market Regimes       | /api/ml/crypto/*, /api/ml/bourse/* | -                                            | Prod   |
| cycle-analysis.html   | Cycle Analysis       | /api/ml/crypto/cycle/*             | bitcoin-cycle-chart.js                       | Prod   |

---

## Pages Analytics

| Page | Title | Endpoints | Modules JS | Statut |
|------|-------|-----------|------------|--------|
| di-backtest.html      | DI Backtest          | /api/di-backtest/*                 | -                                            | Prod   |
| optimization.html     | Portfolio Optimization| /api/portfolio/optimize/*          | -                                            | Prod   |
| portfolio-optimization-advanced.html | Advanced Optimization | /api/portfolio/optimize/* | plotly.js                           | Prod   |
| performance-monitor-unified.html | Performance Monitor | /api/performance/*           | -                                            | Prod   |

---

## Pages Wealth

| Page | Title | Endpoints | Modules JS | Statut |
|------|-------|-----------|------------|--------|
| wealth-dashboard.html | Wealth Dashboard     | /api/wealth/*                      | -                                            | Prod   |
| ai-dashboard.html     | AI Chat              | /ai/chat/*                         | ai-chat.js                                   | Prod   |

---

## Pages Bourse/Saxo

| Page | Title | Endpoints | Modules JS | Statut |
|------|-------|-----------|------------|--------|
| saxo-dashboard.html | Bourse Dashboard | /api/saxo/positions | wealth-saxo-summary.js | Prod |
| bourse-analytics.html | Bourse Analytics | /api/risk/bourse/*, /api/ml/bourse/* | plotly.js, chart.js | Prod |
| bourse-recommendations.html | Bourse Recommendations | /api/ml/bourse/portfolio-recommendations | - | Prod |
| saxo-upload.html | Import Saxo | /api/sources/* | sources-manager.js | Migrating |
| analytics-equities.html | Analytics Equities | /api/wealth/* | wealth-analytics.js | Beta |

**Menu Bourse (dropdown):**

- 📊 Dashboard → saxo-dashboard.html (Overview + Positions)
- 📈 Analytics → bourse-analytics.html (Risk Analysis + Advanced Analytics)
- 💡 Recommendations → bourse-recommendations.html (Recommendations + Market Opportunities)

---

## Pages Test/Debug (54)

| Préfixe | But                      | Exemple            | Statut |
|---------|---------------------------|--------------------|--------|
| test-   | Tests unitaires frontend  | test-di-history.html| Actif  |
| debug-  | Outils de diagnostic      | debug-risk.html     | Actif  |
| clear-  | Nettoyage caches/localStorage | clear-cache.html | Actif  |

**Liste partielle** :
- test-di-history.html
- test-phase-engine.html
- test-contradiction-unified.html
- debug-risk.html
- clear-cache.html
- (49 autres pages test/debug)

---

## Pages Utilities

| Page                  | But                      | Statut |
|-----------------------|---------------------------|--------|
| settings.html         | Configuration utilisateur | Prod   |
| alias-manager.html    | Gestion aliases assets    | Prod   |
| sources-manager.html  | Sources System v2         | Prod   |

---

## Pages Legacy/Deprecated

À documenter au fur et à mesure de l'audit. Critère : pages non référencées dans navigation principale et non maintenues depuis > 6 mois.

---

## Notes

- **Statut** : Prod (production), Beta (fonctionnel mais incomplet), Migrating (en transition), Legacy (déprécié)
- **25 pages production** dans `static/` (inventaire complet via `ls static/*.html`)
- **Test/Debug** : pages additionnelles avec préfixes test-, debug-, clear- dans sous-dossiers

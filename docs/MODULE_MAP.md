# Module Map — JavaScript

**Total** : ~120 modules JS dans `static/`

---

## Core Modules

| Module                                   | Exports clés              | Utilisé par                  | Notes |
|------------------------------------------|---------------------------|------------------------------|-------|
| static/core/risk-dashboard-store.js      | riskStore, getDecision()  | risk-dashboard.html, simulations.html | Store central Risk |
| static/core/unified-insights-v2.js       | computeUnifiedInsights()  | analytics-unified.html       | Prod, DI calcul |
| static/core/phase-engine.js              | detectPhase(), applyTilts() | unified-insights-v2.js     | Phase detection |
| static/core/storage-service.js           | StorageService            | auth-guard.js, pages         | **NEW Fév 2026** Abstraction localStorage |
| static/core/fetcher.js                   | safeFetch, apiCall, fetchCached | pages, modules      | **NEW Fév 2026** Point d'entrée fetch unifié |
| static/core/auth-guard.js                | checkAuth(), getAuthHeaders() | toutes pages auth      | Migré vers StorageService |

---

## Components

| Component                                | Exports                   | Pages                        | Notes |
|------------------------------------------|----------------------------|------------------------------|-------|
| static/components/decision-index-panel.js| renderDecisionIndexPanel() | analytics-unified.html, simulations.html | Trend chip + regime |
| static/components/GovernancePanel.js     | GovernancePanel (class)    | risk-dashboard.html          | Gouvernance |
| static/components/UnifiedInsights.js     | UnifiedInsights (class)    | analytics-unified.html       | Wrapper |

---

## Modules (Domaine)

| Module                                   | Exports                   | Domaine     | Notes |
|------------------------------------------|----------------------------|-------------|-------|
| static/modules/simulation-engine.js      | Fonctions simulation       | Simulation  | Aligne DI prod |
| static/modules/wealth-saxo-summary.js    | fetchSaxoSummary()         | Bourse      | dashboard & saxo-dashboard |
| static/modules/risk-dashboard-toasts.js  | showToast, showS3AlertToast | Risk       | **NEW Fév 2026** Extrait du controller |
| static/modules/risk-dashboard-alerts-history.js | loadAlertsHistory, formatAlertType | Risk | **NEW Fév 2026** Extrait du controller |
| static/utils/di-history.js               | getTodayCH(), makeKey()    | DI History  | Persistence localStorage |

---

## Utilities

| Utility                                  | Exports                   | Usage       | Notes |
|------------------------------------------|----------------------------|-------------|-------|
| static/global-config.js                  | API_BASE_URL               | Global      | Config centralisée |
| static/lazy-loader.js                    | lazyLoad()                 | Navigation  | Iframes lazy |

---

## Notes

- **debug-logger.js** : Non trouvé dans static/utils/ (peut-être inline ou autre emplacement)
- **~120 modules** : Inventaire complet disponible via `find static/ -name "*.js" -type f` (core/, components/, modules/, utils/, selectors/)
- **Exports vérifiés** : decision-index-panel (ligne 999), GovernancePanel (ligne 13), wealth-saxo-summary (fetchSaxoSummary)

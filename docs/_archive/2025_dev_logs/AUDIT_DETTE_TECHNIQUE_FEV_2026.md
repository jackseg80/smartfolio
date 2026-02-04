# Plan de Nettoyage - Dette Technique SmartFolio

## Décisions

- **Scope:** Plan complet (4 phases)
- **Redirections:** Supprimer banks-dashboard.html et banks-manager.html
- **Workflow:** À chaque fin de phase → màj plan, màj docs, tests, commit

## Résumé de l'Audit

| Catégorie | Fichiers | Impact |
|-----------|----------|--------|
| Code mort (HTML/JS/Python) | ~17 fichiers | 1800+ LOC à supprimer |
| Duplications services | 3 services | 800+ LOC redondants |
| Patterns JS répétés | 28 fichiers | Maintenance difficile |
| TODOs/FIXMEs | 61 fichiers | Documentation incomplète |

---

## Phase 1: Suppression Code Mort (Basse Risque) ✅ TERMINÉ

**Note:** `coingecko_safe.py` conservé car utilisé dans `smart_classification.py` (workaround bug aiohttp/CTRL+C)

### 1.1 Fichiers HTML à supprimer (7 fichiers)

| Fichier | Raison |
|---------|--------|
| `static/test_groq_settings.html` | Page de test API Groq |
| `static/test-memory-leak.html` | Diagnostic memory leak |
| `static/ui-components-demo.html` | Galerie composants legacy |
| `static/phase-engine-control.html` | Panel debug Phase Engine |
| `static/sources-unified-section.html` | Fragment HTML non intégré |
| `static/banks-dashboard.html` | Redirection obsolète |
| `static/banks-manager.html` | Redirection obsolète |

### 1.2 Fichiers JS à supprimer (5 fichiers)

| Fichier | Lignes | Raison |
|---------|--------|--------|
| `static/clear-onchain-cache.js` | 25 | Script console manuel |
| `static/memory-leak-diagnostic.js` | 212 | Outil diagnostic |
| `static/modules/export-button-old.js` | 330 | Remplacé par export-button.js |
| `static/ml-card-component.js` | 320 | Aucun import trouvé |
| `static/lazy-controller-loader.js` | 226 | Aucun import trouvé |

### 1.3 Fichiers Python à supprimer (1 fichier)

| Fichier | Lignes | Raison |
|---------|--------|--------|
| `api/execution_dashboard.py` | 447 | Non importé dans router_registration.py |

**Conservé:** `services/coingecko_safe.py` - utilisé par smart_classification.py (workaround aiohttp)

### Validation Phase 1
```bash
# Vérification pré-suppression
grep -r "export-button-old" static/
grep -r "ml-card-component" static/
grep -r "execution_dashboard" api/ services/

# Tests après suppression
pytest tests/unit -v && pytest tests/integration -v
```

---

## Phase 2: Consolidation Services Python

### 2.1 ML Pipeline Manager

**Action:** Supprimer `services/ml_pipeline_manager.py` (11 lignes - stub de compatibilité)

**Migration des imports:**
```python
# Ancien
from services.ml_pipeline_manager import RegimeClassifier, VolatilityPredictor

# Nouveau
from services.ml.models.regime_detector import RegimeClassificationNetwork as RegimeClassifier
from services.ml.models.volatility_predictor import VolatilityLSTM as VolatilityPredictor
```

### 2.2 Pricing Services

**Décision:** NE PAS consolider - architecture correcte
- `pricing.py`: Service core multi-provider
- `pricing_service.py`: Façade Wealth (importe pricing.py)

### 2.3 get_active_user() - Dépréciation

**Fichier:** [api/deps.py](api/deps.py) lignes 109-174

**Action:** Tracker les appels restants et planifier suppression (deadline: Mai 2026)

---

## Phase 3: Refactoring Patterns JS ✅ TERMINÉ

### Créé: `static/core/formatters.js`

Fonctions unifiées:
- `formatNumber(value, decimals)` - nombres avec séparateurs
- `formatCurrency(value, currency, decimals)` - multi-devises
- `formatMoney(usd)` - alias formatCurrency
- `formatUSD(value)` - USD simplifié
- `formatPercentage(value, decimals)` - pourcentages
- `formatDate(date, options)` - dates
- `formatRelativeTime(timestamp)` - temps relatif

### Fichiers migrés:
- `shared-ml-functions.js` → réexporte depuis formatters.js
- `risk-utils.js` → import + réexport
- `wealth-saxo-summary.js` → réexporte formatUSD
- `dashboard-main-controller.js` → utilise formatUSD

### Restant (méthodes de classe, non migrées):
- `ai-components.js` - formatNumber() méthode de classe
- `manual-source-editor.js` - formatNumber() méthode de classe

---

## Phase 4: Nettoyage Documentation ✅ TERMINÉ

### 4.1 legacy-redirects.js → SUPPRIMÉ
- Fichier non importé nulle part
- `performance-monitor.html` supprimé (uniquement référencé par legacy-redirects.js)

### 4.2 TODOs/FIXMEs Triés

**Priorité Haute (Sécurité):**
- `api/dependencies/dev_guards.py:215` - Validation JWT

**Priorité Moyenne (Fonctionnalités):**
- `api/admin_router.py:134` - Comptage modèles ML
- `api/admin_router.py:1201` - Lecture secrets.json avec masking
- `api/risk_bourse_endpoints.py:621` - Intégrer earnings calendar API
- `api/realtime_endpoints.py:253` - Brancher vrai broadcaster

**Priorité Basse (Aspirationnels - à garder pour plus tard):**
- `api/portfolio_endpoints.py:452` - Mode dry-run cleanup
- `api/portfolio_monitoring.py:315` - Vraie date dernière rebal
- `api/scheduler.py:578` - Correlation forecaster
- `services/alerts/alert_engine.py:1314` - Quiet hours
- `services/notifications/monitoring.py:212` - Autres métriques

---

## Fichiers Critiques à Modifier

1. [api/router_registration.py](api/router_registration.py) - Vérifier imports morts
2. [api/deps.py](api/deps.py) - Tracking get_active_user()
3. [static/core/fetcher.js](static/core/fetcher.js) - Étendre error handling
4. [static/components/legacy-redirects.js](static/components/legacy-redirects.js) - Nettoyer redirections

---

## Résultat Final

| Phase | Statut | Impact |
|-------|--------|--------|
| Phase 1 | ✅ Terminé | 13 fichiers, 3282 LOC supprimés |
| Phase 2 | ✅ Terminé | ML Pipeline conservé (compatibilité torch), get_active_user déjà migré |
| Phase 3 | ✅ Terminé | formatters.js créé, 5 fichiers migrés |
| Phase 4 | ✅ Terminé | 2 fichiers, 869 LOC supprimés |
| **Total** | **15 fichiers supprimés** | **~4200 LOC nettoyés** |

### Commits générés

1. `b2028ec` - refactor: supprimer 13 fichiers de code mort (Phase 1)
2. `fa77c3c` - refactor: supprimer legacy-redirects.js et performance-monitor.html (Phase 4)
3. `0392aae` - refactor: centraliser formatCurrency dans static/core/formatters.js (Phase 3)
4. `2a55719` - fix(settings): correction double déclaration formatUSD

---

## Vérification Finale

```bash
# Tests complets
pytest tests/unit -v && pytest tests/integration -v

# Démarrage serveur
python -m uvicorn api.main:app --port 8080

# Health check
curl http://localhost:8080/api/health
```

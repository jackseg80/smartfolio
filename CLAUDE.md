# CLAUDE.md — Guide Agent SmartFolio

> Version condensée pour agents IA. Dernière mise à jour: Fév 2026

## Règles Critiques

### 1. Multi-Tenant OBLIGATOIRE
```python
# Backend: TOUJOURS utiliser get_required_user (header X-User obligatoire)
from api.deps import get_required_user
from services.balance_service import balance_service

@app.get("/endpoint")
async def endpoint(user: str = Depends(get_required_user), source: str = Query("cointracking")):
    res = await balance_service.resolve_current_balances(source=source, user_id=user)

# ⚠️ DEPRECATED: get_active_user (fallback "demo" non sécurisé) → NE PAS UTILISER
```

```javascript
// Frontend: TOUJOURS utiliser window.loadBalanceData()
const balanceResult = await window.loadBalanceData(true);
// NE JAMAIS: fetch(`/balances/current?...`)
```

**Isolation:** `data/users/{user_id}/{source}/`

### 2. Authentication JWT
```javascript
// Frontend: Auth requise sur toutes les pages
import { checkAuth, getAuthHeaders } from './core/auth-guard.js';
await checkAuth();
const response = await fetch('/api/endpoint', { headers: getAuthHeaders() });
```

```python
# Backend: Endpoints protégés avec JWT
from api.deps import get_current_user_jwt
@router.get("/endpoint")
async def endpoint(user: str = Depends(get_current_user_jwt)): pass
```

**Règles:** Login requis, tokens JWT 7 jours, pas de user switcher sans re-login.
**Docs:** [`docs/AUTHENTICATION.md`](docs/AUTHENTICATION.md)

### 3. Système Dual de Scoring
| Métrique | Formule | Range | Usage |
|----------|---------|-------|-------|
| **Score de Régime** | `0.5×CCS + 0.3×OnChain + 0.2×Risk` | 0-100 | Communication du régime marché |
| **Decision Index** | `(C×w₁ + O×w₂ + R×w₃ + S×w₄) × phase_factor` | 0-100 | Score décisionnel stratégique |
| **Allocation Validity** | `total_check.isValid ? 65 : 45` | 65 ou 45 | Check technique V2 allocation |

**Règles:**
- Phase basée UNIQUEMENT sur cycle (<70=bearish, 70-90=moderate, ≥90=bullish)
- Régime "Expansion" (55) + Phase "bearish" (cycle 59<70) est NORMAL
- Risk Score = Positif (0-100), plus haut = plus robuste. NE JAMAIS inverser avec `100 - scoreRisk`
- **Decision Index ≠ Allocation Validity**: DI est continu (0-100), Validity est binaire (65/45)
- **Docs:** [`docs/DECISION_INDEX_V2.md`](docs/DECISION_INDEX_V2.md)

### 4. Design Responsive
- Full responsive: `max-width: none`, Grid `repeat(auto-fit, minmax(300px, 1fr))`
- Breakpoints: 768px (mobile), 1024px (tablet), 1400px (desktop), 2000px (XL)
- Pas de largeur fixe type `max-width: 1200px`

### 5. Autres Règles
- Ne jamais committer `.env` ou clés
- Pas d'URL API en dur → `static/global-config.js`
- Windows: `.venv\Scripts\Activate.ps1` avant tout
- Pas de `--reload` flag → demander restart manuel après modifs backend
- **Git commits:** NE JAMAIS ajouter "Co-Authored-By: Claude" dans les messages de commit

---

## Architecture Essentielle

### Pages Production
```
dashboard.html          # Vue globale + P&L Today
analytics-unified.html  # ML temps réel + Decision Index
risk-dashboard.html     # Risk metrics + quick links
cycle-analysis.html     # Bitcoin cycle analysis
rebalance.html         # Rebalancing + Optimization (6 algos Markowitz)
execution.html         # Exécution temps réel
simulations.html       # Simulateur complet
wealth-dashboard.html   # Patrimoine unifié
monitoring.html        # KPIs système + Alerts History
admin-dashboard.html    # Admin Dashboard (RBAC)
```

### API Namespaces
```
/balances/current, /portfolio/metrics    # Données portfolio
/api/ml/*, /api/risk/*                   # ML + Risk
/api/wealth/patrimoine/*, /api/wealth/*  # Patrimoine + Wealth
/api/sources/*                           # Sources System v2
/execution/governance/*                  # Decision Engine
/admin/*                                 # Admin (RBAC protected)
/proxy/fred/*                            # Macro indicators (DXY, VIX, Bitcoin)
```

### Fichiers Clés
```
api/main.py, api/admin_router.py, api/deps.py
services/balance_service.py, services/portfolio.py
services/execution/governance.py, services/ml/orchestrator.py
services/macro_stress.py                 # DXY/VIX stress → Decision Index penalty
static/global-config.js, static/core/allocation-engine.js
config/users.json
```

---

## Patterns de Code

### Endpoint API
```python
from api.deps import get_required_user
from api.utils import success_response, error_response
from services.balance_service import balance_service

@router.get("/metrics")
async def get_metrics(user: str = Depends(get_required_user), source: str = Query("cointracking")):
    res = await balance_service.resolve_current_balances(source=source, user_id=user)
    return success_response(res.get("items", []))
```

### Response Formatting
```python
from api.utils import success_response, error_response, paginated_response
return success_response(data, meta={"currency": "USD"})
return error_response("Not found", code=404)
return paginated_response(items, total=100, page=1, page_size=50)
```

### Safe ML Model Loading
```python
from services.ml.safe_loader import safe_pickle_load, safe_torch_load
model = safe_pickle_load("cache/ml_pipeline/models/my_model.pkl")
checkpoint = safe_torch_load("cache/ml_pipeline/models/regime.pth", map_location='cpu')
# NE JAMAIS: pickle.load() ou torch.load() direct
```

### Frontend Fetch Multi-Tenant
```javascript
const activeUser = localStorage.getItem('activeUser') || 'demo';
const response = await fetch('/api/risk/dashboard', { headers: { 'X-User': activeUser } });
// TOUJOURS passer X-User sur: /api/risk/*, /api/portfolio/*, /api/wealth/*, /api/saxo/*
```

---

## Structure Données

```
data/users/{user_id}/
  config.json           # Config utilisateur (clés API)
  cointracking/data/    # CSV versionnés automatiquement
  saxobank/data/        # CSV Saxo
  wealth/patrimoine.json # Patrimoine unifié
```

- Versioning: `YYYYMMDD_HHMMSS_{filename}.csv`
- P&L: `data/portfolio_history.json` (clé: user_id + source)

---

## Quick Checks

```bash
# Dev Server (Windows)
.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --port 8080

# Tests
pytest -q tests/unit && pytest -q tests/integration

# Redis
redis-cli ping  # ou: wsl -d Ubuntu bash -c "sudo service redis-server start"
```

### Cache TTL
- On-Chain: 4h | Cycle Score: 24h | ML Sentiment: 15min | Prix crypto: 3min | Risk Metrics: 30min | Macro Stress (DXY/VIX): 4h

---

## Pièges Fréquents

- Oublier `user_id` → Utiliser `Depends(get_required_user)`
- Hardcoder `user_id='demo'` → Dependency injection
- Importer de `api.main` → Utiliser `services.balance_service`
- `fetch()` sans header `X-User` sur endpoints user-specific
- Inverser Risk Score avec `100 - scoreRisk`
- Oublier restart serveur après modifs backend
- Response format incohérent → Utiliser `success_response()`/`error_response()`

---

## Features Avancées (Docs)

| Feature | Doc |
|---------|-----|
| Allocation Engine V2 | [`docs/ALLOCATION_ENGINE_V2.md`](docs/ALLOCATION_ENGINE_V2.md) |
| Portfolio Optimization | [`docs/PORTFOLIO_OPTIMIZATION_GUIDE.md`](docs/PORTFOLIO_OPTIMIZATION_GUIDE.md) |
| Risk Semantics | [`docs/RISK_SEMANTICS.md`](docs/RISK_SEMANTICS.md) |
| Stop Loss System | [`docs/STOP_LOSS_SYSTEM.md`](docs/STOP_LOSS_SYSTEM.md) |
| Stress Testing | [`docs/STRESS_TESTING_MONTE_CARLO.md`](docs/STRESS_TESTING_MONTE_CARLO.md) |
| Monte Carlo | [services/risk/monte_carlo.py](services/risk/monte_carlo.py) |
| Market Opportunities | [`docs/MARKET_OPPORTUNITIES_SYSTEM.md`](docs/MARKET_OPPORTUNITIES_SYSTEM.md) |
| AI Chat Global | [`docs/AI_CHAT_GLOBAL.md`](docs/AI_CHAT_GLOBAL.md) |
| Alerts System | [`docs/ML_ALERT_PREDICTOR_REAL_DATA_OCT_2025.md`](docs/ML_ALERT_PREDICTOR_REAL_DATA_OCT_2025.md) |
| Governance Freeze | [`docs/GOVERNANCE_FIXES_OCT_2025.md`](docs/GOVERNANCE_FIXES_OCT_2025.md) |
| Patrimoine | [`docs/PATRIMOINE_MODULE.md`](docs/PATRIMOINE_MODULE.md) |
| Sources V2 | [`docs/SOURCES_V2.md`](docs/SOURCES_V2.md) |
| Admin Dashboard | [`docs/ADMIN_DASHBOARD.md`](docs/ADMIN_DASHBOARD.md) |
| Export System | [`docs/EXPORT_SYSTEM.md`](docs/EXPORT_SYSTEM.md) |

---

## Admin Dashboard

**URL:** `admin-dashboard.html` | **RBAC:** Rôle `admin` requis (user "jack")

```python
# Backend: TOUJOURS utiliser require_admin_role
from api.deps import require_admin_role
@router.get("/admin/users")
async def list_users(user: str = Depends(require_admin_role)): pass
```

**Rôles:** admin (full), governance_admin, ml_admin, viewer (read-only)

**ML Training Réel:** btc_regime_detector, btc_regime_hmm, stock_regime_detector, volatility_forecaster
- Données: 730j (regime) / 365j (volatility)
- Fichiers: `models/regime/*.pth`, `models/registry.json`

---

## Commandes Utiles

```bash
# P&L snapshot
curl -X POST "localhost:8080/portfolio/snapshot?user_id=jack&source=cointracking"

# Compte bancaire
curl -X POST "localhost:8080/api/wealth/banks/accounts" -H "X-User: jack" \
  -d '{"bank_name":"UBS","balance":5000,"currency":"CHF"}'

# Test RBAC
curl "localhost:8080/admin/health" -H "X-User: jack"     # OK
curl "localhost:8080/admin/health" -H "X-User: demo"    # 403
```

---

## Docs Architecture

- Architecture: `docs/architecture.md`
- Audit: `docs/audit/` (voir README.md dans ce dossier)
- Logging: `docs/LOGGING.md`
- Redis: `docs/REDIS_SETUP.md`
- Security: `docs/SECURITY.md`
- Cache TTL: `docs/CACHE_TTL_OPTIMIZATION.md`

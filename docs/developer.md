# Guide Développeur SmartFolio

>  **Guide principal** : [CLAUDE.md](../CLAUDE.md) — Règles critiques pour agents IA

---

## Installation rapide

### Windows
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn api.main:app --port 8080
```

### Linux/Mac
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --port 8080
```

**Note:** Ne pas utiliser `--reload` en dev - demander restart manuel après modifs backend.

---

## Conventions Critiques

### Multi-Tenant OBLIGATOIRE
```python
# Backend: TOUJOURS utiliser dependencies
from api.deps import get_active_user
@router.get("/endpoint")
async def endpoint(user: str = Depends(get_active_user)): pass
```

```javascript
// Frontend: TOUJOURS utiliser window.loadBalanceData()
const data = await window.loadBalanceData(true);
```

### Response Formatting
```python
from api.utils import success_response, error_response
return success_response(data)  # Pas de format custom
return error_response("Not found", code=404)
```

### Safe ML Loading
```python
from services.ml.safe_loader import safe_pickle_load
model = safe_pickle_load("path/to/model.pkl")
# NE JAMAIS: pickle.load() direct
```

---

## Tests

```bash
# Tests unitaires
pytest tests/unit -v

# Tests intégration
pytest tests/integration -v

# Test spécifique
pytest tests/unit/test_risk_scoring.py -v
```

**Couverture importante :**
- `test_risk_scoring.py` - Sémantique risk score
- `test_dual_window_metrics.py` - Dual-window system
- `test_user_isolation.py` - Multi-tenant

---

## Structure Projet

```
api/           # FastAPI routers et endpoints
services/      # Logique métier (balance, risk, ml, etc.)
static/        # Frontend (HTML, JS, CSS)
config/        # Configuration (users.json, etc.)
data/users/    # Données utilisateurs (isolation)
tests/         # Tests (unit, integration)
docs/          # Documentation
```

---

## Debug & Outils

- API docs: `http://localhost:8080/docs`
- Health check: `GET /healthz`
- Debug paths: `GET /debug/paths`
- Pages test: `static/test_*.html`

---

## Règles Cap (Allocation)

- Utiliser exclusivement `selectCapPercent(state)` pour tout calcul/affichage de cap
- `selectPolicyCapPercent` (principal) et `selectEngineCapPercent` (secondaire)
- Normaliser les valeurs (0–1 → %)

---

## Ressources

- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Guide tests complet
- [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture système
- [AUTHENTICATION.md](AUTHENTICATION.md) - JWT et RBAC


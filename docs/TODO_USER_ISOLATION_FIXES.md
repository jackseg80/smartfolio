# TODO - User Isolation Fixes
> **Date:** 2025-12-29
> **Status:** Partial Fix Applied (analytics_endpoints.py ✅)

## ✅ Fixes Complétés

### 1. localStorage Settings Isolation ✅
**Fichiers:** `static/global-config.js`, `static/core/auth-guard.js`, `static/modules/settings-main-controller.js`

**Fix appliqué:**
- Clé localStorage isolée par user: `smartfolio_settings_${activeUser}`
- Race condition corrigée dans `saveSettings()`
- Settings préservées lors du logout

**Impact:** Clés API ne se perdent plus lors changement utilisateur

---

### 2. analytics_endpoints.py Cache Isolation ✅
**Fichier:** `api/analytics_endpoints.py`

**Fix appliqué:**
- Ajout `user: str = Depends(get_active_user)` aux endpoints:
  - `/performance/summary`
  - `/performance/detailed`
- Cache keys modifiées: `perf_summary:{user}:{days_back}`

**Impact:** Cache analytics isolé par utilisateur

---

## 🔴 Problèmes CRITIQUES (À Fixer Priorité 1)

### 1. HistoryManager Storage Global
**Fichier:** `services/analytics/history_manager.py:209`

**Problème:**
```python
def __init__(self, storage_path: str = "data/rebalance_history.json"):
    # ❌ Storage partagé entre tous les utilisateurs
    self.storage_path = Path(storage_path)
```

**Impact:**
- Sessions de rebalancing partagées entre utilisateurs
- User "demo" voit les sessions de "jack"
- Risque de modification croisée

**Fix proposé:**
```python
def __init__(self, user_id: str = "demo", storage_path: str = None):
    if storage_path is None:
        storage_path = f"data/users/{user_id}/rebalance_history.json"
    self.storage_path = Path(storage_path)
```

**Effort:** 2-3h (refactor HistoryManager + tests)

---

### 2. advanced_analytics_endpoints.py Cache
**Fichier:** `api/advanced_analytics_endpoints.py:23`

**Problème:**
```python
_advanced_cache = {}
# Cache keys probablement SANS user_id
```

**Impact:** Cache analytics avancés partagé entre utilisateurs

**Fix:** Même pattern que analytics_endpoints.py:
1. Ajouter `user: str = Depends(get_active_user)` aux endpoints
2. Modifier cache keys: `{metric}:{user}:{params}`

**Effort:** 30 min (similaire à analytics_endpoints.py)

---

### 3. unified_ml_endpoints.py Cache Mixte
**Fichier:** `api/unified_ml_endpoints.py:22`

**Problème:**
```python
_unified_ml_cache = {}
# Cache mixte: pipeline_status (global OK) + predictions (user-specific?)
```

**À vérifier:**
- Quels endpoints cachent des données user-specific?
- Pipeline status = global (OK de partager)
- Predictions ML = probablement user-specific (portfolio-based)

**Fix:** Audit endpoint par endpoint:
1. Identifier endpoints user-specific vs global
2. Ajouter `user` uniquement aux endpoints user-specific
3. Documenter quels endpoints sont globaux

**Effort:** 1-2h (audit + selective fixes)

---

## 🟡 Problèmes ÉLEVÉS (Priorité 2)

### 4. localStorage API Keys Frontend
**Fichiers:**
- `static/components/ai-chat.js:11,195`
- `static/components/ai-chat-context-builders.js:575-582`

**Problème:**
```javascript
// ❌ Clés API partagées entre utilisateurs
localStorage.setItem('groq_api_key', key);
localStorage.getItem('aiProvider');
```

**Impact:**
- User "demo" configure Groq key → visible par "jack"
- Préférences AI (provider, includeDocs) partagées

**Fix:**
```javascript
const activeUser = localStorage.getItem('activeUser') || 'demo';

// Isolation par user
localStorage.setItem(`${activeUser}_groq_api_key`, key);
localStorage.setItem(`${activeUser}_aiProvider`, provider);

// Ou utiliser window.globalConfig (déjà isolé par user)
window.globalConfig.set('groq_api_key', key);
```

**Effort:** 1h (migration vers globalConfig)

---

### 5. get_active_user() Fallback Silencieux
**Fichier:** `api/deps.py:109-164`

**Problème:**
```python
def get_active_user(x_user: Optional[str] = Header(None)) -> str:
    if not x_user:
        return "demo"  # ❌ Fallback silencieux sans erreur
```

**Impact:**
- Si client oublie header `X-User` → fallback "demo" SANS erreur
- Données renvoyées: portfolio de "demo" au lieu de "jack"

**Fix:**
```python
# Option A: Rendre X-User obligatoire (breaking change)
def get_required_user(x_user: str = Header(...)) -> str:
    return x_user

# Option B: Log warning + metric (non-breaking)
def get_active_user(x_user: Optional[str] = Header(None)) -> str:
    if not x_user:
        logger.warning("Missing X-User header, using default 'demo'")
        # Emit metric for monitoring
        return get_default_user()
    return x_user
```

**Recommandation:** Migrer endpoints sensibles (risk, portfolio, wealth) vers `get_required_user()`

**Effort:** 2h (migration progressive + tests)

---

## 🟢 Problèmes MOYENS (Priorité 3)

### 6. CoinGecko Proxy Cache
**Fichier:** `api/coingecko_proxy_router.py:51`

**Problème:**
```python
cache_key = f"coingecko_simple_price_{','.join(sorted(coin_ids))}"
# Pas de user_id, mais données publiques
```

**Impact:** Faible - Prix crypto identiques pour tous les utilisateurs

**Fix:** Ajouter `user` SEULEMENT si cache impacte user experience
- Si deux users demandent prix différents moments → peuvent recevoir prix stale
- Solution: Soit accepter (prix publics), soit isoler par user

**Effort:** 15 min (optionnel)

---

### 7. signals_endpoints.py Cache
**Fichier:** `api/execution/signals_endpoints.py:39`

**Problème:**
```python
_RECOMPUTE_CACHE = {}
# Cache signals de rebalancing
```

**À vérifier:** Signals basés sur portfolio user-specific ou globaux?

**Effort:** 30 min (audit + fix si nécessaire)

---

## 📋 Plan d'Action Recommandé

### Phase 1 - Fixes Critiques (4-6h)
1. ✅ localStorage settings isolation (COMPLÉTÉ)
2. ✅ analytics_endpoints.py cache (COMPLÉTÉ)
3. ⏳ Refactor HistoryManager pour user isolation (2-3h)
4. ⏳ Fix advanced_analytics_endpoints.py (30 min)
5. ⏳ Audit + fix unified_ml_endpoints.py (1-2h)

### Phase 2 - Fixes Élevés (3-4h)
6. ⏳ Isoler localStorage API keys frontend (1h)
7. ⏳ Migrer endpoints sensibles vers get_required_user() (2h)
8. ⏳ Vider UserSecretsManager cache lors logout (30 min)

### Phase 3 - Fixes Moyens (1-2h)
9. ⏳ CoinGecko cache (optionnel, 15 min)
10. ⏳ signals_endpoints.py audit (30 min)

---

## 🧪 Tests de Validation

Après chaque fix, tester:

```bash
# Test 1: Isolation localStorage
# 1. Login "demo" → Ajouter clé Groq
# 2. Logout → Login "jack" → Vérifier clé vide
# 3. Logout → Login "demo" → Vérifier clé présente

# Test 2: Isolation cache backend
# Terminal 1
curl -H "X-User: demo" "localhost:8080/analytics/performance/summary?days_back=30"

# Terminal 2
curl -H "X-User: jack" "localhost:8080/analytics/performance/summary?days_back=30"

# Vérifier logs: cache keys différents (demo vs jack)

# Test 3: Isolation storage HistoryManager
# 1. User "demo" crée session rebalancing
# 2. User "jack" GET /analytics/sessions
# 3. Vérifier: jack NE VOIT PAS sessions de demo
```

---

## 📊 Métriques de Succès

| Métrique | Avant | Cible |
|----------|-------|-------|
| Caches isolés par user | 1/6 (17%) | 6/6 (100%) |
| localStorage isolé | 1/5 clés (20%) | 5/5 (100%) |
| Storage backend isolé | 0% | 100% |
| Endpoints avec X-User requis | 0% | 80%+ |

---

## 🔗 Références

- [CLAUDE.md](../CLAUDE.md) - Multi-Tenant OBLIGATOIRE
- [docs/AUTHENTICATION.md](AUTHENTICATION.md) - JWT Auth System
- [api/deps.py](../api/deps.py) - get_active_user() vs get_required_user()
- [services/user_secrets.py](../services/user_secrets.py) - UserSecretsManager cache

---

*Ce document sera mis à jour au fur et à mesure des fixes appliqués.*

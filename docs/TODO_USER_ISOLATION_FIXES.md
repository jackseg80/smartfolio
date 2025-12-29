# TODO - User Isolation Fixes
> **Date:** 2025-12-29
> **Status:** Phase 1 COMPLÉTÉE ✅ (4 caches isolés + localStorage + HistoryManager storage)

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

### 3. advanced_analytics_endpoints.py User Isolation ✅

**Fichier:** `api/advanced_analytics_endpoints.py`

**Fix appliqué:**

- Ajout `user: str = Depends(get_active_user)` aux 5 endpoints:
  - `/analytics/advanced/metrics`
  - `/analytics/advanced/timeseries`
  - `/analytics/advanced/drawdown-analysis`
  - `/analytics/advanced/strategy-comparison`
  - `/analytics/advanced/risk-metrics`
- Appels `get_current_balances()` isolés: `user_id=user`
- Helper `_generate_real_performance_data()` accepte `user_id` param
- Documentation cache: Note pour inclure `user_id` si réintroduit

**Impact:** Analytics avancés isolés par utilisateur

**Test validation:**

```powershell
# Demo: total_return_pct = -5.51%, portfolio_values = 10071.33
curl -H "X-User: demo" "http://localhost:8080/analytics/advanced/metrics?days=30"

# Jack: total_return_pct = -5.26%, portfolio_values = 9762.26
curl -H "X-User: jack" "http://localhost:8080/analytics/advanced/metrics?days=30"
```

---

### 4. unified_ml_endpoints.py User Isolation ✅

**Fichier:** `api/unified_ml_endpoints.py`

**Fix appliqué:**

- Ajout `user: str = Depends(get_active_user)` à 1 endpoint:
  - `/api/ml/correlation/matrix/current`
- Appel `get_unified_filtered_balances()` isolé: `user_id=user`
- Documentation cache: Note cache global SAUF correlation (portfolio user-specific)

**Impact:** Matrice de corrélations isolée par utilisateur (basée sur portfolio)

**Test validation:**

```powershell
# Demo: 8 assets (AAVE, LINK, USDC, USDT, BNB, BTC, ETH, SOL)
curl -H "X-User: demo" "http://localhost:8080/api/ml/correlation/matrix/current?window_days=30"

# Jack: ~180 assets (NEWT, HBAR, LINK, WLKN, ... portfolio massif)
curl -H "X-User: jack" "http://localhost:8080/api/ml/correlation/matrix/current?window_days=30"
```

**Note:** Tous les autres endpoints ML (prédictions, sentiment, modèles) sont globaux (données publiques) → Cache partagé OK ✅

---

### 5. HistoryManager Storage Isolation ✅

**Fichiers:** `services/analytics/history_manager.py`, `api/analytics_endpoints.py`

**Fix appliqué:**

- **HistoryManager refactoré:**
  - `__init__` accepte `user_id` (ligne 209)
  - Path isolé: `data/users/{user_id}/rebalance_history.json` (ligne 212)
  - Factory function `get_history_manager(user_id)` créée (ligne 521)
  - Auto-save ajouté dans 4 méthodes (create, snapshot, actions, results)
- **11 endpoints analytics modifiés** avec `user: str = Depends(get_active_user)`:
  - POST `/api/analytics/sessions` (create)
  - GET `/api/analytics/sessions/{id}` (get one)
  - POST `/api/analytics/sessions/{id}/portfolio-snapshot`
  - POST `/api/analytics/sessions/{id}/actions`
  - POST `/api/analytics/sessions/{id}/execution-results`
  - POST `/api/analytics/sessions/{id}/complete`
  - GET `/api/analytics/sessions` (list)
  - GET `/api/analytics/performance/summary`
  - GET `/api/analytics/performance/detailed`
  - GET `/api/analytics/reports/comprehensive`
  - GET `/api/analytics/optimization/recommendations`

**Impact:** Sessions de rebalancing isolées par utilisateur (storage backend + cache)

**Test validation:**

```powershell
# Créer sessions pour demo et jack
curl -X POST "http://localhost:8080/api/analytics/sessions" -H "X-User: demo" -H "Content-Type: application/json" -d '{"target_allocations":{"BTC":40},"source":"test"}'
curl -X POST "http://localhost:8080/api/analytics/sessions" -H "X-User: jack" -H "Content-Type: application/json" -d '{"target_allocations":{"BTC":50},"source":"test"}'

# Vérifier isolation
ls data/users/demo/rebalance_history.json  # ✅ Existe
ls data/users/jack/rebalance_history.json  # ✅ Existe
```

---

## 🔴 Problèmes CRITIQUES (À Fixer Priorité 1)

**Aucun problème critique restant !** 🎉

---

### 6. localStorage API Keys Frontend ✅

**Fichiers:** `static/global-config.js`, `static/components/ai-chat.js`, `static/components/ai-chat-context-builders.js`

**Fix appliqué:**

- **global-config.js:** Ajout de 6 nouvelles clés dans `DEFAULT_SETTINGS`:
  - `groq_api_key`, `claude_api_key`, `grok_api_key`, `openai_api_key`
  - `aiProvider` (default: 'groq')
  - `aiIncludeDocs` (default: true)
- **ai-chat.js:** Migration vers `window.globalConfig`:
  - Constructor: `window.globalConfig.get('aiProvider')`
  - `switchProvider()`: `window.globalConfig.set('aiProvider', newProvider)`
- **ai-chat-context-builders.js:** Migration `buildSettingsContext()`:
  - API keys check: `window.globalConfig.get('groq_api_key')`
  - AI preferences: `window.globalConfig.get('aiProvider')`

**Impact:** Clés API AI isolées par utilisateur (storage automatique dans `smartfolio_settings_${activeUser}`)

**Test validation:**

```javascript
// Console Browser (Page Settings ou Dashboard)

// Test 1: User "demo" - Configurer Groq key
window.globalConfig.set('groq_api_key', 'gsk_demo_test_key_123');
window.globalConfig.set('aiProvider', 'groq');
console.log('Demo Groq:', window.globalConfig.get('groq_api_key')); // "gsk_demo_test_key_123"

// Test 2: Switch to user "jack" (via UI logout/login)
// Puis dans console:
console.log('Jack Groq:', window.globalConfig.get('groq_api_key')); // "" (empty)

// Test 3: User "jack" - Configurer Claude key
window.globalConfig.set('claude_api_key', 'sk-ant-jack_key_456');
window.globalConfig.set('aiProvider', 'claude');

// Test 4: Switch back to "demo"
console.log('Demo Groq:', window.globalConfig.get('groq_api_key')); // "gsk_demo_test_key_123" ✅
console.log('Demo Claude:', window.globalConfig.get('claude_api_key')); // "" ✅

// Vérifier localStorage raw keys
Object.keys(localStorage).filter(k => k.startsWith('smartfolio_settings_'));
// ["smartfolio_settings_demo", "smartfolio_settings_jack"]
```

**Note:** Pas besoin de migration des anciennes clés (localStorage direct → globalConfig), les users reconfigureront naturellement leurs clés dans Settings.

---

## 🟡 Problèmes ÉLEVÉS (Priorité 2)

---

### 7. get_active_user() → get_required_user() Migration ✅

**Fichiers:** `api/risk_endpoints.py`, `api/portfolio_endpoints.py`, `api/wealth_endpoints.py`

**Fix appliqué:**

- **27 endpoints sensibles migrés** vers `get_required_user()`:
  - `api/risk_endpoints.py`: 3 endpoints (dashboard, alerts, monitoring)
  - `api/portfolio_endpoints.py`: 4 endpoints (metrics, snapshot, trend, alerts)
  - `api/wealth_endpoints.py`: 20 endpoints (patrimoine, banks, transactions)

**Impact:** Header `X-User` maintenant OBLIGATOIRE pour endpoints sensibles

**Avant (fallback silencieux):**

```python
# Client oublie header X-User → fallback "demo" SANS erreur
curl "http://localhost:8080/api/risk/dashboard"
# ✅ 200 OK (données "demo") ❌ PAS DÉTECTÉ
```

**Après (erreur explicite):**

```python
# Client oublie header X-User → erreur 422
curl "http://localhost:8080/api/risk/dashboard"
# ❌ 422 {"detail":[{"loc":["header","X-User"],"msg":"field required"}]}

# Avec header X-User → OK
curl -H "X-User: demo" "http://localhost:8080/api/risk/dashboard"
# ✅ 200 OK (données "demo")
```

**Test validation:**

```powershell
# Test 1: Risk endpoint sans header → Erreur 422
curl "http://localhost:8080/api/risk/dashboard"

# Test 2: Portfolio endpoint sans header → Erreur 422
curl "http://localhost:8080/api/portfolio/metrics"

# Test 3: Wealth endpoint sans header → Erreur 422
curl "http://localhost:8080/api/wealth/patrimoine/summary"

# Test 4: Avec header X-User → OK
curl -H "X-User: demo" "http://localhost:8080/api/risk/dashboard"
```

**Note:** `get_active_user()` conservé pour endpoints non-critiques (compatibilité backward)

---

## 🟢 Problèmes MOYENS (Priorité 3)

### 4. CoinGecko Proxy Cache
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

### 5. signals_endpoints.py Cache
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
3. ✅ advanced_analytics_endpoints.py (COMPLÉTÉ - 30 min)
4. ✅ unified_ml_endpoints.py (COMPLÉTÉ - 1h audit + 1 endpoint)
5. ⏳ Refactor HistoryManager pour user isolation (2-3h)

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

| Métrique                      | Avant      | Actuel              | Cible         |
|-------------------------------|------------|---------------------|---------------|
| Caches isolés par user        | 1/6 (17%)  | 3/6 (50%) ✅        | 6/6 (100%)    |
| localStorage isolé (settings) | 0/11 (0%)  | 11/11 (100%) ✅     | 11/11 (100%)  |
| localStorage isolé (AI keys)  | 0/6 (0%)   | 6/6 (100%) ✅       | 6/6 (100%)    |
| Storage backend isolé         | 0/1 (0%)   | 1/1 (100%) ✅       | 1/1 (100%)    |
| Endpoints avec X-User requis  | 0%         | 11/~50 (22%) ⚠️     | 80%+          |

---

## 🔗 Références

- [CLAUDE.md](../CLAUDE.md) - Multi-Tenant OBLIGATOIRE
- [docs/AUTHENTICATION.md](AUTHENTICATION.md) - JWT Auth System
- [api/deps.py](../api/deps.py) - get_active_user() vs get_required_user()
- [services/user_secrets.py](../services/user_secrets.py) - UserSecretsManager cache

---

*Ce document sera mis à jour au fur et à mesure des fixes appliqués.*

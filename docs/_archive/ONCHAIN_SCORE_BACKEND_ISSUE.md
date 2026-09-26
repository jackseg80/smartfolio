# OnChain Score 36↔42 Variability - Backend Root Cause

**Date**: 2025-10-02
**Problem**: OnChain Score varie entre 36 et 42 selon les refreshes
**Root Cause**: Backend `/api/crypto-toolbox` timeout → cache SWR retourne données stales
**Status**: [Warning] **BACKEND ISSUE** - nécessite fix côté serveur Flask (port 8801)

---

## Diagnostic Complet

### Symptômes Observés

**Frontend (risk-dashboard.html)**:
- Hard refresh (Ctrl+Shift+R): **OnChain = 36** [Error]
- Soft refresh (F5): **OnChain = 42** [OK]
- Après quelques minutes: **OnChain = 36** [Error]

**Pattern**: Le score **dégrade progressivement** de 42 → 36 au fil du temps

### Investigation Frontend

**1. Orchestrator appelle fetchCryptoToolboxIndicators()**
```javascript
// risk-data-orchestrator.js:145
fetchAllIndicators({ force: forceRefresh })
  ↓
// onchain-indicators.js:1381
fetchCryptoToolboxIndicators({ force })
  ↓
// onchain-indicators.js:841
fetch(`${apiBase}/api/crypto-toolbox`)
```

**2. Cache SWR (Stale-While-Revalidate)**
```javascript
// onchain-indicators.js:771-788
if (!force && cached && age < TTL_SHOW_MS) {
  return cached;  // Retourne cache même si stale!
}
```

**3. Timeouts & Circuit Breaker**
```javascript
// onchain-indicators.js:809-827
if (_circuitBreakerState.isOpen) {
  return cached; // Retourne cache ancien si backend fail
}
```

### Investigation Backend

**Test 1: Endpoint `/api/crypto-toolbox/indicators`**
```bash
$ curl http://localhost:8080/api/crypto-toolbox/indicators
{"detail":"Not Found"}  # [Error] Endpoint n'existe pas
```

**Test 2: Endpoint `/api/crypto-toolbox`** (le bon)
```bash
$ curl -m 5 http://localhost:8080/api/crypto-toolbox
# TIMEOUT après 5 secondes!
```

**Test 3: API Risk Dashboard**
```bash
$ curl http://localhost:8080/api/risk/dashboard
{
  "onchain_indicators": {},  # [Error] Vide!
  "risk_metrics": {
    "risk_score": 37.0
  }
}
```

### Root Cause Identifiée

**Le backend scraper Flask (port 8801) ne répond PAS:**

1. **Proxy FastAPI → Flask**
   ```
   Frontend → http://localhost:8080/api/crypto-toolbox
            ↓ (proxy)
            → http://localhost:8801/api/crypto-toolbox
            [Error] TIMEOUT (>5s)
   ```

2. **Cache SWR Fallback**
   ```
   Timeout → Circuit breaker OPEN
          → Retourne cache ancien (36)
          OU cache semi-récent (42)
   ```

3. **TTL SWR explique la variabilité**
   ```javascript
   TTL_SHOW_MS  = 5 min   // Serve from cache
   TTL_BG_MS    = 3 min   // Background revalidate
   TTL_HARD_MS  = 30 min  // Force network
   ```

   **Scénario A** (cache récent < 5min):
   - Retourne cache **42** (données fraîches)

   **Scénario B** (cache vieux 30min+):
   - Force network → **Timeout** → Circuit breaker
   - Retourne cache **très ancien** (36)

---

## Solutions

### Solution 1: Fix Backend Scraper (RECOMMANDÉ )

**Problème**: Flask scraper (port 8801) ne répond pas

**Action**: Vérifier pourquoi `/api/crypto-toolbox` timeout

**Checklist**:
```bash
# 1. Vérifier que Flask scraper tourne
ps aux | grep python | grep 8801
# OU
curl http://localhost:8801/health

# 2. Vérifier logs Flask pour erreurs
tail -f logs/flask-scraper.log

# 3. Tester endpoint direct
curl -v http://localhost:8801/api/crypto-toolbox

# 4. Vérifier dépendances scraper
pip list | grep -E "requests|beautifulsoup|selenium"
```

**Causes possibles**:
- [Error] Scraper pas démarré (process mort)
- [Error] Rate limiting externe (APIs tierces bloquées)
- [Error] Timeout scraping (sites web lents)
- [Error] Erreur Python non catchée (crash silencieux)

### Solution 2: Augmenter Timeout Frontend (WORKAROUND )

**Si backend est intrinsèquement lent (>5s pour scraper):**

```javascript
// onchain-indicators.js:844
response = await performanceMonitoredFetch(proxyUrl, {
  timeout: 30000  // 30s au lieu de 5s
});
```

**Inconvénient**: UX dégradée (attente 30s!)

### Solution 3: Fallback Gracieux (PALLIATIF )

**Accepter que backend est instable, montrer état clairement:**

```javascript
// onchain-indicators.js après timeout
if (cached) {
  console.warn(' Backend timeout, using stale cache (age: Xmin)');
  cached._stale = true;
  cached._backend_available = false;
  return cached;
}
```

**Frontend affiche warning:**
```javascript
// analytics-unified.html
if (onchainData._stale) {
  showWarning('OnChain data is stale (backend unavailable)');
}
```

### Solution 4: Mock Data (DEV ONLY )

**Pour tests frontend sans backend:**

```javascript
// onchain-indicators.js
if (import.meta.env?.DEV || window.location.hostname === 'localhost') {
  return getMockIndicators(); // 30 indicateurs simulés
}
```

---

## Impact Actuel

**Scores Calculés**:
- **OnChain = 36**: Cache **très ancien** (30min+, backend timeout)
- **OnChain = 42**: Cache **récent** (<5min, backend a répondu une fois)
- **Risk = 37**: Calculé avec OnChain=36 (formule dépend des métriques)
- **Risk = 50**: Calculé avec OnChain=42 (meilleures métriques)

**Cascade d'Erreurs**:
```
Backend timeout
  ↓
Cache stale (OnChain=36)
  ↓
calculateCompositeScore() utilise vieilles données
  ↓
Risk Score incorrect (37 au lieu de 50)
  ↓
Blended Score incorrect (68 au lieu de 67)
  ↓
Recommandations incohérentes
```

---

## Tests de Validation

### Test 1: Vérifier Backend Disponibilité

```bash
# Terminal 1: Démarrer Flask scraper (si pas déjà running)
cd scraper/
python app.py  # Port 8801

# Terminal 2: Tester endpoint
curl -w "\nTime: %{time_total}s\n" http://localhost:8801/api/crypto-toolbox
```

**Attendu**: Réponse JSON avec 30 indicateurs en <3s

### Test 2: Vérifier Cache SWR

```javascript
// Console browser (risk-dashboard.html)
localStorage.clear(); // Vider tout cache
location.reload();    // Hard refresh

// Observer logs:
// [OK] " SWR: Forcing network (cache cleared)"
// [OK] "Indicators count: 30"
// [OK] "OnChain Score: 42"
```

### Test 3: Simuler Timeout Backend

```javascript
// onchain-indicators.js (temporaire pour test)
const proxyUrl = 'http://localhost:9999/fake'; // Endpoint inexistant
// → Doit retourner cache + warning
```

**Attendu**: Circuit breaker OPEN + cache stale utilisé

---

## Recommandations Finales

### Court Terme (Aujourd'hui)

1. **Redémarrer scraper Flask** (port 8801)
2. **Vérifier logs** pour erreurs Python
3. **Tester endpoint** manuellement

### Moyen Terme (Cette Semaine)

1. **Monitoring backend**:
   - Healthcheck `/health` toutes les 30s
   - Alertes si timeout >3s

2. **Cache plus intelligent**:
   - Marqueur `_stale` visible dans UI
   - Badge "Données anciennes (Xmin)" si backend down

3. **Fallback robuste**:
   - Mock data si dev mode
   - Dernières bonnes données connues en prod

### Long Terme (Refactoring)

1. **Découpler scraping de l'API**:
   - Scraper background (cron job)
   - API lit depuis cache Redis/DB
   - Jamais de timeout API

2. **Observabilité**:
   - Metrics Prometheus (scraper duration, success rate)
   - Dashboards Grafana
   - Alertes PagerDuty si backend down >5min

---

## Liens Utiles

**Code Frontend**:
- [risk-data-orchestrator.js:145](../static/core/risk-data-orchestrator.js#L145) - Appel fetchAllIndicators
- [onchain-indicators.js:762](../static/modules/onchain-indicators.js#L762) - fetchCryptoToolboxIndicators
- [onchain-indicators.js:841](../static/modules/onchain-indicators.js#L841) - Endpoint `/api/crypto-toolbox`

**Code Backend** (à vérifier):
- Flask scraper app: `scraper/app.py` (??)
- Proxy FastAPI: `api/main.py` (route `/api/crypto-toolbox`)

**Autres Docs**:
- [SCORE_UNIFICATION_FIX.md](./SCORE_UNIFICATION_FIX.md) - Fix orchestrator SSOT
- [SCORE_CACHE_HARD_REFRESH_FIX.md](./SCORE_CACHE_HARD_REFRESH_FIX.md) - Fix cache hard refresh

---

**Auteur**: Claude
**Status**: [Warning] BLOQUÉ - Nécessite fix backend Flask scraper
**Priority**: HIGH (impact: scores incorrects, recommandations fausses)


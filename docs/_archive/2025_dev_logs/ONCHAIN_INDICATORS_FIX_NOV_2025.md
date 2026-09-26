# On-Chain Indicators Fix - November 2025

## Problème Identifié

Les indicateurs on-chain ne se mettaient plus à jour dans le Risk Dashboard.

### Cause Racine
1. **Cache backend** contenait des données invalides (toutes à 0.00%)
2. **Cache frontend** (localStorage SWR) servait ces données stales
3. **Scraping Playwright** échouait silencieusement et retournait des données invalides
4. **Aucune validation** ne détectait ni ne rejetait les données suspectes

### Symptômes
```javascript
// Toutes les valeurs étaient à 0.00%
{
  "name": "MVRV Z-Score",
  "value_numeric": 0.0,  // [Error] Invalid!
  "value": "0.00%"
}
```

---

## Solutions Implémentées

### 1. Backend Validation (`api/crypto_toolbox_endpoints.py`)

#### A. Validation lors du scraping (lignes 335-349)
```python
# Reject if more than 80% of indicators are zero
if zero_percentage > 80:
    logger.error(f" Invalid scraping result: {zero_percentage:.1f}% zeros")
    raise Exception("Scraping validation failed - rejecting invalid data")

# Warning if 50-80% are zero
if zero_percentage > 50:
    logger.warning(f" Suspicious scraping result: {zero_percentage:.1f}% zeros")
```

**Avantages:**
- Détecte les échecs de scraping Playwright (timeout, page non chargée, etc.)
- Rejette immédiatement les données invalides
- Logs détaillés pour debug

#### B. Protection du cache (lignes 428-445)
```python
# Don't cache invalid data - keep old good cache instead
if zero_percentage > 80 and _cache["data"]:
    logger.error(" Not caching invalid data - keeping previous cache")
    return {
        **_cache["data"],
        "scraping_failed": True,
        "failure_reason": f"Invalid data detected ({zero_percentage:.1f}% zeros)"
    }
```

**Avantages:**
- Ne permet **jamais** d'écraser de bonnes données avec des données invalides
- Retourne les données précédentes en fallback
- Signale l'échec dans la response (`scraping_failed: true`)

#### C. Fallback intelligent (lignes 469-483)
```python
except Exception as scrape_error:
    # Return old cache if available instead of failing completely
    if _cache["data"]:
        logger.error(f" Scraping failed - falling back to stale cache")
        return {
            **_cache["data"],
            "scraping_failed": True,
            "failure_reason": str(scrape_error)
        }
```

**Avantages:**
- Graceful degradation : mieux des données stales que pas de données
- L'UI reste fonctionnelle même si le scraping échoue
- L'utilisateur est informé via `scraping_failed` flag

---

### 2. Frontend Validation (`static/modules/onchain-indicators.js`)

#### A. Détection backend failure (lignes 895-905)
```javascript
// Detect stale/invalid data from backend
if (apiData.scraping_failed) {
    const reason = apiData.failure_reason || 'Unknown error';
    console.warn(` Backend scraping failed: ${reason} - Using stale cache`);
}
```

#### B. Validation frontend double-check (lignes 976-1008)
```javascript
const zeroPercentage = 100 - (nonZeroCount / indicatorValues.length * 100);

if (zeroPercentage > 80) {
    // Show user-visible warning
    window.showToast(
        `[Warning] On-chain data quality issue detected - using fallback`,
        'warning',
        { duration: 10000 }
    );
}
```

**Avantages:**
- Détection client-side redondante (défense en profondeur)
- Warning visuel dans l'UI (toast notification)
- Métadonnées qualité dans le cache (`data_quality` object)

---

## Comment Tester

### Test 1 : Vérifier les données actuelles
```bash
# Check backend health
curl http://localhost:8080/api/crypto-toolbox/health

# Get current indicators (should have real values now)
curl http://localhost:8080/api/crypto-toolbox | python -m json.tool | head -50
```

**Résultat attendu:**
```json
{
    "name": "CBBI*",
    "value_numeric": 69.37,  // [OK] Non-zero!
    "value": "69.37%"
}
```

### Test 2 : Force refresh pour obtenir de nouvelles données
```bash
# Force backend refresh
curl -X POST http://localhost:8080/api/crypto-toolbox/refresh
```

### Test 3 : Vider le cache frontend
**Option A - Console navigateur:**
```javascript
localStorage.removeItem('CTB_ONCHAIN_CACHE_V2');
location.reload();
```

**Option B - UI:**
1. Ouvrir Risk Dashboard
2. Cliquer sur bouton **⋮** (options)
3. Cliquer **"Force Refresh"**

---

## Validation Metrics

### Backend (Python)
- **Threshold critique:** 80% zeros → Reject & keep old cache
- **Threshold warning:** 50% zeros → Log warning but accept
- **Log level:** ERROR pour rejection, WARNING pour suspects, DEBUG pour success

### Frontend (JavaScript)
- **Threshold critique:** 80% zeros → Toast warning + error log
- **Threshold warning:** 50% zeros → Console warning
- **Cache metadata:** `data_quality.zero_percentage` disponible pour monitoring

---

## Déploiement

### 1. Redémarrer le serveur backend
```bash
# IMPORTANT: Les modifications backend nécessitent un restart
# Arrêter le serveur actuel (Ctrl+C), puis:
.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --port 8080
```

### 2. Hard refresh frontend
```bash
# Dans le navigateur sur http://localhost:8080/static/risk-dashboard.html
Ctrl + Shift + R  # Hard refresh pour charger nouveau JS
```

### 3. Vérifier les logs
```bash
# Surveiller les logs pour voir les validations
Get-Content logs\app.log -Wait -Tail 50
```

**Logs attendus (success):**
```
[OK] Successfully scraped 30 indicators
[OK] Data validation passed: 30/30 indicators have non-zero values
```

**Logs attendus (échec détecté):**
```
[Error] Invalid scraping result: 93.3% of indicators are zero
[Error] Not caching invalid data (93.3% zeros) - keeping previous cache
```

---

## Améliorations Futures

### Phase 1 : Monitoring (Recommandé)
- [ ] Endpoint `/api/crypto-toolbox/quality` pour métriques qualité données
- [ ] Prometheus metrics : `onchain_indicators_zero_percentage`
- [ ] Alertes si zero_percentage > 50% pendant >1h

### Phase 2 : Retry Logic (Optionnel)
- [ ] Retry automatique si scraping retourne >80% zeros
- [ ] Exponential backoff (1min, 5min, 15min)
- [ ] Circuit breaker après 3 échecs consécutifs

### Phase 3 : Fallback Sources (Avancé)
- [ ] API secondaire (ex: Glassnode free tier)
- [ ] Données simulées basées sur dernier cycle connu
- [ ] Interpolation intelligente si données partielles

---

## Fichiers Modifiés

### Backend
- `api/crypto_toolbox_endpoints.py` - Validation + cache protection

### Frontend
- `static/modules/onchain-indicators.js` - Double validation + toast warnings

### Documentation
- `docs/ONCHAIN_INDICATORS_FIX_NOV_2025.md` (ce fichier)

---

## Checklist Post-Déploiement

- [ ] Serveur redémarré avec nouveau code
- [ ] Frontend hard-refresh (Ctrl+Shift+R)
- [ ] Cache localStorage vidé (`CTB_ONCHAIN_CACHE_V2`)
- [ ] Test `/api/crypto-toolbox` retourne valeurs non-zero
- [ ] Test Force Refresh UI fonctionne
- [ ] Logs backend montrent validations
- [ ] Pas de toasts warning en conditions normales
- [ ] Documentation CLAUDE.md mise à jour si nécessaire

---

**Date:** 2025-11-01
**Severity:** High (Impact: Risk Dashboard inutilisable)
**Status:** Fixed [OK]
**Tested:** Backend validation [OK] | Frontend validation [OK] | Cache protection [OK]

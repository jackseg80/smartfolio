# Performance Monitoring - Guide Complet

## Vue d'Ensemble

Le système de monitoring unifié permet de surveiller en temps réel les performances, caches et freshness des données du système Crypto Rebal.

**URL** : `http://localhost:8080/static/performance-monitor-unified.html`

---

## Sections Surveillées

### 1. Frontend Caches

#### OnChain Module Cache
- **Type** : IntelligentCache (classe JavaScript)
- **Données** : Indicateurs on-chain (Fear & Greed, MVRV, NUPL, etc.)
- **TTL** : 10min (fresh) → 30min (stale+bg) → 2h (hard)
- **Métriques** :
  - Cache Size (nombre d'entrées)
  - Hit Rate (%)
  - Total Requests

**Statuts** :
- [OK] **Good** : Hit rate > 70%, cache actif
- **Warning** : Hit rate 30-70%
- [Negative] **Critical** : Hit rate < 30%

---

### 2. Risk Dashboard Store

#### LocalStorage Persistence
- **Clés surveillées** :
  - `risk_score_blended` : Score de risque blended actuel
  - `risk_score_timestamp` : Timestamp du dernier calcul
  - `risk_scores_cache` : Cache persistant (TTL 12h)

**Métriques** :
- Blended Risk Score (valeur actuelle)
- Score Age (minutes depuis dernier update)
- Persistent Cache Age (heures)

**Statuts** :
- [OK] **Good** : Age < 15min (scores), < 12h (cache)
- **Warning** : Age 15-60min (scores)
- [Negative] **Critical** : Age > 60min (scores stale)

---

### 3. OnChain Indicators Cache

#### SWR (Stale-While-Revalidate) Cache
- **Clé** : `CTB_ONCHAIN_CACHE_V2`
- **Données** : 30+ indicateurs crypto-toolbox
- **Stratégie** : Cache first, background revalidation

**Métriques** :
- Indicators Count (nombre d'indicateurs)
- Cache Age (minutes)
- Cache Source (network/cache/stale_cache_fallback)

**Statuts** :
- [OK] **Good** : Age < 10min (fresh)
- **Warning** : Age 10-30min (stale mais acceptable)
- [Negative] **Critical** : Age > 30min (très stale)

---

### 4. LocalStorage Usage

#### Quota Management
- **Quota typique** : 5-10 MB selon navigateur
- **Surveillance** : Éviter dépassement quota

**Métriques** :
- Total Keys (nombre de clés)
- Storage Used (KB)
- Quota Usage (%)

**Statuts** :
- [OK] **Good** : Usage < 50%
- **Warning** : Usage 50-80%
- [Negative] **Critical** : Usage > 80% (risque dépassement)

---

### 5. ML Pipeline Cache

#### LRU Cache Backend (Python)
- **Endpoint** : `/api/ml/cache/stats`
- **Limite** : 5 modèles max, 2048 MB mémoire
- **Type** : LRU (Least Recently Used) eviction

**Métriques** :
- Models Loaded (nombre de modèles en mémoire)
- Memory Usage (MB)
- Cache Hit Rate (%)

**Statuts** :
- [OK] **Good** : Memory < 1500MB, Hit rate > 70%
- **Warning** : Memory 1500-2000MB, Hit rate 30-70%
- [Negative] **Critical** : Memory > 2000MB

---

### 6. API Endpoints Performance

#### Endpoints Critiques Surveillés
1. **Portfolio Metrics** : `/portfolio/metrics`
2. **Balances** : `/balances/current`
3. **ML Status** : `/api/ml/status`

**Métriques** :
- Response Time (ms)
- HTTP Status Code

**Statuts** :
- [OK] **Good** : Latence < 100ms
- **Warning** : Latence 100-500ms
- [Negative] **Critical** : Latence > 500ms ou erreur

---

### 7. Data Freshness

#### Sources de Données
- **Portfolio Snapshots** : Dernière sauvegarde P&L
- **Risk Scores** : Dernière mise à jour scores
- **OnChain Indicators** : Dernière récupération indicateurs

**Métriques** :
- Last Portfolio Snapshot (heures)
- Risk Scores (minutes)
- OnChain Indicators (minutes)

**Statuts** :
- [OK] **Good** : Portfolio < 24h, Risk < 15min, OnChain < 30min
- **Warning** : Portfolio 24-72h, Risk 15-60min, OnChain 30-120min
- [Negative] **Critical** : Portfolio > 72h, Risk > 60min, OnChain > 120min

---

## Contrôles Disponibles

### Refresh All
- Rafraîchit toutes les sections
- Non-bloquant (parallèle)
- Durée typique : 2-5 secondes

### Stress Test
- Lance 10 requêtes vers `/portfolio/metrics` avec délai de 50ms entre chaque
- Évite rate limiting (429) en espaçant les requêtes
- Mesure taux de succès et latence moyenne
- **Objectif** : 100% succès, < 100ms avg par requête

### Clear Caches
- Supprime tous les caches localStorage
- **Garde** : `user_id`, `data_source`, `theme`
- Nécessite confirmation utilisateur

### Auto-Refresh
- Intervalle : 10 secondes
- Toggle : ON/OFF
- Défaut : Activé au démarrage

### Export Report
- Génère JSON complet avec :
  - Tous les logs de session
  - Snapshot localStorage
  - Timestamp export
- Nom fichier : `performance-report-{timestamp}.json`

---

## Activity Log

### Format
```
[HH:MM:SS] Message de log
```

### Niveaux
- **Info** : Opérations normales
- **Warning** : Problèmes non-bloquants
- **Error** : Erreurs nécessitant attention

### Limite
- Max 100 entrées en mémoire
- Auto-scroll vers bas
- Inclus dans export report

---

## Intégration & Développement

### Ajouter une Nouvelle Métrique Frontend

```javascript
async function checkMyNewCache() {
    const container = document.getElementById('myNewSection');
    const metrics = [];

    try {
        // Votre logique de vérification
        const cacheData = localStorage.getItem('MY_CACHE_KEY');

        metrics.push({
            label: 'My Cache Status',
            value: cacheData ? 'Active' : 'Empty',
            status: cacheData ? 'good' : 'neutral'
        });

        addLog('My cache checked successfully');
    } catch (error) {
        metrics.push({
            label: 'Error',
            value: error.message,
            status: 'critical'
        });
        addLog(`My cache check failed: ${error.message}`, 'error');
    }

    renderMetrics(container, metrics);
}

// Ajouter au refreshAll()
async function refreshAll() {
    await Promise.all([
        // ... existing checks
        checkMyNewCache()
    ]);
}
```

### Ajouter un Nouvel Endpoint Backend

```python
@router.get("/cache/my-new-cache/stats")
async def get_my_cache_stats():
    """Get stats for my new cache"""
    try:
        stats = {
            "cache_size": 10,
            "hit_rate": 85.5,
            "memory_mb": 128
        }
        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

Puis ajouter la vérification dans `checkAPIPerformance()`.

---

## Objectifs de Performance

### Frontend
- **OnChain Cache Hit Rate** : > 70%
- **Risk Score Age** : < 15min
- **LocalStorage Usage** : < 50%

### Backend
- **API Response Time (p95)** : < 100ms
- **ML Pipeline Memory** : < 1500MB
- **Cache Hit Rate ML** : > 70%

### Data Freshness
- **Portfolio Snapshots** : < 24h
- **Risk Scores** : < 15min
- **OnChain Indicators** : < 30min

---

## Alertes & Actions

### LocalStorage > 80%
**Action** : Nettoyer vieux caches, augmenter TTL, externaliser données volumineuses

### ML Memory > 1800MB
**Action** : Appeler `/api/ml/memory/optimize`, réduire nombre modèles chargés

### API Latence > 500ms
**Action** : Vérifier logs backend, optimiser requêtes DB, activer caching

### Risk Scores > 60min
**Action** : Vérifier service `risk-dashboard.html` est actif, forcer refresh

---

## Logs & Debugging

### Console Browser
Le monitor utilise `console.log` pour debugging détaillé :
- Chaque métrique collectée
- Erreurs API avec stack trace
- Changements d'état cache

### Export Report
Utiliser pour :
- Post-mortem incidents
- Analyse tendances performance
- Reporting à l'équipe

### Activity Log
Surveiller en temps réel pour :
- Détecter erreurs intermittentes
- Valider opérations manuelles (clear cache, stress test)
- Monitoring continu

---

## Liens Utiles

- **Risk Dashboard** : `/static/risk-dashboard.html`
- **Analytics Unified** : `/static/analytics-unified.html`
- **ML Endpoints** : `/api/ml/cache/stats`, `/api/ml/status`
- **Portfolio Endpoints** : `/portfolio/metrics`, `/balances/current`

---

## Notes Importantes

1. **Auto-refresh** consomme des ressources - désactiver si monitoring passif
2. **Stress test** peut impacter backend sous charge - utiliser modérément
3. **Clear caches** nécessite rechargement pages actives pour éviter incohérences
4. **Export report** ne contient PAS de données sensibles (prix, tokens)
5. Le monitor est **read-only** - aucune modification des données applicatives

---

## Troubleshooting

### "OnChain Cache: Not loaded"
**Normal** si `analytics-unified.html` pas encore visité. Première visite popule le cache.

### "ML Pipeline: Unavailable"
Vérifier que backend FastAPI tourne et endpoint `/api/ml/cache/stats` accessible.

### "No Data" dans Data Freshness
Système pas encore initialisé. Visiter pages principales (dashboard, risk-dashboard) puis refresh monitor.

### Stress Test échoue
Vérifier :
1. Backend FastAPI actif (`uvicorn api.main:app`)
2. Source données configurée (`data_source` localStorage)
3. Logs backend pour erreurs spécifiques

---

**Dernière mise à jour** : 2025-09-30
**Version** : 1.0
**Auteur** : Claude + Jack

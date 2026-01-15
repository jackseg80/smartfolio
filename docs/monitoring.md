# Monitoring SmartFolio

> Guide pour le monitoring système et métier

---

## Pages UI

| Page | Usage | URL |
|------|-------|-----|
| `monitoring.html` | Monitoring métier (KPIs, alertes) | `/static/monitoring.html` |
| `monitoring_advanced.html` | Monitoring technique (système) | `/static/monitoring_advanced.html` |

---

## Endpoints API

### Système (santé)

| Endpoint | Description |
|----------|-------------|
| `GET /healthz` | Health check rapide |
| `GET /api/monitoring/health` | Santé système avec latences |
| `GET /api/monitoring/alerts` | Alertes système actives |
| `GET /debug/paths` | Routes disponibles (debug) |

### Portefeuille (métier)

| Endpoint | Description |
|----------|-------------|
| `GET /api/portfolio/metrics` | Métriques portfolio |
| `GET /api/portfolio/alerts` | Alertes portfolio |
| `GET /api/risk/dashboard` | Dashboard risk complet |

---

## Types de Monitoring

### Métier (Portfolio)
- Déviations d'allocation vs cibles
- Erreurs d'exécution
- Alertes P&L
- Decision Index history

### Technique (Système)
- Connectivité APIs externes (CoinGecko, Saxo, etc.)
- Latences Redis
- Santé ML pipeline
- Métriques Prometheus (optionnel)

---

## Configuration Alertes

Les alertes sont configurées dans `services/alerts/`:
- `alert_engine.py` - Logique principale
- `alert_types.py` - Types d'alertes
- `ml_alert_predictor.py` - Prédiction ML

**Voir:** [ML_ALERT_PREDICTOR_REAL_DATA_OCT_2025.md](ML_ALERT_PREDICTOR_REAL_DATA_OCT_2025.md)

---

## Debug

```bash
# Health check rapide
curl http://localhost:8080/healthz

# Vérifier routes
curl http://localhost:8080/debug/paths

# Mode debug verbose
APP_DEBUG=true python -m uvicorn api.main:app --port 8080
```

---

## Cache TTL

| Donnée | TTL | Notes |
|--------|-----|-------|
| On-Chain | 4h | Données blockchain |
| Cycle Score | 24h | Bitcoin cycle |
| ML Sentiment | 15min | Signaux ML |
| Prix crypto | 3min | Prix temps réel |
| Risk Metrics | 30min | Métriques risque |

**Voir:** [CACHE_TTL_OPTIMIZATION.md](CACHE_TTL_OPTIMIZATION.md)

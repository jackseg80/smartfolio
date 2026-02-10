# Monitoring SmartFolio

> Guide pour le monitoring systeme et metier
>
> See also: [PERFORMANCE_MONITOR.md](PERFORMANCE_MONITOR.md) for the unified performance tracking page.

---

## Page UI

| Page | Usage | URL |
|------|-------|-----|
| `monitoring.html` | System Monitoring (health, KPIs, alertes) | `/static/monitoring.html` |

### Architecture monitoring.html (Feb 2026 Refonte)

- **Auth guard** : `core/auth-guard.js` (JWT + X-User)
- **Fetch** : `apiCall()` depuis `core/fetcher.js` (pas de fetch() brut)
- **View modes** : Simple (2 KPIs + alerts table) / Pro (4 KPIs + alerts table)
- **Auto-refresh** : 30s interval
- **Responsive** : 768px, 1400px breakpoints

### Sections

| Section | Mode | Source |
|---------|------|--------|
| Health Banner | Simple+Pro | `/health/all` |
| KPI: System Status | Simple+Pro | `/health/all` (api, redis, ml) |
| KPI: Active Alerts | Simple+Pro | `/api/alerts/active` + `/api/alerts/health` |
| KPI: Circuit Breakers | Pro | `/health/all` (circuit_breakers) |
| KPI: Scheduler | Pro | `/api/scheduler/health` |
| Alerts History table | Simple+Pro | `/api/alerts/active` (filterable, paginated) |

---

## Endpoints API

### System Health

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health/all` | None | Unified health: overall_status, components (api, redis, ml, alerts, scheduler, circuit_breakers) |
| `GET /api/alerts/health` | None | Alerts engine: engine_status, escalations, anti_noise_stats |
| `GET /api/scheduler/health` | None | Scheduler: enabled, jobs_count, jobs (returns 503 if `RUN_SCHEDULER != 1`) |

### Alerts

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /api/alerts/active` | User (X-User) | Active alerts list (filterable: severity, type, limit, offset) |
| `GET /api/alerts/history` | User (X-User) | Historical alerts |

### Admin (RBAC protected)

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /admin/cache/stats` | Admin | Cache stats (in-memory, Redis) |
| `GET /admin/logs/stats` | Admin | Logs by level, recent errors |
| `GET /admin/status` | Admin | System overview stats |

> Note: Admin endpoints are NOT used in monitoring.html (they are in Settings > Advanced and admin-dashboard.html).

### Portfolio (metier)

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /api/portfolio/metrics` | User | Portfolio metrics |
| `GET /api/risk/dashboard` | User | Risk dashboard complet |

---

## Types de Monitoring

### System (monitoring.html)
- Health status per component (API, Redis, ML, Alerts, Scheduler)
- Circuit breaker states (open/closed/half-open)
- Active alerts count + severity breakdown
- Scheduler jobs status + next run times

### Metier (Portfolio)
- Deviations d'allocation vs cibles
- Erreurs d'execution
- Alertes P&L
- Decision Index history

---

## Configuration Alertes

Les alertes sont configurees dans `services/alerts/`:
- `alert_engine.py` - Logique principale
- `alert_types.py` - Types d'alertes
- `alert_storage.py` - Stockage JSON avec FileLock
- `ml_alert_predictor.py` - Prediction ML

**Voir:** [ML_ALERT_PREDICTOR_REAL_DATA_OCT_2025.md](ML_ALERT_PREDICTOR_REAL_DATA_OCT_2025.md)

---

## Debug

```bash
# Unified health check
curl http://localhost:8080/health/all

# Alerts health
curl http://localhost:8080/api/alerts/health

# Scheduler health (503 if disabled)
curl http://localhost:8080/api/scheduler/health

# Active alerts
curl http://localhost:8080/api/alerts/active -H "X-User: jack"

# Mode debug verbose
APP_DEBUG=true python -m uvicorn api.main:app --port 8080
```

---

## Cache TTL

| Donnee | TTL | Notes |
|--------|-----|-------|
| On-Chain | 4h | Donnees blockchain |
| Cycle Score | 24h | Bitcoin cycle |
| ML Sentiment | 15min | Signaux ML |
| Prix crypto | 3min | Prix temps reel |
| Risk Metrics | 30min | Metriques risque |
| Macro Stress | 4h | DXY/VIX indicators |

**Voir:** [CACHE_TTL_OPTIMIZATION.md](CACHE_TTL_OPTIMIZATION.md)

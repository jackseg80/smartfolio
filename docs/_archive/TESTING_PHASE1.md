# Guide de Test - Phase 1 Système d'Alertes

> **Objectif**: Valider le système d'alertes prédictives production-ready intégrant anti-bruit, RBAC, hot-reload et observabilité.

## Démarrage Rapide

### Prérequis

- Python 3.8+
- FastAPI server démarré : `uvicorn api.main:app --port 8080`
- Dépendances installées : `pip install -r requirements.txt`

### Test Automatisé Complet

```bash
# Windows
test_phase1_alerting.bat

# Linux/Mac  
chmod +x test_phase1_alerting.sh && ./test_phase1_alerting.sh
```

## Tests par Composant

### 1. Tests Unitaires

```bash
# AlertEngine core
python -m pytest tests/unit/test_alert_engine.py -v

# Tests spécifiques
python -m pytest tests/unit/test_alert_engine.py::TestAlertEngine::test_config_hot_reload -v
python -m pytest tests/unit/test_alert_engine.py::TestAlertEngine::test_escalation_s2_to_s3 -v
```

**Couvre**:

- [OK] Initialisation AlertEngine avec config file
- [OK] Hot-reload automatique configuration
- [OK] Évaluation alertes avec signaux ML
- [OK] Escalade automatique 2x S2 → S3
- [OK] Collecte métriques observabilité

### 2. Tests d'Intégration API

```bash
# Tous les endpoints alertes
python -m pytest tests/integration/test_alerts_api.py -v

# Tests spécifiques
python -m pytest tests/integration/test_alerts_api.py::TestAlertsAPI::test_prometheus_metrics -v
python -m pytest tests/integration/test_alerts_api.py::TestAlertsAPI::test_config_reload -v
```

**Couvre**:

- [OK] GET `/api/alerts/active` avec filtres
- [OK] POST `/api/alerts/acknowledge/{alert_id}`
- [OK] POST `/api/alerts/snooze/{alert_id}` avec validation
- [OK] GET `/api/alerts/metrics` (JSON + Prometheus)
- [OK] GET `/api/alerts/health` pour monitoring
- [OK] POST `/api/alerts/config/reload` avec RBAC

### 3. Tests Manuels Interactifs

```bash
# Test workflow complet
python tests/manual/test_alerting_workflows.py

# Test hot-reload config
python tests/manual/test_config_hot_reload.py
```

**Scenarios**:

- Health checks API + composants
- Endpoints gouvernance avec RBAC
- Métriques au format Prometheus
- Hot-reload configuration temps réel
- Validation structure config JSON

## Scenarios de Test Manuels

### Scenario 1: Cycle Complet d'Alerte

```bash
# 1. Démarrer serveur
uvicorn api.main:app --port 8080

# 2. Vérifier santé
curl http://localhost:8080/api/alerts/health

# 3. Lister alertes actives  
curl http://localhost:8080/api/alerts/active

# 4. Acquitter une alerte (si existante)
curl -X POST http://localhost:8080/api/alerts/acknowledge/ALERT_ID

# 5. Vérifier métriques
curl http://localhost:8080/api/alerts/metrics/prometheus
```

### Scenario 2: Test Hot-Reload Config

```bash
# 1. Vérifier config actuelle
curl http://localhost:8080/api/alerts/config/current

# 2. Modifier config/alerts_rules.json
# Changer "config_version": "1.0" → "1.1"

# 3. Déclencher reload (si RBAC permet)
curl -X POST http://localhost:8080/api/alerts/config/reload

# 4. Vérifier changement (attendre ~60s pour auto-reload)
curl http://localhost:8080/api/alerts/config/current
```

### Scenario 3: Test Freeze avec TTL

```bash
# 1. Vérifier état gouvernance
curl http://localhost:8080/api/governance/state

# 2. Freeze avec TTL (nécessite Idempotency-Key)
curl -X POST http://localhost:8080/api/governance/freeze \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: test-$(date +%s)" \
  -d '{
    "reason": "Test freeze TTL", 
    "ttl_minutes": 5
  }'

# 3. Vérifier auto_unfreeze_at dans état
curl http://localhost:8080/api/governance/state
```

## Validation des Résultats

### Critères de Succès

**Tests Unitaires**:

- Configuration hot-reload fonctionne
- Escalade S2→S3 se déclenche correctement  
- Métriques sont collectées

**Tests API**:

- Tous les endpoints répondent (200 ou RBAC 401/403)
- Format Prometheus valide
- Pagination historique fonctionne

**Tests Manuels**:

- Health check retourne "healthy"
- Config reload détecte les modifications
- TTL auto-unfreeze s'affiche dans governance state

### Échecs Normaux

**RBAC Protection** (401/403):

- `/api/alerts/config/reload` - Nécessite rôle "approver"
- `/api/alerts/metrics` - Nécessite rôle "viewer"  
- `/api/governance/freeze` - Nécessite rôle "approver"

**Hot-reload**:

- Peut échouer si fichier config verrouillé
- Délai ~60s pour auto-detection

## Debug Common Issues

### Server Non Accessible

```bash
# Vérifier port
netstat -an | findstr 8080

# Redémarrer serveur
uvicorn api.main:app --port 8080 --log-level debug
```

### Config Hot-Reload Échec

```bash
# Vérifier permissions fichier
ls -la config/alerts_rules.json

# Vérifier format JSON
python -m json.tool config/alerts_rules.json
```

### Tests Unitaires Échouent

```bash
# Installer dépendances test
pip install pytest pytest-asyncio

# Exécuter avec debug
python -m pytest tests/unit/test_alert_engine.py -v -s --tb=short
```

## Métriques de Performance

### Objectifs Phase 1

- **Latence P95**: < 100ms pour endpoints alertes
- **Hot-reload**: < 2s après modification fichier  
- **Storage**: Redis primary + file fallback opérationnel
- **Anti-bruit**: Rate limiting + dedup + hystérésis actifs

### Observabilité

```bash
# Métriques JSON détaillées
curl http://localhost:8080/api/alerts/metrics | jq .

# Format Prometheus
curl http://localhost:8080/api/alerts/metrics/prometheus

# Health components
curl http://localhost:8080/api/alerts/health | jq .components
```

## Validation Finale

**Le système Phase 1 est prêt si**:

- [OK] Script `test_phase1_alerting.bat` passe entièrement
- [OK] Health check retourne "healthy"
- [OK] Config hot-reload fonctionne
- [OK] Métriques Prometheus valides
- [OK] RBAC bloque accès non autorisés (401/403)
- [OK] TTL auto-unfreeze visible dans governance state

**Production Readiness**: Le système respecte les patterns Phase 0, intègre anti-bruit robuste, et fournit observabilité complète pour monitoring operationnel.

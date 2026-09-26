# SmartFolio - Commandes Utiles

Guide rapide des commandes pour le développement quotidien de SmartFolio.

## Commandes Slash Claude Code

### Operations Rapides

| Commande | Description | Exemple |
|----------|-------------|---------|
| `/snapshot` | Créer un snapshot P&L | `/snapshot jack cointracking` |
| `/clear-cache` | Nettoyer tous les caches | `/clear-cache` |
| `/health-check` | Vérifier tous les services | `/health-check` |

### Code Review Automatisé

| Commande | Description | Mode |
|----------|-------------|------|
| `/review-multitenant` | Audit règles multi-tenant | Fork |
| `/review-risk` | Détecter inversions Risk Score | Fork |
| `/review-allocation` | Vérifier Allocation Engine V2 | Fork |

### Documentation

| Commande | Description |
|----------|-------------|
| `/doc-commit` | Mettre à jour docs + commit sans co-auteur |

** Documentation complète:** [.claude/commands/README.md](.claude/commands/README.md)

---

## Commandes Shell

### Démarrage Serveur

**Windows (PowerShell):**
```powershell
# Serveur dev basique
.\start_dev.ps1

# Avec scheduler (P&L snapshots, OHLCV updates)
.\start_dev.ps1 -EnableScheduler

# Avec hot reload (sans Playwright)
.\start_dev.ps1 -Reload

# Port personnalisé
.\start_dev.ps1 -Port 8081
```

**Linux/macOS (Bash):**
```bash
# Serveur dev basique
./start_dev.sh

# Avec scheduler
./start_dev.sh --enable-scheduler

# Avec hot reload
./start_dev.sh --reload

# Port personnalisé
./start_dev.sh --port 8081
```

---

### Tests

```bash
# Tests unitaires
pytest tests/unit -v

# Tests d'intégration
pytest tests/integration -v

# Tests E2E
npm run test:e2e

# Coverage
pytest --cov=services --cov=api --cov-report=html
```

---

### Redis

```bash
# Démarrer Redis
redis-server  # Linux/macOS
wsl -d Ubuntu bash -c "sudo service redis-server start"  # Windows WSL

# Arrêter Redis
redis-cli shutdown  # Linux/macOS
wsl -d Ubuntu bash -c "sudo service redis-server stop"  # Windows WSL

# Vérifier Redis
redis-cli ping  # Doit retourner PONG

# Nettoyer Redis
redis-cli FLUSHALL
```

---

### Docker

```bash
# Build image
docker build -t smartfolio:latest .

# Démarrage auto (pull + build + restart)
./deploy.sh

# Démarrage manuel
docker-compose up -d

# Arrêter
docker-compose down

# Logs
docker-compose logs -f

# Rebuild complet
docker-compose up -d --build

# Cleanup
docker-compose down -v
docker system prune -f
```

---

## Commandes API (curl)

### Health & Config

```bash
# Status application
curl http://localhost:8080/healthz | python -m json.tool

# Config frontend
curl http://localhost:8080/api/config | python -m json.tool

# Health scheduler
curl http://localhost:8080/api/scheduler/health | python -m json.tool
```

### Portfolio

```bash
# Balances actuelles
curl "http://localhost:8080/balances/current?source=cointracking" \
  -H "X-User: demo" | python -m json.tool

# Metrics portfolio
curl "http://localhost:8080/portfolio/metrics?user_id=demo" \
  | python -m json.tool

# Créer snapshot P&L
curl -X POST "http://localhost:8080/portfolio/snapshot?user_id=demo&source=cointracking" \
  | python -m json.tool
```

### ML & Analytics

```bash
# ML Sentiment BTC
curl "http://localhost:8080/api/ml/sentiment/symbol/BTC" \
  | python -m json.tool

# Cycle Score
curl "http://localhost:8080/api/ml/cycle_score" \
  | python -m json.tool

# On-Chain Score
curl "http://localhost:8080/api/ml/onchain_score" \
  | python -m json.tool
```

### Risk

```bash
# Risk dashboard complet
curl "http://localhost:8080/api/risk/dashboard" \
  -H "X-User: demo" | python -m json.tool

# Risk dashboard Saxo (stocks)
curl "http://localhost:8080/api/risk/bourse/dashboard" \
  -H "X-User: jack" | python -m json.tool
```

### Admin (RBAC - user "jack")

```bash
# Liste users
curl "http://localhost:8080/admin/users" \
  -H "X-User: jack" | python -m json.tool

# ML models status
curl "http://localhost:8080/admin/ml/models" \
  -H "X-User: jack" | python -m json.tool

# Train ML model
curl -X POST "http://localhost:8080/admin/ml/train/btc_regime_detector" \
  -H "X-User: jack" | python -m json.tool

# Cache stats
curl "http://localhost:8080/admin/cache/stats" \
  -H "X-User: jack" | python -m json.tool
```

### Sources

```bash
# Liste sources disponibles
curl "http://localhost:8080/api/sources/list" \
  -H "X-User: demo" | python -m json.tool

# Test source
curl "http://localhost:8080/api/sources/test?source_name=cointracking" \
  -H "X-User: demo" | python -m json.tool

# Upload CSV
curl -X POST "http://localhost:8080/api/sources/upload" \
  -H "X-User: demo" \
  -F "file=@data/users/demo/cointracking/data/export.csv" \
  -F "source_name=cointracking" | python -m json.tool
```

---

## Makefile (Commandes Make)

Si Make est installé (Linux/macOS/WSL/Git Bash):

```bash
# Aide
make help

# Installation
make install          # Dépendances
make install-dev      # + dev dependencies
make playwright       # Installer Playwright

# Développement
make dev              # Démarrer serveur
make dev-scheduler    # Avec scheduler
make dev-reload       # Avec hot reload

# Tests
make test             # Tous les tests
make test-unit        # Tests unitaires uniquement
make test-integration # Tests d'intégration uniquement
make test-coverage    # Avec coverage

# Qualité code
make lint             # Linting
make format           # Format code (Black + Ruff)
make security-scan    # Scan vulnérabilités

# Redis
make redis-start      # Démarrer Redis
make redis-stop       # Arrêter Redis
make cache-clear      # Nettoyer cache

# Data
make snapshot         # Créer snapshot P&L
make backup           # Backup user data
make restore          # Restore latest backup

# Monitoring
make health           # Health check simple
make health-full      # Health check complet
make logs             # Tail logs
make logs-error       # Logs erreur uniquement

# Docker
make docker-build     # Build image
make docker-up        # Start containers
make docker-down      # Stop containers
make docker-deploy    # Deploy (rebuild + restart)

# ML
make ml-train         # Entraîner tous les modèles
make ml-status        # Status modèles ML

# Cleanup
make clean            # Nettoyer fichiers temp
make clean-all        # Deep clean (+ cache + logs)
make reset            # Reset environnement complet

# Quick Actions
make quick-start      # Redis + Dev server
make quick-test       # Unit + integration tests
make quick-check      # Lint + unit tests
make ci               # Pipeline CI complète
```

---

## npm Scripts

```bash
# Tests JavaScript
npm test              # Run tests
npm run test:watch    # Watch mode
npm run test:ui       # UI Vitest
npm run test:coverage # Coverage

# Tests E2E
npm run test:e2e        # Run E2E tests
npm run test:e2e:ui     # E2E UI mode
npm run test:e2e:debug  # Debug mode
npm run test:e2e:headed # Headed browser
npm run test:e2e:report # Show report
```

---

## Monitoring & Logs

### Logs Application

```bash
# Tail logs (Windows)
powershell -Command "Get-Content logs\smartfolio.log -Wait -Tail 50"

# Tail logs (Linux/macOS)
tail -f logs/smartfolio.log

# Erreurs seulement (Windows)
powershell -Command "Select-String -Path logs\smartfolio.log -Pattern 'ERROR|CRITICAL' | Select-Object -Last 50"

# Erreurs seulement (Linux/macOS)
grep -E "ERROR|CRITICAL" logs/smartfolio.log | tail -50
```

### Monitoring Temps Réel

URLs à surveiller:
- **Healthz:** http://localhost:8080/healthz
- **Scheduler:** http://localhost:8080/api/scheduler/health
- **Monitoring:** http://localhost:8080/static/monitoring.html
- **API Docs:** http://localhost:8080/docs

---

## Git

### Commits

```bash
# Commit standard (avec co-auteur Claude)
git add .
git commit -m "feat: add new feature

Description détaillée.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Commit docs uniquement (sans co-auteur)
git add docs/
git commit -m "docs: update documentation"

# Amend dernier commit
git commit --amend --no-edit
```

### Pull Requests

```bash
# Créer PR avec gh CLI
gh pr create --title "feat: nouvelle feature" --body "Description"

# Lire commentaires PR
gh pr view 123
gh api repos/owner/repo/pulls/123/comments
```

### Hooks Pre-commit

```bash
# Installer hooks
pip install pre-commit
pre-commit install

# Run manuellement
pre-commit run --all-files

# Skip hooks (si urgent)
git commit --no-verify -m "hotfix: critical fix"
```

---

## Déploiement Production

### Docker Deployment

```bash
# Déploiement automatisé
./deploy.sh

# Déploiement sans rebuild
./deploy.sh --skip-build

# Vérifier santé après déploiement
curl http://localhost:8080/healthz
./scripts/check_production_health.sh
```

### Backup Avant Déploiement

```bash
# Créer backup
tar -czf backups/smartfolio-backup-$(date +%Y%m%d-%H%M%S).tar.gz data/users config/users.json

# Restore si problème
tar -xzf backups/smartfolio-backup-YYYYMMDD-HHMMSS.tar.gz
```

---

## Documentation

### Générer Docs API

```bash
# Régénérer API_REFERENCE.md
python scripts/dev_tools/gen_api_reference.py

# Vérifier liens cassés
python scripts/dev_tools/gen_broken_refs.py
```

### Pages Documentation

- **Index:** [docs/index.md](docs/index.md)
- **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **API Reference:** [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- **User Guide:** [docs/user-guide.md](docs/user-guide.md)
- **CLAUDE.md:** [CLAUDE.md](CLAUDE.md) - Guide pour agents IA

---

## Troubleshooting

### Port déjà utilisé

```bash
# Trouver processus sur port 8080 (Windows)
Get-Process -Id (Get-NetTCPConnection -LocalPort 8080).OwningProcess

# Trouver processus sur port 8080 (Linux/macOS)
lsof -i :8080

# Démarrer sur autre port
.\start_dev.ps1 -Port 8081
```

### Redis non accessible

```bash
# Vérifier status
redis-cli ping

# Redémarrer
wsl -d Ubuntu bash -c "sudo service redis-server restart"  # Windows
sudo service redis-server restart  # Linux
```

### Cache corrompus

```bash
# Nettoyer tous les caches
/clear-cache  # Commande slash

# OU manuellement:
redis-cli FLUSHALL
rm -rf cache/ml_pipeline/*
rm -rf cache/multi_asset/*
```

### Tests qui échouent

```bash
# Nettoyer et relancer
make clean
pytest tests/unit -v --tb=short

# Mode debug
pytest tests/unit -v -s --pdb
```

---

## Support

- **Issues GitHub:** Créer une issue
- **Documentation:** [docs/troubleshooting.md](docs/troubleshooting.md)
- **Runbooks:** [docs/runbooks.md](docs/runbooks.md)

---

**Dernière mise à jour:** Janvier 2026
**Version:** 3.0

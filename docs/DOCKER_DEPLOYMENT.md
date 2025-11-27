# 🐳 SmartFolio - Docker Deployment Guide

Guide de déploiement Docker pour SmartFolio en production (Ubuntu 24.04.2 LTS).

## 📋 Prérequis

- Docker Engine 24.0+
- Docker Compose v2.20+
- Ubuntu 24.04.2 LTS (ou compatible)
- Minimum 4GB RAM, 20GB disque
- Port 8080 disponible

---

## 🚀 Déploiement Rapide (TL;DR)

```bash
# 1. Cloner le projet
git clone https://github.com/your-org/smartfolio.git
cd smartfolio

# 2. Créer .env depuis template
cp .env.docker .env
# Éditer .env et changer DEBUG_TOKEN, ADMIN_KEY

# 3. Lancer
docker-compose -f docker-compose.prod.yml up -d

# 4. Vérifier
docker-compose -f docker-compose.prod.yml logs -f
curl http://localhost:8080/docs
```

---

## 📁 Structure Fichiers

```
smartfolio/
├── docker-compose.prod.yml    # Config Docker Compose production
├── Dockerfile.prod            # Image Docker production
├── .env.docker                # Template variables d'environnement
├── .env                       # Fichier .env réel (gitignored, créer depuis .env.docker)
├── .dockerignore              # Exclusions build Docker
├── data/                      # Données utilisateurs (volume persistant)
├── logs/                      # Logs application (volume persistant)
└── docs/
    └── DOCKER_DEPLOYMENT.md   # Ce fichier
```

---

## ⚙️ Configuration

### 1. Créer le Fichier .env

```bash
# Sur serveur production
cp .env.docker .env
nano .env
```

### 2. Variables Critiques à Modifier

```bash
# .env
PORT=8080                      # Port d'écoute (LAN accessible)
DEBUG=false                    # IMPORTANT: false en production!
ENVIRONMENT=production         # Indique environnement prod
LOG_LEVEL=INFO                 # INFO ou WARNING en prod

# Sécurité - CHANGEZ CES VALEURS!
DEBUG_TOKEN=your-prod-token-xyz123       # Token fort (32+ caractères)
ADMIN_KEY=your-admin-key-abc456          # Clé admin forte

# API Base URL (interne au container)
API_BASE_URL=http://localhost:8080       # NE PAS CHANGER

# Redis (utilise service Docker 'redis')
REDIS_URL=redis://redis:6379/0           # NE PAS CHANGER
```

### 3. Variables Optionnelles

```bash
# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=120       # Ajuster selon charge

# CORS (si frontend externe)
CORS_ORIGINS=https://smartfolio.example.com,https://app.example.com

# Feature flags
RISK_SCORE_V2_ENABLED=true
RUN_SCHEDULER=1
```

---

## 🏗️ Construction et Lancement

### Première Installation

```bash
# 1. Construire l'image (peut prendre 5-10 min)
docker-compose -f docker-compose.prod.yml build

# 2. Lancer les services
docker-compose -f docker-compose.prod.yml up -d

# 3. Vérifier santé
docker-compose -f docker-compose.prod.yml ps
docker-compose -f docker-compose.prod.yml logs smartfolio | tail -20
```

### Commandes Utiles

```bash
# Voir logs en temps réel
docker-compose -f docker-compose.prod.yml logs -f

# Logs filtrés (erreurs uniquement)
docker-compose -f docker-compose.prod.yml logs smartfolio | grep ERROR

# Redémarrer service
docker-compose -f docker-compose.prod.yml restart smartfolio

# Arrêter tout
docker-compose -f docker-compose.prod.yml down

# Arrêter ET supprimer volumes (⚠️ perte données!)
docker-compose -f docker-compose.prod.yml down -v
```

---

## 🔄 Mise à Jour du Code

```bash
# 1. Pull dernières modifications
git pull origin main

# 2. Rebuild image (force rebuild)
docker-compose -f docker-compose.prod.yml build --no-cache

# 3. Restart avec nouvelle image
docker-compose -f docker-compose.prod.yml up -d --force-recreate

# 4. Cleanup images anciennes
docker image prune -f
```

---

## 🩺 Health Checks

### Automatiques (Docker)

- **Redis**: `redis-cli ping` toutes les 10s
- **SmartFolio API**: `curl http://localhost:8080/docs` toutes les 30s
- Start period: 60s (temps d'init Playwright)

### Manuels

```bash
# Status global
docker-compose -f docker-compose.prod.yml ps

# Health smartfolio
docker exec smartfolio-api curl -f http://localhost:8080/health || echo "❌ Failed"

# Health redis
docker exec smartfolio-redis redis-cli ping

# Métriques système
docker stats smartfolio-api smartfolio-redis
```

---

## 📊 Monitoring et Logs

### Logs Docker

```bash
# Logs combinés (tous services)
docker-compose -f docker-compose.prod.yml logs -f

# Logs smartfolio uniquement
docker-compose -f docker-compose.prod.yml logs -f smartfolio

# Logs redis uniquement
docker-compose -f docker-compose.prod.yml logs -f redis

# Dernières 100 lignes
docker-compose -f docker-compose.prod.yml logs --tail=100 smartfolio
```

### Logs Application (dans container)

```bash
# Via volume monté (accessible depuis host)
tail -f logs/app.log

# Depuis host
docker exec smartfolio-api tail -f /app/logs/app.log

# Recherche erreurs
docker exec smartfolio-api grep ERROR /app/logs/app.log | tail -20
```

### Rotation Automatique

- **Logs Docker**: max 10MB × 3 fichiers = 30MB total
- **Logs App**: max 5MB × 3 backups = 15MB total (rotation Python)

---

## 🔧 Dépannage

### Service ne démarre pas

```bash
# 1. Vérifier logs
docker-compose -f docker-compose.prod.yml logs smartfolio

# 2. Vérifier healthcheck
docker inspect smartfolio-api | grep -A 20 Health

# 3. Tester manuellement dans container
docker exec -it smartfolio-api bash
curl http://localhost:8080/docs
```

### Erreur "Port déjà utilisé"

```bash
# Trouver processus sur port 8080
sudo lsof -i :8080
# ou
sudo netstat -tulpn | grep 8080

# Tuer processus
sudo kill -9 <PID>
```

### Redis connection failed

```bash
# Vérifier Redis
docker exec smartfolio-redis redis-cli ping  # Doit répondre PONG

# Restart Redis
docker-compose -f docker-compose.prod.yml restart redis

# Logs Redis
docker-compose -f docker-compose.prod.yml logs redis
```

### Pas d'accès depuis LAN

```bash
# 1. Vérifier firewall Ubuntu
sudo ufw status
sudo ufw allow 8080/tcp

# 2. Vérifier binding container
docker-compose -f docker-compose.prod.yml ps
# Port doit être: 0.0.0.0:8080->8080/tcp

# 3. Tester depuis host
curl http://localhost:8080/docs

# 4. Tester depuis autre machine LAN
curl http://<IP_SERVER>:8080/docs
```

---

## 🔐 Sécurité Production

### Checklist

- [ ] `DEBUG=false` dans .env
- [ ] `DEBUG_TOKEN` changé (32+ caractères aléatoires)
- [ ] `ADMIN_KEY` changé (32+ caractères aléatoires)
- [ ] Firewall activé (`ufw enable`)
- [ ] Port 8080 ouvert uniquement sur LAN (pas Internet)
- [ ] Redis pas exposé publiquement (réseau interne Docker uniquement)
- [ ] .env gitignored (ne jamais commit)
- [ ] Logs rotation activée
- [ ] Backups réguliers `data/` et `redis_data`

### Génération Tokens Sécurisés

```bash
# Token fort (32 caractères)
openssl rand -hex 32

# Ou avec Python
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## 💾 Backup & Restore

### Backup

```bash
# Script backup complet
#!/bin/bash
BACKUP_DIR="/backup/smartfolio/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup data/
cp -r data/ "$BACKUP_DIR/data"

# Backup Redis (dump.rdb)
docker exec smartfolio-redis redis-cli BGSAVE
sleep 5
docker cp smartfolio-redis:/data/dump.rdb "$BACKUP_DIR/redis_dump.rdb"

# Backup .env (sécurisé)
cp .env "$BACKUP_DIR/.env.backup"

# Compress
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "✅ Backup créé: $BACKUP_DIR.tar.gz"
```

### Restore

```bash
# 1. Arrêter services
docker-compose -f docker-compose.prod.yml down

# 2. Extraire backup
tar -xzf backup_20250127_120000.tar.gz

# 3. Restore data/
rm -rf data/
cp -r backup_20250127_120000/data/ ./data/

# 4. Restore Redis
docker-compose -f docker-compose.prod.yml up -d redis
sleep 10
docker cp backup_20250127_120000/redis_dump.rdb smartfolio-redis:/data/dump.rdb
docker-compose -f docker-compose.prod.yml restart redis

# 5. Restore .env
cp backup_20250127_120000/.env.backup .env

# 6. Relancer tout
docker-compose -f docker-compose.prod.yml up -d
```

---

## 📈 Performance Tuning

### Limites Ressources (Optionnel)

Décommenter dans `docker-compose.prod.yml` :

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'      # Max 2 cores
      memory: 4G       # Max 4GB RAM
    reservations:
      cpus: '0.5'      # Min 0.5 core
      memory: 512M     # Min 512MB RAM
```

### Redis Tuning

Ajuster `maxmemory` selon RAM disponible :

```yaml
# docker-compose.prod.yml (ligne 17)
command: >
  redis-server
  --maxmemory 1gb          # Ajuster selon RAM serveur
  --maxmemory-policy allkeys-lru
```

---

## 🌐 Accès depuis Internet (Optionnel)

### Avec Reverse Proxy (Nginx)

```nginx
# /etc/nginx/sites-available/smartfolio
server {
    listen 80;
    server_name smartfolio.example.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### SSL avec Certbot

```bash
sudo certbot --nginx -d smartfolio.example.com
```

---

## 📞 Support

- **Documentation**: `docs/ARCHITECTURE.md`, `CLAUDE.md`
- **Issues**: GitHub Issues
- **Logs**: `logs/app.log` (application), `docker-compose logs` (containers)

---

**Dernière mise à jour**: 2025-01-27
**Version Docker Compose**: 3.8
**SmartFolio Version**: 2.0+

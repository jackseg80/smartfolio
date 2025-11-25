# SmartFolio - Migration Docker vers Linux NUC

Guide complet de migration depuis Windows vers Ubuntu 24.04.2 LTS avec Docker.

---

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Prérequis](#prérequis)
3. [Phase 1 : Préparation (Windows)](#phase-1--préparation-windows)
4. [Phase 2 : Transfert vers NUC](#phase-2--transfert-vers-nuc)
5. [Phase 3 : Build & Démarrage](#phase-3--build--démarrage)
6. [Phase 4 : Tests & Validation](#phase-4--tests--validation)
7. [Workflow Dev Windows → Prod NUC](#workflow-dev-windows--prod-nuc)
8. [Commandes Utiles](#commandes-utiles)
9. [Troubleshooting](#troubleshooting)

---

## Vue d'ensemble

### Architecture Docker

```
┌─────────────────────────────────────────┐
│         NUC Ubuntu 24.04.2 LTS          │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │     Docker Compose Stack          │  │
│  │                                   │  │
│  │  ┌─────────────┐  ┌────────────┐ │  │
│  │  │   Redis     │  │ SmartFolio │ │  │
│  │  │   Alpine    │←→│    API     │ │  │
│  │  │   (Cache)   │  │  (Python)  │ │  │
│  │  └─────────────┘  └────────────┘ │  │
│  │         ↓               ↓         │  │
│  │  ┌──────────────────────────┐    │  │
│  │  │   Named Volume           │    │  │
│  │  │   redis_data             │    │  │
│  │  └──────────────────────────┘    │  │
│  │         ↓               ↓         │  │
│  │  ┌──────────┐    ┌──────────┐   │  │
│  │  │ ./data/  │    │ ./logs/  │   │  │
│  │  │ (bind)   │    │ (bind)   │   │  │
│  │  └──────────┘    └──────────┘   │  │
│  └───────────────────────────────────┘  │
│                 ↑                        │
│          Port 8080 (LAN)                 │
└─────────────────────────────────────────┘
              ↑
       Windows Browser
   http://<nuc-ip>:8080
```

### Nouveaux Fichiers Créés

- **`Dockerfile.prod`** : Image optimisée avec healthcheck, Python 3.11
- **`docker-compose.prod.yml`** : Stack complète (Redis + API + volumes + auto-start)
- **`.env.docker.example`** : Template configuration Docker-ready

### Avantages Docker

✅ **Setup simplifié** : 1 commande pour tout démarrer
✅ **Auto-start** : Redémarrage automatique au boot NUC
✅ **Isolation** : Conteneurs séparés (pas de pollution système)
✅ **Healthchecks** : Redémarrage auto si panne
✅ **Redis intégré** : Cache + streaming opérationnels
✅ **Portabilité** : Fonctionne sur n'importe quel Linux avec Docker

---

## Prérequis

### Sur Windows

- ✅ Git installé
- ✅ Accès SSH au NUC (`ssh user@nuc-ip`)
- ✅ Projet SmartFolio fonctionnel

### Sur NUC Ubuntu 24.04.2 LTS

- ✅ Docker installé (vous l'avez déjà)
- ✅ Docker Compose installé
- ✅ Accès internet (APIs externes : CoinGecko, Binance, etc.)
- ✅ Port 8080 disponible

**Vérifier Docker sur NUC :**

```bash
# Via SSH sur NUC
docker --version          # Doit afficher : Docker version 24.x.x
docker-compose --version  # Doit afficher : docker-compose version 1.29.x ou 2.x
docker ps                 # Doit fonctionner sans erreur
```

---

## Phase 1 : Préparation (Windows)

### Étape 1.1 : Vérifier les fichiers Docker créés

```powershell
# Vérifier que les 3 fichiers existent
ls Dockerfile.prod
ls docker-compose.prod.yml
ls .env.docker.example
```

### Étape 1.2 : Test de build local (OPTIONNEL mais recommandé)

```powershell
# Test build pour détecter erreurs avant transfert NUC
docker-compose -f docker-compose.prod.yml build

# Si succès, vous verrez :
# Successfully built <image-id>
# Successfully tagged smartfolio-api:latest
```

⚠️ **Note Windows** : Le build peut être lent (~5-10 min première fois). Sur NUC Linux, ce sera plus rapide.

### Étape 1.3 : Commit et push vers Git

```powershell
# Ajouter les nouveaux fichiers Docker
git add Dockerfile.prod docker-compose.prod.yml .env.docker.example README-DOCKER.md
git commit -m "feat(docker): add production Docker setup for Linux NUC

- Dockerfile.prod with healthcheck and optimizations
- docker-compose.prod.yml with Redis and auto-start
- .env.docker template for Docker networking
- Migration guide README-DOCKER.md

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

git push
```

---

## Phase 2 : Transfert vers NUC

### Étape 2.1 : Connexion SSH et clone projet

```bash
# Sur NUC (via SSH)
ssh user@<nuc-ip>

# Créer dossier projet (exemple : /opt/smartfolio)
cd /opt
sudo mkdir -p smartfolio
sudo chown $USER:$USER smartfolio
cd smartfolio

# Cloner le repository
git clone <votre-repo-url> .

# Vérifier que les fichiers Docker sont présents
ls -la Dockerfile.prod docker-compose.prod.yml .env.docker.example
```

### Étape 2.2 : Transférer données utilisateurs

```powershell
# Sur Windows (PowerShell)
# Remplacer <nuc-ip> et <user> par vos valeurs

# Transférer data/users/ (portfolios, cache, config)
scp -r data/users/* <user>@<nuc-ip>:/opt/smartfolio/data/users/

# Transférer P&L history
scp data/portfolio_history.json <user>@<nuc-ip>:/opt/smartfolio/data/

# Transférer secrets API (CRITIQUE)
scp data/users/demo/secrets.json <user>@<nuc-ip>:/opt/smartfolio/data/users/demo/
scp data/users/jack/secrets.json <user>@<nuc-ip>:/opt/smartfolio/data/users/jack/
```

**Vérifier transfert réussi :**

```bash
# Sur NUC
ls -R data/users/
# Doit afficher : demo/, jack/ avec leurs fichiers
```

### Étape 2.2b : Transférer Cache Prix Historiques (CRITIQUE ⚠️)

**Problème :** Le cache de prix historiques (`data/price_history/`) contient **127 fichiers JSON** (3000+ jours d'historique BTC, ETH, etc.). Sans ce cache, les **métriques de risque seront incorrectes** :

| Métrique | Avec Cache (Windows) | Sans Cache (Linux) | Impact |
|----------|---------------------|-------------------|--------|
| **Risk Score** | 69.5/100 | 39.6/100 | ❌ -43% |
| **Effective Assets** | 132 | 10 | ❌ -92% |
| **Long-Term Window** | 365 jours | 120 jours | ❌ -67% |
| **Full Intersection** | 1154 jours | 365 jours | ❌ -68% |

**Solution : Transférer le cache complet depuis Windows**

```powershell
# Sur Windows (PowerShell)
# Méthode 1 : SCP direct (RECOMMANDÉ - 1-2 minutes)
scp -r d:\Python\smartfolio\data\price_history\*.json <user>@<nuc-ip>:/tmp/price_cache/

# Si "Permission denied", préparer le dossier sur Ubuntu d'abord :
```

```bash
# Sur NUC (avant le SCP)
mkdir -p /tmp/price_cache
sudo chown $USER:$USER /tmp/price_cache

# Puis après le SCP, déplacer vers le bon dossier :
mkdir -p /opt/smartfolio/data/price_history
mv /tmp/price_cache/*.json /opt/smartfolio/data/price_history/
chown -R $USER:$USER /opt/smartfolio/data/price_history/
```

**Vérifier cache transféré :**

```bash
# Sur NUC
cd /opt/smartfolio

# Compter fichiers (attendu : 127)
ls data/price_history/*.json | wc -l

# Vérifier taille cache (attendu : ~1.6 MB)
du -sh data/price_history/

# Vérifier historique BTC (attendu : 3000+ jours)
python3 -c "import json; data = json.load(open('data/price_history/BTC_1d.json')); print(f'BTC: {len(data)} jours')"
# Sortie attendue : BTC: 3022 jours
```

**⚠️ IMPORTANT :** Sans ce cache, Docker démarrera MAIS les métriques Risk Dashboard seront **incorrectes** (Risk Score -43%, Effective Assets -92%). Les APIs externes (Binance, Kraken) ne fournissent que 365 jours max via API.

**Alternative : Re-téléchargement complet (LENT - 15-30 min)**

Si vous ne pouvez pas transférer depuis Windows :

```bash
# Sur NUC (après Phase 3 - Docker démarré)
docker exec -it smartfolio-api python scripts/download_historical_data.py --days 3000 --all
```

⚠️ **Limitations :** Binance API rate limits peuvent causer des échecs aléatoires. La copie depuis Windows est **toujours préférable**.

### Étape 2.3 : Créer .env depuis template

```bash
# Sur NUC
cd /opt/smartfolio

# Copier template
cp .env.docker.example .env

# Optionnel : éditer si besoin de personnaliser
nano .env
# Vérifier HOST=0.0.0.0, PORT=8080, REDIS_URL=redis://redis:6379/0
```

---

## Phase 3 : Build & Démarrage

### Étape 3.1 : Build image Docker

```bash
# Sur NUC
cd /opt/smartfolio

# Build l'image (première fois : 5-8 min)
docker-compose -f docker-compose.prod.yml build

# Sortie attendue :
# Successfully built <hash>
# Successfully tagged smartfolio-smartfolio:latest
```

### Étape 3.2 : Démarrer le stack

```bash
# Démarrer en background (-d = detached)
docker-compose -f docker-compose.prod.yml up -d

# Sortie attendue :
# Creating network "smartfolio-network" ... done
# Creating volume "smartfolio-redis-data" ... done
# Creating smartfolio-redis ... done
# Creating smartfolio-api ... done
```

### Étape 3.3 : Vérifier démarrage

```bash
# Voir les conteneurs actifs
docker ps

# Sortie attendue :
# CONTAINER ID   IMAGE                      STATUS                 PORTS                    NAMES
# abc123def456   smartfolio-smartfolio      Up 30s (healthy)       0.0.0.0:8080->8080/tcp   smartfolio-api
# 789ghi012jkl   redis:7-alpine             Up 35s (healthy)       6379/tcp                 smartfolio-redis

# Voir les logs en temps réel
docker-compose -f docker-compose.prod.yml logs -f

# Attendre ces lignes :
# smartfolio-redis | Ready to accept connections
# smartfolio-api   | Application startup complete
# smartfolio-api   | Redis ready at redis://redis:6379/0
# smartfolio-api   | Scheduler initialized (RUN_SCHEDULER=1)
```

**Appuyer sur Ctrl+C pour quitter les logs** (conteneurs continuent de tourner).

---

## Phase 4 : Tests & Validation

### Étape 4.1 : Test API local (sur NUC)

```bash
# Sur NUC
curl http://localhost:8080/docs
# Doit retourner HTML (Swagger UI)

curl http://localhost:8080/balances/current?user_id=demo
# Doit retourner JSON avec balances

curl http://localhost:8080/api/ml/sentiment/symbol/BTC
# Doit retourner JSON avec sentiment ML
```

### Étape 4.2 : Test accès LAN (depuis Windows)

```
Ouvrir navigateur Windows :

http://<nuc-ip>:8080/
http://<nuc-ip>:8080/dashboard.html
http://<nuc-ip>:8080/analytics-unified.html
```

**Trouver IP NUC :**

```bash
# Sur NUC
ip addr show | grep inet
# Exemple : inet 192.168.1.50/24
```

### Étape 4.3 : Vérifier healthchecks

```bash
# Sur NUC
docker ps

# Colonne STATUS doit afficher "healthy" :
# Up 2 minutes (healthy)
```

Si "unhealthy", voir [Troubleshooting](#troubleshooting).

### Étape 4.3b : Vérifier Cohérence Métriques Risk Dashboard ⚠️

**CRITIQUE :** Vérifiez que les métriques sur Ubuntu **correspondent** à Windows. Des divergences indiquent un cache prix manquant/incomplet.

```bash
# Sur NUC - Tester API Risk Dashboard
curl -s "http://localhost:8080/api/risk/dashboard?user_id=jack" | jq '{
  risk_score: .data.risk_score,
  effective_assets: .data.effective_assets,
  long_term_days: .data.long_term_window.days,
  long_term_assets: .data.long_term_window.assets,
  full_intersection_days: .data.full_intersection_window.days
}'
```

**Résultats Attendus (avec cache complet) :**

```json
{
  "risk_score": 69.5,
  "effective_assets": 132,
  "long_term_days": 365,
  "long_term_assets": 132,
  "full_intersection_days": 1154
}
```

**❌ Divergences Typiques (cache manquant) :**

```json
{
  "risk_score": 39.6,           // ❌ -43% vs attendu (69.5)
  "effective_assets": 10,       // ❌ -92% vs attendu (132)
  "long_term_days": 120,        // ❌ -67% vs attendu (365)
  "long_term_assets": 10,       // ❌ -92% vs attendu (132)
  "full_intersection_days": 365 // ❌ -68% vs attendu (1154)
}
```

**🔍 Diagnostic si divergence :**

```bash
# 1. Vérifier cache prix existe
ls data/price_history/*.json | wc -l
# Attendu : 127 fichiers
# Si 0-10 : Cache manquant → Voir Étape 2.2b

# 2. Vérifier historique BTC
docker exec smartfolio-api python -c "import json; data = json.load(open('data/price_history/BTC_1d.json')); print(f'BTC: {len(data)} jours')"
# Attendu : BTC: 3022 jours
# Si <365 : Cache incomplet → Re-transférer depuis Windows

# 3. Si cache OK mais métriques incorrectes : Restart API
docker-compose -f docker-compose.prod.yml restart smartfolio
```

**📊 Test Visual (Risk Dashboard Web) :**

Ouvrir dans navigateur Windows : `http://<nuc-ip>:8080/risk-dashboard.html`

**Indicateurs à vérifier :**

| Métrique | Valeur Attendue | Signe Problème |
|----------|----------------|----------------|
| **Risk Score** | 65-75/100 | < 50/100 ❌ |
| **Effective Assets** | 100-150 | < 20 ❌ |
| **Long-Term Window** | "365d, 120+ assets" | "120d, 10 assets" ❌ |
| **Full Intersection** | "1000+ jours" | "365 jours" ❌ |

**✅ Si métriques cohérentes** : Cache OK, Docker opérationnel !

**❌ Si métriques divergent** : Retour Étape 2.2b (transférer cache prix)

### Étape 4.4 : Tester auto-start au boot

```bash
# Sur NUC
sudo reboot

# Attendre 2-3 minutes, puis reconnecter SSH
ssh user@<nuc-ip>

# Vérifier que les conteneurs sont UP
docker ps

# Les 2 conteneurs doivent être présents (smartfolio-api, smartfolio-redis)

# Tester API
curl http://localhost:8080/docs
```

✅ **Si succès = Migration terminée !**

---

## Workflow Dev Windows → Prod NUC

### Après modifications sur Windows

```powershell
# 1. Commit & push
git add .
git commit -m "feat: description changement"
git push
```

### Déploiement sur NUC

```bash
# 2. SSH vers NUC
ssh user@<nuc-ip>
cd /opt/smartfolio

# 3. Pull dernières modifications
git pull

# 4. Rebuild & restart (rebuild seulement si code Python modifié)
docker-compose -f docker-compose.prod.yml up -d --build

# Rebuild : 30-60 sec (Docker cache layers)

# 5. Vérifier logs
docker-compose -f docker-compose.prod.yml logs -f --tail 50

# Attendre "Application startup complete"
```

### ⚠️ Si "git pull" échoue avec conflits

**Symptômes :**

```text
error: Your local changes to the following files would be overwritten by merge:
        config/score_registry.json
        services/ml/model_registry.py
Please commit your changes or stash them before you merge.
Aborting
```

**Cause :** Le serveur Ubuntu (production) ne devrait **jamais** avoir de modifications locales. Windows est la machine de dev (source de vérité).

**Solution 1 : Écraser avec GitHub (RECOMMANDÉ ✅)**

```bash
# Sur NUC - Sauvegarder au cas où
git diff > /tmp/smartfolio_local_changes_$(date +%Y%m%d_%H%M%S).patch

# Écraser avec la version GitHub (production = pas de modifs locales)
git reset --hard origin/main

# Vérifier qu'on est à jour
git log --oneline -5

# Continuer avec rebuild Docker
docker-compose -f docker-compose.prod.yml up -d --build
```

**Solution 2 : Stash et ré-appliquer (si modifications intentionnelles)**

```bash
# Mettre de côté les changements locaux
git stash save "local_changes_$(date +%Y%m%d_%H%M%S)"

# Pull la nouvelle version
git pull origin main

# Ré-appliquer vos changements (peut causer conflits)
git stash pop

# Si conflits, résoudre manuellement :
git status  # Voir fichiers en conflit
nano <fichier_en_conflit>  # Résoudre conflits (<<<<< HEAD ... =====)
git add <fichier_résolu>
git commit -m "fix: merge conflicts"

# Puis rebuild
docker-compose -f docker-compose.prod.yml up -d --build
```

**Solution 3 : Voir les différences avant décision**

```bash
# Voir les fichiers modifiés localement
git status

# Voir le détail des changements
git diff config/score_registry.json
git diff services/ml/model_registry.py

# Si changements sans importance → Solution 1 (reset hard)
# Si changements critiques → Solution 2 (stash)
```

**📌 Bonnes Pratiques Production :**

1. ✅ **Ubuntu NUC = Serveur LECTURE SEULE** (pas de modifs locales)
2. ✅ **Windows = Machine DEV** (commit/push depuis Windows uniquement)
3. ✅ **Toujours `git reset --hard origin/main`** avant rebuild Docker sur NUC
4. ✅ **Logs changements locaux** avant reset (commande `git diff > /tmp/...`)

### Changements data/ ou logs/ uniquement

Si vous modifiez uniquement `data/` ou `logs/` (pas de code Python) :

```bash
# Simple restart (pas de rebuild)
docker-compose -f docker-compose.prod.yml restart
```

---

## Maintenance & Automatisation

### Mise à Jour Quotidienne Cache Prix (RECOMMANDÉ ✅)

**Problème :** Le cache prix historiques vieillit (BTC à J-1, ETH à J-1, etc.). Sans mise à jour, les métriques Risk Dashboard deviennent obsolètes.

**Solution :** Tâche cron pour télécharger **uniquement les 7 derniers jours** (rapide, 2-5 min).

#### Installation Cron Job

```bash
# Sur NUC Ubuntu
crontab -e

# Ajouter cette ligne (mise à jour tous les jours à 2h du matin)
0 2 * * * cd ~/smartfolio && docker exec smartfolio-api python scripts/download_historical_data.py --days 7 --from-portfolio >> ~/smartfolio/logs/price_update.log 2>&1
```

**Explication :**
- `0 2 * * *` : Tous les jours à 2h00 (quand APIs peu chargées)
- `--days 7` : Seulement 7 derniers jours (rapide, évite rate limits)
- `--from-portfolio` : Uniquement assets de votre portfolio (pas tous les 127 assets)
- `>> logs/price_update.log` : Log des succès/échecs

#### Vérifier Cron Actif

```bash
# Lister tâches cron
crontab -l

# Voir logs mise à jour
tail -f ~/smartfolio/logs/price_update.log

# Tester manuellement (sans attendre 2h du matin)
cd ~/smartfolio
docker exec smartfolio-api python scripts/download_historical_data.py --days 7 --from-portfolio
```

**Résultat attendu :**

```text
✅ Téléchargé 7 points pour BTC
✅ Téléchargé 7 points pour ETH
✅ Téléchargé 7 points pour SOL
...
✅ Mis à jour 25/25 symboles
```

#### Alternative : Mise à Jour Hebdomadaire Complète

Si vous voulez **tous les 127 assets** (pas juste portfolio), utilisez cette tâche hebdomadaire :

```bash
# Cron tous les lundis à 3h du matin (plus long, 15-30 min)
0 3 * * 1 cd ~/smartfolio && docker exec smartfolio-api python scripts/download_historical_data.py --days 30 --all >> ~/smartfolio/logs/price_update_weekly.log 2>&1
```

**Note :** `--days 30` (pas 3000) car on fusionne avec cache existant, pas besoin de tout re-télécharger.

### Backup Automatique Data (OPTIONNEL)

**Sauvegarder régulièrement** : `data/users/`, `data/portfolio_history.json`, `data/price_history/`

```bash
# Cron backup hebdomadaire (tous les dimanches à 4h)
0 4 * * 0 tar -czf ~/backups/smartfolio_$(date +\%Y\%m\%d).tar.gz -C ~/smartfolio data/ >> ~/smartfolio/logs/backup.log 2>&1

# Créer dossier backups d'abord
mkdir -p ~/backups
```

**Nettoyage backups anciens (garder 4 semaines) :**

```bash
# Cron cleanup backups (tous les lundis à 5h)
0 5 * * 1 find ~/backups/ -name "smartfolio_*.tar.gz" -mtime +28 -delete
```

### Monitoring Healthcheck

**Vérifier que Docker est toujours "healthy"** et redémarrer si besoin :

```bash
# Cron check health toutes les heures
0 * * * * docker ps --filter "name=smartfolio-api" --filter "health=unhealthy" --format "{{.Names}}" | xargs -r docker restart >> ~/smartfolio/logs/healthcheck.log 2>&1
```

**Explication :**
- Vérifie si `smartfolio-api` est "unhealthy"
- Si oui, redémarre automatiquement le container
- Log dans `healthcheck.log`

---

## Commandes Utiles

### Gestion conteneurs

```bash
# Voir conteneurs actifs
docker ps

# Voir TOUS les conteneurs (actifs + arrêtés)
docker ps -a

# Arrêter le stack
docker-compose -f docker-compose.prod.yml down

# Arrêter ET supprimer volumes (⚠️ perte données Redis)
docker-compose -f docker-compose.prod.yml down -v

# Redémarrer un service spécifique
docker-compose -f docker-compose.prod.yml restart smartfolio
docker-compose -f docker-compose.prod.yml restart redis
```

### Logs & Debug

```bash
# Logs temps réel (tous services)
docker-compose -f docker-compose.prod.yml logs -f

# Logs temps réel (service spécifique)
docker-compose -f docker-compose.prod.yml logs -f smartfolio
docker-compose -f docker-compose.prod.yml logs -f redis

# Logs récents (50 dernières lignes)
docker-compose -f docker-compose.prod.yml logs --tail 50

# Logs depuis timestamp
docker-compose -f docker-compose.prod.yml logs --since 2024-01-01T10:00:00
```

### Accès shell conteneur

```bash
# Shell interactif dans conteneur API
docker exec -it smartfolio-api bash

# Exemples commandes dans conteneur :
ls /app/data/users/           # Voir données users
cat /app/logs/app.log         # Lire logs
python -m pytest tests/       # Lancer tests
exit                          # Sortir

# Shell Redis (redis-cli)
docker exec -it smartfolio-redis redis-cli
# Commandes Redis :
ping               # Doit retourner PONG
keys *             # Voir toutes les clés
get <key>          # Lire valeur
exit               # Sortir
```

### Nettoyage Docker

```bash
# Supprimer images inutilisées (libérer espace)
docker image prune -a

# Supprimer volumes orphelins
docker volume prune

# Nettoyage complet (⚠️ supprime TOUT ce qui n'est pas actif)
docker system prune -a --volumes
```

### Monitoring ressources

```bash
# Utilisation CPU/RAM en temps réel
docker stats

# Sortie :
# CONTAINER ID   NAME               CPU %   MEM USAGE / LIMIT   MEM %   NET I/O
# abc123def456   smartfolio-api     15%     1.2GB / 16GB        7.5%    2MB / 1MB
# 789ghi012jkl   smartfolio-redis   2%      50MB / 16GB         0.3%    500KB / 300KB
```

---

## Troubleshooting

### Problème : Conteneur "unhealthy"

**Symptômes :**
```bash
docker ps
# STATUS: Up 5 minutes (unhealthy)
```

**Solution :**
```bash
# Voir logs détaillés
docker-compose -f docker-compose.prod.yml logs smartfolio

# Erreurs communes :
# - "Redis connection refused" → Vérifier Redis actif
# - "Port 8080 already in use" → Changer PORT dans .env
# - "Playwright browser not found" → Rebuild image
```

### Problème : Redis connection refused

**Symptômes :**
```
ConnectionError: Error 111 connecting to redis:6379. Connection refused.
```

**Solution :**
```bash
# Vérifier Redis actif
docker ps | grep redis

# Si absent, redémarrer
docker-compose -f docker-compose.prod.yml up -d redis

# Vérifier healthcheck Redis
docker inspect smartfolio-redis | grep Health -A 10
```

### Problème : Port 8080 déjà utilisé

**Symptômes :**
```
Error starting userland proxy: listen tcp 0.0.0.0:8080: bind: address already in use
```

**Solution :**
```bash
# Trouver processus utilisant port 8080
sudo lsof -i :8080
# ou
sudo netstat -tlnp | grep 8080

# Tuer processus (remplacer <PID>)
sudo kill <PID>

# OU changer port dans .env
nano .env
# PORT=8081
docker-compose -f docker-compose.prod.yml up -d
```

### Problème : Données utilisateurs introuvables

**Symptômes :**
```
FileNotFoundError: data/users/demo/secrets.json
```

**Solution :**
```bash
# Vérifier ownership volumes
ls -la data/users/

# Si permissions incorrectes :
sudo chown -R $USER:$USER data/

# Vérifier montage volumes Docker
docker inspect smartfolio-api | grep -A 20 Mounts
```

### Problème : Image build échoue

**Symptômes :**
```
ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully
```

**Solution :**
```bash
# Vérifier requirements.txt existe
cat requirements.txt

# Rebuild sans cache
docker-compose -f docker-compose.prod.yml build --no-cache

# Si problème persiste, vérifier connexion internet NUC
ping pypi.org
```

### Problème : Logs "disk full"

**Solution :**
```bash
# Vérifier espace disque
df -h

# Purger logs Docker
docker system prune -a --volumes

# Logs applicatifs (auto-rotation 5MB x4)
ls -lh logs/
# Si trop gros, supprimer manuellement
rm logs/app.log.2 logs/app.log.3
```

---

## Ressources Additionnelles

### Documentation Projet

- **Architecture** : `docs/ARCHITECTURE.md`
- **Redis Setup** : `docs/REDIS_SETUP.md`
- **Multi-tenant** : `CLAUDE.md` (section Multi-Tenant)
- **Logging** : `docs/LOGGING.md`

### Docker Documentation

- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Docker Healthchecks](https://docs.docker.com/engine/reference/builder/#healthcheck)
- [Docker Volumes](https://docs.docker.com/storage/volumes/)

---

## Notes Performances

### Estimations NUC (i5-7260U, 16GB RAM)

| Métrique | Valeur | Commentaire |
|----------|--------|-------------|
| **Build initial** | 5-8 min | Première fois (télécharge images, compile deps) |
| **Rebuild après modif** | 30-60 sec | Docker cache layers (seul code Python recompile) |
| **Startup time** | 20-40 sec | Redis ~5s, API ~15-35s (Playwright init) |
| **RAM utilisée** | 1.5-2.5 GB | Redis ~50MB, API ~1-2GB, Docker engine ~300MB |
| **CPU idle** | 5-10% | Scheduler + warmers |
| **CPU peak** | 40-80% | ML inference + Playwright scraping |

### Optimisations Possibles

Si ressources limitées, décommenter dans `docker-compose.prod.yml` :

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'      # Max 2 cores
      memory: 4G       # Max 4GB RAM
```

---

**🎉 Migration terminée ! Votre SmartFolio tourne maintenant sur Linux avec Docker.**

Pour questions ou problèmes : ouvrir une issue GitHub ou consulter `CLAUDE.md`.

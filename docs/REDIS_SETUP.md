# Redis Setup Guide

> Installation et configuration de Redis pour smartfolio
> Date: Oct 2025

## Pourquoi Redis ?

Redis est une base de données en mémoire (RAM) ultra-rapide utilisée pour:

- **Cache haute performance** - Métriques calculées, prix, données ML
- **Alertes persistantes** - Survivent aux restarts serveur
- **Streaming temps réel** - WebSocket push via Redis Streams
- **Sessions partagées** - Multi-instances en production

**Performance:** Opérations en **microseccondes** vs millisecondes (DB classique)

---

## Installation

### Option 1: WSL2 (Recommandé pour Windows)

**Prérequis:** WSL2 avec Ubuntu installé

```bash
# 1. Ouvrir terminal WSL2/Ubuntu
wsl

# 2. Installer Redis
sudo apt update
sudo apt install -y redis-server

# 3. Configurer Redis pour Windows
sudo sed -i 's/^bind 127.0.0.1/bind 0.0.0.0/' /etc/redis/redis.conf

# 4. Démarrer Redis
sudo service redis-server start

# 5. Tester
redis-cli ping
# Doit afficher: PONG
```

**Démarrage automatique (optionnel):**
```bash
# Ajouter au ~/.bashrc pour démarrage auto
echo "sudo service redis-server start" >> ~/.bashrc
```

### Option 2: Windows Native

**Télécharger:**
- https://github.com/tporadowski/redis/releases
- Télécharger `Redis-x64-5.0.14.1.zip`
- Extraire dans `C:\Redis`

**Lancer:**
```powershell
cd C:\Redis
.\redis-server.exe
```

**Installer comme service Windows:**
```powershell
.\redis-server.exe --service-install
.\redis-server.exe --service-start
```

### Option 3: Linux/macOS

```bash
# Ubuntu/Debian
sudo apt install redis-server

# macOS (Homebrew)
brew install redis
brew services start redis

# Fedora/RHEL
sudo dnf install redis
sudo systemctl start redis
```

---

## Configuration Projet

### 1. Variable d'Environnement

Ajouter dans `.env`:
```bash
# Redis configuration (for caching, alerting, streaming)
REDIS_URL=redis://localhost:6379/0
```

**Format URL Redis:**
```
redis://host:port/db
redis://localhost:6379/0    # DB 0 (default)
redis://localhost:6379/1    # DB 1 (alternate)
rediss://host:6379/0        # SSL/TLS (production)
```

### 2. Vérification Connexion

**Depuis WSL/Linux:**
```bash
redis-cli ping
# PONG

# Infos serveur
redis-cli INFO server | grep uptime
```

**Depuis Python:**
```bash
.venv\Scripts\python.exe -c "import redis; r = redis.Redis(host='localhost', port=6379); print('Connected:', r.ping())"
# Connected: True
```

**Depuis Windows:**
```powershell
Test-NetConnection -ComputerName localhost -Port 6379
# TcpTestSucceeded: True
```

---

## Utilisation dans le Projet

### Streams Redis Actifs

Le serveur FastAPI utilise Redis Streams pour le temps réel:

```python
# Streams enregistrés au démarrage (voir logs)
[OK] Registered consumer for stream: risk_events
[OK] Registered consumer for stream: alerts
[OK] Registered consumer for stream: market_data
[OK] Registered consumer for stream: portfolio_updates
```

### Services Utilisant Redis

**1. Alert Storage** (`services/alerts/alert_storage.py`)
```python
# Stockage persistant des alertes
alert_storage.save(alert)  # Persiste en Redis
active = alert_storage.get_active_alerts()  # Lecture depuis Redis
```

**2. RealtimeEngine** (`services/streaming/realtime_engine.py`)
```python
# Streaming temps réel
realtime_engine.broadcast_event("price_update", data)
# → Tous les clients WebSocket reçoivent l'update
```

**3. Health Monitor** (`services/monitoring/phase3_health_monitor.py`)
```python
# Cache métriques santé
health_monitor.store_metrics(metrics)  # Cache Redis partagé
```

**4. Cache Général** (`api/utils/cache.py`)
```python
from api.utils.cache import cache_get, cache_set

# Cache avec TTL
cache_set("portfolio_metrics", data, ttl=300)  # 5 min
cached = cache_get("portfolio_metrics")
```

---

## Tests et Debug

### Vérifier les Logs Serveur

**Rechercher succès Redis:**
```bash
grep "Redis Streams initialized successfully" logs/app.log
# Doit afficher: INFO services.streaming.realtime_engine: Redis Streams initialized successfully
```

**Rechercher erreurs Redis:**
```bash
grep -i "redis.*error\|failed to initialize redis" logs/app.log
# Ne doit rien afficher si OK
```

### Monitorer Redis en Direct

**Voir les commandes:**
```bash
# Dans WSL
redis-cli MONITOR
# Affiche toutes les commandes Redis en temps réel
```

**Voir les streams:**
```bash
redis-cli XINFO STREAM risk_events
redis-cli XINFO STREAM alerts
```

**Voir les clés:**
```bash
redis-cli KEYS "*"
```

### Tester Manuellement

**Écrire/Lire:**
```bash
redis-cli SET test_key "Hello Redis"
redis-cli GET test_key
# Hello Redis

redis-cli DEL test_key
```

**Pub/Sub:**
```bash
# Terminal 1 (subscriber)
redis-cli SUBSCRIBE test_channel

# Terminal 2 (publisher)
redis-cli PUBLISH test_channel "Test message"
```

---

## Troubleshooting

### Problème: "Connection refused" (Error 22)

**Cause:** Redis n'écoute pas sur l'interface accessible depuis Windows

**Solution:**
```bash
# Dans WSL
sudo sed -i 's/^bind 127.0.0.1/bind 0.0.0.0/' /etc/redis/redis.conf
sudo service redis-server restart
```

### Problème: "degraded mode" dans les logs

**Cause:** REDIS_URL non configuré ou Redis non accessible

**Solution:**
1. Vérifier `.env` contient `REDIS_URL=redis://localhost:6379/0`
2. Tester connexion: `redis-cli ping`
3. Restart serveur FastAPI

### Problème: Redis ne démarre pas au boot

**Solution WSL:**
```bash
# Ajouter au ~/.bashrc
echo "sudo service redis-server start" >> ~/.bashrc
```

**Solution Windows native:**
```powershell
# Installer comme service
cd C:\Redis
.\redis-server.exe --service-install
.\redis-server.exe --service-start
```

### Problème: Redis consomme trop de mémoire

**Configuration limits:**
```bash
# Dans WSL, éditer /etc/redis/redis.conf
sudo nano /etc/redis/redis.conf

# Ajouter/modifier:
maxmemory 256mb
maxmemory-policy allkeys-lru  # Éviction LRU quand plein

# Restart
sudo service redis-server restart
```

---

## Sécurité

### Développement Local

Configuration actuelle (bind 0.0.0.0) est OK pour dev local, mais **ne PAS exposer sur Internet**.

### Production

**1. Activer authentification:**
```bash
# Dans redis.conf
requirepass votre_mot_de_passe_fort
```

**2. Utiliser TLS:**
```bash
REDIS_URL=rediss://password@host:6379/0
```

**3. Bind spécifique:**
```bash
# Seulement localhost
bind 127.0.0.1 ::1
```

**4. Firewall:**
```bash
# Bloquer port 6379 depuis l'extérieur
sudo ufw deny 6379
```

---

## Performance

### Benchmarks Typiques

- **SET/GET:** ~100,000 ops/sec
- **Latence:** < 1ms (local)
- **Mémoire:** ~10-50 MB (projet actuel)
- **CPU:** < 1% (idle), ~5% (actif)

### Monitoring Production

```bash
# Stats en temps réel
redis-cli --stat

# Infos détaillées
redis-cli INFO all
```

---

## Ressources

- Documentation officielle: https://redis.io/docs/
- Redis commands: https://redis.io/commands/
- Redis Streams: https://redis.io/docs/data-types/streams/
- Python client: https://redis-py.readthedocs.io/

---

**Configuration actuelle:** Redis fonctionne en mode développement avec 4 streams actifs (risk_events, alerts, market_data, portfolio_updates) et supporte le cache distribué pour le système d'alertes et le streaming temps réel.

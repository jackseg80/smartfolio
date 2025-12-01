# 🐳 SmartFolio - Docker Deployment Guide

Guide de déploiement et de maintenance de SmartFolio en production avec Docker.
Ce document est la source de vérité pour le déploiement.

**Public cible :** Développeurs, Administrateurs système.
**Environnement cible :** Serveur Linux (ex: Ubuntu 24.04) avec Docker.

---

## 📋 Table des Matières

1. [Architecture Cible](#architecture-cible)
2. [Déploiement Rapide (TL;DR)](#-déploiement-rapide-tldr)
3. [Installation & Configuration](#-installation--configuration)
4. [Déploiement Automatisé (deploy.sh)](#-déploiement-automatisé-deploysh)
5. [Commandes Manuelles](#-commandes-manuelles)
6. [Workflow de Mise à Jour](#-workflow-de-mise-à-jour)
7. [Maintenance et Dépannage](#-maintenance-et-dépannage)
8. [Backup & Restore](#-backup--restore)

---

## Architecture Cible

L'application est conçue pour tourner dans un environnement conteneurisé géré par Docker Compose.

```
┌─────────────────────────────────────────┐
│            Serveur Linux (Hôte)         │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │     Docker Compose Stack          │  │
│  │                                   │  │
│  │  ┌─────────────┐  ┌────────────┐ │  │
│  │  │   Redis     │  │ SmartFolio │ │  │
│  │  │   (Cache)   │←→│    API     │ │  │
│  │  └─────────────┘  └────────────┘ │  │
│  │         ↓               ↓         │  │
│  │  ┌──────────────────────────┐    │  │
│  │  │   Volume Nommé           │    │  │
│  │  │   redis_data             │    │  │
│  │  └──────────────────────────┘    │  │
│  │         ↓               ↓         │  │
│  │  ┌──────────┐    ┌──────────┐   │  │
│  │  │ ./data/  │    │ ./logs/  │   │  │
│  │  │ (Bind)   │    │ (Bind)   │   │  │
│  │  └──────────┘    └──────────┘   │  │
│  └───────────────────────────────────┘  │
│                 ↑                        │
│          Port 8080 (LAN)                 │
└─────────────────────────────────────────┘
```
**Composants clés :**
- **`docker-compose.yml`**: Fichier principal décrivant la stack de services (API, Redis).
- **`Dockerfile.prod`**: Instructions pour construire l'image Docker de production.
- **`.env`**: Fichier de configuration pour les secrets et variables d'environnement.
- **Volumes**:
    - `redis_data` (volume nommé) : Pour la persistance des données Redis.
    - `./data` et `./logs` (bind mounts) : Pour que les données et logs soient directement accessibles sur le serveur hôte.

---

## 🚀 Déploiement Rapide (TL;DR)

Sur le serveur de production :
```bash
# 1. Cloner le projet (si pas déjà fait)
git clone https://github.com/your-org/smartfolio.git
cd smartfolio

# 2. Créer le fichier .env
cp .env.example .env
# Éditer .env et configurer les clés API et tokens.

# 3. Lancer le déploiement automatisé
./deploy.sh

# 4. Vérifier l'état
docker ps
curl http://localhost:8080/docs
```

---

## 🔧 Installation & Configuration

### Prérequis Serveur
- Docker Engine 24.0+
- Docker Compose v2.20+
- Git
- Serveur Linux (Ubuntu 24.04.2 LTS recommandé)
- Minimum 4GB RAM, 20GB disque
- Port 8080 (ou celui configuré) disponible.

### Configuration Initiale

1.  **Cloner le projet**
    ```bash
    git clone <votre-repo-url> /opt/smartfolio
    cd /opt/smartfolio
    ```

2.  **Créer le fichier `.env`**
    Copiez le template et modifiez-le.
    ```bash
    cp .env.example .env
    nano .env
    ```

3.  **Variables Critiques à Modifier dans `.env`**
    ```ini
    # Mettre false en production pour la sécurité et performance
    DEBUG=false
    ENVIRONMENT=production

    # CHANGEZ CES VALEURS ! Utilisez des chaînes longues et aléatoires.
    DEBUG_TOKEN=your-prod-token-xyz123
    ADMIN_KEY=your-admin-key-abc456

    # Clés API pour les services externes
    COINGECKO_API_KEY=your_key_here
    COINTRACKING_API_KEY=your_key_here
    FRED_API_KEY=your_key_here
    ```
    **Générer des tokens sécurisés :**
    ```bash
    # Génère une chaîne de 64 caractères hexadécimaux
    openssl rand -hex 32
    ```

4.  **Transférer l'historique des données (TRÈS IMPORTANT)**
    Le cache de prix (`data/price_history/`) est crucial pour le calcul des métriques de risque. Les APIs publiques ne fournissent qu'un historique limité (ex: 365 jours). Vous devez transférer le cache complet depuis votre machine de développement.

    **Sur votre machine de dev (Windows/Linux/Mac) :**
    ```bash
    # Remplacez <user> et <ip_serveur>
    scp -r data/price_history/*.json <user>@<ip_serveur>:/opt/smartfolio/data/price_history/
    ```
    **Vérification sur le serveur :**
    ```bash
    # Doit retourner un nombre élevé de fichiers (ex: 127)
    ls /opt/smartfolio/data/price_history/ | wc -l
    ```
    Sans cette étape, les métriques de risque seront incorrectes.

---

## 🚀 Déploiement Automatisé (deploy.sh)

Le script `deploy.sh` est la méthode **recommandée** pour tous les déploiements et mises à jour en production. Il automatise le processus pour être rapide, sûr et répétable.

### Usage
```bash
# Déploiement standard (rebuild complet de l'image)
./deploy.sh

# Déploiement rapide (redémarre les conteneurs sans reconstruire l'image)
./deploy.sh --skip-build

# Déploiement forcé (écrase les changements locaux sur le serveur sans demander)
./deploy.sh --force

# Afficher l'aide
./deploy.sh --help
```

### Processus du script
Le script exécute les étapes suivantes :
1.  **Vérification des changements locaux** : Si des modifications existent sur le serveur, il propose de les sauvegarder dans un patch avant de les écraser.
2.  **Pull depuis GitHub** : Récupère la dernière version du code.
3.  **Vérification du cache de prix** : Vous alerte si le cache semble incomplet.
4.  **Reconstruction & Redémarrage Docker** : Reconstruit l'image de l'API et relance la stack.
5.  **Health Check** : Attend que les services soient opérationnels et confirme leur état.

---

## ⚙️ Commandes Manuelles

Utilisez ces commandes pour une gestion plus fine ou pour le débogage.

### Lancement et Arrêt
```bash
# Construire les images et démarrer les services en arrière-plan
docker compose up -d --build

# Démarrer les services sans reconstruire
docker compose up -d

# Arrêter les services
docker compose down

# Arrêter et supprimer les volumes (ATTENTION: perte de données Redis)
docker compose down -v
```

### Consultation des logs
```bash
# Voir les logs de tous les services en temps réel
docker compose logs -f

# Voir les logs d'un service spécifique (ex: l'API)
docker compose logs -f smartfolio

# Afficher les 100 dernières lignes et quitter
docker compose logs --tail=100 smartfolio
```

### Exécuter des commandes dans un conteneur
```bash
# Ouvrir un shell bash dans le conteneur de l'API
docker compose exec smartfolio bash

# Lancer les tests unitaires à l'intérieur du conteneur
docker compose exec smartfolio python -m pytest tests/unit

# Se connecter à l'interface de commande de Redis
docker compose exec redis redis-cli
```

---

## 🔄 Workflow de Mise à Jour

Le workflow de développement et de mise en production est simple :

1.  **Sur votre machine de développement :**
    Faites vos modifications, commitez et pushez sur la branche `main`.
    ```bash
    git add .
    git commit -m "feat: ma nouvelle fonctionnalité"
    git push origin main
    ```

2.  **Sur le serveur de production :**
    Exécutez simplement le script de déploiement.
    ```bash
    # Se connecter au serveur
    ssh <user>@<ip_serveur>
    cd /opt/smartfolio

    # Lancer le script
    ./deploy.sh
    ```

Le script s'occupe de tout.

**Quand utiliser `--skip-build` ?**
- **Rebuild complet (défaut)** : Obligatoire si vous modifiez `Dockerfile.prod` ou `requirements.txt`.
- **Restart rapide (`--skip-build`)** : Suffisant si vous ne modifiez que du code Python (`.py`), des fichiers statiques (HTML/JS) ou de la configuration (`.json`). Le redémarrage ne prend que quelques secondes.

---

## 🔧 Maintenance et Dépannage

### Le service ne démarre pas ou est "unhealthy"
1.  **Consultez les logs** : C'est la première source d'information.
    ```bash
    docker compose logs smartfolio
    ```
2.  **Vérifiez la configuration `.env`** : Une clé API manquante ou un token malformé peut empêcher le démarrage.
3.  **Vérifiez qu'un autre service n'utilise pas le port** :
    ```bash
    sudo lsof -i :8080
    ```

### Nettoyage de Docker
Pour libérer de l'espace disque, vous pouvez nettoyer les ressources Docker non utilisées.
```bash
# Supprimer les conteneurs arrêtés, les réseaux inutilisés et les images pendantes
docker system prune

# Nettoyage plus agressif (supprime aussi les volumes non utilisés)
docker system prune --volumes
```

---

## 💾 Backup & Restore

### Stratégie de Backup
Il est crucial de sauvegarder régulièrement :
1.  Le répertoire `data/` qui contient toutes les données utilisateurs, configurations et l'historique des prix.
2.  Le volume `redis_data` qui contient le cache de session.
3.  Le fichier `.env` qui contient vos secrets.

### Exemple de script de backup
```bash
#!/bin/bash
BACKUP_DIR="/backup/smartfolio/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 1. Sauvegarder le répertoire data/
cp -r /opt/smartfolio/data/ "$BACKUP_DIR/data"

# 2. Sauvegarder les données Redis
docker compose exec redis redis-cli BGSAVE
sleep 5 # Laisser le temps à Redis de sauvegarder sur le disque
docker cp smartfolio-redis:/data/dump.rdb "$BACKUP_DIR/redis_dump.rdb"

# 3. Sauvegarder le fichier .env
cp /opt/smartfolio/.env "$BACKUP_DIR/.env.backup"

# 4. Compresser l'archive
tar -czf "$BACKUP_DIR.tar.gz" -C "/backup/smartfolio" "$(basename $BACKUP_DIR)"
rm -rf "$BACKUP_DIR"

echo "✅ Backup créé: $BACKUP_DIR.tar.gz"
```

### Restauration
La restauration implique de stopper les services, de remplacer les données par celles du backup, et de redémarrer. Assurez-vous de bien comprendre le processus avant de le tenter.
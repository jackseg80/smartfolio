# Guide Configuration Utilisateur - SmartFolio

## Système de Configuration (Nov 2025)

SmartFolio utilise **2 fichiers distincts** par utilisateur pour séparer les préférences UI et les secrets API.

### Structure Fichiers

```
data/users/{user_id}/
├── config.json       ← Préférences UI uniquement (NON sensible)
└── secrets.json      ← Clés API sensibles (NE PAS COMMITTER)
```

### config.json - Préférences UI

**Contenu** : Paramètres d'interface, pas de clés API

```json
{
  "data_source": "cointracking_api",
  "api_base_url": "http://localhost:8080",
  "display_currency": "USD",
  "min_usd_threshold": 1.0,
  "csv_glob": "csv/*.csv",
  "csv_selected_file": null,
  "pricing": "local",
  "refresh_interval": 5,
  "enable_coingecko_classification": true,
  "enable_portfolio_snapshots": true,
  "enable_performance_tracking": true,
  "theme": "auto",
  "debug_mode": false
}
```

### secrets.json - Clés API Sensibles

**Contenu** : Toutes les clés API et secrets

```json
{
  "dev_mode": {
    "enabled": false,
    "mock_data": false
  },
  "coingecko": {
    "api_key": "CG-xxxxx",
    "pro": false
  },
  "cointracking": {
    "api_key": "xxxxx",
    "api_secret": "xxxxx"
  },
  "binance": {
    "api_key": "",
    "api_secret": "",
    "testnet": true
  },
  "kraken": {
    "api_key": "",
    "api_secret": ""
  },
  "exchanges": {
    "default": "binance"
  }
}
```

## Migration depuis l'Ancien Système

Si vous avez des clés API dans `config.json` (ancien système), migrez-les vers `secrets.json` :

### Méthode Automatique

```bash
# Windows
python scripts/consolidate_user_config.py jack

# Linux/Docker
python3 scripts/consolidate_user_config.py jack
```

**Actions du script** :
1. Sauvegarde `config.json` → `config.json.bak`
2. Supprime les clés sensibles de `config.json`
3. Vérifie que `secrets.json` contient les clés

### Méthode Manuelle

1. Ouvrir `config.json` et `secrets.json`
2. Copier les clés API de `config.json` vers `secrets.json`
3. Supprimer ces lignes de `config.json` :
   - `cointracking_api_key`
   - `cointracking_api_secret`
   - `coingecko_api_key`
   - `fred_api_key`

## Déploiement Docker/Production

### Problème Clés Manquantes Après Déploiement

**Symptôme** : "Clés API non trouvées" après `./deploy.sh`

**Cause** : Le fichier `secrets.json` existe en local mais pas dans le container Docker

**Solution** : Copier `secrets.json` sur le serveur avant déploiement

```bash
# Depuis Windows vers serveur Ubuntu (robot2)
scp data/users/jack/secrets.json jack@robot2:~/smartfolio/data/users/jack/

# Puis déployer
ssh jack@robot2
cd ~/smartfolio
./deploy.sh --force
```

### Volumes Docker Montés

Le `docker-compose.yml` monte le dossier `data/` :

```yaml
volumes:
  - ./data:/app/data:rw    # User data (config + secrets)
  - ./logs:/app/logs:rw    # Application logs
```

**Important** : `secrets.json` doit exister sur le serveur AVANT de démarrer Docker !

## Vérification Configuration

### Local (Windows)

```bash
# Vérifier config.json (pas de clés sensibles)
cat data/users/jack/config.json

# Vérifier secrets.json (clés présentes)
cat data/users/jack/secrets.json
```

### Production (Docker/Ubuntu)

```bash
# SSH sur serveur
ssh jack@robot2

# Vérifier secrets.json existe
ls -lh ~/smartfolio/data/users/jack/secrets.json

# Vérifier contenu (masqué)
cat ~/smartfolio/data/users/jack/secrets.json | head -20
```

### Test API Endpoints

```bash
# Test CoinGecko key chargée
curl "http://localhost:8080/api/debug/coingecko-test?user_id=jack"

# Test CoinTracking credentials
curl "http://localhost:8080/api/cointracking/import?user_id=jack"
```

## Sécurité

### ⚠️ NE JAMAIS COMMITTER

```bash
# .gitignore contient déjà:
data/users/*/secrets.json
data/users/*/config.json
```

### ✅ Bonnes Pratiques

1. **Secrets** : Uniquement dans `secrets.json`
2. **Backup** : Garder copie locale sécurisée de `secrets.json`
3. **Docker** : Copier `secrets.json` sur serveur avant déploiement
4. **Rotation** : Changer les clés API régulièrement

## Troubleshooting

### "CoinGecko API key not found"

1. Vérifier `secrets.json` existe : `ls data/users/jack/secrets.json`
2. Vérifier contenu : `"coingecko": {"api_key": "CG-xxxxx"}`
3. Redémarrer serveur : `python -m uvicorn api.main:app --port 8080` (ou Docker restart)

### "CoinTracking credentials not configured"

1. Vérifier `secrets.json` : `"cointracking": {"api_key": "...", "api_secret": "..."}`
2. Vérifier format : `api_secret` doit être présent (pas `secret`)

### Docker ne trouve pas les clés

1. Vérifier volume monté : `docker inspect smartfolio-api | grep Mounts -A 20`
2. Vérifier fichier dans container : `docker exec smartfolio-api ls -lh /app/data/users/jack/`
3. Copier si manquant : `docker cp data/users/jack/secrets.json smartfolio-api:/app/data/users/jack/`

---

**Last Updated:** December 2025
**Related Docs:**
- [SAXO_REDIRECT_FIX.md](SAXO_REDIRECT_FIX.md) - Fix Saxo OAuth2 errors
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture

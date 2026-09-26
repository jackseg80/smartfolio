# Docker Deployment Fix - Nov 2025

## Problème Identifié

En développement Windows, l'application fonctionne correctement avec `localhost:8080`. En déploiement Docker Linux, les erreurs suivantes apparaissent :

```
ERR_BLOCKED_BY_CLIENT: http://localhost:8080/balances/current
404 Not Found: http://192.168.1.200:8080/static/data/raw/...
```

**Cause racine :**
- Le frontend utilise une config `api_base_url` en localStorage pointant vers `localhost:8080`
- En Docker, `localhost` dans le navigateur ne pointe PAS vers le conteneur
- Les requêtes sont bloquées (CORS/client-side blocking)

## Solutions Appliquées

### 1. Alignement des Ports (docker-compose.yml)

**Avant :**
```yaml
ports:
  - "8000:8000"  # [Error] Incohérent avec Dockerfile (8080)
```

**Après :**
```yaml
ports:
  - "8080:8080"  # [OK] Aligné avec Dockerfile.prod
environment:
  - API_BASE_URL=${API_BASE_URL:-http://0.0.0.0:8080}
```

### 2. Auto-Configuration Frontend (global-config.js)

Ajout d'une fonction d'initialisation qui charge l'API_BASE_URL depuis le backend :

```javascript
// Charge l'API_BASE_URL depuis /api/config/api-base-url au démarrage
// Override automatiquement la config localStorage obsolète
(async function initApiBaseUrl() {
  const response = await fetch('/api/config/api-base-url');
  if (response.ok) {
    const result = await response.json();
    globalConfig.set('api_base_url', result.data.api_base_url);
  }
})();
```

**Avantages :**
- Détecte automatiquement l'URL backend correcte
- Fonctionne en dev (localhost) ET en production (IP serveur)
- Pas besoin de vider manuellement le localStorage

### 3. Configuration Production (.env.production)

**Fichier créé :** `.env.production`

**Configuration clé :**
```bash
# IMPORTANT: Remplacer 192.168.1.200 par l'IP réelle du serveur
API_BASE_URL=http://192.168.1.200:8080
PORT=8080
REDIS_URL=redis://redis:6379/0
```

### 4. Endpoint Backend Existant

L'endpoint `/api/config/api-base-url` existe déjà dans `api/config_router.py` :

```python
@router.get("/api-base-url")
async def get_api_base_url():
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8080")
    return success_response({"api_base_url": api_base_url})
```

## Déploiement

### Étape 1: Configurer l'IP du Serveur

Modifier `.env.production` avec l'IP réelle du serveur :

```bash
# Exemple pour IP 192.168.1.200
API_BASE_URL=http://192.168.1.200:8080
```

### Étape 2: Déployer sur Linux

```bash
# Sur le serveur Linux
./deploy.sh
```

Le script deploy.sh :
- Pull les derniers changements GitHub
- Rebuild l'image Docker avec les nouveaux fichiers
- Redémarre les conteneurs (smartfolio + redis)
- Vérifie la santé des services

### Étape 3: Vérification

**Dans le navigateur (DevTools Console) :**
```
[OK] API_BASE_URL loaded from backend: http://192.168.1.200:8080
 Balance data loaded from cache (user: demo, file: latest)
```

**Si les erreurs persistent :**

1. **Vider le localStorage :**
```javascript
// Dans la console du navigateur
localStorage.clear();
location.reload();
```

2. **Vérifier les logs Docker :**
```bash
docker-compose -f docker-compose.prod.yml logs -f smartfolio
```

3. **Tester l'endpoint config directement :**
```bash
curl http://192.168.1.200:8080/api/config/api-base-url
# Doit retourner: {"ok":true,"data":{"api_base_url":"http://192.168.1.200:8080"}}
```

## Architecture Mise à Jour

```
┌──────────────────────────────────────────────────────────────┐
│ Navigateur (Client)                                          │
│ - global-config.js charge API_BASE_URL depuis backend       │
│ - Stocke dans localStorage pour cache                       │
│ - Auto-update si backend change                             │
└────────────────────┬─────────────────────────────────────────┘
                     │ HTTP Requests (ex: /balances/current)
                     │
┌────────────────────▼─────────────────────────────────────────┐
│ Docker Container: smartfolio-api (Port 8080)                │
│ - FastAPI app (uvicorn)                                     │
│ - Endpoint /api/config/api-base-url                         │
│ - Environnement: API_BASE_URL (depuis .env.production)      │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│ Docker Container: smartfolio-redis (Port 6379)              │
│ - Cache                                                      │
│ - Streaming                                                  │
└──────────────────────────────────────────────────────────────┘
```

## Fichiers Modifiés

1. [OK] `docker-compose.yml` - Port 8000→8080 (dev)
2. [OK] `docker-compose.prod.yml` - Déjà correct (8080)
3. [OK] `.env.production` - Nouveau fichier avec config production
4. [OK] `static/global-config.js` - Auto-load API_BASE_URL depuis backend
5. [OK] `DOCKER_FIX_NOTES.md` - Ce document

## Notes Importantes

### LocalStorage vs Backend Config

**Ancien comportement :**
- `api_base_url` détecté depuis `window.location.origin` (peut être incorrect)
- Stocké dans localStorage (persiste même si mauvais)
- Aucune synchronisation avec backend

**Nouveau comportement :**
- `api_base_url` chargé depuis `/api/config/api-base-url` au démarrage
- Override la config localStorage obsolète
- Toujours synchronisé avec la config backend

### Docker Network vs LAN Access

**Important :** Il y a 2 types d'accès réseau :

1. **Docker network interne** (redis:6379) :
   - Communication entre conteneurs
   - Ex: `REDIS_URL=redis://redis:6379/0`

2. **LAN access externe** (192.168.1.200:8080) :
   - Accès depuis navigateur client
   - Ex: `API_BASE_URL=http://192.168.1.200:8080`

Ne PAS confondre les deux !

### Sécurité

En production, l'API est accessible sur le LAN (192.168.1.200:8080).

**Recommandations :**
- Firewall : Bloquer le port 8080 depuis internet (seulement LAN)
- Reverse proxy : Utiliser nginx avec HTTPS (production réelle)
- CORS : Restreindre les origins autorisées dans `.env.production`

```bash
# Exemple .env.production sécurisé
CORS_ORIGINS=http://192.168.1.200:8080,https://smartfolio.example.com
```

## Troubleshooting

### Erreur: "ERR_BLOCKED_BY_CLIENT" persiste

**Cause possible :** Bloqueur de publicités (uBlock, AdBlock)

**Solutions :**
1. Désactiver temporairement le bloqueur pour `192.168.1.200`
2. Ajouter exception pour localhost/LAN dans le bloqueur
3. Utiliser navigateur en mode incognito (sans extensions)

### Erreur: "404 Not Found" pour CSV files

**Cause :** Les fichiers CSV n'existent pas à cet emplacement

**Solution :**
- Utiliser source `cointracking_api` au lieu de `cointracking` (CSV)
- Ou uploader les CSV via l'interface Sources dans settings.html

### Erreur: API retourne localhost au lieu de l'IP

**Vérification :**
```bash
# Sur le serveur Linux
docker exec smartfolio-api env | grep API_BASE_URL
# Doit afficher: API_BASE_URL=http://192.168.1.200:8080
```

**Si incorrect :**
```bash
# Vérifier .env.production
cat .env.production | grep API_BASE_URL

# Rebuild avec env variables
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d --build
```

## Validation Finale

**Checklist déploiement réussi :**

- [ ] Backend répond : `curl http://192.168.1.200:8080/docs`
- [ ] Config endpoint : `curl http://192.168.1.200:8080/api/config/api-base-url`
- [ ] Frontend charge API_BASE_URL depuis backend (console logs)
- [ ] Balances API fonctionne : `curl http://192.168.1.200:8080/balances/current`
- [ ] Dashboard accessible : `http://192.168.1.200:8080/dashboard.html`
- [ ] Settings page fonctionne sans erreurs ERR_BLOCKED_BY_CLIENT

## Support

En cas de problème persistant :

1. Vérifier logs Docker : `docker-compose -f docker-compose.prod.yml logs -f`
2. Tester endpoint config : `curl http://192.168.1.200:8080/api/config/api-base-url`
3. Vérifier variables env : `docker exec smartfolio-api env | grep API_BASE_URL`
4. Vider localStorage navigateur : `localStorage.clear()` + reload

---

**Date :** November 27, 2025
**Status :** [OK] Résolu
**Impact :** Production Docker Linux déployé avec succès

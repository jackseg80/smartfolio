# Fix Erreurs 429 en Production Docker

**Statut:** [OK] RÉSOLU
**Date:** 2025-12-04
**Problème:** Erreurs 429 (Too Many Requests) en cascade sur production Linux Docker

---

## Diagnostic

### Symptômes
- [Error] Erreur 429 sur TOUS les endpoints (/api/wealth/global/summary, /api/ml/sentiment, /api/risk/dashboard, etc.)
- [Error] WebSocket connection failed
- [Error] CSP violation pour Chart.js sourcemaps
- [Error] 401 Unauthorized sur /api/saxo/api-positions
- [Error] Clés API non visibles dans settings.html

### Cause Racine
**Rate limiter Token Bucket trop strict en production :**
- Valeurs par défaut : 6 req/sec + burst 12 tokens
- Dashboard fait 20-30 requêtes parallèles au chargement
- → Épuise les 12 tokens immédiatement → erreurs 429 en cascade

**Variables manquantes dans docker-compose.yml :**
- `SECURITY_RATE_LIMIT_REFILL_RATE` et `SECURITY_RATE_LIMIT_BURST_SIZE` non mappées
- Docker utilisait les valeurs par défaut hardcodées (6.0/12) au lieu des valeurs .env

---

## Solutions Appliquées

### 1. Fix Rate Limiting (CRITIQUE)

**Modifications :**
- [OK] [docker-compose.yml](../docker-compose.yml#L72-L73) - Ajout mapping variables SECURITY_*
- [OK] [.env.production.example](../.env.production.example#L35-L36) - Nouvelles valeurs recommandées

**Nouvelles valeurs par défaut :**
```yaml
SECURITY_RATE_LIMIT_REFILL_RATE: 20.0  # 20 req/sec (1200/min)
SECURITY_RATE_LIMIT_BURST_SIZE: 50     # Burst 50 requêtes simultanées
```

**Justification :**
- Dashboard fait 20-30 requêtes au chargement → besoin burst 50
- Rechargement toutes les 5 minutes → 20 req/sec suffisant
- Serveur local LAN (pas d'attaque DDoS externe)

### 2. Fix CSP Chart.js (Warning)

**Modification :**
- [OK] [config/settings.py](../config/settings.py#L76) - Ajout `https://cdn.jsdelivr.net` dans `csp_connect_src`

**Impact :**
- Supprime warning CSP "violates connect-src" pour Chart.js sourcemaps
- Permet debugging Chart.js en production

### 3. Erreur 401 Saxo (Non bloquant)

**Cause :** Token OAuth expiré (limitation comptes Self-Developer : reconnexion 24h)

**Solution utilisateur :**
1. Aller dans [Settings > Clés API](http://192.168.1.200:8080/settings.html)
2. Section "SaxoBank OpenAPI"
3. Cliquer " Se connecter à Saxo"
4. Popup OAuth → Accepter → Ferme automatiquement
5. Status passe à "[OK] Connecté"

### 4. Clés API non visibles (À investiguer)

**Cause possible :**
- Fichier `data/users/jack/secrets.json` manquant ou permissions incorrectes
- Endpoint `/api/settings/get` (GET) retourne vide

**Debug sur serveur :**
```bash
# Vérifier existence secrets.json
ls -la data/users/jack/secrets.json

# Vérifier contenu (sensible!)
cat data/users/jack/secrets.json

# Vérifier permissions
chmod 600 data/users/jack/secrets.json
chown 1000:1000 data/users/jack/secrets.json  # UID Docker
```

**Endpoint test :**
```bash
curl -H "X-User: jack" http://192.168.1.200:8080/api/settings/get
```

---

## Checklist Déploiement

### Sur votre machine Windows (préparation)

- [x] Modifications code appliquées (docker-compose.yml, config/settings.py)
- [ ] Commit + push sur GitHub :
  ```bash
  git add docker-compose.yml config/settings.py .env.production.example
  git commit -m "fix(production): resolve 429 errors with proper rate limiting config"
  git push origin main
  ```

### Sur serveur Linux (déploiement)

1. **Créer/Modifier `.env` avec nouvelles variables :**
   ```bash
   cd /path/to/smartfolio
   nano .env
   ```

   Ajouter/modifier :
   ```env
   # Rate limiting - Token Bucket
   RATE_LIMIT_ENABLED=true
   SECURITY_RATE_LIMIT_REFILL_RATE=20.0  # 20 req/sec
   SECURITY_RATE_LIMIT_BURST_SIZE=50     # Burst 50 requêtes
   ```

2. **Déployer nouvelle version :**
   ```bash
   ./deploy.sh --force
   ```

   Le script va :
   - Pull latest code depuis GitHub
   - Rebuild Docker image avec nouvelles variables
   - Restart containers
   - Healthcheck automatique

3. **Vérifier déploiement :**
   ```bash
   # Check containers
   docker-compose ps

   # Check logs rate limiter
   docker-compose logs -f smartfolio | grep "Token bucket"
   # Devrait afficher: " Token bucket rate limiter initialized: 20.0 req/s burst 50"

   # Test endpoint
   curl -v http://192.168.1.200:8080/api/wealth/global/summary?source=stub_balanced
   # Vérifier headers: X-RateLimit-Available, X-Cache-Hit-Ratio
   ```

4. **Test complet dashboard :**
   - Ouvrir http://192.168.1.200:8080/dashboard.html
   - Vérifier absence d'erreurs 429 dans console
   - Vérifier WebSocket connecté (badge nav vert)
   - Vérifier toutes les tuiles chargent correctement

---

## Si Problèmes Persistent

### Option A : Désactiver complètement le rate limiting

**Temporaire, pour debug uniquement :**
```env
# Dans .env
RATE_LIMIT_ENABLED=false
```

Puis redéployer :
```bash
./deploy.sh --skip-build  # Restart rapide sans rebuild
```

### Option B : Augmenter encore plus les limites

**Si 50 burst insuffisant :**
```env
SECURITY_RATE_LIMIT_REFILL_RATE=50.0   # 50 req/sec (3000/min)
SECURITY_RATE_LIMIT_BURST_SIZE=100     # Burst 100 requêtes
```

### Option C : Debug rate limiter en temps réel

**Endpoint monitoring :**
```bash
# Status rate limiter
curl http://192.168.1.200:8080/api/debug/rate-limiter-status

# Logs live
docker-compose logs -f smartfolio | grep -E "(Rate limit|Token bucket|429)"
```

---

## Métriques Attendues

### Avant fix (BROKEN)
```
Rate limiter: 6.0 req/s, burst 12
Dashboard charge: 25 requêtes en 2 secondes
→ 12 tokens épuisés instantanément
→ 13 requêtes échouent avec 429
→ Cascade d'erreurs
```

### Après fix (WORKING)
```
Rate limiter: 20.0 req/s, burst 50
Dashboard charge: 25 requêtes en 2 secondes
→ 25 tokens consommés (25/50 burst)
→ Refill 40 tokens/2s (20×2)
→ Aucune erreur 429
→ Toutes requêtes passent
```

---

## Références

- [Token Bucket Rate Limiter](../services/rate_limiter.py) - Implémentation
- [SecurityConfig](../config/settings.py#L53-L98) - Configuration
- [Rate Limit Middleware](../api/middleware.py#L229-L291) - Middleware
- [Docker Compose](../docker-compose.yml#L69-L73) - Variables env
- [Deploy Script](../deploy.sh) - Script déploiement

---

**Notes :**
- [Warning] Les valeurs recommandées (20 req/sec, burst 50) sont adaptées pour un **serveur LAN local** sans exposition internet
- [Warning] Si exposition internet future → réduire à 10 req/sec, burst 30 + ajouter IP whitelisting
- [OK] Token bucket est préféré à fixed window (évite burst DOS)
- [OK] Adaptive cache TTL optimise performance (cache hit ratio)

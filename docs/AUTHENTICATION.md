# Authentication

> État au 29 juillet 2026. Cette documentation remplace l'ancien flux JWT de
> sept jours stocké dans `localStorage`.

SmartFolio prend en charge une migration progressive vers des sessions par
cookies. Le mode de production cible est `cookie`; `legacy` et `dual` existent
uniquement pour éviter une rupture lors du déploiement.

## Modes d'authentification

| `AUTH_MODE` | Comportement | Usage |
| --- | --- | --- |
| `legacy` | Le login retourne le JWT historique et les requêtes utilisent `Authorization: Bearer`. | Baseline et rollback temporaire |
| `dual` | Le login retourne encore le JWT et pose aussi les cookies de session. Le backend accepte Bearer ou cookie si les identités concordent. | Migration du frontend |
| `cookie` | Les tokens ne sont ni stockés ni retournés au JavaScript. L'authentification utilise uniquement les cookies sécurisés. | Production cible |

En modes `dual` et `cookie`, `X-User` ne constitue jamais une preuve
d'identité. S'il est présent, il doit correspondre à la session authentifiée,
sinon la réponse est `403`.

## Session sécurisée

- access token JWT de 15 minutes par défaut dans `smartfolio_access`;
- refresh token opaque de 7 jours par défaut dans `smartfolio_refresh`;
- token CSRF de 7 jours par défaut dans `smartfolio_csrf`;
- cookies `Secure`, `SameSite=Strict` et `Path=/`;
- cookies access et refresh `HttpOnly`;
- refresh token hashé dans Redis, rotation atomique à chaque utilisation et
  rejet d'un token rejoué;
- révocation au logout, au changement de mot de passe et à la désactivation du
  compte;
- cinq échecs de connexion en quinze minutes entraînent une réponse `429`;
- Redis est obligatoire et fonctionne en mode fermé pour `dual` et `cookie`.

Toutes les mutations doivent envoyer le cookie CSRF et sa valeur dans
`X-CSRF-Token`. Le frontend commun utilise `credentials: "include"`, renouvelle
la session avec `/auth/refresh`, puis rejoue au plus une fois la requête.

Les durées se règlent au démarrage avec `AUTH_ACCESS_TOKEN_MINUTES` (5 à
1440 minutes, 15 par défaut) et `AUTH_SESSION_DAYS` (1 à 90 jours, 7 par
défaut). Le premier délai est technique : son expiration ne déconnecte pas
l'utilisateur tant que la session de renouvellement reste valide. Toute
activité qui renouvelle la session repart pour la durée définie.

## Endpoints

| Méthode et chemin | Fonction |
| --- | --- |
| `POST /auth/login` | Vérifie le mot de passe, applique le rate limiting et crée la session |
| `POST /auth/refresh` | Consomme et remplace le refresh token |
| `POST /auth/logout` | Révoque la session et efface les cookies |
| `GET /auth/session` | Retourne l'utilisateur authentifié et ses rôles |
| `GET /auth/verify` | Compatibilité transitoire; aucun token d'URL en mode `cookie` |
| `POST /auth/change-password` | Change le mot de passe et révoque toutes les sessions |

Le login, le refresh, `/healthz`, le favicon, les assets statiques nécessaires
au login et le callback Saxo validé sont publics. Les autres routes API sont
authentifiées par défaut en modes sécurisés.

## Utilisateurs et rôles

Le registre persistant est `data/auth/users.json`. Au premier démarrage avec
`AUTH_USER_STORE=persistent`, il est initialisé par copie de
`config/users.json`; la source historique n'est pas modifiée.

| Rôle | Autorisations |
| --- | --- |
| `viewer` | Lectures authentifiées |
| `governance_admin` | Gouvernance et exécution |
| `ml_admin` | Mutations et entraînements ML |
| `admin` | Administration, debug et maintenance, plus tous les droits |

Le compte `demo` est refusé lorsque `ENVIRONMENT=production`. Il n'existe pas
d'inscription publique. Les nouveaux mots de passe et changements de mot de
passe doivent contenir au moins 12 caractères.

## Configuration

Exemple de configuration cible sur `robot2`:

```dotenv
ENVIRONMENT=production
DEBUG=false
AUTH_MODE=dual
AUTH_ACCESS_TOKEN_MINUTES=15
AUTH_SESSION_DAYS=7
AUTH_USER_STORE=persistent
AUTH_USERS_PATH=/app/data/auth/users.json
AUTH_ALLOW_LEGACY_USER_STORE=false
JWT_SECRET_KEY=<secret aléatoire d'au moins 32 caractères>
REDIS_URL=redis://redis:6379/0
PUBLIC_BASE_URL=https://segalla.ddns.net
CORS_ORIGINS=https://segalla.ddns.net
ALLOWED_HOSTS=segalla.ddns.net,192.168.1.200,localhost,smartfolio-api
```

Le secret JWT et les identifiants ne doivent jamais être commités. En
production, un secret absent, connu ou trop court empêche le démarrage.

## Frontend

Utiliser le point d'entrée commun:

```javascript
import { apiCall } from './core/fetcher.js';

const response = await apiCall('/api/risk/dashboard');
```

En `cookie`, le frontend ne doit pas lire de token, le placer dans
`localStorage`, le mettre dans une URL ou construire manuellement un header
`Authorization`. `auth-guard.js` et `fetcher.js` gèrent la session, le CSRF et
le renouvellement.

## Déploiement sans rupture

1. Capturer la baseline et démarrer la nouvelle version en `legacy`.
2. Vérifier le login et les parcours LAN existants.
3. Passer à `dual`, vérifier Bearer et cookies, puis migrer le frontend.
4. Publier Caddy et observer les erreurs d'authentification.
5. Après les tests E2E et 48 heures stables, passer à `cookie`.
6. Conserver `/auth/verify` pendant une version, sans token d'URL en `cookie`.

Un rollback d'authentification consiste à remettre temporairement
`AUTH_MODE=dual`, ou `legacy` si le frontend historique l'exige, puis à
redémarrer uniquement SmartFolio. Le registre persistant n'est pas supprimé.

Voir aussi [External access with Caddy](EXTERNAL_ACCESS_CADDY.md) et
[Security](SECURITY.md).

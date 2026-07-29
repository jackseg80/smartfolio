# External access with Caddy

> Hôte public: `segalla.ddns.net`  
> Serveur: `robot2` (`192.168.1.200`)  
> État au 29 juillet 2026: proxy déployé, règles routeur 80/443 actives et
> certificat Let's Encrypt valide.

## Rôle de Caddy

Caddy est un reverse proxy. Il reçoit les connexions HTTP/HTTPS, obtient et
renouvelle automatiquement le certificat de `segalla.ddns.net`, puis transmet
les requêtes à SmartFolio. Il ne remplace ni No-IP, ni le routeur, ni
SmartFolio.

```text
Internet
  ├─ TCP 80  → routeur → robot2:8081 → Caddy:80
  └─ TCP 443 → routeur → robot2:8443 → Caddy:443
                                      ↓
                         smartfolio-api:8080

LAN → http://192.168.1.200:8080 → SmartFolio directement
```

Le port 80 local déjà utilisé par `scalp-radar` n'est pas modifié.

## Fichiers

La stack indépendante se trouve dans `deploy/smartfolio-proxy/`:

- `compose.yaml`: Caddy sur 8081/8443 et réseau externe
  `smartfolio-network`;
- `Caddyfile`: configuration publique et certificat automatique;
- `Caddyfile.local`: préflight avec autorité TLS interne;
- `.env.example`: nom public.

Sur `robot2`, la copie opérationnelle est
`/home/jack/smartfolio-proxy`. Elle ne modifie pas la stack SmartFolio.

## Baseline et préflight réalisés

La sauvegarde est dans:

```text
/home/jack/backups/smartfolio/20260729_external_security_baseline
```

Elle contient la configuration Compose résolue, les identifiants d'images, les
logs utiles et une archive de `data/`, `config/users.json` et `.env` avec son
SHA-256.

Le préflight Caddy avec certificat interne a confirmé:

- redirection HTTP sur le port local 8081;
- `/healthz` et le login servis en HTTPS sur 8443;
- accès LAN direct sur 8080 toujours fonctionnel.

Caddy est maintenant démarré sur `robot2`. Le certificat public Let’s Encrypt
pour `segalla.ddns.net` a été émis avec succès.

## Ordre de déploiement

1. Déployer et tester la nouvelle application en `AUTH_MODE=legacy`.
2. Vérifier le dashboard, analytics, risk, wealth, settings, Saxo, alertes et
   exécution sur le LAN.
3. Passer à `AUTH_MODE=dual`, puis tester Bearer, cookies, refresh, logout et
   CSRF.
4. Valider `docker compose config` et Caddy localement.
5. Supprimer la règle routeur publique historique
   `8080 → 192.168.1.200:8080`.
6. Depuis un réseau externe, confirmer que `segalla.ddns.net:8080` est fermé
   et que l'accès LAN 8080 fonctionne encore.
7. Créer uniquement:
   - TCP public `80` → `192.168.1.200:8081`;
   - TCP public `443` → `192.168.1.200:8443`.
8. Démarrer `smartfolio-proxy` et valider le certificat public en 4G/5G.
9. Observer 48 heures avant de passer à `AUTH_MODE=cookie`.

No-IP reste configuré sur le routeur pour `segalla.ddns.net`. Les challenges de
certificat public nécessitent que les ports publics 80 et 443 atteignent Caddy.

## Configuration SmartFolio

```dotenv
ENVIRONMENT=production
DEBUG=false
AUTH_MODE=dual
AUTH_USER_STORE=persistent
AUTH_USERS_PATH=/app/data/auth/users.json
JWT_SECRET_KEY=<secret aléatoire d'au moins 32 caractères>
REDIS_URL=redis://redis:6379/0
SMARTFOLIO_BIND_ADDRESS=192.168.1.200
PUBLIC_BASE_URL=https://segalla.ddns.net
CORS_ORIGINS=https://segalla.ddns.net
ALLOWED_HOSTS=segalla.ddns.net,192.168.1.200,localhost,smartfolio-api
```

Le proxy de confiance doit être limité au conteneur ou au réseau Caddy. Les
logs Caddy sont rotatifs et le callback Saxo est exclu des access logs pour ne
pas enregistrer un code OAuth. Les cookies et headers `Authorization` ne sont
pas inclus dans le format de log.

## Restriction des services internes

SmartFolio et Caddy sont liés explicitement à `192.168.1.200`, ce qui empêche
leur exposition directe via IPv6. Le firewall et les chaînes Docker doivent
encore autoriser les ports 8080, 8000, 3000,
4000, 9000 et 1883 uniquement depuis les réseaux LAN approuvés, en IPv4 et
IPv6. Avant toute règle:

- relever les règles existantes;
- confirmer le sous-réseau LAN réel;
- tester depuis le LAN et depuis un réseau externe;
- conserver une commande de rollback.

Ne pas publier SSH. Après vérification d'une connexion par clé dans une seconde
session, l'authentification SSH par mot de passe peut être désactivée. Cette
opération nécessite les droits administrateur de `robot2`.

## Validation externe

- `http://segalla.ddns.net` redirige vers HTTPS;
- le certificat est valide pour `segalla.ddns.net`;
- login, navigation, refresh et logout fonctionnent;
- le port public 8080 est fermé;
- Host, Origin et CSRF invalides sont rejetés;
- cinq échecs de login entraînent un `429`;
- WebSocket refuse une session ou une origine invalide;
- aucun port interne n'est accessible depuis Internet.

## Rollback

Le rollback réseau ne touche pas SmartFolio:

1. supprimer les deux règles routeur publiques 80/443;
2. arrêter `/home/jack/smartfolio-proxy`;
3. vérifier que `http://192.168.1.200:8080` répond toujours sur le LAN.

Ne jamais recréer une règle publique vers 8080. Pour un rollback applicatif,
restaurer l'image et la configuration archivées, sans supprimer la sauvegarde
ni le registre persistant des utilisateurs.

# Security

> État au 29 juillet 2026. Les affirmations d'audit historiques qui ne sont
> plus vérifiables ont été retirées.

## Modèle de sécurité

SmartFolio est authentifié par défaut en modes `dual` et `cookie`. Les rares
routes publiques sont explicitement autorisées: login, refresh, `/healthz`,
favicon, assets du login et callback OAuth Saxo validé.

Les protections principales sont:

- sessions courtes par cookies sécurisés, refresh rotatif dans Redis et CSRF;
- rôles réels `viewer`, `governance_admin`, `ml_admin` et `admin`;
- isolation des données sous `data/users/{user_id}/`;
- rejet de `X-User` comme authentification en modes sécurisés;
- refus du compte `demo` en production;
- CORS et hôtes autorisés explicitement;
- rate limiting des connexions et révocation des sessions;
- chargement des modèles uniquement avec `safe_pickle_load` et
  `safe_torch_load`;
- fermeture des montages statiques sensibles (`/data`, `/config`, `/tests`) en
  production;
- absence de secrets et de paramètres OAuth sensibles dans les logs Caddy.

La description complète des sessions est dans
[Authentication](AUTHENTICATION.md).

## Exposition réseau

L'API sur `192.168.1.200:8080` reste disponible uniquement sur le LAN. Internet
accède à Caddy sur les ports publics 80 et 443, redirigés vers les ports 8081 et
8443 de `robot2`. Caddy termine TLS puis contacte `smartfolio-api:8080` sur le
réseau Docker privé.

Aucun port public ne doit cibler directement 8080. Les ports 8080, 8000, 3000,
4000, 9000 et 1883 doivent être limités au LAN en IPv4 et IPv6. SSH ne doit pas
être redirigé depuis Internet.

Voir la procédure et le rollback dans
[External access with Caddy](EXTERNAL_ACCESS_CADDY.md).

## Configuration de production

Les valeurs minimales sont:

```dotenv
ENVIRONMENT=production
DEBUG=false
AUTH_MODE=dual
AUTH_USER_STORE=persistent
JWT_SECRET_KEY=<secret aléatoire d'au moins 32 caractères>
REDIS_URL=redis://redis:6379/0
PUBLIC_BASE_URL=https://segalla.ddns.net
CORS_ORIGINS=https://segalla.ddns.net
ALLOWED_HOSTS=segalla.ddns.net,192.168.1.200,localhost,smartfolio-api
```

Le passage à `AUTH_MODE=cookie` intervient après la migration complète du
frontend et la période d'observation. Les secrets restent exclusivement dans
le fichier `.env` non versionné ou dans un gestionnaire de secrets.

## Contrôles bloquants

La CI exécute:

```powershell
ruff check .
mypy api/auth_security.py api/auth_router.py api/deps.py api/middleware_setup.py api/static_files_setup.py config/users.py --ignore-missing-imports --explicit-package-bases --follow-imports=skip
npm test -- --runInBand
pytest -q
bandit -r api services -ll
pip-audit
python scripts/security/check_unsafe_model_loads.py
python scripts/security/check_openapi_english.py
```

État local validé avant publication:

- Ruff: aucune erreur;
- mypy sur la frontière d'authentification: aucune erreur;
- Jest: 85 tests réussis;
- Bandit: aucun problème de sévérité moyenne ou haute;
- descriptions OpenAPI: 454 chemins validés en anglais;
- aucun chargement direct `pickle.load`, `joblib.load` ou `torch.load`.

L'audit de dépendances reste bloquant dans la CI. Il ne faut pas déclarer
« zéro CVE » sans résultat `pip-audit` daté et archivé.

## Checklist avant ouverture

- [ ] Sauvegarde et identifiants des images Docker archivés
- [ ] Tests Python, Jest, Ruff, mypy, OpenAPI et sécurité verts
- [ ] Login et parcours fonctionnels comparés sur le LAN
- [ ] Toute requête protégée sans session retourne `401`
- [ ] Un viewer reçoit `403` sur gouvernance, exécution et mutations
- [ ] Aucun accès aux données d'un autre tenant ou de `demo`
- [ ] Rotation, rejeu de refresh, logout et CSRF testés
- [ ] WebSocket authentifié et Origin hostile rejeté
- [ ] Aucun `NaN`, division par zéro ou inversion du Risk Score
- [ ] Redirection publique historique vers 8080 supprimée
- [ ] Caddy validé localement avant les règles routeur 80/443
- [ ] Certificat public et redirection HTTP vers HTTPS vérifiés en 4G/5G
- [ ] Ports internes inaccessibles depuis Internet
- [ ] Surveillance 48 heures des réponses 401/403/429/5xx, de Redis, des
      certificats et de l'espace disque

## Incident et rollback

En cas de doute:

1. supprimer les règles publiques 80/443;
2. arrêter uniquement la stack `smartfolio-proxy`;
3. conserver SmartFolio sur le LAN;
4. remettre temporairement `AUTH_MODE=dual` ou `legacy` si le problème concerne
   la migration d'authentification;
5. préserver les logs et la sauvegarde avant toute restauration.

Une clé potentiellement exposée doit être révoquée et remplacée. Ne jamais
publier une vulnérabilité, un secret ou un log sensible dans une issue publique.

# Réparations applicatives Robot2 — 7 octobre 2026

## État livré

Corrections locales dans le worktree `C:\Users\jacks\.codex\worktrees\robot2-scheduler-alerts\smartfolio`, basé sur la version de production `131800ca05a7f2a1c87568c19e465dfa98195dc9`. Livraison Git prévue sur `codex/robot2-scheduler-alerts`. Aucun déploiement, redémarrage ou changement de données sur Robot2 pendant la préparation de ce lot. Le DNS et Caddy restent à part.

## Tâches et caches

Le préchauffage appelle désormais des fonctions internes partagées avec les routes, avec utilisateur et source explicites. Il ne fait plus de requête HTTP à l'application et n'a pas besoin d'un JWT interactif. Les routes conservent `Depends(get_required_user)` et leur protection externe.

- `api/cache_warmup.py` agrège les trois opérations par utilisateur. Réponses JSON FastAPI et dictionnaires sont interprétés ; 401/500, erreurs applicatives `ok=false` / `success=false`, exceptions et délais dépassés comptent comme échecs. Les détails de réponse et secrets ne sont pas exposés dans le statut.
- `api/scheduler.py` enregistre `success` uniquement lorsque toutes les opérations attendues réussissent ; un échec partiel ou total est `error`, avec compteurs et opérations concernées. Aucun repli silencieux vers `jack` lorsque la configuration des utilisateurs est invalide.
- `build_portfolio_metrics` et `build_risk_dashboard` partagent le corps applicatif existant avec leurs routes. La comparaison AST confirme que ces deux corps n'ont pas changé. Les clés de cache, paramètres, scoring et valorisation sont conservés.
- Crypto Toolbox utilise le même chemin de cache que sa route. Une erreur de scraping servie avec un ancien cache n'est plus annoncée comme un rafraîchissement réussi.

Limite conservée : le préchauffage concerne la source `cointracking`, comme avant cette réparation. Il ne qualifie pas automatiquement les autres sources ou les cours Saxo.

## Alertes Redis

`services/alerts/alert_storage.py` corrige la chaîne complète création → lecture → état :

- Les nulls JSON ne deviennent plus `userdata:…` dans les scripts Lua.
- La lecture reconnaît les anciennes représentations nulles dans les champs optionnels concernés, sans réécrire les enregistrements. Les vraies dates sont conservées.
- Les dictionnaires et listes JSON sont décodés en types attendus par `Alert`. Les anciens tableaux vides réencodés comme `{}` pour `escalation_sources` sont pris en charge ; les nouvelles écritures préservent les tableaux vides et imbriqués.
- Les dates de report sont comparées comme dates en Python, avec prise en charge des fuseaux ; la comparaison précédente entre date ISO et chaîne d'époque est supprimée.
- Acquittements et reports modifient maintenant le stockage Redis lorsque Redis est le stockage actif. Le script met à jour l'index actif atomiquement. Les dates optionnelles remises à null sont sérialisées correctement.
- Un enregistrement malformé ne masque pas les enregistrements valides. Les erreurs de décodage incrémentent le compteur de dégradation ; si tout le résultat Redis est invalide, le chemin de secours existant reçoit une erreur explicite.

Aucune migration ni purge des anciennes données de production n'a été exécutée. La compatibilité en lecture permet de présenter les alertes historiques correctement après déploiement sans migration immédiate. Le diagnostic antérieur compte environ 48 000 membres dans l'index actif : ne pas assimiler cet index à des alertes métier valides, ni purger ces données sans revue de la rétention. L'impact de l'affichage de ce grand historique reste à mesurer sur Robot2 avant de déclarer le module pleinement qualifié.

## Tâche ML

La version déployée de `StocksMLAdapter.detect_market_regime` lit `capability_service.result`, renvoie une observation descriptive avec `confidence=None` et n'exécute aucun entraînement, même avec l'ancien argument `force_retrain=True`.

L'ancienne tâche `daily_ml_training` est donc conservée pour la traçabilité avec un statut **`skipped`** et une raison explicite : l'adaptateur est en lecture seule. Elle ne construit plus l'adaptateur, ne formate plus une confiance inexistante et n'annonce plus un entraînement qui n'a pas eu lieu. Son nom de tâche devient `Stock regime retraining (disabled)`.

Cette correction ne réactive pas un entraînement automatique. Toute nouvelle tâche d'entraînement doit utiliser la procédure ML qualifiée et constitue un sujet distinct. Aucun modèle n'a été entraîné ou remplacé par ces réparations.

## Diagnostic des données existantes

`scripts/ops/inspect_alert_redis_nulls.py` fournit un inventaire **en lecture seule**, borné à 500 enregistrements par défaut. Il utilise `REDIS_URL`, ne l'affiche pas, et ne produit que des compteurs/date d'index. Il indique si SCAN est exhaustif et distingue l'index actif des états utilisateur.

Exécution ultérieure dans le contexte de l'instance choisie, une fois sa cible vérifiée :

```powershell
python scripts/ops/inspect_alert_redis_nulls.py --limit 500
```

Cet outil n'a pas été exécuté sur Robot2. Une éventuelle réparation matérielle des champs et une rétention Redis restent séparées : sauvegarde, aperçu des entrées concernées, politique explicite, application idempotente et contrôle après écriture. Aucune commande de purge n'est livrée ici.

## Validation

Suite finale : **3 355 tests réussis, 13 ignorés, 17 avertissements**, incluant toute la suite unitaire et les onze régressions Redis réelles. Durée : 72,14 s. Les avertissements concernent principalement des dépréciations existantes et les données constantes des tests de risque.

- Ruff passe sur les dix fichiers Python applicatifs et de tests concernés ; aucune correction de lint globale.
- Onze tests de régression exécutent les scripts contre Redis **7.0.15** réel, dans une instance WSL jetable sur le port réservé `46379`, sans persistance. Chaque test utilise des clés avec un préfixe UUID et nettoie seulement ses clés. La production et les autres instances Redis ne sont jamais ciblées.
- Les régressions couvrent nulls historiques et nouveaux, tableaux/dictionnaires imbriqués, acquittements et résolutions, report avec/sans fuseau, mises à jour Redis/index, enregistrement invalide partiel/total, identité littérale `null` et inventaire en lecture seule.
- Les tests de tâches couvrent succès, échec partiel/total, retour applicatif d'erreur, exceptions, délai dépassé, configuration invalide, séparation utilisateur/source, protection HTTP et utilisation du même cache.
- Pour la suite complète, `ALLOWED_HOSTS=testserver,localhost,127.0.0.1` est défini uniquement dans le processus de test. Sans cela, le middleware refuse la collecte des tests qui importent l'application. Aucune protection d'authentification n'est désactivée pour contourner ce contrôle.
- `git diff --check` passe sur les fichiers suivis modifiés. Les 36 autres fichiers signalés dans ce nouveau worktree ont uniquement des différences préexistantes de fin de ligne ; leurs contenus normalisés correspondent à HEAD. Ils n'ont pas été reformattés ou inclus dans la réparation.

Preuves locales : `outputs/robot2-repairs/final-suite.txt`, `unit-suite.txt`, `diff-check.json`. Les essais ciblés intermédiaires et leurs ajustements sont remplacés par la validation finale ci-dessus.

## Étape suivante sur Robot2

Après autorisation distincte de livraison/déploiement :

1. Préparer un diff/commit limité aux fichiers de ce lot et une version de retour arrière vérifiée, en excluant les différences de fin de ligne sans rapport avec la réparation.
2. Construire et déployer l'image ; comparer les fichiers exécutés/servis à la version approuvée.
3. Vérifier plusieurs cycles du préchauffage et Crypto Toolbox : disparition des 401 concernés, résultats réels des caches et statuts cohérents.
4. Vérifier les alertes dans une session authentifiée, leur report et leur acquittement, ainsi que le volume et le temps de lecture du grand historique.
5. Vérifier que la tâche ML indique bien `skipped` avec sa raison, et que les autres observations ML restent disponibles selon leur contrat.

Le succès des tests locaux ne prouve pas encore ces comportements sur le serveur. Les conteneurs et données de Robot2 sont inchangés par ce lot.


## Déployer la branche pour tester sur Robot2

Lancer `ssh robot2` depuis PowerShell, puis copier ce bloc dans le shell Linux. Il construit l'image et redémarre l'API SmartFolio. Il conserve une étiquette de retour arrière de l'image actuelle. La base attendue est celle vérifiée le 7 octobre : `131800ca`.

Les cinq modifications locales observées sur Robot2 (debug HTML, scripts de téléchargement/maintenance et simulateur) sont hors du lot et peuvent rester présentes. Le contrôle ci-dessous porte sur les fichiers du lot ; Git refuse aussi une collision avec un fichier non suivi. Aucun stash, reset ou checkout forcé n'est utilisé.

```bash
(
  set -eu
  cd /home/jack/smartfolio
  test "$(git rev-parse HEAD)" = "131800ca05a7f2a1c87568c19e465dfa98195dc9"
  git fetch origin codex/robot2-scheduler-alerts
  git diff --quiet HEAD -- api/cache_warmup.py api/crypto_toolbox_endpoints.py api/portfolio_endpoints.py api/risk_endpoints.py api/scheduler.py services/alerts/alert_storage.py scripts/ops/inspect_alert_redis_nulls.py tests/unit/test_scheduler.py tests/unit/test_cache_warmup.py tests/integration/test_alert_storage_redis_regression.py docs/audit/ROBOT2_APPLICATION_REPAIRS_2026-10-07.md
  rollback_tag=smartfolio-rollback:before-robot2-repairs-20261007
  if docker image inspect "$rollback_tag" >/dev/null 2>&1; then
    echo "Rollback tag already exists; stop to preserve it."
    exit 1
  fi
  docker image tag "$(docker inspect --format '{{.Image}}' smartfolio-api)" "$rollback_tag"
  git switch --detach origin/codex/robot2-scheduler-alerts
  git log -1 --oneline
  docker compose build smartfolio
  docker compose up -d --no-deps smartfolio
  docker compose ps smartfolio
  curl -fsS http://192.168.1.200:8080/healthz
)
```

Si le contrôle de version ou de fichiers refuse la commande, conserver la sortie et examiner l'écart ; ne pas remplacer ce contrôle par un reset forcé. Si la construction échoue, l'API actuelle continue avec son image précédente, mais le checkout est déjà sur la branche de test : cela ne prouve pas que les corrections sont exécutées.

### Vérifier la version dans le conteneur

Ce bloc compare les fichiers applicatifs réellement exécutés à HEAD, en neutralisant les différences LF/CRLF. Aucun accès aux variables d'environnement ou données utilisateur.

```bash
cd /home/jack/smartfolio
python3 - <<'PY'
import hashlib, json, subprocess
files = ['api/cache_warmup.py', 'api/scheduler.py', 'api/crypto_toolbox_endpoints.py',
         'api/portfolio_endpoints.py', 'api/risk_endpoints.py', 'services/alerts/alert_storage.py']
def digest(value):
    return hashlib.sha256(value.replace(b'\r\n', b'\n')).hexdigest()
expected = {f: digest(subprocess.check_output(['git', 'show', 'HEAD:' + f])) for f in files}
code = ('import hashlib,json; from pathlib import Path; files=' + repr(files)
        + '; print(json.dumps({f:hashlib.sha256(Path("/app",f).read_bytes().replace(b"\\r\\n",b"\\n")).hexdigest() for f in files}))')
actual = json.loads(subprocess.check_output(['docker', 'exec', 'smartfolio-api', 'python', '-B', '-c', code]))
mismatches = [f for f in files if actual.get(f) != expected[f]]
print(json.dumps({'head': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                  'mismatched_files': mismatches}, indent=2))
raise SystemExit(bool(mismatches))
PY
```

### Tester les comportements

- Se connecter à SmartFolio via son adresse LAN habituelle. Le DNS public ne participe pas à ce test.
- Attendre le prochain préchauffage, au maximum environ dix minutes après le démarrage du scheduler, puis consulter son statut dans Monitoring. Si le statut est `error`, examiner les opérations signalées : une source vide ou indisponible doit maintenant être visible, plutôt qu'un faux `success`.
- Vérifier les alertes actives, leur temps de chargement et les dates. Un test d'acquittement modifie un état métier : choisir une alerte que l'utilisateur souhaite effectivement acquitter. Un report peut servir à contrôler le filtrage avec la même précaution.
- Crypto Toolbox se rafraîchit aux horaires planifiés 08:00/20:00 Europe/Zurich ; la tâche ML conservée prend le statut `skipped` à son prochain passage quotidien vers 03:00. Les anciens statuts Redis peuvent rester visibles jusqu'à ce passage.
- Ne pas déclencher le préchauffage dans un nouveau processus `docker exec` pour prétendre réchauffer le cache mémoire du worker API : ce serait le cache d'un autre processus.

Lecture seule des trois statuts persistants, sans appeler les tâches :

```bash
docker exec smartfolio-api python -B -c 'import asyncio,json; from api.scheduler import get_job_status_persistent; d=asyncio.run(get_job_status_persistent()); print(json.dumps({k:d.get(k) for k in ["api_warmers","crypto_toolbox_refresh","daily_ml_training"]},indent=2))'
```

Comparer les champs `last_run` à l'heure de déploiement ; un ancien `success` ou `error` ne qualifie pas la nouvelle image. `/healthz` HTTP 200 et Docker `healthy` restent des contrôles de disponibilité.

### Retour arrière

Si le test nécessite un retour à l'image sauvegardée, exécuter dans le même shell Linux :

```bash
(
  set -eu
  cd /home/jack/smartfolio
  docker image inspect smartfolio-rollback:before-robot2-repairs-20261007 >/dev/null
  git switch --detach 131800ca05a7f2a1c87568c19e465dfa98195dc9
  docker image tag smartfolio-rollback:before-robot2-repairs-20261007 smartfolio-smartfolio:latest
  docker compose up -d --no-deps --no-build --force-recreate smartfolio
  docker compose ps smartfolio
  curl -fsS http://192.168.1.200:8080/healthz
)
```

Ce retour arrière restaure le code exécuté. Il ne restaure pas les éventuels acquittements/reports d'alertes réalisés pendant le test ni les caches renouvelés. Aucune opération de migration ou purge de Redis n'est incluse dans la livraison.

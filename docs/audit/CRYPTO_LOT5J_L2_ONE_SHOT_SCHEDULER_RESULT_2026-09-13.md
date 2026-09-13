# Résultat — lot 5J, déclenchement L2 aligné et mono-instance

Date : 13 septembre 2026<br>
Périmètre : tests hors ligne et une exécution publique unique sur une borne UTC. Aucun planificateur permanent, compte, ordre, modèle ou environnement SmartFolio.

## Décision

Le lot est **`GO_OPERATIONAL_ONE_SHOT` selon le contrat préenregistré**.

Le processus s'est déclenché 14 millisecondes après la borne de 11:00 UTC, a validé et écrit les trois carnets en 2,105 secondes au total, puis s'est arrêté et a libéré son verrou.

Ce Go prouve le comportement d'une exécution unique alignée. Il n'autorise pas encore une tâche récurrente pendant 420 jours.

## Essai réel aligné

| Étape | Résultat |
|---|---:|
| Borne cible | 2026-09-13 11:00:00 UTC |
| Début du processus de capture | +14 ms |
| BTC-USDT durable | +1 267 ms |
| ETH-USDT durable | +1 764 ms |
| SOL-USDT durable | +2 105 ms |
| Captures | 3/3 |
| Erreurs | 0 |
| Instruments manquants sur la borne | 0 |
| Backfill | Aucun |
| Verrou restant après arrêt | Aucun |

Les trois captures sont très largement sous la limite préenregistrée de 120 secondes. Les créneaux antérieurs de la journée, où aucun collecteur n'était actif, restent explicitement manquants et n'ont pas été reconstitués.

## Comportements validés hors ligne

- la prochaine borne calculée est strictement future et alignée sur 15 minutes ;
- un second processus est refusé pendant que le verrou existe ;
- seul le propriétaire du verrou peut le retirer ;
- une arrivée après 120 secondes produit `SKIPPED_LATE`, zéro requête et zéro backfill ;
- une panne simulée sur un actif produit `PARTIAL_FAILURE` et conserve la capture déjà réussie ;
- les chemins sortant du dossier prévu sont refusés pour les captures et le verrou ;
- les timestamps de génération sont exclus de l'identité reproductible.

## Reproductibilité

Artifact final : `crypto-forecast-okx-l2-one-shot-scheduler-v1-3b0bb767f3d9b748`.

- résultat SHA-256 : `a34689f0b682d9afa6338f413a21c99eb7d8f141c0761df892cd846f9e3f65e3` ;
- manifeste journalier SHA-256 : `8133bc4701c6f58efc812ff591479abe6df4f9a549a99e9476795c4122e76d95` ;
- configuration SHA-256 : `2160d3e3a5a232720fd2994029d68915a54d8286f0402157c79f2ecdba79a229` ;
- code du déclencheur SHA-256 : `f41a875a2489f0bca0b5cbdca5929b67eb7121c82aa33cd60041017a20d974e` ;
- code du collecteur SHA-256 : `f85a1245344c400e37ac4bb7f7539d6f1d571951af004a238d5170e6a2a29c4b`.

Une relecture entièrement hors ligne a produit le même identifiant, le même résultat et le même manifeste, octet pour octet. Les deux exécutions ont libéré leur verrou.

## Contrôles et limites

- six tests unitaires spécifiques au déclencheur réussis ;
- six tests du collecteur sous-jacent toujours réussis ;
- Ruff et Black conformes ;
- aucune tâche Windows, cron ou démarrage automatique ;
- aucun processus laissé actif ;
- aucun secret, compte, ordre ou modèle ;
- aucune API, interface, port local ou production modifié ;
- un seul essai réel : la fiabilité sur veille, redémarrage et plusieurs jours n'est pas encore observée ;
- aucune validation prédictive ou économique.

## Suite recommandée

Le collecteur et son déclenchement unique sont techniquement prêts. La prochaine décision n'est plus un simple test de code : il faut choisir comment obtenir l'historique utile.

La voie prospective est légère mais exige 420 jours. Avant de l'activer, le prochain lot recommandé doit tester une acquisition historique progressive sur un très petit nombre de journées : télécharger une archive à la fois, extraire seulement les snapshots alignés, vérifier leur artifact, puis mesurer le coût réel de transfert et de traitement. Aucune suppression temporaire ni extension du volume ne doit être implicite.

L'installation d'une tâche récurrente et toute campagne d'acquisition restent des autorisations séparées.

Il n'y a rien à tester dans le navigateur : l'application et la production restent inchangées.

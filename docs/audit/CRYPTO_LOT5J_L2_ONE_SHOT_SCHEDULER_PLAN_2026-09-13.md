# Plan préenregistré — lot 5J, déclenchement L2 aligné et mono-instance

Date de gel : 13 septembre 2026, avant l'essai aligné.

## Objectif

Valider un déclenchement unique sur la prochaine borne UTC de 15 minutes, sous 120 secondes, avec verrou mono-instance et comportement sûr après retard ou panne réseau. Aucun planificateur permanent ne sera installé.

## Entrée figée

- collecteur source : `crypto-forecast-okx-l2-prospective-collector-v1-9bb1d74206beaac2` ;
- résultat source SHA-256 : `802fae34a45dda4f7baf9ccb95de5eed9fe439b39c511de71045304832b31bef` ;
- endpoint public OKX, trois instruments spot, 400 niveaux par côté ;
- grille UTC de 15 minutes ;
- tolérance maximale : 120 secondes après la borne.

## Contrat du déclencheur

1. La prochaine borne est strictement postérieure à l'heure de démarrage et alignée sur 15 minutes.
2. Un fichier créé exclusivement protège l'ensemble de la séquence ; un second processus échoue sans requête.
3. Le verrou n'est retiré que par son propriétaire ; un verrou abandonné est signalé et jamais supprimé automatiquement.
4. Si le processus arrive après la borne plus 120 secondes, le créneau est `SKIPPED_LATE` sans requête ni backfill.
5. Chaque réponse réussie est validée et écrite immédiatement.
6. Une panne partielle conserve les succès, publie `PARTIAL_FAILURE` et laisse les autres instruments manquants.
7. Une exécution complète exige trois captures valides, chacune terminée sous 120 secondes.
8. Le journal et le manifeste sont déterministes ; les timestamps de génération sont exclus de l'identité.

## Critères Go/No-Go

Le lot est **Go opérationnel limité** seulement si :

- les tests couvrent calcul de borne, verrou concurrent, retard après veille, panne partielle et reproduction ;
- l'essai réel termine les trois captures sous 120 secondes ;
- aucune capture n'est remplie ou antidatée ;
- aucun verrou ou processus ne reste actif après l'essai ;
- une relecture hors ligne reproduit le même identifiant et les mêmes empreintes ;
- les tests ciblés, Ruff et Black passent.

Un échec réseau ou un dépassement du délai reste un résultat No-Go valide ; les seuils ne seront pas modifiés après observation.

## Hors périmètre

- tâche Windows, cron, service ou démarrage automatique ;
- campagne de 420 jours ;
- rattrapage des créneaux manqués ;
- modèle, feature, cible, backtest ou allocation ;
- compte, secret, ordre, dérivé ou levier ;
- API SmartFolio, interface, port local ou production ;
- suppression d'artifact, commit, push ou déploiement.

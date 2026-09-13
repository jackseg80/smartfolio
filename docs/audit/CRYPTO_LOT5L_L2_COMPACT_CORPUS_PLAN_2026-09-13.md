# Plan préenregistré — lot 5L, corpus L2 compact multi-périodes

Date de gel : 13 septembre 2026, avant exécution sur les neuf archives.

## Objectif

Étendre le mécanisme validé au lot 5K aux neuf archives OKX déjà acquises afin de produire un petit corpus de carnets complets couvrant trois dates, trois actifs et deux cadences historiques. Aucun nouveau téléchargement et aucune suppression ne sont autorisés.

## Entrées figées

- dates UTC : `2023-04-01`, `2024-07-01` et `2026-07-01` ;
- instruments spot : BTC-USDT, ETH-USDT et SOL-USDT ;
- neuf archives déjà épinglées par taille et SHA-256 ;
- volume compressé total : `1 181 503 619` octets ;
- snapshots natifs valides attendus : `8 927` ;
- créneaux alignés attendus : `863/864` ;
- manque attendu : ETH-USDT, `2023-04-01 00:00:00 UTC` ;
- méthode source : artifact 5K `crypto-forecast-okx-l2-progressive-extraction-v1-bbdf89ecd6809b32`.

### Addendum d'intégrité avant consommation

Le premier essai consommateur du lot 5M a démontré que cet artifact 5K initial omettait les séparateurs de ligne entre objets JSON. Sans changer la méthode, la grille, les seuils ou les snapshots sélectionnés, le producteur a été corrigé puis rejoué. La source de méthode valable devient `crypto-forecast-okx-l2-progressive-extraction-v1-8ca057800a4ec6e4`, résultat SHA-256 `ded85d4b08ec400fe2af1d63b2bb0c0e6a920827de7f5ea8f53a482d0de6ae34`. L'identifiant initial reste ci-dessus pour préserver la chronologie du préenregistrement, mais il ne doit plus être consommé.

## Méthode gelée

1. Vérifier la cohérence de la configuration et les neuf identités d'archive avec le manifeste d'acquisition.
2. Lire les archives une par une, dans l'ordre date puis instrument, sans extraire leur membre brut sur disque.
3. Réutiliser strictement la validation et la sélection 5K : snapshots autonomes seulement, grille de 15 minutes, fenêtre symétrique ±1 000 ms, préférence antérieure en cas d'égalité.
4. Conserver séparément les timestamps de grille, source et disponibilité causale.
5. Publier un fichier JSONL gzip déterministe par date et instrument.
6. Préserver explicitement le créneau manquant, sans interpolation ni remplissage.
7. Exclure les durées d'exécution de l'identité reproductible.
8. Refuser tout payload dont le chemin, la taille ou le SHA-256 diffère du résultat annoncé.

## Critères Go/No-Go

Le lot est **Go technique** seulement si :

- les neuf archives correspondent aux entrées gelées ;
- les `8 927` snapshots natifs sont tous valides et non croisés ;
- `863` carnets sont retenus et le seul manque est exactement celui préenregistré ;
- les répartitions avant/exact/après et les écarts maximaux reproduisent les neuf groupes du lot 5G ;
- le corpus gzip total ne dépasse pas 32 MiB ;
- deux lectures complètes produisent le même identifiant, le même résultat et les mêmes neuf fichiers ;
- les tests ciblés, Ruff et Black passent.

Toute divergence reste un No-Go ; aucun seuil ne sera ajusté après observation.

## Hors périmètre

- nouvelle archive, téléchargement ou abonnement ;
- suppression ou modification des sources ;
- tâche récurrente, service ou démarrage automatique ;
- modèle, feature prédictive, cible, backtest ou allocation ;
- API, interface, port local ou production ;
- compte, secret, ordre, dérivé ou levier ;
- commit, push ou déploiement.

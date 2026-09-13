# Plan préenregistré — lot 5H, faisabilité d'un historique L2 continu

Date de gel : 13 septembre 2026, avant les trois requêtes de mesure.

## Objectif

Décider si une quantité minimale d'historique continu pour des cibles à 7 et 30 jours est matériellement réaliste, sans télécharger d'autres archives. Comparer deux voies : les archives historiques complètes et une collecte prospective des seuls snapshots nécessaires.

Ce lot est un contrôle de volume, de cadence et de validité technique. Il ne lance pas de collecte longue durée et ne valide aucune feature ou prévision.

## Fenêtre minimale figée

Le plancher de faisabilité est fixé à `420` jours calendaires continus :

- entraînement : 180 jours ;
- purge : 30 jours ;
- validation : 60 jours ;
- purge : 30 jours ;
- calibration : 30 jours ;
- purge : 30 jours ;
- test final : 60 jours.

Les trois purges sont chacune égales à la cible maximale de 30 jours. Cette fenêtre est un minimum d'ingénierie pour séparer chronologiquement les phases, pas une preuve de puissance statistique suffisante.

## Voie historique

Le coût sera extrapolé sur 420 jours à partir de la distribution déjà gelée au lot 5D pour une journée complète BTC + ETH + SOL : médiane `490,61 MB`, P95 `715,3065 MB`, maximum `767,69 MB`.

Décision **Go historique local** uniquement si les trois projections restent sous `20 GiB`. Aucun fichier d'archive ne sera téléchargé dans ce lot.

## Voie prospective

Trois requêtes publiques et sans authentification seront réalisées, une par instrument, sur `GET /api/v5/market/books` avec `sz=400`. Elles mesureront la taille exacte du JSON non compressé et une taille gzip déterministe.

Pour chaque réponse, le contrôle exige :

- code OKX `0` et un carnet unique ;
- 400 niveaux bid et 400 niveaux ask ;
- prix et quantités strictement positifs et finis ;
- bids décroissants, asks croissants et meilleur bid inférieur au meilleur ask ;
- timestamp âgé d'au plus 60 secondes et ne dépassant pas la fin de requête de plus de 5 secondes ;
- réponse non compressée inférieure ou égale à 128 KiB.

La projection retient 96 captures par jour et par instrument, soit 288 appels par jour pendant 420 jours. Le lot est **Go prospectif de faisabilité** uniquement si :

- les trois réponses passent les contrôles ;
- la projection JSON brute reste sous `10 GiB` ;
- la projection gzip, multipliée par un facteur de sécurité de 3, reste sous `5 GiB` ;
- une salve de trois requêtes reste sous la limite officielle de 40 requêtes par 2 secondes.

Les payloads du pilote et leurs empreintes seront conservés dans un artifact. Une relecture hors ligne devra reproduire le même résultat et le même identifiant.

## Livrables prévus

1. `config/crypto_forecast_okx_l2_collection_feasibility.json` — contrat gelé.
2. `services/forecasting/okx_l2_collection_feasibility.py` — validation et projections.
3. `scripts/run_crypto_forecast_l2_collection_feasibility.py` — pilote public ou relecture locale.
4. `tests/unit/test_crypto_forecast_okx_l2_collection_feasibility.py` — validation, budget et reproductibilité.
5. `docs/audit/CRYPTO_LOT5H_L2_COLLECTION_FEASIBILITY_RESULT_2026-09-13.md` — résultat final.

## Hors périmètre

- nouveau téléchargement d'archive historique ;
- démarrage d'un service ou d'une collecte récurrente ;
- modèle, feature, cible, backtest, allocation ou seuil de trading ;
- compte, secret, ordre, dérivé ou levier ;
- API SmartFolio, interface, port local ou production ;
- suppression des fichiers 5E ;
- commit, push ou déploiement.

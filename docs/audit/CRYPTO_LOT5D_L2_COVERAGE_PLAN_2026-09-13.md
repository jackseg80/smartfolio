# Plan préenregistré — lot 5D, couverture et coût des snapshots L2 OKX

Date de gel : 13 septembre 2026, avant toute interrogation de couverture multi-date.

## Objectif

Mesurer la disponibilité et la taille des archives L2 à 400 niveaux pour BTC-USDT, ETH-USDT et SOL-USDT sur un échantillon calendaire couvrant l'historique annoncé, sans télécharger aucune archive.

Ce lot doit déterminer si un prochain échantillon snapshot-only multi-actifs reste matériellement borné. Il ne construit pas de dataset de modèle.

## Source

- endpoint public : `GET https://www.okx.com/api/v5/public/market-data-history` ;
- module : `4`, carnet à 400 niveaux ;
- type : `SPOT` ;
- agrégation : `daily` ;
- instruments : `BTC-USDT`, `ETH-USDT`, `SOL-USDT` ;
- authentification : aucune.

Seules les métadonnées JSON de l'endpoint sont admises. Les URL d'archives retournées ne doivent pas être appelées dans ce lot.

## Dates figées

Le relevé utilise le premier jour UTC de chaque trimestre disponible, indépendamment des prix, volumes ou résultats de modèle :

- 2023 : `2023-04-01`, `2023-07-01`, `2023-10-01` ;
- 2024 : `2024-01-01`, `2024-04-01`, `2024-07-01`, `2024-10-01` ;
- 2025 : `2025-01-01`, `2025-04-01`, `2025-07-01`, `2025-10-01` ;
- 2026 : `2026-01-01`, `2026-04-01`, `2026-07-01`.

Cela représente 14 dates et 42 couples instrument-date attendus.

## Contrôles

Pour chaque couple instrument-date :

1. présence ou absence explicite du fichier ;
2. date UTC exacte ;
3. nom conforme à l'instrument, au module et à la date ;
4. URL HTTPS sous `static.okx.com` ;
5. taille positive et finie en MB ;
6. absence de doublon ;
7. conservation du fournisseur, des paramètres et de la date de collecte ;
8. aucun secret, compte ou téléchargement d'archive.

Les absences restent absentes. Aucun remplacement de date n'est permis.

## Statistiques figées

- couverture totale et par instrument ;
- taille min, médiane, P95 et maximum par instrument ;
- taille totale BTC + ETH + SOL pour chaque date complète ;
- distribution min, médiane, P95 et maximum de ces totaux quotidiens ;
- extrapolation indicative à 30 jours et 365,25 jours depuis la médiane quotidienne, explicitement présentée comme estimation ;
- taille cumulée des trois dates candidates figées ci-dessous.

## Critères de décision

Le relevé est **valide** si :

- au moins 38 des 42 fichiers attendus sont présents ;
- chaque instrument est présent sur au moins 12 des 14 dates ;
- aucune métadonnée présente n'est invalide ou dupliquée ;
- deux exécutions normalisées produisent le même identifiant et les mêmes empreintes.

Le prochain échantillon multi-actifs est **matériellement faisable** seulement si les trois dates suivantes possèdent chacune BTC, ETH et SOL et si leur taille compressée cumulée ne dépasse pas `2 000 MB` :

- `2023-04-01` ;
- `2024-07-01` ;
- `2026-07-01`.

Ce seuil ne vaut pas autorisation de téléchargement. Même si le critère passe, le prochain lot devra préenregistrer séparément l'acquisition, le traitement en flux, la rétention et le nettoyage éventuel des archives.

## Livrables prévus

1. `config/crypto_forecast_okx_l2_coverage.json` — dates, instruments et seuils gelés.
2. `services/forecasting/okx_l2_coverage.py` — validation et normalisation des métadonnées.
3. `scripts/run_crypto_forecast_l2_coverage.py` — collecte publique sans archive.
4. `tests/unit/test_crypto_forecast_okx_l2_coverage.py` — trous, doublons, URL, tailles et reproductibilité.
5. `docs/audit/CRYPTO_LOT5D_L2_COVERAGE_RESULT_2026-09-13.md` — résultat et budget recommandé.

## Hors périmètre

- téléchargement d'une archive L2 ;
- calcul de feature depuis un carnet ;
- modèle, cible, allocation ou seuil de trading ;
- API, interface, port 8082 ou production 8080 ;
- compte, clé exchange, ordre, levier ou dérivé ;
- commit, push ou déploiement.

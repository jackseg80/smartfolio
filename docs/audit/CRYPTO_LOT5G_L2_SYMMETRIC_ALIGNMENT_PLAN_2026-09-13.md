# Plan préenregistré — lot 5G, alignement causal symétrique des snapshots L2

Date de gel : 13 septembre 2026, avant toute exécution de l'alignement symétrique.

## Objectif

Tester une grille commune de 15 minutes qui accepte les petites oscillations de l'horloge native avant ou après chaque borne, sans masquer le moment réel où une observation devient disponible.

Ce lot répond au No-Go du lot 5F. Il ne modifie ni les résultats 5E/5F, ni les snapshots source.

## Entrée figée

- artifact source : `crypto-forecast-okx-l2-sample-v1-973d02b919af2d1c` ;
- résultat source SHA-256 : `b32045a424853d22de978a8a3302873c74f64ef0d1d91203455853b713a3d6e2` ;
- métriques source SHA-256 : `27c102f090b48521267b95f11e7c2aafb09327cd85cefe0ca75d4a0b4984e58e` ;
- trois instruments et trois journées du lot 5E ;
- aucune requête réseau ou lecture supplémentaire des archives brutes.

## Règle de sélection figée

Pour chacun des 96 créneaux de 15 minutes par instrument-date :

1. considérer uniquement les snapshots valides situés entre `-1 000 ms` et `+1 000 ms` autour de la borne ;
2. sélectionner celui dont l'écart absolu à la borne est minimal ;
3. en cas d'égalité absolue, préférer le snapshot antérieur ou exactement sur la borne ;
4. conserver le décalage signé et absolu ;
5. définir l'heure de disponibilité causale comme le maximum entre la borne de grille et le timestamp source ;
6. publier un créneau manquant si aucun snapshot n'est dans la fenêtre ;
7. interdire le remplissage, l'interpolation, les deltas et la réutilisation d'un snapshot sur deux créneaux.

Un snapshot antérieur est déjà connu à l'heure de grille. Un snapshot postérieur ne peut être utilisé qu'à son propre timestamp. Cette distinction doit rester explicite dans chaque ligne.

## Critères Go/No-Go

Le lot est **Go technique d'alignement** seulement si :

- les neuf séries attendues sont présentes ;
- chacune publie exactement 96 créneaux ;
- chacune possède au moins 95 créneaux disponibles et valides ;
- chaque écart absolu est inférieur ou égal à 1 000 ms ;
- les heures de disponibilité ne précèdent jamais ni la borne ni la source ;
- le résultat totalise exactement 864 lignes ;
- un snapshot source n'est jamais sélectionné deux fois ;
- la mutation de snapshots non sélectionnés ne change aucune ligne alignée ;
- deux exécutions produisent le même identifiant et les mêmes empreintes.

Le seuil 95/96 reste celui du lot 5F. Ce lot est une correction méthodologique préenregistrée après deux pilotes exploratoires, pas une validation indépendante ou prédictive.

## Livrables prévus

1. `config/crypto_forecast_okx_l2_symmetric_alignment.json` — contrat gelé.
2. `services/forecasting/okx_l2_symmetric_alignment.py` — sélection et disponibilité causales.
3. `scripts/run_crypto_forecast_l2_symmetric_alignment.py` — exécution hors ligne.
4. `tests/unit/test_crypto_forecast_okx_l2_symmetric_alignment.py` — fenêtres, causalité, trous et reproductibilité.
5. `docs/audit/CRYPTO_LOT5G_L2_SYMMETRIC_ALIGNMENT_RESULT_2026-09-13.md` — résultat final.

## Hors périmètre

- réseau, nouveau téléchargement ou extraction brute ;
- modification ou suppression des artifacts précédents ;
- modèle, cible 7/30 jours, backtest ou allocation ;
- API, interface, port local ou production ;
- ordre, compte, secret, levier ou dérivé ;
- commit, push ou déploiement.

# Plan préenregistré — lot 5F, normalisation temporelle des snapshots L2

Date de gel : 13 septembre 2026, avant toute exécution de la normalisation.

## Objectif

Tester si les snapshots natifs hétérogènes du lot 5E peuvent former une grille causale commune de 15 minutes, sans nouveau téléchargement, interpolation, propagation d'état ou utilisation des deltas.

Ce lot répond uniquement au changement de cadence observé entre 2023–2024 et 2026. Le No-Go strict du lot 5E reste inchangé.

## Entrée figée

- artifact source : `crypto-forecast-okx-l2-sample-v1-973d02b919af2d1c` ;
- résultat source SHA-256 : `b32045a424853d22de978a8a3302873c74f64ef0d1d91203455853b713a3d6e2` ;
- métriques source SHA-256 : `27c102f090b48521267b95f11e7c2aafb09327cd85cefe0ca75d4a0b4984e58e` ;
- instruments : BTC-USDT, ETH-USDT et SOL-USDT ;
- journées UTC : 1er avril 2023, 1er juillet 2024 et 1er juillet 2026 ;
- lignes snapshot natives observées : au plus `10 000` admises ;
- accès réseau : aucun.

Les métriques source doivent être re-hachées avant lecture. Une ligne non finie, invalide, dupliquée ou hors du périmètre gelé rend le lot invalide.

## Grille et règle de sélection

Pour chaque instrument et chaque journée :

1. créer exactement 96 créneaux UTC espacés de `900 000 ms`, de 00:00 à 23:45 ;
2. pour chaque créneau, choisir le premier snapshot valide dont le timestamp est supérieur ou égal au créneau ;
3. accepter ce snapshot seulement si son retard est compris entre `0` et `1 000 ms` inclus ;
4. si aucun snapshot ne satisfait cette fenêtre, publier un créneau explicitement indisponible ;
5. ne jamais utiliser un snapshot antérieur, un delta, une valeur future au-delà de la seconde, une interpolation ou un remplissage ;
6. publier le timestamp de grille, le timestamp source et le retard exact afin que la disponibilité réelle soit traçable.

Le timestamp causal d'une ligne disponible est le timestamp source, pas le début théorique du créneau.

## Critères Go/No-Go

Le lot est **Go technique de normalisation** seulement si :

- les neuf séries attendues sont présentes ;
- chacune publie exactement 96 créneaux ;
- chacune possède au moins 95 créneaux disponibles et valides, soit au plus un manque explicite par journée ;
- chaque ligne sélectionnée respecte le retard maximal de 1 000 ms ;
- le résultat contient exactement 864 lignes de grille ;
- aucun snapshot natif ne peut servir à deux créneaux ;
- une mutation des snapshots non sélectionnés ne modifie aucune ligne sélectionnée ;
- deux exécutions produisent le même identifiant et les mêmes empreintes.

Le seuil de 95/96 correspond à une couverture journalière minimale de 98,95 %. Il formalise le cas découvert au lot 5E ; ce lot reste une étude de faisabilité, pas une confirmation indépendante ni une validation prédictive.

## Mesures prévues

- couverture disponible par instrument-date ;
- créneaux manquants explicites ;
- retard min, médian, P95 et maximum ;
- nombre de snapshots natifs sélectionnés et ignorés ;
- spread, profondeurs et déséquilibres existants, sans recalcul depuis les updates.

## Livrables prévus

1. `config/crypto_forecast_okx_l2_normalization.json` — contrat gelé.
2. `services/forecasting/okx_l2_normalization.py` — validation et sélection causale.
3. `scripts/run_crypto_forecast_l2_normalization.py` — exécution hors ligne.
4. `tests/unit/test_crypto_forecast_okx_l2_normalization.py` — causalité, trous et reproductibilité.
5. `docs/audit/CRYPTO_LOT5F_L2_NORMALIZATION_RESULT_2026-09-13.md` — résultat et limites.

## Hors périmètre

- requête réseau ou nouvelle archive ;
- reconstruction des deltas ;
- fabrication d'un créneau manquant ;
- cible à 7/30 jours, modèle, sélection de feature, backtest ou trading ;
- API, interface, port local ou production ;
- suppression des archives du lot 5E ;
- commit, push ou déploiement.

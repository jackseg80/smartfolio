# Lot 5Q — Plan gelé des features causales de funding

Date de gel : 13 septembre 2026. Ce document précède la construction des features.

## Question

Peut-on transformer l'historique validé au lot 5P en une table quotidienne dont chaque valeur est effectivement disponible à la date de décision, sans qu'aucune observation future puisse modifier le passé ?

Ce lot contrôle la causalité de la transformation. Il ne joint aucune cible et n'entraîne aucun modèle.

## Entrée immuable

- Artifact : `crypto-forecast-binance-funding-history-v1-6288d6a1dca91185`.
- Manifeste SHA-256 : `5d3f4320926c1615e961a17c64ca911fcc35173c4daae6013665e49ebb5fbf48`.
- Sept CSV d'événements et leurs empreintes exactes sont figés dans `config/crypto_forecast_binance_funding_features.json`.
- Période : du 1er juin 2022 au 31 août 2026, soit 1 553 dates et 10 871 lignes instrument-date attendues.

Tout écart d'identifiant, d'empreinte, de schéma, de taille, de nombre d'observations ou de période entraîne un arrêt.

## Temps de décision

Pour la ligne du jour UTC `d`, l'instant de décision est `d + 1 jour à 00:00:00 UTC`. Seuls les règlements tels que `calc_time < instant de décision` peuvent contribuer aux features. Un règlement horodaté exactement au lendemain à 00:00 appartient au lendemain et est exclu.

Les agrégats d'un jour utilisent donc les événements publiés pendant ce jour. Les fenêtres mobiles se terminent au jour courant et ne lisent jamais un jour ultérieur.

## Features gelées

Agrégats du jour : somme, moyenne, dernière valeur, minimum, maximum, écart-type population, part strictement positive et intervalle moyen en heures.

Historique : sommes sur 3, 7 et 30 jours ; moyennes des sommes quotidiennes sur 7 et 30 jours ; changements de somme à 1 et 7 jours ; z-score de la somme quotidienne sur 30 jours.

- Une fenêtre n'est renseignée que lorsqu'elle possède tous ses jours.
- Le z-score vaut zéro si les 30 sommes quotidiennes sont constantes.
- Les 29 premiers jours restent explicitement non éligibles au modèle.
- À partir du 30e jour, les 16 features doivent être finies et présentes.
- Aucun centrage, mise à l'échelle, remplissage, interpolation ou winsorisation n'est autorisé dans ce lot.

## Test de mutation du futur

Trois coupures sont figées : 31 décembre 2023, 31 décembre 2024 et 31 décembre 2025.

Pour chaque coupure, une copie en mémoire altère fortement tous les taux et intervalles strictement postérieurs, et ajoute une observation contrefactuelle postérieure par jour. Toutes les lignes de features dont la date est antérieure ou égale à la coupure doivent rester exactement identiques, champ par champ. Le nombre de divergences accepté est zéro.

## Critères Go/No-Go

**Go causal** uniquement si :

1. les sept entrées passent la provenance et les empreintes ;
2. les 10 871 lignes sont produites dans l'ordre instrument-date prévu ;
3. les 10 668 lignes après warm-up ont 16 features finies ;
4. aucune cible, normalisation, requête réseau ou mutation de la source n'est utilisée ;
5. les trois tests de mutation donnent zéro divergence sur toutes les lignes antérieures ou égales à leur coupure ;
6. deux constructions indépendantes produisent le même identifiant et les mêmes empreintes de contenu.

Tout autre résultat est **No-Go causal**. Aucun seuil ni définition ne sera modifié après observation.

Un Go causal autorise seulement le lot 5R, qui préenregistrera la comparaison modèle de référence contre modèle enrichi. Il ne constitue pas une preuve de pouvoir prédictif.

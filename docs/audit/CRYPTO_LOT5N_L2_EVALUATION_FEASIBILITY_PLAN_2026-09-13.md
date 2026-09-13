# Plan préenregistré — lot 5N, faisabilité des cibles et de l'évaluation L2

Date de gel : 13 septembre 2026, avant calcul de faisabilité.<br>
Entrée : table 5M revue `crypto-forecast-okx-l2-features-v1-de3238a8dcf7a722`.

## Objectif

Déterminer si les trois journées L2 permettent une évaluation prédictive conforme au protocole causal déjà utilisé par SmartFolio. Ce lot ne crée aucune cible et n'entraîne aucun modèle. Il doit échouer fermé si les journées indépendantes, la durée, l'univers ou les partitions sont insuffisants.

## Unité indépendante gelée

- l'unité temporelle indépendante est la date UTC d'observation ;
- les 96 créneaux intrajournaliers d'une même date ne sont jamais considérés comme 96 journées indépendantes ;
- toutes les lignes d'une date restent dans la même partition afin d'interdire la fuite entre entraînement, calibration, test et confirmation ;
- un couple instrument-date est éligible avec au moins `95/96` créneaux disponibles ;
- la réduction des 25 features intrajournalières en vecteur journalier n'est pas sélectionnée dans ce lot, puisqu'aucun modèle n'est autorisé.

## Cibles gelées pour une éventuelle phase ultérieure

Les horizons restent ceux du mandat et du lot 3 : `7` et `30` jours calendaires.

- prix d'entrée : clôture spot OKX `1Dutc` à la fin de la journée d'observation ;
- prix de sortie : clôture `1Dutc` exactement 7 ou 30 jours calendaires après l'entrée ;
- disponibilité de la cible : strictement après la clôture de la barre de sortie ;
- rendement absolu : `prix_sortie / prix_entrée - 1` ;
- excès défensif : rendement absolu moins le rendement nul du cash USD ;
- rendement relatif à BTC : rendement de l'instrument moins rendement BTC sur les mêmes bornes ;
- labels hausse/surperformance : dérivés seulement si les rendements existent, puis calibrés sur une période chronologique distincte ;
- toute clôture manquante reste indisponible, sans recherche nearest, interpolation ou remplissage.

La série quotidienne cible devra être épinglée par artifact et SHA-256 avant toute matérialisation. Elle n'est pas lue dans ce lot, car l'insuffisance des observations L2 doit être décidée indépendamment des résultats futurs.

## Protocole d'évaluation gelé

Les minimums reprennent sans réduction la configuration causale du lot 3 :

| Partition | Durée calendaire gelée | Dates indépendantes minimales |
|---|---:|---:|
| Entraînement initial | 730 jours | 120 |
| Calibration distincte | 183 jours | 120 |
| Test de développement | 183 jours | 120 |
| Confirmation finale gelée | 365 jours | 120 |
| **Total** | **1 461 jours hors purges** | **480 dates distinctes** |

Trois frontières sont purgées de `30` jours, soit une étendue calendaire minimale de `1 551` jours. Le panel de rotation exige au moins cinq instruments éligibles. Aucun split aléatoire, aucune normalisation globale et aucune sélection sur la confirmation finale ne sont permis.

## Contrôles à exécuter

1. Vérifier l'artifact 5M, son résultat, sa table, sa taille et leurs SHA-256.
2. Vérifier les 864 clés de grille, leur ordre, leur disponibilité et les neuf groupes instrument-date.
3. Compter les dates UTC distinctes et les couples instrument-date éligibles sans gonfler l'échantillon avec les lignes intrajournalières.
4. Mesurer l'étendue calendaire inclusive et le nombre d'instruments.
5. Comparer ces valeurs aux seuils gelés et publier chaque contrôle en Pass/Fail.
6. Produire deux artifacts indépendants de même identité.

## Critères Go/No-Go

Le lot est **Go pour une future évaluation** seulement si tous les contrôles d'intégrité passent et si :

- au moins `480` dates UTC observées indépendantes sont présentes, soit 120 par partition ;
- leur étendue calendaire atteint `1 551` jours ;
- chaque partition peut recevoir le nombre gelé de dates sans partager une journée avec une autre partition ;
- au moins cinq instruments sont disponibles pour le panel ;
- aucune ligne manquante n'est fabriquée.

Sinon, la décision est **No-Go prédictif**. Aucun seuil ne sera abaissé après observation et les 863 lignes ne seront pas présentées comme 863 échantillons indépendants.

## Hors périmètre

- matérialisation ou consultation des rendements futurs ;
- agrégation ou sélection des features ;
- modèle, calibration, backtest, allocation ou signal ;
- téléchargement, collecte récurrente, API, interface ou production ;
- ordre, secret, compte, dérivé, levier, suppression, commit, push ou déploiement.

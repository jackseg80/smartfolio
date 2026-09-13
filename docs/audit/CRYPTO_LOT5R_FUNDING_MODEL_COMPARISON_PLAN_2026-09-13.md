# Lot 5R — Plan gelé de comparaison prédictive du funding

Date de gel : 13 septembre 2026. Ce document précède tout entraînement 5R et toute lecture de performance.

## Question

Les 16 features causales de funding du lot 5Q améliorent-elles réellement, hors échantillon, la probabilité de surperformance des actifs face à BTC par rapport au même modèle utilisant uniquement les 17 variables historiques de prix et volume ?

La comparaison porte sur la valeur marginale du funding. Architecture, paramètres, lignes, cibles, prétraitement et fenêtres sont identiques entre les deux variantes.

## Entrées immuables

- Dataset prix/cibles : `crypto-forecast-dataset-v1-6f1490cfee2de334`, SHA-256 `9e1d0d9ad6d418e6ce03448bd82b416e19d5ab953a1aa83ea02bc37c1a76d537`.
- Features funding : `crypto-forecast-binance-funding-features-v1-f32e0489dbb258fd`, table SHA-256 `fc1b78a7ba21919f676d82e952c6dc3dec2566d4e00d23dadfddeee6ce19d313`.
- Actifs : ADA, BTC, ETH, LINK, LTC, SOL et XRP.
- Période commune : du 1er juillet 2022 au 1er août 2026, soit 1 493 dates et 10 451 lignes attendues.
- Cibles : surperformance face à BTC à 7 et 30 jours, déjà calculée causalement dans le dataset validé.

Une jointure exacte `(date, actif)` est exigée. Une date incomplète, un doublon, une empreinte différente ou une ligne non finie entraîne un arrêt ; aucune ligne propre à une seule variante n'est autorisée.

## Variantes et modèle

- `baseline` : 17 variables historiques déjà utilisées aux lots 3/3b — prix, volatilité, drawdown, rendement relatif à BTC, volume coté et nombre de transactions.
- `baseline_plus_funding` : les mêmes 17 variables, plus les 16 variables causales 5Q.

Les deux variantes utilisent exclusivement `HistGradientBoostingClassifier` avec `max_depth=3`, `max_iter=100`, `learning_rate=0.05` et `random_state=42`. Ces paramètres sont repris du candidat existant et ne seront pas balayés.

Les variables finies et non constantes sont sélectionnées sur l'entraînement uniquement, puis centrées et réduites avec les statistiques de ce même entraînement. La calibration de Platt utilise uniquement la période de calibration distincte.

## Fenêtres chronologiques gelées

Chaque frontière est séparée par une purge de 30 jours.

| Bloc | Entraînement | Calibration | Test |
|---|---|---|---|
| Développement 1 | 2022-07-01 → 2023-06-30 | 2023-07-31 → 2023-10-28 | 2023-11-28 → 2024-05-27 |
| Développement 2 | 2022-07-01 → 2023-12-29 | 2024-01-29 → 2024-04-27 | 2024-05-28 → 2024-11-25 |
| Confirmation finale | 2022-07-01 → 2025-03-04 | 2025-04-04 → 2025-07-02 | 2025-08-02 → 2026-08-01 |

La confirmation finale ne sert jamais à choisir modèle, variable ou seuil.

## Métriques et seuils préenregistrés

Le Brier mesure la qualité probabiliste globale, plus bas étant meilleur. L'AUC quotidienne mesure le classement transversal des sept actifs, plus haut étant meilleur.

Un horizon passe uniquement si les six conditions suivantes sont toutes vraies :

1. amélioration moyenne du Brier sur les deux développements ≥ 0,002 ;
2. amélioration du Brier final ≥ 0,002 ;
3. amélioration moyenne de l'AUC quotidienne en développement ≥ 0,01 ;
4. amélioration de l'AUC quotidienne finale ≥ 0,01 ;
5. aucune période de développement ne dégrade le Brier de plus de 0,001 ;
6. aucune période de développement ne dégrade l'AUC quotidienne de plus de 0,01.

Le lot obtient un **Go prédictif** seulement si les horizons 7 et 30 jours passent tous les deux. Un seul horizon positif, une amélioration seulement finale ou une AUC meilleure avec un Brier dégradé produisent un **No-Go prédictif**.

Accuracy, log-loss et erreur de calibration sont conservés comme diagnostics mais ne peuvent pas renverser la décision.

## Sorties et limites

- résultats par fenêtre, variante et horizon ;
- prédictions datées permettant l'audit ;
- métadonnées du prétraitement prouvant qu'il est ajusté sur l'entraînement ;
- décision mécanique par critère ;
- deux exécutions reproductibles.

Aucun coût, portefeuille ou seuil d'allocation n'est testé ici. Le lot 5S économique et une démonstration locale sur le port 8082 ne sont autorisés qu'après un Go prédictif 5R. Aucun réseau, ordre ou changement de production n'est inclus.

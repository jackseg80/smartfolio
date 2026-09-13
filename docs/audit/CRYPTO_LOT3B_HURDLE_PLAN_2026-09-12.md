# Plan préenregistré — lot 3b hurdle et allocation hybride

Date de gel : 12 septembre 2026, avant exécution de l'expérience.

## Objectif

Tester une seule hypothèse nouvelle : les probabilités calibrées de hausse ou de surperformance contiennent une information de classement que la régression directe par RMSE écrase vers zéro. Convertir cette probabilité en rendement attendu peut rendre le signal économiquement testable sans sélectionner un modèle sur la confirmation finale.

## Méthode gelée

Pour chaque fenêtre, tâche et classifieur déjà configurés :

1. ajuster le classifieur sur l'entraînement et calibrer sa probabilité sur la calibration distincte, comme au lot 3 ;
2. calculer sur l'entraînement uniquement la moyenne du rendement cible quand le label vaut 1 et quand il vaut 0 ;
3. produire le rendement hurdle attendu : `p × moyenne_positive + (1-p) × moyenne_negative` ;
4. conserver la sélection probabiliste par Brier moyen de développement ;
5. ne pas utiliser la confirmation finale pour choisir le modèle, les seuils ou les politiques.

Aucun nouveau modèle, balayage d'hyperparamètres ou nouvelle feature n'est ajouté.

## Comparaisons économiques

- `forecast` : marché et rotations utilisent la probabilité et le rendement hurdle.
- `hybrid` : le niveau d'exposition utilise la référence BTC/SMA200 5/2 ; seules les rotations utilisent la probabilité et le rendement hurdle.
- `reference` : règles réactives inchangées.
- Benchmarks, cadence hebdomadaire, exécution à la clôture suivante, caps et coûts restent identiques au lot 4.

Le rendement hurdle doit dépasser le coût aller-retour de 0,60 % et la probabilité doit franchir le seuil déjà gelé. Les coûts doublés restent un scénario de sensibilité, sans modification du seuil de décision.

## Critère de décision

L'hypothèse n'est pas retenue si elle reste inactive, si elle dégrade la confirmation finale, si son avantage disparaît aux coûts doublés ou si elle repose seulement sur une période. Une amélioration de classement sans amélioration économique ne suffit pas.

## Fichiers concernés

- `services/forecasting/evaluation.py` et ses tests : rendement hurdle causal.
- `services/forecasting/allocation_backtest.py` et ses tests : variante hybride.
- `scripts/run_crypto_forecast_allocation.py` : comparaison et diagnostics.
- Artifacts isolés sous `outputs/` et rapport séparé ; aucune API, UI ou production.

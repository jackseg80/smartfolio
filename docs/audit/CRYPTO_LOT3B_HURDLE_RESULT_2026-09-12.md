# Résultat lot 3b — rendement hurdle et allocation hybride

Date d'exécution : 12 septembre 2026<br>
Statut : hypothèse rejetée selon le plan préenregistré.<br>
Périmètre : recherche hors ligne uniquement ; aucun ordre réel, branchement UI ou changement de configuration de production.

## Conclusion

La conversion des probabilités calibrées en rendements attendus hurdle rend les rotations altcoins actives sur le développement, contrairement à la régression directe qui restait à zéro. Cette activité ne crée toutefois pas d'avantage économique robuste : elle réduit légèrement le rendement et le Sharpe face au même socle BTC/ETH sans altcoins, tout en ajoutant du turnover et des frais.

Sur la confirmation finale, le signal n'alloue aucun altcoin. Il évite ainsi les pertes supplémentaires des rotations réactives, mais produit exactement le même résultat que le socle sans altcoins. Cette égalité est une abstention correcte, pas une preuve de valeur prédictive.

Le critère préenregistré n'est donc pas satisfait. Les seuils ne sont pas réajustés après lecture de la confirmation finale et le signal ne doit pas être branché sur Rebalance, Execution ou une interface utilisateur.

## Méthode testée

Pour chaque classifieur déjà sélectionné, les moyennes de rendement des classes positive et négative sont calculées exclusivement sur l'entraînement. Le rendement attendu est ensuite :

`probabilité × moyenne_positive + (1 - probabilité) × moyenne_negative`

- `forecast` utilise les prévisions hurdle pour le régime marché et les rotations ;
- `hybrid` conserve le régime BTC/SMA200 réactif et utilise le hurdle uniquement pour les rotations ;
- `reference` conserve toutes les règles réactives du lot 4.

Aucun nouveau modèle, balayage d'hyperparamètres, feature ou seuil n'a été ajouté. La confirmation finale n'a pas servi à la sélection.

## Activation des signaux

| Bloc | Jours | Marché hurdle favorable | Marché hurdle défavorable | Jours avec au moins une rotation admissible | Lignes de rotation admissibles |
|---|---:|---:|---:|---:|---:|
| Développement 2021-07-02 → 2024-07-03 | 1 098 | 0 | 0 | 382 | 1 058 sur 10 980 |
| Confirmation 2025-09-05 → 2026-08-12 | 342 | 0 | 0 | 142 | 214 sur 3 420 |

Le marché reste neutre dans les deux blocs : la variante `forecast` complète demeure donc intégralement en cash. La variante `hybrid` peut tester les rotations parce que son exposition de base vient de la règle SMA200.

## Résultats hybrides aux coûts normaux

| Bloc | Politique | Rendement net | Drawdown maximal | Sharpe | Turnover | Frais | Exposition alt moyenne |
|---|---|---:|---:|---:|---:|---:|---:|
| Développement | Socle 20/60 sans alt | 64,919 % | -29,107 % | 0,8026 | 5,0066 | 1,708 % | 0,000 % |
| Développement | Hybride 20/60, cap alt 30 % | 64,909 % | -28,930 % | 0,8006 | 6,1983 | 2,107 % | 0,502 % |
| Développement | Hybride 20/60, cap alt 60 % | 64,888 % | -28,898 % | 0,7999 | 6,4670 | 2,197 % | 0,602 % |
| Développement | Socle 20/80 sans alt | 88,977 % | -33,477 % | 0,8284 | 6,3913 | 2,213 % | 0,000 % |
| Développement | Hybride 20/80, cap alt 30 % | 88,929 % | -33,139 % | 0,8268 | 7,5958 | 2,624 % | 0,501 % |
| Développement | Hybride 20/80, cap alt 60 % | 88,839 % | -32,942 % | 0,8253 | 8,4002 | 2,900 % | 0,802 % |
| Confirmation | Socle et hybrides 20/60 | -18,477 % | -22,659 % | -1,3983 | 2,1878 | 0,616 % | 0,000 % |
| Confirmation | Socle et hybrides 20/80 | -21,729 % | -26,321 % | -1,3451 | 2,9872 | 0,827 % | 0,000 % |

Sur le développement, les petites améliorations de drawdown ne compensent pas la baisse de rendement, de Sharpe et l'augmentation des frais. Aux coûts doublés, l'écart défavorable devient plus net :

- socle 20/60 : 62,458 %, contre 61,868 % et 61,717 % pour les deux hybrides ;
- socle 20/80 : 85,382 %, contre 84,665 % et 84,132 % pour les deux hybrides.

Sur la confirmation, les variantes hybrides n'effectuent aucune rotation altcoin et restent identiques à leur socle. Les variantes réactives avec altcoins font moins bien, mais cela ne suffit pas à valider le hurdle puisque celui-ci n'apporte aucune amélioration au socle.

## Reproductibilité et intégrité

- Dataset : `crypto-forecast-dataset-v1-6f1490cfee2de334`
- Expérience : `crypto-forecast-experiment-v1-5ab5bb01eabe7468`
- SHA-256 résultats modèle : `79ada19451341f6f919fad5cf50c64d663a5938ec6107770ed9797e4a2110950`
- SHA-256 prédictions : `57a7854fdbe4baf3b998f6ffaa0080f037ef4deb68c72b80be5fa5bd27e1cf3c`
- Allocation : `crypto-forecast-allocation-backtest-v1-88d6d78424cee3ec`
- SHA-256 résultats allocation : `086e0aae44baad2c744d7a89a914230b1fe8c96321548b03cfe55642d2fac8ea`
- SHA-256 quotidien : `bb6c7d956ac86ed5f1e5e96545093aaeece84d01011cf7dd56607d1938c655ea`

Une seconde exécution isolée a reproduit exactement les deux identifiants et les quatre empreintes. Les fichiers correspondent à leurs manifestes, ne contiennent aucune valeur `NaN` ou infinie, et déclarent `real_orders_created=false` et `production_configuration_changed=false`.

Validation du code : 31 tests ciblés réussis ; Ruff et Black conformes. L'unique avertissement vient de la dépréciation Starlette/httpx déjà extérieure à ce lot.

## Décision et suite responsable

L'hypothèse hurdle est rejetée. Un réglage a posteriori des probabilités ou du seuil de coût introduirait un biais de sélection et n'est pas entrepris.

Le prochain lot éventuel doit apporter une information réellement nouvelle et datée — par exemple liquidité/carnet d'ordres, données on-chain ou sentiment disponibles historiquement, ou validation multi-exchange — avec un nouveau protocole préenregistré. Tant qu'un tel signal n'améliore pas plusieurs périodes après coûts, aucun test utilisateur de l'interface ni intégration en production n'est justifié.

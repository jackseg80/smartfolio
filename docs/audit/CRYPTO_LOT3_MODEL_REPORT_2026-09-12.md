# [ARCHIVE] Rapport lot 3 — première exécution sur cache historique

> Ce rapport décrit l'expérience initiale `0fd2f7e401f0931c`. Il est remplacé par `CRYPTO_LOT3_REFRESH_REPORT_2026-09-12.md`, fondé sur l'acquisition publique datée, dix actifs et les variables de volume.

Date d'exécution : 12 septembre 2026<br>
Périmètre : comparaison hors ligne des modèles, sans API, sans UI, sans ordre et sans modification de la production.

## Conclusion

Le lot 3 est techniquement terminé, mais son résultat prédictif est négatif sur les données actuellement disponibles.

Sur les six tâches évaluables (BTC, ETH et ETH relatif à BTC, à 7 et 30 jours), la référence constante à zéro obtient la plus faible RMSE moyenne sur les fenêtres de développement. Pour les probabilités de hausse ou de surperformance, la prévalence calculée sur l'entraînement seul obtient le meilleur Brier score moyen. Ces choix ont été gelés avant la confirmation finale d'un an.

Ce résultat interdit de présenter Ridge, la régression logistique ou le gradient boosting comme une amélioration validée. Il ne démontre pas qu'aucune prévision crypto n'est possible : il démontre que les variables de prix bornées, les modèles et l'historique figés de cette expérience ne battent pas les références simples selon le protocole retenu.

Les rotations entre altcoins et entre groupes restent indisponibles. Seuls BTC et ETH satisfont la longueur d'historique inchangée, contre cinq actifs requis pour une comparaison transversale et trois groupes requis pour une rotation de groupes.

## Artefacts et reproductibilité

- Dataset figé : `crypto-forecast-dataset-v1-618b482c94e44311`
- SHA-256 du dataset : `57f8aaf73eb8d2eb5fe9976d94335948d3be9e4a738b8ee6c44a60d1c5683ded`
- Expérience : `crypto-forecast-experiment-v1-0fd2f7e401f0931c`
- SHA-256 de la configuration : `25751c29693dcadcd0a122021ac00f9206b1d313f6c76bead0c73372a1b328ee`
- SHA-256 du moteur d'évaluation : `9ad8367af59ef108828a32ef57e0d831071e7127c2d8afc4cf23ab24425eddeb`
- SHA-256 du lanceur : `f0f866655fc8cf3cdc7ad5a373d2ed03642b4abeb54f039eb3f9059976bb4a19`
- Environnement : Python 3.13.13, NumPy 2.4.0, pandas 2.3.3 et scikit-learn 1.8.0
- SHA-256 des résultats : `e7bec0e918ba68aea31675f14f1b8ede151cf7d4ec8e4d6aeadc8021bf7ae164`
- SHA-256 des prédictions : `2c58fbae64409c9937a7243983927ed7114ece649dd71260c9bf0f1f26a31be6`

Le manifeste lie le dataset, la configuration, le code d'évaluation, les résultats et les prédictions. Une modification de l'un de ces éléments produit un nouvel identifiant d'expérience au lieu d'écraser silencieusement le résultat.

## Protocole appliqué

- Entraînement initial : minimum 730 jours, puis fenêtre expansive.
- Calibration : 183 jours distincts pour les probabilités apprises.
- Test de développement : 183 jours.
- Progression : 183 jours.
- Purge : 30 jours entre entraînement et calibration, puis entre calibration et test.
- Confirmation finale : les 365 derniers jours complets, réservés avant toute sélection.
- Nombre de fenêtres de développement : sept pour chacune des six tâches. Leur dernier test se termine avant la période de calibration finale ; la sélection est donc gelée avant cette calibration et avant la confirmation finale.
- Lignes causales utilisables : 2 894 à 7 jours et 2 871 à 30 jours.
- Prétraitement : sélection, moyenne et écart-type ajustés sur l'entraînement seul.
- Calibration : Platt sur la période de calibration seulement. La prévalence de référence reste calculée sur l'entraînement seulement.

Les dates de confirmation finale sont du 4 février 2025 au 3 février 2026 pour l'horizon 7 jours, et du 12 janvier 2025 au 11 janvier 2026 pour l'horizon 30 jours. Cette différence vient de la disponibilité réelle des cibles futures, pas d'un découpage choisi après observation des scores.

## Candidats comparés

Régression : zéro, momentum au même horizon, Ridge (`alpha=1`) et gradient boosting histogramme (`max_depth=3`, `max_iter=100`, `learning_rate=0.05`, graine 42).

Classification : prévalence de l'entraînement, régression logistique (`C=1`) et gradient boosting histogramme avec les mêmes paramètres bornés. Aucun balayage massif de paramètres n'a été effectué.

## Résultats de sélection sur les fenêtres de développement

| Tâche | Horizon | RMSE référence zéro | Meilleur modèle appris | Écart appris vs référence | Brier prévalence | Meilleur classifieur appris | Écart appris vs référence |
|---|---:|---:|---|---:|---:|---|---:|
| Rendement BTC | 7 j | 0,090309 | Ridge, 0,093237 | +3,24 % | 0,252088 | Gradient boosting, 0,265811 | +5,44 % |
| Rendement BTC | 30 j | 0,210680 | Ridge, 0,222651 | +5,68 % | 0,257697 | Gradient boosting, 0,329267 | +27,77 % |
| Rendement ETH | 7 j | 0,109554 | Ridge, 0,115820 | +5,72 % | 0,252527 | Gradient boosting, 0,258638 | +2,42 % |
| Rendement ETH | 30 j | 0,254702 | Ridge, 0,297017 | +16,61 % | 0,261890 | Logistique, 0,280643 | +7,16 % |
| ETH relatif à BTC | 7 j | 0,063452 | Ridge, 0,070225 | +10,68 % | 0,252901 | Logistique, 0,255833 | +1,16 % |
| ETH relatif à BTC | 30 j | 0,139260 | Ridge, 0,179443 | +28,85 % | 0,259674 | Logistique, 0,282882 | +8,94 % |

Un écart positif signifie ici une erreur supérieure, donc une dégradation. Les résultats JSON contiennent aussi MAE, précision directionnelle, corrélation de rang temporelle, log-loss, précision au seuil 0,5 et erreur de calibration à dix intervalles pour chaque modèle et chaque fenêtre.

## Confirmation finale des modèles gelés

| Tâche | Horizon | Modèle de rendement gelé | RMSE finale | Modèle probabiliste gelé | Brier final | Erreur de calibration finale |
|---|---:|---|---:|---|---:|---:|
| Rendement BTC | 7 j | Zéro | 0,055921 | Prévalence entraînement | 0,250709 | 0,026949 |
| Rendement BTC | 30 j | Zéro | 0,109293 | Prévalence entraînement | 0,257155 | 0,100044 |
| Rendement ETH | 7 j | Zéro | 0,107375 | Prévalence entraînement | 0,250681 | 0,026419 |
| Rendement ETH | 30 j | Zéro | 0,253257 | Prévalence entraînement | 0,260049 | 0,128767 |
| ETH relatif à BTC | 7 j | Zéro | 0,072381 | Prévalence entraînement | 0,249547 | 0,019352 |
| ETH relatif à BTC | 30 j | Zéro | 0,183732 | Prévalence entraînement | 0,244773 | 0,021211 |

La confirmation finale décrit le comportement des modèles sélectionnés auparavant. Elle n'a pas été utilisée pour modifier les modèles, leurs paramètres, les variables ou les politiques.

## Limites qui empêchent une conclusion économique

1. Le cache historique s'arrête au 10 février 2026 alors que l'expérience est exécutée le 12 septembre 2026.
2. Le format historique ne fournit pas le fournisseur de chaque observation, les volumes datés, ni un registre complet des cotations et radiations. Un biais de survivants reste possible.
3. Le lot 3 mesure la qualité prédictive. Il ne simule pas encore une allocation, une exécution à la clôture suivante, les abstentions, les frais de 0,20 % et le glissement de 0,10 % par montant négocié, ni le scénario de coûts doublés. Ces éléments appartiennent au lot 4.
4. Les classements transversaux d'actifs et de groupes sont indisponibles faute d'au moins cinq actifs et trois groupes satisfaisant le protocole long. Aucune métrique de classement n'est inventée pour ces sorties.
5. Le cycle Bitcoin, l'on-chain et le sentiment ne sont pas inclus : leur historique disponible au moment de chaque décision n'est pas encore suffisamment prouvé pour une comparaison causale.

## Décision proposée

Ne pas exposer ces modèles dans l'interface et ne pas utiliser leurs sorties pour un rééquilibrage. Le prochain travail utile n'est pas une intégration visuelle : c'est soit compléter des historiques datés et traçables pour rendre les rotations évaluables, soit ouvrir un lot 4 explicitement expérimental où les références simples restent le centre de la comparaison économique. Dans les deux cas, la production 8080 doit rester inchangée et tout futur test manuel doit se faire sur un serveur isolé, par exemple le port 8082.

## Validation technique

- 18 tests ciblés des lots 2 et 3 passent.
- Analyse statique Ruff : aucun problème.
- Formatage Black : conforme.
- Aucune route API, page HTML, configuration de production ou logique d'ordre n'a été modifiée.

# Lot 5R — Résultat de la comparaison prédictive du funding

Date : 13 septembre 2026<br>
Verdict : **NO_GO_PREDICTIVE_FUNDING**

## Conclusion

L'ajout des 16 features causales de funding du lot 5Q au modèle de référence n'apporte pas une amélioration hors échantillon suffisamment robuste. Aucun des deux horizons préenregistrés ne passe l'ensemble des six critères. Le lot 5S économique et une démonstration locale sur le port 8082 ne sont donc pas autorisés.

Ce No-Go ne remet pas en cause la qualité ou la causalité des données 5P–5Q. Il signifie que, dans ce protocole fixe, leur valeur prédictive marginale n'est pas démontrée.

## Protocole exécuté

Le plan a été gelé dans `CRYPTO_LOT5R_FUNDING_MODEL_COMPARISON_PLAN_2026-09-13.md` avant toute lecture de performance.

- mêmes 10 451 lignes et mêmes cibles pour les deux variantes ;
- sept actifs, du 1er juillet 2022 au 1er août 2026 ;
- modèle `HistGradientBoostingClassifier` identique et sans balayage de paramètres ;
- variante de référence avec 17 features prix/volume ;
- variante enrichie avec les mêmes 17 features plus les 16 features funding ;
- deux fenêtres de développement et une confirmation finale de 365 jours ;
- purge chronologique de 30 jours ;
- sélection, centrage et réduction appris sur l'entraînement uniquement ;
- calibration de Platt sur une période séparée ;
- confirmation finale jamais utilisée pour modifier le modèle ou les seuils.

## Décision par horizon

Une valeur positive indique que la variante enrichie est meilleure. Chaque horizon devait satisfaire simultanément les six critères préenregistrés.

| Critère | Seuil | 7 jours | Passe 7 j | 30 jours | Passe 30 j |
|---|---:|---:|:---:|---:|:---:|
| Amélioration Brier moyenne, développement | ≥ 0,002 | +0,001729 | Non | +0,007652 | Oui |
| Amélioration Brier, confirmation finale | ≥ 0,002 | -0,000740 | Non | -0,003977 | Non |
| Amélioration AUC quotidienne moyenne, développement | ≥ 0,010 | -0,007027 | Non | -0,014749 | Non |
| Amélioration AUC quotidienne, confirmation finale | ≥ 0,010 | +0,021021 | Oui | -0,021939 | Non |
| Stabilité Brier sur chaque développement | dégradation ≤ 0,001 | échec | Non | conforme | Oui |
| Stabilité AUC sur chaque développement | dégradation ≤ 0,010 | conforme | Oui | échec | Non |

Résultat mécanique : **0 horizon sur 2 passe**.

## Métriques détaillées

| Horizon | Fenêtre | Brier référence | Brier enrichi | Gain Brier | AUC référence | AUC enrichie | Gain AUC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 j | Développement 1 | 0,207448 | 0,208866 | -0,001418 | 0,640949 | 0,634327 | -0,006623 |
| 7 j | Développement 2 | 0,231262 | 0,226384 | +0,004877 | 0,673779 | 0,666348 | -0,007431 |
| 7 j | Confirmation finale | 0,211671 | 0,212411 | -0,000740 | 0,639821 | 0,660842 | +0,021021 |
| 30 j | Développement 1 | 0,185910 | 0,177462 | +0,008448 | 0,624946 | 0,608606 | -0,016340 |
| 30 j | Développement 2 | 0,250717 | 0,243861 | +0,006855 | 0,674025 | 0,660867 | -0,013158 |
| 30 j | Confirmation finale | 0,222437 | 0,226414 | -0,003977 | 0,643594 | 0,621655 | -0,021939 |

Le profil est contradictoire, pas simplement faible : à 7 jours le classement final progresse, mais la qualité probabiliste finale se dégrade et les développements ne confirment pas le gain d'AUC. À 30 jours le Brier progresse en développement, tandis que le classement se dégrade, puis les deux métriques reculent sur la confirmation finale. Ce comportement ne satisfait pas une exigence de robustesse.

## Reproductibilité et preuves

Deux exécutions indépendantes produisent des résultats et des prédictions identiques.

- artifact : `crypto-forecast-funding-model-comparison-v1-f6804348361648f4` ;
- résultat SHA-256 : `a595cc00e11b5c4eb3a0495b31612b0b330093840b272bb17becfa9ea837f57c` ;
- prédictions SHA-256 : `ff20afbe29b127d104666208d32c458c753bc80445a69215cad64e410ad07fdb` ;
- configuration SHA-256 : `e5bfa48204132ac1e2808b5182a0218df15ff076d2a65039af4dc7f6bc66e80b` ;
- code de comparaison SHA-256 : `de71cb3970a6399cf3fb5f523e9b5c14367fd27dc7204e73cfad205b0c50c411` ;
- 20 412 prédictions datées ;
- égalité exacte des résultats et des prédictions entre les deux exécutions.

La validation finale compte 129 tests unitaires de prévision crypto réussis. Ruff et Black sont conformes sur les trois nouveaux fichiers 5R. L'unique avertissement Starlette/httpx est extérieur à ce chantier.

## Portée de la décision

- Les artifacts 5P et 5Q restent des preuves de données compactes, complètes et causales.
- Les features funding ne doivent pas être intégrées au produit ou à l'allocation sur la base de cette expérience.
- Aucun seuil n'est retouché après observation de la confirmation finale.
- Aucun test économique 5S, aucune interface locale 8082 et aucun test utilisateur ne sont nécessaires.
- Aucun réseau de production, ordre réel, déploiement ou réglage de production n'a été utilisé ou modifié.

La chaîne de recherche prévue est close sur un résultat négatif mais exploitable. Une reprise future demanderait une hypothèse ou une source d'information réellement nouvelle, avec un nouveau plan préenregistré ; elle ne doit pas recycler ces résultats pour ajuster les seuils a posteriori.

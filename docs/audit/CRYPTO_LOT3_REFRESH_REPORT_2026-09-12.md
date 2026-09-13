# Rapport lot 3 actualisé — prévisions crypto causales enrichies

> **Rapport technique actuel du lot 3.** Pour l'état global, la décision économique et le lot suivant, consulter `CRYPTO_FORECAST_RESEARCH_STATUS_2026-09-13.md`.

Date d'exécution : 12 septembre 2026<br>
Périmètre : acquisition publique en lecture seule, dataset causal et comparaison hors ligne. Aucun service, ordre ou paramètre de production n'a été modifié.

## Conclusion

Le lot 3 est techniquement reproductible, mais il ne valide pas encore une prévision directement exploitable pour une allocation.

- Pour les rendements absolus BTC et ETH à 7 et 30 jours, la référence constante à zéro reste meilleure en RMSE moyenne que les modèles appris.
- Certains classifieurs apprennent un classement relatif hors entraînement. Le signal le plus stable concerne les groupes face à BTC.
- Les régressions de rendement relatif restent faibles ou instables. À 30 jours pour les actifs, la référence zéro est sélectionnée par RMSE.
- Les variables de volume datées améliorent légèrement certaines probabilités, notamment la rotation de groupes à 30 jours, sans résoudre l'estimation du rendement attendu.

Une bonne AUC de classement ne suffit pas : sans rendement estimé positif dépassant les coûts, aucune rotation ne doit être proposée.

## Acquisition et dataset

- Artifact d'acquisition : `crypto-forecast-history-acquisition-v1-7d67c83f07ff241e`
- Fournisseur déclaré : `binance_spot_public_market_data`, base `https://data-api.binance.vision`
- Lecture publique sans identifiant ni clé API, du 1er janvier 2017 au 11 septembre 2026
- Cotation USDT ; aucune équivalence avec un USD sans risque n'est supposée
- Univers : BTC, ETH, SOL, ADA, XRP, LINK, LTC, BCH, BNB et DOT
- De 2 216 à 3 313 observations par actif, sans jour calendaire manquant après la première cotation
- Quatre bougies raccourcies du 8 février 2018 sont conservées et signalées
- OHLCV normalisés en conservant les décimales du fournisseur ; payloads JSON bruts non archivés

Le dataset contient 46 765 lignes : 28 891 actifs, 14 561 groupes et 3 313 marché. La couverture complète des features est de 89,9519 %, 89,2040 % et 93,9934 %. Les cibles 30 jours couvrent environ 98,96 % à 99,09 % des lignes.

Les quatre variables de volume sont rétrospectives : variation du volume coté à 7 et 30 jours, rapport à sa moyenne 30 jours et rapport du nombre de transactions à sa moyenne 30 jours. Elles utilisent seulement la clôture de la date de décision et son passé.

## Protocole

- Entraînement expansif initial d'au moins 730 jours.
- Calibration distincte de 183 jours, tests de 183 jours, progression de 183 jours.
- Purge de 30 jours aux deux frontières.
- Confirmation finale de 365 jours par tâche, jamais utilisée pour la sélection.
- Prétraitement appris sur l'entraînement uniquement.
- Candidats bornés : zéro, momentum, Ridge, régression logistique et gradient boosting histogramme.
- Sélection principale : RMSE pour les rendements, Brier pour les probabilités.
- Classement secondaire : corrélation de rang et AUC quotidiennes transversales.

## Résultats de rotation

| Cible | Horizon | Modèle probabiliste sélectionné | Brier développement | AUC développement | Brier final | AUC finale |
|---|---:|---|---:|---:|---:|---:|
| Actif vs BTC | 7 j | Gradient boosting | 0,227422 | 0,6314 | 0,221756 | 0,5776 |
| Actif vs BTC | 30 j | Gradient boosting | 0,227003 | 0,6370 | 0,215161 | 0,6144 |
| Groupe vs BTC | 7 j | Gradient boosting | 0,203701 | 0,7184 | 0,199621 | 0,7171 |
| Groupe vs BTC | 30 j | Gradient boosting | 0,219922 | 0,7280 | 0,199123 | 0,7057 |
| Actif L1 vs groupe | 7 j | Prévalence entraînement | 0,247448 | 0,5000 | 0,250030 | 0,5000 |
| Actif L1 vs groupe | 30 j | Prévalence entraînement | 0,245633 | 0,5000 | 0,255456 | 0,5000 |

Pour les lignes L1, une logistique candidate classe parfois mieux que le hasard, mais elle n'est pas retenue par le Brier de développement. Les corrélations finales des régressions sont généralement faibles ou négatives ; la candidate Ridge L1 à 30 jours atteint 0,1795 sans être sélectionnée par RMSE.

## Artifacts reproductibles

- Dataset : `crypto-forecast-dataset-v1-6f1490cfee2de334`
- SHA-256 dataset : `9e1d0d9ad6d418e6ce03448bd82b416e19d5ab953a1aa83ea02bc37c1a76d537`
- Expérience : `crypto-forecast-experiment-v1-e9745a3d6fbd445a`
- SHA-256 configuration : `05107a247b5441f411a8e867acb992bb12ca53a40f4cc9e21aaee18ba06a2183`
- SHA-256 moteur : `ea755faba913a9058ebd6542bfe1866d69045879ab68791aae385d4bc6b03dd6`
- SHA-256 résultats : `e564b12d474c7f350881976bc65a78bc8a4fbc5bb04e5bfaaadc74b95bedd412`
- SHA-256 prédictions : `2701a026d80ecd22fc3e70eee48fd94107621bfee7bee8271a1ca22bb312cd89`

Une seconde exécution isolée a produit exactement le même identifiant et les mêmes hashes de sortie.

## Limites et décision

L'univers vient de la première observation Binance et d'un snapshot actuel `TRADING`, pas d'un registre historique complet des cotations. Les groupes utilisent la taxonomie actuelle. USDT n'est pas traité comme une preuve de rendement sans risque. Le volume coté et le nombre de transactions ne prouvent pas la profondeur du carnet ni le coût personnel. Cycle, on-chain et sentiment restent exclus faute d'historique causal prouvé.

Les probabilités de groupes peuvent rester un diagnostic secondaire. Elles ne constituent pas une recommandation et doivent passer le verrou économique du lot 4.

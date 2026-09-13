# État canonique — recherche crypto SmartFolio

Dernière mise à jour : 13 septembre 2026<br>
Périmètre : lots 0 à 5R terminés ; chaîne de recherche close sur un No-Go prédictif du funding.<br>
Statut produit : chaîne sécurisée en production, capacité prédictive non démontrée.

## Résumé décisionnel

SmartFolio dispose maintenant d'une chaîne crypto plus sûre, traçable et cohérente pour le portefeuille spot. Les lots 0 et 1 ont corrigé la provenance des données, la comptabilité du backtest, les propositions d'allocation et les garde-fous d'exécution. Ces corrections ont été validées puis promues en production le 12 septembre 2026.

Les lots de recherche 2 à 4 restent isolés de l'application :

- le dataset causal et reproductible est disponible ;
- certaines probabilités classent correctement les groupes ou actifs face à BTC ;
- les modèles ne prédisent pas assez bien les rendements absolus ou relatifs pour produire un avantage économique robuste après frais ;
- le test hurdle préenregistré a été rejeté ;
- les features causales de funding n'améliorent pas robustement le modèle de référence sur les horizons 7 et 30 jours ;
- aucune prévision expérimentale ne doit être exposée dans Rebalance, Execution ou une nouvelle interface.

La décision actuelle est donc : **conserver en production les sécurisations des lots 0 et 1, mais ne pas utiliser les modèles des lots 2 à 4 pour piloter une allocation réelle**.

## Ce qui est en production

Les changements fonctionnels des lots 0 et 1 ont été intégrés à `main` et déployés sur le port 8080 depuis la base validée `78a7aca`, avec la clôture documentaire `b656d40`.

Ils couvrent notamment :

- l'identité utilisateur et la source explicites ;
- la référence réelle du portefeuille et sa provenance ;
- le Risk Score positif de robustesse, jamais inversé ;
- la séparation entre cible théorique, proposition réalisable et exécution ;
- le respect intégral du budget défensif ;
- des propositions datées, expirables et liées au snapshot ;
- un backtest par quantités, cash, prix propres à chaque actif et coûts nets ;
- l'abstention lorsque des données ou sorties vérifiées sont indisponibles.

Ces éléments rendent le système plus sûr. Ils ne prouvent pas qu'une stratégie bat une référence de marché.

## État des lots

| Lot | Objet | État | Décision |
|---|---|---|---|
| 0 | Référence personnelle, identité, provenance et contrats | Terminé et en production | Conservé |
| 1 | Allocation, proposition, comptabilité et garde-fous | Terminé et en production | Conservé |
| 2 initial | Dataset causal construit depuis l'ancien cache | Archivé comme étape intermédiaire | Remplacé par l'acquisition publique actualisée |
| 2 actualisé | Historique Binance public, OHLCV et volume datés | Terminé hors ligne | Base actuelle de recherche |
| 3 initial | Première comparaison de modèles | Archivé | Remplacé par le lot 3 actualisé |
| 3 actualisé | Modèles causaux et probabilités calibrées | Terminé hors ligne | Diagnostic seulement |
| 4 | Traduction des prévisions en allocations après coûts | Terminé hors ligne | Aucun avantage démontré |
| 3b | Rendement hurdle et allocation hybride | Terminé hors ligne | Hypothèse rejetée |
| 5A | Données quotidiennes multi-plateformes | Terminé hors ligne | OKX validé comme source de contrôle |
| 5B | Pilote de liquidité L2 | Terminé hors ligne | Deltas rejetés ; snapshots autonomes à étudier séparément |
| 5C | Pilote L2 snapshot-only | Terminé hors ligne | Go technique limité à SOL-USDT sur une journée |
| 5D | Couverture et coût L2 multi-périodes | Terminé hors ligne | 42/42 métadonnées valides ; échantillon borné à 1 126,78 MB |
| 5E | Échantillon snapshot-only multi-actifs | Terminé hors ligne | No-Go strict : cadence historique hétérogène malgré 8 927 snapshots valides |
| 5F | Grille causale après-borne | Terminé hors ligne | No-Go : 814/864 créneaux disponibles, seuil par série non atteint |
| 5G | Alignement causal symétrique | Terminé hors ligne | Go technique : 863/864 créneaux, aucune valeur fabriquée |
| 5H | Faisabilité d'un historique L2 continu | Terminé hors ligne | Archives locales No-Go ; collecte prospective Go matériel |
| 5I | Noyau du collecteur prospectif L2 | Terminé hors ligne | Go technique ; aucun service permanent démarré |
| 5J | Déclenchement L2 aligné et mono-instance | Terminé, essai public borné | Go opérationnel unique : démarrage +14 ms, 3/3 captures |
| 5K | Extraction historique L2 progressive | Terminé hors ligne | Go technique : 288 carnets, 544,81 MB vers 1,08 MB |
| 5L | Corpus L2 compact multi-périodes | Terminé hors ligne | Go technique : 863/864 carnets, 1,18 GB vers 2,57 MB |
| 5M | Features L2 causales instantanées | Terminé hors ligne | Go technique : 25 features, 864 lignes, zéro divergence avec 5G |
| 5N | Faisabilité des cibles et de l'évaluation L2 | Terminé hors ligne | No-Go prédictif : 3 dates indépendantes sur 480 requises |
| 5O | Faisabilité de signaux historiques compacts | Terminé, lecture seule | Funding Binance prioritaire en Go pilote ; Coin Metrics en réserve |
| 5P | Acquisition historique du funding Binance | Terminé hors ligne | Go qualité : 357 archives vérifiées, 7 actifs et 1 553 jours complets |
| 5Q | Features quotidiennes de funding | Terminé hors ligne | Go causal : 16 features, 10 871 lignes et zéro divergence future-mutation |
| 5R | Comparaison prédictive du funding | Terminé hors ligne | No-Go : 0 horizon sur 2 passe tous les critères gelés |

## Base de recherche actuellement valable

### Données

- Fournisseur : données publiques Binance Spot, sans identifiant ni clé.
- Univers : BTC, ETH, SOL, ADA, XRP, LINK, LTC, BCH, BNB et DOT, cotés en USDT.
- Période maximale : du 1er janvier 2017 au 11 septembre 2026, selon la date de cotation de chaque actif.
- Dataset : 46 765 lignes, avec prix, volumes cotés et nombre de transactions rétrospectifs.
- Artifact : `crypto-forecast-dataset-v1-6f1490cfee2de334`.
- SHA-256 : `9e1d0d9ad6d418e6ce03448bd82b416e19d5ab953a1aa83ea02bc37c1a76d537`.

Les groupes reposent encore sur la taxonomie actuelle et l'univers historique n'est pas un registre complet de listings/delistings. USDT n'est pas assimilé à un actif sans risque.

### Modèles

Les comparaisons utilisent des découpages chronologiques purgés, un prétraitement appris sur l'entraînement uniquement, une calibration séparée et une confirmation finale qui ne sert pas à la sélection.

- Les références constantes restent meilleures pour les rendements absolus BTC et ETH à 7 et 30 jours.
- Les probabilités de groupes face à BTC atteignent une AUC finale d'environ 0,72 à 7 jours et 0,71 à 30 jours.
- Le classement relatif ne se transforme pas en rendement net robuste.
- Une AUC positive ne suffit jamais à autoriser une rotation si le rendement attendu ne couvre pas les frais.

### Allocation

Le lot 4 simule une décision à la clôture, une exécution à la clôture suivante, une revue hebdomadaire et des coûts de 0,30 % par montant négocié, avec un scénario doublé.

La prévision complète reste en cash parce que le signal marché ne franchit pas les seuils. La variante hybride utilise le régime réactif BTC/SMA200 et les prévisions uniquement pour les rotations. Elle devient active pendant le développement, mais ajoute du turnover et des frais sans améliorer le rendement ou le Sharpe face au même socle sans altcoins. Sur la confirmation finale, elle n'alloue aucun altcoin et reste identique au socle.

## Preuves et reproductibilité

Les artifacts finaux utiles sont :

| Élément | Identifiant | Empreinte principale |
|---|---|---|
| Historique public | `crypto-forecast-history-acquisition-v1-7d67c83f07ff241e` | Manifeste d'acquisition |
| Dataset actuel | `crypto-forecast-dataset-v1-6f1490cfee2de334` | `9e1d0d9a…a76d537` |
| Modèles avec hurdle | `crypto-forecast-experiment-v1-5ab5bb01eabe7468` | résultats `79ada194…110950` |
| Prédictions hurdle | même artifact | `57a7854f…1cf3c` |
| Allocation hurdle | `crypto-forecast-allocation-backtest-v1-88d6d78424cee3ec` | résultats `086e0aae…fac8ea` |
| Résultats quotidiens | même artifact | `bb6c7d95…655ea` |
| Snapshots autonomes | `crypto-forecast-okx-l2-snapshot-pilot-v1-abc1878b6784aeb4` | résultat `6489b2ca…feb54` |
| Couverture L2 | `crypto-forecast-okx-l2-coverage-v1-9b584745ea2d87c6` | résultat `6d885321…f84a0` |
| Échantillon L2 | `crypto-forecast-okx-l2-sample-v1-973d02b919af2d1c` | résultat `b32045a4…a3d6e2` |
| Normalisation L2 | `crypto-forecast-okx-l2-normalization-v1-14438dc7f0ab2f0c` | résultat `6c087991…8f879` |
| Alignement L2 | `crypto-forecast-okx-l2-symmetric-alignment-v1-f201b14037ec95c9` | résultat `c864b55a…a3f6a` |
| Faisabilité collecte L2 | `crypto-forecast-okx-l2-collection-feasibility-v1-7a415096e211446f` | résultat `ce188e14…50b0b` |
| Pilote collecteur L2 | `crypto-forecast-okx-l2-prospective-collector-v1-9bb1d74206beaac2` | résultat `802fae34…1bef` |
| Déclenchement aligné L2 | `crypto-forecast-okx-l2-one-shot-scheduler-v1-3b0bb767f3d9b748` | résultat `a34689f0…65e3` |
| Extraction progressive L2 corrigée | `crypto-forecast-okx-l2-progressive-extraction-v1-8ca057800a4ec6e4` | résultat `ded85d4b…6ae34` |
| Corpus compact L2 corrigé | `crypto-forecast-okx-l2-compact-corpus-v1-1df479332e50e3d1` | résultat `bb597dbc…05e59` |
| Features L2 causales | `crypto-forecast-okx-l2-features-v1-de3238a8dcf7a722` | résultat `4f1d5bb9…071ac` |
| Faisabilité évaluation L2 | `crypto-forecast-okx-l2-evaluation-feasibility-v1-22462ef298ab08b9` | résultat `695137a6…4eff18` |
| Écran signaux compacts | `crypto-forecast-compact-signal-feasibility-v1-9fb8f124741be790` | résultat `aab5e5c8…c81531` |
| Historique funding Binance | `crypto-forecast-binance-funding-history-v1-6288d6a1dca91185` | manifeste `5d3f4320…5fbf48` |
| Features funding causales | `crypto-forecast-binance-funding-features-v1-f32e0489dbb258fd` | résultat `2d2efd92…e2b821` |
| Comparaison prédictive funding | `crypto-forecast-funding-model-comparison-v1-f6804348361648f4` | résultat `a595cc00…37f57c` |

Les chaînes modèle et allocation ont chacune été rejouées dans un répertoire isolé avec les mêmes identifiants et empreintes. Les fichiers correspondent aux manifestes, ne contiennent aucune valeur non finie et déclarent qu'aucun ordre réel ni réglage de production n'a été créé.

Validation technique actuelle : 129 tests unitaires de prévision crypto réussis, Ruff et Black conformes sur le lot 5R. L'avertissement Starlette/httpx observé est extérieur à ce chantier. Le module `coverage` a subi une violation d'accès Windows après une première passe ciblée ; la régression finale a donc été exécutée avec `--no-cov`.

## Ordre de lecture documentaire

1. Le présent fichier est la synthèse canonique et le point de reprise.
2. `CRYPTO_LOTS_0_1_HANDOFF_2026-09-09.md` décrit la chaîne sécurisée aujourd'hui en production.
3. `CRYPTO_LOT3_REFRESH_REPORT_2026-09-12.md` décrit les données publiques actualisées et les résultats ML.
4. `CRYPTO_LOT4_ALLOCATION_REPORT_2026-09-12.md` décrit la traduction économique initiale.
5. `CRYPTO_LOT3B_HURDLE_PLAN_2026-09-12.md` conserve le protocole gelé avant l'expérience hurdle.
6. `CRYPTO_LOT3B_HURDLE_RESULT_2026-09-12.md` contient la décision finale de rejet.

`CRYPTO_LOT2_DATASET_REPORT_2026-09-12.md` et `CRYPTO_LOT3_MODEL_REPORT_2026-09-12.md` restent disponibles comme historique, mais ne décrivent plus la meilleure base de recherche actuelle.

## Lot 5 — conclusion

Le lot 5 doit tester si une information de marché réellement nouvelle améliore les décisions. Il ne doit pas modifier les seuils du lot 3b sur les mêmes données.

Ordre prévu :

1. auditer la disponibilité historique et les licences de données multi-plateformes ;
2. définir un noyau commun d'actifs et de périodes sans fabriquer les jours manquants ;
3. ajouter des mesures causales de liquidité disponibles à la date de décision ;
4. comparer les divergences de prix, volume et liquidité entre plateformes ;
5. préenregistrer les features, modèles, coûts et critères Go/No-Go ;
6. construire un dataset versionné et reproductible ;
7. valider hors échantillon sur plusieurs périodes et plateformes ;
8. ne créer une démonstration locale sur le port 8082 que si un avantage économique robuste subsiste après coûts.

Le premier jalon du lot 5 est un rapport de faisabilité des sources. Aucun abonnement payant, secret, ordre, déploiement ou intégration produit n'est inclus dans ce jalon.

Ce jalon est maintenant documenté dans `CRYPTO_LOT5_FEASIBILITY_REPORT_2026-09-13.md`. Sa décision est un Go limité pour l'acquisition quotidienne OKX/USDT, avec un pilote L2 séparé et borné ; aucun téléchargement L2 massif n'est autorisé.

Le lot 5A quotidien est terminé dans `CRYPTO_LOT5_CROSS_EXCHANGE_RESULT_2026-09-13.md`. Les dix actifs passent les critères de cohérence prix préenregistrés et les deux exécutions sont reproductibles. OKX est donc une source de contrôle recevable, mais ses clôtures apportent presque la même information directionnelle que Binance. Le prochain gain potentiel doit venir de la microstructure L2, pas d'un empilement de prix quotidiens similaires.

Le lot 5B est terminé dans `CRYPTO_LOT5B_L2_PILOT_RESULT_2026-09-13.md`. L'archive SOL-USDT contient 1 813 848 messages et 96 snapshots complets, mais aucun identifiant de séquence. Conformément au plan gelé, la reconstruction fondée sur les deltas est rejetée : une perte de mise à jour ne serait pas détectable. Les snapshots espacés d'environ 15 minutes restent une piste séparée et causalement plus propre ; aucune donnée L2 n'est intégrée au modèle actuel.

Le lot 5C snapshot-only est terminé dans `CRYPTO_LOT5C_SNAPSHOT_ONLY_RESULT_2026-09-13.md`. Les 96 snapshots passent tous les critères gelés et deux exécutions produisent les mêmes sorties. Les 1 813 752 updates sont ignorées par construction et un test de mutation confirme qu'elles ne peuvent modifier aucune feature snapshot. Ce Go est purement technique sur une paire et une journée ; il n'autorise ni modèle, ni intégration produit, ni téléchargement historique massif.

Le lot 5D de couverture est terminé dans `CRYPTO_LOT5D_L2_COVERAGE_RESULT_2026-09-13.md`. Les 42 fichiers attendus sont présents et leurs métadonnées passent les contrôles gelés. La journée complète médiane pèse `490,61 MB` et les trois dates candidates BTC + ETH + SOL totalisent `1 126,78 MB`. Ce résultat rend un prochain échantillon borné matériellement faisable, mais n'autorise aucun téléchargement : l'acquisition et la rétention doivent être préenregistrées dans un lot séparé.

Le lot 5E est terminé dans `CRYPTO_LOT5E_L2_SAMPLE_RESULT_2026-09-13.md`. Les neuf archives ont été acquises et les 8 927 snapshots observés sont tous valides, mais le contrat unique de 96 snapshots échoue : les données 2023–2024 sont principalement à une minute, tandis que 2026 est à 15 minutes avec un décalage initial de quelques millisecondes. Le résultat strict reste No-Go et reproductible. Une éventuelle grille temporelle commune doit être définie dans un nouveau plan avant réanalyse ; aucun modèle n'est autorisé.

Le lot 5F est terminé dans `CRYPTO_LOT5F_L2_NORMALIZATION_RESULT_2026-09-13.md`. La règle préenregistrée sélectionne uniquement un snapshot situé dans la seconde après chaque quart d'heure. Elle conserve 814 des 864 créneaux et échoue au seuil sur les six séries 2023–2024 ; les trois séries 2026 passent à 96/96. Le No-Go est reproductible, les 50 absences restent explicites et aucune valeur n'est fabriquée. Une règle symétrique autour de la borne nécessiterait un nouveau lot préenregistré.

Le lot 5G est terminé dans `CRYPTO_LOT5G_L2_SYMMETRIC_ALIGNMENT_RESULT_2026-09-13.md`. La règle symétrique préenregistrée conserve 863 des 864 créneaux et les neuf séries passent le seuil de 95/96. Les timestamps source et de disponibilité causale restent distincts ; le seul manque ETH-USDT à minuit le 1er avril 2023 n'est pas rempli. Ce Go est technique et reproductible, mais ne justifie encore ni acquisition historique large ni modèle.

Le lot 5H est terminé dans `CRYPTO_LOT5H_L2_COLLECTION_FEASIBILITY_RESULT_2026-09-13.md`. Un historique complet de 420 jours représenterait environ 192 à 300 GiB et est rejeté localement. Trois probes publics à 400 niveaux montrent qu'une collecte prospective à 15 minutes représenterait environ 2,83 GiB bruts, 0,69 GiB gzip ou 2,06 GiB avec le facteur de sécurité ×3. Le Go est uniquement matériel et technique : aucun collecteur longue durée n'a été démarré et aucune valeur prédictive n'est démontrée.

Le lot 5I est terminé dans `CRYPTO_LOT5I_L2_PROSPECTIVE_COLLECTOR_RESULT_2026-09-13.md`. Le noyau local écrit les captures atomiquement, rend les répétitions identiques idempotentes, refuse les conflits sans écrasement et reconstruit les absences depuis les fichiers durables. Le pilote public conserve trois carnets valides sur une seule borne et se reproduit hors ligne. Son lancement manuel, environ 12 minutes 52 après la borne, ne prouve pas encore la future tolérance opérationnelle de 120 secondes. Aucun service ou planificateur n'a été installé.

Le lot 5J est terminé dans `CRYPTO_LOT5J_L2_ONE_SHOT_SCHEDULER_RESULT_2026-09-13.md`. L'essai réel s'est déclenché 14 ms après la borne de 11:00 UTC et les trois captures étaient durables après 2,105 s. Le verrou mono-instance a été libéré, aucun backfill n'a été tenté et la relecture hors ligne est identique. Les pannes partielles et réveils hors délai sont couverts par tests. Ce Go reste limité à une exécution unique : aucune tâche récurrente n'est installée.

Le lot 5K est terminé dans `CRYPTO_LOT5K_L2_PROGRESSIVE_EXTRACTION_RESULT_2026-09-13.md`. Les trois archives du 1er juillet 2024, soit `544 810 138` octets, ont été lues séquentiellement sans extraction brute et réduites à `1 076 624` octets de carnets complets alignés. Les 288 créneaux sont présents, les 4 320 snapshots natifs sont valides et les 14 854 068 updates sont ignorées. Le premier essai consommateur 5M a révélé que les objets étaient concaténés sans retour à la ligne ; les artifacts initiaux 5K–5L sont donc remplacés. Après correction et test de lecture réelle, deux passages complets reproduisent l'artifact 5K corrigé et ses données JSONL valides.

Le lot 5L est terminé dans `CRYPTO_LOT5L_L2_COMPACT_CORPUS_RESULT_2026-09-13.md`. Les neuf archives déjà acquises, soit `1 181 503 619` octets, deviennent neuf fichiers JSONL déterministes totalisant `2 574 579` octets. Les 8 927 snapshots natifs sont valides, 863 des 864 créneaux sont conservés et le seul manque ETH-USDT de 2023 reste explicite. Les 34 798 969 updates sont ignorées. Deux lectures indépendantes reproduisent le résultat et les neuf empreintes corrigées. Ce corpus couvre seulement trois journées isolées et ne permet pas encore une validation prédictive.

Le lot 5M est terminé dans `CRYPTO_LOT5M_L2_FEATURES_RESULT_2026-09-13.md`. Il produit 25 features instantanées sur 863 lignes disponibles et conserve une ligne manquante, soit la grille complète de 864 lignes. Les 12 945 comparaisons numériques avec les 15 métriques du lot 5G donnent zéro divergence. La table et le résultat sont reproductibles ; aucune cible, valeur future, normalisation, agrégation ou entraînement n'est inclus. Le lot 5N suivant en contrôle la faisabilité avant tout modèle.

Le lot 5N est terminé dans `CRYPTO_LOT5N_L2_EVALUATION_FEASIBILITY_RESULT_2026-09-13.md`. L'intégrité 5M et les neuf couples instrument-date passent, mais les 864 lignes ne représentent que trois dates UTC indépendantes. Le protocole gelé en exige au moins 480, une étendue de 1 551 jours et cinq instruments ; les valeurs observées sont respectivement 3, 1 188 et 3. Le verdict reproductible est No-Go prédictif. Aucun prix futur, cible ou modèle n'a été utilisé. La prochaine décision porte sur une acquisition L2 beaucoup plus longue et plus large, pas sur un choix de modèle.

Le lot 5O est terminé dans `CRYPTO_LOT5O_COMPACT_SIGNAL_FEASIBILITY_RESULT_2026-09-13.md`. L'écran de cinq familles, fondé sur les documentations officielles et des métadonnées sans téléchargement de série, classe le funding Binance USDⓈ-M en première position : sept actifs ont des archives mensuelles présentes en juin 2022 et août 2026, pour un volume estimé très inférieur à 1 Mo compressé. Le verdict est Go pilote, car la première observation exacte, la continuité interne et les doublons n'ont pas encore été contrôlés. `AdrActCnt` et `TxCnt` de Coin Metrics couvrent aussi sept actifs sur une longue période, mais restent en seconde piste tant que leur politique de révision point-in-time n'est pas verrouillée. OKX dérivés, FRED comme nouveauté et Fear & Greed sont rejetés pour ce prochain lot. Aucun dataset, cible, feature ou modèle n'a été créé.

Le lot 5P est terminé dans `CRYPTO_LOT5P_BINANCE_FUNDING_ACQUISITION_RESULT_2026-09-13.md`. Les 357 archives mensuelles et leurs 357 empreintes officielles passent les contrôles préenregistrés. Les sept instruments couvrent sans journée manquante les 1 553 jours du 1er juin 2022 au 31 août 2026, pour seulement 316 272 octets compressés. Deux acquisitions indépendantes donnent le même artifact `crypto-forecast-binance-funding-history-v1-6288d6a1dca91185` et les 14 fichiers normalisés sont identiques. Le verdict est Go qualité, pas Go prédictif : aucun agrégat, cible ou modèle n'a encore été créé.

Le lot 5Q est terminé dans `CRYPTO_LOT5Q_BINANCE_FUNDING_FEATURES_RESULT_2026-09-13.md`. Il produit 16 features sur 10 871 lignes, dont 10 668 éligibles après un warm-up explicite de 30 jours. Les mutations postérieures aux 31 décembre 2023, 2024 et 2025 changent respectivement 6 818, 4 256 et 1 701 lignes futures, sans modifier une seule des 4 053, 6 615 et 9 170 lignes protégées. Deux constructions indépendantes reproduisent l'artifact `crypto-forecast-binance-funding-features-v1-f32e0489dbb258fd`. Le verdict est Go causal, pas Go prédictif.

Le lot 5R est terminé dans `CRYPTO_LOT5R_FUNDING_MODEL_COMPARISON_RESULT_2026-09-13.md`. Le modèle de référence et le même modèle enrichi par les 16 features funding ont été comparés sur deux développements chronologiques et une confirmation finale séparée, avec purge de 30 jours et prétraitement appris sur l'entraînement uniquement. À 7 jours, l'AUC finale progresse de 0,021021, mais le Brier final se dégrade de 0,000740 et les développements ne confirment pas le gain de classement. À 30 jours, le Brier progresse en développement, puis le Brier et l'AUC se dégradent sur la confirmation finale. Aucun des deux horizons ne passe tous les critères gelés. Deux exécutions indépendantes reproduisent l'artifact `crypto-forecast-funding-model-comparison-v1-f6804348361648f4` et ses 20 412 prédictions.

Le verdict final est **No-Go prédictif funding**. Il n'autorise ni test économique 5S, ni démonstration locale sur le port 8082, ni intégration produit. Aucun lot expérimental obligatoire ne reste dans la chaîne prévue. Une éventuelle reprise future devra partir d'une source ou d'une hypothèse réellement nouvelle et d'un nouveau protocole préenregistré, sans retoucher les seuils à partir de ces résultats.

## Garde-fous de reprise

- Spot uniquement ; aucun levier, short ou poids négatif.
- Production 8080 et recherche locale/test 8082 restent séparées.
- Les artifacts sous `outputs/` sont des preuves locales et ne doivent pas être présentés comme une fonctionnalité produit.
- Aucun ajustement a posteriori fondé sur la confirmation finale.
- Toute donnée manquante reste explicitement indisponible.
- Commit, push, déploiement et activation financière nécessitent chacun une autorisation distincte.

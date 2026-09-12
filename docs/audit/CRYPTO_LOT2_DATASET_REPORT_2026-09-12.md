# Lot 2 crypto - dataset causal et validation temporelle

Date : 12 septembre 2026

## Statut

Le lot 2 fournit un dataset commun, daté et reproductible pour les futures comparaisons ML. Il ne contient aucun modèle entraîné, aucune probabilité présentée comme calibrée, aucune recommandation d'allocation et aucune action financière.

L'artefact local construit deux fois à l'identique est :

- version : `crypto-forecast-dataset-v1-618b482c94e44311` ;
- hash SHA-256 du CSV : `57f8aaf73eb8d2eb5fe9976d94335948d3be9e4a738b8ee6c44a60d1c5683ded` ;
- hash SHA-256 du code constructeur : `80f7582dd9071dbc7fd98e238f025bd52c0d8a74ccd134d4a0d64a7a96413bdf` ;
- 78 486 lignes ;
- 126 membres inventoriés, dont 121 éligibles aux features de prix et 5 exclus ;
- aucune entrée source rejetée.

Le CSV, le manifeste des entrées et de leurs hashes, l'univers et la couverture sont conservés sous `outputs/crypto-forecast-lot2/crypto-forecast-dataset-v1-618b482c94e44311/`. Les historiques sources ne sont ni copiés ni modifiés. Le nom de version dépend du contrat, du hash du code constructeur, des hashes d'entrée et de l'univers. Une seconde construction a reproduit la même version et le même hash.

## Conventions

- Une date de décision est un jour calendaire UTC. Les features utilisent uniquement les clôtures quotidiennes disponibles jusqu'à cette date incluse.
- Aucun remplissage vers le futur et aucune recherche `nearest` ne sont autorisés. Les jointures externes utilisent uniquement `as-of` vers le passé et conservent la date source pour audit.
- Les labels de recherche mesurent le rendement entre la clôture de décision et la clôture située exactement 7 ou 30 jours calendaires plus tard. L'exécution à la clôture suivante, les frais et le glissement appartiennent à la future couche de décision/backtest.
- La référence défensive est du cash USD avec un rendement total supposé nul. C'est une hypothèse explicite, pas une observation de marché ni une promesse de stabilité d'un stablecoin.
- Les paniers de groupe sont équipondérés entre les membres connus à la date de décision et leurs poids sont maintenus jusqu'à l'horizon. Si un membre n'a pas de prix de décision ou de sortie, la cible reste indisponible. Aucun membre n'est retiré avant renormalisation.
- Les colonnes `label_*` sont des événements binaires observés a posteriori. Le dataset ne contient aucune colonne de probabilité ; les probabilités ne pourront être publiées qu'après calibration au lot 3.
- Les volumes et la liquidité restent indisponibles, car le cache historique utilisé ne les date pas.

## Features et cibles

Les features strictement rétrospectives sont les rendements 7/30/90 jours, les distances aux moyennes mobiles 30/90/200 jours, les volatilités passées 7/30/60 jours, le drawdown depuis le plus haut des 90 derniers jours et les rendements relatifs à BTC et au groupe.

Les lignes sont séparées par objet :

- marché : rendement futur BTC et excès face au cash USD ;
- groupe : rendement du panier équipondéré et rendement relatif à BTC ;
- actif : rendement futur, rendement relatif au groupe et rendement relatif à BTC.

Chaque ligne conserve `decision_date`, `information_cutoff`, état de l'historique, état des features, nombre de membres connus et champs d'indisponibilité des cibles 7/30 jours.

## Couverture mesurée

| Objet | Lignes | Features complètes | Cible 7 j | Cible 30 j | Cible relative principale 7 j | Cible relative principale 30 j |
|---|---:|---:|---:|---:|---:|---:|
| Marché BTC | 3 100 | 93,5806 % | 99,7742 % | 99,0323 % | 99,7742 % face au cash | 99,0323 % face au cash |
| Groupes | 12 093 | 58,5380 % | 81,1378 % | 79,2359 % | 81,1378 % face à BTC | 79,2359 % face à BTC |
| Actifs | 63 293 | 16,9260 % | 94,5618 % | 90,2011 % | 31,4427 % face au groupe | 30,3161 % face au groupe |

Période observée : du 17 août 2017 au 10 février 2026. Seuls BTC et ETH possèdent au moins 730 observations ; les 119 autres actifs éligibles n'atteignent pas les deux années minimales prévues pour le protocole complet du lot 3.

## Causalité et validation

Le test central reconstruit le dataset après modification de toutes les observations postérieures à une date de coupure. Le hash des prix de décision, états d'univers et features antérieures reste identique, tandis que les labels futurs concernés peuvent changer.

Les autres garde-fous vérifiés sont :

- jointure `as-of` exclusivement vers le passé ;
- frontières communes entre actifs ;
- purge de 30 jours avant validation et test ;
- aucun split aléatoire des observations réelles ;
- sélection des colonnes et normalisation apprises sur l'entraînement uniquement ;
- correction de `prev_realized_vol_7d` dans l'ancien générateur : la feature utilise maintenant une volatilité rétrospective et non une cible future décalée.

## Limites et indisponibilités

- Le cache est ancien au regard de la date du rapport : sa dernière clôture est le 10 février 2026. Une actualisation contrôlée sera nécessaire avant toute étude contemporaine.
- L'ancien format de cache ne conserve pas le fournisseur de chaque observation. Chaque fichier est hashé, mais la provenance par point reste indisponible.
- L'univers historique est inféré depuis la première observation présente dans le cache. Il ne prouve pas les dates historiques de cotation, de délisting ou la composition passée complète. Le biais de survivant reste possible.
- Aucun délisting explicite n'est disponible dans l'entrée actuelle. La valeur zéro du compteur signifie « aucune information datée fournie », pas « aucun actif délisté ».
- Les cinq membres du groupe `Stablecoins` (`PAXG`, `TUSD`, `USD`, `USDC`, `USDT`) sont exclus des observations ML par prudence, car l'ancien service peut générer certains historiques stables. `PAXG` suit ici la taxonomie actuelle et devra être reclassé ou documenté séparément avant une étude sur l'or.
- La faible couverture des cibles relatives au groupe vient du contrat strict : un membre manquant rend le panier indisponible au lieu d'être supprimé silencieusement.

## Point exact de reprise pour le lot 3

Le lot 3 peut commencer sur la version figée ci-dessus avec les références nulles/momentum, Ridge, régression logistique et gradient boosting prévus, en utilisant les splits chronologiques purgés et le prétraitement train-only fournis.

Le protocole long est exploitable pour le marché BTC et, sous réserve de la définition de sa cible, pour ETH. Les conclusions générales sur les groupes et rotations altcoins doivent rester indisponibles tant qu'un historique d'au moins deux ans, une provenance plus précise et un univers daté avec listings/delistings n'ont pas été fournis. Aucun résultat du lot 3 ne devra être présenté comme applicable au portefeuille réel avant validation hors échantillon, calibration et comparaison nette de coûts.

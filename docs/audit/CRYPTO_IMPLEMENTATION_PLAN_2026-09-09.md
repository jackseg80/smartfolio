# SmartFolio : prévision et amélioration du rendement/risque du portefeuille

Date : 9 septembre 2026. Plan de référence pour la reprise avec Sol, à valider par l'utilisateur avant implémentation. Il remplace les plans conversationnels précédents : **la prévision appartient au chantier principal**, et non à une amélioration ML facultative ultérieure.

Références : [audit décisionnel](CRYPTO_DECISION_CHAIN_AUDIT_2026-09-09.md) et [audit ML](CRYPTO_ML_AUDIT_2026-09-09.md). Ces audits établissent des défauts techniques ; ils ne démontrent pas encore une amélioration économique du portefeuille de l'utilisateur.

## 1. Objectif et périmètre confirmé

Partir du portefeuille crypto réel de l'utilisateur et chercher un meilleur compromis entre rendement, pertes possibles et coûts. Essayer d'anticiper les mouvements et rotations suffisamment tôt pour améliorer les décisions, tout en préservant un capital déjà constitué.

- Spot uniquement ; aucun levier ni position négative.
- Horizon d'investissement : quelques semaines à quelques mois. Premiers horizons de prévision : **7 et 30 jours calendaires**. Le 30 jours pilote la recherche stratégique ; le 7 jours sert à évaluer l'anticipation tactique et défensive.
- Couvrir BTC, ETH, les groupes existants (dont L1, L2) et les actifs éligibles du portefeuille ou d'une liste explicitement autorisée.
- Conserver un socle BTC/ETH et permettre une rotation importante vers les altcoins lorsque les résultats le justifient. BTC/ETH ne sont pas considérés comme sans risque.
- Stablecoins : évaluer l'intérêt de réduire l'exposition crypto ; distinguer ce rôle des risques propres de liquidité, émetteur et décrochage. Ne pas leur attribuer un potentiel de hausse artificiel.
- Aucun minimum implicite par catégorie ni protection obligatoire des positions détenues. Les contraintes explicites restent applicables.
- Revue hebdomadaire et possibilité de propositions défensives anticipées ; exécution manuelle sur les exchanges, avec simulation et export.
- Le niveau de risque acceptable et la politique active seront choisis après comparaison. Aucun plafond de drawdown garanti, aucune promesse de détecter les sommets ou les actifs qui vont « exploser ».

**Livraison finale :** moteur en comparaison/observation, prévisions réellement évaluées, propositions traçables et dossier présentant les avantages et limites par rapport au portefeuille actuel. Pas de bascule automatique en production financière.

## 2. Architecture : prévoir, décider, vérifier

Construire une logique Python canonique, partagée par l'API, le backtest et la simulation, avec trois responsabilités distinctes.

1. **Prévisions de marché** : dépendantes des observations de marché et de l'univers daté, pas des poids personnels du portefeuille. Produire des résultats à 7/30 jours par marché, groupe et actif lorsque les données le permettent.
2. **Décision de portefeuille** : combiner les prévisions disponibles avec les positions réelles, contraintes, coûts et exposition. Une probabilité favorable n'impose pas un achat ; une position déjà concentrée peut ne pas être renforcée.
3. **Proposition réalisable** : appliquer les limites de mouvement, les disponibilités et les frais, puis expliquer les écarts entre cible théorique et proposition intermédiaire.

Chaque calcul utilise un snapshot cohérent : utilisateur, source, positions, prix, observations datées, univers autorisé, versions et paramètres. Un même snapshot et une même politique produisent les mêmes résultats.

Le Risk Score reste une mesure positive de robustesse, distincte d'une prévision de marché. Les anciens CCS, On-Chain Composite, DI et régimes restent identifiés comme diagnostics de leur version. Aucun score ou pourcentage de confiance artificiel pour remplir l'interface.

Une prévision expose au minimum : objet, horizon, cible prédite, estimation, probabilités si calibrées, incertitude si estimable, date des observations, modèle/version et état de validation. Qualité des données, probabilité statistique et résultat du contrôle d'allocation sont des champs distincts. Une sortie indisponible reste indisponible.

Interfaces : `POST /api/decision/preview`, `GET /api/decision/proposals/{id}`, et prise en charge de `proposal_id` dans `/rebalance/plan`. La preview rassemble prévisions, résultats des politiques et comparaison aux positions. Le serveur résout les positions et les cibles ; le navigateur ne fournit pas une allocation concurrente dans le mode par identifiant.

JWT obligatoire sur les nouvelles routes, cohérence avec `X-User`, isolation par utilisateur/source et format de réponse du projet. Propositions immuables, invalidées par un changement de positions, de source ou de politique. Aucun nouveau mécanisme d'envoi d'ordres.

## 3. Lots à réaliser dans l'ordre

### Lot 0 — Référence personnelle et contrats

Résoudre la source crypto effectivement sélectionnée par l'utilisateur, sans supposer que les données démo ou le dernier CSV trouvé représentent son portefeuille. Enregistrer quantités, valeurs, plateformes, positions bloquées et couverture des historiques.

Présenter concentrations et contributions au risque. Pour la covariance crypto : rendements journaliers, Ledoit-Wolf et annualisation 365 jours, fenêtre cible 365 jours et minimum de 90 observations communes. Revoir les conventions du module d'optimisation existant avant réutilisation, notamment pondération de covariance et annualisation. Ne pas brancher aveuglément son maximum de Sharpe.

Distinguer historique réel des positions, simulation de la composition actuelle et risque estimé aujourd'hui. Ne jamais retirer silencieusement les actifs sans historique puis renormaliser le reste. Une analyse partielle indique sa couverture et ne donne pas de classement global trompeur.

**Fin :** référence utilisateur vérifiable, contrats d'entrée/sortie fixés et jeux de tests déterministes.

### Lot 1 — Sécurisation et comptabilité

Transformer les reproductions des audits en tests, puis corriger : cibles négatives, budget stablecoins contradictoire, identité des propositions, données de secours présentées comme observations, sorties ML fictives et comptabilité du backtest.

Tenir quantités et cash, laisser dériver les poids entre opérations, déduire les coûts et calculer toutes les métriques depuis la même courbe nette. Généraliser la comptabilité aux actifs réellement simulés ; ne pas représenter toutes les altcoins par les rendements BTC.

**Fin :** invariants vérifiés, aucune donnée fictive utilisée comme prévision réelle et résultats comptables reproductibles.

### Lot 2 — Données de prévision et validation temporelle

Créer un jeu de données daté commun aux modèles. A la date t :

- Marché : rendement futur BTC à 7/30 jours et rendement excédentaire face à la référence défensive explicitement définie.
- Groupe : rendement futur du panier de groupe et rendement relatif à BTC.
- Actif : rendement futur et rendement relatif à son groupe ; BTC reste la référence relative secondaire commune.
- Probabilités : hausse et surperformance de la référence. Les coûts de l'opération personnelle sont appliqués ensuite par la couche décisionnelle.

Les paniers de groupe sont équipondérés pour définir les cibles de recherche, avec membres connus à t et poids maintenus pendant l'horizon. Ils ne constituent pas une allocation recommandée. Documenter les conventions et les actifs non disponibles, délistés ou sans prix de sortie ; ne pas les supprimer des résultats.

Premier jeu de features : rendements passés à 7/30/90 jours, distance aux moyennes 30/90/200 jours, volatilité passée 7/30/60 jours, drawdown passé 90 jours, rendements relatifs à BTC et au groupe. Volume/liquidité seulement si l'historique daté est disponible. Pas de reconstruction fictive de l'on-chain ou du sentiment avec des prix.

Corriger la fuite de volatilité identifiée : les entrées utilisent uniquement des observations antérieures à la décision. Découper toutes les lignes multi-actifs aux mêmes dates, normaliser sur l'entraînement seul, et purger 30 jours aux frontières pour les labels futurs qui se chevauchent. Aucun split aléatoire ni recherche de date « nearest » pouvant accéder au futur.

**Fin :** le test « changer le futur ne change aucune feature ni décision passée » passe ; couverture et limites de chaque cible sont documentées.

### Lot 3 — Prévisions réellement comparées

Implémenter un ensemble borné de candidats, sans campagne massive de recherche de paramètres :

- Références : rendement nul, tendance/momentum simple et probabilité de classe estimée sur l'entraînement uniquement.
- Modèles simples : régression Ridge avec alpha=1 et régression logistique avec C=1, après normalisation sur train seulement.
- Candidat non linéaire : gradient boosting d'histogrammes, profondeur maximale 3, 100 itérations, taux d'apprentissage 0,05 et graine 42 ; versions régression/classification selon la cible.
- Calibration des probabilités sur une période chronologique distincte de l'entraînement ; aucun ajustement sur le test final.

Entraîner séparément les horizons et les cibles. Conserver ces réglages comme premières hypothèses versionnées. Toute modification devient une nouvelle expérience enregistrée, pas un remplacement silencieux du résultat initial.

Mesurer erreurs de rendement, classement relatif des actifs/groupes, Brier score et calibration des probabilités. Si l'historique ou la couverture ne permet pas d'évaluer une sortie, la déclarer expérimentale ou indisponible. Les modèles HMM ou neuronaux existants ne sont pas certifiés par leurs anciennes métriques : corriger leur provenance et leur contrat avant tout usage comparatif.

Le cycle Bitcoin, l'on-chain et le sentiment restent des candidats supplémentaires. Leur intégration exige des observations disponibles à la date simulée et une comparaison avec/sans la contribution. Ils ne doivent pas retarder la première comparaison des modèles de prix.

**Fin :** performances réellement hors entraînement, artifacts identifiables, calibration évaluée et aucune inférence inventée. Un résultat négatif ou non concluant est un résultat acceptable du lot.

### Lot 4 — Traduction des prévisions en allocations comparables

Conserver une référence réactive : BTC/SMA200, cinq clôtures favorables à l'entrée, deux défavorables à la sortie. Comparer profils d'exposition 20/60 % et 20/80 %, chacun avec BTC/ETH seuls, plafond altcoins 30 %, ou plafond 60 % de la poche risquée. Répartition BTC/ETH initiale 60/40. Ces six politiques sont des hypothèses de test.

Référence rotation : tendance au-dessus de SMA90 et rendement relatif BTC positif à 90 jours, confirmations 5/2 jours. Part altcoins = plafond multiplié par la fraction de l'univers éligible confirmée favorable ; nulle en marché BTC défavorable. Pondération des altcoins selon inverse de volatilité passée 60 jours, plafond de test 10 % du portefeuille par altcoin.

Ajouter une variante prévisionnelle correspondante à chaque politique, pour isoler l'apport des prévisions en gardant les mêmes contraintes et coûts :

- Etat marché fondé sur la prévision BTC à 30 jours : favorable si probabilité de surperformance de la référence défensive >= 0,60 et excédent de rendement estimé supérieur au coût aller-retour ; défavorable si probabilité <= 0,40. Entre les seuils, conserver l'état précédent. Initialisation sans état exploitable : observation, pas d'action.
- Eligibilité de rotation fondée sur la probabilité de surperformance à 30 jours de l'actif face à BTC >= 0,60 et rendement relatif estimé supérieur au coût aller-retour. Les résultats de groupe sont affichés et servent à vérifier la cohérence, sans ajouter un second multiplicateur.
- Un signal défensif à 7 jours confirmé sur deux observations quotidiennes (probabilité de rendement BTC négatif >= 0,65 et rendement estimé négatif au-delà des coûts) autorise une proposition anticipée de réduction. Il ne déclenche aucun ordre.
- Garder les confirmations, plafonds, budgets et règles de pondération identiques à la référence. Publier ces seuils comme paramètres expérimentaux ; ne pas prétendre qu'ils sont optimaux.
- Une prévision indispensable indisponible ne bascule pas silencieusement sur la règle de tendance. La référence reste affichée séparément et la variante prévisionnelle s'abstient.

Rendre explicites budgets souhaité/effectif et contraintes. Une nouvelle cible doit respecter le budget final ; le surplus impossible à attribuer est traité avant validation du budget, jamais par une correction cachée ensuite.

**Fin :** comparaison avec/sans prévisions, aux mêmes risques et coûts mesurés, y compris le coût de transition depuis les positions réelles. Aucun modèle n'est retenu uniquement pour une bonne accuracy.

### Lot 5 — Interface, propositions manuelles et bilan

Afficher simplement : portefeuille actuel, anticipations à 7/30 jours, incertitude, allocation proposée, évolution du risque estimé, coûts et raisons des changements. Réserver les scores détaillés à une vue secondaire. Les textes visibles restent en anglais.

Appliquer un déplacement proportionnel vers la cible avec facteur commun respectant les caps, sans inventer de position négative pour équilibrer les mouvements. Réévaluer financement, frais et minimums après calcul des actions, plateforme par plateforme. Aucun transfert implicite entre exchanges, aucune vente au-delà des quantités disponibles. Les positions inconnues/bloquées exigent une revue ; les contraintes impossibles ne sont pas silencieusement relâchées.

Une proposition ne vaut pas exécution. Rafraîchir les positions ou obtenir confirmation avant de recalculer un mouvement supposé déjà effectué. Le mode comparaison n'alimente pas automatiquement les anciens connecteurs d'exécution.

**Fin :** parcours preview -> simulation -> export vérifié, bilan comparatif et mode observation. La décision de risque et la bascule restent à l'utilisateur.

## 4. Protocole économique et tests d'acceptation

- Référence principale : maintien des quantités du portefeuille actuel à partir du snapshot. Rétrospectivement, distinguer les positions historiques connues et la simulation hypothétique de la composition actuelle. Ajouter les références simples BTC, BTC/ETH et répartition fixe avec stablecoins.
- Fenêtres chronologiques progressives : entraînement expansif, six mois de calibration puis six mois de test, progression de six mois ; minimum deux ans d'entraînement initial. Réserver les douze derniers mois disponibles à une confirmation finale après gel des modèles, paramètres et politiques. Si les données ne permettent pas ce protocole, ne pas raccourcir silencieusement les exigences : rapporter une évaluation partielle.
- Reconstituer chaque prévision historique avec le modèle qui aurait pu être entraîné à cette date. Aucun modèle entraîné jusqu'à aujourd'hui appliqué rétroactivement comme s'il avait existé auparavant.
- Signaux disponibles avant transaction ; avec clôtures journalières seules, exécution simulée à la clôture suivante. Mesurer les périodes sans signal et les abstentions.
- Coûts de comparaison initiaux : 0,20 % de frais et 0,10 % de glissement par montant négocié, plus un scénario avec coûts doublés. Utiliser les coûts renseignés quand disponibles. Pas de rendement gratuit attribué aux stablecoins.
- Publier rendement net, volatilité, Sortino, Sharpe, drawdown maximal, récupération, pire mois, turnover, frais et exposition moyenne. Présenter les différences de risque au lieu d'assimiler plus de rendement à une amélioration.
- Tester la stabilité entre périodes, marchés difficiles et variantes de coûts. Ne pas traiter les labels journaliers à 30 jours comme des observations indépendantes ; employer des blocs temporels d'au moins 30 jours pour les estimations d'incertitude.
- Sans composition historique datée des groupes et de l'univers, qualifier le résultat de conditionnel à l'univers retenu et limiter les conclusions. Ne pas annoncer une validation générale des altseasons ou du risque stablecoin.
- Tests déterministes : budget 80 % stablecoins conservé ; aucune cible négative ni dépassement ; portefeuille moitié BTC/moitié cash avec BTC 100 -> 200 -> 100 sans transaction revenant au capital initial ; frais cohérents ; causalité temporelle ; isolation utilisateur/source ; rejet des propositions périmées ; prédictions réellement liées à un artifact et à un horizon.

**Critère de réussite :** savoir dire si les prévisions et les changements proposés améliorent suffisamment le compromis rendement/risque du portefeuille pour justifier les coûts, ou constater que l'avantage n'est pas établi. Aucune supériorité ni aucun niveau de perte acceptable n'est présupposé.

## 5. Consigne de reprise

Après validation de ce plan, Sol commence par les lots 0 et 1, puis avance dans l'ordre avec bilan par lot. Préserver les changements locaux étrangers au chantier et ne pas reprendre l'audit entier sans nouveau besoin démontré. Aucun commit, déploiement, ordre réel ou changement de stratégie active n'est inclus. Le premier cycle utilise les modèles bornés ci-dessus ; une nouvelle architecture neuronale n'est pas un prérequis.

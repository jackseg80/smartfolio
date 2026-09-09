# Complément d'audit : modèles ML crypto

Date : 9 septembre 2026. Complément au [premier audit](D:/Python/smartfolio/docs/audit/CRYPTO_DECISION_CHAIN_AUDIT_2026-09-09.md). Usage : allocation au comptant, quelques semaines à quelques mois.

## Avis

Les modèles ML nécessitent aussi une révision avant de servir de justification à des rééquilibrages réels. Le problème ne se réduit pas au choix entre HMM et réseau neuronal : certaines voies de production contiennent des constantes, l'entraînement de volatilité présente une fuite de données futures démontrée, et les métadonnées ne relient pas correctement chaque entraînement au fichier de modèle utilisé.

Il existe du travail ML réel : HMM BTC/ETH, réseaux de régime et de volatilité, calibration par température dans certaines voies, données historiques, modèles sauvegardés et métriques. Leur existence ne démontre toutefois pas leur valeur prédictive ni leur utilisation par chaque écran.

Périmètre : lecture des routes utilisées par les écrans crypto, de l'orchestrateur et des entraînements, lecture des métadonnées et HMM sauvegardés via le chargeur sécurisé du projet, reproduction hors ligne de la causalité des features. Aucun entraînement, aucun remplacement de modèle, aucun ordre. Les modèles en mémoire du serveur et l'état authentifié du navigateur n'ont pas été inspectés. Les dates ci-dessous sont celles des artifacts locaux, pas une affirmation sur un processus serveur en cours.

## 1. Plusieurs sorties « ML » ne viennent pas d'une inférence du modèle

### Façade API appelée par les écrans

`GET /api/ml/predictions/live` retourne des volatilités BTC/ETH fixes (0,0734 et 0,0892), un Fear & Greed fixe à 58 et un régime Correction qui devient Bull Market si le modèle est déclaré chargé. Le chargement d'un modèle ne mesure pas l'état du marché. La réponse contient néanmoins `based_on_training: true`.

`GET /api/ml/regime/current` utilise cette voie. L'onglet ML d'Analyse et certains contextes du chat IA l'appellent. Ce n'est donc pas uniquement du code inutilisé.

Sources : [réponse live](D:/Python/smartfolio/api/ml/prediction_endpoints.py:200), [alias](D:/Python/smartfolio/api/ml/prediction_endpoints.py:174), [consommation dans Analyse](D:/Python/smartfolio/static/modules/analytics-unified-tabs-controller.js:165), [AI Dashboard](D:/Python/smartfolio/static/ai-dashboard.html:2072).

L'enrichissement multi-horizon de `/api/ml/predict` contient également des rendements attendus fixes : +0,1 % à un jour, +2,5 % jusqu'à sept jours, +8 % au-delà. Volatilité, intervalles et confiance sont construits par formules prédéfinies, sans inférence sur les données du marché dans cette fonction : [multi-horizon](D:/Python/smartfolio/api/ml/prediction_endpoints.py:481).

### Voie orchestrateur → gouvernance

Lorsque le modèle est marqué `ready`, l'orchestrateur appelle `_get_regime_predictions()`, qui retourne toujours Bull Market, probabilité 0,75, stabilité 0,82 et durée prévue de 45 jours. `_get_correlation_forecasts()` retourne des corrélations courantes de 0,65 et prévues de 0,58. Ces résultats peuvent être consommés comme signaux par la gouvernance.

La volatilité de cette voie est calculée depuis la volatilité historique des prix, avec confiance fixe de 0,75 ; la fonction n'appelle pas le LSTM sauvegardé. Cette statistique historique peut être utile, mais son nom ne doit pas la faire passer pour une prévision neuronale validée. La convention d'unité doit aussi être explicite : le calcul multiplie par racine de 365 même pour une demande d'horizon d'un jour.

Sources : [aiguillage selon état ready](D:/Python/smartfolio/services/ml/orchestrator.py:396), [régime et corrélation](D:/Python/smartfolio/services/ml/orchestrator.py:690), [volatilité historique](D:/Python/smartfolio/services/ml/orchestrator.py:451), [consommation par gouvernance](D:/Python/smartfolio/services/execution/governance.py:260).

### Sentiment

La route `/api/ml/sentiment/symbol/{symbol}`, utilisée dans Analyse et dans les ajustements de cibles, lit un score de gouvernance. Son détail social et news est ensuite obtenu par multiplication de ce score, avec noms de plateformes et nombre d'articles prédéfinis. Ces champs ne constituent pas une preuve d'analyse de ces sources dans cette route. En cas d'exception, une valeur peut même provenir d'une empreinte du symbole et du nombre de jours.

Un moteur d'analyse de sentiment distinct existe ; cela ne justifie pas d'inventer les sous-sources de cette réponse : [route](D:/Python/smartfolio/api/ml/prediction_endpoints.py:336), [détail construit](D:/Python/smartfolio/api/ml/prediction_endpoints.py:372), [autre moteur appelé par l'orchestrateur](D:/Python/smartfolio/services/ml/orchestrator.py:522).

**Priorité haute :** distinguer explicitement statistiques historiques, heuristiques, modèles entraînés et données indisponibles. Un label « live » ou « trained » doit être lié à une inférence vérifiable, un artifact précis et une date d'observation.

## 2. L'entraînement de la volatilité voit des données futures

Le chemin réel de `scripts/train_models.py` définit `realized_vol[i]` comme la dispersion des sept rendements suivants. Cette série sert de cible, ce qui est normal. Mais `realized_vol[end_idx-1]` est réintroduit comme entrée `prev_rv`, présentée comme volatilité de la veille. Décaler une cible future d'un jour ne transforme pas cette cible en observation passée.

Sources : [définition forward](D:/Python/smartfolio/scripts/train_models.py:248), [réintroduction en feature](D:/Python/smartfolio/scripts/train_models.py:358), [cible](D:/Python/smartfolio/scripts/train_models.py:385).

### Reproduction déterministe

Le probe exécute la fonction de génération actuelle extraite par AST, avec cache de prix simulé et sans accès réseau. Le passé est identique jusqu'au prix d'indice 40 ; seul le futur à partir de l'indice 44 change.

| Vérification | Résultat |
|---|---:|
| Prix passés identiques | Oui |
| Features de régime identiques | Oui |
| Colonne d'entrée modifiée | 11, `prev_rv` |
| Valeur initiale | 0,00333265 |
| Valeur après modification du futur | 0,12257055 |

Donc la feature change de 0,33 % à 12,26 % alors qu'aucune information disponible à la date de décision n'a changé. C'est une fuite temporelle démontrée. Les cibles futures sont autorisées à changer ; les entrées ne le sont pas.

Le split temporel et la normalisation sur train seulement sont de bons éléments du chemin volatilité actuel. Ils ne compensent pas une fuite déjà présente dans les features. Il faut reconstruire une volatilité réalisée strictement rétrospective, vérifier les index, purger les frontières pour les cibles qui se chevauchent, puis réentraîner et revalider.

Les métadonnées BTC/ETH/SOL sauvegardées incluent toutes `prev_realized_vol_7d`. Cela confirme la pertinence de ce contrat de features pour les artifacts présents, sans reconstruire exactement le code historique ayant produit chaque fichier. Aucun taux précis de dégradation après correction n'est estimable à partir de ce seul test.

## 3. Le HMM BTC/ETH détecte des états, mais leurs noms ne sont pas correctement garantis

La voie `/api/ml/crypto/regime` utilise réellement `BTCRegimeDetector`, avec un fichier propre à BTC ou ETH et un système hybride de règles. Il faut la distinguer de `/api/ml/regime/current` décrit plus haut.

Cependant, `train_hmm()` entraîne un modèle non supervisé, choisit la meilleure vraisemblance d'entraînement parmi cinq initialisations et garde directement les noms fixes 0=Bear, 1=Correction, 2=Bull, 3=Expansion. Aucune étape d'association des états appris à des critères économiques n'est présente dans cette méthode.

L'ordre des états d'un HMM peut varier après apprentissage ; le numéro d'un état n'est pas son sens économique. C'est explicitement signalé par la [documentation hmmlearn](https://hmmlearn.readthedocs.io/en/0.3.3/tutorial.html).

Sources : [noms fixes](D:/Python/smartfolio/services/ml/models/btc_regime_detector.py:45), [entraînement](D:/Python/smartfolio/services/ml/models/btc_regime_detector.py:203), [interprétation directe](D:/Python/smartfolio/services/ml/models/btc_regime_detector.py:326).

L'inspection du HMM BTC sauvegardé renforce ce doute : l'état nommé Bull Market présente une moyenne de rendement journalier d'environ -0,395 %, un drawdown moyen de -33,5 % et une tendance à 30 jours de -3,7 %. L'état Correction présente une tendance à 30 jours de +32,9 %. Ces moyennes sont des caractéristiques des états appris, pas des rendements futurs. Elles appellent une vérification des noms, pas une recommandation de trading.

Les règles peuvent remplacer le résultat HMM si leur confiance dépasse un seuil. Elles ne corrigent pas l'association des états dans les autres cas. Autre incohérence : après remplacement par les règles, les probabilités renvoyées restent celles du HMM, ce qui peut rendre le régime annoncé et la distribution affichée discordants : [fusion et probabilités](D:/Python/smartfolio/services/ml/models/btc_regime_detector.py:359).

Enfin, la confiance HMM est mélangée à un prior uniforme fixé, et certaines confiances de règles sont prédéfinies. Cela limite l'extrême confiance, sans établir qu'une probabilité annoncée de 85 % réussit effectivement 85 % du temps.

## 4. Classification de régime et prédiction de rendement sont différentes

Dans le script d'entraînement, le réseau de régime apprend principalement des labels produits par des règles sur les rendements, drawdowns et tendances déjà présents dans la fenêtre des features. Une accuracy élevée mesure surtout sa capacité à reproduire cette classification.

Elle ne démontre pas qu'il anticipe une hausse à sept ou trente jours, ni qu'il bat la règle qui fabrique les labels. Le registre annonce pourtant un horizon 7d pour les modèles de régime.

La route de « forecast » BTC contient également des probabilités de scénarios prédéfinies, notamment 0,8 pour « la tendance continue ». Ce sont des hypothèses de scénario à présenter comme telles, pas des probabilités calibrées : [labels](D:/Python/smartfolio/scripts/train_models.py:283), [scénarios](D:/Python/smartfolio/api/ml_crypto_endpoints.py:442).

## 5. La validation et le registre ne permettent pas de certifier les métriques affichées

### Prétraitement et découpage

Le réseau de régime du script normalise toutes les données avant le split train/validation/test. De plus, la liste est construite actif par actif puis découpée par position dans la liste : un split 60/20/20 ne garantit donc pas une frontière chronologique commune entre actifs.

L'autre classe `RegimeDetector` normalise également avant la séparation et utilise un split aléatoire. Certains labels de cette classe réétiquettent rétrospectivement une plage passée quand une condition ultérieure est satisfaite. Ces labels ne doivent pas être présentés comme les décisions qui auraient été disponibles en temps réel.

Sources : [script](D:/Python/smartfolio/scripts/train_models.py:431), [ordre des actifs](D:/Python/smartfolio/scripts/train_models.py:217), [autre classe](D:/Python/smartfolio/services/ml/models/regime_detector.py:805), [réétiquetage](D:/Python/smartfolio/services/ml/models/regime_detector.py:75). La [documentation scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html) demande d'apprendre les transformations sur l'entraînement uniquement ; sa [documentation de validation temporelle](https://scikit-learn.org/stable/modules/cross_validation) explique les limites des découpages aléatoires pour les séries temporelles.

### Fichiers présents et provenance

| Artifact lu | Date d'entraînement enregistrée | Observation |
|---|---|---|
| Régime neuronal | 10 février 2026 | Accuracy 89,70 %, 641 échantillons test |
| HMM BTC | 21 octobre 2025 | 1 746 observations, fenêtre demandée 1 825 jours |
| HMM ETH | 7 février 2026 | 411 observations malgré une fenêtre demandée 3 650 jours |
| Volatilité BTC | 3 novembre 2025 | 592 échantillons test |
| Volatilité ETH | 3 novembre 2025 | 71 échantillons test |
| Volatilité SOL | 9 février 2026 | 65 échantillons test |

Le registre associe au HMM BTC les mêmes métriques et date de février que le réseau neuronal. Le dispatcher Admin envoie les modèles de type/noms « regime » vers `save_models(train_regime=True)`, qui entraîne le réseau neuronal et écrit ses fichiers, sans appeler `train_hmm()`. Le HMM BTC local reste daté d'octobre. Une actualisation du registre n'est donc pas une preuve de renouvellement de ce HMM.

Sources : [dispatch Admin](D:/Python/smartfolio/services/ml/training_executor.py:369), [sauvegarde neuronale](D:/Python/smartfolio/scripts/train_models.py:916), [mise à jour des métriques](D:/Python/smartfolio/services/ml/training_executor.py:580).

Le champ `data_source: synthetic` des métadonnées de régime est lui-même codé en dur dans le script, y compris lorsqu'Admin demande `real_data=True`. Il serait incorrect d'en conclure que le modèle présent a forcément été entraîné sur données synthétiques. La provenance est ambiguë et doit être réparée : [champ codé en dur](D:/Python/smartfolio/scripts/train_models.py:618).

Le rapport historique de validation BTC fourni dans le dépôt vérifie des seuils et un cas courant d'octobre 2025 ; il indique lui-même que la validation historique par fenêtres n'est pas implémentée : [rapport](D:/Python/smartfolio/data/ml_predictions/btc_regime_validation_report.json).

## Ce que je conserverais et ce que je changerais

Conserver les prix historiques, les artifacts et métadonnées comme base de traçabilité, les règles économiques explicites et la séparation BTC/ETH. Conserver aussi les réseaux comme candidats de recherche, sans utiliser leurs métriques actuelles comme certification de performance future.

Ordre proposé :

1. Remplacer les réponses fixes/simulées par une inférence identifiable ou un état indisponible. Afficher honnêtement les statistiques historiques et les scénarios heuristiques.
2. Corriger la fuite de volatilité, le prétraitement et les frontières temporelles. Introduire un test systématique : modifier le futur ne doit jamais changer les features disponibles aujourd'hui.
3. Relier chaque bouton d'entraînement au bon modèle et au bon artifact. Enregistrer hash du fichier, période réelle, nombre d'observations, paramètres, version du code et jeu d'évaluation.
4. Définir et vérifier le sens des états HMM à chaque entraînement ; évaluer le régime courant avec les seules données connues à la date simulée.
5. Pour la volatilité, comparer d'abord au dernier niveau réalisé, à une moyenne exponentielle et à une régression simple multi-horizon. Comparer sur la même cible future, en unités explicites, avec erreurs hors échantillon et couverture des intervalles. Un LSTM doit apporter un gain mesurable sur ces références.
6. Pour une prévision directionnelle, définir une cible future distincte, par exemple rendement relatif BTC/cash à 30 jours, et commencer par un modèle régularisé avec peu de features. Calibrer les probabilités et tester le résultat économique net de coûts, pas seulement l'accuracy.
7. Reconnecter ensuite les modèles validés à la chaîne de décision canonique et mesurer leur valeur ajoutée par retrait successif. Un bon modèle de volatilité prédit l'amplitude des variations, pas leur sens.

Ces recommandations définissent des candidats et un protocole ; aucun modèle alternatif n'a été déclaré gagnant sur des données historiques corrigées pendant cet audit.

## Preuves produites

[Probe hors ligne](D:/Python/smartfolio/outputs/audit-crypto-2026-09-09/probe_ml.py) et [résultats complets](D:/Python/smartfolio/outputs/audit-crypto-2026-09-09/ml-results.json). Le probe a terminé sans erreur, a démontré la fuite de données et a lu les six artifacts mentionnés. Pas de nouvelle campagne d'entraînement ou de validation de performance ; pas de prétention à un audit exhaustif de tous les modèles boursiers ou des modules ML secondaires.

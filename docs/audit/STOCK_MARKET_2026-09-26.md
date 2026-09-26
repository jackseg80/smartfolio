# Audit Stock Market — 26 septembre 2026

## Périmètre et preuve

Revue du parcours Saxo CSV/API et des pages `saxo-dashboard.html`, `bourse-analytics.html` et `bourse-recommendations.html`, jusqu'aux services de prix, de risque, de modèles ML, de recommandations et d'opportunités. Les constats proviennent du code, de tests locaux et, après réparation du serveur de test, de parcours authentifiés dans le navigateur. Un export Saxo local réel a été chargé ; il date du 18 janvier 2026 et ne prouve pas les positions actuelles du compte. Le parcours Saxo API réel et la qualité financière des signaux restent à valider.

## Constats et corrections locales

| Priorité | Constat | Correction locale | État |
| --- | --- | --- | --- |
| P0 | Un échec du fournisseur pouvait produire des prix synthétiques et alimenter risque et ML ; le navigateur pouvait réafficher indéfiniment une ancienne réponse après une erreur. | Suppression de cette substitution ; erreur explicite, caches historiques séparés par une nouvelle version, provenance vérifiée, rejet d'un dernier prix vieux de plus de sept jours pour une décision actuelle, et aucun repli sur un cache navigateur périmé pour la bourse. | Implémenté, tests locaux pour les prix ; contrôle statique du cache navigateur. |
| P0 | Le CSV Saxo sélectionné pouvait être remplacé silencieusement par le plus récent ; le cash pouvait provenir d'un autre export. | Sélection exacte du fichier pour positions et cash, échec explicite si absent. | Implémenté, tests locaux. |
| P0 | Le tableau de bord pouvait mélanger le cash CSV au mode API ou remplacer un échec de lecture du cash par zéro. | Cash du mode API lu uniquement dans la réponse API ; le tableau de bord ne publie pas un total CSV lorsque la lecture du cash sélectionné échoue. | Implémenté, contrôle statique de la page ; parcours réel à valider. |
| P0 | Un score de risque était possible avec seulement une partie des titres et avec des rendements en devises natives comparés en USD. | Conversion des cours non USD par des séries FX historiques datées ; score indisponible si les cours ou le change manquent ; mêmes dates de début ET de fin pour les rendements comparés ; cash et concentration calculés sur les positions sélectionnées. | Implémenté, tests locaux. |
| P1 | Les métriques de bénéfices supposaient une date à 90 jours et une hausse de volatilité de 50 % sans événements vérifiés. | Métriques indisponibles sans dates réelles ; aucun chiffre prospectif inventé. | Implémenté, tests locaux. |
| P1 | Labels de régime et normalisation ML pouvaient incorporer des informations futures ; la volatilité utilisait une annualisation crypto. | Labels causaux, séparation chronologique entraînement/validation/test, normaliseur appris sur entraînement, 252 séances pour les actions. | Implémenté, tests de causalité ; entraînement réel à confirmer. |
| P1 | Une prédiction ML ou un intervalle de confiance pouvait être présenté sans validation hors échantillon. | Modèles actions de volatilité et de régime publiés seulement s'ils battent leurs références simples sur test temporel ; sinon observations/règles clairement identifiées. GET ne lance plus d'entraînement. Prévision de corrélation désactivée en attente d'une validation temporelle. | Implémenté, tests locaux ; performance réelle non démontrée. |
| P1 | Les horizons courts, moyens et longs partageaient des signaux ou historiques inadéquats. | Force relative sur 10/21/63 séances et momentum sectoriel sur 42/126/252 séances, selon l'horizon ; fenêtre de collecte correspondante ; signal absent si données insuffisantes. | Implémenté, tests locaux. |
| P1 | Les lectures ML publiques n'exigeaient pas systématiquement l'identité utilisateur, et un paramètre GET pouvait forcer un réentraînement. | Identité exigée sur les routes ML Stock Market ; réentraînement réservé à la route administrateur POST. | Implémenté, tests locaux. |
| P1 | « Confiance » pouvait être comprise comme probabilité de rendement ; les achats pouvaient dépasser le cash connu. | Libellé « signal agreement », plafonnement par le cash confirmé, absence de taille d'achat si cash inconnu. | Implémenté, tests locaux. |
| P1 | Des opportunités pouvaient mélanger secteurs et géographies, inclure des titres déjà détenus ou non cotés dans les données, et suggérer des ventes sur dates d'achat supposées. | Comparaison sectorielle uniquement, secteur fournisseur contrôlé, titres détenus/non tarifés exclus ; ventes proposées seulement avec date d'achat vérifiée et motif fort. | Implémenté, tests locaux. |
| P2 | La diversification était déduite de la seule volatilité, et une valeur neutre pouvait masquer des données fondamentales manquantes. | Composantes non vérifiées laissées indisponibles ; poids des composantes disponibles renormalisés ; « confiance » opportunité correspond à la couverture de données. | Implémenté, tests locaux. |

## Pertinence des opportunités sur 1 à 3 mois

Le nouvel horizon court privilégie une observation récente du prix relatif et du secteur. Il supprime les signaux si l'historique manque, vérifie que le titre appartient au secteur indiqué par le fournisseur et évite de proposer un titre déjà détenu. Le classement reste un **filtre d'étude**, pas une prévision de rendement : il manque encore un test historique sans biais sur cet horizon, les coûts de transaction, les événements de résultats à venir et une comparaison à des alternatives investissables adaptées au compte.

## Plan de suite, dans l'ordre

1. **P0 — Rapprochement Saxo réel.** Les séries FX historiques et les contrats de devise sont implémentés. Valider désormais sur un export Saxo réel sélectionné et une réponse API réelle, incluant cash, compte multidevise et titres CHF/EUR/USD ; vérifier les valeurs USD, les cotations en pence et les jours fériés. Les limites de fraîcheur sont des règles techniques à calibrer par marché.
2. **P0 — Parcours réel.** Vérifier Saxo CSV et API, les trois pages, les états sans données et les droits utilisateur sur un environnement de test. Mesurer les temps de réponse et la cohérence entre positions, analytics et recommandations.
3. **P1 — Validation ML.** Exécuter des réentraînements glissants hors échantillon avec embargo adapté à chaque horizon, comparaison à des références simples, calibration, stabilité temporelle et suivi de dérive. N'afficher une prévision que si elle gagne sur ces critères et que le modèle reste récent. Ajouter la même preuve avant de réactiver la corrélation.
4. **P1 — Qualité des recommandations.** Tester séparément les horizons d’opportunités 1–3 mois, 6–12 mois et 2–3 ans, et les horizons de recommandations 1–2 semaines, 1 mois et 3–6 mois avec rendements nets de coûts, drawdown, turnover, couverture de l'univers, comparaison à un ETF de référence et intervalles d'incertitude. Définir des seuils minimums avant qu'un classement soit présenté comme exploitable.
5. **P1 — Univers et contexte.** Remplacer la liste statique de candidats par un univers investissable propre au compte (place, devise, liquidité, type d'instrument). Ajouter calendriers de résultats/dividendes, données fondamentales horodatées et contrôles de concentration et de corrélation avec les positions réelles.
6. **P2 — Exploitation.** Versionner les hypothèses et les modèles, conserver source/horodatage/couverture dans les réponses et l'interface, surveiller erreurs fournisseur et qualité du cache, puis tester la charge et les temps limites.

## Reprise après crash : erreurs supplémentaires corrigées

| Sujet | Erreur vérifiée | Correction |
| --- | --- | --- |
| Rechargement ML | Après redémarrage, les métadonnées vides étaient contrôlées avant de charger le modèle enregistré ; sa calibration pouvait aussi changer au rechargement. | Chargement avant contrôle de qualité ; préservation de la température calibrée des modèles actions. |
| Saxo API | Le risque traitait le cache « positions + cash » comme une liste. Les valeurs de tous les comptes et les prix natifs étaient convertis comme des EUR. | Lecture du contrat réel du cache ; distinction devise du compte, valeur en devise de base et prix de cotation. Valeurs converties en USD, prix d’achat/courants conservés dans leur devise. Nouvelle version du cache Saxo. |
| Change | Les taux de secours fixes pouvaient être assimilés à des taux vérifiés. | Valorisation Saxo exigeant un taux fournisseur récent ; conversion des historiques avec taux datés, sans valeur future et avec report limité à trois jours calendaires. |
| Cotation | L’ISIN ou un symbole générique pouvait prendre priorité sur la place explicitement indiquée. | Place explicite prioritaire ; conflit et place inconnue rejetés ; devise du cours obtenue du fournisseur, normalisation des pence. |
| Observations | Une ligne fournisseur sans cours pouvait bloquer tout un historique ; des rendements de durées différentes pouvaient être comparés. | Exclusion des lignes invalides, sans remplissage de prix ; alignement du début et de la fin de chaque rendement. |
| Lots | Deux lots de même valeur pouvaient être assimilés à une ligne agrégée et une ligne de détail, puis l’un supprimé. | Suppression de cette déduction ; conservation de tous les lots, somme des valeurs et poids. |
| Signaux manquants | Des valeurs NaN/Inf devenaient des signaux neutres avec un accord artificiellement élevé. | Recommandation absente si signal obligatoire manquant ; couverture réduite si secteur manquant ; liste des titres non analysés renvoyée et affichée. |
| Montants recommandés | La place restante dans un secteur était traitée comme une limite de valeur totale du titre ; un plafonnement cash ne mettait pas à jour le pourcentage. | Limite appliquée à l’incrément ; montant et pourcentage cohérents avec le cash connu. |
| Décision finale | Un BUY devenu HOLD/SELL pouvait conserver son ancien montant d’achat et ses anciens objectifs. | Taille et conseil recalculés après contraintes ; anciens objectifs retirés après changement de décision. Instruments à levier laissés en revue pour le dimensionnement. |
| Force relative | Rendements natifs comparés à un benchmark USD, avec une séance manquante dans la fenêtre. | Comparaison en USD aux mêmes dates et exactement N intervalles. Prix des objectifs affichés dans leur devise native. |
| Positions courtes | Le filtre de montant positif pouvait masquer les positions vendeuses du risque global. | Rejet explicite des positions courtes/CFD et du cash emprunté sur ce parcours de risque long-only. |

### Validation réalisée

- Suite élargie : **195 tests réussis, 14 ignorés, 7 avertissements**, en 179 secondes. Les tests ignorés dépendent notamment de CSV Saxo locaux absents dans la copie isolée ; ils ne constituent pas une validation du compte réel.
- Après les dernières modifications : **19 tests réussis**, dont trois nouveaux cas (parcours complet de recommandation sans secteur, calibration conservée au rechargement, position courte non masquée). Ces 19 tests recouvrent en partie la suite précédente et ne doivent pas être additionnés intégralement.
- Syntaxe : 38 fichiers Python, neuf scripts intégrés aux trois pages et `fetcher.js` contrôlés. Contrôle des différences ciblées sans erreur d’espacement.
- Données publiques réelles : portefeuille **fictif** composé de NESN sur SIX et AAPL sur Nasdaq, avec benchmark SPY, change CHF/USD historique et cash. Calcul réussi avec **78 rendements alignés**, dernier cours commun au **24 septembre 2026**. La ligne Nestlé du 25 septembre était vide chez le fournisseur et a été exclue. Ce test valide le chemin technique, pas une rentabilité.
- Les avertissements observés concernent des dépréciations Starlette/httpx et des noms de champs Pydantic existants.

### Références des contrats de données

Les champs Saxo distinguent devise du compte, valeurs de base et valeurs d’instrument : [balances](https://www.developer.saxo/openapi/referencedocs/port/v1/balances/post__port__subscriptions/schema-balanceresponse), [positions](https://www.developer.saxo/openapi/referencedocs/port/v1/positions/get__port__me/schema-positionresponse), [valeurs en devise de base](https://www.developer.saxo/openapi/referencedocs/port/v1/netpositions/get__port__netpositionid/schema-netpositiondynamic). Le chargement de cours ajustés et de métadonnées de devise suit le [code du fournisseur yfinance](https://github.com/ranaroussi/yfinance/blob/main/yfinance/scrapers/history.py).

## Limites restant ouvertes

- Les tailles, seuils de vente, cibles sectorielles et scores sont encore des règles heuristiques ; ils ne prouvent pas un avantage financier.
- Les listes de candidats et certains noms sont statiques. La vérification du prix et du secteur réduit les erreurs, mais l’identité complète, la liquidité et l’éligibilité Saxo doivent être contrôlées.
- Le régime ML apprend des labels construits à partir des observations du marché. Battre la classe majoritaire mesure cette classification ; cela ne prouve pas un rendement futur supérieur.
- Une politique d’âge maximal des modèles, un suivi de dérive et des validations glissantes restent à implémenter. Les calendriers de résultats et le test des coûts restent ouverts.
- Le risque avec poids actuels est une simulation historique à poids constants. Ce n’est pas la performance réalisée du portefeuille ni un moteur de marge pour produits dérivés.
- Les parcours locaux authentifiés décrits ci-dessous fonctionnent dans le navigateur. Les états partiels, le compte Saxo API réel et la charge multi-utilisateur restent à vérifier.

## Réparation du serveur présenté pour les essais manuels

### Causes des erreurs signalées

- Le premier serveur sur le port 8081 utilisait une copie basée sur le commit `2a92a1be` du 13 septembre. La base courante du projet est `a792a644` du 26 septembre. Les corrections ont été reportées par fusion à trois versions dans une nouvelle copie isolée basée sur ce dernier commit ; les changements sans rapport ont été préservés.
- La copie de données de test était incomplète : les données/configurations crypto nécessaires au fournisseur n'y figuraient pas. La configuration locale complète a été reprise, sans afficher ni versionner ses secrets.
- L'identifiant de source générique `saxo:saxobank_csv` était envoyé comme nom de fichier. La sélection stricte du CSV échouait donc. Un résolveur partagé distingue désormais source configurée, véritable nom de CSV, entrée manuelle et API. Les pages attendent l'initialisation du contexte et réagissent au changement de source.
- Les erreurs HTTP perdaient leur description, produisant `http_error` ou `undefined`. Le détail utile est maintenant conservé.
- L'export historique contient Roche `ROG:xvtx`. Depuis le 17 mars 2026, la cotation est `ROP`. Le fournisseur ne fournit plus le cours courant de ROG. Le chargement applique cette correspondance documentée, garde le symbole d'origine et la provenance, et respecte la date d'effet. [Communication officielle Roche du 16 mars 2026](https://www.roche.com/investors/updates/inv-update-2026-03-16).
- L'affichage de volatilité retirait la place de cotation du symbole. Elle est maintenant conservée ; le modèle est chargé sous le symbole de la cotation résolue. Sans modèle validé, l'interface affiche une volatilité observée sur 30 séances, sans fausses prévisions à un et sept jours.

### Contrôles sur la nouvelle base

- **190 tests ciblés distincts réussis** : 28 tests de contrats, résolution de source, données, change, recommandations et risque ; 161 tests complémentaires ; un nouveau cas de résolution de la cotation du modèle. Les trois cas existants relancés avec ce dernier ne sont pas recomptés. Avertissements de dépendances existants uniquement.
- Contrôle syntaxique Python et scripts des pages ; contrôle des différences ciblées sans erreur d'espacement et absence de marqueurs de conflit.
- Connexion par le formulaire normal avec un compte de test local isolé, puis navigation réelle : sources crypto et actions actives, tableau de bord avec 30 positions, risque calculé avec 100 % de couverture des prix, matrice de corrélation à 30 titres, 30 recommandations et 18 résultats du scan d'opportunités « 1–3 Months ».
- Aucun appel de validation n'a passé d'ordre. Entraînements et tâches planifiées automatiques sont désactivés dans ce serveur de test. Le témoin global de gouvernance peut donc rester `STALE` ; cela ne décrit pas la fraîcheur des prix boursiers affichée séparément.
- Le portefeuille demeure celui du CSV du **18 janvier 2026** ; les cours communs de risque sont datés du **24 septembre 2026**. Pour analyser le portefeuille actuel, importer un export récent et son cash correspondant dans cette copie de test.
- La présence de 18 résultats valide le fonctionnement du scan ; elle ne démontre ni rendement futur ni pertinence financière hors échantillon. Les modèles sans validation restent identifiés comme observations/règles. La classification sectorielle des lignes historiques et le simulateur d'impact méritent encore un contrôle de couverture, notamment lorsque la catégorie `Other` domine.

## État de publication et suites

Les corrections ont été fusionnées dans `main` via la [PR #60](https://github.com/jackseg80/smartfolio/pull/60), commit `13bd9461`. Les contrôles GitHub de tests, lint et sécurité ont réussi. Aucun déploiement applicatif n'a été effectué. Le serveur local de validation écoute sur `http://127.0.0.1:8081` lorsqu'il est démarré ; son export Saxo est daté du 18 janvier 2026. Le rapprochement avec le compte actuel et un backtest hors échantillon restent nécessaires avant d’affirmer qu’une opportunité à 1–3 mois est pertinente financièrement.

# Régressions signalées lors du contrôle humain de l’aperçu

## Constat et causes vérifiées

Le contrôle précédent des pages par défaut était insuffisant pour valider les onglets et actions. Le retour utilisateur invalide la conclusion de disponibilité fonctionnelle générale.

- Journaux robot2 : POST Rebalance, DI Backtest et Optimization rejetés 403 par la protection de l’aperçu. Ces routes calculent des propositions/simulations ; elles ne passent pas d’ordres. Elles sont désormais autorisées exactement pour POST, pas PUT/DELETE.
- Taxonomie absente de l’aperçu : 102 alias de base. Au contrôle direct authentifié de production, `/taxonomy` renvoie 224 alias (le navigateur utilisateur indiquait 249, écart encore à expliquer). Montage du fichier de référence production en lecture seule dans l’aperçu.
- Risk Score pour la même source cointracking_api et fenêtres 365/90 : 71.5541 production, 69.6327 aperçu avant restauration. Attribution causale complète à vérifier après restauration, sans ajuster les scores pour les faire coïncider.
- Alertes 503 : moteur volontairement non initialisé. Restauration d’un stockage d’alertes isolé en standby, sans démarrer le scheduler ou la gouvernance. L’absence d’alertes dans cet espace n’est pas un historique de production.
- PUT implicites de source bourse pendant les lectures : bloqués par le montage protégé. Restauration de contexte sans persistance implicite ; changements de compte sur disque restent interdits.
- Optimization : ancien chargement CSV global sans user_id. Remplacement par get_unified_filtered_balances avec identité authentifiée et source exacte. L’ancien endpoint d’optimization backtest produisait chiffres fixes/aléatoires : réponse 501 explicite désormais.
- Cycle : sections retirées trop largement. Position descriptive et comparaisons/anatomie rétablies à partir des clôtures Bitcoin avec reçu vérifié. Seuls les cycles disposant de la clôture du halving sont normalisés ; pics rétrospectifs, cycle actuel incomplet, aucune date future prédite.
- Scans : lectures séquentielles et retries longs. Lectures publiques bornées à quatre simultanées ; attente navigateur limitée, sans répétition automatique de calcul coûteux.
- Stock Analytics : probabilités absentes indiquées explicitement ; délai ML adapté. Une sortie absente n’est pas remplacée par une prédiction artificielle.

## Tests exécutés avant reconstruction

- 93 tests ML, régressions et optimisation : succès.
- 34 tests contrats/recommandations/auth bourse : succès ; ALLOWED_HOSTS de test configuré.
- 12 tests protection aperçu : succès, dont calcul autorisé et autres méthodes interdites.
- 15 tests frontend ML/source : succès, configuration Jest ESM existante.
- Ruff : succès avant assemblage final.

## Contrôle fonctionnel restant

L’aperçu corrigé doit être contrôlé sur les parcours et boutons signalés, desktop et mobile. Les réponses HTTP seules ne constituent pas une preuve de rendu. Rapport de contrôle réel et différences de scores à ajouter après les essais. Aucun déploiement de production autorisé.

## Contrôle intermédiaire v6

- Après restauration de la taxonomie : 224 alias renvoyés par les deux APIs. Risk Score aperçu 71.5410 contre 71.5541 production, écart résiduel 0.0130 à caractériser.
- DI Backtest : deux POST réussis et 441 points dans le graphique equity, compte jack.
- Optimization : POST réussi, source cointracking_api, 36 poids tracés / 37 lignes de table.
- Cycle : position observée, trois lignes d’anatomie, 1440 points dans les graphiques comparatifs.
- Nouveaux contrôles exposent une session fraîche sans source bourse locale malgré une configuration serveur existante : restauration par GET de la source configurée, sans PUT.
- Rebalance consomme une liste de stratégies comme un dictionnaire, donnant des identifiants numériques. Normalisation par strategy.id et rendu immédiat des modèles statiques avant les calculs dynamiques.
- Alias Manager ajoutait les alias inconnus locaux aux entrées de la taxonomie en leur attribuant Others : compteur des alias enregistrés séparé des propositions non classifiées ; erreur API affichée sans liste fictive.
- Les captures privées restent uniquement dans cache/ml-preview sur robot2.


## Contrôle fonctionnel v7 et finitions v8

Les conclusions de disponibilité générale de v5 sont remplacées par ces contrôles après le retour utilisateur.

- 19 parcours authentifiés jack : 13 pages desktop et six mobile, widths 1440/390, source crypto cointracking_api et source bourse restaurée depuis la configuration serveur. Aucun débordement du document, aucune exception JavaScript, aucune région encore en chargement au relevé. Les images privées restent sur robot2 ; contrôle automatique du DOM et des actions, validation visuelle humaine encore requise.
- DI Backtest : 441 points et les deux calculs POST 200. Son fonctionnement est distinct de l'ancien backtest d'optimization, désormais explicitement 501 car les chiffres étaient fabriqués.
- Rebalance : POST 200, 11 éléments de résumé, 88 actions, 25 alias inconnus à revoir. Distribution produite en SVG, pas Chart.js : comptage des vrais segments confirmé au contrôle final. Les modifications de taxonomie restent bloquées dans l'aperçu.
- Optimization : 37 lignes de poids et 36 points tracés, source personnelle cointracking_api, succès du POST.
- Cycle : position disponible, trois lignes d'anatomie, 1440 points comparatifs. Les périodes non couvertes restent absentes et les pics sont rétrospectifs.
- Market Regimes : BTC et ETH ont chacun 286 observations tracées, sans erreur. Une règle économique ne fournit pas de probabilité évaluée : Confidence unavailable est attendu. Les légendes de confiance fictive et de fusion avec HMM ont été corrigées dans v8.
- Stock Analytics : 91 sections ML, disponibilité Available/Partial et raison explicite de l'absence de probabilités de régime.
- Recommendations : 30 lignes, source Saxo CSV personnelle. Les deux scans terminent et répondent 200. Opportunities ne produit aucun candidat ; une exposition industrielle non classifiée, notamment ETF, peut remplir un gap apparent. Une liste vide ne prouve pas un portefeuille équilibré. Raison affichée dans v8, sans invention de candidat.
- AI Dashboard : quatre onglets accessibles desktop/mobile. Le compteur indique les artefacts chargés globalement sur le serveur, pas un nombre de modèles prévisionnels utilisables sur les positions choisies.
- Alias : 224 références enregistrées, communes aux deux APIs. Les 25 propositions du dernier plan local expliquent les 249 de l'ancien navigateur ; elles restent séparées et non classifiées.
- Monitoring : environnement isolated_preview et scheduler intentionnellement désactivé ; pas de fausse dégradation de ce seul fait. Standby des alertes dans le stockage isolé, sans historique production.

### Comparaison privée et journaux

Comparaison conservée sur robot2, détails financiers non exportés. GRI et exposition par groupe identiques entre production et aperçu. Risk Score frais : 71.55327 contre 71.54007, différence 0.01321. Les cohortes historiques diffèrent pour EGLD3 (résolu EGLD), HMSTR, JUNO, TUSD et VANRY, dont les fichiers avaient des prix invalides. Les 119 historiques publics acceptés conservent exactement les empreintes de production. Aucun ajustement artificiel du score pour supprimer cet écart.

Journaux pendant les contrôles v7 : zéro traceback et zéro exception PermissionError/FileNotFoundError/ValueError/TimeoutError/ReadTimeout/HTTPStatusError. Seul statut HTTP en erreur : scheduler/health 503 intentionnel. Les journaux bruts restent sur robot2.

### Contrôles requis

- Python global : 3422 succès, 27 skipped, 30 warnings, couverture 51.35 % (seuil requis 30 %), 196.79 s.
- Jest global : 134 succès, 13 suites ; renouvelé après les textes v8.
- Linux dans le conteneur : 46 tests ML/régressions réussis. Le processus de test utilise ENVIRONMENT=staging, sans changer le serveur. Les essais avec production puis une valeur test non reconnue ont échoué sur la configuration de la fixture ; aucun changement de logique pour contourner la protection production.
- Protection aperçu : 12 contrôles ; méthodes d'écriture, entraînement et ordres bloqués, calculs explicitement autorisés.
- Les évaluations de volatilité et leurs 58 artefacts n'ont pas changé : les preuves causales restent valides. Les méthodes et limites sont décrites dans le protocole et le rapport de livraison.

Production non modifiée, aucun commit, fusion, push ou déploiement de production. Prochaine étape : validation humaine de l'aperçu, puis décision séparée concernant l'intégration et la publication.


### Complément au contrôle v8 : collecte externe

Le contrôle strict final v8 a correctement échoué sur Crypto Toolbox 502 : il ne faut pas ignorer cette erreur pour déclarer toutes les APIs saines. La collecte de production renvoie 200 mais scraping_failed=true, avec 30 indicateurs en cache ; elle ne constitue pas une collecte fraîche réussie. L'aperçu a révélé un transport Playwright fermé. La récupération pouvait conserver un objet navigateur déconnecté et ignorer le redémarrage. Correction v9 : lancement sérialisé, destruction du driver déconnecté avant redémarrage, nettoyage après échec. Trois tests dédiés vérifient récupération, appels concurrents et absence de données inventées à l'échec. La capacité externe reste indisponible si la collecte ou sa validation échoue ; ce n'est pas une prévision ML. Le résultat de la collecte après correction est relevé séparément.


## État final v9

- Collecte externe : HTTP 200, 30 indicateurs, cached=false, sans scraping_failed, sans copie du cache de production. Le cache servi en production reste signalé scraping_failed=true lors de la comparaison. La fraîcheur observée concerne cet essai, pas une garantie de disponibilité permanente du fournisseur.
- Python global après correction backend : 3425 succès, 27 skipped, 30 warnings, couverture 51.36 % / seuil 30 %, 215.25 s.
- Linux sur l'image finale : 61 tests ML/régressions/protection, succès. Frontend : 134 tests, succès ; Python/browser recovery n'altère pas le JavaScript vérifié en v8.
- Root et données production restent RO ; image de production et date de démarrage inchangées. Aperçu final smartfolio-ml-preview:20260930-v9. Aucun entraînement, ordre, intégration des prévisions aux allocations ou publication de production.
- Les historiques acceptés peuvent évoluer naturellement en production entre deux essais. L'exclusion documentée des prix invalides est maintenue, sans remplissage artificiel ni modification de formule pour aligner le score.
- La vérification humaine de l'ensemble des comportements et de la présentation reste à faire sur 8083. Les images privées et détails de comparaison financière restent sur robot2.


Attestation finale stricte, 30 septembre 2026 à 21:12 UTC : 19 parcours complets v7 réutilisés pour les fonctions inchangées et sept parcours ciblés desktop/mobile v9 réussis. Aucune erreur API inattendue, aucun débordement, valeurs de test et sources conformes. La seule réponse 503 attendue est celle du scheduler intentionnellement désactivé. Production : image et démarrage inchangés, healthz 200 ; aperçu v9 healthy, root et montages production RO. Espace libre 36.47 GiB. Les conclusions ne remplacent pas la validation humaine et ne prouvent pas la rentabilité future d'une prévision.
## Complément v10/v11 — second retour utilisateur et Admin (1er octobre 2026)

Les attestations v9 ne couvraient pas le second rendu du panneau Risk d’Analytics ni l’apparition du menu Admin sur l’origine LAN 8083. Elles ne suffisaient donc pas pour les problèmes signalés ensuite.

### Corrections

- Admin : suppression de la visibilité forcée par le port 8080/localhost et des anciennes clés `user_roles`. La navigation attend la vérification de session et utilise `userInfo.id/roles`. L’accès reste limité au rôle `admin` par l’API. Les appels Admin transmettent explicitement les headers d’authentification ; aucun rôle du registre n’est modifié.
- Analytics / Risk Dashboard / orchestrateur : paramètres communs (source personnelle, seuil configuré, historique 365 jours, rendements 90 jours, dual window, v2_active, fichier choisi). Le second renderer Analytics arrondit le score, conserve zéro et n’utilise plus les anciennes clés globales de scores. Le rendu principal lit le score de ses métriques API. Les légendes historiques ne prétendent plus représenter un cycle ou une fenêtre de volatilité de 30 jours.
- Comparaison fraîche sur robot2, mêmes paramètres : production 71.5537813157, aperçu v10 71.5407245705. L’ancien 87 de la capture n’est pas reproduit. Les détails financiers de cette comparaison restent sur robot2. Aucune formule ni donnée n’est modifiée pour forcer une égalité.
- Cycle : quatre lignes conservées, avec couverture Complete/Partial/Ongoing/Unavailable. Les drawdowns du cycle 2 sont accessibles indépendamment de la clôture au halving. Le cycle 1 reste sans série faute de données vérifiées ; la normalisation du cycle 2 reste indisponible faute d’ancre. Les anciens schémas de prix sans provenance ne sont pas réintroduits. La position présente temps observé, phase et indice heuristiques, méthode et limites explicites ; aucune date de pic future ni précision prédictive.
- Stock ML : tableau de 30 actifs pour le compte contrôlé, avec 7/30 jours et régime ; détails repliables pour raisons, séances cibles, méthode, validation et provenance. Plus de 91 grandes cartes répétitives.
- Données stock : conservation des acquisitions brutes et de leurs empreintes ; exclusion documentée de clôtures invalides en fin de série et de dates hors séance. Aucun remplissage de trous intérieurs. AGGS.SW et SLHN.SW redeviennent vérifiables ; quatre évaluations chronologiques retiennent EWMA. IWDA.AS, XGDU.MI et SMH.L restent non évaluables avec leurs séances manquantes. Les transformations/artefacts précédemment validés ne changent pas de code_version.
- Opportunities : classification calculée sur une copie, réutilisée pour le scan et le simulateur ; les positions originales ne sont pas modifiées. Allocation du simulateur cohérente avec celle du scan au lieu de Other 100 %. Onze secteurs sont décrits par allocation connue, cible et plage d’écart. Une exposition non classée peut rendre le gap indéterminé ; elle ne devient pas une recommandation d’achat. Les protections des ventes sont conservées.
- AI Dashboard : compteurs portant sur l’univers choisi : 12/50 prévisions disponibles, 6/25 actifs couverts et 7 diagnostics lors du contrôle crypto. Les artefacts chargés par le serveur restent dans Model Status. Les onglets sont conservés. Aucun agrégat de confiance ou Decision Index inventé.
- CSS v11 : en-tête et simulateur boursiers, contrôle de stress Analytics. Défilement local des tables et retour à la ligne des contrôles mobiles.

### Preuves déjà obtenues

- Windows global v10 : 3429 tests Python réussis, 27 skipped, 30 warnings ; couverture 51.48 % (seuil 30 %).
- JavaScript : 138 tests, 14 suites, réussis.
- Linux sur v10 : 61 tests de contrats/causalité/source/régressions réussis. Premier lancement hérité d’ENVIRONMENT=production : rejet correct de la fixture de capture privée ; tentative ENVIRONMENT=test refusée par la configuration ; reprise avec development uniquement pour le processus pytest, serveur inchangé.
- Linux : 62/62 artefacts chargés et inférés, 13 intervalles validés, zéro erreur. Empreinte canonique e4f990e0500052d28d65361280a6198c2d3a7a441c73908549145c4a21143c92 inchangée.
- v10, LAN 8083, session vérifiée jack sans rôles préinjectés : Admin visible sur les 14 passages ; Overview, Users, Cache et ML en consultation, HTTP 200. Cycle, ML Stock, allocation du simulateur et compteurs AI vérifiés desktop et mobile. Dix passages complets réussis.
- Le contrôle initial a également détecté deux débordements mobiles, corrigés en v11, et deux checks Risk prenant par erreur la réponse secondaire non paramétrée pour la réponse principale. Le contrôle final attend la réponse v2_active et le rendu, au lieu d’un délai fixe seul. Les passages concernés sont répétés ; les autres preuves sont réutilisées.
- Contrats HTML concernés après CSS : 23 tests réussis, 1 skipped, 1 warning.
- Le contrôle automatique a refusé une préparation pouvant être interprétée comme une exportation privée. Le contrôleur finalement utilisé ne conserve ni réponses financières ni captures ; seulement des booléens, compteurs, chemins/statuts et le score explicitement comparé.

### Environnement isolé

Aperçu final : smartfolio-ml-preview:20260930-v11, image sha256:55820f9f87f0f50ce93d4ab4ff03873da6096795e1531552e2b9f124dad85179. Paquet public 331 fichiers, 62 artefacts, SHA256 f2f7a5400a00328cfaad812574d3a69479672de153b79ccc8f0efca51edc4aeb. Robot2 dispose d’environ 36.17 GiB libres après cette image. Session de test conservée, mêmes secrets de session privés sur robot2. Les fichiers jack, registre et alias de production restent montés en lecture seule. Production : même image 0c724b8… et démarrage 2026-09-28T09:01:59.068663028Z, healthz 200. Aucun commit, push, fusion ou déploiement de production.

Les mutations Admin, l’entraînement, le scheduler et les opérations financières restent désactivés dans l’aperçu. La validation humaine de l’interface reste requise. L’outil interactif du navigateur demeure indisponible à cause de l’erreur ACL Windows ; les contrôles exécutés utilisent Playwright sur robot2 et ne prouvent pas l’intégralité du parcours de saisie du mot de passe dans le navigateur de l’utilisateur.
### Attestation finale v11

Les six reprises ciblées réussissent : Analytics Risk desktop/mobile affiche 72/100, égal au score arrondi de son API, et le drawdown affiché correspond aux métriques du même appel ; Risk Dashboard desktop/mobile utilise le périmètre commun. Les pages Stock Analytics et Opportunities mesurent chacune 390 px de largeur documentaire pour un viewport de 390 px. Zéro erreur JavaScript non gérée et zéro réponse HTTP inattendue sur ces passages. Les dix autres passages v10, portant sur les fonctions inchangées, restent valides ; les lectures Admin ont bien été réalisées sur LAN 8083 avec les rôles récupérés depuis /auth/session, pas préinjectés dans userInfo.

Douze tests de la protection d’aperçu réussissent dans l’image finale. Relecture finale : 62/62 artefacts disponibles et 13 intervalles validés. Logs v11 : aucune Traceback, PermissionError, FileNotFoundError ou TimeoutError ; seule réponse 503 du scheduler volontairement désactivé. Production et montages RO réattestés, root du conteneur RO, état healthy et pages de connexion HTTP 200. Aucun changement de production, aucune écriture Admin/financière, aucune fusion et aucun entraînement déclenché par les consultations.

Les séries antérieures à la couverture vérifiée, les séances intérieures manquantes, les probabilités prédictives non évaluées et les gaps sectoriels indéterminés restent explicitement limités. Ces résultats techniques ne remplacent ni le test humain du navigateur de jack, ni une preuve de performance financière future.

## Revue du 1 octobre - cycles, fraicheur et erreurs de logs

- Comparaison authentifiee des DOM production 8080 et apercu v11, par lecture sur robot2 : frise narrative retiree, table 10 lignes devenue 4 cycles, deux premieres courbes absentes. Le graphique principal conserve ses 4287 observations dans les deux versions. La production affichait des phases futures comme deja acquises ; cet affichage n'est pas une preuve predictive.
- Reference Coin Metrics Community PriceUSD : 2010-07-18 a 2026-05-23, 5789 jours consecutifs, URL/empreinte/recu conserves. Cette archive est utilisee uniquement pour les cycles termines 1-3. Le cycle 4 garde exclusivement les clotures Binance USDT. Aucun raccord de fournisseurs au milieu d'un cycle, aucune utilisation de cette archive pour les forecasts.
- Frise graphique heuristique restauree (bornes explicites en mois). Comparaison quatre cycles, drawdowns causaux, table 13 mesures par cycle : ancrage, prix/dates extremes observes, durees, rendements, dernier prix et fournisseur. La courbe normalisee utilise une echelle logarithmique pour garder les quatre courbes lisibles. Les extrema historiques restent retrospectifs.
- Forecasts crypto : l'acquisition precedente etait figee au 29 septembre. Dataset d'evaluation immuable + extension publique complete des observations dans un cache propre, recue et empreintee, sans entrainement lors des GET. Chaque prefixe historique doit rester exactement identique. Provenance distingue dataset d'evaluation et dataset d'inference. Erreur fournisseur ou donnees invalides conservent la vraie date, et la barriere de fraicheur reste active. Les consultations concurrentes sont bornees et les echecs HTTP ne repetent pas l'acquisition pendant une minute.
- Les 62 artefacts ont ete reevalues explicitement avec le meme protocole et les memes acquisitions gelees, puis publies uniquement dans le checkout isole. BTC 7/30 jours disponible avec observation du 30 septembre en verification locale ; aucune precision directionnelle n'est revendiquee.
- Advanced Analytics : suppression du connecteur incompatible user_id et de toutes les performances simulees. Source authentifiee selectionnee, historique commun sans remplissage, base normalisee 100, annualisation crypto 365, mois composes et drawdowns reels y compris non recuperes. Metadonnees explicites : reconstruction retrospective a poids actuels, couverture partielle possible, pas NAV reel ni backtest. Comparaison de strategies indisponible sans serie validee.
- Mapping Yahoo partage pour SLHN.SW et WRDUSW.SW (ligne CHF connue), au lieu de SLHN/WRDUSW_CHF.SW. Le cours de WRDUSW en USD n'est pas suppose interchangeable avec une cotation CHF.
- CryptoToolbox : lecture publique directe confirme les placeholders zeros a t=0 et les chiffres charges a t=10s. Attente d'hydratation remplace le delai fixe 0.5s ; le rejet d'une vraie reponse invalide reste en place.
- Market Opportunities ne fait pas l'objet d'une nouvelle recherche dans ce lot. La classification incomplete et le besoin utilisateur de recommandations restent a traiter dans une discussion dediee.

Source historique et attribution : Coin Metrics Community, https://github.com/coinmetrics/data ; licence CC BY-NC 4.0. Prix USD de reference, differents des clotures d'une place Binance USDT. L'archive n'est pas actualisee au 1 octobre ; elle ne fournit donc jamais le cycle courant.

### Verification complementaire stocks et observation du 1 octobre

- L'extension publique applique aussi le calendrier des seances aux actions/ETF. Une acquisition Yahoo ajustee recente doit retrouver les clotures de la periode de recouvrement (tolerance de representation numerique uniquement) : une revision de split/dividende est refusee et requiert une reevaluation explicite. Pas de remplissage des seances absentes. AAPL atteint le 30 septembre par cette extension, sans changer le dataset gele.
- Les preuves fournisseur nouvelles sont conservees dans le cache public prive a l'aperçu (reponse JSON Binance, observations CSV Yahoo du SDK), avec empreinte et version SDK. Aucune position personnelle n'y est enregistree.
- Controles locaux : 3443 tests globaux passes, 27 ignores, couverture 51.88% (seuil 30%). 138 tests JavaScript / 14 suites passes. 67 tests ML/source/regressions passes avant l'ajout stocks ; 15 tests du lot du 1 octobre passent avec le controle de revisions Yahoo et de seances manquantes.
- La consultation Admin reste protegee par la session authentifiee et le role existant ; les ecritures restent bloquees dans l'aperçu. Aucun changement en production 8080.

### Controle authentifie de l'image et correction du dernier historique perime

- Premier controle UI v12 : cycles et Admin passent en 1440/390, quatre courbes [1319, 1402, 1440, 894] observations, 13 lignes de mesures, frise 5 reperes. Le test AI capturait un appel legacy de benchmarks (6 forecasts) ; le controle a ete corrige pour exiger explicitement le mode portfolio et ses 25 actifs.
- Advanced revele un historique de position arrete au 13 fevrier 2026 : son intersection forcait toutes les positions dans une vieille fenetre. Le cohort filtre maintenant les observations perimees, exige une profondeur minimale explicite et coupe avant le dernier trou, sans remplissage ni lecture future.
- Verification sur robot2 avec jack, source cointracking_api, dans un processus de lecture du code corrige : 348 rendements communs, du 17 octobre 2025 au 30 septembre 2026, couverture en valeur courante 95.4278%, un historique perime exclu. Ces dates, le nombre d'observations, la couverture et le caractere retrospectif sont retournes par l'API et affiches. Le cache historique n'est pas certifie comme dataset de forecasting.
- Collecteur CryptoToolbox force en GET de lecture : HTTP 200, 30 indicateurs, cache false, scraping_failed false (1.1s sur ce controle). Les zéros de chargement ne sont plus publies.
- Controles globaux avant la derniere correction de cohorte : 3446 passes / 27 ignores, couverture 51.93%. 84 passes / 1 ignore sous Linux, 138 tests JavaScript. Le test supplementaire sur une position a historique perime passe dans les 18 tests du lot.

### Attestation finale du lot - apercu v12 (1 octobre 2026)

- Image en service : sha256:d989d9d1131f313658f449b9ab857cadffd409472d5876a5276147b891653315. Paquet : 89c1b9e4d0d4192a766f960af8e019246cf65b515a942943609fe9aaec9dbcc5 (341 fichiers, 62 artefacts). Les preuves finales sont externes au paquet de code, relevees apres son lancement.
- Cycles, AI Dashboard et Analytics : controles authentifies jack en desktop 1440 et mobile 390 passent, sans debordement ni erreur JavaScript. AI en mode portfolio : 25 lignes, 12/50 forecasts disponibles, BTC/ETH 7/30 jours disponibles, observation du 30 septembre, identites dataset d'evaluation/d'inference distinctes. Aucun resultat de benchmark utilise comme preuve du portefeuille.
- Admin : Users, Cache et ML repondent 200 en desktop/mobile. Le controle attend maintenant la fin de l'initialisation authentifiee du module avant de cliquer. Aucun changement de role et aucune ecriture Admin.
- Advanced : HTTP 200, source cointracking_api, methode actuelle et couverture explicites. L'historique perime et le trou ne sont pas remplis : fenetre continue 348 rendements, couverture 95.4278%, etat Partial. Date et methode sont visibles.
- Cotations corrigees testees en lecture publique sur robot2 : SLHN.SW et WRDUSW.SW renvoient chacun un historique et la devise CHF. Leur derniere barre fournisseur porte le 1 octobre ; la chaine previsionnelle exclut volontairement le jour UTC incomplet et reste sur la cloture complete du 30 septembre.
- Tests definitifs : 3447 passes / 27 ignores en Python, couverture 51.96%; 138 tests JavaScript / 14 suites; 85 passes / 1 ignore en Linux cible. 18 tests du lot couvrent extensions, prefixe gele, erreurs fournisseur, sessions, revisions de prix ajustes, isolation utilisateur/source, absence de mock et exclusion des historiques perimes.
- Logs du conteneur courant sur les 12 minutes controlees : 0 lignes ERROR, 0 Traceback, aucune recurrence des trois familles signalees (signature balances, alias Yahoo, zeros de scraping). Controle HTTP des pages : aucune erreur dans les lectures retenues.
- Production 8080 : image sha256:0c724b8b8ff87d73885bf0e109276c7b1da2b211f3c9c497b33f9bcda544f2aa et demarrage 2026-09-28T09:01:59.068663028Z inchanges, health HTTP 200. Apercu healthy, donnees de production montees en lecture seule, rootfs RO, UID 1000, limite 3 GiB / 2 CPU. 35.68 GiB libres environ, sans suppression d'image.
- Le lanceur attend desormais healthz 200 avant d'annoncer l'URL, pour eviter un controle premature pendant l'initialisation. Les captures privees du navigateur et les payloads financiers n'ont pas ete exportes.

Limites maintenues : six actifs du top 25 sont couverts par les previsions de volatilite. Aucun modele de prix/direction, aucune probabilite directionnelle, aucun proxy de jeton enveloppe et aucune integration d'allocation ne sont ajoutes. Market Opportunities reste un travail distinct sur la couverture sectorielle et le besoin de recommandations. La verification fonctionnelle de ce lot ne remplace pas la validation visuelle finale de l'utilisateur ni une autorisation de publication en production.

## Chargement ML bourse et encodage - 1 octobre 2026 (apercu v13)

### Cause reproduite et corrections

- Avec jack et son CSV Saxo configure, la requete portfolio stocks depassait le timeout de 60 secondes. Une requete equivalente avec un delai plus long renvoyait HTTP 200 et 91 resultats (30 actifs x trois sorties, plus la correlation). Le message generique masquait donc un timeout, pas une absence de toutes les predictions.
- Les calendriers exchange_calendars etaient reconstruits pour chaque lecture, puis relus pour le diagnostic, les deux horizons et la correlation. Le cache borne de 64 calendriers publics utilise des cles comprenant la place et les bornes exactes; le changement de jour change la cle. Aucun prix, portefeuille, resultat personnel ni prediction n'est mis dans ce cache. Les controles de hash, prefixe historique, sessions completes et revisions Yahoo restent appliques.
- Nouveau composant stock-ml-insights.js : etat Loading explicite, erreur distinguee pour session expiree, acces refuse, timeout/reseau ou HTTP, verification compte/source/marche/scope, rejet d'une reponse tardive apres changement de compte/source/fichier. Une erreur renvoie false au gestionnaire de sections afin que l'onglet puisse reessayer. Les chiffres precedents sont retires lors du chargement et d'une panne.
- Les sequences UTF-8 corrompues etaient litterales dans bourse-analytics.html (points de suspension, tirets, fleches, separateurs). Elles sont remplacees par des signes ASCII. Les messages de repli ML visibles encore corrompus dans AI Dashboard et Cycle Analysis sont nettoyes et affiches en anglais.
- Aucun changement des calculs de prevision, des artefacts evalues ou des regles d'allocation. Empreinte canonique conservee : 753382359cb9d21bfa992304d1acc27903585c433451a699be449cad8d2eabd1.

### Preuves finales

- Image : sha256:88d929489739ef104a7aeec255af1f3366b7424315db658c87611b99682479fe. Paquet v13 : 460254fd4f27565ca1e7525f6952bce07986e0c56045a4e8155503ef8005e8b7, 344 fichiers, 62 artefacts; aucune donnee utilisateur ni cle dans l'image.
- Controle navigateur authentifie jack sur l'image finale, desktop 1440 et mobile 390 : compte, source saxobank et CSV configure concordent. HTTP 200; 30 lignes d'actifs; 38/60 previsions de volatilite disponibles; statut Partial. Chargement mesure 45.76 s et 39.58 s. Aucun caractere corrompu dans les textes charges ou de chargement, aucune erreur JavaScript, aucun debordement horizontal.
- Python unit/integration : 3449 passes, 27 ignores, 30 avertissements, couverture 51.96%, seuil 30% respecte. Log : outputs/ml-reliability/global-pytest-stock-loading-v13-final.log. La premiere tentative sans ALLOWED_HOSTS a ete rejetee correctement; la collecte des anciens E2E necessitant un serveur local a ete arretee. La verification fonctionnelle utilise le navigateur authentifie sur robot2:8083.
- JavaScript : 151 tests passes / 15 suites. Les 13 nouveaux tests couvrent zero reel, resultat partiel, session expiree, acces refuse, timeout, erreur HTTP, reprise apres echec, changement utilisateur/source/fichier et rendu sur.
- Linux : 73 tests ML/regressions passent sur l'image avec la logique finale; apres nettoyage des seuls messages UI, 21 tests du lot passent sur l'image finale. Le test local supplementaire de textes visibles passe aussi. Les tests de calendrier comparent le cache au calendrier original, distinguent les places et verifient le changement de jour.
- Dernieres 12 minutes des logs du conteneur final : aucune ligne ERROR ni Traceback. Source privee et registry d'auth restent montes en lecture seule; aucun payload financier, mot de passe ou pixel prive exporte.
- Production 8080 : image et demarrage inchanges, health HTTP 200. Apercu healthy sur 8083, rootfs RO, UID 1000, 2 CPU / 3 GiB. Environ 35.4 GiB libres; aucune image supprimee. Checkout principal toujours a792a644; corrections conservees dans le checkout ML isole, aucun commit, fusion ou publication production.

Limites : le chargement bourse reste de l'ordre de 40-46 secondes; les 22 autres demandes de prevision restent indisponibles avec leurs raisons, sans estimation de remplacement. Les nouveaux resultats sont des volatilites annualisees a 7/30 jours, pas des previsions de cours ou de hausse.

### Relais propose pour une discussion Market Opportunities

- Objectif : produire un scan utile et explicable pour la source et le CSV Saxo selectionnes, avec donnees verifiees et limites visibles.
- Etat : le scan ne classe aucune opportunite. La ventilation sectorielle verifiee est d'environ 44%; 55.9% reste non ventile, notamment des fonds/ETF. Les intervalles de manque sont indetermines; zero resultat ne prouve pas un portefeuille equilibre. Cette mesure vient des controles et retours precedents, sans nouveau scan dans ce lot.
- Decisions : ne pas deviner la composition des fonds ni convertir les budgets sectoriels en ordres; conserver la coherence source, fichier, devise et date.
- Fichiers utiles : docs/MARKET_OPPORTUNITIES_SYSTEM.md, api/ml_bourse_endpoints.py, services/ml/bourse/opportunity_scanner.py, services/ml/bourse/recommendations_orchestrator.py, static/bourse-recommendations.html, et le present rapport.
- Points ouverts : decomposition sectorielle datee des fonds/ETF, mapping ISIN/place/devise, donnees marche par candidat, politique des cibles et criteres de classement/vente.
- Prochaine action : audit en lecture seule des classifications exactes et du chemin scan -> classements -> simulation, puis plan borne propre a ce module avant modification.

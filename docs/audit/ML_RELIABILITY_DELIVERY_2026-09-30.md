# Remise à niveau ML SmartFolio — livraison du 30 septembre 2026

## Résultat

Implémentation isolée sur la base réellement observée en production :
806338b522e89647c3f49260e0f418dbea1da761. Prévisualisation initiale : localhost:8082; aperçu connecté ensuite sur robot2:8083,
avec accord explicite pour un montage en lecture seule du compte jack.
Aucune publication, intégration aux allocations, nouvelle recherche directionnelle,
ni automatisation financière n'a été exécutée.

Sur 84 évaluations actif/horizon, 58 produisent une estimation
de volatilité validée rétrospectivement et 26 restent non évaluables.
Méthodes retenues : {'ridge': 12, 'ewma': 44, 'lstm': 2}. 12 intervalles à 90 % ont passé la couverture
de confirmation; les autres restent absents. Ces résultats ne prédisent pas un prix,
une hausse ou un rendement financier. Les raisons d'indisponibilité sont visibles.

## Lot 1 — référence, compte et écrans

### Observations de production, par lectures seules

- Image exécutée : sha256:0c724b8b8ff87d73885bf0e109276c7b1da2b211f3c9c497b33f9bcda544f2aa,
  démarrée le 28 septembre à 09:01:59 UTC; empreinte HTML AI Dashboard :
  ae03117de525b153309e8818b7eae8f2f020ebb90b5d7eb49414a013fff29932.
  Le HEAD du répertoire seul n'est pas une preuve de l'image; les deux ont été relevés.
- Baseline du 30 septembre à 11:45 UTC : 39 fichiers de modèles, 11 artefacts de
  volatilité présents et aucun modèle de volatilité chargé. Le monitoring affichait
  quatre entrées saines avec confiance 0,5 malgré aucune inférence enregistrée.
  Le HTML comportait 82,3 % et les prédictions étaient absentes.
- Contrôle authentifié à 13:15 UTC : compte jack, CoinTracking API, 191 lignes après filtrage du seuil USD;
  source Saxo CSV, 30 positions, import du 22 septembre. L'identité JWT et X-User
  ont été vérifiées dans la session utilisée pour ces lectures.
- La première démonstration locale utilisait demo et ne validait pas le périmètre
  personnel. Elle a été remplacée par le compte jack et la copie minimale expressément
  autorisée : symboles, ordre, sources et dates uniquement. Aucun montant, quantité,
  secret ou fichier de portefeuille n'a été copié.

### Corrections et preuves

Inventaire : [capacités et consommateurs](ML_RELIABILITY_INVENTORY_2026-09-30.md).
AI Dashboard conserve ses onglets, démarre sur les actifs du compte, expose 25/50/250
actifs et le nombre omis; les benchmarks sont explicites. Les états inconnus,
performances, dates et confiances fictives ont été retirés. Les chevauchements
d'icônes et le débordement mobile des tooltips ont été corrigés.

Les pages Analytics, Market Regimes, Cycle Analysis, Dashboard et bourse distinguent
diagnostics, probabilités d'état, corrélations historiques et indicateur externe.
Les textes ajoutés à l'interface sont en anglais. Les autres valeurs financières
du compte ne sont pas répliquées dans le preview minimal.

## Lot 2 — contrat et fonctionnement

Le service partagé fournit valeurs nulles, raisons, dates, unités, fournisseur,
dataset, version du code, empreinte et période d'entraînement. Les adaptateurs
legacy harmonisent predictions/prediction/volatility_forecast. Un succès HTTP
n'affirme plus que toutes les prédictions sont disponibles.

Le snapshot /api/ml/overview ne dépend pas d'un ensemble de gouvernance.
Utilisateur/source/fichier sélectionné sont vérifiés; aucun fichier plus récent,
autre source, token enveloppé ou ETF homonyme ne remplace silencieusement une donnée.
Les réponses retardées sont invalidées après changement de périmètre.

Les fichiers présents, chargements et inférences réussies ont des comptes distincts,
liés à l'empreinte courante. Chargement sécurisé et compatibilité des features :
échec explicite, aucune substitution fictive. GET et ouverture de page n'entraînent
rien. Entraînement réservé à une action administrateur; publication atomique et
scellée après validation des fichiers, métadonnées et reproduction de l'inférence.
Les nouveaux artefacts portent governance_eligible=False.

Les lectures coûteuses du service s'exécutent hors de la boucle asynchrone.
Dans le preview, Redis absent provoquait environ 22 secondes d'attente lors du
contrôle de session : Redis a été désactivé uniquement localement, JWT maintenu.
Ce réglage local ne constitue pas une modification de la production.

## Lot 3 — diagnostics et prévisions de risque

[Protocole gelé](ML_RELIABILITY_PROTOCOL_2026-09-30.md) :
clôtures quotidiennes vérifiées, cibles 7/30 jours calendaires, annualisation
365/252, première séance cible prévue en bourse, transformations entraînement seul,
purge par fin de cible, 730 jours minimum, calibration/test de 183 jours,
avance de 183 jours et 365 jours réservés à la confirmation.

Comparaison bornée : persistence, EWMA 0,94, Ridge alpha=1 et LSTM corrigé.
Un candidat appris doit gagner au moins 5 % de QLIKE, deux tiers des fenêtres,
respecter MAE et conserver son avantage en confirmation. Sinon la référence simple
est retenue. Les modèles directionnels, funding et hurdle rejetés ne sont pas relancés.

Les états HMM sont A–D sans mapping économique vérifié; leurs probabilités ne
prédisent pas une hausse. Les règles économiques et cycles sont descriptifs.
Les reconstructions historiques sont rétrospectives. Les corrélations historiques
restent utilisables; les prévisions Transformer restent expérimentales.

### Estimations retenues

QLIKE ci-dessous est celui de la confirmation (plus faible = meilleur).
Chaque rapport JSON conserve les scores des quatre candidats par fenêtre,
la purge, les périodes, le dataset et l'empreinte de l'artefact.

| Marché | Actif | 7 jours : méthode / QLIKE | 30 jours : méthode / QLIKE | Intervalles 90 % publiés |
|---|---|---|---|---|
| crypto | ADA | ewma / 0.5645 | ridge / 0.1245 | 7j |
| crypto | BCH | ridge / 0.6249 | ridge / 0.3022 | Aucun |
| crypto | BNB | ewma / 0.6795 | ewma / 0.6242 | Aucun |
| crypto | BTC | ridge / 0.4321 | ewma / 0.4740 | 7j |
| crypto | DOT | ewma / 0.8167 | ewma / 0.3288 | Aucun |
| crypto | ETH | ewma / 0.6410 | ewma / 0.4138 | 7j, 30j |
| crypto | LINK | ewma / 0.4323 | ewma / 0.2290 | 7j |
| crypto | LTC | ewma / 0.6892 | ewma / 0.3513 | Aucun |
| crypto | SOL | ewma / 0.4112 | ewma / 0.3440 | 7j, 30j |
| crypto | XRP | ewma / 0.7687 | ewma / 0.5705 | 7j |
| stocks | AAPL | ewma / 0.5276 | ridge / 0.2142 | Aucun |
| stocks | AGGS.SW | Unavailable | Unavailable | Aucun |
| stocks | AMD | ewma / 0.8094 | ridge / 0.2215 | 7j |
| stocks | BRK-B | ewma / 0.5367 | ridge / 0.1348 | Aucun |
| stocks | CHSPI.SW | Unavailable | Unavailable | Aucun |
| stocks | CRSP | ewma / 0.5537 | ridge / 0.0842 | Aucun |
| stocks | CSPX.AS | Unavailable | Unavailable | Aucun |
| stocks | FLXI.DE | Unavailable | Unavailable | Aucun |
| stocks | GLEN.L | Unavailable | Unavailable | Aucun |
| stocks | GOOGL | ewma / 0.4406 | ridge / 0.1820 | 7j |
| stocks | IWDA.AS | Unavailable | Unavailable | Aucun |
| stocks | KO | ewma / 0.5064 | ridge / 0.1736 | Aucun |
| stocks | META | ewma / 0.8191 | ewma / 0.4366 | Aucun |
| stocks | MSFT | ewma / 0.7720 | ridge / 0.4592 | Aucun |
| stocks | NVDA | ewma / 0.5035 | ewma / 0.0990 | Aucun |
| stocks | PLTR | lstm / 0.6522 | lstm / 0.1658 | 7j, 30j |
| stocks | QQQ | ewma / 0.5211 | ewma / 0.2434 | Aucun |
| stocks | SLHN.SW | Unavailable | Unavailable | Aucun |
| stocks | SMH.L | Unavailable | Unavailable | Aucun |
| stocks | SPY | ewma / 0.5332 | ewma / 0.2113 | Aucun |
| stocks | SSLN.L | Unavailable | Unavailable | Aucun |
| stocks | TSLA | ewma / 0.4004 | ewma / 0.1105 | Aucun |
| stocks | UBSG.SW | Unavailable | Unavailable | Aucun |
| stocks | UETW.DE | Unavailable | Unavailable | Aucun |
| stocks | VGK | ewma / 0.3920 | ewma / 0.2089 | Aucun |
| stocks | VWO | ewma / 0.5163 | ewma / 0.2516 | Aucun |
| stocks | WFRD | ewma / 0.3540 | ridge / 0.0399 | Aucun |
| stocks | WRDUSW_CHF.SW | Unavailable | Unavailable | Aucun |
| stocks | XGDU.MI | Unavailable | Unavailable | Aucun |
| stocks | XLI | ewma / 0.4354 | ewma / 0.1004 | Aucun |
| stocks | XLP | ewma / 0.2991 | ewma / 0.0889 | Aucun |
| stocks | XLU | ewma / 0.4510 | ewma / 0.0541 | Aucun |

### Conclusions et limites

Les deux LSTM retenus concernent PLTR, à 7 et 30 jours. Ce résultat spécifique
ne justifie pas d'étendre le réseau aux autres actifs ni une prévision de prix.

BTC à 7 jours retient Ridge : gain QLIKE de développement 8,39 %, 8 fenêtres
gagnantes sur 10 et gain de confirmation 13,66 %. BTC à 30 jours conserve EWMA :
le contrôle MAE empêche de retenir Ridge malgré son gain QLIKE. L'intervalle BTC
30 jours est omis (couverture 74,63 %). SPY/QQQ conservent EWMA; leurs intervalles
couvrent 100 % de la confirmation et sont omis car trop larges.

Les historiques boursiers refusés contiennent des clôtures invalides ou ne
satisfont pas les contrôles : ils ne sont ni remplis ni remplacés par un proxy.
Les autres symboles du compte sans acquisition/artefact vérifié restent indisponibles,
y compris les tokens enveloppés et les symboles à identité incertaine.

Cette confirmation est rétrospective; ses fenêtres contiennent des cibles qui se
chevauchent. Aucune significativité statistique ni rentabilité n'est revendiquée.
Plusieurs exécutions ont réparé purge/calendriers, rafraîchi les données,
vérifié les métadonnées, les empreintes et l'encodage. Aucun seuil ni hyperparamètre
n'a été modifié pour optimiser la confirmation après ces répétitions.

Les actions utilisent l'historique actuellement ajusté par le fournisseur pour
splits/dividendes, sans archive des révisions disponible à chaque date. Une preuve
historique de décision en temps réel n'est donc pas établie. Les artefacts appris
conservent leur période d'entraînement antérieure à la confirmation; aucun refit
sur cette confirmation n'a été publié.

## Lot 4 — contrôles et livraison

- Suite globale : 3 404 tests Python réussis, 27 ignorés, couverture 51,09 %.
  La correction finale du proxy fournisseur et le test d'empreinte sont inclus
  dans cette suite globale; 31 tests ciblés ont également réussi. 119 tests JavaScript, 11 suites.
- Ruff et mypy : réussis; OpenAPI anglais : 457 chemins.
  Bandit : zéro problème de sévérité moyenne/haute; pip-audit : aucune vulnérabilité connue.
- Contrats : sorties complètes/partielles/absentes/périmées/zéro, artefact incompatible,
  altération après chargement, chargement/API/session en échec.
- JWT réel, X-User discordant, session expirée et rôle administrateur sont testés.
  Deux utilisateurs, deux sources et deux CSV ont des contrôles d'isolation.
- Absence d'entraînement en lecture même avec force_retrain; normalisation,
  purge, mutation du futur, LSTM causal et reproduction des artefacts vérifiées.
- Les tests globaux ont nécessité l'autorisation des hôtes test/testserver.
  Les compteurs de limitation sont remis à zéro entre tests; la limitation reste
  active à l'intérieur d'un test. La production n'est pas assouplie.
- Les modifications étrangères du checkout original ont été préservées.
  Le report sur la base production a conservé les corrections de devises, source,
  retry et scoring. Les différences de fins de ligne héritées sont exclues du paquet.


### Vérification visuelle et dernières corrections

AI Dashboard, Stock Analytics, Market Regimes, Cycle Analysis, Unified Analytics
et Monitoring ont été contrôlés dans un seul onglet Chrome sous `jack`, à 390 et
1440 pixels. Aucun débordement horizontal de page ne subsiste; les tableaux peuvent
défiler à l'intérieur de leur carte. Les quatre onglets AI sont conservés.

L'aperçu ML personnel utilise uniquement la capture privée consentie des symboles,
ordre, sources et dates, acquise le 30 septembre à 13:15 UTC. Ce n'est pas une connexion
au portefeuille en direct. Aucun montant, quantité, profil complet ou clé fournisseur
n'a été copié. Stock Analytics a donc été vérifié dans son état explicite « source
non configurée », sans prétendre reproduire les données financières complètes.

Une erreur d'authentification CoinGecko était propagée en 401 et provoquait à tort
une déconnexion SmartFolio. Le proxy traduit désormais les refus fournisseur 401/403
en 502; la session SmartFolio continue à être contrôlée normalement. Trois cas de
régression couvrent 401, 403 et 503. La session locale `jack` reste authentifiée.

Les critères boursiers du tableau Regimes correspondent aux règles sur 200 clôtures,
avec leur méthode descriptive. Aucun « Unavailable% » ni zéro d'export de substitution.
Sur Cycles, les performances alternatives codées en dur et les esquisses de prix sans
reçu fournisseur sont retirées. Leur état est Unavailable avec raison. L'indice d'ajustement
à trois événements historiques curated est affiché sur 100 comme heuristique rétrospective;
il ne devient ni une accuracy prévisionnelle ni une précision publiée aux consommateurs.
L'action manuelle d'analyse historique et l'action Alternatives ont été vérifiées.

Les preuves se trouvent dans outputs/ml-reliability : rapports par actif/horizon,
reçus SHA-256, logs des contrôles et checks-summary.json.
Le paquet portable possède son manifeste SHA-256 et exclut entièrement les données
utilisateur et la capture privée. [Procédure réversible](ML_RELIABILITY_RELEASE_2026-09-30.md).

La vérification Linux/Python 3.11 et le contrôle authentifié de l'image isolée ont
été exécutés sur robot2. Le [rapport de l'aperçu](ML_RELIABILITY_ROBOT2_PREVIEW_2026-09-30.md)
détaille les corrections de dépendances et de source révélées lors du test réel.
La validation humaine et les contrôles de publication restent nécessaires. Aucune publication en production
n'est autorisée par cette livraison. Le journal de consultations est disponible mais
désactivé; aucun entraînement récurrent ni automatisme financier n'est ajouté.

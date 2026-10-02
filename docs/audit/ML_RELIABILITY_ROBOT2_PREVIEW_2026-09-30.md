# Aperçu ML sur robot2 — 30 septembre 2026

## État et périmètre

Aperçu isolé : http://192.168.1.200:8083/static/ai-dashboard.html.
La production reste sur 8080, image
`sha256:0c724b8b8ff87d73885bf0e109276c7b1da2b211f3c9c497b33f9bcda544f2aa`,
démarrage de référence `2026-09-28T09:01:59.068663028Z`, base Git
`806338b522e89647c3f49260e0f418dbea1da761`.
Aucun commit, push, fusion, nettoyage Docker ou déploiement de production effectué.
Le retour utilisateur reste nécessaire : le fonctionnement technique ne certifie
pas toutes les fonctions financières du projet.

## Incident et corrections

La première proposition de test était prématurée. L'aperçu avait les historiques
vérifiés du nouveau service ML, mais pas le cache public d'historiques attendu
par les consommateurs historiques Risk et Crypto Regimes. L'artefact HMM ETH,
présent dans l'image exécutée en production, manquait également dans l'image de base.
Ces omissions expliquaient Risk 400, GRI indisponible et BTC/ETH sans historique.

- Copie strictement publique dans les propres volumes de l'aperçu : 119 historiques
  journaliers, 4,35 Mo, et les deux artefacts HMM BTC/ETH exécutés en production.
  Les empreintes et périodes sont enregistrées dans `public-dependency-receipt.json`
  sur robot2. Sept fichiers contenant des prix invalides sont exclus sans remplacement.
  Cette copie reproduit les dépendances de production; la provenance fournisseur de
  ce cache legacy n'est pas certifiée. Elle ne sert pas à valider les prévisions.
- Une session neuve ne chargeait pas la préférence de source dans le navigateur.
  `selected-source.js` lit `/api/users/sources`, vérifie l'identité et reprend seulement
  la source configurée et disponible. Une sélection explicite reste prioritaire;
  source absente, session invalide ou changement d'identité restent indisponibles.
- Le cache Risk distingue désormais toutes les fenêtres et options de couverture.
  Une consultation 30 jours ne peut pas fournir le résultat 365 jours à Advanced.
- La chronologie conserve les états HMM A–D après lissage, distincts des diagnostics
  économiques. Les états ne sont plus convertis implicitement en Bear/Bull.
- Retrait de deux badges aléatoires; aucune date d'observation fictive pour ces badges.
- Le GRI absent ne devient plus zéro. Les calculs manuels de stress sont autorisés
  dans l'aperçu, tout en bloquant les écritures financières et l'entraînement.
- Correction mobile du header et de la grille Risk : vérifiée sur contenu réel,
  document 390 px pour viewport 390 px après application du correctif CSS.

## Isolation et compte réel

Seuls `data/users/jack` et le registre de connexion existant sont montés en lecture
seule, après accord explicite. Ils restent sur robot2, hors de l'image. Redis, cache,
logs, volumes de modèles, clé JWT et réseau Docker de test sont séparés. Le système
racine est en lecture seule; les artefacts de risque validés ont aussi un montage RO.
L'API tourne avec UID/GID 1000, limite 3 Go RAM / 2 CPU. Aucun scheduler, entraînement,
ordre ni journal d'inférences n'est activé. Les cookies entrants et sortants sont
isolés pour éviter de modifier la session de production, partagée entre ports.

Contrôle authentifié : jack / CoinTracking API et CSV Saxo sélectionné (22 septembre,
30 positions). CoinTracking fournit 580 entrées brutes, dont 191 dépassent le seuil
USD du Risk Dashboard. Ce sont deux comptages différents; aucun compte demo ne sert
à cette vérification. La date d'observation du portefeuille API n'est pas exposée par
le fournisseur actuel et reste absente. Le dashboard dit désormais «Source entries».
Sur le périmètre sélectionné : 12/50 estimations crypto et 34/60 estimations actions
sont disponibles; les autres gardent leurs raisons d'indisponibilité.

## Preuves de fonctionnement

- 45 tests ciblés Windows, puis 54 tests Linux (contrats, cache, identité, HMM,
  garde de l'aperçu). Les 58 artefacts validés ont aussi produit 58 inférences lors
  du contrôle Linux; 12 intervalles publiables. Le code de calcul et ses artefacts
  sont inchangés par les corrections de présentation et de source.
- 128 tests frontend / 12 suites, dont neuf tests de restauration de source.
- Contrôle global Python : 3413 passent dans la première exécution, 27 sont sautés.
  Trois smoke échouaient car le host de leur client (`test`) n'était pas autorisé
  par l'environnement de test. Ils passent tous après correction de cette seule
  configuration. Couverture globale 51,14 %. Aucune règle de production assouplie.
- JWT manquant : 401; identité divergente : 403; session expirée : 401;
  écriture financière : 403.
- Navigateur headless sur robot2, compte jack, session neuve : six pages desktop,
  puis Risk / Market Regimes / Advanced en 390 px. Les erreurs principales sont
  résolues. GRI numériquement rendu, six scénarios présents; boutons Stress Test
  et Monte Carlo vérifiés (modal / graphique sans erreur).
- Les captures financières restent uniquement dans le cache privé de l'aperçu sur
  robot2. Seuls les résumés techniques sans montants sont utilisés dans le rapport.

## Limites et validations restantes

Les alertes automatiques sont désactivées dans l'aperçu : leurs lectures renvoient
503, sans fausse liste de production. Crypto Toolbox a également renvoyé 502 sur
une collecte externe; aucun résultat de remplacement n'est inventé. Les prédictions
partielles et les expériences rejetées ne deviennent pas des signaux d'allocation.

Les simulations de stress et Monte Carlo héritées restent des outils de scénarios,
sans validation de probabilités futures établie par ce chantier. Leur fonctionnement
technique n'est pas une validation prévisionnelle. La revue exhaustive de leurs
hypothèses, probabilités affichées et couverture ne doit pas être déclarée achevée
sur la base des seuls smoke tests de l'aperçu.

Une publication exige la validation humaine, la revue finale du périmètre et une
autorisation distincte. Le préfixe de test et les indisponibilités attendues restent
visibles pendant cette étape.

## Reproduire / arrêter

Scripts publics : `deploy/ml-preview/`. Le paquet de code vérifie les empreintes et
exclut les données utilisateur. `setup_robot2.py` vérifie host, image de production,
base Git, marge disque et labels avant toute action limitée aux conteneurs de test.
Les journaux techniques se trouvent dans `/home/jack/smartfolio-ml-preview-20260930`.
Les images précédentes sont conservées. Ne pas nettoyer Docker sans accord distinct.

Pour arrêter le test (sans toucher à la production), utiliser les noms exacts :
`docker stop smartfolio-ml-preview-api smartfolio-ml-preview-redis`.
Retour utilisateur à la production : http://192.168.1.200:8080.

## Attestation finale de l'aperçu

Image exécutée : `sha256:5a24fb1464f995b14d819321dabb7e0d3a9b0506ad0ec0c569e8dfbb03ca630e` (`smartfolio-ml-preview:20260930-v5`), état healthy.
Dernier contrôle : Risk, Market Regimes et Advanced, chacun à 1440 et 390 px,
compte jack, source restaurée depuis la préférence authentifiée. Six contrôles
réussis : aucune erreur visible, aucune exception JavaScript, aucun débordement
du document et aucune erreur API inattendue. Les seuls statuts en erreur relevés
sont ceux explicités ci-dessus (alertes 503; collecte Crypto Toolbox 502).
Production : même image et même StartedAt que la référence, healthz 200.
Login de l'aperçu : HTTP 200. Dossier jack et registre : montages RO; UID/GID
1000:1000; racine RO. Espace libre restant : 37.06 GiB.
Paquet public v5 : SHA-256 `51ce354aaccd826da42f459803c9583c3eb86ca2c6da81870618c00524c41b68`.
Kit public v5 : SHA-256 `d712a4b06e244ea50be0e40998e60b44b036f39ec0d47bebeb6dbacaef3744d8`.
Les captures privées restent sur robot2. La validation humaine est en attente.


## Révision après les régressions signalées

L'attestation v5 ci-dessus ne suffit pas pour les parcours réels ; elle est remplacée par le [rapport de corrections et contrôles fonctionnels](ML_RELIABILITY_REPORTED_REGRESSIONS_2026-09-30.md). Version finale préparée : smartfolio-ml-preview:20260930-v8. Source crypto du compte jack, source bourse restaurée, taxonomie de production en lecture seule. Validation humaine encore requise. Le scheduler et les écritures financières restent désactivés dans le test.


Attestation v8 : healthy, image sha256:25cd0266064701add0a74ca18a07ab052cde95714d1681fab7cab06804c0fbe1. Paquet code 7b52b445e7a2e3f9d7530785ac43fb2bb852dd8274dd9791be8c196b34ae5ae4. Production : image et démarrage strictement identiques à la référence, healthz 200. Montages production RO et racine RO, UID 1000:1000, limite 2 CPU/3 GiB, espace libre 36.61 GiB. 58 tests Linux (ML, régressions et protection) passent sur cette image. Les helpers de contrôle ont été affinés après le kit initial ; leurs résultats finaux sont conservés sur robot2.


Révision finale v9 après correction de récupération Playwright : image sha256:7e657cbba244c1fbc140328878013c2643bad2ee282dd0c96fb737fffccaef3f, paquet code 21b58862c2f4ed4ea194fbb8e26a557baccae2959dac94f973923eb54eacee84. Collecte externe fraîche réussie (30 indicateurs, HTTP 200) ; 3425 tests Python globaux et 61 tests Linux. Le [rapport des régressions](ML_RELIABILITY_REPORTED_REGRESSIONS_2026-09-30.md) remplace les conclusions v5/v8. Publication de production non autorisée ; validation humaine encore en attente.


Contrôle final strict à 21:12 UTC : sept parcours ciblés v9 et 19 parcours complets antérieurs réutilisés réussis, aucune erreur API inattendue, aperçu healthy et production inchangée, marge disque 36.47 GiB. Les attestations techniques sont disponibles dans outputs/ml-reliability/verification-summary-final.json et preview-final-state.json ; les captures et comparaisons financières détaillées restent sur robot2.

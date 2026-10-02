# Market Opportunities — correction de présentation, 2026-10-02

État : correction publiée sur l’aperçu existant 8083 et contrôlée sur le compte sélectionné. Ce document complète MARKET_OPPORTUNITIES_DEPLOYMENT_2026-10-02.md et actualise uniquement la version de présentation. Aucun changement de production, ordre financier, commit ou push.

## Cause et correction

Le renderer créait déjà des tables HTML mais sans styles de cellules, séparateurs, en-têtes ni alignement des nombres. Les nombreux paragraphes et motifs répétés rendaient le résultat difficile à parcourir. Les anciens contrôles de largeur ne suffisaient pas à valider la qualité visuelle.

Les quatre fichiers modifiés sont static/bourse-recommendations.html, static/components/market-opportunities.js, static/css/bourse-recommendations.css et static/tests/market-opportunities.test.js. Ajout de tables structurées avec en-têtes, alternance de lignes, alignement numérique, badges et liens de preuve ; cartes de chiffres clés ; panneaux dépliables de provenance, méthode, instruments et revues de positions. Les limites essentielles restent visibles. Styles limités à #opportunities et utilisant les variables du thème existant. Défilement horizontal interne et région accessible au clavier pour les tables étroites. Tous les libellés visibles restent anglais.

## Validation et limites

160 tests frontend réussis dans 16 suites ; 9 tests ciblés réussis après la dernière simplification du renderer. Le backend et les autres fichiers du chantier ML sont identiques au paquet r3 : comparaison des paquets, quatre fichiers de présentation/test changés, aucune addition ni suppression.

Contrôles navigateur réels à 1440 et 390 pixels : source et CSV sélectionnés concordants, 11 lignes de secteurs, 11 candidats et 30 revues présents ; absence de débordement de page et d’erreur JavaScript ; bordures, padding et alignement numérique appliqués ; ouvertures/fermetures des panneaux fonctionnelles. Le changement d’horizon invalide les résultats. Contrôles JWT, refus d’un autre CSV, export agrégé et refus de scénario avec valeurs manquantes conservés. Captures clair/sombre examinées ; contenus privés remplacés sur robot2 avant toute capture ou export d’image.

Les deux valeurs de positions manquantes continuent de bloquer les bornes globales et la simulation. La couverture reste celle du sous-ensemble valorisé. Aucun résultat ou score financier ajouté pour habiller l’écran.

## Publication et reprise

Image courante : smartfolio-ml-preview:market-opportunities-v2-20261002-ui, sha256:61d9334cf9cf2fbce98957d8e8d03e01927d306bfb1c77f51a3a3edce1538392 ; démarrage 2026-10-02T10:37:46.403192347Z. Paquet sha256:420470a65a98256349c4e4a67457436f79747abdc2b41f878f178331a2d931ad. 113 fichiers publics servis concordent avec le paquet (bannière d’aperçu prise en compte), aucun écart. Les deux /healthz répondent 200, zéro ERROR/traceback depuis le démarrage lors du contrôle final.

Production : image et date de démarrage inchangées. Environnement et montages de l’aperçu préservés, données privées toujours en lecture seule. Retour arrière intermédiaire conservé : smartfolio-ml-preview-api-before-ui-layout-20261002 ; le retour arrière v13 reste conservé séparément.

Preuves publiques/agrégées dans D:/Python/smartfolio/outputs/market-opportunities : ui-browser-checks-20261002.json, ui-runtime-20261002.json, ui-package-diff-20261002.json et captures *-redacted.png. Tests ciblés dans le checkout isolé : outputs/market-opportunities/ui-tests-20261002.log. Scripts de publication et validation : deploy_preview_20261002_ui.py et browser_validate_ui_20261002.py, copies sur robot2 dans /home/jack/smartfolio-ml-preview-20260930/.

Prochaine action éventuelle : retour utilisateur sur la présentation. Aucun nouveau contrôle backend requis sans changement de calcul ou de contrat. Production, commit et push nécessitent toujours une autorisation distincte.

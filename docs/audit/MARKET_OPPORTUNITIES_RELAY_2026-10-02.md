# Market Opportunities — reprise validée le 2 octobre 2026

La reprise locale a terminé les contrôles interrompus. Aucun déploiement, commit, push, ordre ou nouvel environnement. La prochaine étape est une validation API/visuelle après autorisation de publication sur l’aperçu existant 8083 ; production 8080 reste inchangée.

## État vérifié

- 72 tests ciblés réussis, dont alias suisse sans ISIN, mauvais ISIN, mauvaise devise et refus d’un historique CHF pour une cotation USD.
- Suite unitaire complète : 3286 réussis, 13 ignorés, 27 avertissements ; couverture 49,46 % supérieure au seuil de 30 %. Journal : outputs/market-opportunities/unit-with-coverage-2026-10-02.log.
- Interface : preuve précédente réutilisée, aucun changement JS depuis : 16 suites et 156 tests réussis ; outputs/market-opportunities/frontend-tests.log.
- 1440 fichiers de référence : aucune suppression ; seuls les sept fichiers déclarés diffèrent de la référence. Autres endpoints du fichier API octet pour octet identiques à la sauvegarde initiale.
- Exception historique conservée : config/score_registry.json a vu son last_updated régénéré par un ancien test. Original exact inconnu ; ne pas restaurer aveuglément. Test isolé dans tmp_path ; registre exclu du paquet.

## Compte réel — agrégats uniquement

Vérification du moteur en mémoire dans le conteneur d’aperçu existant, sans installation et sans export des données privées. Source et cash concordants avec le CSV sélectionné par jack. 30 positions, 30 ISIN, aucune date d’acquisition, 2 valorisations manquantes. Import 2026-09-22T10:00:36 ; date de valorisation financière inconnue ; cash 2026-09-22T10:01:15.693200Z. Valorisation EUR et cotations CHF/EUR/GBP/USD.

Couverture 52,5119 % du sous-ensemble valorisé ; 47,4881 % non classifié de ce sous-ensemble. 15 profils secondaires d’actions, 1 décomposition de fonds datée acceptée, 14 sans source acceptée. 11 évaluations sectorielles Indeterminate. 11 candidats examinés : 8 avec scores historiques calculés, 3 déjà détenus, 0 indisponible. Aucune vente automatique. Scénario réel bloqué par les valeurs manquantes. La variation légère de couverture depuis le 1er octobre ne prouve pas une amélioration de diversification.

Les limites restent visibles : composition de 14 fonds, dates financières, acquisitions/fiscalité/stop orders, politique personnelle, recouvrement des constituants, géographie économique et Risk Score validé indisponibles. Horizons et scores sont descriptifs de l’historique, sans rendement futur validé. Les simulations mécaniques ont été testées sur fixtures explicites, pas affichées comme des résultats du compte réel.

## Préparation et prochaine action

Paquet reconstruit sous outputs/ml-reliability/ml-reliability-market-opportunities-v2-2026-10-02.zip avec manifeste et SHA256. Les 356 entrées ont des empreintes vérifiées, le code correspond au checkout actuel et aucun chemin de données privées ou .env n’est inclus. La construction Docker et la validation visuelle du module déployé restent à effectuer après autorisation. Le paquet sans date précédent est périmé. Kit d’aperçu : outputs/market-opportunities/market-opportunities-preview-kit.zip. Préserver le guard courant et ses 13 POST existants ; seul le calcul scenario est ajouté. Ordres, entraînements et écritures de configuration restent refusés. Les fichiers privés restent montés en lecture seule sur robot2, hors images et rapports.

Après accord : appliquer ces deux paquets seulement à l’aperçu ML existant, préserver ses montages et la possibilité de revenir à l’image actuelle, puis vérifier jack/source/CSV, résultats et limites, mobile/desktop, erreurs et refus des écritures. Aucune publication de production, aucun commit/push ni ordre autorisé.

Fichiers : docs/audit/MARKET_OPPORTUNITIES_AUDIT_2026-10-01.md ; docs/MARKET_OPPORTUNITIES_SYSTEM.md ; services/ml/bourse/{market_snapshot,fund_exposure,market_analysis}.py ; static/components/market-opportunities.js ; api/ml_bourse_endpoints.py. Vérification agrégée : D:/Python/smartfolio/outputs/market-opportunities/verify_robot2.py. Utiliser .venv et node_modules existants du projet principal. Ne pas relancer l’installateur de staging.

## Autorisation et progression de publication

L’utilisateur a répondu « ok, go » à la publication de l’aperçu 8083 et aux contrôles. Une première image a été lancée, puis le contrôle navigateur a trouvé l’erreur de contrat safeFetch décrite dans l’audit. Correction locale scan/scénario et tests avec helper réel : 159 tests frontend / 16 suites réussis. Une image corrigée et une nouvelle validation du vrai parcours sont en préparation. Le conteneur initial v13 reste conservé ; production, données privées, clés et environnement n’ont pas changé. Les statistiques de paquet et empreintes sont à prendre dans le dernier manifeste et la preuve de publication, pas dans la première archive du 2 octobre.

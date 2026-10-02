# Market Opportunities — publication et relais final du 2 octobre 2026

État courant : module v2 publié et validé sur l’aperçu ML existant, http://192.168.1.200:8083/static/bourse-recommendations.html, onglet Market Opportunities. L’utilisateur a autorisé cette publication et ses contrôles par « ok, go ». Ce rapport remplace les mentions provisoires de publication/validation encore en préparation des relais précédents. Aucun déploiement de production, commit, push, ordre financier ou nouvel environnement.

## Publication vérifiée

- Image actuelle : sha256:4ffe67a32a9f33c4b64a1a1221edffdf8e7e945c6cafcd78c15cfec36fd6d0dd.
- Tag : smartfolio-ml-preview:market-opportunities-v2-20261002-r3.
- Démarrage : 2026-10-02T09:36:15.871443053Z ; état healthy ; healthz HTTP 200.
- Paquet : outputs/ml-reliability/ml-reliability-market-opportunities-v2-2026-10-02-r3.zip.
- SHA256 : 4bf5786e0a315fd64ab2e18c30654c8e96bb085a71d317e358c4acd51f475042 ; 356 entrées vérifiées, 62 artefacts publics. Les paquets sans r3 sont dépassés.
- 114 fichiers réellement servis comparés au paquet, y compris le guard ; transformation HTML connue limitée à l’injection idempotente du bandeau d’aperçu.
- Environnement et neuf montages identiques à l’aperçu initial ; données de jack et auth montées RO ; rootfs RO. Aucun fichier de compte ni .env ajouté à l’image. Aucun secret retourné dans les sorties.
- Scheduler et entraînement automatique désactivés. Guard : 13 POST existants conservés et seul calcul scenario ajouté ; PUT/PATCH/DELETE scenario refusés ; écritures financières/configuration refusées sur handler synthétique, isolation des cookies vérifiée. Aucun appel réel d’ordre financier.

Production inchangée : smartfolio-api, image sha256:0c724b8b8ff87d73885bf0e109276c7b1da2b211f3c9c497b33f9bcda544f2aa, démarrage 2026-09-28T09:01:59.068663028Z, en fonctionnement et healthz HTTP 200. Aucun ERROR, traceback ou échec Market Opportunities dans la fenêtre des logs de l’image finale ; aucun log brut exporté.

## Compte réel et interface

Vérifications sur jack, source bourse et CSV réellement sélectionnés, cash du même CSV. Aucune lecture d’un autre fichier de positions ni d’un compte démo. Les essais de mauvaise identité/fichier sont refusés avant chargement. Seuls les agrégats autorisés sont présents dans les rapports.

- 30 positions, 30 ISIN, aucune date d’acquisition et 2 positions sans valeur source.
- Couverture sectorielle : 52,51 % du sous-ensemble valorisé ; 47,49 % non ventilé dans ce sous-ensemble. Le portefeuille complet ne peut pas être mesuré puisque deux valeurs manquent.
- 15 classifications secondaires de sociétés, 1 ventilation de fonds datée acceptée, 14 fonds sans ventilation vérifiée acceptée.
- 11 évaluations sectorielles Indeterminate avec bornes complètes indisponibles ; zéro écart certain ne démontre pas l’équilibre.
- 11 ETF sectoriels examinés : 8 avec scores effectivement calculés sur 189 sessions USD, fenêtre finissant le 1er octobre 2026 ; 3 cotations déjà détenues exclues et expliquées. Ce sont des candidats descriptifs, pas des recommandations d’achat validées.
- Zéro vente automatique, raison explicite. Simulation réelle refusée avec explication des valorisations manquantes ; aucun résultat fictif de scénario montré.
- Import 2026-09-22T10:00:36, date financière de valorisation inconnue, cash 2026-09-22T10:01:15.693200Z ; valeurs EUR et cotations CHF/EUR/GBP/USD. Import, cash et données de marché ne sont pas synchrones ; limites visibles.

Desktop 1440 px et mobile 390 px : scan HTTP 200, source/identité concordantes, résultats et raisons affichés, méthode et absence de prévision expliquées, aucun pageerror ou débordement horizontal du document. Les tableaux ont un défilement interne sur petit écran. Un changement d’horizon supprime les anciens résultats. L’export contient uniquement les contrôles agrégés approuvés, aucun nom/ISIN/valeur de position, nom de CSV, hash privé ou clé.

Refus API validés : token absent, identité incohérente, autre CSV demandé, Europe comme cible sectorielle, écriture de configuration et scénario avec valeurs manquantes. Cache-Control inclut no-store avec no-cache et must-revalidate. Les données privées restent sur robot2. Les captures de contrôle sont masquées avant création pour supprimer fichier/cash et toutes les identités/tables d’instruments ; les vues masquées desktop/mobile ont été inspectées. Aucun pixel de position ou montant n’est exporté.

## Corrections et tests

Le contrôle déployé initial a révélé un défaut de transport : safeFetch retourne déjà le JSON dans data, tandis que le contrôleur attendait response.json(). Scan et scénario consomment désormais son vrai contrat et les erreurs visibles. Les mocks initiaux ont été remplacés/complétés par des tests avec le vrai helper, dont décodage tardif, succès, erreur et scénario synthétique. La suite frontend finale passe : 16 suites et 159 tests. La lisibilité mobile a été ajustée après inspection des captures. Aucun middleware commun n’a été modifié.

Backend : 3286 tests unitaires réussis, 13 ignorés, couverture 49,46 %. Preuve réutilisée car aucun changement Python depuis cette validation. 72 tests ciblés incluent les alias suisses et refus de substitutions de devise/ISIN. Les calculs de scénarios sont testés sur fixtures identifiées comme synthétiques, jamais présentées comme résultat réel.

Préservation : les autres endpoints API restent octet pour octet identiques à leur sauvegarde préintervention. Aucune suppression des 1440 fichiers de référence. Les modifications ML existantes ont été conservées dans le paquet et les artefacts publics montés ont été comparés avant bascule. Exception historique : un ancien test a régénéré last_updated dans config/score_registry.json, déjà modifié avant cette intervention. Original exact non récupéré ; test corrigé pour utiliser tmp_path, registre exclu de la livraison. Ne pas restaurer aveuglément ce fichier.

## Limites et prochaine étape

Données encore nécessaires : les deux valorisations manquantes avec date financière fiable, ventilations réelles/datées des 14 fonds restants, politique personnelle si souhaitée, acquisitions/lots fiscaux/stop orders et autres contraintes de vente. Les chevauchements de constituants et la géographie économique restent non calculés. Aucun Risk Score 0–100 validé ni rendement futur validé n’est ajouté. Il faut progresser par sources vérifiables et conserver les indisponibilités visibles, sans proxy silencieux. Aucune écriture dans les données de compte n’est autorisée par cette publication.

Retour arrière initial disponible : conteneur arrêté smartfolio-ml-preview-api-before-market-opportunities-20261002, image v13 sha256:88d929489739ef104a7aeec255af1f3366b7424315db658c87611b99682479fe. Conserver l’image actuelle avant toute restauration, contrôler les identités/labels, puis ne renommer/redémarrer que les conteneurs d’aperçu ; production reste intouchée. Attention : le rollback du script r3 restaure r2, tandis que ce conteneur initial permet de revenir à v13. Les conteneurs intermédiaires sont gardés arrêtés ; pas de nettoyage automatique.

## Reprise et fichiers utiles

- Diagnostic/méthode : docs/audit/MARKET_OPPORTUNITIES_AUDIT_2026-10-01.md ; docs/MARKET_OPPORTUNITIES_SYSTEM.md.
- Moteur : services/ml/bourse/{market_snapshot,fund_exposure,market_analysis}.py ; api/ml_bourse_endpoints.py.
- UI : static/components/market-opportunities.js ; static/bourse-recommendations.html.
- Preuves locales : outputs/market-opportunities/{runtime-final-2026-10-02,deployed-functional-checks-2026-10-02,package-verification-2026-10-02,validation-summary-2026-10-02}.json ; frontend-tests-2026-10-02.log ; unit-with-coverage-2026-10-02.log.
- Scripts publics de publication/contrôle : D:/Python/smartfolio/outputs/market-opportunities/deploy_preview_20261002_r3.py et browser_validate_20261002.py ; copies sur robot2 dans /home/jack/smartfolio-ml-preview-20260930/. Build/conclusions publics dans market-opportunities-20261002-r3/ ; données/captures privées restent sur robot2.
- Checkout existant : C:/Users/jacks/.codex/worktrees/ml-reliability-current/smartfolio. Réutiliser .venv et node_modules du projet principal ; ne pas relancer l’installateur de staging.

Prochaine action : retour utilisateur sur cet aperçu ou lot de données vérifiables restant. Ne pas publier en production, committer, pousser ni créer d’ordre sans autorisation distincte.

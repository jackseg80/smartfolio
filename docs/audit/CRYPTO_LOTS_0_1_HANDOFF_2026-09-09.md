# Relais de reprise - Lots crypto 0 et 1

Date de clôture : 9 septembre 2026

Ce document permet de reprendre le chantier sans recommencer l'audit. Le plan validé reste `CRYPTO_IMPLEMENTATION_PLAN_2026-09-09.md`. Les lots 0 et 1 sont terminés dans le répertoire de travail. Aucun commit, déploiement, ordre, activation automatique ou changement de stratégie financière réelle n'a été effectué.

## Objectif confirmé

- Portefeuille crypto spot uniquement, sans levier ni poids négatif.
- Horizon principal : quelques semaines à quelques mois.
- Recherche d'un meilleur compromis rendement/risque pour le portefeuille réel, avec rotations BTC, ETH, altcoins et stablecoins.
- Prévisions futures à 7 et 30 jours à construire et valider dans les lots 2 et 3.
- Revue hebdomadaire, propositions défensives possibles, simulation et export; exécution manuelle sur les exchanges.
- Une donnée manquante reste indisponible. Aucun score neutre, rendement, régime ou niveau de confiance n'est inventé pour remplir l'interface.

## Décisions d'architecture appliquées

1. Séparer les observations, les scores, la décision théorique, la proposition réalisable et l'exécution.
2. Conserver le Risk Score comme score positif de robustesse. Ne jamais utiliser `100 - Risk Score`.
3. Ne pas imposer de minimum caché par classe d'actifs ni protéger automatiquement les positions détenues.
4. Préserver intégralement un budget défensif élevé en stablecoins lorsqu'il est calculé, y compris 80 %.
5. Appliquer un plafond de mouvement par un facteur commun vers la cible. Les mouvements restent financés, de même sens que la cible et sans position négative.
6. Lier chaque allocation suggérée à l'utilisateur, la source, un snapshot et une date. Rejeter une proposition d'une autre identité, périmée ou calculée avant un changement matériel du portefeuille.
7. Donner la priorité à une politique explicitement manuelle. Les automatismes passent en `Freeze` lorsque les sorties vérifiées sont indisponibles.
8. Utiliser des rendements quotidiens, Ledoit-Wolf, 365 jours d'annualisation, une fenêtre cible de 365 jours et au moins 90 observations communes pour la covariance crypto personnelle.

## Lot 0 terminé - référence personnelle et contrats

Le module `services/decision/portfolio_reference.py` et la route authentifiée `GET /api/risk/portfolio-reference` fournissent une référence déterministe liée à l'utilisateur et à la source demandée :

- quantités, valeurs, plateformes, positions bloquées et positions non valorisées conservées;
- snapshot et date d'observation explicites;
- concentrations par actif et plateforme, plus HHI;
- covariance Ledoit-Wolf annualisée sur 365 jours;
- couverture historique par actif et en pourcentage du portefeuille original;
- contributions au risque limitées et nommées comme celles de la poche couverte;
- statut `complete`, `partial` ou `unavailable`, sans retrait silencieux ni renormalisation trompeuse;
- distinction explicite entre risque de prix des stablecoins et risques d'émetteur ou de liquidité non couverts.

Le service résout le portefeuille par `balance_service` avec l'identité authentifiée et une source explicite. Il ne choisit ni `demo`, ni un CSV global, ni le dernier fichier trouvé. Les anciens points d'entrée ML ne devinent plus un univers de portefeuille à partir de clés ou de fichiers locaux.

Les routes actives Risk qui lisent le portefeuille (`dashboard`, métriques, corrélation, stress tests, Monte Carlo, attribution et alertes) exigent elles aussi la source. Les contrôleurs concernés affichent une indisponibilité lorsqu'aucune source n'est sélectionnée au lieu de revenir silencieusement à CoinTracking.

Tests principaux : `tests/unit/test_portfolio_reference.py` et `tests/unit/test_ml_source_contract.py`.

## Lot 1 terminé - sécurisation et comptabilité

### Allocation et proposition

- Les cibles doivent être finies, non négatives et totaliser 100 %; une somme invalide est rejetée.
- Le budget stablecoins doit exister et n'est plus limité implicitement à 60 %.
- Les floors et l'incumbency sont désactivés par défaut.
- Le plafond de mouvement réduit proportionnellement tous les écarts; il ne devient jamais un plafond d'exposition risquée.
- La cible théorique reste distincte des positions courantes.
- La proposition porte l'utilisateur, la source, le snapshot, la composition par groupe et la date. Elle expire après deux heures et est rejetée après un changement matériel de valeur ou de composition.
- Les modèles de requête n'acceptent plus une proposition sans cibles.
- Une écriture de taxonomie échouée n'est plus annoncée comme sauvegardée localement.

### Données, ML et gouvernance

- Les rendements, volatilités, régimes, sentiments, corrélations, confiances et métriques de portefeuille fictifs ont été supprimés des chemins examinés.
- L'ancien endpoint ML unifié ne génère plus de sentiment ou de Risk Score aléatoire et ne renvoie plus `0.0` comme pseudo-prédiction lors d'un échec.
- Une prédiction sans calibrateur, métriques hors échantillon et métadonnées de qualité observées est rejetée.
- Un entraînement sans métadonnées vérifiables échoue au lieu de publier des performances inventées ou un statut `mock`.
- Le score blended exige cycle, on-chain et Risk Score; aucune composante manquante n'est remplacée par 50.
- Les indicateurs on-chain très majoritairement nuls sont rejetés. Un cache périmé de moins de deux heures est étiqueté `stale`; au-delà, il devient indisponible. La date observée n'est plus remplacée par l'heure d'affichage.
- Les états et signaux de gouvernance ne fabriquent plus de portefeuille, cycle, régime ou confiance. Une sortie ML incomplète entraîne `Freeze`, sauf politique manuelle explicite.
- Le `Risk Score` reste positif de bout en bout.

### Backtest

- Le moteur tient les quantités et le cash, laisse dériver les poids entre opérations, déduit les coûts et calcule les métriques depuis la même courbe nette.
- Chaque actif utilise sa propre série de prix. Une altcoin n'hérite plus du rendement BTC; une série manquante pour un actif détenu ou ciblé provoque une erreur explicite.
- Les poids initiaux sont explicites.
- La réplique historique conserve l'allocation courante lorsqu'un score ou une donnée causale manque.
- La sélection d'un score de cycle ou du Decision Index utilise uniquement une observation disponible à la date simulée; aucune recherche `nearest` ne peut choisir une observation future dans cette stratégie.

Tests principaux : `tests/unit/test_di_backtest_accounting.py`, `tests/unit/test_replica_abstention.py`, `tests/unit/test_policy_unavailable_signals.py`, `tests/unit/test_rebalance_proposal_identity.py`, `tests/unit/test_ml_unavailable_contract.py`, `tests/integration/test_signals_recompute_contract.py` et les tests JavaScript d'invariants/allocation/on-chain.

## Validations finales

- Suite Python unitaire complète exécutée pendant le chantier : **2942 réussis, 10 ignorés**. Six avertissements numériques connus, sans échec.
- Contrôle Python final ciblé après les dernières corrections : **218 réussis**, puis **5 réussis** sur le contrat de la route Risk.
- Suite JavaScript finale : **96 réussis**.
- Vérification de syntaxe réussie sur les fichiers JavaScript modifiés.
- `git diff --check` propre.

Le fichier `config/score_registry.json` peut apparaître modifié à cause de ses fins de ligne historiques et du cache d'index Git; son contenu est identique à `HEAD` (identifiant de blob Git `123e4e06e175a05f36ac59855e4c19e77dd23e7f`).

## Limites à conserver en tête

Les lots 0 et 1 rendent la chaîne plus sûre et auditable; ils ne prouvent pas encore une capacité prédictive. Les règles de cycle, le Decision Index et les allocations restent des heuristiques tant que les lots 2 à 4 n'ont pas produit une validation causale hors échantillon, après coûts, face à des références simples.

Des modules de simulation et d'anciennes stratégies de backtest contiennent encore des valeurs par défaut dans leur périmètre explicitement simulé. Les features historiques basées sur des proxys de prix, les recherches temporelles `nearest` restantes et la calibration des modèles appartiennent au lot 2. Elles ne doivent pas être présentées comme preuve de performance aujourd'hui.

## Point exact de reprise - Lot 2

1. Construire le jeu de données daté commun aux horizons 7 et 30 jours.
2. Définir les cibles marché, groupe et actif, ainsi que les références BTC et défensive.
3. Définir l'univers connu à chaque date, avec actifs délistés et prix de sortie manquants explicitement suivis.
4. Supprimer les proxys on-chain/sentiment reconstruits avec les prix dans les features de prévision.
5. Remplacer les recherches temporelles `nearest` restantes par des jointures causales `as-of` vers le passé.
6. Ajuster normalisation et sélection de variables sur l'entraînement seul, avec purge de 30 jours aux frontières.
7. Ajouter le test central : modifier le futur ne change aucune feature ni décision passée.
8. Documenter la couverture, les indisponibilités et les conventions avant de comparer des modèles au lot 3.

## Etat Git et propriété des fichiers

Changements locaux préexistants à préserver et ne pas attribuer à ce chantier :

- `data/users/demo/wealth/patrimoine.json`
- `.agents/`
- `AGENTS.md`

Les audits, le plan, ce relais, les nouveaux tests et les autres fichiers modifiés de la chaîne crypto appartiennent au chantier. Le dossier `outputs/audit-crypto-2026-09-09/` contient les sondes d'audit. Ne rien committer, mettre en stage, pousser, déployer ou exécuter sur un exchange sans autorisation explicite distincte.

## Contrôle fonctionnel du 10 septembre 2026

La comparaison entre la production du NUC et la branche de test a expliqué les écarts observés :

- la production Rebalance utilisait des données de démonstration (`source=mock-data`, prix simulés), tandis que la branche de test utilisait le portefeuille CoinTracking réel; ses regroupements et montants ne constituent donc pas une référence fiable pour cette comparaison;
- la production limitait encore implicitement les stablecoins à 60 %, même lorsque Risk recommandait 73 % ou davantage; la branche corrigée conserve le budget défensif calculé;
- les différences du DI Backtest proviennent du nouveau calcul par actif, de la dérive réelle des poids, des frais et des jointures causales. Le libellé de l'écart avec BTC précise désormais qu'il s'agit de points de pourcentage;
- les alertes `Failed to propose targets` et `Failed to apply strategy` venaient de l'application automatique d'une stratégie avant la fin du chargement. L'application automatique a été supprimée et les boutons restent indisponibles tant que leurs entrées ne sont pas complètes;
- les suffixes numériques ajoutés par CoinTracking sont résolus uniquement lorsque le symbole de base existe déjà dans la taxonomie. `FRAX`, `HYPE`, `PLUME`, `RARI` et `VVV` ont reçu un groupe explicite;
- une position inconnue reste désormais inchangée, ne peut pas financer un achat et impose une revue avant exécution. Rebalance affiche le montant protégé;
- `LLY`, `DOTA`, `PEON` et `TRUTH` ont été retirés par l'utilisateur dans CoinTracking. Le rafraîchissement du 10 septembre à 08:27 affichait encore 192 actifs : leur disparition doit être vérifiée après propagation de la source.

Validations locales après ces corrections : **70 tests Python ciblés réussis**, **101 tests JavaScript réussis** et `git diff --check` propre. Ces corrections sont encore locales au moment de cette note; elles doivent être committées, poussées puis installées sur l'environnement de test avant une nouvelle validation visuelle.

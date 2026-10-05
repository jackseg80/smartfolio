# Plan : valorisation Saxo CSV historique et actuelle

Date : 5 octobre 2026. Statut : plan validé par l'utilisateur ; implémentation locale et qualification réseau du portefeuille complet réalisées après autorisation spécifique. Aucun déploiement effectué.

## Résultat d'implémentation et validation

- `services/saxo_valuation_service.py` conserve les montants bruts du CSV sans FX actuel en mode export et fournit une valorisation commune en mode actuel.
- `services/saxo_quote_service.py` récupère des cotations Yahoo datées sans génération de cours ; cache de 30 minutes, mutualisation par verrou et délai de 60 secondes entre tentatives rapprochées.
- API `/api/saxo/valuation` protégée par identité requise et session/JWT ; concordance entre identité et contexte utilisateur contrôlée.
- Stock Market expose les deux modes, la fraîcheur et la couverture. Le rafraîchissement récupère les cours ; la minuterie ne travaille que lorsque la page est visible. Les changements de contexte annulent les résultats devenus obsolètes.
- Les résumés généraux et exports utilisent la même base actuelle ; les exports Stock Market reprennent le mode affiché. Les valeurs manquantes ne sont pas renommées en zéro. Le P&L Saxo quotidien n'est pas calculé par comparaison avec d'anciens snapshots CSV non qualifiés ; le P&L quotidien global indique alors sa portée crypto.
- La référence historique reste dans sa devise originale. Aucune conversion historique n'est inventée lorsque les taux historiques sont indisponibles. Le cash enregistré à une autre date est exclu du total historique et signalé.
- Les splits documentés sont appliqués aux quantités ; les CSV de plus de deux ans ou sans date/couverture suffisante restent marqués comme quantités non vérifiées. Les fusions/scissions nécessitent une vérification supplémentaire. Obligations et dérivés restent sur une référence d'export explicitement partielle tant qu'un modèle adapté n'est pas disponible.
- Qualification réelle de trois listings publics réussie : MSFT/USD, SLHN.SW/CHF et AGGS.SW/CHF, avec prix et horodatages du marché le 5 octobre 2026. Le test initial dans le bac à sable ne disposait pas d'accès réseau ; le contrôle autorisé hors bac à sable a réussi.
- 152 tests ciblés réussis après le traitement Roche, sept avertissements de dépréciation existants. Contrat navigateur isolé réussi : actuel/historique, EUR historique, valeur absente, rafraîchissement, desktop et mobile. Fixtures API/graphique/auth utilisées pour le navigateur ; l'authentification de l'API est contrôlée séparément par les tests.
- Preuves visuelles et contrat navigateur : `outputs/saxo-valuation/historical-desktop.png`, `historical-mobile.png`, `browser-contract.json`.
- La revue automatique a initialement refusé la transmission des symboles/date du portefeuille à Yahoo. L'utilisateur a ensuite donné son autorisation spécifique ; la qualification a été exécutée dans ce périmètre, sans identifiants Saxo ni opération de compte.
- Qualification locale du fichier sélectionné (export du 18 janvier 2026), à `2026-10-05T21:54:20Z` : 30 positions sur 30 actualisées, aucune valeur manquante, couverture complète selon les contrôles de cours, quantité et FX. 29 cours de clôture et un cours classé intraday par les métadonnées de séance Yahoo ; le TTL ne promet pas un cours en temps réel.
- Le premier essai avait échoué pour `ROG.SW`. [Roche confirme l'échange automatique 1:1 de ROG vers ROP le 17 mars 2026](https://www.roche.com/investors/updates/inv-update-2026-03-16). Le fournisseur traite désormais ce remplacement daté et sourcé, conserve le symbole original et expose le symbole coté `ROP.SW` et l'opération ; l'historique Yahoo couvre la date du CSV et la quantité est conservée.
- Somme positions + cash = total ; total de l'export actuel identique. Le CSV source reste identique (SHA-256 avant/après). Le mode export conserve l'EUR d'origine et s'exécute avec les appels cours/FX remplacés par des exceptions de contrôle : aucun appel actuel effectué. Cash d'une autre date exclu de la vue historique. Preuve : `outputs/saxo-valuation/local-selected-qualification.json` (métadonnées et statuts, sans montants).
- Report sur le dernier `main` dans une branche isolée le 6 octobre 2026 : conservation des évolutions du dashboard et utilisation des taux FX vérifiés, sans taux de référence présenté comme actuel. Validation sur cette base : 151 tests Python réussis, 14 tests nécessitant des CSV privés absents du checkout isolé ignorés ; sept tests du graphique réussis, contrat navigateur réussi, lint ciblé et contrôle OpenAPI anglais réussis. Les 152 tests et la qualification réelle ci-dessus concernent le checkout initial du 5 octobre, et ne prouvent pas un déploiement.
- Redémarrage manuel du backend requis pour charger la nouvelle API. Pas de commit, push, ordre ni déploiement réalisé.

Commandes de contrôle : activation de `.venv`, puis pytest ciblé sur les deux nouveaux fichiers de tests, les exports, FX, isolation Saxo et import du prix moyen, avec `--no-cov`; `node tests/e2e/saxo-valuation-check.cjs`; vérifications de syntaxe Python/JavaScript et `git diff --check`.

## Objectif et décisions proposées

Afficher par défaut la valorisation estimée des positions importées aux derniers cours disponibles, avec une vue `At export` fidèle au fichier. Conserver les deux bases distinctes et afficher leurs dates, devises, sources et limites.

- Sélecteur `Current valuation` / `At export` dans Stock Market.
- Cours récupérés à l'ouverture si le cache dépasse 30 minutes, puis toutes les 30 minutes lorsque la page est visible. Pas de tâche permanente nécessaire.
- Bouton `Refresh prices` qui demande réellement un renouvellement des cours ; limiter et mutualiser les appels pour éviter les rafraîchissements concurrents.
- Calcul serveur unique pour positions, total, cash, pondérations et allocations. Métadonnées conservées dans les résumés et exports.
- Mode API Saxo distinct ; les positions CSV ne deviennent pas un compte synchronisé.
- Interface et erreurs en anglais, documentation en français.

## Contrôles réalisés et preuves locales

Base examinée : HEAD `a792a644` du 26 septembre 2026. Le checkout comporte des modifications préexistantes étrangères à cette tâche. Aucun fichier applicatif ni donnée utilisateur modifié lors du diagnostic. Version déployée non vérifiée.

1. `connectors/saxo_import.py` lit `Market Value` et convertit ce montant en USD via `fx_service.convert`. Aucun nouveau cours de titre n'intervient dans ce calcul.
2. `adapters/saxo_adapter.py` relit le CSV sélectionné. `static/saxo-dashboard.html` additionne ses valeurs USD et le cash ; Refresh contourne le cache de présentation de cinq minutes.
3. `services/fx_service.py:convert` accepte `asof`, mais ne l'utilise pas pour sélectionner des taux historiques. Fournir une date à cette fonction ne résout donc pas le problème.
4. Le service générique `services/pricing_service.py` appelle un moteur principalement crypto et peut lire un fichier statique : ce parcours ne constitue pas un fournisseur boursier qualifié.
5. `services/risk/bourse/data_fetcher.py` possède une résolution de places de cotation, mais télécharge des séries journalières et peut remplacer un échec Yahoo par des données générées. Ne pas utiliser ce résultat comme un cours actuel authentique.
6. Le CSV sélectionné localement possède une colonne `Valeur actuelle (EUR)`, un prix actuel et des symboles avec suffixes de place. Les champs `Date de valeur` et `Dernière mise à jour` ne fournissent pas une date d'export fiable : le premier comporte des dates anciennes, le second des heures seules.
7. Certains noms de fichiers distinguent clairement date d'import et date du document d'origine. Le préfixe d'import ou le mtime ne doit pas être présenté comme date de valorisation historique certaine.
8. Les résumés Saxo du dashboard général, le résumé global et les exports ont des parcours séparés. Ils doivent recevoir la même base de valorisation et résoudre le même fichier pour les positions et le cash.

Ces constats prouvent le comportement local, pas la disponibilité ni la qualité actuelle d'un fournisseur externe. Aucun appel à Saxo connecté, opération de marché ou déploiement effectué.

## Étapes d'implémentation

### 1. Préserver et décrire la référence d'export

- Conserver montant brut, devise de valorisation, quantité, prix du fichier si présent, identifiants, date d'import et date de valorisation séparément.
- Détecter la devise depuis les colonnes explicites avant tout défaut ; distinguer devise du montant et devise de cotation.
- Extraire la date du document d'origine lorsqu'elle est fiable ; sinon indiquer `unknown` ou `inferred` avec provenance. Ne pas utiliser une date d'achat/valeur comme date d'export.
- `At export` affiche d'abord les montants dans la devise originale. Une conversion historique dans une autre devise exige un taux historique daté et sourcé ; si indisponible, conserver la devise originale sans inventer une équivalence USD.
- Pour le cash saisi séparément, conserver sa propre date et signaler lorsqu'il ne constitue pas un solde historique à la date du CSV. Ne pas promettre un total historique complet si ce solde manque.

### 2. Qualifier un fournisseur de cours sur un petit échantillon

- Réutiliser la résolution des identifiants/places existante, mais créer un parcours de cotations sans génération de données ni fallback crypto.
- Vérifier un échantillon US, Suisse et ETF/autre devise avant la généralisation : instrument exact, place, devise/unité de cotation, prix, horodatage du marché et provenance.
- Distinguer `quote_at`, `fetched_at` et type du cours : intraday, clôture, cache ancien, référence du CSV. Le TTL de 30 minutes ne garantit pas un cours intraday.
- Si les données réellement disponibles sont des clôtures, les afficher comme telles. Le fournisseur et la cadence utile seront confirmés par cet essai, sans achat ni abonnement implicite.
- Utiliser des prix de cotation non ajustés pour valoriser des quantités actuelles. Pour un ancien CSV, traiter les changements de quantité liés aux splits ou autres opérations documentées ; sinon signaler une quantité non vérifiée et éviter une prétendue valorisation actuelle complète. Ne pas prendre une série ajustée des dividendes comme prix de valorisation.

### 3. Service de valorisation commun

- Nouveau service `services/saxo_valuation_service.py` et fournisseur boursier dédié si nécessaire.
- Entrées explicites : utilisateur, fichier exact, mode de valorisation, devise et demande de rafraîchissement.
- Réutiliser/rationaliser `resolve_saxo_file_key` pour positions et cash ; un fichier explicitement demandé mais absent produit une erreur, pas le remplacement silencieux par un autre CSV.
- Mode actuel : quantité admissible × prix authentique × FX actuel, plus cash enregistré converti dans sa devise réelle.
- Réponse : positions, totaux, allocations, cash, couverture en nombre de positions, provenance des prix/FX, dates, alertes et état complet/partiel.
- En cas d'échec : dernier vrai cours avec date si disponible ; à défaut référence CSV clairement identifiée. Ne pas convertir une indisponibilité en zéro ni présenter un total mixte comme intégralement actualisé.
- Écart depuis l'export uniquement lorsque les deux bases sont comparables : même périmètre, devise et traitement des quantités. Sinon ne pas afficher un pourcentage trompeur.
- Cache des cotations par instrument exact/place/unité ; cache des résultats par utilisateur, fichier/version, mode et devise. Isolation stricte des données privées.

### 4. API et intégration des vues

- Ajouter une API de valorisation dans `api/saxo_endpoints.py` avec authentification JWT, utilisateur requis et format de réponse standard. Vérifier que le propriétaire JWT correspond au contexte utilisateur.
- `static/saxo-dashboard.html` et `static/css/saxo-dashboard.css` : sélecteur, rafraîchissement, dates et états de couverture. Toutes les cartes, tableaux et allocations consomment la même réponse.
- Renouveler les données au changement d'utilisateur/source/mode ; invalider les anciens caches. Suspendre la minuterie lorsque la page est masquée et supprimer les requêtes obsolètes à un changement de contexte.
- `static/modules/wealth-saxo-summary.js`, `api/wealth_endpoints.py` et `services/portfolio_export_service.py` : reprendre la base commune et indiquer le mode/date dans les exports ; éviter un dashboard global figé alors que Stock Market est actualisé.
- Ne pas changer implicitement les séries historiques des analyses ML/risque. Examiner leurs consommateurs si une pondération actuelle est exposée par le nouveau service.

## Validation requise

- Tests de référence : CSV ancien inchangé malgré un changement des cours/FX actuels ; devise originale conservée ; conversion historique indisponible explicite ; date d'import différente de date d'export.
- Tests de valorisation actuelle : plusieurs places/devises, unité de cotation, lots multiples sans doublon, cash du même fichier, quantité affectée par split, instrument non reconnu.
- Tests de panne/fraîcheur : cours absent ou ancien, clôture, échec fournisseur, cache 30 minutes, vrai renouvellement manuel, aucune donnée synthétique, total partiel explicite.
- Tests de sécurité : utilisateurs A/B, CSV inexistant, fichier hors périmètre utilisateur, absence d'identité/JWT ou identité incohérente ; aucun fallback interutilisateur.
- Tests de cohérence : somme positions + cash = total ; mêmes valeurs/base dans Stock Market, résumé global et export ; écart depuis export seulement si comparable.
- Régressions ciblées : `test_saxo_adapter_isolation.py`, `test_saxo_import_avg_price.py`, `test_portfolio_export_service.py`, `test_global_export_endpoint.py`, `test_export_formatter.py` et tests FX si modifiés ; ajouter tests du service et de l'API.
- Vérification navigateur responsive, changement de source/mode, page masquée/réouverte, panne et comparaison échantillon. Vérifier les informations de date/prix du fournisseur au moment de l'essai.
- Activer `.venv` avant exécution Python. Les tests n'ont pas été exécutés pour ce plan documentaire ; ils seront exécutés après implémentation.

## Limites et prochaine action

Prochaine action après validation : créer le socle de référence immuable et qualifier les cotations sur un petit échantillon, puis implémenter le service commun, intégrer les vues et terminer les contrôles.

Le prix courant n'actualise pas les opérations du compte, dividendes encaissés ni cash depuis l'export. Les anciens portefeuilles nécessitent une vérification des opérations sur titres. Aucune installation de fournisseur payant, commit, push ou mise en production inclus dans ce plan. Après modifications backend, redémarrage manuel requis, sans `--reload`.

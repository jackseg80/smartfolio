# Market Opportunities — diagnostic, refonte et contrôles du 1er octobre 2026

Le module est désormais refondu dans le checkout ML existant. Le code n'est pas déployé : production et aperçu 8083 conservent leurs images et dates de démarrage. La vérification réelle du nouveau moteur a été exécutée en mémoire dans le conteneur d'aperçu, en lisant uniquement les données du compte autorisé. Aucun montant, nom de position, fichier sélectionné, ISIN privé, clé ou hash de portefeuille n'est reproduit ici.

## Diagnostic et preuves

| Étape | Fait vérifié | Cause / conséquence |
|---|---|---|
| Source → CSV | Source Saxo CSV sélectionnée ; positions et cash rattachés au même fichier | Pas de preuve d'un compte démo ou d'un autre CSV utilisé lors du contrôle réel |
| CSV → positions | 32 lignes de fichier : 30 positions identifiées et 2 lignes de synthèse ; 30 ISIN ; 15 actions et 15 ETF | Les réponses normalisées anciennes perdaient ISIN, asset class, dates et devises de cotation |
| Valeurs | 28 valeurs numériques et 2 valeurs de position absentes | L'ancien parseur transformait l'absence de valeur en zéro : incertitude financière masquée |
| Devises | Valeur du CSV explicitement en EUR ; cotations CHF, EUR, GBP et USD | Ne pas convertir la valeur du compte avec la devise de l'instrument ; pas de parité ni de taux de secours pour le nouveau module |
| Dates | Import 2026-09-22 10:00:36 ; cash sauvegardé 2026-09-22 10:01:15.693200Z ; aucune date d'acquisition des 30 positions | La date d'import n'est pas une date financière confirmée. Les dates de notification/valeur ne prouvent pas une date d'achat |
| Identité | Mapping explicite place/cotation ; vérification de devise des historiques | Une ligne suisse déclarée USD ne peut pas emprunter silencieusement l'historique CHF. Le fournisseur peut manquer pour la cotation exacte |
| Classification ancienne | Environ 44,08 % du montant valorisé ventilé ; 55,92 % non ventilé ; fonds diversifiés sans décomposition ; profils sans date sectorielle effective | Une catégorie géographique ne doit pas remplir un secteur économique. Une étiquette ETF ne prouve pas les poids de ses secteurs |
| Écarts anciens | 11 assessments Indeterminate dans l'aperçu ; 0 écart certain | L'incertitude de composition explique ces résultats. Elle ne prouve pas un portefeuille équilibré |
| Candidats anciens | Exploration déclenchée seulement après les écarts certains : 0 candidat | Défaut de conception : une information sectorielle manquante empêchait même l'examen du marché |
| Classement ancien | Fenêtres historiques et composantes partielles ; diversification absente ; problème d'unité du dividend yield dans le composite historique | Ce composite ne constitue pas une preuve financière. Le nouveau module l'abandonne au profit d'un écran historique explicite |
| Ventes anciennes | 2 positions protégées, 28 exclues pour date de détention manquante ; aucune vente éligible | Critères restrictifs avant analyse ; absence d'éligibilité, pas preuve de qualité du portefeuille. Cash absent du scan ; zéro besoin de financement pouvait apparaître suffisant |
| Simulation ancienne | Sans achats appliqués ; Risk Score jamais calculé ; risque latent de vente recherchée par `symbol` sur des positions ayant `instrument_id` | Pas de changement ne mesure pas un bénéfice de réallocation ; risque de conservation des montants à corriger |
| Logs / environnements | Aucun ERROR/Traceback dans la fenêtre d'audit initiale ; conteneurs toujours en fonctionnement et inchangés au contrôle final | Les zéros venaient de chemins fonctionnels silencieux ; l'aperçu ajoutait des limites visibles mais gardait la chaîne restrictive |

Le constat des deux valeurs absentes complète et corrige le premier diagnostic : les deux zéros observés dans le parseur ne sont pas des valorisations vérifiées. Le nouveau module conserve les deux positions avec une valeur indisponible. Il bloque les bornes sectorielles du portefeuille complet et sa simulation plutôt que donner un total artificiel.

## Résultat du moteur refondu sur le compte réel

- 30 positions conservées, dont 30 avec ISIN, 0 avec date d'acquisition et 2 sans valeur source.
- Cohérence source/CSV/cash confirmée ; aucune autre sélection utilisée.
- 15 classifications d'actions issues d'un profil secondaire explicitement identifié, 1 ventilation de fonds acceptée depuis l'émetteur avec date et ISIN exacts, 14 fonds sans décomposition acceptée.
- Couverture **52,49 % du sous-ensemble valorisé** ; reste non ventilé **47,51 % de ce sous-ensemble**. La couverture du portefeuille complet est inconnue puisque deux valeurs manquent. Ce chiffre n'est pas une preuve de meilleure diversification.
- 11 catégories sectorielles Indeterminate, avec bornes complètes indisponibles.
- Univers initial : 11 ETF sectoriels examinés, 3 cotations déjà détenues exclues et 8 candidats avec scores effectivement calculés sur des historiques USD complets. Classement descriptif, sans prévision validée ni instruction d'achat.
- 0 vente automatique, avec les limites d'éligibilité expliquées. Le contrôle réel n'a soumis aucun scénario financier ni aucun ordre. La conservation des montants et les historiques de scénario sont vérifiés par des tests synthétiques identifiés comme tels.

## Lots : décisions prises, critères et données encore nécessaires

| Lot | Mise en œuvre / suite | Sources requises | Critères de qualité |
|---|---|---|---|
| 1 — Contrat du portefeuille | Implémenté : sélection stricte, champs riches, valeur/devise distinctes, cash du même CSV, dates et nulls | CSV sélectionné et son cash existant ; FX public vérifié | Jamais substituer un fichier/utilisateur ; conserver les positions non valorisées ; pas de valeur ou date inventée |
| 2 — Classification | Implémenté : secteurs, géographie et classes séparés ; deux adaptateurs publics iShares, un applicable au contrôle réel | Émetteur, ISIN exact, table Fund, observation datée ; profil secondaire pour une action | Ventilation ≤45 jours, non future, poids finis et cohérents, résidu conservé. Jamais remplacer le fonds réel par son indice |
| 3 — Écarts et candidats | Implémenté : bornes conservatrices, cible générique explicitée, politique personnelle optionnelle, exploration indépendante des écarts | Référence codée documentée ; politique personnelle si saisie ; univers public limité et profils courants | Aucun équilibre déduit d'un zéro ; raisons d'exclusion visibles ; ISIN manquant et chevauchement de fonds signalés |
| 4 — Historique / classement | Implémenté : USD historique, calendrier exact, jours complets, fenêtres entières, score et méthode affichés | Cours ajustés Yahoo et FX historiques réels pour la cotation exacte | Pas de proxy silencieux ; rejet des trous/devise ambiguë/staleness ; pas de rang entre dates différentes ; pas de probabilité ou rendement futur inventés |
| 5 — Revue et scénario | Implémenté : revues de détention et simulation manuelle avec IDs, coûts et conservation du cash ; volatilité et corrélation historiques optionnelles | Valeur de chaque position, cash cohérent ; tous les historiques exacts pour le risque ; frais/slippage saisis | Pas de vente automatique sans contraintes vérifiées ; pas de capital créé ; aucun ordre ; indisponible si une valeur ou un historique nécessaire manque |
| 6 — Sources restantes | Ouvert : adaptateurs supplémentaires pour les 14 fonds sans source acceptée ; dates financières et deux valorisations manquantes | Tables réelles du fonds ou fichiers de l'émetteur datés. Franklin et UBS demandent une vérification dédiée ; une fiche d'indice ne suffit pas | Documenter part/class/ISIN, unité, date, source et couverture. Ne pas viser 100 % par une approximation |
| 7 — Validation de l'interface | Tests DOM terminés ; validation visuelle/API du module déployé encore à faire | Publication autorisée du paquet dans l'aperçu 8083 existant, avec les mêmes montages privés RO | Refaire les seuls contrôles agrégés autorisés sur jack, vérifier mobile/desktop et erreurs de source/scénario ; production reste inchangée |

Les acquisitions, lots fiscaux, stop orders, contraintes personnelles de vente, géographie réelle des revenus/fonds et chevauchements de constituants restent non vérifiés. Les fonds achetés dans un scénario ne reçoivent pas une composition supposée. Aucun Risk Score 0–100 validé n'est ajouté ; la volatilité historique éventuelle est une mesure distincte.

## Validation et préservation

Les tests couvrent sélection/fichiers/devises, valeurs absentes et format pandas 3, identité/date/poids des fonds, bornes, conservation et dépassement de cash, identité de vente, fenêtres/historiques manquants, dates communes de rendement, transmission API, réponses tardives et injection DOM. Les observations publiques iShares datées du 29 septembre 2026 ont été relues ; leurs dix lignes visibles couvrent 98,04 % et 97,97 %, sans renormalisation. Une cotation publique US et une cotation publique suisse ont validé le chemin de calendrier et FX historique jusqu'au 30 septembre 2026, avec exclusion de la journée courante.

Les autres endpoints du fichier API sont conservés octet pour octet. Les modifications fonctionnelles du chantier ML hors de ce module sont préservées. Le contrôle par empreintes a révélé qu'un test préexistant supprimait et recréait `config/score_registry.json` : son `last_updated` a été régénéré pendant les tests. Ce fichier était déjà modifié avant l'intervention ; aucune restauration de son ancien timestamp non récupéré n'est prétendue. Le test utilise désormais `tmp_path` pour éviter toute nouvelle écriture dans le registre du checkout. Le registre est exclu du paquet selon la règle préexistante ; aucune configuration de production n'a changé.

Le paquet de code prépare aussi un garde-fou d'aperçu qui autorise uniquement le nouveau POST de calcul tout en bloquant entraînements, configuration et ordres. Le garde-fou actuellement déployé n'est pas modifié. Les sources privées et clés restent sur robot2, hors des images et de ce rapport. Aucun commit, push, nouveau worktree/environnement ou changement de production n'a été réalisé.

La preuve des tests et le manifeste de paquet sont enregistrés dans les sorties locales du checkout isolé. Les contrôles sur le compte réel se limitent aux agrégats autorisés. L'export de l'interface suit la même limite : pas de noms de positions, montants, identifiants de fichier ou hashes privés.

## Reprise et validation du 2 octobre 2026

Les derniers correctifs sont validés : 72 tests ciblés et 3286 tests unitaires réussis, 13 ignorés, couverture 49,46 %. Les 156 tests frontend précédents restent applicables, sans changement JS depuis leur succès. Contrôle de préservation : aucune suppression parmi 1440 fichiers, sept changements attendus dont l’exception de timestamp déjà documentée ; autres endpoints API byte-identiques.

Le compte réel a été revérifié en mémoire, sans déploiement : 30 positions, 30 ISIN, 2 valeurs absentes ; sélection et cash concordants ; 52,51 % de couverture du seul sous-ensemble valorisé, 47,49 % non ventilé, 11 assessments Indeterminate ; 8 candidats historiques calculés, 3 déjà détenus et zéro vente automatique. Les dates de source restent celles du contrôle précédent. La couverture du portefeuille complet reste inconnue et la simulation réelle reste bloquée.

Images et démarrages de production/aperçu inchangés au contrôle. Publication sur le seul aperçu 8083 et vérification visuelle/API restent soumises à autorisation. Le relais du 2 octobre et les journaux datés remplacent l’état de validation interrompu du relais du 1er octobre. Aucun commit, push ou ordre.

## Contrôle du module publié et correction du transport

La publication sur le seul aperçu 8083 a été autorisée le 2 octobre. La première validation déployée a découvert un défaut absent des premiers mocks : safeFetch retourne déjà {ok,status,data,error}, tandis que le contrôleur appelait response.json(). L’API répondait 200 mais l’affichage échouait avec une erreur technique. Scan et scénario consomment désormais response.data et response.error. Les tests de contrat utilisent le vrai helper ; ils couvrent succès, erreur de validation, scénario synthétique et changement de contexte pendant son décodage asynchrone. Suite frontend complète : 16 suites, 159 tests réussis. La preuve backend reste valide (aucun changement Python depuis 3286 tests et 49,46 % de couverture).

Les refus API déployés ont réussi : token absent, identité incohérente, autre CSV, Europe comme cible sectorielle, écriture de configuration et simulation avec valorisations manquantes. La validation desktop/mobile du contrôleur corrigé reste à terminer. L’environnement, les neuf montages et la production sont conservés ; le conteneur initial v13 est gardé arrêté pour retour arrière. Aucune donnée privée n’est incluse dans la nouvelle image.

## Lisibilité et contrôle fonctionnel final

Le contrôleur corrigé a réussi les scans réels en 1440 px et 390 px : même utilisateur/CSV, 30 positions, 11 assessments, 11 candidats examinés dont 8 avec scores calculés sur des fenêtres finissant le 1er octobre ; exclusions, manque de valorisation, absence de vente automatique et export agrégé correctement affichés. Les derniers ajustements donnent aux tableaux une largeur minimale avec défilement interne sur mobile et traduisent le scope technique en anglais lisible. Les 159 tests frontend passent après ces ajustements. Les deux faux échecs du premier contrôle r2 étaient des assertions de format : no-store reste présent dans no-cache, no-store, must-revalidate et les placeholders -- et — expriment le même effacement. Aucun middleware n’a été modifié.

Les captures de contrôle masquent avant création les paragraphes contenant fichier/cash ainsi que toutes les tables d’identités/instruments. Seuls agrégats et pixels sans positions ni montants sont retournés. Le résultat détaillé de publication est sauvegardé séparément après validation de cette dernière image.

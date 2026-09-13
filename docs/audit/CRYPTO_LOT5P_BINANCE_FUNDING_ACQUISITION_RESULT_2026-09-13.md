# Lot 5P — Résultat de l'acquisition du funding Binance

Date : 13 septembre 2026<br>
Plan gelé : [CRYPTO_LOT5P_BINANCE_FUNDING_ACQUISITION_PLAN_2026-09-13.md](./CRYPTO_LOT5P_BINANCE_FUNDING_ACQUISITION_PLAN_2026-09-13.md)

## Verdict

**Go qualité.** L'historique public du funding réalisé Binance USDⓈ-M satisfait tous les critères préenregistrés pour BTC, ETH, SOL, ADA, XRP, LINK et LTC du 1er juin 2022 au 31 août 2026 inclus.

Ce verdict autorise uniquement le lot 5Q, consacré aux features causales et au test de mutation du futur. Il ne démontre ni pouvoir prédictif, ni rendement, ni avantage après coûts.

## Résultat observé

| Contrôle | Seuil gelé | Résultat |
|---|---:|---:|
| Instruments | au moins 5 | 7 |
| Mois par instrument | 51 | 51 |
| Archives ZIP + CHECKSUM | 357 | 357 validées |
| Jours calendaires par instrument | au moins 1 551 | 1 553 |
| Observation quotidienne | au moins 1 | aucune journée manquante |
| Écart maximal autorisé | 24 h | seuil respecté partout |
| Doublons | 0 | 0 |
| Empreintes officielles discordantes | 0 | 0 |
| Taille totale ZIP | au plus 16 777 216 octets | 316 272 octets |
| Reproductibilité des fichiers normalisés | aucune différence | 14/14 identiques |

BTC, ETH, ADA, XRP, LINK et LTC ont chacun 4 659 observations. SOL en a 4 734, car certaines périodes contiennent plus de trois règlements quotidiens ; cette cadence publiée est conservée telle quelle et n'est pas ramenée artificiellement à trois observations.

Les timestamps vont de `2022-06-01T00:00:00Z` à `2026-08-31T16:00:00.001Z`. Les décalages milliseconde publiés par la source sont conservés ; la date UTC, l'ordre strict et l'unicité restent contrôlés.

## Provenance et reproductibilité

- Artifact final : `crypto-forecast-binance-funding-history-v1-6288d6a1dca91185`.
- Manifeste principal SHA-256 : `5d3f4320926c1615e961a17c64ca911fcc35173c4daae6013665e49ebb5fbf48`.
- Configuration SHA-256 : `d8bb38736c81e9bf341677b57e83155193e559be4e0041a40a15ea986b326810`.
- Chaîne de code acquisition + validation SHA-256 : `054c22ee85c3061adbf18873646ebfceffd343772ccb475abd67dbddc81ec1d8`.
- Deux acquisitions indépendantes donnent le même identifiant ; les 14 CSV d'événements et de couverture ont zéro différence d'empreinte.

Chaque archive a été lue en flux avec arrêt immédiat au plafond, puis contrôlée contre le fichier `CHECKSUM` officiel avant ouverture. Un seul CSV au nom attendu est accepté ; sa taille décompressée est bornée. Les archives brutes ne sont pas conservées après validation. Le manifeste retient leur nom, leur taille, leur empreinte officielle, leur empreinte calculée et leur nombre de lignes.

## Validation technique

- 5 tests ciblés : réussis.
- Régression finale de la recherche crypto : 110 tests réussis en 6,36 s.
- Ruff : conforme.
- Black : conforme.
- Avertissement restant : dépréciation Starlette/httpx déjà extérieure à ce chantier.

Une première passe des 4 tests initiaux avait déclenché une violation d'accès dans la génération XML de `coverage` après réussite des assertions. La validation finale a donc été rejouée avec `--no-cov`; les tests applicatifs eux-mêmes ne sont pas en échec.

## Limites

- Le funding provient de contrats perpétuels USDⓈ-M ; il est étudié comme information de marché, pas comme autorisation d'utiliser levier ou vente à découvert.
- Le corpus est historique. Ce lot n'a pas encore prouvé la disponibilité exacte de chaque agrégat au moment d'une décision quotidienne.
- Aucun taux n'a encore été agrégé, décalé ou joint aux prix et cibles.
- Aucun modèle, allocation, interface, service permanent, changement de production ou ordre réel n'a été créé.

## Suite autorisée

Le lot 5Q doit geler avant construction :

1. l'heure de décision UTC et la règle stricte de disponibilité des règlements ;
2. les agrégats quotidiens passés uniquement, leurs fenêtres et leurs décalages ;
3. la règle de jointure aux prix sans accès aux cibles ;
4. un test de mutation du futur exigeant l'identité parfaite de toutes les lignes antérieures à la coupure.

Le lot 5R ne comparera le modèle de référence au modèle enrichi qu'après réussite du lot 5Q. Le test économique et une éventuelle démonstration locale restent conditionnels à une amélioration hors échantillon réelle.

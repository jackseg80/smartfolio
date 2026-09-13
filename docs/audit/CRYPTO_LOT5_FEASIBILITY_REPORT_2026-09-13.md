# Rapport de faisabilité — lot 5 multi-plateformes et liquidité

Date : 13 septembre 2026<br>
Périmètre : documentation officielle et sondes publiques en lecture seule. Aucun fichier de marché massif, secret, compte, ordre ou changement de production.

## Décision

Le jalon de faisabilité est **Go limité** pour un collecteur quotidien OKX/USDT et **Go conditionnel** pour un petit pilote de carnet L2 OKX. Il reste **No-Go** pour un téléchargement L2 massif ou pour toute nouvelle affirmation de performance.

La priorité recommandée est :

1. utiliser OKX/USDT pour comparer causalement prix et volumes quotidiens avec Binance/USDT sur les dix actifs actuels ;
2. réserver Coinbase/USD comme contrôle secondaire du prix et de l'effet de cotation USDT ;
3. conserver le pilote L2 OKX borné comme contrôle de faisabilité et poursuivre uniquement par snapshots autonomes ;
4. conserver Kraken et Bybit comme solutions de contrôle, sans multiplier immédiatement les collecteurs.

## Matrice des sources

| Source | Historique spot annoncé | Champs utiles | Limites importantes | Décision |
|---|---|---|---|---|
| Binance Public Data | Archives quotidiennes/mensuelles, toutes les paires supportées | OHLC, volume base/quote, transactions, volumes taker ; trades et aggTrades séparés | Référence déjà utilisée ; pas de nouvel historique L2 spot librement établi dans ce jalon | Conserver comme référence |
| Coinbase Exchange | Candles publiques paginables, 300 maximum par requête | OHLC et volume ; cotations USD/USDT | Intervalles sans transaction absents ; volume sans compteur de trades ; carnet API actuel, pas archive historique annoncée | Contrôle USD secondaire |
| Kraken | Archives OHLCVT depuis le début de chaque marché | OHLC, volume et nombre de trades | La page officielle consultée indique encore une couverture complète jusqu'à fin T3 2024 ; fichiers via Google Drive ; trous lorsque zéro transaction | Contrôle/solution de secours |
| Bybit | Klines historiques spot, 1 000 maximum par page | OHLC, volume base et turnover quote | La réponse spot actuelle ne fournit pas de date de lancement exploitable ; carnet disponible en instantané, pas comme archive historique prouvée ici | Contrôle USDT secondaire |
| OKX | Candles de plusieurs années ; archives séparées | OHLC, volumes base/quote, état de clôture ; trades historiques ; L2 haute résolution | Candles téléchargeables annoncées depuis juillet 2023 et L2 depuis mars 2023 ; volume L2 potentiellement très important | Source nouvelle prioritaire |

## Vérifications en direct

Les catalogues publics interrogés le 13 septembre 2026 montrent que BTC, ETH, SOL, ADA, XRP, LINK, LTC, BCH, BNB et DOT ont tous une paire active USD ou USDT sur les quatre plateformes sondées.

- Coinbase : les dix actifs ont une paire USD ; plusieurs ont aussi une paire USDT.
- Bybit : les dix paires `ASSETUSDT` sont `Trading`.
- OKX : les dix actifs ont une paire USDT active. Les dates de listing déclarées sont le 29 janvier 2021 pour neuf actifs et le 20 décembre 2022 pour BNB.
- Kraken : les dix actifs ont une paire USD ou USDT active ; Bitcoin y est exposé sous l'alias `XBT`.

Trois petites sondes quotidiennes ont confirmé les schémas sans conserver de payload massif :

| Source | Période sondée | Lignes | Champs | Résultat |
|---|---|---:|---:|---|
| Coinbase BTC-USD | 2018-01-01 → 2018-01-10 | 10 | 6 | OHLCV historique disponible |
| Bybit BTCUSDT | 2022-01-01 → 2022-01-10 | 10 | 7 | OHLCV + turnover disponibles |
| OKX BTC-USDT | dix jours avant 2022-01-10 | 10 | 9 | OHLCV quote/base, bougies toutes confirmées |

Ces sondes prouvent le fonctionnement de l'accès public et le schéma sur un cas. Elles ne prouvent pas encore la couverture complète de chaque actif.

## Sources officielles consultées

- [Binance Public Data](https://github.com/binance/binance-public-data/blob/master/README.md)
- [Coinbase Exchange — candles](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles)
- [Coinbase Exchange — produits](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-all-known-trading-pairs)
- [Kraken — archives OHLCVT](https://support.kraken.com/articles/360047124832-downloadable-historical-ohlcvt-open-high-low-close-volume-trades-data)
- [Bybit — klines](https://bybit-exchange.github.io/docs/v5/market/kline)
- [Bybit — instruments](https://bybit-exchange.github.io/docs/v5/market/instrument)
- [OKX — API market data](https://www.okx.com/docs-v5/en/)
- [OKX — données historiques](https://www.okx.com/en-us/historical-data)

OKX annonce des transactions tick par tick depuis septembre 2021, des candles téléchargeables depuis juillet 2023 et un carnet L2 haute résolution depuis mars 2023. Les endpoints courants de carnet des autres plateformes ne sont pas assimilés à un historique.

## Contrat proposé pour l'acquisition quotidienne

### Périmètre

- Source primaire nouvelle : OKX Spot.
- Paires : BTC-USDT, ETH-USDT, SOL-USDT, ADA-USDT, XRP-USDT, LINK-USDT, LTC-USDT, BCH-USDT, BNB-USDT et DOT-USDT.
- Intervalle : `1Dutc` uniquement.
- Début demandé : date de listing déclarée par OKX, sans extrapolation antérieure.
- Fin : dernière journée UTC entièrement close.
- Source de comparaison : artifact Binance existant, jamais modifié.

### Champs normalisés

- plateforme et instrument natif ;
- date UTC et timestamp d'ouverture ;
- open, high, low, close ;
- volume base, volume quote et état `confirmed` ;
- date de listing issue du snapshot d'instruments ;
- URL, paramètres, date de collecte, hashes des réponses normalisées et version du collecteur.

### Règles de qualité

- rejeter toute bougie non confirmée ;
- trier et dédupliquer strictement les timestamps ;
- ne jamais remplir un jour manquant avec le prix précédent ;
- signaler chaque trou et chaque différence de calendrier ;
- conserver les décimales textuelles du fournisseur ;
- comparer les closes et volumes seulement après alignement exact de la journée UTC ;
- ne pas considérer USDT comme du cash sans risque.

## Pilote de liquidité proposé

Le carnet L2 est la seule source identifiée dans ce jalon qui apporte une information structurellement différente des candles Binance.

Le pilote doit rester borné à BTC-USDT, ETH-USDT et SOL-USDT, sur une courte période et avec une taille maximale décidée avant téléchargement. Il doit d'abord vérifier :

- URL et format exacts des archives ;
- fréquence réelle et continuité des snapshots/mises à jour ;
- reconstruction du carnet et gestion des séquences ;
- taille compressée et décompressée ;
- calcul causal du spread, de la profondeur à 10/25/50 points de base et du déséquilibre bid/ask ;
- absence de redistribution ou d'usage interdit par les conditions applicables.

Le pilote a ensuite été exécuté séparément dans `CRYPTO_LOT5B_L2_PILOT_RESULT_2026-09-13.md`. Une archive SOL-USDT d'un jour et 47,43 MB a été analysée sous plafonds préenregistrés. Le flux complet de deltas est rejeté parce qu'aucune ligne ne contient d'identifiant de séquence ; les 96 snapshots complets espacés d'environ 15 minutes restent une piste distincte. Aucun fichier BTC/ETH, modèle ou composant produit n'a été ajouté.

## Limite de validation

La confirmation finale 2025-09-05 → 2026-08-12 a déjà été observée dans les lots précédents. Elle ne peut plus être présentée comme un nouveau holdout aveugle. Le lot 5 pourra mesurer la robustesse multi-plateforme et effectuer des tests de recherche, mais une preuve finale exigera soit :

- un protocole gelé évalué sur une plateforme tenue complètement hors développement ;
- soit une période prospective encore non observée après gel du modèle.

## Étape suivante autorisable

Implémenter et tester un collecteur OKX quotidien isolé, produire un artifact versionné pour les dix paires, puis mesurer la couverture et les divergences face à Binance. Cette étape ne touche ni l'API SmartFolio, ni l'interface, ni les ports 8080/8082.

# Résultat lot 5O — faisabilité de signaux historiques compacts

Date : 13 septembre 2026<br>
Statut : écran documentaire et métadonnées terminé ; aucune série historique téléchargée.<br>
Décision : **Go pilote pour le funding Binance USDⓈ-M**, avec Coin Metrics on-chain en seconde piste conditionnelle.

## Conclusion

Le meilleur prochain essai n'est plus le carnet L2. Le funding réalisé des contrats perpétuels Binance apporte une information de positionnement/coût distincte des OHLCV spot, dispose d'archives qui encadrent la fenêtre requise pour au moins sept actifs du panel et représente un volume de données dérisoire face aux archives L2.

Le verdict reste **GO_PILOT**, pas Go prédictif : l'existence des fichiers de début et de fin ne prouve pas encore la première observation exacte, la continuité de chaque date ni l'absence de doublons. Le prochain lot devra télécharger uniquement ce corpus compact, vérifier toutes les observations et épingler les fichiers bruts avec leurs checksums avant de construire une feature.

Coin Metrics est une seconde piste recevable pour `AdrActCnt` et `TxCnt`. Son catalogue Community prouve une longue couverture quotidienne sur sept actifs, mais un pilote doit encore geler les réponses, les définitions et la politique de révision point-in-time. Il ne faut pas injecter dans un ancien backtest une valeur révisée ultérieurement en prétendant qu'elle était connue à la date simulée.

## Matrice de décision

| Rang | Famille | Décision | Signal retenu | Motif principal |
|---:|---|---|---|---|
| 1 | Binance USDⓈ-M | **GO_PILOT** | Funding réalisé | Historique public compact et multi-actifs ; continuité interne à vérifier |
| 2 | Coin Metrics Community | **GO_PILOT** | `AdrActCnt`, `TxCnt` | Vraies métriques réseau quotidiennes ; politique de révision à verrouiller |
| 3 | OKX dérivés | **NO_GO** | Aucun | Aucun archive fixe ni horizon officiel prouvant les 1 551 jours dans la documentation examinée |
| 4 | FRED VIX/DXY | **NO_GO comme nouveauté** | Contrôle existant | Déjà utilisé par `services/macro_stress.py` |
| 5 | Alternative.me | **NO_GO** | Contrôle descriptif seulement | Série Bitcoin unique, partiellement dérivée de volatilité/momentum et déjà référencée par SmartFolio |

## Preuves Binance

Le dépôt officiel [Binance Public Data](https://github.com/binance/binance-public-data) décrit un accès public aux fichiers quotidiens et mensuels, leur publication différée, les checksums et les mises à jour éventuelles d'archives. La documentation USDⓈ-M définit le endpoint public de [Funding Rate History](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data), son `fundingTime`, son taux réalisé et l'ordre ascendant des résultats.

Quatorze requêtes `HEAD`, sans lecture du contenu, ont confirmé les archives mensuelles de juin 2022 et août 2026 pour :

- BTC, ETH, SOL, ADA, XRP, LINK et LTC ;
- 14 réponses HTTP 200 sur 14 ;
- fichiers compressés observés entre 857 et 995 octets par actif-mois ;
- estimation de `355 215` octets compressés pour sept actifs et 51 mois ; même une borne brute volontairement extrême de 100 fois cette taille reste à environ 35,5 Mo, très loin du plafond préenregistré de 256 MiB.

Le funding et son timestamp sont observables après le prélèvement. Une future feature quotidienne devra donc être construite uniquement avec les paiements dont `fundingTime` est antérieur ou égal à l'heure de décision. Les fichiers et checksums seront épinglés : le dépôt officiel précise que des archives peuvent être corrigées ultérieurement.

L'open interest Binance ne passe pas le même écran : l'endpoint officiel d'[Open Interest Statistics](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data) limite l'historique REST au dernier mois. Il ne peut donc pas fournir à lui seul les 1 551 jours gelés.

## Preuves Coin Metrics

La documentation officielle [API Conventions](https://docs.coinmetrics.io/api) indique que l'API Community ne requiert pas de clé, utilise UTC et met ses données communautaires à disposition gratuitement pour un usage non commercial sous licence Creative Commons. Le [catalogue API v4](https://docs.coinmetrics.io/api/v4/) fournit pour chaque couple actif-métrique la fréquence, `min_time` et `max_time`.

Une lecture du catalogue, sans série temporelle, a trouvé `AdrActCnt` et `TxCnt` en fréquence quotidienne jusqu'au 12 septembre 2026 pour BTC, ETH, ADA, XRP, LINK, LTC et BCH. Le début commun le plus récent parmi ces sept actifs est le 23 septembre 2017. SOL est absent pour ces métriques, DOT s'arrête au 3 juin 2022 et l'ancien actif BNB au 22 avril 2019 ; ils ne sont pas comptés.

La documentation précise qu'une fréquence `1d` correspond à une journée se terminant à 00:00 UTC dans les [conventions de fréquence](https://docs.coinmetrics.io/resources/faqs). Avec sept actifs et 1 551 jours, l'ordre de grandeur est 10 857 lignes ; même une borne très conservatrice de 1 000 octets par ligne reste sous 11 Mo.

Le catalogue prouve l'étendue, pas l'absence de trous ni la version effectivement connue chaque jour. C'est pourquoi Coin Metrics reste en Go pilote conditionnel et ne devance pas le funding Binance.

## Témoins rejetés

- **OKX** : l'[API V5](https://app.okx.com/docs-v5/en/) documente le funding paginé et l'open interest courant, mais le présent écran n'a trouvé ni archive fixe ni garantie de rétention prouvant la période gelée. La documentation montre aussi que la méthode et la fréquence du funding ont changé selon les contrats ; il faudrait conserver ces régimes, pas les lisser rétroactivement.
- **FRED** : [VIXCLS](https://fred.stlouisfed.org/series/VIXCLS) et [DTWEXBGS](https://fred.stlouisfed.org/series/DTWEXBGS) sont des séries quotidiennes longues et compactes, mais SmartFolio les consomme déjà. Elles restent des contrôles macro, pas la nouvelle information recherchée.
- **Alternative.me** : l'[API Fear & Greed](https://alternative.me/crypto/fear-and-greed-index/) expose tout l'historique avec attribution, mais le fournisseur la décrit comme Bitcoin-centric et explique qu'une moitié du composite vient de volatilité et momentum. La série échoue donc à la couverture de cinq actifs et à la nouveauté par rapport au pipeline actuel.

## Reproductibilité

- Artifact : `crypto-forecast-compact-signal-feasibility-v1-9fb8f124741be790`
- SHA-256 résultat : `aab5e5c8b47fc017bfbc84d3d47673ac3016d67bdc13659db924b74632c81531`
- SHA-256 matrice CSV : `74693bc439a94d03d9426906ac56e0b752c0a939fab7c3daef2d8c9d6d01d5e3`
- SHA-256 configuration : `9307916b0658f282f5e657a2538cc473d340ba431be175e2658ddc20ec1c2f28`
- SHA-256 preuves figées : `fc1e93986f519b7e2c54bb641b7ff1b38f1b1bb0a57b694db7adbb830d745db5`
- SHA-256 évaluateur : `71c728320d27e20716babcbfe9b6efcf7e0ce62ce9d4bec256ea2de573f93821`

Deux exécutions dans des répertoires isolés produisent le même identifiant, le même résultat et la même matrice. Les 105 tests unitaires de prévision crypto réussissent. Une première commande strictement limitée au nouveau fichier avait uniquement échoué au seuil global de couverture du dépôt, bien que ses cinq assertions soient déjà passées ; la relance ciblée puis la suite élargie confirment le résultat.

## Limites et sécurité

- Les requêtes externes ont porté sur des pages officielles, deux lectures de catalogue et des en-têtes HTTP ; aucun fichier historique n'a été téléchargé.
- Les estimations de taille ne valident ni la complétude ni la qualité interne des séries.
- `dataset_downloaded=false`, `future_targets_read=false`, `features_materialized=false`, `model_trained=false`, `backtest_run=false`.
- Aucun compte, secret, portefeuille, ordre, dérivé réel, API produit, interface ou production n'a été touché.
- Aucun commit, push ou déploiement n'a été effectué.

## Suite recommandée

Préenregistrer un lot 5P limité au funding Binance USDⓈ-M sur les sept actifs prouvés, du 16 juin 2022 au 31 août 2026. Ce lot devra :

1. télécharger seulement les fichiers mensuels funding et leurs checksums ;
2. refuser tout fichier manquant, checksum invalide, timestamp dupliqué ou observation hors contrat ;
3. produire une table quotidienne causale avec sommes, moyennes, dispersion et changements passés, sans cible ;
4. tester la mutation future avant toute jointure avec les rendements 7/30 jours ;
5. décider ensuite, séparément, si les données sont assez propres pour une comparaison avec/sans funding.

La piste Coin Metrics reste en réserve. Elle ne doit être collectée qu'après le pilote funding, ou si celui-ci échoue à ses contrôles de continuité.

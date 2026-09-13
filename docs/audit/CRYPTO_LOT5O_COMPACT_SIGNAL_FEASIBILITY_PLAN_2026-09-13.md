# Plan préenregistré — lot 5O, faisabilité de signaux historiques compacts

Date de gel : 13 septembre 2026, avant consultation des catalogues et documentations externes.<br>
Entrée : décision No-Go prédictive du lot 5N, causée par seulement trois dates L2 indépendantes.

## Objectif

Identifier une famille de données qui apporte une information réellement distincte des prix spot quotidiens, tout en offrant assez d'historique pour le protocole causal SmartFolio. Ce lot est un écran documentaire en lecture seule : il ne télécharge aucun dataset, ne matérialise aucune feature et ne lit aucun rendement futur.

## Seuils gelés

Le candidat doit satisfaire simultanément :

- première observation au plus tard le 16 juin 2022, soit une étendue minimale de `1 551` jours au 13 septembre 2026 ;
- au moins `480` dates UTC indépendantes, sans compter plusieurs observations intrajournalières comme plusieurs journées ;
- au moins cinq actifs du panel gelé BTC, ETH, SOL, ADA, XRP, LINK, LTC, BCH, DOT et BNB ;
- volume brut estimé d'au plus `256 MiB` pour cinq actifs sur la période minimale ;
- timestamps et moment de disponibilité suffisamment définis pour décaler causalement chaque feature ;
- accès officiel, public et en lecture seule, avec une route d'archive ou de récupération reproductible ;
- information non reconstruite à partir des seuls OHLCV spot déjà testés ;
- conditions de réutilisation pour la recherche identifiées avant toute collecte.

Les horizons prédictifs restent gelés à 7 et 30 jours. Le protocole ultérieur, s'il est autorisé, conservera les partitions chronologiques, trois purges de 30 jours, une calibration distincte et une confirmation finale gelée du lot 3/5N.

## Familles candidates gelées

1. dérivés USDⓈ-M Binance : funding, premium index et métriques d'open interest ;
2. métriques on-chain Coin Metrics Community : adresses actives, transactions, frais et valeur transférée ;
3. dérivés OKX : funding et open interest ;
4. macro FRED VIX/DXY, comme témoin de disponibilité longue mais déjà présent dans SmartFolio ;
5. Crypto Fear & Greed d'Alternative.me, comme témoin de sentiment à couverture potentiellement mono-série.

Aucun candidat ne sera ajouté ou retiré après lecture des résultats. Une famille peut être rejetée pour un signal précis sans invalider ses autres signaux si les routes et rétentions diffèrent.

## Preuves autorisées

- documentations, catalogues, dépôts ou conditions officiels du fournisseur ;
- métadonnées décrivant cadence, couverture, début de série, rétention et méthode d'accès ;
- inspection locale du code SmartFolio pour mesurer la nouveauté du signal.

Les blogs tiers, agrégateurs commerciaux et affirmations non datées ne peuvent pas établir un critère. Une lacune documentaire reste une lacune ; elle n'est pas convertie en hypothèse favorable.

## Décision

- **GO_PRIMARY** : tous les critères sont prouvés par les sources officielles ; la famille peut faire l'objet d'un plan de collecte séparé.
- **GO_PILOT** : aucun échec dur n'est connu, mais un échantillon borné ou une clarification des conditions reste nécessaire.
- **NO_GO** : un critère dur échoue ou n'est pas démontrable sans réduire les seuils après observation.

Le classement privilégie l'information incrémentale et la causalité, pas la quantité de colonnes. Une série déjà utilisée dans SmartFolio ou dérivée des OHLCV ne devient pas nouvelle parce qu'elle est facile à obtenir.

## Hors périmètre

- téléchargement de fichiers historiques ou appel d'endpoint de données ;
- collecte récurrente, secret, compte payant ou abonnement ;
- feature, cible, modèle, backtest, allocation ou signal ;
- API, interface, configuration de production ou test utilisateur ;
- ordre, dérivé réel, levier, commit, push ou déploiement.

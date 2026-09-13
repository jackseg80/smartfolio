# Plan préenregistré — lot 5, faisabilité multi-plateformes et liquidité

Date de gel : 13 septembre 2026, avant tout téléchargement ou développement multi-plateforme.

## Objectif

Déterminer si des sources publiques, datées et reproductibles peuvent apporter une information réellement nouvelle par rapport au dataset Binance quotidien du lot 3 : validation des prix/volumes sur plusieurs plateformes et, si disponible, profondeur ou spread historiques.

Ce jalon ne cherche pas encore une performance de modèle. Il doit conclure par une sélection justifiée des sources et un contrat de données, ou par un No-Go explicite.

## Sources candidates

- Binance Public Data : référence actuelle et, éventuellement, transactions agrégées historiques.
- Coinbase Exchange : OHLCV spot en USD et registre actuel des produits.
- Kraken : archives OHLCVT officielles et registre actuel des paires.
- Bybit : klines spot en USDT et date de lancement déclarée des instruments.
- OKX : candles spot, registre des instruments et archive historique L2 annoncée.

Seules les documentations et données officielles des plateformes sont recevables pour ce jalon.

## Critères de recevabilité

Une source n'est retenue que si :

1. l'accès de marché est public et en lecture seule, sans clé de trading ;
2. le fournisseur, l'URL, le schéma, la date de collecte et les empreintes peuvent être conservés ;
3. les dates sont interprétables en UTC sans interpolation silencieuse ;
4. l'historique couvre au moins BTC et ETH, plus un noyau utile d'altcoins ;
5. les unités de prix et de volume sont explicites ;
6. les trous, listings, délistings et changements de symbole restent visibles ;
7. les conditions d'utilisation permettent une recherche locale ;
8. le volume et le coût de stockage restent proportionnés au test.

Pour une source L2, il faut en plus des snapshots ou mises à jour historiques horodatés, un format documenté et une période permettant une validation hors échantillon. Un carnet actuel accessible par API n'est pas présenté comme historique.

## Ordre des vérifications

1. vérifier les documentations officielles, limites et profondeur historique annoncée ;
2. vérifier la présence actuelle des paires de l'univers commun ;
3. sonder un petit nombre de réponses publiques sans enregistrer de secret ;
4. estimer la couverture et le volume avant tout téléchargement massif ;
5. rédiger la matrice Go/No-Go et choisir le contrat minimal ;
6. seulement ensuite proposer l'implémentation du collecteur.

## Règles de décision

- Les OHLCV multi-plateformes servent d'abord à vérifier la robustesse et les divergences de marché ; ils ne sont pas automatiquement ajoutés comme features.
- La priorité va à une donnée différente de l'OHLCV Binance déjà testé, notamment spread, profondeur ou déséquilibre de carnet historiques.
- Un historique L2 trop court peut servir à une étude séparée, mais ne doit pas être mélangé à la confirmation finale déjà utilisée.
- Aucun seuil du lot 3b n'est réoptimisé sur ces nouvelles observations.
- Aucun téléchargement L2 massif, abonnement payant ou acceptation contractuelle supplémentaire n'est réalisé sans décision séparée.

## Livrables du jalon

- rapport de faisabilité et matrice des sources ;
- noyau d'actifs/paires et périodes communes proposées ;
- estimation du stockage et des appels ;
- contrat de provenance, trous et normalisation ;
- décision Go/No-Go pour l'acquisition ;
- plan de tests du futur collecteur.

## Hors périmètre

- modèle, allocation ou optimisation de seuils ;
- API SmartFolio, interface, port 8082 ou production 8080 ;
- clés privées, comptes exchange, ordres ou portefeuille réel ;
- données dérivées, levier, short ou futures.

## Contrat préenregistré de comparaison Binance–OKX

Ajouté le 13 septembre 2026 avant le calcul des divergences.

Pour chaque actif, la comparaison utilise uniquement les journées UTC présentes sur les deux plateformes. Aucune interpolation n'est permise. Les métriques figées sont :

- nombre et période des journées communes ;
- corrélation des rendements quotidiens ;
- écart absolu médian, 95e percentile et maximum entre les clôtures, en points de base ;
- écart absolu médian des rendements quotidiens, en points de base ;
- corrélations des variations de volume coté à 7 et 30 jours, sans comparer directement les niveaux de volume.

Une série passe le contrôle de cohérence prix si elle possède au moins 1 000 observations communes, une corrélation de rendements d'au moins 0,995, un écart médian de clôture d'au plus 50 points de base et un 95e percentile d'au plus 300 points de base. Un échec reste publié et n'est pas retiré de l'univers après observation.

Les corrélations de volume sont descriptives : aucun seuil de sélection n'est fixé dans ce jalon. Le résultat sert à décider si OKX est une source de validation recevable, pas à sélectionner un modèle.

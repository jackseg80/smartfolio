# Plan préenregistré — lot 5B, pilote de carnet L2 OKX

Date de gel : 13 septembre 2026, après consultation des seules métadonnées publiques OKX et avant tout téléchargement d'archive L2.

## Décision de cadrage

Le pilote utilise le module public OKX `4`, carnet spot à 400 niveaux. Le module `6` à 50 niveaux n'est pas retenu : OKX annonce sa dépréciation progressive et, pour la même journée et les mêmes paires, ses fichiers sont plus volumineux que ceux du module `4`.

Le premier échantillon est fixé ainsi :

- instrument : `SOL-USDT` spot ;
- journée : `2026-09-10` en UTC ;
- granularité : archive quotidienne ;
- fichier annoncé : `SOL-USDT-L2orderbook-400lv-2026-09-10.tar.gz` ;
- taille annoncée par l'API : `47.43 MB` ;
- source : `GET /api/v5/public/market-data-history`, sans authentification.

La date est la plus récente journée théoriquement complète selon le délai T+3 annoncé par OKX au moment du gel. SOL est retenu parce que son archive est la plus petite des trois paires candidates. Aucun résultat de marché ou de modèle n'a servi à choisir cette date ou cette paire.

## Objectif

Vérifier, sur un échantillon borné, si l'archive permet de reconstruire causalement un carnet et de calculer des variables de microstructure sans ambiguïté : spread, profondeur proche du mid-price, déséquilibre bid/ask et continuité temporelle.

Ce pilote ne mesure pas la valeur prédictive, ne modifie pas le dataset quotidien et n'entraîne aucun modèle.

## Limites du pilote

- téléchargement compressé maximal : `55,000,000` octets ;
- somme des membres décompressés maximale : `2 GiB` ;
- aucun membre brut n'est extrait durablement sur disque ;
- au plus `10,000,000` enregistrements sont lus ;
- une seule archive, une seule paire, une seule journée ;
- arrêt immédiat si la taille reçue dépasse le plafond, si le format diffère des métadonnées ou si l'archive est invalide.

L'archive compressée peut être conservée localement avec son empreinte SHA-256. Les sorties dérivées doivent rester compactes : manifeste, inventaire des membres, contrat de schéma, contrôles de qualité et statistiques agrégées.

## Contrôles obligatoires

1. enregistrer l'URL de métadonnées, l'URL du fichier, la date de collecte, la taille annoncée, la taille reçue et le SHA-256 ;
2. inventorier les membres du `tar.gz` sans extraction persistante ;
3. identifier les champs, unités, types d'événement, snapshots et mises à jour ;
4. vérifier l'ordre des timestamps et, si présents, la continuité `seqId/prevSeqId` ;
5. refuser toute interpolation ou réparation silencieuse d'un trou ;
6. reconstruire seulement lorsque les règles snapshot/delta sont non ambiguës ;
7. contrôler prix et quantités finis et non négatifs, meilleur bid inférieur ou égal au meilleur ask hors état explicitement documenté ;
8. produire, si le carnet est reconstructible, spread en points de base, profondeur bid/ask dans 10, 25 et 50 points de base et déséquilibre normalisé ;
9. distinguer clairement les métriques complètes des métriques calculées sur un préfixe borné.

Clarification d'implémentation gelée avant l'analyse complète : les métriques de carnet sont observées sur une grille UTC d'une minute, en état « as-of ». Une mise à jour postérieure à la minute observée ne peut donc pas modifier sa valeur. Le maintien du dernier état du carnet entre deux mises à jour est la sémantique normale d'un carnet, pas une interpolation de prix ou de quantité.

## Règles Go/No-Go

Le pilote est **Go technique** seulement si l'archive est reproductible, parsable sans hypothèse cachée, possède un ancrage temporel exploitable et permet de reconstruire au moins 99 % des séquences observées sans interpolation.

Il est **No-Go** si le schéma ou les unités restent ambigus, si les snapshots nécessaires manquent, si les ruptures de séquence ne sont pas détectables, si les limites sont dépassées ou si l'accès nécessite un compte, une clé ou une acceptation supplémentaire non autorisée.

Même en cas de Go technique, une décision séparée sera requise avant d'étendre le volume à BTC/ETH, plusieurs journées ou une expérience prédictive.

## Conditions d'usage

Le pilote reste une recherche locale, publique et en lecture seule. Il ne redistribue pas les données OKX et reste soumis à l'accord API et aux conditions régionales applicables. Aucun secret, compte exchange, ordre, levier, dérivé, service SmartFolio, interface ou environnement de production n'entre dans le périmètre.

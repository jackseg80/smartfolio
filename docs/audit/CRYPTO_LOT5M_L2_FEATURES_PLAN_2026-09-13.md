# Plan préenregistré — lot 5M, features L2 causales

Date de gel : 13 septembre 2026, avant calcul des features sur le corpus 5L.

## Objectif

Définir et produire une table de features instantanées à partir des 863 carnets complets du corpus 5L. Ce lot ne contient aucune cible, aucun rendement futur, aucune agrégation temporelle et aucun entraînement de modèle.

## Sources figées

- corpus : `crypto-forecast-okx-l2-compact-corpus-v1-e229d0bfb9fef72b` ;
- résultat corpus SHA-256 : `51f8f01cf63cb89444c7841b58475d6675e89fab85353ce84a51edc29b26e758` ;
- neuf fichiers gzip épinglés individuellement par taille et SHA-256 ;
- référence de contrôle : métriques alignées du lot 5G, SHA-256 `cb761d58f71713eafc11976de4bd0be68eab7542711fca211d1fd9aa0a31218e` ;
- grille attendue : 864 lignes, 863 disponibles et une absence ETH-USDT à minuit UTC le 1er avril 2023.

### Addendum d'intégrité avant calcul réussi

Le premier lancement s'est arrêté avant tout calcul de feature : le lecteur JSONL a détecté que le corpus 5L initial concaténait les objets sans séparateur de ligne. Aucun critère, feature ou seuil n'a été consulté ni modifié. Le producteur a été corrigé et testé, puis 5K et 5L ont été régénérés deux fois depuis les mêmes archives.

L'entrée effectivement utilisée est le corpus corrigé `crypto-forecast-okx-l2-compact-corpus-v1-1df479332e50e3d1`, résultat SHA-256 `bb597dbc71f9b001610c44c8f0772a9f2c46ecee74509974ccd3bb8203105e59`. Les deux références initiales restent ci-dessus afin de préserver l'historique du gel, mais l'artifact `e229d0bfb9fef72b` est remplacé et ne doit plus être consommé.

## Features gelées

Chaque ligne disponible est calculée uniquement depuis son propre snapshot :

- meilleur bid, meilleur ask, mid, spread et nombres de niveaux ;
- tailles au meilleur bid et ask, déséquilibre top-of-book ;
- microprice et son écart au mid en points de base ;
- profondeurs notionnelles bid/ask et déséquilibres dans les bandes 5, 10, 25 et 50 bps ;
- part des cinq niveaux les plus proches dans la profondeur à 50 bps, séparément au bid et à l'ask.

Aucune différence temporelle, moyenne mobile, normalisation transversale, forward fill ou interpolation n'est autorisée.

## Contrôles gelés

1. Vérifier l'identité et les limites de taille des neuf fichiers avant lecture.
2. Lire chaque gzip ligne par ligne, avec limites sur volume décompressé, taille de ligne et nombre de lignes.
3. Vérifier les timestamps de grille, source et disponibilité ainsi que l'unicité de chaque clé.
4. Produire explicitement la ligne indisponible avec des cellules de features vides.
5. Exiger des nombres finis, un spread positif, des déséquilibres dans `[-1, 1]`, un microprice entre bid et ask et des parts dans `[0, 1]`.
6. Comparer les 15 métriques déjà présentes au lot 5G avec tolérances absolue `1e-9` et relative `1e-12`.
7. Vérifier par mutation qu'une modification d'un snapshot futur ne change aucune feature antérieure.
8. Exclure le timestamp de génération de l'identité et reproduire l'artifact dans un second répertoire.

## Critères Go/No-Go

Le lot est **Go technique features** seulement si :

- les 864 lignes sont produites dans l'ordre chronologique gelé ;
- 863 lignes disposent de 25 features finies et une seule reste explicitement indisponible ;
- aucune divergence n'existe avec les métriques 5G de référence ;
- le test de mutation future et la reproduction complète passent ;
- les tests ciblés, Ruff et Black passent.

Toute divergence reste un No-Go ; aucune feature ni tolérance ne sera modifiée après observation.

## Hors périmètre

- cible, rendement futur, label, modèle ou sélection de feature ;
- backtest, allocation, signal ou seuil de décision ;
- nouvelle archive, téléchargement ou collecte récurrente ;
- API, interface, port local ou production ;
- compte, secret, ordre, dérivé ou levier ;
- suppression, commit, push ou déploiement.

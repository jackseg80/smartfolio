# Résultat — lot 5N, faisabilité des cibles et de l'évaluation L2

Date d'exécution : 13 septembre 2026<br>
Décision : **NO-GO PRÉDICTIF**

## Conclusion

La table 5M est intègre et tous ses neuf couples instrument-date atteignent le seuil technique de couverture. Elle ne permet cependant pas une expérience prédictive conforme au protocole causal SmartFolio.

Les `864` lignes de grille ne représentent que `3` dates UTC indépendantes. Les compter comme 864 échantillons constituerait une pseudo-réplication : des lignes de la même journée partageraient le même environnement de marché et des cibles 7/30 jours fortement communes.

Conformément au plan gelé, aucune cible n'a été créée, aucun prix futur n'a été lu et aucun modèle n'a été entraîné.

## Contrôles

| Contrôle | Observé | Requis | Écart | Verdict |
|---|---:|---:|---:|---|
| Lignes totales | 864 | 864 | 0 | Pass |
| Lignes disponibles | 863 | 863 | 0 | Pass |
| Lignes manquantes | 1 | 1 | 0 | Pass |
| Couples instrument-date éligibles | 9 | 9 | 0 | Pass |
| Dates UTC indépendantes | 3 | 480 | -477 | **Fail** |
| Étendue calendaire inclusive | 1 188 jours | 1 551 jours | -363 | **Fail** |
| Instruments du panel | 3 | 5 | -2 | **Fail** |

Le manque ETH-USDT de `2023-04-01 00:00:00 UTC` ne rend pas sa journée inéligible : le groupe conserve `95/96` observations, exactement le minimum préenregistré.

## Contrat de cibles conservé pour une phase ultérieure

- horizons : 7 et 30 jours calendaires ;
- entrée : clôture OKX spot `1Dutc` à la fin de la journée d'observation ;
- sortie : clôture `1Dutc` exactement à l'horizon ;
- disponibilité : seulement après la clôture de la barre de sortie ;
- rendements : absolu, excès par rapport au cash USD à rendement nul et relatif à BTC ;
- labels : dérivés uniquement lorsque les rendements existent, avec calibration chronologique distincte ;
- manque : indisponible explicitement, sans nearest, interpolation ou remplissage.

Une source quotidienne OKX devra être épinglée par artifact et SHA-256 avant toute future matérialisation. Elle n'a volontairement pas été consultée ici.

## Protocole d'évaluation gelé

- fenêtres calendaires : 730 jours d'entraînement, 183 de calibration, 183 de test de développement et 365 de confirmation finale ;
- trois purges de 30 jours ;
- étendue totale minimale : 1 551 jours ;
- minimum conservateur : 120 dates indépendantes par partition, soit 480 au total ;
- toutes les lignes d'une journée restent dans la même partition ;
- panel minimal : cinq instruments ;
- aucun split aléatoire ni sélection sur la confirmation finale.

## Reproductibilité

- artifact final revu : `crypto-forecast-okx-l2-evaluation-feasibility-v1-22462ef298ab08b9` ;
- résultat SHA-256 : `695137a6e2c1ad640a0dc5b724c70c08e2d04ad160f2e281a06de5e3dd4eff18` ;
- configuration SHA-256 : `3a8bf93c7bc79daf6ced1a9e769163d62c77dac9eb730df0145a356a2e4b3dd7` ;
- code de faisabilité SHA-256 : `78a4d02799d598ceff86dfe5029c33ef44019cefcd9e4936cc07e75f5218a88a`.

Deux exécutions indépendantes ont produit le même identifiant et le même résultat. Le timestamp de manifeste est exclu de l'identité.

## Validation

- 5 tests unitaires 5N réussis ;
- refus explicite de compter les lignes intrajournalières comme échantillons indépendants ;
- refus d'une provenance 5M altérée ;
- couverture insuffisante maintenue comme No-Go sans corrompre le statut d'intégrité ;
- 100 tests unitaires de recherche crypto réussis au total ;
- Ruff et Black conformes ;
- seul l'avertissement Starlette/httpx déjà extérieur au chantier subsiste.

## Garde-fous confirmés

- aucun prix futur lu et aucune cible matérialisée ;
- aucune agrégation ou sélection de feature ;
- aucun modèle, backtest, allocation ou signal ;
- aucune requête réseau, collecte, suppression, API, interface ou production ;
- aucun secret, ordre, commit, push ou déploiement.

## Décision de reprise

La branche prédictive L2 doit rester arrêtée. Les features 5M demeurent une preuve technique, pas une amélioration démontrée du portefeuille.

La prochaine décision n'est pas un choix de modèle : il faut soit autoriser un nouveau plan d'acquisition couvrant au moins cinq instruments et visant 480 journées éligibles sur une étendue de 1 551 jours, soit clore la piste prédictive L2 et conserver uniquement le socle quotidien actuel. Les seuils ne doivent pas être abaissés pour rendre les trois journées exploitables.

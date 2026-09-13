# Résultat — lot 5M, features L2 causales

Date d'exécution : 13 septembre 2026<br>
Décision : **GO_TECHNIQUE POUR LA TABLE DE FEATURES L2**

## Conclusion

Le corpus 5L corrigé produit une table déterministe de `864` lignes : `863` lignes disponibles avec `25` features finies et une ligne explicitement indisponible pour ETH-USDT à `2023-04-01 00:00:00 UTC`.

Ce Go valide seulement le calcul causal et la traçabilité des features instantanées. Il ne mesure aucune capacité prédictive : aucune cible, aucun rendement futur, aucune normalisation, aucune agrégation temporelle et aucun modèle ne sont présents dans ce lot.

## Incident d'intégrité en entrée

Le premier lancement s'est arrêté avant le calcul car les anciens fichiers `jsonl.gz` 5K–5L concaténaient les objets sans retour à la ligne. Le défaut a été corrigé au niveau du producteur et protégé par un test de décompression et de parsing ligne par ligne. Les artifacts initiaux restent conservés mais sont remplacés ; les références valables sont désormais :

- extraction 5K : `crypto-forecast-okx-l2-progressive-extraction-v1-8ca057800a4ec6e4` ;
- corpus 5L : `crypto-forecast-okx-l2-compact-corpus-v1-1df479332e50e3d1` ;
- résultat 5L SHA-256 : `bb597dbc71f9b001610c44c8f0772a9f2c46ecee74509974ccd3bb8203105e59`.

Aucune feature, tolérance, règle d'alignement ou observation n'a été changée à la suite de cet incident.

## Table produite

| Contrôle | Résultat |
|---|---:|
| Lignes attendues / produites | 864 / 864 |
| Lignes disponibles | 863 |
| Lignes manquantes explicites | 1 |
| Features par ligne disponible | 25 |
| Comparaisons numériques avec 5G | 12 945 |
| Divergences avec 5G | 0 |
| Erreur absolue maximale | 0 |

Les features couvrent les meilleurs prix et tailles, le spread, le microprice, les déséquilibres, les profondeurs notionnelles à 5/10/25/50 bps et la concentration des cinq premiers niveaux dans la bande de 50 bps.

## Garanties causales

- chaque ligne disponible est calculée depuis un seul snapshot autonome ;
- le timestamp de grille, le timestamp source et le timestamp de disponibilité restent séparés ;
- la disponibilité vaut le maximum entre grille et source ;
- une mutation d'un snapshot futur ne modifie pas une ligne antérieure ;
- le manque attendu reste une ligne vide, sans interpolation ni forward fill ;
- les noms `top5_50bps` sont verrouillés à exactement cinq niveaux et 50 bps ;
- les politiques de temps, de manque et d'absence de cible sont refusées si elles diffèrent du contrat gelé.

## Reproductibilité

- artifact final revu : `crypto-forecast-okx-l2-features-v1-de3238a8dcf7a722` ;
- résultat SHA-256 : `4f1d5bb9c37dfaac655c3affe3aad7f7ddf6b911bd269a47429971247c8071ac` ;
- table CSV SHA-256 : `0175ce8c9725e109cf7744cee8c07ca337e559ea614aade0decd2e232a5ff799` ;
- configuration SHA-256 : `2b6b16f3a55c9319aa1eba8e6d908fec47840f32d3560c6249a4e39e8a144edf` ;
- code des features SHA-256 : `7db105ebb6e98f739c95b88a0d551058cad38a4ad4c20fbbba22730fd8e37f96`.

Deux générations indépendantes ont produit le même identifiant, le même résultat et la même table CSV. Les timestamps propres aux manifestes sont exclus de l'identité scientifique.

## Validation

- 11 tests unitaires 5M réussis ;
- 6 tests 5K réussis, dont la régression de framing JSONL ;
- 95 tests unitaires de recherche crypto réussis au total ;
- Ruff conforme sur les fichiers modifiés ;
- Black conforme ;
- seul l'avertissement Starlette/httpx déjà extérieur au chantier subsiste.

## Garde-fous confirmés

- aucune requête réseau, archive nouvelle ou donnée supprimée ;
- aucune cible, sélection de feature, normalisation ou entraînement ;
- aucun secret, compte, ordre, dérivé ou levier ;
- aucune API, interface, tâche récurrente ou production touchée ;
- aucun commit, push ou déploiement.

## Limites et suite

Les observations couvrent seulement trois journées isolées. Les 863 lignes intrajournalières ne constituent donc pas 863 journées indépendantes et ne peuvent pas justifier une conclusion prédictive.

Cette suite a été exécutée au lot 5N. Le contrôle préenregistré conclut au No-Go prédictif : les trois journées sont insuffisantes et renvoient à une décision séparée sur l'acquisition prospective.

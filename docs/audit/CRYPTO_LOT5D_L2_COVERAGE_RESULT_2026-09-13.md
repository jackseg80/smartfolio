# Résultat — lot 5D, couverture et coût des snapshots L2 OKX

Date : 13 septembre 2026<br>
Périmètre : métadonnées publiques de 42 archives potentielles, sans téléchargement d'archive L2, compte, secret, ordre, modèle ou environnement SmartFolio.

## Décision

Le relevé passe les critères préenregistrés avec une couverture de **42/42**. Le prochain échantillon snapshot-only multi-actifs, limité aux trois dates gelées, est **matériellement faisable** : les neuf archives existent et totalisent `1 126,78 MB`, sous le plafond de `2 000 MB`.

La décision normalisée est `GO_BOUNDED_SAMPLE_METADATA`.

Ce Go porte uniquement sur la disponibilité et le volume annoncé des fichiers. Il ne vaut pas autorisation de les télécharger et ne prouve ni la qualité interne des snapshots, ni une valeur prédictive.

## Relevé préenregistré

- source : endpoint public OKX `GET /api/v5/public/market-data-history` ;
- module : `4`, carnet Spot à 400 niveaux ;
- instruments : `BTC-USDT`, `ETH-USDT`, `SOL-USDT` ;
- dates : premier jour UTC de 14 trimestres entre avril 2023 et juillet 2026 ;
- couples instrument-date attendus : `42` ;
- fichiers présents et valides : `42` ;
- fichiers manquants : `0` ;
- archives téléchargées : `0`.

Chaque métadonnée présente possède la date et le nom attendus, une URL HTTPS sous `static.okx.com`, une taille positive et aucun doublon. Aucune date de remplacement n'a été introduite.

## Distribution des tailles annoncées

| Périmètre | Observations | Minimum | Médiane | P95 | Maximum |
|---|---:|---:|---:|---:|---:|
| BTC-USDT | 14 | 55,35 MB | 193,89 MB | 275,389 MB | 356,34 MB |
| ETH-USDT | 14 | 50,83 MB | 165,445 MB | 283,8835 MB | 295,72 MB |
| SOL-USDT | 14 | 34,33 MB | 104,855 MB | 180,349 MB | 183,82 MB |
| Journée complète BTC + ETH + SOL | 14 | 144,10 MB | 490,61 MB | 715,3065 MB | 767,69 MB |

La taille varie fortement avec la période. La journée isolée du lot 5C ne constituait donc pas une base suffisante pour budgéter un historique.

À partir de la médiane observée de `490,61 MB` par journée complète :

- 30 jours représenteraient environ `14 718,30 MB` ;
- 365,25 jours représenteraient environ `179 195,30 MB`.

Ces valeurs sont des extrapolations indicatives, pas des volumes garantis.

## Échantillon candidat gelé

| Date UTC | BTC-USDT | ETH-USDT | SOL-USDT | Total |
|---|---:|---:|---:|---:|
| 2023-04-01 | 55,35 MB | 54,42 MB | 34,33 MB | 144,10 MB |
| 2024-07-01 | 179,19 MB | 161,91 MB | 178,48 MB | 519,58 MB |
| 2026-07-01 | 213,05 MB | 168,98 MB | 81,07 MB | 463,10 MB |
| **Cumul** | 447,59 MB | 385,31 MB | 293,88 MB | **1 126,78 MB** |

Cet échantillon couvre trois actifs et trois périodes éloignées. Un lot ultérieur devra encore préenregistrer le téléchargement, la vérification des archives, le traitement en flux et la politique de rétention avant toute acquisition.

## Reproductibilité

Artifact final : `crypto-forecast-okx-l2-coverage-v1-9b584745ea2d87c6`.

- résultat SHA-256 : `6d88532136cc96d7a9c081492a67119359b5c40d02f5f0913af480f9cd5f84a0` ;
- métadonnées CSV SHA-256 : `e5d9d80b5d50657735a5b229b3d15c4f6be67a1955906c44d5f25fd043210d2d` ;
- configuration SHA-256 : `7f7dd3dfc3099a1e29733e13e6d70eb7b33eac4bf4f7034f2cf02e9b8b5a812a` ;
- code d'acquisition SHA-256 : `ae68969651a1bec4e76abf0aa85f30f2c22eae52650b98c5bde0c95cdb03b547`.

Une seconde interrogation identique, écrite dans un répertoire séparé, a produit le même identifiant et les mêmes empreintes de résultat et de métadonnées.

## Garanties et limites

- aucune archive L2 n'a été appelée ou téléchargée ;
- aucune donnée de marché brute n'a été extraite ;
- aucun compte ni identifiant OKX n'a été utilisé ;
- aucun modèle, seuil de trading ou allocation n'a été créé ;
- aucune API, interface, production ou configuration utilisateur n'a été modifiée ;
- la présence et la taille annoncée ne prouvent pas que les fichiers sont intègres ou que leurs snapshots respectent le contrat du lot 5C ;
- la disponibilité future de ces URL n'est pas garantie.

## Sources officielles

- [OKX — API V5 et endpoint public Historical Market Data](https://www.okx.com/docs-v5/en/)
- [OKX — Historical Market Data](https://www.okx.com/en-us/historical-data)
- [OKX — API Agreement](https://www.okx.com/en-gb/help/okx-api-agreement)

## Suite recommandée

Le lot 5E sur les **neuf archives figées** est maintenant terminé. Les fichiers ont été acquis et conservés, mais le contrôle strict conclut No-Go à cause d'une cadence historique hétérogène. Le résultat complet se trouve dans `CRYPTO_LOT5E_L2_SAMPLE_RESULT_2026-09-13.md`.

Il n'y a toujours rien à tester dans le navigateur : l'application et la production restent inchangées.

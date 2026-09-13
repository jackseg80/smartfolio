# Résultat — lot 5C, carnet OKX par snapshots autonomes

Date : 13 septembre 2026<br>
Périmètre : 96 snapshots d'une archive publique SOL-USDT déjà acquise. Aucun nouveau téléchargement, compte, secret, ordre, modèle ou environnement SmartFolio.

## Décision

Le collecteur snapshot-only est **Go technique** sur le périmètre préenregistré.

Les `96/96` snapshots sont autonomes, valides, non croisés et espacés d'environ 15 minutes. Les `1 813 752` mises à jour intermédiaires sont comptées mais ne contribuent à aucune feature. Le problème de continuité des deltas identifié au lot 5B est donc éliminé par construction.

Ce Go ne constitue pas une validation prédictive. Il autorise seulement à considérer les snapshots complets comme une source de données candidate pour une étude de couverture bornée.

## Entrée

- fournisseur : OKX Historical Market Data, Spot, module `4` à 400 niveaux ;
- instrument : `SOL-USDT` ;
- date : `2026-09-10` UTC ;
- archive : `SOL-USDT-L2orderbook-400lv-2026-09-10.tar.gz` ;
- taille compressée : `49 738 619` octets ;
- taille interne : `298 647 927` octets ;
- SHA-256 : `5021fdb3a0413e8f8f03524dcce39805cbdc53e2b922596d519f44254b0f3c46`.

Le membre brut a été lu en flux et n'a jamais été extrait durablement sur disque.

## Critères préenregistrés

| Critère | Résultat | Décision |
|---|---:|---|
| Nombre exact de snapshots | 96 | Passe |
| Snapshots valides et non croisés | 96/96 | Passe |
| Premier snapshot | 00:00:00.000 UTC | Passe |
| Dernier snapshot | 23:45:00.004 UTC | Passe |
| Intervalle attendu | 900 000 ms ± 1 000 ms | Passe |
| Intervalle observé min/médian/max | 899 991 / 900 000 / 900 007 ms | Passe |
| Updates utilisées dans les features | 0 | Passe |
| Sorties reproductibles | identiques | Passe |

Aucun seuil n'a été modifié après observation.

## Variables descriptives

Chaque ligne est calculée uniquement avec les niveaux présents dans le snapshot correspondant. Il n'y a ni propagation d'état, ni interpolation entre snapshots.

| Mesure | Médiane | P95 | Maximum |
|---|---:|---:|---:|
| Spread | 0,990 bp | 1,008 bp | 1,013 bp |
| Profondeur bid à 10 bp | 197 913 USDT | 268 197 USDT | 359 851 USDT |
| Profondeur ask à 10 bp | 187 783 USDT | 260 751 USDT | 280 902 USDT |
| Déséquilibre à 10 bp | 0,027 | 0,243 | 0,486 |
| Profondeur bid à 50 bp | 1 081 596 USDT | 1 313 176 USDT | 1 703 010 USDT |
| Profondeur ask à 50 bp | 886 303 USDT | 1 014 287 USDT | 1 134 492 USDT |

Ces chiffres décrivent une journée et une paire. Ils ne doivent pas être interprétés comme des seuils de trading ou une liquidité garantie.

## Test d'isolation central

Le test remplace toutes les mises à jour intermédiaires par des valeurs différentes, y compris un bid artificiellement extrême. Les 96 lignes de features snapshot restent strictement identiques.

Ce test prouve que le nouveau chemin snapshot-only ne dépend pas accidentellement des deltas rejetés au lot 5B.

## Reproductibilité

Artifact final : `crypto-forecast-okx-l2-snapshot-pilot-v1-abc1878b6784aeb4`.

- résultat SHA-256 : `6489b2ca5468d4127145456591f42516b3316eaea4e7d0c15f53de4af54feb54` ;
- métriques SHA-256 : `146819c51b9f20426f9a275c9208a8245409ce93ab2dc5a639261ed744cc6f7c` ;
- configuration SHA-256 : `a186ae540b1a9fa18891e22191a92f9d163f05878bda589dd0e64e652d8fdd18` ;
- code d'analyse SHA-256 : `4b63844aa1fc3dff46991a1ca9007fc04d0e6135aff3cad0a02cbfc280d6e1e8`.

Une seconde exécution isolée a produit le même identifiant et les mêmes empreintes, octet pour octet.

Validation du code : les trois tests propres au lot 5C réussissent et la suite ciblée complète des lots de recherche totalise `42/42` tests réussis. Ruff et Black sont conformes. L'avertissement Starlette/httpx observé pendant pytest est extérieur à ce chantier.

## Coût avant extension

Les métadonnées du 10 septembre 2026 indiquent `249,61 MB` compressés pour une seule journée BTC + ETH + SOL. À taille quotidienne constante, trente jours représenteraient environ `7,49 GB` et un an environ `91,1 GB`. Cette extrapolation sert uniquement à borner le stockage ; la taille réelle varie selon l'activité.

Un historique large ne doit donc pas être téléchargé à l'aveugle. L'étape suivante doit d'abord interroger uniquement les métadonnées de plusieurs dates calendaires préenregistrées, sans télécharger les archives, afin d'estimer la distribution réelle des tailles et la stabilité du nombre de snapshots.

## Sources officielles

- [OKX — API V5 et endpoint public Historical Market Data](https://www.okx.com/docs-v5/en/)
- [OKX — Historical Market Data](https://www.okx.com/en-us/historical-data)
- [OKX — API Agreement](https://www.okx.com/en-gb/help/okx-api-agreement)

## Limites et suite

- une seule paire et une seule journée ;
- aucune validation de stabilité inter-actifs ou inter-périodes ;
- aucune cible à 7/30 jours ;
- aucune sélection de feature ou de modèle ;
- aucune conclusion après frais ;
- aucune modification produit.

Le lot 5D de **faisabilité de couverture** est maintenant terminé. Les 42 métadonnées attendues sont valides et les neuf archives de l'échantillon candidat totalisent environ `1,13 GB`. Ce résultat est documenté dans `CRYPTO_LOT5D_L2_COVERAGE_RESULT_2026-09-13.md` et ne vaut pas autorisation de téléchargement.

Il n'y a rien à tester dans le navigateur : l'API, l'interface et la production restent inchangées.

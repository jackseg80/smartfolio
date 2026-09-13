# Résultat — lot 5E, échantillon snapshot-only multi-actifs

Date : 13 septembre 2026<br>
Périmètre : neuf archives publiques OKX Spot, trois actifs et trois journées UTC. Aucun compte, secret, ordre, modèle, environnement SmartFolio ou extraction brute durable.

## Décision

Le lot est **`NO_GO_TECHNICAL` selon le contrat préenregistré**.

Les neuf archives ont été téléchargées et vérifiées, et les `8 927/8 927` snapshots observés sont valides et non croisés. L'échec vient de l'hypothèse de cadence unique :

- en 2023 et 2024, les archives contiennent environ un snapshot par minute, soit 1 439 ou 1 440 par jour, au lieu des 96 attendus ;
- en 2026, les trois archives contiennent bien 96 snapshots espacés de 15 minutes, mais le premier arrive de 2 à 6 millisecondes après minuit au lieu de l'égalité exacte exigée.

Ce résultat ne signifie donc pas que les snapshots sont corrompus ou moins fiables. Il montre que le format temporel historique n'est pas homogène et que le contrat issu de la journée du lot 5C ne peut pas être appliqué tel quel à toutes les périodes.

Les seuils n'ont pas été modifiés après observation.

## Acquisition

| Date UTC | Instrument | Taille réelle | SHA-256 |
|---|---|---:|---|
| 2023-04-01 | BTC-USDT | 58 040 178 octets | `8bcda2a4741e060615c4aa41792a1cd352430e95e822229fef9eb9e745e0f202` |
| 2023-04-01 | ETH-USDT | 57 063 560 octets | `c0d0a09767e91a8527c094ef34834d5b7789a52e362232ccafafdbb024dc0890` |
| 2023-04-01 | SOL-USDT | 35 996 898 octets | `c15de3aeb333e5286ab4f40fe6f2f9ce83b6a7a4116b7c9da3d2127e4ee1cd88` |
| 2024-07-01 | BTC-USDT | 187 890 212 octets | `0c04a4025fc155f164df9d090960b0a5e94154df003579226f31c3529de0f1a6` |
| 2024-07-01 | ETH-USDT | 169 773 160 octets | `85dd7739bcba0f1fb29506349c76cb2495654428881e54f5e72954469c9144bb8` |
| 2024-07-01 | SOL-USDT | 187 146 766 octets | `8f7e81512a2edc9a3f5fb7dec7ba6a833e3644e8a8bdf532a797334212d5ca615` |
| 2026-07-01 | BTC-USDT | 223 399 810 octets | `975421f369661edc45e80a70923248760a37bb07f85b54aa9d9e829b75c218bb4` |
| 2026-07-01 | ETH-USDT | 177 184 431 octets | `ae0af8d5c7bab437a72ae6d362c1812ed13200f82ed9d0b55f2c0770ace1b9d5f` |
| 2026-07-01 | SOL-USDT | 85 008 604 octets | `c21100ac72c55c67fd898d7610868737b7c6e99696411900476941f0749f04d7c` |

Total réel : `1 181 503 619` octets, soit environ `1 126,77 MiB`. Toutes les tailles respectent les métadonnées gelées et les plafonds du lot.

Les fichiers validés sont conservés sous `outputs/crypto-forecast-l2-sample/raw/`. Ils ne doivent être supprimés qu'après une autorisation distincte.

## Contrôle par archive

| Date UTC | Instrument | Enregistrements | Updates ignorées | Snapshots | Valides | Cadence médiane | Décision |
|---|---|---:|---:|---:|---:|---:|---|
| 2023-04-01 | BTC-USDT | 2 793 048 | 2 791 608 | 1 440 | 1 440 | 60 000 ms | No-Go |
| 2023-04-01 | ETH-USDT | 2 951 430 | 2 949 991 | 1 439 | 1 439 | 60 000 ms | No-Go |
| 2023-04-01 | SOL-USDT | 1 706 751 | 1 705 311 | 1 440 | 1 440 | 60 000 ms | No-Go |
| 2024-07-01 | BTC-USDT | 5 265 650 | 5 264 210 | 1 440 | 1 440 | 60 000 ms | No-Go |
| 2024-07-01 | ETH-USDT | 4 691 619 | 4 690 179 | 1 440 | 1 440 | 60 000 ms | No-Go |
| 2024-07-01 | SOL-USDT | 4 901 119 | 4 899 679 | 1 440 | 1 440 | 60 000 ms | No-Go |
| 2026-07-01 | BTC-USDT | 5 517 408 | 5 517 312 | 96 | 96 | 900 000 ms | No-Go |
| 2026-07-01 | ETH-USDT | 4 364 115 | 4 364 019 | 96 | 96 | 900 000 ms | No-Go |
| 2026-07-01 | SOL-USDT | 2 616 756 | 2 616 660 | 96 | 96 | 900 000 ms | No-Go |

Au total, `34 798 969` actions `update` ont été comptées et ignorées. Elles n'ont contribué à aucune feature.

### Détail des écarts temporels

- 2023 BTC et SOL commencent 13 et 33 ms après minuit ; ETH commence à 00:01:00.028 et n'a que 1 439 snapshots.
- 2024 BTC, ETH et SOL commencent 9, 15 et 450 ms après minuit.
- 2026 BTC, ETH et SOL commencent 2, 6 et 4 ms après minuit ; leur cadence de 15 minutes passe la tolérance de ±1 000 ms.
- Les six archives 2023–2024 échouent au nombre attendu, à la cadence attendue et à l'égalité exacte de la borne initiale.
- Les trois archives 2026 échouent uniquement à l'égalité exacte de la borne initiale.

## Reproductibilité

Artifact final : `crypto-forecast-okx-l2-sample-v1-973d02b919af2d1c`.

- résultat SHA-256 : `b32045a424853d22de978a8a3302873c74f64ef0d1d91203455853b713a3d6e2` ;
- métriques SHA-256 : `27c102f090b48521267b95f11e7c2aafb09327cd85cefe0ca75d4a0b4984e58e` ;
- résumé des archives SHA-256 : `40644c02ea2b4a1d6891e36079a6f4edb27bb1b882ced3d47ea123b45d0fa034` ;
- identité d'acquisition SHA-256 : `458044e69fd4c5e439ffbd1a53ace7ce54d3a6e226f13acbf1ba36f4a45ae3d5` ;
- configuration SHA-256 : `79f9aa0d9ee0b5740e26b0b0a0947706aad04b4fc95a4a890fe6ccbaef541de5` ;
- orchestration SHA-256 : `99144761d65ec06ecde7fcacf49e8a3510f90f3c03fe508bc2caeed84277c66b` ;
- analyse snapshot SHA-256 : `4b63844aa1fc3dff46991a1ca9007fc04d0e6135aff3cad0a02cbfc280d6e1e8`.

Une seconde analyse a relu et re-haché les mêmes neuf archives sans requête réseau. Elle a produit le même identifiant et les mêmes empreintes de résultat, métriques et résumé, octet pour octet.

## Garanties et limites

- téléchargement public uniquement, sans authentification ;
- traitement séquentiel et borné ;
- aucune extraction durable du membre brut ;
- aucun delta utilisé dans les features ;
- aucun modèle, target, backtest ou seuil de trading ;
- aucune modification de l'API, de l'interface ou de la production ;
- trois journées seulement : le point exact du changement de cadence n'est pas identifié ;
- les métriques restent descriptives et ne démontrent aucun avantage économique.

## Suite recommandée

Le lot 5F de normalisation temporelle est maintenant terminé. Sa règle limitée aux snapshots situés après la borne ne conserve que 814 des 864 créneaux et reste No-Go. Le résultat est documenté dans `CRYPTO_LOT5F_L2_NORMALIZATION_RESULT_2026-09-13.md` ; le présent No-Go 5E n'est pas réécrit rétroactivement.

Il n'y a rien à tester dans le navigateur : l'application et la production restent inchangées.

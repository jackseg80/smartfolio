# Plan préenregistré — lot 5C, carnet OKX par snapshots autonomes

Date de gel : 13 septembre 2026, après le No-Go du lot 5B et avant tout calcul snapshot-only.

## Objectif

Vérifier si les snapshots complets contenus dans l'archive L2 OKX peuvent produire des variables de microstructure causales et reproductibles sans dépendre des mises à jour intermédiaires dépourvues d'identifiants de séquence.

Le lot utilise uniquement l'archive SOL-USDT du 10 septembre 2026 déjà acquise et vérifiée au lot 5B. Aucun nouveau fichier de marché n'est téléchargé.

## Contrat de données

- fournisseur : OKX, historique public Spot, module `4` à 400 niveaux ;
- instrument : `SOL-USDT` ;
- journée : `2026-09-10` UTC ;
- SHA-256 de l'archive : `5021fdb3a0413e8f8f03524dcce39805cbdc53e2b922596d519f44254b0f3c46` ;
- événements admis pour les features : `action=snapshot` uniquement ;
- événements `action=update` : comptés puis ignorés intégralement ;
- aucune propagation, interpolation ou reconstruction entre deux snapshots.

Chaque snapshot est une observation autonome à son timestamp. Les variables d'un snapshot sont calculées exclusivement avec ses propres niveaux bid/ask.

## Variables figées

Pour chaque snapshot :

- meilleur bid, meilleur ask, mid-price et spread en points de base ;
- nombre de niveaux de chaque côté ;
- profondeur notionnelle bid et ask en USDT à 10, 25 et 50 points de base du mid-price ;
- déséquilibre normalisé bid/ask pour chaque bande.

Le troisième élément entier de chaque niveau est conservé comme compteur auxiliaire observé, sans signification économique attribuée faute de contrat exhaustif du format historique dans la documentation publique consultée.

## Critères Go/No-Go

Le collecteur snapshot-only est **Go technique** uniquement si :

1. les `96` snapshots annoncés par le lot 5B sont tous retrouvés ;
2. les timestamps sont strictement croissants ;
3. le premier snapshot se situe au début de la journée UTC et le dernier à moins de 15 minutes de sa fin ;
4. chaque intervalle entre snapshots reste dans `900 000 ms ± 1 000 ms` ;
5. chaque snapshot possède au moins un niveau bid et ask, sans dépasser 400 par côté ;
6. les prix et quantités sont finis et non négatifs ;
7. le meilleur bid ne dépasse jamais le meilleur ask ;
8. toutes les métriques produites sont finies ;
9. deux exécutions séparées produisent le même identifiant et les mêmes empreintes.

Tout échec reste publié. Aucun seuil ne sera ajusté après observation.

## Limites

Un Go technique ne démontrera que la possibilité d'extraire des snapshots propres sur cette paire et cette journée. Il ne démontrera pas :

- la stabilité sur BTC, ETH, d'autres actifs ou d'autres périodes ;
- la qualité prédictive à 7 ou 30 jours ;
- un avantage économique après frais ;
- la faisabilité d'un historique complet en volume de téléchargement.

Les métadonnées du lot 5B évaluent déjà BTC + ETH + SOL à `249,61 MB` compressés par jour. Toute extension de dates ou d'actifs exige donc un nouveau budget préenregistré.

## Fichiers prévus

1. `config/crypto_forecast_okx_l2_snapshot_pilot.json` — entrée, bandes et critères gelés.
2. `services/forecasting/okx_l2_snapshot_pilot.py` — lecture en flux et calcul indépendant des snapshots.
3. `scripts/run_crypto_forecast_l2_snapshot_pilot.py` — exécution hors ligne.
4. `tests/unit/test_crypto_forecast_okx_l2_snapshot_pilot.py` — isolation des updates, causalité et critères de décision.
5. `docs/audit/CRYPTO_LOT5C_SNAPSHOT_ONLY_RESULT_2026-09-13.md` — résultat et suite éventuelle.

## Hors périmètre

- entraînement ou sélection de modèle ;
- modification du dataset quotidien actuel ;
- API, interface, port 8082 ou production 8080 ;
- ordre, portefeuille réel, clé exchange, levier ou dérivé ;
- commit, push ou déploiement.

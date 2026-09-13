# Résultat — lot 5B, pilote de carnet L2 OKX

Date : 13 septembre 2026<br>
Périmètre : une archive publique OKX Spot, une paire et une journée UTC. Aucun compte, secret, ordre, modèle, service SmartFolio ou environnement de production.

## Décision

Le pilote est **No-Go technique pour une reconstruction fondée sur toutes les mises à jour L2**.

La raison est précise : les `1 813 848` enregistrements de la journée ne contiennent jamais `seqId` ni `prevSeqId`. L'ordre temporel est lisible, mais une mise à jour absente ne serait pas détectable. Le protocole préenregistré interdisait de réparer ou d'ignorer silencieusement ce risque et exigeait une continuité vérifiable à au moins 99 %.

Ce No-Go ne signifie pas que l'archive est corrompue ou sans intérêt. Elle contient `96` snapshots autonomes, espacés d'environ 15 minutes. Une étude suivante peut donc utiliser **uniquement les snapshots complets** et ignorer les deltas non vérifiables. Cette piste doit avoir son propre protocole avant toute extension de volume.

## Entrée figée

- source : endpoint public OKX `GET /api/v5/public/market-data-history` ;
- module : `4`, carnet à 400 niveaux ;
- instrument : `SOL-USDT` spot ;
- date : `2026-09-10` UTC ;
- archive : `SOL-USDT-L2orderbook-400lv-2026-09-10.tar.gz` ;
- taille annoncée : `47.43 MB` ;
- taille reçue : `49 738 619` octets ;
- SHA-256 : `5021fdb3a0413e8f8f03524dcce39805cbdc53e2b922596d519f44254b0f3c46` ;
- membre interne : `298 647 927` octets, lu en flux et jamais extrait durablement.

Les plafonds gelés étaient de 55 000 000 octets compressés, 2 Gio décompressés et 10 000 000 d'enregistrements. Ils ont tous été respectés.

## Format observé

Le membre interne est un JSON par ligne. Chaque message possède exactement les champs `instId`, `action`, `ts`, `asks` et `bids`.

- le premier état et les remises à zéro périodiques utilisent `action=snapshot` ;
- les changements intermédiaires utilisent `action=update` ;
- chaque niveau contient trois valeurs textuelles observées : prix, quantité et un compteur auxiliaire entier dont la signification exacte n'est pas attribuée ici ;
- une quantité nulle supprime le niveau ;
- les timestamps sont ordonnés et couvrent la journée UTC jusqu'à `23:59:59.654` ;
- l'écart maximal entre deux messages consécutifs est `1 044 ms`.

La documentation publique décrit le catalogue, les tailles et les URL de fichiers, mais ne fournit pas dans la section consultée un contrat exhaustif du schéma interne de l'archive. Le troisième champ reste donc conservé comme valeur observée et ne doit pas recevoir une sémantique économique non prouvée.

## Contrôles de carnet

L'analyse a reconstruit un état causal sur une grille d'une minute : aucune mise à jour postérieure n'est utilisée pour une minute antérieure.

- échantillons minute : `1 440` ;
- carnets valides et non croisés : `1 440 / 1 440` ;
- snapshots : `96` ;
- mises à jour : `1 813 752` ;
- cadence médiane des snapshots : `900 000 ms` ;
- couverture des champs de séquence : `0 %`.

Les statistiques utilisant les deltas sont publiées à titre descriptif uniquement :

| Mesure | Médiane | P95 | Maximum |
|---|---:|---:|---:|
| Spread | 0,991 bp | 1,008 bp | 2,001 bp |
| Profondeur bid à 10 bp | 193 217 USDT | 262 128 USDT | 878 342 USDT |
| Profondeur ask à 10 bp | 188 179 USDT | 261 285 USDT | 506 974 USDT |
| Déséquilibre à 10 bp | 0,015 | 0,264 | 0,657 |

Ces valeurs prouvent que le parseur et les calculs fonctionnent sur le fichier. Elles ne prouvent ni l'exhaustivité des deltas, ni une capacité prédictive, ni un avantage après frais.

## Reproductibilité

Artifact final : `crypto-forecast-okx-l2-pilot-v1-896ad4df91db342d`.

- résultat SHA-256 : `b96811727de664cb615d2b63c83eec32f685fdaeac510d8408ce20d4c40f2ab4` ;
- métriques minute SHA-256 : `80f65470c48b5611b5628fa68f8e3207e721f47851abd5d5c5058c6526c82e3f` ;
- configuration SHA-256 : `b0fde44b45921be636b494ffb6e9eb11d17d4e23cffc182e02f218adba974789` ;
- code d'analyse SHA-256 : `5318a0ff93c5cb3e3b5f3ae6108492f6484c686a6021071fd8f7fdc19ca5b29c`.

Une seconde exécution dans un répertoire séparé a produit le même identifiant, le même résultat et le même fichier de métriques, octet pour octet.

Validation du code : les trois tests propres au pilote réussissent et la suite ciblée complète des lots de recherche totalise `39/39` tests réussis. Ruff et Black sont conformes. L'avertissement Starlette/httpx observé pendant pytest est extérieur à ce chantier.

## Taille et choix du module

Les métadonnées publiques du même jour montrent :

| Module | BTC + ETH + SOL, compressé | État |
|---|---:|---|
| 400 niveaux (`4`) | 249,61 MB | format actuel retenu |
| 50 niveaux (`6`) | 493,00 MB | dépréciation progressive annoncée |

Le format 50 niveaux aurait donc été à la fois plus gros et moins durable. Aucune archive BTC ou ETH n'a été téléchargée.

## Sources officielles

- [OKX — API V5, données historiques et paramètres des modules](https://www.okx.com/docs-v5/en/)
- [OKX — page des données historiques](https://www.okx.com/en-us/historical-data)
- [OKX — accord API](https://www.okx.com/en-gb/help/okx-api-agreement)

OKX indique que les données de carnet sont généralement disponibles à T+3, que le module `4` fournit 400 niveaux, que le module `5` fournit 5 000 niveaux depuis le 1er novembre 2025 et que le module `6` à 50 niveaux sera progressivement retiré. L'accès utilisé ici est public, sans authentification ; l'usage reste soumis aux conditions régionales et à l'accord API applicables.

## Suite recommandée

Ne pas intégrer les deltas de cette archive au modèle.

La prochaine étape raisonnable est une étude **snapshot-only** : calculer les mêmes variables exclusivement sur les 96 carnets complets, vérifier la stabilité de leur cadence sur quelques journées déterminées à l'avance, puis estimer le coût réel d'une couverture historique. Elle doit rester séparée du modèle tant que sa couverture, sa provenance et son coût de stockage ne sont pas jugés recevables.

Il n'y a rien à tester dans le navigateur à ce stade : aucune interface, API locale ou production n'a été modifiée.

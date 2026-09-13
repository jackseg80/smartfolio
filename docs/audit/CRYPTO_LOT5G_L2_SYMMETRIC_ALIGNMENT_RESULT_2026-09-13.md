# Résultat — lot 5G, alignement causal symétrique des snapshots L2

Date : 13 septembre 2026<br>
Périmètre : alignement hors ligne des snapshots du lot 5E. Aucun réseau, téléchargement, delta, modèle ou environnement SmartFolio.

## Décision

Le lot est **`GO_TECHNICAL_ALIGNMENT` selon le contrat préenregistré**.

La grille contient `864` créneaux et `863` observations disponibles, soit une couverture de `99,884 %`. Les neuf séries passent le seuil de 95 créneaux sur 96. Le seul manque reste explicitement indisponible : ETH-USDT à 00:00 UTC le 1er avril 2023.

Ce Go valide uniquement la méthode d'alignement sur trois journées. Il ne valide aucune feature prédictive, stratégie ou allocation.

## Résultats par série

| Date UTC | Instrument | Disponibles | Manquants | Avant la borne | Exactement | Après la borne | Écart absolu maximal |
|---|---|---:|---:|---:|---:|---:|---:|
| 2023-04-01 | BTC-USDT | 96 | 0 | 8 | 2 | 86 | 299 ms |
| 2023-04-01 | ETH-USDT | 95 | 1 | 9 | 5 | 81 | 278 ms |
| 2023-04-01 | SOL-USDT | 96 | 0 | 2 | 5 | 89 | 491 ms |
| 2024-07-01 | BTC-USDT | 96 | 0 | 16 | 5 | 75 | 73 ms |
| 2024-07-01 | ETH-USDT | 96 | 0 | 9 | 3 | 84 | 64 ms |
| 2024-07-01 | SOL-USDT | 96 | 0 | 5 | 9 | 82 | 450 ms |
| 2026-07-01 | BTC-USDT | 96 | 0 | 0 | 8 | 88 | 9 ms |
| 2026-07-01 | ETH-USDT | 96 | 0 | 0 | 3 | 93 | 9 ms |
| 2026-07-01 | SOL-USDT | 96 | 0 | 0 | 16 | 80 | 9 ms |

Au total :

- 49 snapshots sont antérieurs à leur borne de grille ;
- 56 tombent exactement sur la borne ;
- 758 sont postérieurs à la borne ;
- 8 064 snapshots natifs ne sont pas sélectionnés.

Tous les écarts absolus sélectionnés sont inférieurs à 500 ms, donc sous la limite gelée de 1 000 ms.

## Horodatage causal

Chaque ligne disponible conserve :

- la borne théorique de grille ;
- le timestamp source ;
- l'écart signé et absolu ;
- l'heure de disponibilité causale.

Pour les 49 snapshots antérieurs, l'heure de disponibilité est la borne de grille. Pour les 758 snapshots postérieurs, elle est le timestamp source. Une observation n'est donc jamais présentée comme disponible avant d'exister.

Le créneau ETH-USDT manquant à minuit le 1er avril 2023 ne possède aucune métrique et n'est pas remplacé.

## Reproductibilité

Artifact final : `crypto-forecast-okx-l2-symmetric-alignment-v1-f201b14037ec95c9`.

- résultat SHA-256 : `c864b55a687ff7a2abb020637967af3970ee6eb88a95056b162e3aced12a3f6a` ;
- métriques alignées SHA-256 : `cb761d58f71713eafc11976de4bd0be68eab7542711fca211d1fd9aa0a31218e` ;
- configuration SHA-256 : `3b63153d7ecf01b6298ab70aea0d5d0c5ca8452173518942f36c02e16547df29` ;
- code d'alignement SHA-256 : `b53bc6fc2964f206bf5a0f82620ff4abd39b2da2bf353256a5e69f0219eaa212`.

Une seconde exécution a produit le même identifiant et les mêmes empreintes, octet pour octet.

## Contrôles

- le snapshot le plus proche est sélectionné dans la fenêtre symétrique ;
- une égalité préfère le snapshot antérieur ;
- aucun snapshot ne sert à deux créneaux ;
- la mutation d'un snapshot non sélectionné ne modifie aucune sortie ;
- aucun remplissage, interpolation ou delta ;
- aucune valeur non finie ;
- aucun accès réseau, compte, secret, ordre ou production.

## Limites

- trois actifs et trois journées seulement ;
- règle conçue après les constats exploratoires 5E et 5F ;
- un seul exemple de cadence native par grande période ;
- aucune cible à 7 ou 30 jours ;
- aucune preuve que les métriques L2 améliorent une prévision après coûts ;
- un historique continu resterait volumineux à acquérir.

## Suite recommandée

Ne pas passer directement au modèle. Le prochain jalon doit décider, sans téléchargement aveugle, quelle quantité minimale de journées continues serait nécessaire pour tester des cibles à 7 et 30 jours et si son coût de stockage/transfert reste acceptable.

Une alternative plus sobre est une collecte prospective des seuls snapshots nécessaires, qui évite les millions d'updates historiques mais exige plusieurs mois avant une validation sérieuse.

Il n'y a rien à tester dans le navigateur : l'application et la production restent inchangées.

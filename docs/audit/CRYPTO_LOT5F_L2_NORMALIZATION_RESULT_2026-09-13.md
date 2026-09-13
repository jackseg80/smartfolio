# Résultat — lot 5F, normalisation temporelle des snapshots L2

Date : 13 septembre 2026<br>
Périmètre : normalisation hors ligne de l'artefact snapshot du lot 5E. Aucun réseau, téléchargement, delta, modèle ou environnement SmartFolio.

## Décision

Le lot est **`NO_GO_TECHNICAL_NORMALIZATION` selon le contrat préenregistré**.

La grille contient bien 864 créneaux, mais seulement `814` possèdent un snapshot valide dans la fenêtre autorisée, soit une couverture de `94,21 %`. Le seuil exigeait au moins 95 créneaux disponibles sur 96 pour chacune des neuf séries ; seules les trois séries de 2026 le passent.

Les `50` créneaux absents restent explicitement indisponibles. Aucune interpolation, propagation ou substitution par un delta n'a été effectuée.

## Résultats par série

| Date UTC | Instrument | Snapshots natifs | Disponibles | Manquants | Couverture | Retard maximal sélectionné |
|---|---|---:|---:|---:|---:|---:|
| 2023-04-01 | BTC-USDT | 1 440 | 88 | 8 | 91,67 % | 299 ms |
| 2023-04-01 | ETH-USDT | 1 439 | 86 | 10 | 89,58 % | 278 ms |
| 2023-04-01 | SOL-USDT | 1 440 | 94 | 2 | 97,92 % | 491 ms |
| 2024-07-01 | BTC-USDT | 1 440 | 80 | 16 | 83,33 % | 73 ms |
| 2024-07-01 | ETH-USDT | 1 440 | 87 | 9 | 90,63 % | 64 ms |
| 2024-07-01 | SOL-USDT | 1 440 | 91 | 5 | 94,79 % | 450 ms |
| 2026-07-01 | BTC-USDT | 96 | 96 | 0 | 100 % | 9 ms |
| 2026-07-01 | ETH-USDT | 96 | 96 | 0 | 100 % | 9 ms |
| 2026-07-01 | SOL-USDT | 96 | 96 | 0 | 100 % | 9 ms |

La règle choisissait uniquement le premier snapshot situé entre le créneau et une seconde après celui-ci. Les séries à une minute ont des snapshots de part et d'autre de certaines bornes de quart d'heure. Lorsqu'un snapshot tombe juste avant la borne, il est ignoré par ce contrat et le suivant, environ une minute plus tard, est trop éloigné.

## Causalité et données ignorées

- `814` snapshots natifs sont sélectionnés exactement une fois ;
- `8 113` snapshots natifs ne sont pas utilisés ;
- une mutation des valeurs d'un snapshot non sélectionné ne modifie aucune sortie normalisée ;
- le timestamp source et le retard sont conservés sur chaque ligne disponible ;
- les lignes absentes ne contiennent aucune métrique fabriquée ;
- aucun événement `update` n'est présent dans l'entrée ou utilisé.

## Reproductibilité

Artifact final : `crypto-forecast-okx-l2-normalization-v1-14438dc7f0ab2f0c`.

- résultat SHA-256 : `6c087991ce71f57eed6155992b52e3961c04a19403ee54346951b8167578f879` ;
- métriques normalisées SHA-256 : `5a930c3f25339b77cced88a472f13c655e1278fe4e16ec54de63cbd6761388ba` ;
- configuration SHA-256 : `18237728c712241d76a99027d1e417b0482e5c515e8e9e6bc91de5c6151eafdd` ;
- code de normalisation SHA-256 : `1f164826a79cd6721d806a493c1c997915c2712a4c47218b0a646af977ab7806`.

Une seconde exécution a relu le même fichier source épinglé et produit le même identifiant et les mêmes empreintes, octet pour octet.

## Interprétation

Le No-Go ne remet pas en cause la validité des 8 927 snapshots du lot 5E. Il montre que la règle « uniquement après le créneau » perd trop de points lorsque l'horloge native oscille légèrement autour de la borne.

Ce lot a été conçu après la découverte du changement de cadence au lot 5E. Il s'agit d'une étude de faisabilité méthodologique, pas d'une confirmation indépendante ni d'une preuve prédictive.

## Suite recommandée

Le lot 5G a testé, sans nouveau téléchargement, le snapshot valide **le plus proche dans une fenêtre symétrique de ±1 seconde** :

- un snapshot antérieur est déjà connu à l'heure de grille ;
- un snapshot postérieur n'est disponible qu'à son timestamp réel ;
- la disponibilité causale doit donc être `max(timestamp de grille, timestamp source)` ;
- un créneau sans snapshot dans la fenêtre reste manquant ;
- le test de mutation des snapshots non sélectionnés reste obligatoire.

Cette règle a été préenregistrée puis validée sur 863 des 864 créneaux. Le résultat se trouve dans `CRYPTO_LOT5G_L2_SYMMETRIC_ALIGNMENT_RESULT_2026-09-13.md`. Les No-Go 5E et 5F restent conservés tels quels.

Il n'y a rien à tester dans le navigateur : l'application et la production restent inchangées.

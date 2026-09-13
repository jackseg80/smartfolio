# Plan préenregistré — lot 5K, extraction historique L2 progressive

Date de gel : 13 septembre 2026, avant lecture complète des trois archives retenues.

Corrections de transcription avant production d'un artifact final : deux contrôles initiaux se sont arrêtés successivement sur les vérifications ETH puis SOL. Leurs SHA-256 ont ensuite été comparés automatiquement et recopiés à l'identique depuis le manifeste d'acquisition. Aucun seuil ni résultat analytique n'a été modifié.

## Objectif

Vérifier qu'un historique L2 compact peut être construit progressivement sans conserver en mémoire ni extraire sur disque le contenu décompressé des archives OKX. Ce pilote mesure le coût réel de lecture et la réduction de volume, sans télécharger ni supprimer de donnée.

## Entrées figées

- date UTC : `2024-07-01` ;
- instruments spot : BTC-USDT, ETH-USDT et SOL-USDT ;
- trois archives déjà acquises et épinglées par taille et SHA-256 ;
- volume brut compressé total : `544 810 138` octets ;
- résultat source échantillon : `b32045a424853d22de978a8a3302873c74f64ef0d1d91203455853b713a3d6e2` ;
- résultat source alignement : `c864b55a687ff7a2abb020637967af3970ee6eb88a95056b162e3aced12a3f6a`.

## Méthode gelée

1. Vérifier avant lecture la taille et l'empreinte de chaque archive.
2. Lire une seule archive à la fois et un seul membre régulier, sans extraction du membre brut sur disque.
3. Ignorer toutes les actions `update` ; chaque action `snapshot` est validée comme carnet autonome.
4. Pour chaque borne de 15 minutes, conserver le snapshot valide le plus proche dans une fenêtre symétrique de ±1 000 ms, avec préférence pour l'observation antérieure en cas d'égalité.
5. Publier séparément les timestamps de grille, source et disponibilité ; la disponibilité vaut le maximum des timestamps de grille et source.
6. Ne jamais remplir, interpoler ou propager un carnet manquant.
7. Écrire un fichier JSONL gzip déterministe par instrument, puis un résultat et un manifeste atomiques.
8. Mesurer le temps de lecture séparément de l'identité reproductible de l'artifact.

## Critères Go/No-Go

Le pilote est **Go technique** seulement si :

- les trois archives correspondent exactement aux tailles et SHA-256 gelés ;
- les 288 créneaux sont disponibles et contiennent des carnets valides non croisés ;
- les répartitions avant/exact/après et les écarts maximaux reproduisent le lot 5G ;
- le volume compact total ne dépasse pas 64 MiB ;
- les fichiers compacts sont déterministes et un second passage reproduit l'identifiant et les empreintes ;
- les tests prouvent aussi l'ignorance des updates, la préférence antérieure en cas d'égalité, l'absence explicite et les limites de sécurité ;
- les tests ciblés, Ruff et Black passent.

Toute divergence reste un No-Go ; aucun seuil ne sera ajusté après observation.

## Hors périmètre

- nouveau téléchargement ou acquisition massive ;
- suppression ou modification des archives sources ;
- collecte récurrente, tâche Windows, service ou démarrage automatique ;
- modèle, feature prédictive, cible, backtest ou allocation ;
- API, interface, port local ou production ;
- secret, compte, ordre, dérivé ou levier ;
- commit, push ou déploiement.

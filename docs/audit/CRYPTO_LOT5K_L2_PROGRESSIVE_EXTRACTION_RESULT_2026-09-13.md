# Résultat — lot 5K, extraction historique L2 progressive

Date d'exécution : 13 septembre 2026<br>
Décision : **GO_TECHNIQUE POUR L'EXTRACTION PROGRESSIVE**

## Conclusion

Le mécanisme sait transformer, une archive après l'autre, `544 810 138` octets d'archives OKX déjà présentes en `1 076 624` octets de carnets complets utiles. La réduction est de `99,8024 %`, soit un facteur d'environ `506,04`, tout en conservant exactement les 288 snapshots préenregistrés.

Ce résultat valide une méthode de stockage et d'extraction. Il ne démontre aucune capacité prédictive et n'autorise ni téléchargement historique massif, ni collecte récurrente, ni intégration au produit.

## Résultats par instrument

| Instrument | Archive | Messages | Updates ignorées | Snapshots natifs valides | Retenus | Volume compact gzip | Temps final |
|---|---:|---:|---:|---:|---:|---:|---:|
| BTC-USDT | 187 890 212 o | 5 265 650 | 5 264 210 | 1 440/1 440 | 96 | 439 556 o | — |
| ETH-USDT | 169 773 160 o | 4 691 619 | 4 690 179 | 1 440/1 440 | 96 | 389 661 o | — |
| SOL-USDT | 187 146 766 o | 4 901 119 | 4 899 679 | 1 440/1 440 | 96 | 247 407 o | — |
| **Total** | **544 810 138 o** | **14 858 388** | **14 854 068** | **4 320/4 320** | **288** | **1 076 624 o** | **63,62 s** |

La reproduction indépendante corrigée a pris `62,97 s`. Ces temps dépendent de l'état de la machine et sont exclus de l'identité de l'artifact.

## Correction de format découverte au lot 5M

Le premier essai de lecture consommateur du lot 5M a révélé que l'artifact initial 5K concaténait les objets JSON sans séparateur de ligne. Les octets et leurs empreintes étaient reproductibles, mais les fichiers portant l'extension `jsonl.gz` n'étaient pas des JSON Lines valides.

Le producteur ajoute désormais un retour à la ligne après chaque objet, et un test décompresse puis parse réellement toutes les lignes. L'artifact initial `crypto-forecast-okx-l2-progressive-extraction-v1-bbdf89ecd6809b32` reste conservé comme trace historique mais est **remplacé et interdit comme entrée**. Les règles de sélection, les 288 snapshots et leurs valeurs n'ont pas changé.

## Alignement reproduit

| Instrument | Avant la borne | Exactement | Après la borne | Écart absolu maximal | Manquants |
|---|---:|---:|---:|---:|---:|
| BTC-USDT | 16 | 5 | 75 | 73 ms | 0 |
| ETH-USDT | 9 | 3 | 84 | 64 ms | 0 |
| SOL-USDT | 5 | 9 | 82 | 450 ms | 0 |

Ces valeurs correspondent exactement au lot 5G. Les timestamps de grille, source et disponibilité sont tous conservés. Aucun remplissage, interpolation ou état propagé depuis les updates n'est utilisé.

## Reproductibilité et intégrité

- artifact final corrigé : `crypto-forecast-okx-l2-progressive-extraction-v1-8ca057800a4ec6e4` ;
- résultat SHA-256 : `ded85d4b08ec400fe2af1d63b2bb0c0e6a920827de7f5ea8f53a482d0de6ae34` ;
- configuration SHA-256 : `bd1fa48baf88bba01d282155b6dfe2af1340f77ac5bd2b6e990db66d06fcf5c9` ;
- code d'extraction SHA-256 : `108858b86b57609597ec116476e8f05356e4a4a372cc13042c11157324c476c8` ;
- BTC compact SHA-256 : `6306f8df16f264918444e2a0ece64cdc3bf66ebf9cf9942073d04a498a4c9d89` ;
- ETH compact SHA-256 : `9fd0f5f5ab625b466ac3cc52c502cb7bea2005108283ee1a3eda12ec67f110c3` ;
- SOL compact SHA-256 : `2d5f72512691c81773cbe437e3a1ee74ae75030e10e726bd7c84103f046e7265`.

Deux lectures complètes depuis les archives brutes ont produit le même identifiant, le même résultat et les mêmes trois fichiers gzip. Le manifeste refuse désormais un payload dont la taille ou l'empreinte ne correspond pas au résultat annoncé.

## Incidents de préexécution

Deux premiers passages se sont arrêtés avant publication d'un artifact à cause de coquilles de transcription dans les SHA-256 ETH puis SOL. Les valeurs ont été réconciliées automatiquement avec le manifeste d'acquisition existant. Aucun seuil, règle de sélection ou résultat observé n'a été modifié.

## Validation

- 6 tests unitaires ciblés réussis, dont la décompression et le parsing ligne par ligne du payload ;
- mutation des updates sans effet sur les snapshots compacts ;
- préférence pour le snapshot antérieur en cas d'égalité ;
- absence conservée explicitement et décision No-Go associée ;
- refus d'un chemin source hors périmètre, d'un hash source incorrect et d'un payload compact altéré ;
- Ruff conforme ;
- Black conforme ;
- aucun avertissement nouveau, hormis l'avertissement Starlette/httpx déjà extérieur au chantier.

## Garde-fous confirmés

- aucune requête réseau ;
- aucun membre brut extrait sur disque ;
- aucune archive source modifiée ou supprimée ;
- aucun secret, compte ou ordre ;
- aucun modèle entraîné ;
- aucune API, interface, tâche récurrente ou production touchée ;
- aucun commit, push ou déploiement.

## Suite proposée

Le prochain lot raisonnable est une extension compacte aux neuf archives déjà acquises, sans nouveau téléchargement. Elle vérifierait le même format sur les trois dates et les deux cadences historiques avant toute décision séparée concernant une collecte récurrente ou de nouvelles archives.

# Résultat — lot 5L, corpus L2 compact multi-périodes

Date d'exécution : 13 septembre 2026<br>
Décision : **GO_TECHNIQUE POUR LE CORPUS COMPACT**

## Conclusion

Les neuf archives OKX déjà présentes ont été transformées séquentiellement en neuf fichiers JSON Lines compacts et reproductibles. Les `1 181 503 619` octets sources deviennent `2 574 579` octets gzip, soit une réduction de `99,7821 %` et un facteur d'environ `458,91`.

Les `8 927` snapshots natifs sont tous valides et non croisés. Le corpus conserve exactement `863/864` créneaux ; le seul manque reste ETH-USDT à minuit UTC le 1er avril 2023, conformément au plan.

Ce Go valide le format et la méthode de stockage sur trois journées isolées. Le corpus n'est pas un historique continu et ne démontre aucune capacité prédictive.

## Correction de format découverte au lot 5M

Le premier essai de lecture du lot suivant a révélé que le corpus initial concaténait les objets JSON sans retour à la ligne. Ses empreintes étaient cohérentes, mais le contenu ne respectait pas le contrat `jsonl.gz`. Le producteur 5K a été corrigé, testé par décompression et parsing ligne par ligne, puis les neuf archives ont été relues deux fois.

L'artifact initial `crypto-forecast-okx-l2-compact-corpus-v1-e229d0bfb9fef72b` reste conservé comme trace historique mais est **remplacé et interdit comme source**. Le corpus corrigé conserve les mêmes 863 snapshots et le même manque ; seules la délimitation des enregistrements, les tailles gzip et les empreintes changent.

## Résultats par groupe

| Date UTC | Instrument | Natifs valides | Retenus | Manquants | Avant | Exact | Après | Écart maximal | Gzip |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023-04-01 | BTC-USDT | 1 440 | 96 | 0 | 8 | 2 | 86 | 299 ms | 126 209 o |
| 2023-04-01 | ETH-USDT | 1 439 | 95 | 1 | 9 | 5 | 81 | 278 ms | 121 731 o |
| 2023-04-01 | SOL-USDT | 1 440 | 96 | 0 | 2 | 5 | 89 | 491 ms | 107 255 o |
| 2024-07-01 | BTC-USDT | 1 440 | 96 | 0 | 16 | 5 | 75 | 73 ms | 439 556 o |
| 2024-07-01 | ETH-USDT | 1 440 | 96 | 0 | 9 | 3 | 84 | 64 ms | 389 661 o |
| 2024-07-01 | SOL-USDT | 1 440 | 96 | 0 | 5 | 9 | 82 | 450 ms | 247 407 o |
| 2026-07-01 | BTC-USDT | 96 | 96 | 0 | 0 | 8 | 88 | 9 ms | 436 592 o |
| 2026-07-01 | ETH-USDT | 96 | 96 | 0 | 0 | 3 | 93 | 9 ms | 412 429 o |
| 2026-07-01 | SOL-USDT | 96 | 96 | 0 | 0 | 16 | 80 | 9 ms | 293 739 o |

Les neuf répartitions avant/exact/après et leurs écarts maximaux reproduisent exactement le lot 5G. Les `34 798 969` updates sont ignorées par construction.

## Reproductibilité

- artifact final corrigé : `crypto-forecast-okx-l2-compact-corpus-v1-1df479332e50e3d1` ;
- résultat SHA-256 : `bb597dbc71f9b001610c44c8f0772a9f2c46ecee74509974ccd3bb8203105e59` ;
- configuration SHA-256 : `ac9f9393f51980069d4c15bd4c86d33fd9b12050ab3e11f60f71700785df7e5f` ;
- code corpus SHA-256 : `89f5b48dca72e7736b567d387731f8df3d39c537563ed24ee22e57aa70565431` ;
- code extraction 5K SHA-256 : `108858b86b57609597ec116476e8f05356e4a4a372cc13042c11157324c476c8`.

Deux lectures complètes et indépendantes des neuf archives ont produit le même identifiant, le même résultat et les mêmes neuf empreintes de données. Le premier passage corrigé a pris `139,29 s` et le second `136,95 s`. Ces durées sont exclues de l'identité.

## Validation

- 10 tests ciblés 5K–5L réussis ;
- 95 tests de recherche crypto réussis au total après l'ajout des tests 5M et de la régression JSONL ;
- configuration réconciliée automatiquement avec le manifeste d'acquisition ;
- chemins source et destination bornés ;
- absences limitées à des bornes UTC uniques et préenregistrées ;
- payloads contrôlés par taille et SHA-256 avant publication ;
- Ruff conforme ;
- Black conforme ;
- seul l'avertissement Starlette/httpx déjà extérieur au chantier subsiste.

## Garde-fous confirmés

- aucune requête réseau ou nouvelle archive ;
- aucun membre brut extrait sur disque ;
- aucune source modifiée ou supprimée ;
- aucun remplissage, interpolation ou update utilisé ;
- aucun secret, compte, ordre ou modèle ;
- aucune API, interface, tâche récurrente ou production touchée ;
- aucun commit, push ou déploiement.

## Suite proposée

Le corpus permet maintenant de préenregistrer un contrat de features L2 hors ligne, puis de produire une table causale depuis les 863 carnets sans entraîner de modèle. Toute extension de l'historique ou activation récurrente restera une décision séparée.

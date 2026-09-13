# Résultat lot 5A — cohérence quotidienne Binance–OKX

Date : 13 septembre 2026<br>
Statut : contrôle de données réussi ; aucune sélection de modèle.<br>
Périmètre : dix paires spot USDT, journées UTC closes jusqu'au 11 septembre 2026.

## Conclusion

OKX est recevable comme seconde source quotidienne pour la recherche SmartFolio. Les dix actifs passent les quatre critères de cohérence prix préenregistrés : couverture, corrélation des rendements, écart médian et 95e percentile des clôtures.

Les clôtures Binance et OKX sont très proches : l'écart absolu médian varie d'environ 0,7 à 2,6 points de base selon l'actif. La corrélation des rendements quotidiens est comprise entre 0,9997 et 0,99999.

Les volumes évoluent aussi dans le même sens, mais avec une stabilité variable : les corrélations des variations à 7 jours vont de 0,6155 à 0,9359, et celles à 30 jours de 0,8349 à 0,9760. Ces mesures restent descriptives ; aucun seuil de sélection n'avait été fixé pour le volume.

Ce résultat autorise une étude de robustesse multi-plateforme. Il ne valide pas un modèle, une allocation ou une nouvelle feature.

## Artifact OKX

- Artifact : `crypto-forecast-okx-history-v1-61c1d94d41ced680`
- Fournisseur : `okx_spot_public_market_data`
- Accès : endpoints publics, lecture seule, sans identifiant ni clé
- Taille : environ 2,46 Mo
- Jours manquants entre première et dernière observation : zéro
- BTC, ETH, SOL, ADA, XRP, LINK, LTC, BCH et DOT : 2 052 observations du 29 janvier 2021 au 11 septembre 2026
- BNB : 1 361 observations du 21 décembre 2022 au 11 septembre 2026
- SHA-256 configuration : `4b6d52a3f641da30ebb6f9795c0059717b989e33ab7dc6978d7ca5face93d248`
- SHA-256 collecteur : `3eac2f1b779e7fc50a6ee945752164a84fdbade68c758d339d52abb6a48c8414`

Une seconde acquisition isolée a reproduit le même identifiant et les mêmes hashes de prix/OHLCV pour les dix actifs.

## Critères préenregistrés

Une série devait respecter simultanément :

- au moins 1 000 journées communes ;
- corrélation des rendements quotidiens d'au moins 0,995 ;
- écart absolu médian de clôture d'au plus 50 points de base ;
- 95e percentile de l'écart absolu d'au plus 300 points de base.

Aucun actif n'a été retiré après observation des résultats.

## Résultats par actif

| Actif | Jours communs | Corrélation rendement | Écart médian clôture | P95 clôture | Maximum | Date du maximum | Corr. volume 7 j | Corr. volume 30 j | Statut |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| ADA | 2 052 | 0,999944 | 2,106 pb | 7,580 pb | 32,083 pb | 2021-02-08 | 0,9240 | 0,9481 | Pass |
| BCH | 2 052 | 0,999927 | 2,623 pb | 8,652 pb | 20,630 pb | 2023-08-17 | 0,6155 | 0,9198 | Pass |
| BNB | 1 361 | 0,999861 | 1,961 pb | 6,822 pb | 23,708 pb | 2024-03-15 | 0,8268 | 0,8401 | Pass |
| BTC | 2 052 | 0,999982 | 0,673 pb | 2,436 pb | 17,936 pb | 2021-05-12 | 0,7419 | 0,8349 | Pass |
| DOT | 2 052 | 0,999911 | 2,351 pb | 9,450 pb | 61,100 pb | 2025-10-10 | 0,9196 | 0,9314 | Pass |
| ETH | 2 052 | 0,999985 | 0,755 pb | 3,052 pb | 18,879 pb | 2025-10-10 | 0,9059 | 0,9252 | Pass |
| LINK | 2 052 | 0,999934 | 2,367 pb | 8,185 pb | 46,794 pb | 2025-10-10 | 0,9042 | 0,8743 | Pass |
| LTC | 2 052 | 0,999944 | 1,812 pb | 6,307 pb | 59,278 pb | 2021-05-12 | 0,8958 | 0,9137 | Pass |
| SOL | 2 052 | 0,999706 | 1,299 pb | 7,563 pb | 338,340 pb | 2021-02-24 | 0,8810 | 0,9036 | Pass |
| XRP | 2 052 | 0,999978 | 1,348 pb | 5,058 pb | 31,640 pb | 2025-10-10 | 0,9359 | 0,9760 | Pass |

## Valeurs extrêmes

SOL présente un écart isolé de 338,34 points de base le 24 février 2021. Cette valeur n'est ni supprimée ni winsorisée. Le 95e percentile de SOL reste à 7,56 points de base, ce qui respecte le critère préenregistré.

Plusieurs actifs présentent leur maximum le 10 octobre 2025. Le résultat indique une divergence de clôture ce jour-là ; il n'en déduit pas automatiquement une erreur de fournisseur ou une opportunité exploitable.

## Reproductibilité

- Artifact de validation : `crypto-forecast-cross-exchange-validation-v1-3b425b8039391fbf`
- SHA-256 identité stable : `3b425b8039391fbf3c5d95cded105887bbae81d2cfbaa78d48306221f9ab3f6c`
- SHA-256 fichier quotidien aligné : `50e666b4ed281efab59466c42bc990f9c58983038ef1035328a496e0f3574be4`
- SHA-256 configuration : `06bd4715437047cced00b67d114ae2286d43cbc814f5a4fc98ab4e281e1c429d`
- SHA-256 moteur de validation : `2e004153446d11054bd427532a02126104b4eb934b515a2b66dde908b4901e8a`

La première implémentation incluait par erreur l'heure de génération du manifeste dans l'identité de validation. Les données alignées étaient déjà identiques, mais l'identifiant variait. Le calcul a été corrigé et couvert par un test : l'identité dépend maintenant du contenu stable du manifeste, tandis qu'un changement métier reste détecté.

Deux exécutions basées sur les deux acquisitions OKX ont ensuite produit exactement le même identifiant, la même identité stable et le même hash quotidien. Le CSV final représente les dix premiers rendements indisponibles par des cellules vides et ne contient aucune valeur `NaN` ou infinie.

## Sécurité et portée

- `model_selection_performed=false`
- `production_configuration_changed=false`
- `real_orders_created=false`
- aucune API ou interface SmartFolio modifiée
- aucun accès au portefeuille réel
- aucune donnée L2 téléchargée

## Décision de reprise

Le lot 5A quotidien est validé comme couche de contrôle. La suite utile n'est pas d'ajouter mécaniquement les prix OKX au modèle : ils contiennent presque la même information directionnelle que Binance.

La prochaine étape à valeur ajoutée est le lot 5B, un pilote très borné de données L2 OKX afin d'évaluer spread, profondeur et déséquilibre de carnet. Son format, sa taille maximale, sa période et ses règles de reconstruction doivent être gelés avant téléchargement.

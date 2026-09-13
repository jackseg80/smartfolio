# Lot 5Q — Résultat des features causales de funding

Date : 13 septembre 2026<br>
Plan gelé : [CRYPTO_LOT5Q_BINANCE_FUNDING_FEATURES_PLAN_2026-09-13.md](./CRYPTO_LOT5Q_BINANCE_FUNDING_FEATURES_PLAN_2026-09-13.md)

## Verdict

**Go causal.** Les 16 variables quotidiennes de funding respectent l'instant de décision gelé et les trois mutations du futur donnent zéro divergence dans toutes les lignes protégées.

Ce verdict autorise uniquement la comparaison prédictive préenregistrée du lot 5R. Il ne démontre pas que le funding améliore un modèle ou une allocation.

## Table produite

| Élément | Résultat |
|---|---:|
| Instruments | 7 |
| Dates par instrument | 1 553 |
| Lignes totales | 10 871 |
| Features | 16 |
| Warm-up explicite | 29 jours par instrument |
| Lignes éligibles après warm-up | 10 668 |
| Valeurs non finies dans les lignes éligibles | 0 |
| Cibles ou rendements futurs | 0 |

Chaque ligne du jour `d` ne contient que les règlements dont l'horodatage est strictement antérieur au lendemain à 00:00 UTC. Les fenêtres de 3, 7 et 30 jours se terminent au jour courant. Les fenêtres incomplètes restent vides ; aucun remplissage ou calcul futur n'est utilisé.

## Mutation du futur

| Coupure | Lignes protégées comparées | Divergences protégées | Lignes futures effectivement changées |
|---|---:|---:|---:|
| 31 décembre 2023 | 4 053 | 0 | 6 818 |
| 31 décembre 2024 | 6 615 | 0 | 4 256 |
| 31 décembre 2025 | 9 170 | 0 | 1 701 |

Dans chaque copie contrefactuelle, tous les taux et intervalles postérieurs à la coupure sont modifiés et une observation future supplémentaire est ajoutée par jour. Le fait que les lignes futures changent confirme que le test agit réellement ; l'absence totale de divergence avant ou à la coupure confirme l'invariance recherchée.

## Provenance et reproductibilité

- Entrée : `crypto-forecast-binance-funding-history-v1-6288d6a1dca91185`.
- Artifact 5Q : `crypto-forecast-binance-funding-features-v1-f32e0489dbb258fd`.
- Résultat SHA-256 : `2d2efd9272931e7aadfa4217a864e4b7dbb701b764bf464524a202d6f1e2b821`.
- Table SHA-256 : `fc1b78a7ba21919f676d82e952c6dc3dec2566d4e00d23dadfddeee6ce19d313`.
- Manifeste principal SHA-256 : `41784882a5f30464eb97f71655c4c82411e616906056f7b0142b6b2b45a5a8fe`.
- Configuration SHA-256 : `55a06cdde74e27007757dde50c4fe81317ed02373a5938b0bd143fcbe6b8cfa8`.
- Chaîne de code SHA-256 : `b8521da2dc179e14c1428276dc8a528f39afd968b3c53fdc93564a92485412b3`.
- Deux constructions indépendantes donnent le même identifiant, le même résultat et la même table de 3 200 924 octets.

## Validation technique

- 11 tests ciblés : réussis.
- Régression de la recherche crypto : 121 tests réussis en 6,39 s.
- Ruff : conforme.
- Black : conforme.
- Avertissement restant : dépréciation Starlette/httpx déjà extérieure à ce chantier.

La régression utilise `--no-cov` en raison de la violation d'accès Windows déjà isolée dans le générateur XML de `coverage`, après réussite des assertions. Ce problème d'outil ne bloque pas les tests applicatifs.

## Limites

- Les features sont causalement construites, mais leur utilité prédictive reste inconnue.
- Le funding de contrats perpétuels est utilisé comme information ; aucun levier, short ou ordre n'est autorisé.
- Aucune normalisation n'est faite ici. Dans le lot 5R, elle devra être apprise sur l'entraînement uniquement.
- Aucun modèle, allocation, interface, réseau, service permanent ou changement de production n'a été utilisé.

## Suite autorisée

Le lot 5R doit figer avant entraînement :

1. les mêmes frontières chronologiques que la référence, avec purge de 30 jours ;
2. un modèle de référence sans funding et le même modèle avec funding ;
3. le prétraitement appris sur l'entraînement uniquement ;
4. des critères d'amélioration sur validation et confirmation finale qui ne permettent aucun réglage rétrospectif ;
5. une décision Go/No-Go distincte pour les horizons 7 et 30 jours.

Le test économique 5S et une démonstration locale sur le port 8082 restent interdits tant que 5R n'a pas montré une amélioration hors échantillon robuste.

# Rapport lot 4 — traduction économique des prévisions crypto

Date d'exécution : 12 septembre 2026<br>
Périmètre : simulation hors ligne et conditionnelle, sans portefeuille historique daté, ordre réel ni modification de production.

## Conclusion

Le lot 4 ne démontre aucun avantage économique des prévisions actuelles.

Les variantes prévisionnelles s'abstiennent sur les 1 098 jours de développement communs et les 342 jours de confirmation finale communs. Les régressions BTC et actif à 30 jours sélectionnées valent zéro : aucun rendement estimé ne dépasse le coût aller-retour de 0,60 %. Les probabilités seules ne sont volontairement pas autorisées à déclencher une allocation.

L'abstention évite les pertes de la période finale défavorable, mais ce n'est pas une preuve prédictive : elle reste aussi en cash pendant toute la période de développement positive. Le système ne distingue donc pas économiquement les deux périodes.

## Protocole économique

- Signaux observés chaque jour ; revue tous les sept jours calendaires.
- Exécution simulée à la clôture suivante.
- Frais et glissement : 0,30 % par montant négocié, puis scénario doublé à 0,60 %.
- Six politiques réactives : exposition 20/60 % ou 20/80 %, en BTC/ETH, avec plafond altcoins 30 % du portefeuille ou 60 % de la poche risquée.
- Socle BTC/ETH 60/40 ; rotation inverse de volatilité 60 jours ; cap 10 % par altcoin.
- Référence marché BTC/SMA200 avec confirmations 5/2 ; rotation SMA90 et relatif BTC 90 jours avec confirmations 5/2.
- Repères : cash, BTC, BTC/ETH 60/40 et poches fixes 20 %, 60 % ou 80 % avec cash sans rendement.
- Quantités et cash dérivent entre opérations ; le turnover n'inclut pas deux fois la jambe cash.

## Résultats aux coûts normaux

| Période | Variante | Rendement net | Drawdown maximal | Sharpe | Turnover | Frais cumulés |
|---|---|---:|---:|---:|---:|---:|
| Développement 2021-07-02 → 2024-07-03 | Cash | 0,00 % | 0,00 % | n/a | 0,00 | 0,00 % |
| Développement | BTC conservé | 64,06 % | -76,63 % | 0,57 | 1,00 | 0,30 % |
| Développement | Poche fixe 20 % BTC/ETH | 10,67 % | -25,86 % | 0,32 | 0,20 | 0,06 % |
| Développement | Réactive 20/60 BTC/ETH | 64,92 % | -29,11 % | 0,80 | 5,01 | 1,71 % |
| Développement | Réactive 20/80, alt 60 % poche risquée | 94,29 % | -35,43 % | 0,85 | 13,12 | 5,27 % |
| Développement | Prévisionnelle, toutes politiques | 0,00 % | 0,00 % | n/a | 0,00 | 0,00 % |
| Confirmation 2025-09-05 → 2026-08-12 | Cash | 0,00 % | 0,00 % | n/a | 0,00 | 0,00 % |
| Confirmation | BTC conservé | -42,55 % | -52,97 % | -1,15 | 1,00 | 0,30 % |
| Confirmation | Poche fixe 20 % BTC/ETH | -9,61 % | -12,72 % | -1,44 | 0,20 | 0,06 % |
| Confirmation | Réactive 20/60 BTC/ETH | -18,48 % | -22,66 % | -1,40 | 2,19 | 0,62 % |
| Confirmation | Réactive 20/80, alt 60 % poche risquée | -22,95 % | -27,43 % | -1,29 | 4,64 | 1,28 % |
| Confirmation | Prévisionnelle, toutes politiques | 0,00 % | 0,00 % | n/a | 0,00 | 0,00 % |

Les règles réactives réduisent fortement le drawdown face à BTC pendant le développement, mais perdent encore sur la confirmation et sont moins défensives que la poche fixe 20 %. Avec coûts doublés, la variante 20/80 avec altcoins passe de 94,29 % à 86,78 % sur le développement et de -22,95 % à -24,02 % sur la confirmation.

## Diagnostic des seuils

| Bloc | Jours | Marché favorable | Marché défavorable | Marché neutre | Rotations franchissant probabilité + rendement | Défensif 7 j |
|---|---:|---:|---:|---:|---:|---:|
| Développement commun | 1 098 | 0 | 0 | 1 098 | 0 sur 10 980 | 0 |
| Confirmation commune | 342 | 0 | 0 | 342 | 0 sur 3 420 | 0 |

Les probabilités de rotation atteignent 0,7574 puis 0,7863, mais le rendement relatif sélectionné reste zéro. Le verrou économique fonctionne : une probabilité élevée ne suffit pas à financer une rotation.

## Artifact final

- Artifact : `crypto-forecast-allocation-backtest-v1-a450458ceb1c802b`
- SHA-256 dataset : `9e1d0d9ad6d418e6ce03448bd82b416e19d5ab953a1aa83ea02bc37c1a76d537`
- SHA-256 résultats prévisionnels : `e564b12d474c7f350881976bc65a78bc8a4fbc5bb04e5bfaaadc74b95bedd412`
- SHA-256 prédictions : `2701a026d80ecd22fc3e70eee48fd94107621bfee7bee8271a1ca22bb312cd89`
- SHA-256 résultats allocation : `d59cd2c549d6dda22b6b0aa9cd5318e6a0431912c39e9007372f4ffddc228d3a`
- SHA-256 quotidien : `06c5f5da68dc7448802a377325c80835ec7a68cd757e3562b44a3ed2305c2d7a`

Une seconde exécution isolée a reproduit exactement l'identifiant et les hashes.

## Limites et décision

Chaque bloc démarre hypothétiquement à 100 % cash : sans composition réelle datée, le coût de transition n'est pas mesurable. La confirmation commune ne contient que 342 jours à cause de l'intersection causale des horizons 7 et 30 jours. Les résultats restent conditionnels à Binance/USDT, à la taxonomie actuelle et à un cash sans rendement ni risque stablecoin simulé.

Ne pas brancher ces modèles sur Rebalance, Execution ou une stratégie active. Une interface d'allocation prévisionnelle serait trompeuse à ce stade : sa seule sortie honnête serait « insuffisamment démontré / abstention ». Le prochain essai doit améliorer et valider l'estimation du rendement relatif avec une hypothèse bornée et préenregistrée.

## Mise à jour lot 3b

L'hypothèse suivante a été préenregistrée puis testée dans `CRYPTO_LOT3B_HURDLE_RESULT_2026-09-12.md`. Le rendement hurdle active certaines rotations, mais ne crée pas d'avantage économique robuste après coûts ; l'hypothèse est rejetée et aucun branchement produit n'est autorisé.

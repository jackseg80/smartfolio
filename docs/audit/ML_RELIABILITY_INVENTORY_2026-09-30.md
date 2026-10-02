# Inventaire ML — 30 septembre 2026

## Périmètre et sources de vérité

Le service partagé services/ml/reliability.py calcule les résultats de lecture.
services/ml/risk_evaluation.py utilise les mêmes features, horizons et estimateurs
pour la comparaison chronologique. api/schemas/ml_contract.py décrit les valeurs,
les états, la validation et la provenance. Aucun GET ne lance un entraînement.

| Capacité | Calcul et fournisseur | Artefact / entraînement | API et consommateurs | Conclusion |
|---|---|---|---|---|
| Volatilité crypto | Clôtures quotidiennes Binance Spot USDT, rendements logarithmiques, annualisation 365 | JSON scellé dans models/validated_risk; comparaison explicite persistence/EWMA/Ridge/LSTM | /api/ml/overview, /unified/predict, adaptateurs legacy; AI Dashboard, Analytics | Prévisions 7/30 jours seulement après confirmation |
| Volatilité bourse | Yahoo Finance/yfinance, auto_adjust=True, calendriers de place, annualisation 252 | JSON scellé; transformations entraînement seul; calendrier vérifié | /api/ml/overview, /api/ml/bourse/forecast; AI Dashboard, Bourse Analytics | Prévision validée rétrospectivement ou Unavailable |
| Régime économique | Drawdown sur 200 clôtures et distance à MA200; critères explicites | Pas de réseau requis; règles descriptives | Overview, Analytics, Market Regimes, Dashboard, bourse | Diagnostic, aucune probabilité de hausse |
| États HMM crypto | Caractéristiques et moyennes apprises; probabilités des états | Artefact legacy sécurisé, features compatibles; GET sans réentraînement | /api/ml/crypto/regime; Market Regimes, graphiques BTC/ETH | Experimental; A–D sans correspondance économique vérifiée |
| Histoire des régimes | Reconstruction avec modèle ajusté plus tard | Historique rétrospectif explicitement identifié | /api/ml/crypto/regime/history; graphiques régimes | Ne prouve pas une décision possible en temps réel |
| Corrélation historique | Pearson, 90 rendements quotidiens communs; sources vérifiées | Aucun entraînement | Overview et diagnostic bourse | Descriptive; Partial si actifs absents/périmés |
| Transformer corrélation | Modèle legacy de corrélation future | Aucun comparatif indépendant confirmé | Registre et adaptateurs de corrélation | Experimental; prévisions masquées |
| Cycle Bitcoin | Heuristique de phase, cycle et distances; composante de scoring existante | Aucun modèle prédictif validé | Cycle Analysis, Risk Dashboard, Analytics | Diagnostic heuristique; allocation existante préservée |
| Fear & Greed | Observation Alternative.me avec date et fournisseur | Aucun modèle ML | Overview, Analytics, AI Dashboard | Indicateur externe; absent si fournisseur indisponible |
| Sentiment ML par actif | Aucun adaptateur indépendant validé connecté | Legacy non validé | Endpoint legacy sentiment, registre | Unavailable; aucune valeur neutre de substitution |
| Signaux techniques bourse | Accord de règles RSI/MACD/Bollinger, observations requises | Pas de probabilité calibrée | /api/ml/bourse/signals, Bourse Recommendations | Descriptive; accord de règles ≠ probabilité |
| Direction crypto / funding / hurdle | Recherche causale existante du 13 septembre | Expériences closes; aucune relance | config/ml_capability_registry.json et recherche canonique | Rejected; aucune intégration aux allocations |
| Alertes ML | Prédicteur désactivé | Réactivation hors chantier | Alerts, AI Dashboard, monitoring | Rejected / disabled |
| Modèles legacy | Fichiers .pth/.keras/.pkl, métadonnées et schémas vérifiés | Présence, chargement et inférence distingués; chargement sécurisé | /api/ml/status, monitoring, registry | Experimental tant que la confirmation manque |
| Santé et performances | Comptes observés et métriques des artefacts validés | Santé, confiance et performances inconnues restent nulles | /monitoring/health, /metrics/{model}; Monitoring, Model Status | Aucune santé inventée à partir des fichiers |
| Gouvernance / allocation | Chaîne de gouvernance existante | governance_eligible=False pour les nouveaux artefacts | Allocation / execution existantes | Aucune nouvelle intégration ni automatisme financier |

## Contrat et identité

Chaque résultat indique actif, marché, diagnostic/forecast, cible, horizon, unité,
valeur éventuellement nulle, état, raison, dates et validation. Provenance :
fournisseur, dataset, code, empreinte d'artefact et période d'entraînement.
Les diagnostics sans apprentissage n'ont pas de période d'entraînement.
Les probabilités non évaluées sont absentes. Seuls les intervalles avec couverture
de confirmation entre 85 % et 95 % sont publiés.

Les requêtes personnelles portent l'utilisateur authentifié, la source et le fichier
CSV sélectionné. Le service refuse un changement silencieux de source ou de fichier.
Les mappings Saxo conservent la place; un ETF Londres n'est pas remplacé par son
homonyme américain. Les tokens enveloppés et symboles suffixés ne sont pas assimilés
à BTC, ETH ou SOL. Aucun cache global de résultat personnel n'est ajouté.

Le dashboard ouvre les actifs du compte; les benchmarks sont un mode explicite.
Limites d'affichage : 25, 50 ou jusqu'à 250 actifs, avec le nombre d'actifs omis.
Les événements de source/fichier invalident l'identité de la requête en cours.
La copie minimale autorisée sert uniquement au preview et est refusée en production.

Les anciennes alternatives de cycle à performances fixes et les comparaisons de prix illustratives sans reçu fournisseur sont indisponibles. Seul l'ajustement heuristique rétrospectif à des événements curated reste affiché, explicitement sans preuve de prévision.

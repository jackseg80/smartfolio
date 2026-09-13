# Plan préenregistré — lot 5E, échantillon snapshot-only multi-actifs

Date de gel : 13 septembre 2026, avant tout téléchargement des neuf archives candidates.

## Objectif

Télécharger puis contrôler l'échantillon L2 borné autorisé après le lot 5D : BTC-USDT, ETH-USDT et SOL-USDT sur trois dates UTC éloignées. L'analyse doit utiliser uniquement les snapshots complets et ignorer toutes les mises à jour intermédiaires.

Ce lot mesure la stabilité technique inter-actifs et inter-périodes. Il ne construit aucun modèle prédictif.

## Entrée figée

- artifact de couverture : `crypto-forecast-okx-l2-coverage-v1-9b584745ea2d87c6` ;
- métadonnées SHA-256 : `e5d9d80b5d50657735a5b229b3d15c4f6be67a1955906c44d5f25fd043210d2d` ;
- instruments : `BTC-USDT`, `ETH-USDT`, `SOL-USDT` ;
- dates UTC : `2023-04-01`, `2024-07-01`, `2026-07-01` ;
- archives attendues : `9` ;
- taille annoncée cumulée : `1 126,78 MiB` ;
- source : neuf URL HTTPS `static.okx.com` déjà validées par le lot 5D.

Aucun autre instrument, date ou fichier de remplacement n'est permis.

## Acquisition bornée

1. Vérifier l'empreinte des métadonnées avant toute requête d'archive.
2. Vérifier qu'après le volume annoncé il restera au moins `5 GiB` libres.
3. Télécharger séquentiellement, un fichier à la fois, dans un fichier temporaire local.
4. Refuser un fichier de plus de `400 MiB` ou un cumul supérieur à `2 000 MiB`.
5. Exiger que la taille réelle reste à `128 KiB` au plus de la taille annoncée, arrondie au centième de MiB.
6. Calculer SHA-256 pendant l'écriture et renommer le fichier temporaire seulement après succès.
7. N'utiliser aucun cookie, compte, secret ou en-tête d'authentification.

Une tentative temporaire incomplète peut être nettoyée automatiquement avant une nouvelle tentative. Un fichier achevé n'est jamais écrasé silencieusement : il est relu, re-haché et doit encore respecter la métadonnée gelée.

## Contrôle snapshot-only

Pour chacune des neuf archives :

- exactement un membre régulier, lu en flux sans extraction durable ;
- membre non compressé inférieur ou égal à `4 GiB` ;
- au plus `25 000 000` enregistrements et `8 MiB` par ligne JSON ;
- instrument et journée UTC strictement conformes ;
- timestamps non décroissants ;
- exactement `96` snapshots ;
- premier snapshot à minuit UTC ;
- cadence de `900 000 ms ± 1 000 ms` et couverture jusqu'au dernier quart d'heure ;
- chaque snapshot contient de 1 à 400 niveaux uniques et non nuls par côté ;
- carnet non vide, non croisé et métriques finies ;
- les actions `update` sont comptées mais ne contribuent à aucune feature.

Les mesures descriptives gelées sont le spread, les profondeurs bid/ask et le déséquilibre à 10, 25 et 50 points de base.

## Critères Go/No-Go

Le lot est **Go technique** seulement si :

- les `9/9` archives sont acquises et hachées ;
- les `9/9` contrôles individuels sont `GO_TECHNICAL` ;
- les `864/864` snapshots attendus sont valides ;
- aucune mise à jour n'entre dans les features ;
- une seconde analyse des mêmes archives vérifiées, sans retéléchargement, produit le même identifiant et les mêmes empreintes.

Un seul échec conserve les preuves disponibles mais rend la décision globale `NO_GO_TECHNICAL`. Les seuils ne seront pas assouplis après observation.

## Rétention et reproductibilité

- Les neuf archives validées restent sous `outputs/crypto-forecast-l2-sample/raw/` jusqu'à une autorisation explicite de nettoyage.
- Le contenu brut n'est jamais extrait sur disque.
- La reproduction relit les mêmes archives épinglées par SHA-256 ; elle ne sollicite pas de nouveau le fournisseur.
- Les artifacts normalisés excluent les timestamps de génération de leur identité.

## Livrables prévus

1. `config/crypto_forecast_okx_l2_sample.json` — contrat gelé.
2. `services/forecasting/okx_l2_sample.py` — acquisition, validation et agrégation.
3. `scripts/run_crypto_forecast_l2_sample.py` — exécution initiale et reproduction locale.
4. `tests/unit/test_crypto_forecast_okx_l2_sample.py` — métadonnées, limites et reproductibilité.
5. `docs/audit/CRYPTO_LOT5E_L2_SAMPLE_RESULT_2026-09-13.md` — résultats et limites.

## Hors périmètre

- deltas reconstruits ou continuité de séquence supposée ;
- cible à 7/30 jours, modèle, sélection de feature ou backtest ;
- API, interface, port local, production 8080 ou réglage utilisateur ;
- ordre, levier, dérivé, compte ou clé exchange ;
- suppression des archives validées ;
- commit, push ou déploiement.

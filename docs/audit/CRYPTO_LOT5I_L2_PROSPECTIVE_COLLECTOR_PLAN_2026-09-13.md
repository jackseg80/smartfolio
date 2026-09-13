# Plan préenregistré — lot 5I, collecteur prospectif L2 robuste

Date de gel : 13 septembre 2026, avant toute nouvelle requête publique du lot.

## Objectif

Construire et tester le noyau local d'une future collecte prospective OKX à 15 minutes pour BTC-USDT, ETH-USDT et SOL-USDT. Le lot doit prouver la sûreté du stockage et de la reprise, pas démarrer une collecte permanente.

## Entrée et cadence figées

- source de faisabilité : `crypto-forecast-okx-l2-collection-feasibility-v1-7a415096e211446f` ;
- résultat source SHA-256 : `ce188e14463d58bb8d44990cc2ca1fce0bdc99eafe1b66cc1f653ec528f50b0b` ;
- endpoint public : `GET /api/v5/market/books`, `sz=400` ;
- trois instruments spot ;
- grille UTC de 15 minutes, soit 96 créneaux et 288 captures attendues par jour ;
- tolérance opérationnelle prévue : au plus 120 secondes après la borne ;
- pilote ponctuel : retard maximal de 15 minutes, car aucun service n'est laissé en attente jusqu'à la prochaine borne.

Le retard plus large du pilote ne modifie pas la future tolérance opérationnelle. Il valide uniquement la chaîne requête-écriture-relecture.

## Contrat de stockage

1. Chaque capture possède une clé unique `(jour UTC, borne UTC, instrument)`.
2. Le payload et ses métadonnées sont sérialisés de manière canonique puis compressés en gzip déterministe.
3. L'écriture se fait dans le répertoire final avec un suffixe temporaire, synchronisation disque, puis remplacement atomique.
4. Une capture déjà présente avec la même empreinte est idempotente.
5. Une capture différente sur la même clé est un conflit et ne remplace jamais l'original.
6. Les fichiers temporaires orphelins sont signalés mais jamais supprimés automatiquement.
7. Le manifeste journalier est reconstructible depuis les captures présentes.
8. Les créneaux dus mais absents sont listés ; aucun remplissage ni interpolation.
9. Une journée ne peut être déclarée complète que si ses 288 clés attendues sont présentes et valides.

## Critères Go/No-Go

Le lot est **Go technique du collecteur** uniquement si :

- l'écriture atomique produit une capture relisible et conforme ;
- la répétition identique ne crée aucun second fichier ;
- un doublon différent est refusé sans modifier le premier fichier ;
- une reprise reconstruit le même manifeste ;
- un fichier temporaire orphelin est signalé sans suppression ;
- les absences dues sont explicites et les créneaux futurs ne sont pas marqués manquants ;
- une mutation postérieure ne peut modifier une capture antérieure ;
- le pilote public écrit exactement trois captures valides pour une seule borne ;
- deux analyses du répertoire pilote produisent les mêmes empreintes ;
- les tests ciblés, Ruff et Black passent.

## Livrables prévus

1. `config/crypto_forecast_okx_l2_prospective_collector.json` — contrat gelé.
2. `services/forecasting/okx_l2_prospective_collector.py` — stockage, validation et reprise.
3. `scripts/run_crypto_forecast_l2_prospective_collector.py` — pilote ponctuel ou relecture locale.
4. `tests/unit/test_crypto_forecast_okx_l2_prospective_collector.py` — cas nominaux et pannes.
5. `docs/audit/CRYPTO_LOT5I_L2_PROSPECTIVE_COLLECTOR_RESULT_2026-09-13.md` — résultat final.

## Hors périmètre

- service Windows, tâche planifiée, cron ou processus longue durée ;
- attente active de la prochaine borne ;
- nouvelle archive historique ;
- modèle, feature, cible, backtest, allocation ou seuil de trading ;
- compte, secret, ordre, dérivé ou levier ;
- API SmartFolio, interface, port local ou production ;
- suppression d'artifact ;
- commit, push ou déploiement.

# Résultat — lot 5H, faisabilité d'un historique L2 continu

Date : 13 septembre 2026<br>
Périmètre : projection de 420 jours et trois réponses publiques OKX. Aucun téléchargement d'archive, compte, ordre, modèle, service continu ou environnement SmartFolio.

## Décision

La voie des archives historiques complètes est **`NO_GO_LOCAL_ARCHIVE_HISTORY`**. La voie de collecte prospective des seuls snapshots nécessaires est **`GO_PROSPECTIVE_COLLECTION_FEASIBILITY`** selon le contrat préenregistré.

Cela signifie qu'il est matériellement raisonnable de construire l'historique à partir de maintenant, à raison d'un carnet toutes les 15 minutes pour BTC, ETH et SOL. Cela ne signifie pas encore que cette donnée améliore une prévision, ni qu'un collecteur permanent a été autorisé ou démarré.

## Fenêtre minimale

Le plancher est de `420` jours continus : 180 jours d'entraînement, 60 de validation, 30 de calibration et 60 de test final, séparés par trois purges de 30 jours. Les purges couvrent la cible maximale à 30 jours.

Cette durée est un minimum d'ingénierie pour éviter le chevauchement des cibles entre les phases. Elle ne garantit pas à elle seule une puissance statistique suffisante.

## Archives historiques : No-Go local

La projection utilise les tailles journalières déjà gelées au lot 5D :

| Hypothèse | Volume sur 420 jours |
|---|---:|
| Journée médiane | 191,90 GiB |
| Journée P95 | 279,80 GiB |
| Journée maximale observée | 300,29 GiB |

Les trois scénarios dépassent largement le plafond local préenregistré de 20 GiB et l'espace libre observé sur `D:` pendant le lot, environ 24,1 GiB. Aucun nouvel archive n'a été téléchargé.

## Collecte prospective : Go de faisabilité

Les trois réponses publiques ont été demandées avec une profondeur de 400 niveaux par côté. Elles passent toutes les validations de structure, d'ordre des prix, de quantités, de carnet non croisé, de taille et d'horodatage.

| Instrument | JSON brut | Gzip déterministe | Bids | Asks | Décalage fournisseur maximal toléré |
|---|---:|---:|---:|---:|---:|
| BTC-USDT | 25 874 octets | 5 421 octets | 400 | 400 | 1 096 ms après l'horloge locale |
| ETH-USDT | 24 544 octets | 5 326 octets | 400 | 400 | 1 108 ms après l'horloge locale |
| SOL-USDT | 24 810 octets | 7 538 octets | 400 | 400 | 1 098 ms après l'horloge locale |
| **Cycle de trois actifs** | **75 228 octets** | **18 285 octets** | — | — | — |

Le léger décalage positif des timestamps fournisseur reste sous la tolérance préenregistrée de 5 secondes. Il est compatible avec un décalage d'horloge entre la machine et OKX ; aucune observation n'a été antidatée.

À 96 cycles par jour pendant 420 jours :

- appels publics : `120 960`, soit 288 par jour ;
- volume JSON brut projeté : `2,825 GiB` ;
- volume gzip projeté : `0,687 GiB` ;
- stockage prudent avec facteur de sécurité ×3 : `2,060 GiB`.

Ces valeurs passent les plafonds gelés de 10 GiB bruts et 5 GiB conservés. Une salve de trois requêtes toutes les 15 minutes reste aussi très inférieure à la limite officielle de 40 requêtes par 2 secondes.

## Reproductibilité

Artifact final : `crypto-forecast-okx-l2-collection-feasibility-v1-7a415096e211446f`.

- résultat SHA-256 : `ce188e14463d58bb8d44990cc2ca1fce0bdc99eafe1b66cc1f653ec528f50b0b` ;
- métadonnées des probes SHA-256 : `a5b5c67f02df61284584b69835fc1a4c6c9ff6dce9476ae95bb62174ebbc5534` ;
- configuration SHA-256 : `c679f1ed035513e52bfb8f30b3a3331cd25c9a581939dc7071379a1bbad68880` ;
- code de faisabilité SHA-256 : `c62cc0279d07f8b9252e35a03b809cca27b56610a7728d11875b397c98ff8a8b`.

Une seconde exécution a relu les trois réponses conservées sans réseau. Elle a produit le même identifiant, le même résultat et les mêmes empreintes.

## Contrôles et limites

- trois requêtes `GET` publiques, sans authentification ;
- aucune archive historique supplémentaire ;
- aucun service planifié ou processus laissé actif ;
- aucun modèle, feature, target, backtest ou allocation ;
- aucun ordre, secret ou compte ;
- aucune modification de l'API, de l'interface ou de la production ;
- trois réponses ponctuelles seulement : les projections supposent que leur forme reste représentative ;
- la disponibilité future de l'endpoint public n'est pas garantie.

## Sources officielles

- [OKX API V5 — Market Data et GET /api/v5/market/books](https://www.okx.com/docs-v5/en/)
- [OKX API V5 — règles générales de limitation](https://www.okx.com/docs-v5/en/)

## Suite recommandée

Ne pas télécharger les archives complètes et ne pas passer au modèle. Si le chantier continue, le prochain lot doit concevoir puis tester un collecteur prospectif robuste et très petit : écriture atomique, absence de doublons, détection des créneaux manquants, manifeste quotidien et reprise après interruption.

Le démarrage d'une collecte longue durée devra rester une décision séparée, car il crée un service récurrent pendant environ 420 jours.

Il n'y a rien à tester dans le navigateur : l'application et la production restent inchangées.

# Lot 5P — Plan gelé d'acquisition du funding Binance

Date de gel : 13 septembre 2026. Ce document précède le téléchargement du corpus.

## Question

Les archives mensuelles publiques de funding réalisé Binance USDⓈ-M offrent-elles un historique suffisamment complet, intègre et compact pour autoriser, dans un lot ultérieur, la construction de features quotidiennes causales ?

Ce lot contrôle uniquement les données. Il ne crée ni cible, ni feature prédictive, ni modèle, ni allocation.

## Périmètre gelé

- Source : Binance Public Data, contrats perpétuels USDⓈ-M, répertoire mensuel `fundingRate`.
- Actifs : BTC, ETH, SOL, ADA, XRP, LINK et LTC, tous cotés en USDT.
- Période retenue : du 1er juin 2022 au 31 août 2026 inclus, soit 1 553 jours calendaires.
- Archives attendues : 51 mois × 7 instruments = 357 fichiers ZIP, chacun accompagné de son fichier `CHECKSUM` officiel.
- Accès : GET publics uniquement, sans compte, identifiant, secret ou ordre.
- Budget : 65 536 octets maximum par ZIP, 1 048 576 octets maximum décompressés par CSV et 16 777 216 octets maximum pour l'ensemble des ZIP.

La période est close avant l'expérience. Aucun mois, actif ou seuil ne sera ajouté après observation du résultat.

## Contrôles obligatoires

Le lecteur doit échouer de manière fermée si l'un des cas suivants se produit :

1. archive ou `CHECKSUM` absents ;
2. empreinte SHA-256 officielle invalide ou non concordante ;
3. budget individuel ou total dépassé ;
4. ZIP vide, corrompu, contenant plusieurs fichiers ou un chemin non sûr ;
5. schéma CSV non reconnu, horodatage invalide ou taux non numérique/non fini ;
6. ligne hors de son mois contractuel ;
7. doublon `(instrument, horodatage)` ou ordre temporel non strict ;
8. moins d'une observation sur un jour UTC ;
9. intervalle entre deux observations supérieur à 24 heures ;
10. première ou dernière journée contractuelle absente ;
11. moins de 1 551 jours calendaires par instrument.

Les jours manquants ne seront ni interpolés ni remplis. Les taux exacts publiés seront conservés sous forme textuelle dans une table normalisée d'événements.

## Sorties attendues

- un artifact immuable avec un manifeste de provenance ;
- un CSV d'événements vérifiés par instrument ;
- le relevé de chaque ZIP, de sa taille et de ses empreintes officielle et calculée ;
- un résumé quotidien de couverture uniquement, sans agrégat utilisable comme feature ;
- deux exécutions indépendantes donnant le même identifiant et les mêmes empreintes de contenu.

Les fichiers ZIP bruts ne sont pas conservés après validation : le manifeste et les CSV normalisés doivent suffire à reproduire et auditer l'acquisition.

## Décision

- **Go qualité** : les 357 archives passent tous les contrôles et les sept instruments couvrent chacun toute la période contractuelle.
- **No-Go** : tout autre résultat. La cause est consignée ; aucun seuil n'est assoupli après coup.

Un Go qualité autorise seulement le lot 5Q : construction de features quotidiennes avec disponibilité explicite et test de mutation du futur. Il ne constitue pas une preuve prédictive.

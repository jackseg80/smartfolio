# Livraison et retour arrière ML — 30 septembre 2026

## État

Implémentation préparée dans un checkout isolé issu de la version production
806338b522e89647c3f49260e0f418dbea1da761. Après autorisation, un aperçu Docker
séparé a été construit sur robot2:8083. Le compte jack et le registre de connexion
sont montés en lecture seule; aucune donnée utilisateur ne figure dans l'image.
Aucun commit, push, fusion ou déploiement de production n'a été exécuté.
Voir le [rapport de l'aperçu robot2](ML_RELIABILITY_ROBOT2_PREVIEW_2026-09-30.md).

Le paquet produit par scripts/package_ml_reliability.py contient les seuls fichiers
modifiés sémantiquement, les nouveaux fichiers autorisés, les historiques publics
vérifiés, les artefacts JSON et les rapports d'évaluation. Un manifeste SHA-256
contrôle chaque entrée. Aucun portefeuille, montant, quantité, clé, session, profil
utilisateur, fichier .env ou capture privée n'est inclus.

## Conditions avant publication, après autorisation distincte

1. Relire le manifeste et vérifier toutes ses empreintes. Confirmer que la production
   exécute toujours la version/image de référence; une évolution impose un nouveau
   report des corrections et les contrôles correspondants.
2. Intégrer les fichiers sur une branche de livraison propre basée sur la version
   production appropriée. Les différences de fins de ligne existantes ne doivent
   pas provoquer la réécriture des fichiers étrangers au chantier.
3. Construire et vérifier une image Linux/Python 3.11 isolée avec les dépendances
   du requirements.txt. Le contrôle Windows/Python 3.13 ne remplace pas cette étape.
   Exécuter les contrôles de contrat, sécurité, chargement JSON et calendriers.
4. Les données publiques data/ml_verified sont exclues du Dockerfile par
   .dockerignore; les installer explicitement dans le volume de données, puis
   contrôler les reçus et leurs empreintes. Les modèles JSON doivent être présents
   dans l'image ou dans un volume de modèles explicitement monté.
5. Enregistrer l'image précédente et la configuration de lancement sans exposer
   les secrets; conserver les anciens fichiers publics remplacés. Sauvegarder
   séparément le volume utilisateur existant, sans l'incorporer au paquet.
6. Exiger ML_AUTO_TRAIN=0, conserver RUN_SCHEDULER selon le fonctionnement existant,
   ne pas activer de scheduler ML ni modifier la gouvernance. Supprimer la variable
   ML_PORTFOLIO_SNAPSHOT. Maintenir JWT, cohérence X-User et rôle administrateur.
7. Présenter l'image isolée sur un port distinct et vérifier la session jack,
   CoinTracking API, le CSV Saxo sélectionné, les dates et les actifs réellement
   disponibles. Vérifier également une session expirée et une seconde identité.
8. Publier seulement après ces contrôles et l'autorisation de production.
   L'ancien deploy.sh --force ne convient pas à une livraison qui doit préserver
   des changements locaux.

## Retour arrière

Image de référence observée le 30 septembre :
sha256:0c724b8b8ff87d73885bf0e109276c7b1da2b211f3c9c497b33f9bcda544f2aa.
Conteneur : smartfolio-api. Démarrage observé : 2026-09-28T09:01:59.068663028Z.
Volumes observés : données et logs; aucun volume de modèles n'était monté.

En cas d'échec de session, de chargement ou de contrat, relancer l'image précédente
avec les mêmes paramètres et volumes. Restaurer les fichiers publics remplacés
depuis leur sauvegarde. Ne pas réinitialiser les profils ou portefeuilles.
Contrôler healthz, la session authentifiée, les sources, puis AI Dashboard et bourse.
Les nouveaux artefacts peuvent rester archivés hors du chemin actif, jamais être
déclarés validés par un ancien chargeur.

## Observation après publication éventuelle

ML_INFERENCE_JOURNAL=1 active, après publication autorisée, un journal SQLite
par utilisateur. Il enregistre les consultations manuelles de prévisions
disponibles, source, dates cibles, versions et provenance, sans quantités/montants.
Ce journal est désactivé dans le preview et par défaut. Il ne réentraîne rien et
n'exécute aucun ordre. La comparaison avec les réalisations est une analyse
ultérieure, pas un entraînement récurrent.

## Preview privé

La copie locale autorisée contient uniquement les symboles, l'ordre, les sources
et les dates. Elle est datée du contrôle authentifié; ce n'est pas une connexion
locale aux clés de production. Les données financières des autres pages ne sont pas répliquées localement.
L'aperçu robot2 utilise ensuite, après accord explicite distinct, un montage en
lecture seule du dossier jack existant sur robot2. Ses caches, Redis, clé JWT,
logs et port sont indépendants. Les cookies de production ne sont pas modifiés. Les sources de marché et les modèles utilisés pour l'inférence
restent les données publiques vérifiées.

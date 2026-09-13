# Résultat — lot 5I, collecteur prospectif L2 robuste

Date : 13 septembre 2026<br>
Périmètre : noyau de stockage local, tests de panne et un pilote public de trois captures. Aucun service permanent, compte, ordre, modèle ou environnement SmartFolio.

## Décision

Le lot est **`GO_TECHNICAL_COLLECTOR` selon le contrat préenregistré**.

Le collecteur sait écrire une capture atomiquement, reprendre depuis les fichiers durables, refuser un conflit sans écraser l'original, signaler les absences et les fichiers temporaires orphelins, puis reproduire le même artifact hors ligne.

Ce Go valide le noyau de stockage. Il n'autorise pas encore une tâche planifiée pendant 420 jours et ne démontre aucune valeur prédictive.

## Contrat validé hors ligne

- clé unique par journée, borne UTC et instrument ;
- contenu canonique compressé en gzip déterministe ;
- création atomique sans remplacement d'une capture existante ;
- répétition du même payload idempotente ;
- payload différent sur la même clé rejeté comme conflit ;
- capture antérieure inchangée octet pour octet après ajout d'un créneau ultérieur ;
- manifeste entièrement reconstruit depuis les captures ;
- créneaux dus absents explicitement listés ;
- créneaux futurs exclus des absences ;
- fichier temporaire orphelin signalé mais conservé ;
- chemin de reprise sortant de l'artifact rejeté.

## Pilote public ponctuel

Une seule série de trois requêtes publiques OKX a été conservée pour la borne du 13 septembre 2026 à 10:15 UTC.

| Instrument | Retard du pilote | Taille stockée gzip | État |
|---|---:|---:|---|
| BTC-USDT | 771 837 ms | 5 840 octets | Valide |
| ETH-USDT | 772 228 ms | 5 686 octets | Valide |
| SOL-USDT | 772 623 ms | 8 128 octets | Valide |
| **Total** | — | **19 654 octets** | 3/3 |

Le pilote a été lancé manuellement environ 12 minutes et 52 secondes après la borne. Il passe la tolérance ponctuelle préenregistrée de 15 minutes, mais **ne constitue pas une preuve de respect de la future tolérance opérationnelle de 120 secondes**. Cette dernière devra être testée avec un déclenchement réellement aligné.

Le manifeste de la journée indique :

- 288 captures théoriques sur une journée complète ;
- 126 captures déjà dues au moment du pilote ;
- 3 captures présentes ;
- 123 captures antérieures explicitement manquantes, car la collecte n'existait pas avant 10:15 UTC ;
- 162 captures pas encore dues, donc non marquées manquantes ;
- zéro fichier temporaire orphelin ;
- aucune valeur remplie ou interpolée.

La journée reste logiquement incomplète. C'est le comportement attendu pour un pilote commencé en milieu de journée.

## Incident détecté pendant le pilote

La première tentative locale a été arrêtée avant toute requête et toute écriture : la fonction de capture 5H refusait le nouveau numéro de schéma 5I. Le raccordement a été séparé sans modifier la validation 5H, et un test dédié couvre maintenant le schéma du collecteur et l'univers figé.

## Reproductibilité

Artifact final : `crypto-forecast-okx-l2-prospective-collector-v1-9bb1d74206beaac2`.

- résultat SHA-256 : `802fae34a45dda4f7baf9ccb95de5eed9fe439b39c511de71045304832b31bef` ;
- manifeste journalier SHA-256 : `1c901b51ffed0e6c3694dc96df4a1abc98ad980e09ce25d195333b291a6c5c12` ;
- configuration SHA-256 : `8a4d0b40aa3c5bdcf29f69ad17d3aec8aed3cbebff8efaa0fe1aecf985c8e85e` ;
- code du collecteur SHA-256 : `061606376154121030a6d0fe7fc11c8daff757f61fc24015d3ab5cf102c3fb06`.

Une relecture hors ligne des trois captures a produit le même identifiant, le même résultat et le même manifeste, octet pour octet.

## Contrôles et limites

- six tests unitaires spécifiques réussis ;
- Ruff et Black conformes ;
- aucune suppression automatique d'orphelin ;
- aucun secret, compte ou ordre ;
- aucun processus laissé actif ;
- aucune API, interface, port local ou production modifié ;
- aucune preuve d'un déclenchement réel sous 120 secondes ;
- aucune surveillance, alerte ou rotation de stockage ;
- aucune validation prédictive ou économique.

## Suite recommandée

Le prochain lot doit tester le déclenchement opérationnel sans lancer la campagne de 420 jours : verrou mono-instance, calcul de la prochaine borne, délai maximal de 120 secondes, comportement après veille ou panne réseau et un essai court réellement aligné sur une borne.

L'installation d'une tâche planifiée et le démarrage de la collecte longue durée restent une autorisation séparée.

Il n'y a rien à tester dans le navigateur : l'application et la production restent inchangées.

# Archive - Données Anciennes

Ce dossier contient les données archivées qui ne sont plus utilisées activement mais conservées pour référence historique.

## Contenu

### monitoring/
**Archivé le :** 15 novembre 2025
**Raison :** Données de monitoring d'août 2025 (3+ mois d'ancienneté)
**Fichiers :** 7 fichiers JSON (1.5 MB)
- `metrics_binance_2025-08-23.json` (361 KB)
- `metrics_binance_2025-08-24.json` (104 KB)
- `metrics_enhanced_simulator_2025-08-23.json` (376 KB)
- `metrics_enhanced_simulator_2025-08-24.json` (108 KB)
- `metrics_kraken_2025-08-24.json` (103 KB)
- `metrics_simulator_2025-08-23.json` (369 KB)
- `metrics_simulator_2025-08-24.json` (106 KB)

### execution_history/
**Archivé le :** 15 novembre 2025
**Raison :** Historique d'exécution d'août 2025 (3+ mois d'ancienneté)
**Fichiers :** 1 fichier JSON (14 KB)
- `sessions_2025-08-23.json` (14 KB)

## Politique de Rétention

- **Données récentes (< 30 jours) :** Conservées dans `data/monitoring/` et `data/execution_history/`
- **Données anciennes (> 90 jours) :** Archivées dans `data/_archive/`
- **Suppression définitive :** Après 1 an d'archivage (optionnel)

## Restauration

Pour restaurer des données archivées :

```bash
# Exemple: restaurer un fichier de monitoring
cp data/_archive/monitoring/metrics_simulator_2025-08-23.json data/monitoring/
```

---

*Archive créée automatiquement lors du nettoyage du codebase (Nov 2025)*

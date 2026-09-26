# Snapshots Automatiques de Portfolio

Ce système permet de créer automatiquement des **snapshots quotidiens** de votre portfolio pour le **tracking P&L**.

## Fichiers

- **`daily_snapshot.ps1`** : Script principal qui crée le snapshot
- **`setup_daily_snapshot_task.ps1`** : Configure la tâche planifiée Windows
- **Logs** : `data/logs/snapshots.log`

---

## Installation (une seule fois)

### 1. Ouvrir PowerShell en tant qu'Administrateur

Clic droit sur PowerShell > **"Exécuter en tant qu'administrateur"**

### 2. Exécuter le script de configuration

```powershell
cd D:\Python\smartfolio
.\scripts\setup_daily_snapshot_task.ps1
```

**Options disponibles** :
```powershell
# Changer l'heure d'exécution (par défaut 00:00)
.\scripts\setup_daily_snapshot_task.ps1 -Time "02:30"

# Changer l'utilisateur/source
.\scripts\setup_daily_snapshot_task.ps1 -UserId "demo" -Source "cointracking"
```

### 3. Vérifier la tâche

Ouvrir **"Planificateur de tâches"** Windows :
- Aller dans `Bibliothèque du Planificateur de tâches > CryptoRebal`
- Vous devriez voir **"Crypto Portfolio Daily Snapshot"**

---

## Test Manuel

### Tester le script directement

```powershell
.\scripts\daily_snapshot.ps1
```

Avec options :
```powershell
.\scripts\daily_snapshot.ps1 -UserId "jack" -Source "cointracking_api" -MinUsd 1.0
```

### Tester la tâche planifiée

```powershell
# Exécuter immédiatement (sans attendre minuit)
Start-ScheduledTask -TaskPath "\CryptoRebal\" -TaskName "Crypto Portfolio Daily Snapshot"

# Vérifier les logs
Get-Content data\logs\snapshots.log -Tail 10
```

---

## Vérifier les Snapshots

### Via API

```powershell
# Voir les métriques P&L
Invoke-RestMethod "http://localhost:8080/portfolio/metrics?source=cointracking_api&user_id=jack" | ConvertTo-Json -Depth 5
```

### Via fichier JSON

Le fichier `data/portfolio_history.json` contient tous les snapshots :

```powershell
Get-Content data\portfolio_history.json | ConvertFrom-Json | Format-Table date, user_id, source, total_value_usd
```

---

## Gestion de la Tâche

### Voir le statut

```powershell
Get-ScheduledTask -TaskPath "\CryptoRebal\" -TaskName "Crypto Portfolio Daily Snapshot"
```

### Désactiver temporairement

```powershell
Disable-ScheduledTask -TaskPath "\CryptoRebal\" -TaskName "Crypto Portfolio Daily Snapshot"
```

### Réactiver

```powershell
Enable-ScheduledTask -TaskPath "\CryptoRebal\" -TaskName "Crypto Portfolio Daily Snapshot"
```

### Supprimer

```powershell
Unregister-ScheduledTask -TaskPath "\CryptoRebal\" -TaskName "Crypto Portfolio Daily Snapshot" -Confirm:$false
```

---

## Logs

Les logs sont enregistrés dans **`data/logs/snapshots.log`** :

```
[2025-10-02 00:00:15] Snapshot OK - user=jack source=cointracking_api
[2025-10-03 00:00:12] Snapshot OK - user=jack source=cointracking_api
[2025-10-04 00:00:08] ERROR - user=jack source=cointracking_api - API timeout
```

Voir les derniers logs :
```powershell
Get-Content data\logs\snapshots.log -Tail 20
```

---

## Fonctionnement

1. **Chaque jour à minuit** (ou heure configurée), la tâche planifiée s'exécute
2. Le script appelle `POST /portfolio/snapshot`
3. Un nouveau snapshot est créé dans `data/portfolio_history.json`
4. Le dashboard peut calculer le **P&L quotidien** en comparant avec le snapshot précédent

---

## Dépannage

### La tâche ne s'exécute pas

1. Vérifier que le serveur FastAPI est démarré (`http://localhost:8080`)
2. Vérifier les logs : `data\logs\snapshots.log`
3. Tester manuellement : `.\scripts\daily_snapshot.ps1`

### Erreur "ExecutionPolicy"

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### API timeout (429)

L'API CoinTracking a des limites de rate. Attendez quelques minutes et réessayez.

---

## Historique de Rétention

Par défaut, le système garde **365 snapshots maximum** par utilisateur/source (voir `services/portfolio.py:384`).

Les snapshots les plus anciens sont automatiquement supprimés.

---

**Dernière mise à jour** : 2025-10-02


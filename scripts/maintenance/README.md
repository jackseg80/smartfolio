# Scripts de Maintenance - Crypto Rebal Starter

Hub central pour les utilitaires de maintenance du projet.

## Scripts Disponibles

### `clean_tree.ps1` (À créer)

Nettoyage automatique de l'arborescence de développement.

**Actions** :
- Supprime tous les dossiers `__pycache__`
- Supprime tous les fichiers `.pyc`, `.pyo`
- Supprime les logs à la racine (hors `data/logs/`)
- Nettoie les fichiers temporaires (`temp_*.json`, `*.tmp`, `*.bak`)

**Usage** :
```powershell
.\scripts\maintenance\clean_tree.ps1
```

**Contenu suggéré** :
```powershell
# clean_tree.ps1
Write-Host " Nettoyage de l'arborescence..." -ForegroundColor Cyan

# Supprimer __pycache__
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Write-Host "[OK] __pycache__ supprimés" -ForegroundColor Green

# Supprimer .pyc / .pyo
Get-ChildItem -Path . -Recurse -Include "*.pyc", "*.pyo" -File | Remove-Item -Force
Write-Host "[OK] Fichiers .pyc/.pyo supprimés" -ForegroundColor Green

# Supprimer logs à la racine (garder data/logs/)
Get-ChildItem -Path . -Filter "*.log" -File -Depth 0 | Remove-Item -Force
Write-Host "[OK] Logs racine supprimés" -ForegroundColor Green

# Supprimer temporaires
Get-ChildItem -Path . -Recurse -Include "temp_*.json", "*_temp.json", "*.tmp", "*.bak" -File | Remove-Item -Force
Write-Host "[OK] Fichiers temporaires supprimés" -ForegroundColor Green

Write-Host " Nettoyage terminé!" -ForegroundColor Green
```

---

### `verify_gitignore.ps1` (À créer)

Vérifie que les fichiers générés ne sont pas trackés par git.

**Actions** :
- Vérifie que `__pycache__/`, `*.pyc` ne sont pas dans `git ls-files`
- Vérifie que `*.log` ne sont pas trackés
- Alerte si des fichiers sensibles sont présents (`.env` non exemple)

**Usage** :
```powershell
.\scripts\maintenance\verify_gitignore.ps1
```

**Contenu suggéré** :
```powershell
# verify_gitignore.ps1
Write-Host " Vérification .gitignore..." -ForegroundColor Cyan

$errors = @()

# Vérifier .pyc
$pycTracked = git ls-files | Select-String -Pattern "\.pyc$"
if ($pycTracked) {
    $errors += ".pyc files are tracked!"
}

# Vérifier logs
$logsTracked = git ls-files | Select-String -Pattern "\.log$"
if ($logsTracked) {
    $errors += "Log files are tracked!"
}

# Vérifier .env (sauf .env.example)
$envTracked = git ls-files | Select-String -Pattern "^\.env$"
if ($envTracked) {
    $errors += ".env file is tracked! (SECURITY RISK)"
}

if ($errors.Count -gt 0) {
    Write-Host "[Error] Erreurs détectées:" -ForegroundColor Red
    $errors | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
} else {
    Write-Host "[OK] .gitignore correct" -ForegroundColor Green
}
```

---

### `archive_cleanup.ps1` (À créer)

Nettoie les fichiers d'archive obsolètes (>90 jours).

**Actions** :
- Liste les fichiers dans `static/archive/` datant de >90 jours
- Propose suppression interactive
- Crée un rapport de nettoyage

**Usage** :
```powershell
.\scripts\maintenance\archive_cleanup.ps1 [-Days 90] [-Force]
```

---

### `smoke_test.ps1` (À créer)

Tests rapides des endpoints critiques post-déploiement.

**Actions** :
- Ping `/health`, `/openapi.json`
- Teste endpoints essentiels :
  - `/api/risk/status`
  - `/api/analytics/summary` (si existe)
  - `/balances/current`
- Vérifie headers CORS

**Usage** :
```powershell
.\scripts\maintenance\smoke_test.ps1 [-BaseUrl "http://localhost:8080"]
```

**Contenu suggéré** :
```powershell
# smoke_test.ps1
param(
    [string]$BaseUrl = "http://localhost:8080"
)

Write-Host " Smoke tests - $BaseUrl" -ForegroundColor Cyan

$errors = @()

# Test 1: Health check
try {
    $health = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 5
    Write-Host "[OK] /health OK" -ForegroundColor Green
} catch {
    $errors += "/health FAIL: $_"
}

# Test 2: OpenAPI
try {
    $openapi = Invoke-RestMethod -Uri "$BaseUrl/openapi.json" -TimeoutSec 5
    if (-not $openapi.info) { throw "Invalid OpenAPI response" }
    Write-Host "[OK] /openapi.json OK" -ForegroundColor Green
} catch {
    $errors += "/openapi.json FAIL: $_"
}

# Test 3: Risk status
try {
    $risk = Invoke-RestMethod -Uri "$BaseUrl/api/risk/status" -TimeoutSec 5
    if ($risk.success -ne $true) { throw "Risk status returned success=false" }
    Write-Host "[OK] /api/risk/status OK" -ForegroundColor Green
} catch {
    $errors += "/api/risk/status FAIL: $_"
}

# Test 4: Balances
try {
    $balances = Invoke-RestMethod -Uri "$BaseUrl/balances/current?source=stub_balanced" -TimeoutSec 10
    Write-Host "[OK] /balances/current OK" -ForegroundColor Green
} catch {
    $errors += "/balances/current FAIL: $_"
}

# Résumé
if ($errors.Count -gt 0) {
    Write-Host "`n[Error] Tests échoués:" -ForegroundColor Red
    $errors | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
} else {
    Write-Host "`n Tous les tests sont passés!" -ForegroundColor Green
}
```

---

## Utilisation Recommandée

### Avant commit
```powershell
.\scripts\maintenance\clean_tree.ps1
.\scripts\maintenance\verify_gitignore.ps1
```

### Après déploiement
```powershell
.\scripts\maintenance\smoke_test.ps1 -BaseUrl "https://production.example.com"
```

### Maintenance mensuelle
```powershell
.\scripts\maintenance\archive_cleanup.ps1 -Days 90
```

---

## TODO

- [ ] Créer `clean_tree.ps1`
- [ ] Créer `verify_gitignore.ps1`
- [ ] Créer `smoke_test.ps1`
- [ ] Créer `archive_cleanup.ps1`
- [ ] Ajouter hooks pre-commit pour clean_tree
- [ ] Intégrer smoke_test dans CI/CD
- [ ] Documenter dans CLAUDE.md

---

**Dernière mise à jour** : 2025-09-30
**Auteur** : Audit architecture cleanup

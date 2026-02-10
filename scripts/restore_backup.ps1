# restore_backup.ps1
# Restauration d'un backup SmartFolio
# Usage: .\scripts\restore_backup.ps1 -UserId "jack" -BackupFile "backup_20260210_030000.zip" [-DryRun]
#
# IMPORTANT: Toujours utiliser -DryRun d'abord pour verifier les fichiers

param(
    [Parameter(Mandatory=$true)]
    [string]$UserId,

    [Parameter(Mandatory=$true)]
    [string]$BackupFile,

    [string]$BaseUrl = "http://localhost:8080",
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Write-Host ""
Write-Host "=== SmartFolio Backup Restore ===" -ForegroundColor Cyan
Write-Host "User:   $UserId" -ForegroundColor Gray
Write-Host "Backup: $BackupFile" -ForegroundColor Gray
Write-Host "Mode:   $(if ($DryRun) { 'DRY RUN (simulation)' } else { 'REAL RESTORE' })" -ForegroundColor $(if ($DryRun) { "Yellow" } else { "Red" })
Write-Host ""

$headers = @{
    "X-User" = "jack"
    "Content-Type" = "application/json"
}

# Etape 1: Verifier le backup
Write-Host "Step 1: Verifying backup integrity..." -ForegroundColor Cyan
try {
    $backupPath = "data\backups\$UserId\$BackupFile"
    if (-not (Test-Path $backupPath)) {
        Write-Host "ERROR: Backup file not found at $backupPath" -ForegroundColor Red
        exit 1
    }

    $verifyBody = @{ zip_path = $backupPath } | ConvertTo-Json
    $verifyResp = Invoke-RestMethod `
        -Uri "$BaseUrl/admin/backups/verify" `
        -Method POST `
        -Headers $headers `
        -Body $verifyBody `
        -TimeoutSec 30

    if ($verifyResp.ok -and $verifyResp.data.ok) {
        Write-Host "  Integrity OK: $($verifyResp.data.file_count) files, SHA256=$($verifyResp.data.checksum_sha256.Substring(0,16))..." -ForegroundColor Green
    } else {
        Write-Host "  Integrity FAILED: $($verifyResp.data.error)" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "  Verification error: $_" -ForegroundColor Red
    exit 1
}

# Etape 2: Dry run ou restauration
$dryRunValue = if ($DryRun) { "true" } else { "false" }
$actionLabel = if ($DryRun) { "Simulating restore" } else { "Restoring files" }

Write-Host ""
Write-Host "Step 2: $actionLabel..." -ForegroundColor Cyan

if (-not $DryRun) {
    Write-Host ""
    Write-Host "  WARNING: This will overwrite existing files!" -ForegroundColor Red
    $confirm = Read-Host "  Type 'RESTORE' to confirm"
    if ($confirm -ne "RESTORE") {
        Write-Host "  Aborted." -ForegroundColor Yellow
        exit 0
    }
}

try {
    $restoreBody = @{
        zip_path = "data\backups\$UserId\$BackupFile"
        user_id = $UserId
        dry_run = [System.Convert]::ToBoolean($dryRunValue)
    } | ConvertTo-Json

    $restoreResp = Invoke-RestMethod `
        -Uri "$BaseUrl/admin/backups/restore" `
        -Method POST `
        -Headers $headers `
        -Body $restoreBody `
        -TimeoutSec 120

    if ($restoreResp.ok -and $restoreResp.data.ok) {
        $data = $restoreResp.data
        if ($DryRun) {
            Write-Host ""
            Write-Host "  DRY RUN Results:" -ForegroundColor Yellow
            Write-Host "  Files that would be restored: $($data.file_count)" -ForegroundColor Yellow
            Write-Host "  Target: $($data.target_dir)" -ForegroundColor Gray
            Write-Host ""
            Write-Host "  Files:" -ForegroundColor Gray
            foreach ($f in $data.files_to_restore) {
                Write-Host "    - $f" -ForegroundColor Gray
            }
            Write-Host ""
            Write-Host "  Run again WITHOUT -DryRun to apply." -ForegroundColor Cyan
        } else {
            Write-Host ""
            Write-Host "  Restore completed: $($data.file_count) files restored" -ForegroundColor Green
            Write-Host "  Target: $($data.target_dir)" -ForegroundColor Gray

            # Log
            $logEntry = "[$timestamp] RESTORE OK - user=$UserId backup=$BackupFile files=$($data.file_count)"
            Add-Content -Path "data\logs\backups.log" -Value $logEntry -ErrorAction SilentlyContinue
        }
        exit 0
    } else {
        Write-Host "  Restore failed: $($restoreResp.data.error)" -ForegroundColor Red
        exit 1
    }

} catch {
    Write-Host "  Restore error: $_" -ForegroundColor Red
    exit 1
}

# setup_backup_task.ps1
# Configure une tache planifiee Windows pour les backups automatiques
# Usage: .\scripts\setup_backup_task.ps1 [-Time "03:00"] [-Uninstall]
#
# Necessite elevation administrateur

param(
    [string]$Time = "03:00",
    [switch]$Uninstall = $false
)

$taskName = "SmartFolio-DailyBackup"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $scriptDir
$backupScript = Join-Path $scriptDir "backup_all.ps1"

if ($Uninstall) {
    Write-Host "Removing scheduled task '$taskName'..." -ForegroundColor Yellow
    try {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "Task removed successfully." -ForegroundColor Green
    } catch {
        Write-Host "Task not found or already removed." -ForegroundColor Gray
    }
    exit 0
}

# Verifier que le script existe
if (-not (Test-Path $backupScript)) {
    Write-Host "ERROR: backup_all.ps1 not found at $backupScript" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== SmartFolio Backup Task Setup ===" -ForegroundColor Cyan
Write-Host "Task:   $taskName" -ForegroundColor Gray
Write-Host "Time:   $Time daily" -ForegroundColor Gray
Write-Host "Script: $backupScript" -ForegroundColor Gray
Write-Host ""

try {
    # Action: lancer PowerShell avec le script
    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$backupScript`" -Verify" `
        -WorkingDirectory $projectDir

    # Trigger: tous les jours a l'heure specifiee
    $trigger = New-ScheduledTaskTrigger -Daily -At $Time

    # Settings
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable:$false `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

    # Verifier si la tache existe deja
    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Updating existing task..." -ForegroundColor Yellow
        Set-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings | Out-Null
    } else {
        Write-Host "Creating new task..." -ForegroundColor Cyan
        Register-ScheduledTask `
            -TaskName $taskName `
            -Action $action `
            -Trigger $trigger `
            -Settings $settings `
            -Description "SmartFolio daily backup - data/users/* with rotation (7d/4w/12m)" `
            -RunLevel Limited | Out-Null
    }

    Write-Host ""
    Write-Host "Scheduled task '$taskName' configured successfully!" -ForegroundColor Green
    Write-Host "Next run: $(Get-Date -Format 'yyyy-MM-dd') $Time" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Cyan
    Write-Host "  View:    Get-ScheduledTask -TaskName '$taskName'" -ForegroundColor Gray
    Write-Host "  Run now: Start-ScheduledTask -TaskName '$taskName'" -ForegroundColor Gray
    Write-Host "  Remove:  .\scripts\setup_backup_task.ps1 -Uninstall" -ForegroundColor Gray

    exit 0

} catch {
    Write-Host "ERROR: Failed to create scheduled task: $_" -ForegroundColor Red
    Write-Host "Try running PowerShell as Administrator." -ForegroundColor Yellow
    exit 1
}

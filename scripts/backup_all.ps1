# backup_all.ps1
# Sauvegarde complete des donnees utilisateur SmartFolio
# Usage: .\scripts\backup_all.ps1 [-UserIds "jack,demo"] [-IncludeSecrets] [-Verify]
#
# Retention automatique: 7 daily / 4 weekly / 12 monthly

param(
    [string]$BaseUrl = "http://localhost:8080",
    [string]$UserIds = "",
    [switch]$IncludeSecrets = $false,
    [switch]$Verify = $false,
    [switch]$SkipRetention = $false
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$logFile = "data\logs\backups.log"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $entry = "[$timestamp] $Level - $Message"
    Write-Host $entry -ForegroundColor $(
        switch ($Level) {
            "ERROR" { "Red" }
            "WARN"  { "Yellow" }
            "OK"    { "Green" }
            default { "Cyan" }
        }
    )
    Add-Content -Path $logFile -Value $entry -ErrorAction SilentlyContinue
}

Write-Log "=== SmartFolio Backup Started ==="

# Build request body
$body = @{
    include_secrets = $IncludeSecrets.IsPresent
}
if ($UserIds -ne "") {
    $body["user_ids"] = @($UserIds -split ",")
}

try {
    $headers = @{
        "X-User" = "jack"
        "Content-Type" = "application/json"
    }

    # Trigger backup via admin API
    $response = Invoke-RestMethod `
        -Uri "$BaseUrl/admin/backups/create-now" `
        -Method POST `
        -Headers $headers `
        -Body ($body | ConvertTo-Json) `
        -TimeoutSec 120

    if ($response.ok -eq $true) {
        $data = $response.data
        Write-Log "Backup completed: $($data.total_ok) OK, $($data.total_failed) failed" "OK"

        # Detail par utilisateur
        foreach ($uid in $data.users.PSObject.Properties) {
            $u = $uid.Value
            if ($u.ok) {
                $sizeMB = [math]::Round($u.zip_size / 1MB, 2)
                Write-Log "  $($uid.Name): $($u.file_count) files, ${sizeMB}MB, SHA256=$($u.checksum_sha256.Substring(0,16))..."
            } else {
                Write-Log "  $($uid.Name): FAILED - $($u.error)" "ERROR"
            }
        }

        # Verification optionnelle
        if ($Verify) {
            Write-Log "Running backup verification..."
            $verifyResp = Invoke-RestMethod `
                -Uri "$BaseUrl/admin/backups/status" `
                -Method GET `
                -Headers $headers `
                -TimeoutSec 30
            if ($verifyResp.ok) {
                Write-Log "Verification OK: $($verifyResp.data.total_backups) backups, $($verifyResp.data.total_size_mb)MB total" "OK"
            }
        }

        # Retention (sauf si skip)
        if (-not $SkipRetention) {
            Write-Log "Applying retention policy..."
            $retentionResp = Invoke-RestMethod `
                -Uri "$BaseUrl/admin/backups/apply-retention" `
                -Method POST `
                -Headers $headers `
                -TimeoutSec 60
            if ($retentionResp.ok) {
                $totalDeleted = ($retentionResp.data.PSObject.Properties.Value | Measure-Object -Sum).Sum
                Write-Log "Retention applied: $totalDeleted old backups removed" "OK"
            }
        }

        exit 0
    } else {
        Write-Log "Backup failed: $($response.error)" "ERROR"
        exit 1
    }

} catch {
    Write-Log "Error: $_" "ERROR"
    exit 1
}

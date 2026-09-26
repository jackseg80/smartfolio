# Test de la Configuration Centralisée
# Vérifie que le système de configuration unifié fonctionne

$BASE_URL = "http://localhost:8001"
$HEADERS = @{ "X-User" = "jack" }

Write-Host "=== Test Configuration Centralisée ===" -ForegroundColor Green

# 1. Test avec différentes sources
Write-Host "`n1. Test des différentes sources de données..." -ForegroundColor Yellow

$sources = @("stub", "cointracking", "cointracking_api")

foreach ($source in $sources) {
    Write-Host "  Testing source: $source" -ForegroundColor Cyan
    
    try {
        $response = Invoke-RestMethod -Uri "$BASE_URL/balances/current?source=$source" -Method Get -Headers $HEADERS -ErrorAction Stop
        $itemCount = $response.items.Count
        $sourceUsed = $response.source_used
        Write-Host "     OK - $itemCount assets (source: $sourceUsed)" -ForegroundColor Green
        
        # Test portfolio metrics avec cette source
        $metricsResponse = Invoke-RestMethod -Uri "$BASE_URL/portfolio/metrics?source=$source" -Method Get -Headers $HEADERS -ErrorAction Stop
        if ($metricsResponse.ok) {
            $totalValue = '{0:F0}' -f $metricsResponse.metrics.total_value_usd
            $assetCount = $metricsResponse.metrics.asset_count
            $groupCount = $metricsResponse.metrics.group_count
            Write-Host "     Analytics: $totalValue USD, $assetCount assets, $groupCount groupes" -ForegroundColor White
        }
    }
    catch {
        Write-Host "    [Error] Erreur: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Write-Host ""
}

# 2. Test des modes de pricing
Write-Host "`n2. Test des modes de pricing..." -ForegroundColor Yellow

$pricingModes = @("local", "auto")

foreach ($pricing in $pricingModes) {
    Write-Host "  Testing pricing: $pricing" -ForegroundColor Cyan
    
    try {
        # Créer un plan de rebalancing avec ce mode de pricing
        $payload = @{
            "target_allocation" = @{
                "BTC" = 50
                "ETH" = 30
                "Others" = 20
            }
        }
        
        $response = Invoke-RestMethod -Uri "$BASE_URL/rebalance/plan?source=cointracking&pricing=$pricing" -Method Post -Headers $HEADERS -Body ($payload | ConvertTo-Json) -ContentType "application/json" -ErrorAction Stop
        
        if ($response.ok) {
            $totalValue = '{0:F0}' -f $response.portfolio_summary.total_value_usd
            Write-Host "     OK - Portfolio: $totalValue USD (pricing: $pricing)" -ForegroundColor Green
        } else {
            Write-Host "     Warning: $($response.message)" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "    [Error] Erreur: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# 3. Test des endpoints portfolio avec paramètres
Write-Host "`n3. Test endpoints portfolio avec paramètres..." -ForegroundColor Yellow

$tests = @(
    @{ endpoint = "/portfolio/metrics"; params = "?source=cointracking"; name = "Métriques Portfolio" },
    @{ endpoint = "/portfolio/trend"; params = "?days=7"; name = "Tendance 7 jours" },
    @{ endpoint = "/taxonomy/suggestions"; params = ""; name = "Suggestions Taxonomie" }
)

foreach ($test in $tests) {
    Write-Host "  Testing: $($test.name)" -ForegroundColor Cyan
    
    try {
        $url = "$BASE_URL$($test.endpoint)$($test.params)"
        $response = Invoke-RestMethod -Uri $url -Method Get -Headers $HEADERS -ErrorAction Stop
        
        if ($response.ok) {
            Write-Host "     OK" -ForegroundColor Green
        } else {
            Write-Host "    [Warning] Response: $($response.message)" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "    [Error] Erreur: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# 4. Test de consistance des données
Write-Host "`n4. Test de consistance entre sources..." -ForegroundColor Yellow

$sourcesToCompare = @("stub", "cointracking")
$results = @{}

foreach ($source in $sourcesToCompare) {
    try {
        $balances = Invoke-RestMethod -Uri "$BASE_URL/balances/current?source=$source" -Method Get -Headers $HEADERS -ErrorAction Stop
        $metrics = Invoke-RestMethod -Uri "$BASE_URL/portfolio/metrics?source=$source" -Method Get -Headers $HEADERS -ErrorAction Stop
        
        $results[$source] = @{
            "asset_count" = $balances.items.Count
            "total_value" = if ($metrics.ok) { $metrics.metrics.total_value_usd } else { 0 }
            "source_used" = $balances.source_used
        }
    }
    catch {
        $results[$source] = @{ "error" = $_.Exception.Message }
    }
}

Write-Host "  Comparaison des sources:" -ForegroundColor Cyan
foreach ($source in $results.Keys) {
    $data = $results[$source]
    if ($data.error) {
        Write-Host "    $source : [Error] $($data.error)" -ForegroundColor Red
    } else {
        $value = if ($data.total_value -gt 0) { '{0:F0} USD' -f $data.total_value } else { 'N/A' }
        Write-Host "    $source : $($data.asset_count) assets, $value (via $($data.source_used))" -ForegroundColor White
    }
}

# 5. Instructions pour tester l'interface
Write-Host "`n=== Instructions Interface Web ===" -ForegroundColor Green
Write-Host "Pour tester la configuration unifiée:" -ForegroundColor Yellow
Write-Host "1. Ouvrir: static\settings.html dans le navigateur" -ForegroundColor White
Write-Host "2. Changer la source (stub/csv/api) et sauvegarder" -ForegroundColor White
Write-Host "3. Aller sur le Dashboard et vérifier que les données changent" -ForegroundColor White
Write-Host "4. Vérifier les indicateurs de source dans l'en-tête" -ForegroundColor White
Write-Host ""
Write-Host "Pages à tester:" -ForegroundColor Cyan
Write-Host "   Dashboard: file:///$(Get-Location)\static\dashboard.html" -ForegroundColor Gray
Write-Host "   Rebalancing: file:///$(Get-Location)\static\rebalance.html" -ForegroundColor Gray
Write-Host "   Settings: file:///$(Get-Location)\static\settings.html" -ForegroundColor Gray

Write-Host "`n=== Test terminé ===" -ForegroundColor Green
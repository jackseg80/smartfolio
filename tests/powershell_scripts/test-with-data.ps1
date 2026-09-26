# Test with real portfolio data
$base = "http://127.0.0.1:8080"

Write-Host " Testing with Portfolio Data" -ForegroundColor Green

# Test portfolio optimization with sample data
$optimizationRequest = @{
    objective = "max_sharpe"
    lookback_days = 90
    expected_return_method = "mean_reversion"
    conservative = $true
    custom_constraints = @{
        max_weight = 0.4
        max_sector_weight = 0.6
        min_diversification_ratio = 0.3
    }
    include_current_weights = $true
} | ConvertTo-Json -Depth 3

Write-Host "`n Testing Portfolio Optimization..." -ForegroundColor Yellow

try {
    $headers = @{'Content-Type' = 'application/json'}
    $result = Invoke-RestMethod -Uri "$base/api/portfolio/optimization/optimize?min_usd=100" -Method Post -Body $optimizationRequest -Headers $headers
    
    Write-Host "[OK] Optimization successful!" -ForegroundColor Green
    Write-Host "   - Expected Return: $([math]::Round($result.expected_return * 100, 2))%" -ForegroundColor Cyan
    Write-Host "   - Volatility: $([math]::Round($result.volatility * 100, 2))%" -ForegroundColor Cyan
    Write-Host "   - Sharpe Ratio: $([math]::Round($result.sharpe_ratio, 2))" -ForegroundColor Cyan
    Write-Host "   - Assets optimized: $($result.weights.PSObject.Properties.Count)" -ForegroundColor Cyan
    
    if ($result.rebalancing_trades.Count -gt 0) {
        Write-Host "   - Trades needed: $($result.rebalancing_trades.Count)" -ForegroundColor Yellow
        $result.rebalancing_trades | Select-Object -First 3 | ForEach-Object {
            Write-Host "     • $($_.action.ToUpper()) $($_.symbol): $($_.usd_amount) USD" -ForegroundColor Gray
        }
    }
} catch {
    Write-Host "[Error] Portfolio optimization failed: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Message -match "price.*data") {
        Write-Host "    Tip: You may need historical price data. Try with stub data first." -ForegroundColor Yellow
    }
}

# Test backtesting with sample parameters
Write-Host "`n Testing Backtesting..." -ForegroundColor Yellow

$backtestRequest = @{
    strategy = "equal_weight"
    assets = @("BTC", "ETH", "SOL")
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    initial_capital = 10000
    rebalance_frequency = "monthly"
    benchmark = "BTC"
} | ConvertTo-Json -Depth 3

try {
    $headers = @{'Content-Type' = 'application/json'}
    $backtest = Invoke-RestMethod -Uri "$base/api/backtesting/run" -Method Post -Body $backtestRequest -Headers $headers
    
    Write-Host "[OK] Backtest successful!" -ForegroundColor Green
    Write-Host "   - Strategy: $($backtest.strategy_name)" -ForegroundColor Cyan
    Write-Host "   - Total Return: $([math]::Round($backtest.metrics.total_return * 100, 2))%" -ForegroundColor Cyan
    Write-Host "   - Sharpe Ratio: $([math]::Round($backtest.metrics.sharpe_ratio, 2))" -ForegroundColor Cyan
    Write-Host "   - Max Drawdown: $([math]::Round($backtest.metrics.max_drawdown * 100, 2))%" -ForegroundColor Cyan
    
} catch {
    Write-Host "[Error] Backtesting failed: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Message -match "price.*data") {
        Write-Host "    Tip: Historical price data needed. Check data/price_history/" -ForegroundColor Yellow
    }
}

# Test ML training (quick test)
Write-Host "`n Testing ML Training..." -ForegroundColor Yellow

$mlRequest = @{
    assets = @("BTC", "ETH", "SOL")
    lookback_days = 365
    include_market_indicators = $false
    save_models = $true
} | ConvertTo-Json -Depth 3

try {
    $headers = @{'Content-Type' = 'application/json'}
    $mlResult = Invoke-RestMethod -Uri "$base/api/ml/train" -Method Post -Body $mlRequest -Headers $headers
    
    Write-Host "[OK] ML training started!" -ForegroundColor Green
    Write-Host "   - Assets: $($mlResult.assets -join ', ')" -ForegroundColor Cyan
    Write-Host "   - Estimated duration: $($mlResult.estimated_duration_minutes) minutes" -ForegroundColor Cyan
    
} catch {
    Write-Host "[Error] ML training failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n Testing Complete!" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green
Write-Host " Web Interfaces:" -ForegroundColor Cyan
Write-Host "   • Portfolio Optimization: http://localhost:8080/static/portfolio-optimization.html" -ForegroundColor Gray
Write-Host "   • API Documentation: http://localhost:8080/docs" -ForegroundColor Gray
Write-Host "   • Risk Dashboard: http://localhost:8080/static/risk-dashboard.html" -ForegroundColor Gray

Write-Host "`n Ready to use!" -ForegroundColor Green
/**
 * AI Chat Context Builders - Page-specific data extractors
 * Each builder extracts relevant data from the current page for AI analysis
 */

/**
 * Dashboard - Global portfolio view
 */
export async function buildDashboardContext() {
    const context = {
        page: 'Dashboard - Global Portfolio View'
    };
    const activeUser = localStorage.getItem('activeUser') || 'demo';
    const activeSource = localStorage.getItem('activeSource') || 'cointracking';

    try {
        // 1. Load crypto balance data
        const balanceResult = await window.loadBalanceData(true);

        // Total value and positions (crypto)
        if (balanceResult.data && balanceResult.data.items) {
            const items = balanceResult.data.items;
            const cryptoTotal = items.reduce((sum, item) => sum + (item.value_usd || 0), 0);

            context.crypto = {
                total_value: cryptoTotal,
                positions_count: items.length,
                top_positions: items
                    .sort((a, b) => (b.value_usd || 0) - (a.value_usd || 0))
                    .slice(0, 10)
                    .map(item => ({
                        symbol: item.symbol || item.coin,
                        name: item.name || '',
                        value: item.value_usd || 0,
                        weight: (item.value_usd / cryptoTotal) * 100,
                        pnl_pct: item.pnl_pct || 0
                    }))
            };
        }

        // 2. Load Saxo/Bourse positions
        try {
            const saxoResponse = await fetch(`/api/saxo/positions?user_id=${activeUser}`, {
                headers: { 'X-User': activeUser }
            });
            if (saxoResponse.ok) {
                const saxoData = await saxoResponse.json();
                if (saxoData.positions && saxoData.positions.length > 0) {
                    const bourseTotal = saxoData.positions.reduce((sum, pos) => sum + (pos.market_value || 0), 0);
                    context.bourse = {
                        total_value: bourseTotal,
                        positions_count: saxoData.positions.length,
                        top_positions: saxoData.positions.slice(0, 5).map(pos => ({
                            symbol: pos.instrument_id,
                            value: pos.market_value || 0,
                            weight: pos.weight * 100,
                            pnl: pos.pnl
                        }))
                    };
                }
            }
        } catch (err) {
            console.warn('[AI Chat] Failed to load Saxo positions:', err);
        }

        // 3. Load Patrimoine/Wealth data
        try {
            const wealthResponse = await fetch('/api/wealth/patrimoine/summary', {
                headers: { 'X-User': activeUser }
            });
            if (wealthResponse.ok) {
                const wealthData = await wealthResponse.json();
                context.patrimoine = {
                    net_worth: wealthData.net_worth || 0,
                    liquidity: wealthData.breakdown?.liquidity || 0,
                    tangible: wealthData.breakdown?.tangible || 0,
                    liabilities: wealthData.total_liabilities || 0
                };
            }
        } catch (err) {
            console.warn('[AI Chat] Failed to load wealth data:', err);
        }

        // 4. Load Risk Score (from unified store, not API)
        // Note: API returns structural risk score (78.9), but dashboard displays
        // the blended risk score (71) from the unified store
        if (window.riskStore) {
            const storeState = window.riskStore.getState();
            context.risk_score = storeState.scores?.risk || 0;
        } else {
            console.warn('[AI Chat] Risk store not available, risk score will be missing');
        }

        // 5. Load Decision Index & Analytics
        try {
            const govResponse = await fetch(`/execution/governance/state?source=${activeSource}`, {
                headers: { 'X-User': activeUser }
            });
            if (govResponse.ok) {
                const govData = await govResponse.json();
                context.decision_index = govData.scores?.decision || 0;
                context.phase = govData.phase?.phase_now || 'unknown';
                context.regime_components = govData.scores?.components || {};
            }
        } catch (err) {
            console.warn('[AI Chat] Failed to load governance state:', err);
        }

        // 6. Load ML Sentiment
        try {
            const sentimentResponse = await fetch('/api/ml/sentiment/unified', {
                headers: { 'X-User': activeUser }
            });
            if (sentimentResponse.ok) {
                const sentimentData = await sentimentResponse.json();
                context.ml_sentiment = sentimentData.aggregated_sentiment?.score || 0;
            }
        } catch (err) {
            console.warn('[AI Chat] Failed to load ML sentiment:', err);
        }

        // 7. Load Market Regime
        try {
            const regimeResponse = await fetch('/api/ml/regime/current', {
                headers: { 'X-User': activeUser }
            });
            if (regimeResponse.ok) {
                const regimeData = await regimeResponse.json();
                context.regime = regimeData.regime_prediction?.regime_name || 'unknown';
            }
        } catch (err) {
            console.warn('[AI Chat] Failed to load regime:', err);
        }

    } catch (error) {
        console.error('[AI Chat] Error building dashboard context:', error);
        context.error = 'Failed to load portfolio data';
    }

    return context;
}

/**
 * Risk Dashboard - Risk metrics and alerts
 */
export async function buildRiskDashboardContext() {
    const context = {
        page: 'Risk Dashboard - Risk Analysis'
    };

    try {
        const activeUser = localStorage.getItem('activeUser') || 'demo';
        const activeSource = localStorage.getItem('activeSource') || 'cointracking';

        // PRIORITY: Use riskStore (same data source as frontend UI)
        if (window.riskStore) {
            const storeState = window.riskStore.getState();

            // Risk score (blended, same as UI displays)
            context.risk_score = storeState.scores?.risk || 0;

            // Portfolio summary from store (correct property: portfolioSummary, not portfolio)
            if (storeState.portfolioSummary) {
                context.portfolio_value = storeState.portfolioSummary.total_value || 0;
                context.num_assets = storeState.portfolioSummary.num_assets || 0;
            }

            // Risk metrics from store (correct property: riskMetrics, not metrics)
            if (storeState.riskMetrics) {
                const metrics = storeState.riskMetrics;

                // VaR is in decimal format in store, convert to USD
                const portfolioValue = storeState.portfolioSummary?.total_value || 0;
                const varDecimal = metrics.var_95_1d || metrics.var_95 || 0;
                context.var_95 = varDecimal * portfolioValue;

                context.max_drawdown = metrics.max_drawdown;
                context.sharpe_ratio = metrics.sharpe_ratio;
                context.sortino_ratio = metrics.sortino_ratio;
                context.hhi = metrics.hhi || metrics.concentration_hhi;
                context.volatility = metrics.volatility_annualized || metrics.volatility;
            }
        } else {
            // FALLBACK: Use API if store not available
            console.warn('[AI Chat] Risk store not available, using API fallback');

            const response = await fetch(`/api/risk/dashboard?source=${activeSource}`, {
                headers: { 'X-User': activeUser }
            });

            if (response.ok) {
                const data = await response.json();

                // Risk score from API
                if (data.risk_score !== undefined) {
                    context.risk_score = data.risk_score;
                } else if (data.risk_metrics?.risk_score !== undefined) {
                    context.risk_score = data.risk_metrics.risk_score;
                }

                // Portfolio summary
                const portfolioValue = data.portfolio_summary?.total_value || 0;
                if (data.portfolio_summary) {
                    context.portfolio_value = portfolioValue;
                    context.num_assets = data.portfolio_summary.num_assets;
                }

                // Risk metrics
                const metrics = data.metrics || data.risk_metrics;
                if (metrics) {
                    const varDecimal = metrics.var_95_1d || metrics.var_95 || 0;
                    context.var_95 = varDecimal * portfolioValue;

                    context.max_drawdown = metrics.max_drawdown;
                    context.sharpe_ratio = metrics.sharpe_ratio;
                    context.sortino_ratio = metrics.sortino_ratio;
                    context.hhi = metrics.hhi || metrics.concentration_hhi;
                    context.volatility = metrics.volatility_annualized || metrics.volatility;
                }
            } else {
                console.error('[AI Chat] Risk dashboard API error:', response.status, response.statusText);
                context.error = `API error: ${response.status}`;
            }
        }

        // Active alerts
        try {
            const alertsResponse = await fetch('/api/alerts/active', {
                headers: { 'X-User': activeUser }
            });

            if (alertsResponse.ok) {
                const alertsData = await alertsResponse.json();
                // API returns List[AlertResponse] directly (not wrapped in {ok, alerts})
                if (Array.isArray(alertsData)) {
                    context.alerts = alertsData.map(alert => ({
                        severity: alert.severity,
                        type: alert.alert_type,
                        message: alert.data?.current_value ?
                            `${alert.alert_type}: ${alert.data.current_value}` :
                            alert.alert_type,
                        created_at: alert.created_at
                    }));
                }
            }
        } catch (err) {
            console.warn('[AI Chat] Failed to load alerts:', err);
        }

        // Market cycles - Load from governance state
        try {
            const govResponse = await fetch(`/execution/governance/state?source=${activeSource}`, {
                headers: { 'X-User': activeUser }
            });

            if (govResponse.ok) {
                const govData = await govResponse.json();

                // Cycle score from components
                const cycleScore = govData.scores?.components?.trend_regime || 0;
                context.cycle_score = cycleScore;

                // Calculate market phase
                if (cycleScore < 70) {
                    context.market_phase = 'bearish';
                } else if (cycleScore < 90) {
                    context.market_phase = 'moderate';
                } else {
                    context.market_phase = 'bullish';
                }

                // Dominance phase (btc/eth/large/alt)
                context.dominance_phase = govData.phase?.phase_now || 'unknown';
                context.phase_confidence = govData.phase?.confidence || 0;
            }
        } catch (err) {
            console.warn('[AI Chat] Failed to load cycles from governance:', err);
        }

    } catch (error) {
        console.error('[AI Chat] Error building risk context:', error);
        context.error = 'Failed to load risk data';
    }

    return context;
}

/**
 * Analytics Unified - ML analysis and decision index
 */
export async function buildAnalyticsContext() {
    const context = {
        page: 'Analytics Unified - ML Analysis'
    };

    try {
        const activeUser = localStorage.getItem('activeUser') || 'demo';
        const activeSource = localStorage.getItem('activeSource') || 'cointracking';

        // 1. Load Decision Index & Phase from Governance
        try {
            const govResponse = await fetch(`/execution/governance/state?source=${activeSource}`, {
                headers: { 'X-User': activeUser }
            });
            if (govResponse.ok) {
                const govData = await govResponse.json();
                context.decision_index = govData.scores?.decision || 0;
                context.dominance_phase = govData.phase?.phase_now || 'unknown';  // btc/eth/large/alt
                context.regime_components = govData.scores?.components || {};

                // Calculate market phase from cycle score (allocation-engine.js logic)
                const cycleScore = govData.scores?.components?.trend_regime || 0;
                context.cycle_score = cycleScore;
                if (cycleScore < 70) {
                    context.market_phase = 'bearish';
                } else if (cycleScore < 90) {
                    context.market_phase = 'moderate';
                } else {
                    context.market_phase = 'bullish';
                }
            }
        } catch (err) {
            console.warn('[AI Chat] Failed to load governance state:', err);
        }

        // 2. Load ML Sentiment
        try {
            const sentimentResponse = await fetch('/api/ml/sentiment/unified', {
                headers: { 'X-User': activeUser }
            });
            if (sentimentResponse.ok) {
                const sentimentData = await sentimentResponse.json();
                const rawScore = sentimentData.aggregated_sentiment?.score || 0;
                // Convert from [-1, 1] to [0, 100] scale: 50 + (score × 50)
                context.ml_sentiment = 50 + (rawScore * 50);
                context.ml_sentiment_label = sentimentData.aggregated_sentiment?.label || 'unknown';
            }
        } catch (err) {
            console.warn('[AI Chat] Failed to load ML sentiment:', err);
        }

        // 3. Load Market Regime
        try {
            const regimeResponse = await fetch('/api/ml/regime/current', {
                headers: { 'X-User': activeUser }
            });
            if (regimeResponse.ok) {
                const regimeData = await regimeResponse.json();
                context.regime = regimeData.regime_prediction?.regime_name || 'unknown';
                context.regime_confidence = regimeData.regime_prediction?.confidence || 0;
            }
        } catch (err) {
            console.warn('[AI Chat] Failed to load regime:', err);
        }

        // 4. Load Risk Score (from unified store, not API)
        if (window.riskStore) {
            const storeState = window.riskStore.getState();
            context.risk_score = storeState.scores?.risk || 0;
        }

        // 5. Volatility forecasts (if available in global cache)
        if (window.lastVolatilityForecasts) {
            context.volatility_forecasts = window.lastVolatilityForecasts;
        }

    } catch (error) {
        console.error('[AI Chat] Error building analytics context:', error);
        context.error = 'Failed to load analytics data';
    }

    return context;
}

/**
 * Saxo Dashboard - Stock portfolio (existing implementation adapted)
 * Uses actual property names from saxo-dashboard.html data structure
 */
export async function buildSaxoContext() {
    const context = {
        page: 'Saxo Dashboard - Portfolio Bourse',
        timestamp: new Date().toISOString()
    };

    try {
        // Get current portfolio data (from global state set by saxo-dashboard.html)
        const data = window.currentPortfolioData;

        if (!data) {
            context.error = 'Aucune donnée de portefeuille chargée';
            return context;
        }

        const summary = data.summary || {};
        const savedCash = window.portfolioCash || 0;

        // Portfolio summary (use actual property names from saxo-dashboard.html)
        const totalValue = Number(summary.total_value_usd || 0);
        const totalValueIncludesCash = data.totalValueIncludesCash || false;
        const totalWithCash = totalValueIncludesCash ? totalValue : (totalValue + savedCash);

        context.total_value = totalWithCash;
        context.total_positions = summary.total_positions || data.positions?.length || 0;
        context.cash = savedCash;

        // P&L information
        if (summary.total_pnl_usd !== undefined) {
            context.total_pnl = summary.total_pnl_usd;
            context.total_pnl_pct = summary.total_pnl_pct || 0;
        }

        // Top 15 positions with details including stop loss if available
        if (data.positions && Array.isArray(data.positions)) {
            context.positions = data.positions.slice(0, 15).map(p => {
                const pos = {
                    symbol: p.Symbol || p.symbol || '?',
                    name: p.Name || p.name || '',
                    value: p.MarketValue || p.market_value_usd || p.value || 0,
                    weight: p.Weight || p.weight || 0,
                    pnl: p.PnL || p.pnl || 0,
                    pnl_pct: p.PnLPercent || p.pnl_pct || 0,
                    sector: p.Sector || p.sector || 'Unknown'
                };

                // Add stop loss if available
                if (p.StopLoss || p.RecommendedStopLoss) {
                    pos.stop_loss = {
                        recommended: p.RecommendedStopLoss || p.StopLoss,
                        method: p.StopLossMethod || 'Fixed Variable',
                        risk_reward: p.RiskReward || 0
                    };
                }

                return pos;
            });
        }

        // Sector allocation
        if (summary.sector_allocation) {
            context.sectors = summary.sector_allocation;
        }

        // Asset allocation
        if (summary.asset_allocation) {
            context.asset_allocation = summary.asset_allocation;
        }

        // Currency exposure
        if (summary.currency_exposure) {
            context.currencies = summary.currency_exposure;
        }

        // Risk metrics if available
        if (data.risk_metrics) {
            context.risk_score = data.risk_metrics.score || 0;
            context.volatility = data.risk_metrics.volatility || 0;
        }

        // Market Opportunities
        if (window.lastOpportunitiesData) {
            const oppsData = window.lastOpportunitiesData;
            context.market_opportunities = {
                horizon: window.currentHorizon || 'medium',
                gaps: (oppsData.gaps || []).map(gap => ({
                    sector: gap.sector,
                    current: gap.current_pct,
                    target: gap.target_pct,
                    gap_pct: gap.gap_pct
                })),
                top_opportunities: (oppsData.opportunities || []).slice(0, 10).map(opp => ({
                    symbol: opp.symbol,
                    name: opp.name,
                    type: opp.type,
                    score: opp.score,
                    amount: opp.amount_usd,
                    sector: opp.sector
                })),
                suggested_sales: (oppsData.suggested_sales || []).map(sale => ({
                    symbol: sale.symbol,
                    current_weight: sale.current_weight_pct,
                    suggested_reduction: sale.sell_pct,
                    reason: sale.reason
                }))
            };
        }

        // Risk metrics from riskDashboardData
        if (window.riskDashboardData) {
            const riskData = window.riskDashboardData;
            context.risk_metrics_detailed = {
                sharpe_ratio: riskData.sharpe_ratio,
                sortino_ratio: riskData.sortino_ratio,
                max_drawdown: riskData.max_drawdown,
                var_95: riskData.var_95,
                concentration_top3: riskData.concentration_top3,
                concentration_hhi: riskData.hhi
            };
        }

    } catch (error) {
        console.error('Error building Saxo context:', error);
        context.error = 'Failed to load portfolio data';
    }

    return context;
}

/**
 * Wealth Dashboard - Patrimoine and net worth
 */
export async function buildWealthContext() {
    const context = {
        page: 'Wealth Dashboard - Patrimoine'
    };

    try {
        const activeUser = localStorage.getItem('activeUser') || 'demo';

        // Fetch wealth summary
        const response = await fetch('/api/wealth/patrimoine/summary', {
            headers: { 'X-User': activeUser }
        });

        if (response.ok) {
            const data = await response.json();

            // Parse response (API returns data directly, no wrapper)
            context.net_worth = data.net_worth || 0;
            context.total_assets = data.total_assets || 0;
            context.total_liabilities = data.total_liabilities || 0;

            // Assets breakdown
            if (data.breakdown) {
                context.liquidity = data.breakdown.liquidity || 0;
                context.tangible = data.breakdown.tangible || 0;
                context.insurance = data.breakdown.insurance || 0;
                context.liabilities = data.breakdown.liability || 0;
            }

            // Counts
            if (data.counts) {
                context.counts = data.counts;
            }
        }

    } catch (error) {
        console.error('[AI Chat] Error building wealth context:', error);
        context.error = 'Failed to load wealth data';
    }

    return context;
}

/**
 * Settings - Configuration and API keys status
 */
export async function buildSettingsContext() {
    const context = {
        page: 'Settings - Configuration'
    };

    try {
        const activeUser = localStorage.getItem('activeUser') || 'demo';

        // Get user configuration from localStorage
        context.user_id = activeUser;
        context.active_source = localStorage.getItem('activeSource') || 'cointracking';
        context.pricing_mode = localStorage.getItem('pricingMode') || 'local';
        context.display_currency = localStorage.getItem('displayCurrency') || 'USD';
        context.theme = localStorage.getItem('theme') || 'auto';
        context.min_usd_threshold = parseFloat(localStorage.getItem('minUsdThreshold') || '1.0');

        // ✅ USER ISOLATION: Check which API keys are configured (without exposing values)
        // Use globalConfig which is already isolated per user
        const apiKeys = {
            cointracking_api: !!window.globalConfig?.get('cointracking_api_key'),
            cointracking_secret: !!window.globalConfig?.get('cointracking_api_secret'),
            coingecko: !!window.globalConfig?.get('coingecko_api_key'),
            fred: !!window.globalConfig?.get('fred_api_key'),
            groq: !!window.globalConfig?.get('groq_api_key'),
            claude: !!window.globalConfig?.get('claude_api_key'),
            grok: !!window.globalConfig?.get('grok_api_key'),
            openai: !!window.globalConfig?.get('openai_api_key')
        };
        context.configured_apis = Object.entries(apiKeys)
            .filter(([_, configured]) => configured)
            .map(([name, _]) => name);

        // Check Saxo OAuth status
        try {
            const saxoResponse = await fetch('/api/saxo/auth/status', {
                headers: { 'X-User': activeUser }
            });
            if (saxoResponse.ok) {
                const saxoData = await saxoResponse.json();
                const status = saxoData.data;
                context.saxo_oauth = {
                    connected: status.status === 'connected',
                    environment: status.environment || 'unknown',
                    expires_at: status.expires_at || null
                };
            }
        } catch (err) {
            console.warn('[AI Chat] Failed to load Saxo OAuth status:', err);
        }

        // Get enabled features
        context.features = {
            coingecko_classification: localStorage.getItem('enable_coingecko_classification') !== 'false',
            portfolio_snapshots: localStorage.getItem('enable_portfolio_snapshots') !== 'false',
            performance_tracking: localStorage.getItem('enable_performance_tracking') !== 'false'
        };

        // ✅ USER ISOLATION: Get AI provider preference (from globalConfig)
        context.ai_provider = window.globalConfig?.get('aiProvider') || 'groq';
        context.ai_include_docs = window.globalConfig?.get('aiIncludeDocs') !== false;

    } catch (error) {
        console.error('[AI Chat] Error building settings context:', error);
        context.error = 'Failed to load settings data';
    }

    return context;
}

/**
 * Generic fallback context builder
 */
export async function buildGenericContext() {
    return {
        page: 'SmartFolio',
        timestamp: new Date().toISOString()
    };
}

/**
 * Context builders registry
 * Maps page identifiers to their builder functions
 */
export const contextBuilders = {
    'dashboard': buildDashboardContext,
    'risk-dashboard': buildRiskDashboardContext,
    'analytics-unified': buildAnalyticsContext,
    'saxo-dashboard': buildSaxoContext,
    'wealth-dashboard': buildWealthContext,
    'settings': buildSettingsContext,
    'generic': buildGenericContext
};

/**
 * Get context builder for a specific page
 * @param {string} pageId - Page identifier
 * @returns {Function} Context builder function
 */
export function getContextBuilder(pageId) {
    return contextBuilders[pageId] || buildGenericContext;
}

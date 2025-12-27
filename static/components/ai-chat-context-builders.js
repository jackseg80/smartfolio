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

        // 4. Load Risk Score
        try {
            const riskResponse = await fetch('/api/risk/dashboard', {
                headers: { 'X-User': activeUser }
            });
            if (riskResponse.ok) {
                const riskData = await riskResponse.json();
                context.risk_score = riskData.risk_metrics?.risk_score || 0;
            }
        } catch (err) {
            console.warn('[AI Chat] Failed to load risk score:', err);
        }

        // 5. Load Decision Index & Analytics
        try {
            const govResponse = await fetch('/execution/governance/state', {
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

        // Debug: Log user
        console.log('[AI Chat] Building Risk Dashboard context for user:', activeUser);

        // Fetch risk dashboard data
        const response = await fetch('/api/risk/dashboard', {
            headers: { 'X-User': activeUser }
        });

        // Debug: Log response status
        console.log('[AI Chat] Risk dashboard API response status:', response.status);

        if (response.ok) {
            const data = await response.json();

            // Debug: Log data structure
            console.log('[AI Chat] Risk dashboard data received:', {
                has_risk_score: data.risk_score !== undefined,
                has_risk_metrics: !!data.risk_metrics,
                has_portfolio_summary: !!data.portfolio_summary,
                keys: Object.keys(data)
            });

            // Risk score (check both locations)
            if (data.risk_score !== undefined) {
                context.risk_score = data.risk_score;
            } else if (data.risk_metrics?.risk_score !== undefined) {
                context.risk_score = data.risk_metrics.risk_score;
            }

            // Risk metrics (check both data.metrics and data.risk_metrics)
            const metrics = data.metrics || data.risk_metrics;
            if (metrics) {
                context.var_95 = metrics.var_95_1d || metrics.var_95;
                context.max_drawdown = metrics.max_drawdown;
                context.sharpe_ratio = metrics.sharpe_ratio;
                context.sortino_ratio = metrics.sortino_ratio;
                context.hhi = metrics.hhi || metrics.concentration_hhi;
                context.volatility = metrics.volatility_annualized || metrics.volatility;
            }

            // Portfolio summary
            if (data.portfolio_summary) {
                context.portfolio_value = data.portfolio_summary.total_value;
                context.num_assets = data.portfolio_summary.num_assets;
            }

            // Debug: Log extracted context
            console.log('[AI Chat] Risk context extracted:', {
                risk_score: context.risk_score,
                has_metrics: !!(context.var_95 || context.max_drawdown)
            });
        } else {
            console.error('[AI Chat] Risk dashboard API error:', response.status, response.statusText);
            context.error = `API error: ${response.status}`;
        }

        // Active alerts
        try {
            const alertsResponse = await fetch('/api/alerts/active', {
                headers: { 'X-User': activeUser }
            });

            if (alertsResponse.ok) {
                const alertsData = await alertsResponse.json();
                if (alertsData.ok && alertsData.alerts) {
                    context.alerts = alertsData.alerts.map(alert => ({
                        severity: alert.severity,
                        message: alert.message
                    }));
                    console.log('[AI Chat] Loaded', context.alerts.length, 'active alerts');
                }
            }
        } catch (err) {
            console.warn('[AI Chat] Failed to load alerts:', err);
        }

        // Market cycles (if available in global state)
        if (window.lastCyclesData) {
            context.cycles = window.lastCyclesData;
            console.log('[AI Chat] Using cached cycles data');
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
        // Get unified state
        const unifiedState = window.getUnifiedState ? window.getUnifiedState() : {};

        // Decision Index
        if (unifiedState.decision_index !== undefined) {
            context.decision_index = unifiedState.decision_index;
        }

        // ML Sentiment
        if (unifiedState.ml_sentiment !== undefined) {
            context.ml_sentiment = unifiedState.ml_sentiment;
        }

        // Market phase
        if (unifiedState.phase) {
            context.phase = unifiedState.phase;
        }

        // Regime scores
        if (unifiedState.regime) {
            context.regime = {
                ccs: unifiedState.regime.ccs_score,
                onchain: unifiedState.regime.onchain_score,
                risk: unifiedState.regime.risk_score,
                total: unifiedState.regime.total_score
            };
        }

        // Volatility forecasts (if available)
        if (window.lastVolatilityForecasts) {
            context.volatility_forecasts = window.lastVolatilityForecasts;
        }

    } catch (error) {
        console.error('Error building analytics context:', error);
        context.error = 'Failed to load analytics data';
    }

    return context;
}

/**
 * Saxo Dashboard - Stock portfolio (existing implementation adapted)
 */
export async function buildSaxoContext() {
    const context = {
        page: 'Saxo Dashboard - Portfolio Bourse',
        timestamp: new Date().toISOString()
    };

    try {
        // Get current portfolio data (from global state)
        const portfolioData = window.currentPortfolioData;

        if (!portfolioData) {
            context.error = 'No portfolio data loaded';
            return context;
        }

        // Portfolio summary
        const totalValue = portfolioData.totalValue || 0;
        const cash = window.portfolioCash || 0;

        context.total_value = totalValue + cash;
        context.total_positions = portfolioData.positions ? portfolioData.positions.length : 0;
        context.cash = cash;

        // P&L
        if (portfolioData.totalPnL !== undefined) {
            context.total_pnl = portfolioData.totalPnL;
            context.total_pnl_pct = portfolioData.totalPnLPct || 0;
        }

        // Top 15 positions with stop loss info
        if (portfolioData.positions) {
            context.positions = portfolioData.positions
                .slice(0, 15)
                .map(pos => ({
                    symbol: pos.symbol,
                    name: pos.name || '',
                    value: pos.marketValue || 0,
                    weight: pos.weight || 0,
                    pnl: pos.pnl || 0,
                    pnl_pct: pos.pnlPct || 0,
                    sector: pos.sector || 'Unknown',
                    stop_loss: pos.stopLoss ? {
                        recommended: pos.stopLoss.recommended,
                        method: pos.stopLoss.method,
                        risk_reward: pos.stopLoss.riskReward
                    } : null
                }));
        }

        // Sector allocation
        if (portfolioData.sectorAllocation) {
            context.sectors = portfolioData.sectorAllocation;
        }

        // Asset allocation
        if (portfolioData.assetAllocation) {
            context.asset_allocation = portfolioData.assetAllocation;
        }

        // Currency exposure
        if (portfolioData.currencyExposure) {
            context.currencies = portfolioData.currencyExposure;
        }

        // Risk metrics
        if (window.riskDashboardData) {
            const risk = window.riskDashboardData;
            context.risk_score = risk.risk_score;
            context.volatility = risk.volatility;

            if (risk.metrics) {
                context.risk_metrics_detailed = {
                    sharpe_ratio: risk.metrics.sharpe_ratio,
                    sortino_ratio: risk.metrics.sortino_ratio,
                    max_drawdown: risk.metrics.max_drawdown,
                    var_95: risk.metrics.var_95,
                    concentration_top3: risk.metrics.concentration_top3,
                    concentration_hhi: risk.metrics.hhi
                };
            }
        }

        // Market Opportunities
        if (window.lastOpportunitiesData) {
            const opps = window.lastOpportunitiesData;
            context.market_opportunities = {
                horizon: window.currentHorizon || 'medium',
                gaps: opps.gaps || [],
                top_opportunities: (opps.opportunities || []).slice(0, 10),
                suggested_sales: opps.suggested_sales || []
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

            if (data.ok) {
                // Net worth
                context.net_worth = data.net_worth || 0;
                context.total_assets = data.total_assets || 0;

                // Assets breakdown
                if (data.assets) {
                    context.assets = data.assets;
                }

                // Liabilities
                if (data.liabilities) {
                    context.liabilities = data.liabilities;
                }

                // Liquidity
                context.liquidity = data.liquidity || 0;
            }
        }

    } catch (error) {
        console.error('Error building wealth context:', error);
        context.error = 'Failed to load wealth data';
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

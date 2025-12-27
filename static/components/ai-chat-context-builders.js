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

    try {
        // Load balance data
        const balanceResult = await window.loadBalanceData(true);

        // Get unified state if available
        const unifiedState = window.getUnifiedState ? window.getUnifiedState() : {};

        // Total value and positions
        if (balanceResult.data && balanceResult.data.items) {
            const items = balanceResult.data.items;
            context.total_value = items.reduce((sum, item) => sum + (item.value_usd || 0), 0);
            context.total_positions = items.length;

            // Top 10 crypto positions
            context.positions = items
                .sort((a, b) => (b.value_usd || 0) - (a.value_usd || 0))
                .slice(0, 10)
                .map(item => ({
                    symbol: item.symbol || item.coin,
                    name: item.name || '',
                    value: item.value_usd || 0,
                    weight: (item.value_usd / context.total_value) * 100,
                    pnl_pct: item.pnl_pct || 0
                }));
        }

        // Market regime
        if (unifiedState.regime) {
            context.regime = {
                ccs: unifiedState.regime.ccs_score,
                onchain: unifiedState.regime.onchain_score,
                risk: unifiedState.regime.risk_score,
                total: unifiedState.regime.total_score
            };
        }

        // Decision Index
        if (unifiedState.decision_index) {
            context.decision_index = unifiedState.decision_index;
        }

        // ML Sentiment
        if (unifiedState.ml_sentiment) {
            context.ml_sentiment = unifiedState.ml_sentiment;
        }

    } catch (error) {
        console.error('Error building dashboard context:', error);
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

        // Fetch risk dashboard data
        const response = await fetch('/api/risk/dashboard', {
            headers: { 'X-User': activeUser }
        });

        if (response.ok) {
            const data = await response.json();

            // Risk score
            if (data.risk_score !== undefined) {
                context.risk_score = data.risk_score;
            }

            // Risk metrics
            if (data.metrics) {
                context.var_95 = data.metrics.var_95;
                context.max_drawdown = data.metrics.max_drawdown;
                context.sharpe_ratio = data.metrics.sharpe_ratio;
                context.sortino_ratio = data.metrics.sortino_ratio;
                context.hhi = data.metrics.hhi || data.metrics.concentration_hhi;
                context.volatility = data.metrics.volatility;
            }
        }

        // Active alerts
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
            }
        }

        // Market cycles (if available in global state)
        if (window.lastCyclesData) {
            context.cycles = window.lastCyclesData;
        }

    } catch (error) {
        console.error('Error building risk context:', error);
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

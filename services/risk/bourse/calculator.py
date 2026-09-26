"""
Bourse Risk Calculator - Orchestrates all risk calculations
Main entry point for risk analytics on stock portfolios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from .metrics import (
    calculate_var_historical,
    calculate_var_parametric,
    calculate_var_montecarlo,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_beta,
    calculate_risk_score,
    calculate_calmar_ratio
)
from .data_fetcher import BourseDataFetcher

logger = logging.getLogger(__name__)


class IncompleteBourseMarketData(ValueError):
    """Risk metrics must not be published for an incomplete price universe."""

    def __init__(self, symbols: List[str], coverage: float, reason: Optional[str] = None):
        self.symbols = symbols
        self.coverage = coverage
        super().__init__(reason or f"Market prices unavailable for: {', '.join(symbols)}")


class BourseRiskCalculator:
    """
    Orchestrates risk calculations for bourse portfolios
    Combines multiple risk metrics into comprehensive risk assessment
    """

    def __init__(self, data_source: str = "yahoo"):
        self.data_fetcher = BourseDataFetcher()
        self.data_source = data_source
        self.cache = {}

    async def calculate_portfolio_risk(
        self,
        positions: List[Dict[str, Any]],
        benchmark: str = "SPY",
        lookback_days: int = 252,
        risk_free_rate: float = 0.03,
        var_method: str = "historical",
        cash_amount: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics for a portfolio

        Args:
            positions: List of portfolio positions with ticker, quantity, market_value
            benchmark: Benchmark ticker (default SPY for S&P500)
            lookback_days: Historical lookback period
            risk_free_rate: Annual risk-free rate
            var_method: VaR calculation method ("historical", "parametric", "montecarlo")

        Returns:
            Dictionary with comprehensive risk metrics
        """
        logger.info(f"Calculating portfolio risk for {len(positions)} positions")

        try:
            # Calculate total portfolio value
            invested_value = sum(pos.get('market_value_usd', 0) for pos in positions)
            portfolio_value = invested_value + cash_amount

            if invested_value <= 0 or portfolio_value <= 0:
                raise ValueError("Invested portfolio value is zero")

            # Fetch historical data for all positions
            # Round to start of day for cache consistency (same window all day)
            end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer

            position_data = {}
            unavailable_symbols = []
            covered_value = 0.0
            fx_currencies = set()
            for pos in positions:
                ticker = pos.get('ticker') or pos.get('symbol')
                if not ticker:
                    logger.warning(f"Position missing ticker: {pos}")
                    unavailable_symbols.append('<missing ticker>')
                    continue
                if ticker in position_data:
                    position_data[ticker]['weight'] += pos['market_value_usd'] / portfolio_value
                    covered_value += pos['market_value_usd']
                    continue

                try:
                    prices = await self.data_fetcher.fetch_historical_prices(
                        ticker,
                        start_date,
                        end_date,
                        source=self.data_source,
                        isin=pos.get('isin'),
                    )
                    currency = prices.attrs.get('native_currency') or pos.get('currency') or 'USD'
                    currency = currency.upper()
                    if currency != 'USD':
                        rates = await self.data_fetcher.fetch_historical_fx(
                            currency, start_date - timedelta(days=7), end_date,
                        )
                        prices = self._convert_prices_to_usd(prices, rates)
                        fx_currencies.add(currency)
                    position_data[ticker] = {
                        'prices': prices,
                        'weight': pos['market_value_usd'] / portfolio_value,
                        'position': pos
                    }
                    covered_value += pos['market_value_usd']
                except Exception as e:
                    logger.error(f"Failed to fetch data for {ticker}: {e}")
                    unavailable_symbols.append(ticker)

            if unavailable_symbols or not position_data:
                raise IncompleteBourseMarketData(
                    unavailable_symbols or ['<all positions>'],
                    covered_value / invested_value,
                )

            # Fetch benchmark data
            try:
                benchmark_prices = await self.data_fetcher.fetch_benchmark_prices(
                    benchmark,
                    start_date,
                    end_date
                )
                portfolio_returns = self._calculate_portfolio_returns({
                    **position_data,
                    '__benchmark__': {'prices': benchmark_prices, 'weight': 0.0},
                })
                benchmark_returns = benchmark_prices['close'].pct_change(fill_method=None)
                aligned = pd.concat(
                    [portfolio_returns.rename('portfolio'), benchmark_returns.rename('benchmark')],
                    axis=1,
                    join='inner',
                ).replace([np.inf, -np.inf], np.nan).dropna()
                if len(aligned) < 20:
                    raise ValueError("Insufficient aligned benchmark returns")
            except Exception as e:
                raise IncompleteBourseMarketData([benchmark], 1.0) from e

            # Calculate risk metrics
            risk_metrics = self._calculate_all_metrics(
                aligned['portfolio'].to_numpy(dtype=float),
                aligned['benchmark'].to_numpy(dtype=float),
                portfolio_value,
                risk_free_rate,
                var_method
            )
            risk_metrics['concentration'] = self._calculate_concentration_metrics(
                positions, portfolio_value, cash_amount
            )

            # Add metadata
            risk_metrics['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': portfolio_value,
                'invested_value': invested_value,
                'cash_amount': cash_amount,
                'return_currency': 'USD',
                'historical_fx_currencies': sorted(fx_currencies),
                'fx_max_carry_days': 3,
                'return_alignment': 'matching_start_and_end_dates',
                'price_origin': 'yahoo',
                'observations': len(aligned),
                'price_asof': str(aligned.index.max().date()),
                'positions_count': len(positions),
                'lookback_days': lookback_days,
                'risk_free_rate': risk_free_rate,
                'benchmark': benchmark,
                'var_method': var_method
            }

            logger.info(f"Portfolio risk calculated: Score={risk_metrics['risk_score']['risk_score']}")
            return risk_metrics

        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            raise

    @staticmethod
    def _convert_prices_to_usd(prices: pd.DataFrame, rates: pd.Series) -> pd.DataFrame:
        """Align dated FX without looking ahead; tolerate at most three calendar days."""
        rates = rates.loc[~rates.index.duplicated(keep='last')].sort_index()
        rates = rates.where(np.isfinite(rates) & (rates > 0)).dropna()
        aligned = rates.reindex(prices.index, method='ffill', tolerance=pd.Timedelta(days=3))
        if aligned.isna().any():
            raise ValueError("Historical FX does not cover every stock observation")
        converted = prices.copy()
        for column in ('open', 'high', 'low', 'close', 'adjusted_close'):
            if column in converted:
                converted[column] = converted[column] * aligned
        converted.attrs['native_currency'] = 'USD'
        converted.attrs['historical_fx_applied'] = True
        return converted

    def _calculate_portfolio_returns(
        self,
        position_data: Dict[str, Dict]
    ) -> pd.Series:
        """
        Calculate weighted portfolio returns

        Args:
            position_data: Dictionary of position data with prices and weights

        Returns:
            Array of portfolio returns
        """
        # Compute returns before intersecting dates.  Reindexing prices first can
        # introduce missing values for a ticker and make its return series shorter
        # than the other positions (as seen with partially quoted securities).
        returns_by_ticker = {}
        starts_by_ticker = {}
        common_dates = None
        for ticker, data in position_data.items():
            close_prices = data['prices']['close']
            close_prices = close_prices.loc[
                ~close_prices.index.duplicated(keep='last')
            ].sort_index()
            returns = close_prices.pct_change(fill_method=None).replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            if returns.empty:
                raise ValueError(f"No usable return data for {ticker}")

            returns_by_ticker[ticker] = returns
            starts_by_ticker[ticker] = pd.Series(close_prices.index, index=close_prices.index).shift(1)
            common_dates = (
                returns.index
                if common_dates is None
                else common_dates.intersection(returns.index)
            )

        if common_dates is None or common_dates.empty:
            raise ValueError("No common return dates are available across positions")

        common_dates = common_dates.sort_values()
        # A two-session return after a local holiday must not be paired with
        # another exchange's one-session return carrying the same end date.
        starts = pd.DataFrame({ticker: series.reindex(common_dates)
                               for ticker, series in starts_by_ticker.items()})
        common_dates = common_dates[starts.nunique(axis=1).eq(1).to_numpy()]
        if common_dates.empty:
            raise ValueError("No matching return periods are available across positions")
        portfolio_returns = np.zeros(len(common_dates))
        for ticker, data in position_data.items():
            returns = returns_by_ticker[ticker].reindex(common_dates).to_numpy(dtype=float)
            portfolio_returns += returns * data['weight']

        return pd.Series(portfolio_returns, index=common_dates, name='portfolio')

    def _calculate_all_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        portfolio_value: float,
        risk_free_rate: float,
        var_method: str
    ) -> Dict[str, Any]:
        """
        Calculate all risk metrics

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            portfolio_value: Total portfolio value
            risk_free_rate: Risk-free rate
            var_method: VaR calculation method

        Returns:
            Dictionary with all metrics
        """
        # VaR calculation
        if var_method == "historical":
            var_result = calculate_var_historical(returns, 0.95, portfolio_value)
        elif var_method == "parametric":
            var_result = calculate_var_parametric(returns, 0.95, portfolio_value)
        elif var_method == "montecarlo":
            var_result = calculate_var_montecarlo(returns, 0.95, portfolio_value)
        else:
            raise ValueError(f"Unknown VaR method: {var_method}")

        # Volatility calculations
        vol_30d = calculate_volatility(returns, window=30, annualize=True)
        vol_90d = calculate_volatility(returns, window=90, annualize=True)
        vol_252d = calculate_volatility(returns, window=None, annualize=True)

        # Risk-adjusted returns
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
        sortino = calculate_sortino_ratio(returns, risk_free_rate)

        # Reconstruct prices from returns for drawdown
        prices = (1 + returns).cumprod() * 100  # Start at 100

        # Max drawdown
        dd_metrics = calculate_max_drawdown(prices)

        # Beta
        beta = calculate_beta(returns, benchmark_returns)

        # Calmar ratio
        calmar = calculate_calmar_ratio(returns, prices)

        # Composite risk score
        risk_score_result = calculate_risk_score(
            var_result['var_percentage'],
            vol_252d,
            sharpe,
            dd_metrics['max_drawdown'],
            beta
        )

        # Compile results
        return {
            'risk_score': risk_score_result,
            'traditional_risk': {
                'var_95_1d': var_result['var_percentage'],
                'var_monetary': var_result.get('var_monetary', 0),
                'var_method': var_result['method'],
                'volatility_30d': vol_30d,
                'volatility_90d': vol_90d,
                'volatility_252d': vol_252d,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'max_drawdown': dd_metrics['max_drawdown'],
                'max_drawdown_pct': dd_metrics['max_drawdown_pct'],
                'drawdown_days': dd_metrics['drawdown_days'],
                'beta_portfolio': beta
            },
            'concentration': {},
            'alerts': self._generate_alerts(risk_score_result, var_result, vol_30d, dd_metrics)
        }

    def _calculate_concentration_metrics(
        self, positions: List[Dict[str, Any]], portfolio_value: float, cash_amount: float = 0.0
    ) -> Dict[str, Any]:
        """Calculate weights from selected holdings; unknown classifications stay unknown."""
        by_symbol: Dict[str, float] = {}
        for position in positions:
            symbol = position.get('ticker') or position.get('symbol')
            if not symbol:
                continue
            by_symbol[symbol] = by_symbol.get(symbol, 0.0) + max(
                0.0, float(position.get('market_value_usd') or 0.0)
            )
        weights = sorted((value / portfolio_value for value in by_symbol.values()), reverse=True)
        if cash_amount > 0:
            weights.append(cash_amount / portfolio_value)
        weights.sort(reverse=True)
        hhi = sum(weight ** 2 for weight in weights)
        return {
            'top5_pct': round(sum(weights[:5]) * 100, 2),
            'largest_position_pct': round(max(weights, default=0.0) * 100, 2),
            'herfindahl_index': round(hhi, 4),
            'effective_positions': round(1 / hhi, 2) if hhi > 0 else None,
            'sector_max_pct': None,
            'geography_us_pct': None,
            'classification_coverage': 0.0,
        }

    def _generate_alerts(
        self,
        risk_score: Dict[str, Any],
        var_result: Dict[str, float],
        volatility: float,
        dd_metrics: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate risk alerts based on thresholds

        Args:
            risk_score: Risk score result
            var_result: VaR calculation result
            volatility: Current volatility
            dd_metrics: Drawdown metrics

        Returns:
            List of alert dictionaries
        """
        alerts = []

        # High risk score alert
        if risk_score['risk_score'] < 30:
            alerts.append({
                'severity': 'critical',
                'type': 'risk_score',
                'message': f"Critical risk level: Score {risk_score['risk_score']}/100"
            })
        elif risk_score['risk_score'] < 50:
            alerts.append({
                'severity': 'warning',
                'type': 'risk_score',
                'message': f"High risk level: Score {risk_score['risk_score']}/100"
            })

        # High VaR alert
        if var_result['var_percentage'] < -0.03:  # VaR > 3%
            alerts.append({
                'severity': 'warning',
                'type': 'var',
                'message': f"High Value at Risk: {var_result['var_percentage']*100:.2f}% daily loss possible"
            })

        # High volatility alert
        if volatility > 0.30:  # >30% annualized
            alerts.append({
                'severity': 'warning',
                'type': 'volatility',
                'message': f"High volatility: {volatility*100:.1f}% annualized"
            })

        # Deep drawdown alert
        if dd_metrics['max_drawdown'] < -0.20:  # >20% drawdown
            alerts.append({
                'severity': 'warning',
                'type': 'drawdown',
                'message': f"Significant drawdown: {dd_metrics['max_drawdown_pct']:.1f}%"
            })

        return alerts

    async def calculate_position_level_var(
        self,
        positions: List[Dict[str, Any]],
        lookback_days: int = 252
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate VaR contribution for each position

        Args:
            positions: List of positions
            lookback_days: Historical lookback period

        Returns:
            Dictionary mapping ticker to VaR metrics
        """
        position_vars = {}

        for pos in positions:
            ticker = pos.get('ticker') or pos.get('symbol')
            if not ticker:
                continue

            try:
                # Fetch prices
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days + 30)

                prices = await self.data_fetcher.fetch_historical_prices(
                    ticker,
                    start_date,
                    end_date,
                    source=self.data_source
                )

                # Calculate returns
                returns = self.data_fetcher.calculate_returns(prices)

                # Calculate VaR
                var_result = calculate_var_historical(
                    returns,
                    confidence_level=0.95,
                    portfolio_value=pos['market_value_usd']
                )

                position_vars[ticker] = {
                    'var_percentage': var_result['var_percentage'],
                    'var_monetary': var_result['var_monetary'],
                    'position_value': pos['market_value_usd']
                }

            except Exception as e:
                logger.error(f"Failed to calculate VaR for {ticker}: {e}")

        return position_vars

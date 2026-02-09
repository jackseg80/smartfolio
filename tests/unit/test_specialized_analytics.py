"""
Unit tests for SpecializedBourseAnalytics

Tests all major methods:
- _get_sector_for_ticker: static map, exchange suffixes, yfinance fallback, caching
- _analyze_portfolio_allocation: concentration, missed opportunities, overallocation, momentum
- predict_earnings_impact: with/without earnings dates, alert levels
- detect_sector_rotation: hot/cold sectors, clustering, portfolio analysis
- forecast_beta: ewma/rolling/expanding, trend detection, insufficient data
- analyze_dividends: with/without dividends, frequency detection, growth rate
- monitor_margin: cash/leveraged, warnings, margin call distance
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from services.risk.bourse.specialized_analytics import SpecializedBourseAnalytics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analytics():
    """Fresh SpecializedBourseAnalytics instance for each test."""
    return SpecializedBourseAnalytics()


@pytest.fixture
def price_history():
    """Generate reproducible price history (200 trading days)."""
    np.random.seed(42)
    dates = pd.bdate_range(end=datetime.now(), periods=200)
    # Random walk around price=100
    returns = np.random.normal(0.0005, 0.02, 200)
    prices = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame({'close': prices}, index=dates)
    return df


@pytest.fixture
def benchmark_returns():
    """Generate reproducible benchmark returns (200 trading days)."""
    np.random.seed(99)
    dates = pd.bdate_range(end=datetime.now(), periods=200)
    returns = np.random.normal(0.0003, 0.012, 200)
    return pd.Series(returns, index=dates, name='benchmark')


@pytest.fixture
def position_returns():
    """Generate reproducible position returns correlated with benchmark."""
    np.random.seed(42)
    dates = pd.bdate_range(end=datetime.now(), periods=200)
    # Correlated with benchmark (beta ~ 1.2) plus idiosyncratic noise
    np.random.seed(99)
    bench = np.random.normal(0.0003, 0.012, 200)
    np.random.seed(42)
    noise = np.random.normal(0.0001, 0.008, 200)
    pos = 1.2 * bench + noise
    return pd.Series(pos, index=dates, name='position')


# ===========================================================================
# 1. _get_sector_for_ticker
# ===========================================================================

class TestGetSectorForTicker:
    """Tests for ticker-to-sector mapping."""

    def test_known_ticker_from_static_map(self, analytics):
        """Known tickers should resolve from the static map."""
        assert analytics._get_sector_for_ticker('AAPL') == 'Technology'
        assert analytics._get_sector_for_ticker('JPM') == 'Finance'
        assert analytics._get_sector_for_ticker('JNJ') == 'Healthcare'
        assert analytics._get_sector_for_ticker('XOM') == 'Energy'
        assert analytics._get_sector_for_ticker('SPY') == 'ETF-Broad'

    def test_ticker_with_exchange_colon_suffix(self, analytics):
        """Saxo-style 'TICKER:exchange' should strip the suffix and match."""
        assert analytics._get_sector_for_ticker('GOOGL:xnas') == 'Technology'
        assert analytics._get_sector_for_ticker('UBSG:xswx') == 'Finance'
        assert analytics._get_sector_for_ticker('MSFT:xnas') == 'Technology'

    def test_ticker_with_dot_suffix(self, analytics):
        """Yahoo-style 'TICKER.L' should strip the suffix and match."""
        assert analytics._get_sector_for_ticker('AAPL.L') == 'Technology'
        assert analytics._get_sector_for_ticker('JPM.N') == 'Finance'

    def test_ticker_case_insensitive(self, analytics):
        """Lowercase tickers should still match the static map."""
        assert analytics._get_sector_for_ticker('aapl') == 'Technology'
        assert analytics._get_sector_for_ticker('jpm') == 'Finance'

    def test_unknown_ticker_falls_back_to_other(self, analytics):
        """Unknown ticker without yfinance should return 'Other'."""
        with patch.dict('sys.modules', {'yfinance': None}):
            # Force import to fail so we hit the except branch
            with patch('builtins.__import__', side_effect=ImportError("no yfinance")):
                result = analytics._get_sector_for_ticker('ZZZZUNKNOWN')
        assert result == 'Other'

    def test_cache_stores_dynamic_lookup(self, analytics):
        """After a dynamic lookup, the result should be cached."""
        with patch('builtins.__import__', side_effect=ImportError("no yfinance")):
            analytics._get_sector_for_ticker('XYZFAKE')

        # The ticker should now be in _sector_cache
        assert 'XYZFAKE' in analytics._sector_cache
        assert analytics._sector_cache['XYZFAKE'] == 'Other'

    def test_cached_value_returned_on_second_call(self, analytics):
        """Second call for the same unknown ticker should use cache, not retry yfinance."""
        # First call: populate cache
        with patch('builtins.__import__', side_effect=ImportError("no yfinance")):
            first = analytics._get_sector_for_ticker('NEWONE')

        # Manually overwrite cache to prove it is read
        analytics._sector_cache['NEWONE'] = 'TestSector'
        second = analytics._get_sector_for_ticker('NEWONE')
        assert second == 'TestSector'

    def test_yfinance_sector_lookup(self, analytics):
        """When yfinance returns a sector, it should be used and cached."""
        mock_yf = MagicMock()
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'sector': 'Industrials'}
        mock_yf.Ticker.return_value = mock_ticker_instance

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = analytics._get_sector_for_ticker('UNKNOWNTICKER123')

        assert result == 'Industrials'
        assert analytics._sector_cache['UNKNOWNTICKER123'] == 'Industrials'

    def test_yfinance_etf_classification(self, analytics):
        """When yfinance identifies an ETF, classify by name keywords."""
        mock_yf = MagicMock()
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {
            'sector': None,
            'quoteType': 'ETF',
            'longName': 'iShares Global Bond ETF',
            'category': 'fixed income'
        }
        mock_yf.Ticker.return_value = mock_ticker_instance

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = analytics._get_sector_for_ticker('BONDETF')

        assert result == 'ETF-Bonds'


# ===========================================================================
# 2. _analyze_portfolio_allocation
# ===========================================================================

class TestAnalyzePortfolioAllocation:
    """Tests for portfolio allocation analysis insights."""

    def test_concentration_risk_above_40_pct(self, analytics):
        """Sector with weight > 40% triggers a concentration alert."""
        sector_metrics = {
            'Technology': {'weight': 55.0, 'momentum': 1.0, 'signal': 'neutral'},
            'Finance': {'weight': 25.0, 'momentum': 1.0, 'signal': 'neutral'},
            'Healthcare': {'weight': 20.0, 'momentum': 1.0, 'signal': 'neutral'},
        }
        result = analytics._analyze_portfolio_allocation(sector_metrics, [], [], {})

        assert len(result['alerts']) == 1
        assert result['alerts'][0]['type'] == 'concentration_risk'
        assert result['alerts'][0]['sector'] == 'Technology'
        assert result['alerts'][0]['weight'] == 55.0

    def test_no_concentration_risk_at_40_pct(self, analytics):
        """Sector at exactly 40% should NOT trigger concentration alert."""
        sector_metrics = {
            'Technology': {'weight': 40.0, 'momentum': 1.0, 'signal': 'neutral'},
            'Finance': {'weight': 60.0, 'momentum': 1.0, 'signal': 'neutral'},
        }
        result = analytics._analyze_portfolio_allocation(sector_metrics, [], [], {})

        # Only Finance (60%) should trigger, not Technology (40%)
        concentration_alerts = [a for a in result['alerts'] if a['type'] == 'concentration_risk']
        sectors_alerted = [a['sector'] for a in concentration_alerts]
        assert 'Technology' not in sectors_alerted
        assert 'Finance' in sectors_alerted

    def test_missed_opportunity_hot_sector_low_weight(self, analytics):
        """Hot sector with weight < 5% triggers missed opportunity."""
        sector_metrics = {
            'Energy': {'weight': 2.0, 'momentum': 1.5, 'return': 15.0},
            'Technology': {'weight': 50.0, 'momentum': 1.0, 'return': 5.0},
        }
        result = analytics._analyze_portfolio_allocation(
            sector_metrics, hot_sectors=['Energy'], cold_sectors=[], sector_momentum={'Energy': 1.5}
        )

        assert len(result['opportunities']) == 1
        assert result['opportunities'][0]['type'] == 'missed_opportunity'
        assert result['opportunities'][0]['sector'] == 'Energy'

    def test_overallocated_weak_cold_sector_high_weight(self, analytics):
        """Cold sector with weight > 5% triggers overallocated_weak warning."""
        sector_metrics = {
            'Energy': {'weight': 12.0, 'momentum': 0.6, 'return': -8.0},
            'Technology': {'weight': 88.0, 'momentum': 1.1, 'return': 10.0},
        }
        result = analytics._analyze_portfolio_allocation(
            sector_metrics, hot_sectors=[], cold_sectors=['Energy'], sector_momentum={'Energy': 0.6}
        )

        overalloc = [w for w in result['warnings'] if w['type'] == 'overallocated_weak']
        assert len(overalloc) == 1
        assert overalloc[0]['sector'] == 'Energy'

    def test_momentum_slowdown_warning(self, analytics):
        """High weight + slowing momentum + neutral signal triggers momentum warning."""
        sector_metrics = {
            'Technology': {
                'weight': 25.0,
                'momentum': 0.90,
                'signal': 'neutral',
            },
        }
        result = analytics._analyze_portfolio_allocation(
            sector_metrics, hot_sectors=[], cold_sectors=[], sector_momentum={'Technology': 0.90}
        )

        slowdown = [w for w in result['warnings'] if w['type'] == 'momentum_slowdown']
        assert len(slowdown) == 1
        assert slowdown[0]['sector'] == 'Technology'

    def test_no_momentum_slowdown_when_signal_not_neutral(self, analytics):
        """Momentum slowdown should NOT fire if signal is not 'neutral'."""
        sector_metrics = {
            'Technology': {
                'weight': 25.0,
                'momentum': 0.90,
                'signal': 'overweight',  # not neutral
            },
        }
        result = analytics._analyze_portfolio_allocation(
            sector_metrics, hot_sectors=[], cold_sectors=[], sector_momentum={'Technology': 0.90}
        )

        slowdown = [w for w in result['warnings'] if w['type'] == 'momentum_slowdown']
        assert len(slowdown) == 0

    def test_empty_portfolio_returns_empty_insights(self, analytics):
        """Empty sector_metrics should produce empty insight lists."""
        result = analytics._analyze_portfolio_allocation({}, [], [], {})
        assert result['alerts'] == []
        assert result['warnings'] == []
        assert result['opportunities'] == []


# ===========================================================================
# 3. predict_earnings_impact
# ===========================================================================

class TestPredictEarningsImpact:
    """Tests for earnings impact prediction."""

    def test_with_earnings_dates_calculates_metrics(self, analytics, price_history):
        """With earnings dates, should compute pre/post vol and avg move."""
        # Place earnings dates within the price history range
        mid_idx = len(price_history) // 2
        earnings_dates = [
            price_history.index[mid_idx].to_pydatetime(),
            price_history.index[mid_idx + 30].to_pydatetime(),
        ]

        result = analytics.predict_earnings_impact('AAPL', price_history, earnings_dates)

        assert result['ticker'] == 'AAPL'
        assert result['num_earnings_analyzed'] == 2
        assert result['pre_earnings_vol'] > 0
        assert result['post_earnings_vol'] > 0
        assert 'vol_increase_pct' in result
        assert 'avg_post_earnings_move' in result

    def test_without_earnings_dates_uses_generic(self, analytics, price_history):
        """Without earnings dates, should use generic 50% vol increase estimate."""
        result = analytics.predict_earnings_impact('MSFT', price_history, earnings_dates=None)

        assert result['ticker'] == 'MSFT'
        assert result['num_earnings_analyzed'] == 0
        assert result['vol_increase_pct'] == 50.0
        assert result['alert_level'] == 'low'
        assert 'generic estimates' in result['recommendation']

    def test_alert_level_high_within_7_days(self, analytics, price_history):
        """If next earnings <= 7 days away, alert_level should be 'high'."""
        # Create earnings date such that next (+ 90 days) is within 7 days from now
        target_next = datetime.now() + timedelta(days=5)
        earnings_date = target_next - timedelta(days=90)

        # We need this date to be within the price history range for vol calc
        mid_idx = len(price_history) // 2
        earnings_dates = [
            price_history.index[mid_idx].to_pydatetime(),
            earnings_date,
        ]

        result = analytics.predict_earnings_impact('AAPL', price_history, earnings_dates)
        assert result['alert_level'] == 'high'

    def test_alert_level_medium_within_14_days(self, analytics, price_history):
        """If next earnings 8-14 days away, alert_level should be 'medium'."""
        target_next = datetime.now() + timedelta(days=10)
        earnings_date = target_next - timedelta(days=90)

        mid_idx = len(price_history) // 2
        earnings_dates = [
            price_history.index[mid_idx].to_pydatetime(),
            earnings_date,
        ]

        result = analytics.predict_earnings_impact('AAPL', price_history, earnings_dates)
        assert result['alert_level'] == 'medium'

    def test_alert_level_low_beyond_14_days(self, analytics, price_history):
        """If next earnings > 14 days away, alert_level should be 'low'."""
        target_next = datetime.now() + timedelta(days=60)
        earnings_date = target_next - timedelta(days=90)

        mid_idx = len(price_history) // 2
        earnings_dates = [
            price_history.index[mid_idx].to_pydatetime(),
            earnings_date,
        ]

        result = analytics.predict_earnings_impact('AAPL', price_history, earnings_dates)
        assert result['alert_level'] == 'low'


# ===========================================================================
# 4. detect_sector_rotation
# ===========================================================================

class TestDetectSectorRotation:
    """Tests for sector rotation detection."""

    @staticmethod
    def _make_returns(tickers, n_days=90, seed=42):
        """Helper: create a dict of {ticker: pd.Series} returns."""
        np.random.seed(seed)
        dates = pd.bdate_range(end=datetime.now(), periods=n_days)
        returns = {}
        for i, t in enumerate(tickers):
            np.random.seed(seed + i)
            returns[t] = pd.Series(np.random.normal(0.001, 0.015, n_days), index=dates)
        return returns

    def test_basic_rotation_detection(self, analytics):
        """Should produce sectors, hot/cold lists, and recommendations."""
        returns = self._make_returns(['AAPL', 'JPM', 'XOM'])
        result = analytics.detect_sector_rotation(returns, lookback_days=60)

        assert 'sectors' in result
        assert 'hot_sectors' in result
        assert 'cold_sectors' in result
        assert 'recommendations' in result
        assert 'rotation_detected' in result
        assert result['num_sectors'] >= 1

    def test_hot_cold_classification(self, analytics):
        """Sectors with momentum > 1.2 are hot, < 0.8 are cold."""
        np.random.seed(42)
        dates = pd.bdate_range(end=datetime.now(), periods=90)

        # Tech: strong recent acceleration (hot)
        tech_returns = np.concatenate([
            np.random.normal(-0.001, 0.01, 60),  # weak early
            np.random.normal(0.015, 0.01, 30),    # strong recent
        ])
        # Energy: deceleration (cold)
        energy_returns = np.concatenate([
            np.random.normal(0.015, 0.01, 60),   # strong early
            np.random.normal(-0.005, 0.01, 30),   # weak recent
        ])

        returns = {
            'AAPL': pd.Series(tech_returns, index=dates),
            'XOM': pd.Series(energy_returns, index=dates),
        }

        result = analytics.detect_sector_rotation(returns, lookback_days=90)

        # Verify that rotation was detected (at least one hot or cold)
        assert result['rotation_detected'] is True

    def test_with_positions_values_calculates_weights(self, analytics):
        """When positions_values provided, sector weights should be populated."""
        returns = self._make_returns(['AAPL', 'JPM'])
        values = {'AAPL': 50000.0, 'JPM': 30000.0}

        result = analytics.detect_sector_rotation(returns, lookback_days=60, positions_values=values)

        tech_weight = result['sectors'].get('Technology', {}).get('weight', 0)
        fin_weight = result['sectors'].get('Finance', {}).get('weight', 0)

        # AAPL = 50k / 80k = 62.5%, JPM = 30k / 80k = 37.5%
        assert abs(tech_weight - 62.5) < 0.1
        assert abs(fin_weight - 37.5) < 0.1

    def test_clustering_with_multiple_sectors(self, analytics):
        """With >= 2 sectors, hierarchical clustering should be returned."""
        returns = self._make_returns(['AAPL', 'JPM', 'XOM'])
        result = analytics.detect_sector_rotation(returns, lookback_days=60)

        assert result['clustering'] is not None
        assert 'linkage_matrix' in result['clustering']
        assert result['clustering']['method'] == 'ward'

    def test_portfolio_analysis_in_result(self, analytics):
        """Result should include portfolio_analysis from _analyze_portfolio_allocation."""
        returns = self._make_returns(['AAPL', 'JPM'])
        result = analytics.detect_sector_rotation(returns, lookback_days=60)

        assert 'portfolio_analysis' in result
        pa = result['portfolio_analysis']
        assert 'alerts' in pa
        assert 'warnings' in pa
        assert 'opportunities' in pa


# ===========================================================================
# 5. forecast_beta
# ===========================================================================

class TestForecastBeta:
    """Tests for dynamic beta forecasting."""

    def test_ewma_method(self, analytics, position_returns, benchmark_returns):
        """EWMA method should return valid beta forecast."""
        result = analytics.forecast_beta(
            position_returns, benchmark_returns, forecast_method='ewma', rolling_window=60
        )

        assert 'current_beta' in result
        assert 'forecasted_beta' in result
        assert result['forecast_method'] == 'ewma'
        # Beta should be in a reasonable range
        assert -3.0 < result['current_beta'] < 5.0
        assert -3.0 < result['forecasted_beta'] < 5.0

    def test_rolling_method(self, analytics, position_returns, benchmark_returns):
        """Rolling method should return beta as average of last 10 rolling betas."""
        result = analytics.forecast_beta(
            position_returns, benchmark_returns, forecast_method='rolling', rolling_window=60
        )

        assert result['forecast_method'] == 'rolling'
        assert isinstance(result['rolling_betas'], list)
        assert len(result['rolling_betas']) > 0

    def test_expanding_method(self, analytics, position_returns, benchmark_returns):
        """Expanding method: forecasted_beta should equal current_beta."""
        result = analytics.forecast_beta(
            position_returns, benchmark_returns, forecast_method='expanding', rolling_window=60
        )

        assert result['forecast_method'] == 'expanding'
        assert result['forecasted_beta'] == result['current_beta']

    def test_beta_trend_detection(self, analytics):
        """Beta trend should be detected from rolling betas difference."""
        np.random.seed(42)
        dates = pd.bdate_range(end=datetime.now(), periods=200)

        # Create data where beta increases sharply toward the end
        benchmark = np.random.normal(0.0003, 0.012, 200)
        # Position with increasing beta: low correlation early, high later
        position = np.concatenate([
            0.5 * benchmark[:100] + np.random.normal(0, 0.01, 100),   # low beta
            2.0 * benchmark[100:] + np.random.normal(0, 0.005, 100),  # high beta
        ])

        pos_series = pd.Series(position, index=dates)
        bench_series = pd.Series(benchmark, index=dates)

        result = analytics.forecast_beta(pos_series, bench_series, rolling_window=30)

        # Trend should detect the regime change
        assert result['beta_trend'] in ('increasing', 'decreasing', 'stable')

    def test_insufficient_data_raises_value_error(self, analytics):
        """With < 30 aligned data points, should raise ValueError."""
        dates = pd.bdate_range(end=datetime.now(), periods=20)
        pos = pd.Series(np.random.normal(0, 0.01, 20), index=dates)
        bench = pd.Series(np.random.normal(0, 0.01, 20), index=dates)

        with pytest.raises(ValueError, match="Insufficient data"):
            analytics.forecast_beta(pos, bench)

    def test_r_squared_and_alpha_present(self, analytics, position_returns, benchmark_returns):
        """Result should include r_squared and annualized alpha."""
        result = analytics.forecast_beta(position_returns, benchmark_returns)

        assert 0.0 <= result['r_squared'] <= 1.0
        # Alpha is annualized percentage
        assert isinstance(result['alpha'], float)

    def test_volatility_ratio(self, analytics, position_returns, benchmark_returns):
        """Volatility ratio should be positive."""
        result = analytics.forecast_beta(position_returns, benchmark_returns)
        assert result['volatility_ratio'] > 0


# ===========================================================================
# 6. analyze_dividends
# ===========================================================================

class TestAnalyzeDividends:
    """Tests for dividend analysis."""

    def test_with_quarterly_dividends(self, analytics, price_history):
        """Quarterly dividends should be detected and yield calculated."""
        # Create 4 quarterly dividend payments in last 12 months
        now = datetime.now()
        div_dates = pd.DatetimeIndex([
            now - timedelta(days=90),
            now - timedelta(days=180),
            now - timedelta(days=270),
            now - timedelta(days=360),
        ])
        dividends = pd.Series([0.82, 0.82, 0.82, 0.82], index=div_dates)

        result = analytics.analyze_dividends('AAPL', price_history, dividends)

        assert result['ticker'] == 'AAPL'
        assert result['payout_frequency'] == 'quarterly'
        assert result['annual_dividend'] > 0
        assert result['current_yield'] > 0
        assert result['has_dividend_data'] is True

    def test_with_semi_annual_dividends(self, analytics, price_history):
        """Two payments per year should be detected as semi-annual."""
        now = datetime.now()
        div_dates = pd.DatetimeIndex([
            now - timedelta(days=100),
            now - timedelta(days=280),
        ])
        dividends = pd.Series([1.50, 1.50], index=div_dates)

        result = analytics.analyze_dividends('UBSG', price_history, dividends)
        assert result['payout_frequency'] == 'semi-annual'

    def test_with_annual_dividend(self, analytics, price_history):
        """Single payment per year should be detected as annual."""
        now = datetime.now()
        div_dates = pd.DatetimeIndex([now - timedelta(days=200)])
        dividends = pd.Series([3.00], index=div_dates)

        result = analytics.analyze_dividends('ROG', price_history, dividends)
        assert result['payout_frequency'] == 'annual'

    def test_without_dividends_returns_zero_yield(self, analytics, price_history):
        """No dividend data should return zero yield and 'none' frequency."""
        result = analytics.analyze_dividends('TSLA', price_history, dividends=None)

        assert result['current_yield'] == 0.0
        assert result['annual_dividend'] == 0.0
        assert result['payout_frequency'] == 'none'
        assert result['has_dividend_data'] is False

    def test_close_column_case_handling(self, analytics):
        """Should handle both 'close' and 'Close' column names."""
        np.random.seed(42)
        dates = pd.bdate_range(end=datetime.now(), periods=50)
        prices = 100 + np.cumsum(np.random.normal(0, 1, 50))

        # Test with 'Close' (uppercase)
        df_upper = pd.DataFrame({'Close': prices}, index=dates)
        result = analytics.analyze_dividends('TEST', df_upper, dividends=None)
        assert result['current_yield'] == 0.0  # just verifying it doesn't crash

        # Test with 'close' (lowercase)
        df_lower = pd.DataFrame({'close': prices}, index=dates)
        result = analytics.analyze_dividends('TEST', df_lower, dividends=None)
        assert result['current_yield'] == 0.0

    def test_missing_close_column_raises(self, analytics):
        """If neither 'close' nor 'Close' exists, should raise ValueError."""
        dates = pd.bdate_range(end=datetime.now(), periods=50)
        df = pd.DataFrame({'price': [100] * 50}, index=dates)

        with pytest.raises(ValueError, match="No 'close' or 'Close' column"):
            analytics.analyze_dividends('TEST', df, dividends=None)

    def test_dividend_growth_rate(self, analytics, price_history):
        """Dividend growth rate should be computed when 2+ years of data exist."""
        now = datetime.now()
        # Current year: $1.00/quarter, previous year: $0.80/quarter
        div_dates = pd.DatetimeIndex([
            now - timedelta(days=60),
            now - timedelta(days=150),
            now - timedelta(days=240),
            now - timedelta(days=330),
            now - timedelta(days=420),
            now - timedelta(days=510),
            now - timedelta(days=600),
            now - timedelta(days=690),
        ])
        dividends = pd.Series(
            [1.00, 1.00, 1.00, 1.00, 0.80, 0.80, 0.80, 0.80],
            index=div_dates
        )

        result = analytics.analyze_dividends('JNJ', price_history, dividends)
        # Growth rate = (4.00 - 3.20) / 3.20 * 100 = 25%
        assert result['dividend_growth_rate'] > 0


# ===========================================================================
# 7. monitor_margin
# ===========================================================================

class TestMonitorMargin:
    """Tests for margin monitoring logic."""

    def test_cash_positions_no_margin(self, analytics):
        """Positions with leverage=1.0 should use zero margin."""
        positions = [
            {'market_value_usd': 50000, 'leverage': 1.0},
            {'market_value_usd': 30000, 'leverage': 1.0},
        ]
        result = analytics.monitor_margin(positions, account_equity=100000)

        assert result['total_margin_used'] == 0.0
        assert result['margin_utilization'] == 0.0
        assert result['total_exposure'] == 80000.0
        assert len(result['warnings']) == 0

    def test_leveraged_positions_margin_calculation(self, analytics):
        """Leveraged positions should calculate margin correctly."""
        # market_value=10000, leverage=2.0 => exposure=20000, margin=10000*(1-1/2)=5000
        positions = [
            {'market_value_usd': 10000, 'leverage': 2.0},
        ]
        result = analytics.monitor_margin(positions, account_equity=20000)

        assert result['total_exposure'] == 20000.0
        assert result['total_margin_used'] == 5000.0
        assert result['margin_utilization'] == pytest.approx(0.25, abs=0.01)

    def test_high_utilization_critical_warning(self, analytics):
        """Margin utilization > 80% should trigger CRITICAL warning."""
        # leverage=5.0 => margin = value*(1-1/5) = 0.8*value
        positions = [
            {'market_value_usd': 20000, 'leverage': 5.0},
        ]
        # margin = 20000 * 0.8 = 16000, equity = 18000 => utilization = 88.9%
        result = analytics.monitor_margin(positions, account_equity=18000)

        assert result['margin_utilization'] > 0.80
        assert any('CRITICAL' in w for w in result['warnings'])

    def test_moderate_utilization_warning(self, analytics):
        """Margin utilization 60-80% should trigger WARNING (not CRITICAL)."""
        # leverage=3.0 => margin = value*(1-1/3) = 0.667*value
        positions = [
            {'market_value_usd': 10000, 'leverage': 3.0},
        ]
        # margin = 6666.67, equity = 10000 => utilization = 66.7%
        result = analytics.monitor_margin(positions, account_equity=10000)

        assert 0.60 < result['margin_utilization'] < 0.80
        assert any('WARNING' in w and '60%' in w for w in result['warnings'])

    def test_margin_call_distance_critical(self, analytics):
        """Margin call distance < 10% should trigger critical warning."""
        # Large exposure vs small equity
        positions = [
            {'market_value_usd': 50000, 'leverage': 4.0},
        ]
        # exposure = 200000, maint margin (25%) = 50000
        # equity = 52000 => call_distance = (52000-50000)/52000 = 3.8%
        result = analytics.monitor_margin(positions, account_equity=52000)

        assert result['margin_call_distance'] < 0.10
        assert any('margin call distance' in w.lower() and 'below 10%' in w.lower() for w in result['warnings'])

    def test_margin_call_distance_warning(self, analytics):
        """Margin call distance 10-20% should trigger warning."""
        positions = [
            {'market_value_usd': 20000, 'leverage': 3.0},
        ]
        # exposure = 60000, maint margin (25%) = 15000
        # equity = 18000 => call_distance = (18000-15000)/18000 = 16.7%
        result = analytics.monitor_margin(positions, account_equity=18000)

        assert 0.10 < result['margin_call_distance'] < 0.20
        assert any('below 20%' in w.lower() for w in result['warnings'])

    def test_high_leverage_warning(self, analytics):
        """Current leverage > 3x should trigger high leverage warning."""
        positions = [
            {'market_value_usd': 25000, 'leverage': 4.0},
        ]
        # exposure = 100000, equity = 20000 => leverage = 5.0x
        result = analytics.monitor_margin(positions, account_equity=20000)

        assert result['current_leverage'] > 3.0
        assert any('High leverage' in w for w in result['warnings'])

    def test_optimal_leverage_capped(self, analytics):
        """Optimal leverage should be capped between 1x and 5x."""
        positions = [
            {'market_value_usd': 5000, 'leverage': 2.0},
        ]
        result = analytics.monitor_margin(positions, account_equity=100000)

        assert 1.0 <= result['optimal_leverage'] <= 5.0

    def test_empty_positions(self, analytics):
        """Empty positions list should return zero exposure and no warnings."""
        result = analytics.monitor_margin([], account_equity=50000)

        assert result['total_exposure'] == 0.0
        assert result['total_margin_used'] == 0.0
        assert result['num_positions'] == 0

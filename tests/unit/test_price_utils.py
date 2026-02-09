"""
Comprehensive tests for services/price_utils.py

Tests cover all 6 public functions with 30+ test cases.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from services.price_utils import (
    price_history_to_series,
    price_history_to_dataframe,
    validate_price_data,
    validate_price_data_integrity,
    resample_to_daily,
    calculate_returns_dataframe,
)


def _ts(dt):
    return int(dt.timestamp())


def _make_price_history(n=60, start_price=100.0, daily_return=0.01, start_date=None):
    if start_date is None:
        start_date = datetime(2025, 1, 1)
    history = []
    price = start_price
    for i in range(n):
        dt = start_date + timedelta(days=i)
        history.append((_ts(dt), round(price, 4)))
        price *= 1 + daily_return * (1 + 0.3 * np.sin(i))
    return history


def _make_df(n=60, cols=None, start_price=100.0):
    if cols is None:
        cols = ["BTC", "ETH"]
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    data = {}
    for c in cols:
        prices = start_price * np.cumprod(1 + rng.normal(0.001, 0.02, n))
        data[c] = prices
    return pd.DataFrame(data, index=dates)


class TestPriceHistoryToSeries:
    def test_empty_list_returns_empty_series(self):
        s = price_history_to_series([], "BTC")
        assert isinstance(s, pd.Series)
        assert s.empty
        assert s.name == "BTC"
        assert s.dtype == float

    def test_single_element(self):
        ts = _ts(datetime(2025, 6, 1))
        s = price_history_to_series([(ts, 50000.0)], "BTC")
        assert len(s) == 1
        assert s.iloc[0] == 50000.0
        assert s.name == "BTC"

    def test_normal_list_sorted_index(self):
        dt1 = datetime(2025, 1, 1)
        dt2 = datetime(2025, 1, 3)
        dt3 = datetime(2025, 1, 2)
        history = [(_ts(dt2), 200.0), (_ts(dt1), 100.0), (_ts(dt3), 150.0)]
        s = price_history_to_series(history, "ETH")
        assert list(s.values) == [100.0, 150.0, 200.0]
        assert s.index.is_monotonic_increasing

    def test_duplicates_keep_last(self):
        dt = datetime(2025, 3, 15)
        ts = _ts(dt)
        history = [(ts, 100.0), (ts, 200.0)]
        s = price_history_to_series(history, "SOL")
        assert len(s) == 1
        assert s.iloc[0] == 200.0

    def test_name_preserved(self):
        history = _make_price_history(5)
        s = price_history_to_series(history, "DOGE")
        assert s.name == "DOGE"

    def test_datetime_index_type(self):
        history = _make_price_history(3)
        s = price_history_to_series(history, "BTC")
        assert isinstance(s.index, pd.DatetimeIndex)

    def test_prices_converted_to_float(self):
        dt = datetime(2025, 1, 1)
        history = [(_ts(dt), 42000)]
        s = price_history_to_series(history, "BTC")
        assert isinstance(s.iloc[0], float)


class TestPriceHistoryToDataframe:
    def test_empty_dict_returns_empty_dataframe(self):
        df = price_history_to_dataframe({})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_single_asset(self):
        history = _make_price_history(10)
        df = price_history_to_dataframe({"BTC": history})
        assert "BTC" in df.columns
        assert len(df) == 10

    def test_multiple_assets(self):
        h1 = _make_price_history(10, start_price=100)
        h2 = _make_price_history(10, start_price=3000)
        df = price_history_to_dataframe({"BTC": h1, "ETH": h2})
        assert set(df.columns) == {"BTC", "ETH"}

    def test_skips_empty_histories(self):
        h1 = _make_price_history(10)
        df = price_history_to_dataframe({"BTC": h1, "ETH": []})
        assert "BTC" in df.columns
        assert len(df.columns) == 1

    def test_all_empty_histories_returns_empty(self):
        df = price_history_to_dataframe({"BTC": [], "ETH": []})
        assert df.empty

    def test_forward_fill_applied(self):
        base = datetime(2025, 1, 1)
        h1 = [(_ts(base + timedelta(days=i)), 100.0 + i) for i in range(5)]
        h2 = [(_ts(base + timedelta(days=i)), 200.0 + i) for i in range(2, 5)]
        df = price_history_to_dataframe({"BTC": h1, "ETH": h2})
        assert not df.isnull().any().any()

    def test_dropna_removes_leading_nans(self):
        base = datetime(2025, 1, 1)
        h1 = [(_ts(base + timedelta(days=i)), 100.0) for i in range(5)]
        h2 = [(_ts(base + timedelta(days=i)), 200.0) for i in range(3, 5)]
        df = price_history_to_dataframe({"A": h1, "B": h2})
        assert len(df) <= 5
        assert not df.isnull().any().any()


class TestValidatePriceData:
    def test_empty_dataframe_returns_false(self):
        assert validate_price_data(pd.DataFrame()) is False

    def test_insufficient_days_returns_false(self):
        df = _make_df(n=10)
        assert validate_price_data(df, min_days=30) is False

    def test_sufficient_days_returns_true(self):
        df = _make_df(n=60)
        assert validate_price_data(df, min_days=30) is True

    def test_custom_min_days(self):
        df = _make_df(n=5)
        assert validate_price_data(df, min_days=5) is True
        assert validate_price_data(df, min_days=6) is False

    def test_no_columns_returns_false(self):
        df = pd.DataFrame(index=pd.date_range("2025-01-01", periods=60))
        assert validate_price_data(df) is False

    def test_too_many_nans_returns_false(self):
        df = _make_df(n=60)
        total_cells = df.shape[0] * df.shape[1]
        nan_count = int(total_cells * 0.25)
        rng = np.random.default_rng(0)
        for _ in range(nan_count):
            r = rng.integers(0, df.shape[0])
            c = rng.integers(0, df.shape[1])
            df.iloc[r, c] = np.nan
        assert validate_price_data(df) is False

    def test_exactly_at_nan_threshold_passes(self):
        df = _make_df(n=60, cols=["BTC"])
        nan_count = int(len(df) * 0.20)
        df.iloc[:nan_count, 0] = np.nan
        assert validate_price_data(df) is True

    def test_just_over_nan_threshold_fails(self):
        df = _make_df(n=100, cols=["BTC"])
        nan_count = int(len(df) * 0.21)
        df.iloc[:nan_count, 0] = np.nan
        assert validate_price_data(df) is False


class TestValidatePriceDataIntegrity:
    def test_empty_df_returns_invalid(self):
        result = validate_price_data_integrity(pd.DataFrame())
        assert result["valid"] is False
        assert "Empty DataFrame" in result["anomalies"]

    def test_valid_data_passes(self):
        df = _make_df(n=100)
        result = validate_price_data_integrity(df)
        assert result["valid"] is True
        assert result["anomalies"] == []
        assert result["flagged_assets"] == []
        assert len(result["volatility_per_asset"]) == 2

    def test_zero_prices_detected(self):
        df = _make_df(n=60)
        df.iloc[10, 0] = 0.0
        result = validate_price_data_integrity(df)
        assert result["valid"] is False
        assert any("zero/negative" in a for a in result["anomalies"])

    def test_negative_prices_detected(self):
        df = _make_df(n=60)
        df.iloc[10, 0] = -5.0
        result = validate_price_data_integrity(df)
        assert result["valid"] is False
        assert any("zero/negative" in a for a in result["anomalies"])

    def test_flat_prices_detected(self):
        dates = pd.date_range("2025-01-01", periods=60, freq="D")
        df = pd.DataFrame({"FLAT": [100.0] * 60}, index=dates)
        result = validate_price_data_integrity(df)
        assert result["valid"] is False
        assert "FLAT" in result["flagged_assets"]
        assert any("identical" in a for a in result["anomalies"])

    def test_extreme_daily_change_detected(self):
        df = _make_df(n=60)
        df.iloc[30, 0] = df.iloc[29, 0] * 3.0
        result = validate_price_data_integrity(df, max_daily_change=0.99)
        assert result["valid"] is False
        assert any("extreme" in a.lower() for a in result["anomalies"])

    def test_low_volatility_detected(self):
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        prices = 100.0 + np.linspace(0, 0.01, 100)
        df = pd.DataFrame({"STABLE": prices}, index=dates)
        result = validate_price_data_integrity(df, min_volatility_threshold=0.05)
        assert result["valid"] is False
        assert any("Volatility" in a for a in result["anomalies"])

    def test_reject_on_anomaly_raises(self):
        df = _make_df(n=60)
        df.iloc[10, 0] = 0.0
        with pytest.raises(ValueError, match="integrity check failed"):
            validate_price_data_integrity(df, reject_on_anomaly=True)

    def test_reject_on_anomaly_false_does_not_raise(self):
        df = _make_df(n=60)
        df.iloc[10, 0] = 0.0
        result = validate_price_data_integrity(df, reject_on_anomaly=False)
        assert result["valid"] is False

    def test_volatility_per_asset_populated(self):
        df = _make_df(n=100, cols=["BTC", "ETH", "SOL"])
        result = validate_price_data_integrity(df)
        assert set(result["volatility_per_asset"].keys()) == {"BTC", "ETH", "SOL"}
        for v in result["volatility_per_asset"].values():
            assert isinstance(v, float)

    def test_returns_dict_structure(self):
        df = _make_df(n=60)
        result = validate_price_data_integrity(df)
        assert "valid" in result
        assert "anomalies" in result
        assert "flagged_assets" in result
        assert "volatility_per_asset" in result

    def test_single_row_df_no_valid_returns(self):
        dates = pd.date_range("2025-01-01", periods=1, freq="D")
        df = pd.DataFrame({"BTC": [100.0]}, index=dates)
        result = validate_price_data_integrity(df)
        assert result["valid"] is False
        assert "No valid returns" in result["anomalies"][0]

    def test_multiple_anomalies_per_asset(self):
        dates = pd.date_range("2025-01-01", periods=60, freq="D")
        prices = [100.0] * 60
        prices[5] = 0.0
        df = pd.DataFrame({"BAD": prices}, index=dates)
        result = validate_price_data_integrity(df)
        assert result["valid"] is False
        assert len(result["anomalies"]) >= 2

    def test_custom_volatility_threshold(self):
        df = _make_df(n=100)
        result = validate_price_data_integrity(df, min_volatility_threshold=0.001)
        assert result["valid"] is True

    def test_custom_max_daily_change(self):
        df = _make_df(n=200)
        result = validate_price_data_integrity(df, max_daily_change=0.01)
        assert isinstance(result["valid"], bool)


class TestResampleToDaily:
    def test_empty_df_returns_as_is(self):
        df = pd.DataFrame()
        result = resample_to_daily(df)
        assert result.empty

    def test_already_daily_preserves_data(self):
        df = _make_df(n=30)
        result = resample_to_daily(df)
        assert len(result) >= 30

    def test_hourly_to_daily(self):
        dates = pd.date_range("2025-01-01", periods=48, freq="h")
        df = pd.DataFrame({"BTC": np.linspace(100, 200, 48)}, index=dates)
        result = resample_to_daily(df)
        assert len(result) <= 3
        assert not result.isnull().any().any()

    def test_forward_fill_gaps(self):
        dates = [datetime(2025, 1, 6), datetime(2025, 1, 8), datetime(2025, 1, 10)]
        df = pd.DataFrame(
            {"BTC": [100.0, 200.0, 300.0]}, index=pd.DatetimeIndex(dates)
        )
        result = resample_to_daily(df)
        assert len(result) == 5
        assert not result.isnull().any().any()

    def test_uses_last_price_of_day(self):
        dates = pd.to_datetime(
            ["2025-01-01 09:00", "2025-01-01 15:00", "2025-01-02 10:00"]
        )
        df = pd.DataFrame({"BTC": [100.0, 150.0, 200.0]}, index=dates)
        result = resample_to_daily(df)
        assert result.iloc[0, 0] == 150.0


class TestCalculateReturnsDataframe:
    def test_empty_df_returns_as_is(self):
        df = pd.DataFrame()
        result = calculate_returns_dataframe(df)
        assert result.empty

    def test_basic_returns(self):
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        df = pd.DataFrame({"BTC": [100.0, 110.0, 121.0]}, index=dates)
        result = calculate_returns_dataframe(df)
        assert len(result) == 3
        assert pd.isna(result.iloc[0, 0])
        assert abs(result.iloc[1, 0] - 0.10) < 1e-10
        assert abs(result.iloc[2, 0] - 0.10) < 1e-10

    def test_custom_periods(self):
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {"BTC": [100.0, 110.0, 121.0, 133.1, 146.41]}, index=dates
        )
        result = calculate_returns_dataframe(df, periods=2)
        assert pd.isna(result.iloc[0, 0])
        assert pd.isna(result.iloc[1, 0])
        assert abs(result.iloc[2, 0] - 0.21) < 1e-10

    def test_multiple_columns(self):
        df = _make_df(n=30, cols=["BTC", "ETH", "SOL"])
        result = calculate_returns_dataframe(df)
        assert set(result.columns) == {"BTC", "ETH", "SOL"}
        assert len(result) == 30

    def test_first_row_is_nan(self):
        df = _make_df(n=10)
        result = calculate_returns_dataframe(df)
        assert result.iloc[0].isna().all()

    def test_preserves_index(self):
        df = _make_df(n=10)
        result = calculate_returns_dataframe(df)
        pd.testing.assert_index_equal(result.index, df.index)

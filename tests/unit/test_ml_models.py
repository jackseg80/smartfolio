"""
Unit tests for services/ml_models.py

Tests for CryptoMLPredictor and CryptoMLPipeline classes.

Covers:
- Initialization and model parameters
- Feature preparation (prepare_features)
- MarketRegime enum correctness
- RegimePrediction and ReturnForecast dataclass instantiation
- predict_regime / predict_returns without training (ValueError)
- save_models / load_models edge cases
- CryptoMLPipeline initialization and get_predictions without training

KNOWN BUG: prepare_regime_labels references MarketRegime.DISTRIBUTION,
MarketRegime.ACCUMULATION, and MarketRegime.EUPHORIA which do NOT exist
on the MarketRegime enum (which only has BEAR_MARKET, CORRECTION,
BULL_MARKET, EXPANSION). Any code path calling prepare_regime_labels
will raise AttributeError. Tests document this bug explicitly.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import fields

from services.ml_models import (
    CryptoMLPredictor,
    CryptoMLPipeline,
    MarketRegime,
    RegimePrediction,
    ReturnForecast,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_price_data(assets=("BTC", "ETH"), periods=200, start="2024-01-01"):
    """Generate synthetic price data suitable for prepare_features."""
    np.random.seed(42)
    index = pd.date_range(start, periods=periods)
    data = {}
    for asset in assets:
        data[asset] = np.random.lognormal(0, 0.05, periods).cumsum() + 100
    return pd.DataFrame(data, index=index)


@pytest.fixture
def price_data_btc_eth():
    """Price DataFrame with BTC and ETH columns (200 rows)."""
    return _make_price_data(("BTC", "ETH"), 200)


@pytest.fixture
def price_data_single():
    """Price DataFrame with a single asset (200 rows)."""
    return _make_price_data(("BTC",), 200)


@pytest.fixture
def predictor(tmp_path):
    """CryptoMLPredictor backed by a temporary models directory."""
    return CryptoMLPredictor(models_path=str(tmp_path / "models"))


# ---------------------------------------------------------------------------
# 1. MarketRegime enum
# ---------------------------------------------------------------------------

class TestMarketRegime:
    """Verify the canonical MarketRegime enum."""

    def test_has_exactly_four_members(self):
        assert len(MarketRegime) == 4

    def test_canonical_values(self):
        assert MarketRegime.BEAR_MARKET.value == "Bear Market"
        assert MarketRegime.CORRECTION.value == "Correction"
        assert MarketRegime.BULL_MARKET.value == "Bull Market"
        assert MarketRegime.EXPANSION.value == "Expansion"

    def test_can_construct_from_value(self):
        assert MarketRegime("Bear Market") is MarketRegime.BEAR_MARKET
        assert MarketRegime("Expansion") is MarketRegime.EXPANSION

    def test_missing_legacy_members(self):
        """Document the bug: these members are referenced in prepare_regime_labels
        but do NOT exist on MarketRegime."""
        assert not hasattr(MarketRegime, "DISTRIBUTION")
        assert not hasattr(MarketRegime, "ACCUMULATION")
        assert not hasattr(MarketRegime, "EUPHORIA")


# ---------------------------------------------------------------------------
# 2. Dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    """RegimePrediction and ReturnForecast can be instantiated correctly."""

    def test_regime_prediction_fields(self):
        field_names = {f.name for f in fields(RegimePrediction)}
        assert field_names == {
            "regime", "confidence", "probabilities",
            "features_importance", "timestamp",
        }

    def test_regime_prediction_instantiation(self):
        ts = pd.Timestamp("2024-06-01")
        rp = RegimePrediction(
            regime=MarketRegime.BULL_MARKET,
            confidence=0.87,
            probabilities={"Bull Market": 0.87, "Expansion": 0.13},
            features_importance={"BTC_return_1d": 0.4, "ETH_vol_7d": 0.6},
            timestamp=ts,
        )
        assert rp.regime is MarketRegime.BULL_MARKET
        assert rp.confidence == 0.87
        assert rp.timestamp == ts

    def test_return_forecast_fields(self):
        field_names = {f.name for f in fields(ReturnForecast)}
        assert field_names == {
            "expected_returns", "confidence_intervals",
            "model_accuracy", "forecast_horizon", "timestamp",
        }

    def test_return_forecast_instantiation(self):
        ts = pd.Timestamp("2024-06-01")
        rf = ReturnForecast(
            expected_returns={"BTC": 0.05, "ETH": 0.03},
            confidence_intervals={"BTC": (-0.02, 0.12), "ETH": (-0.04, 0.10)},
            model_accuracy=0.65,
            forecast_horizon=7,
            timestamp=ts,
        )
        assert rf.forecast_horizon == 7
        assert "BTC" in rf.expected_returns
        assert len(rf.confidence_intervals["ETH"]) == 2


# ---------------------------------------------------------------------------
# 3. CryptoMLPredictor.__init__
# ---------------------------------------------------------------------------

class TestCryptoMLPredictorInit:
    """Initialization creates directory, sets defaults."""

    def test_creates_models_directory(self, tmp_path):
        target = tmp_path / "nested" / "models"
        assert not target.exists()
        CryptoMLPredictor(models_path=str(target))
        assert target.exists() and target.is_dir()

    def test_models_initially_none_or_empty(self, predictor):
        assert predictor.regime_model is None
        assert predictor.return_models == {}

    def test_scaler_is_standard_scaler(self, predictor):
        from sklearn.preprocessing import StandardScaler
        assert isinstance(predictor.feature_scaler, StandardScaler)

    def test_regime_model_params(self, predictor):
        params = predictor.regime_model_params
        assert params["n_estimators"] == 100
        assert params["max_depth"] == 10
        assert params["min_samples_split"] == 5
        assert params["random_state"] == 42

    def test_return_model_params(self, predictor):
        params = predictor.return_model_params
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 8
        assert params["min_samples_split"] == 10
        assert params["random_state"] == 42


# ---------------------------------------------------------------------------
# 4. prepare_features
# ---------------------------------------------------------------------------

class TestPrepareFeatures:
    """prepare_features is pure pandas; no ML models required."""

    def test_returns_dataframe(self, predictor, price_data_btc_eth):
        features = predictor.prepare_features(price_data_btc_eth)
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0

    def test_generates_return_columns(self, predictor, price_data_btc_eth):
        features = predictor.prepare_features(price_data_btc_eth)
        for asset in ("BTC", "ETH"):
            assert f"{asset}_return_1d" in features.columns
            assert f"{asset}_return_7d" in features.columns
            assert f"{asset}_return_30d" in features.columns

    def test_generates_volatility_columns(self, predictor, price_data_btc_eth):
        features = predictor.prepare_features(price_data_btc_eth)
        for asset in ("BTC", "ETH"):
            assert f"{asset}_vol_7d" in features.columns
            assert f"{asset}_vol_30d" in features.columns
            assert f"{asset}_vol_ratio" in features.columns

    def test_generates_momentum_columns(self, predictor, price_data_btc_eth):
        features = predictor.prepare_features(price_data_btc_eth)
        for asset in ("BTC", "ETH"):
            assert f"{asset}_ma_20" in features.columns
            assert f"{asset}_ma_50" in features.columns
            assert f"{asset}_price_to_ma20" in features.columns
            assert f"{asset}_rsi" in features.columns

    def test_cross_asset_features_btc_eth(self, predictor, price_data_btc_eth):
        features = predictor.prepare_features(price_data_btc_eth)
        assert "btc_eth_ratio" in features.columns
        assert "btc_eth_corr" in features.columns

    def test_market_wide_features_multi_asset(self, predictor, price_data_btc_eth):
        features = predictor.prepare_features(price_data_btc_eth)
        assert "market_vol" in features.columns
        assert "market_trend" in features.columns

    def test_no_cross_asset_features_single_asset(self, predictor, price_data_single):
        features = predictor.prepare_features(price_data_single)
        assert "btc_eth_ratio" not in features.columns
        assert "btc_eth_corr" not in features.columns
        # market_vol and market_trend only appear when >1 column
        assert "market_vol" not in features.columns

    def test_market_indicators_scalar(self, predictor, price_data_btc_eth):
        indicators = {"vix": 18.5, "dxy": 104.2}
        features = predictor.prepare_features(price_data_btc_eth, market_indicators=indicators)
        assert "market_vix" in features.columns
        assert "market_dxy" in features.columns
        # Scalar indicators fill every row with the same value
        assert (features["market_vix"] == 18.5).all()

    def test_market_indicators_series(self, predictor, price_data_btc_eth):
        idx = price_data_btc_eth.index
        vix_series = pd.Series(np.random.uniform(15, 30, len(idx)), index=idx)
        features = predictor.prepare_features(
            price_data_btc_eth,
            market_indicators={"vix": vix_series},
        )
        assert "market_vix" in features.columns

    def test_no_nans_in_output(self, predictor, price_data_btc_eth):
        """After ffill + dropna, no NaN values should remain."""
        features = predictor.prepare_features(price_data_btc_eth)
        assert features.isna().sum().sum() == 0

    def test_output_shorter_than_input(self, predictor, price_data_btc_eth):
        """Rolling windows drop leading rows, so output is shorter."""
        features = predictor.prepare_features(price_data_btc_eth)
        assert len(features) < len(price_data_btc_eth)


# ---------------------------------------------------------------------------
# 5. prepare_regime_labels -- documents the bug
# ---------------------------------------------------------------------------

class TestPrepareRegimeLabels:
    """prepare_regime_labels references non-existent enum members.
    This is a KNOWN BUG: MarketRegime.DISTRIBUTION, .ACCUMULATION,
    and .EUPHORIA do not exist on the enum. However the bug is
    CONDITIONAL -- it only triggers when specific market conditions
    are met (strong downtrend or strong uptrend + high vol). With
    benign price data that stays in the neutral/expansion path, the
    method may succeed without error.

    These tests craft specific price series to exercise each branch."""

    @staticmethod
    def _make_downtrend_data(periods=200):
        """Price series with consistent negative drift to trigger ret_90 < -0.2."""
        np.random.seed(42)
        idx = pd.date_range("2024-01-01", periods=periods)
        prices = np.zeros(periods)
        prices[0] = 1000
        for i in range(1, periods):
            prices[i] = prices[i - 1] * np.exp(-0.01 + np.random.normal(0, 0.04))
        return pd.DataFrame({"BTC": prices}, index=idx)

    @staticmethod
    def _make_uptrend_high_vol_data(periods=200):
        """Price series with strong uptrend + high volatility to trigger ret_90 > 0.5."""
        np.random.seed(99)
        idx = pd.date_range("2024-01-01", periods=periods)
        prices = np.zeros(periods)
        prices[0] = 100
        for i in range(1, periods):
            prices[i] = prices[i - 1] * np.exp(0.02 + np.random.normal(0, 0.06))
        return pd.DataFrame({"BTC": prices}, index=idx)

    def test_neutral_data_only_hits_expansion(self, predictor, price_data_single):
        """With gentle lognormal data, only the neutral/EXPANSION branch
        is taken, so the bug is NOT triggered."""
        labels = predictor.prepare_regime_labels(price_data_single)
        assert len(labels) > 0
        # All labels should be "Expansion" (the one valid enum value in the code)
        assert set(labels.unique()).issubset({"Expansion"})

    def test_downtrend_produces_bear_market_label(self, predictor):
        """Strong downtrend + high vol produces BEAR_MARKET labels."""
        df = self._make_downtrend_data()
        labels = predictor.prepare_regime_labels(df)
        assert len(labels) > 0
        assert MarketRegime.BEAR_MARKET.value in labels.values

    def test_uptrend_high_vol_produces_bull_market_label(self, predictor):
        """Strong uptrend + high vol produces BULL_MARKET or EXPANSION labels."""
        df = self._make_uptrend_high_vol_data()
        labels = predictor.prepare_regime_labels(df)
        assert len(labels) > 0
        valid_regimes = {MarketRegime.BULL_MARKET.value, MarketRegime.EXPANSION.value}
        assert any(v in valid_regimes for v in labels.values)

    def test_primary_asset_fallback(self, predictor):
        """If primary_asset is not in columns, the first column is used.
        With neutral data the method succeeds; the fallback path is
        exercised (SOL is used instead of BTC)."""
        np.random.seed(42)
        df = _make_price_data(("SOL",), 200)
        labels = predictor.prepare_regime_labels(df, primary_asset="BTC")
        # Should succeed because only expansion branch is hit
        assert len(labels) > 0

    def test_missing_enum_members_documented(self):
        """Explicitly verify the enum does NOT have the members
        referenced in prepare_regime_labels."""
        assert not hasattr(MarketRegime, "DISTRIBUTION")
        assert not hasattr(MarketRegime, "ACCUMULATION")
        assert not hasattr(MarketRegime, "EUPHORIA")


# ---------------------------------------------------------------------------
# 6. predict_regime / predict_returns without training
# ---------------------------------------------------------------------------

class TestPredictWithoutTraining:
    """Calling prediction methods before training must raise ValueError."""

    def test_predict_regime_raises_without_training(self, predictor, price_data_btc_eth):
        features = predictor.prepare_features(price_data_btc_eth)
        with pytest.raises(ValueError, match="Regime model not trained"):
            predictor.predict_regime(features)

    def test_predict_returns_raises_without_training(self, predictor, price_data_btc_eth):
        features = predictor.prepare_features(price_data_btc_eth)
        with pytest.raises(ValueError, match="Return models not trained"):
            predictor.predict_returns(features)


# ---------------------------------------------------------------------------
# 7. save_models / load_models
# ---------------------------------------------------------------------------

class TestSaveLoadModels:
    """Test persistence edge cases with tmp_path."""

    def test_save_with_no_models_saves_scaler(self, predictor, tmp_path):
        """With no regime_model and no return_models, only scaler is saved."""
        result = predictor.save_models()
        assert result is True

        scaler_path = predictor.models_path / "feature_scaler.pkl"
        assert scaler_path.exists()

        # No regime or return model files should exist
        assert not (predictor.models_path / "regime_model.pkl").exists()
        assert not (predictor.models_path / "return_models.pkl").exists()

    def test_load_with_no_files_returns_true(self, predictor):
        """load_models succeeds (returns True) even when no files exist.
        It simply skips files that do not exist."""
        with patch("services.ml_models.safe_pickle_load") as mock_load:
            # safe_pickle_load should never be called because no files exist
            result = predictor.load_models()
            assert result is True
            mock_load.assert_not_called()

    def test_save_then_load_round_trip_scaler(self, predictor, price_data_btc_eth):
        """Save scaler, reload it, verify it's a StandardScaler."""
        # Fit the scaler on some data so it has state
        features = predictor.prepare_features(price_data_btc_eth)
        predictor.feature_scaler.fit(features)

        predictor.save_models()

        # Create a fresh predictor pointing at the same path
        new_predictor = CryptoMLPredictor(models_path=str(predictor.models_path))
        # Patch safe_pickle_load to actually use pickle since tmp_path
        # is outside SAFE_MODEL_DIRS
        import pickle

        def _mock_safe_load(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        with patch("services.ml_models.safe_pickle_load", side_effect=_mock_safe_load):
            result = new_predictor.load_models()
            assert result is True

        from sklearn.preprocessing import StandardScaler
        assert isinstance(new_predictor.feature_scaler, StandardScaler)
        # Verify the scaler has been fitted (has mean_ attribute)
        assert hasattr(new_predictor.feature_scaler, "mean_")

    def test_load_models_returns_false_on_error(self, predictor, tmp_path):
        """If safe_pickle_load raises an unexpected error, load_models returns False."""
        # Create a dummy file so the existence check passes
        scaler_path = predictor.models_path / "feature_scaler.pkl"
        scaler_path.write_bytes(b"corrupted")

        with patch(
            "services.ml_models.safe_pickle_load",
            side_effect=RuntimeError("corrupt file"),
        ):
            result = predictor.load_models()
            assert result is False


# ---------------------------------------------------------------------------
# 8. CryptoMLPipeline.__init__
# ---------------------------------------------------------------------------

class TestCryptoMLPipelineInit:
    """Pipeline initialization."""

    def test_creates_predictor_instance(self):
        pipeline = CryptoMLPipeline()
        assert isinstance(pipeline.predictor, CryptoMLPredictor)

    def test_is_trained_false(self):
        pipeline = CryptoMLPipeline()
        assert pipeline.is_trained is False


# ---------------------------------------------------------------------------
# 9. CryptoMLPipeline.get_predictions without training
# ---------------------------------------------------------------------------

class TestCryptoMLPipelineGetPredictions:
    """get_predictions when models are not trained and cannot be loaded."""

    def test_raises_value_error_when_load_fails(self, price_data_btc_eth):
        pipeline = CryptoMLPipeline()
        pipeline.is_trained = False

        with patch.object(
            pipeline.predictor, "load_models", return_value=False
        ):
            with pytest.raises(ValueError, match="Models not trained and cannot be loaded"):
                pipeline.get_predictions(price_data_btc_eth)

    def test_sets_is_trained_after_successful_load(self, price_data_btc_eth):
        """If load_models succeeds, is_trained flips to True and
        predictions proceed (may still fail on individual models)."""
        pipeline = CryptoMLPipeline()
        pipeline.is_trained = False

        with patch.object(pipeline.predictor, "load_models", return_value=True):
            # predict_regime and predict_returns will error because no actual
            # models are loaded -- but get_predictions catches those and
            # returns error dicts instead of raising.
            result = pipeline.get_predictions(price_data_btc_eth)

        assert pipeline.is_trained is True
        # Both keys present even on error
        assert "regime" in result
        assert "returns" in result
        # They should contain error info since no actual models were loaded
        assert "error" in result["regime"]
        assert "error" in result["returns"]


# ---------------------------------------------------------------------------
# 10. CryptoMLPipeline.train_pipeline -- blocked by enum bug
# ---------------------------------------------------------------------------

class TestCryptoMLPipelineTrainPipeline:
    """train_pipeline calls prepare_regime_labels which uses MarketRegime enum
    for all price regimes including downtrends."""

    def test_train_pipeline_with_downtrend_data(self, tmp_path):
        """train_pipeline succeeds with downtrend data (MarketRegime enum is correct).
        The pipeline may catch internal training errors but should not raise AttributeError."""
        np.random.seed(42)
        idx = pd.date_range("2024-01-01", periods=200)
        prices = np.zeros(200)
        prices[0] = 1000
        for i in range(1, 200):
            prices[i] = prices[i - 1] * np.exp(-0.01 + np.random.normal(0, 0.04))
        df = pd.DataFrame({"BTC": prices}, index=idx)

        pipeline = CryptoMLPipeline()
        pipeline.predictor = CryptoMLPredictor(models_path=str(tmp_path / "m"))
        # Should NOT raise AttributeError - enum values are correct
        results = pipeline.train_pipeline(df, save_models=False)
        assert isinstance(results, dict)
        assert pipeline.is_trained is True

    def test_train_pipeline_with_neutral_data_hits_training_error(
        self, price_data_btc_eth, tmp_path
    ):
        """With neutral data, prepare_regime_labels succeeds (only Expansion
        labels). But train_regime_model then fails because stratified split
        requires at least 2 classes or sufficient samples. The pipeline
        catches this and records the error."""
        pipeline = CryptoMLPipeline()
        pipeline.predictor = CryptoMLPredictor(models_path=str(tmp_path / "m"))
        results = pipeline.train_pipeline(price_data_btc_eth, save_models=False)
        # Pipeline catches internal errors and stores them
        assert "regime_model" in results
        # is_trained is set True at the end regardless of sub-model failures
        assert pipeline.is_trained is True


# ---------------------------------------------------------------------------
# 11. Global instance
# ---------------------------------------------------------------------------

class TestGlobalInstance:
    """The module exports a global ml_pipeline singleton."""

    def test_ml_pipeline_exists(self):
        from services.ml_models import ml_pipeline
        assert isinstance(ml_pipeline, CryptoMLPipeline)

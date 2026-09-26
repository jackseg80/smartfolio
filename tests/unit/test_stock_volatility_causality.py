import numpy as np
import pandas as pd

from services.ml.models.volatility_predictor import VolatilityPredictor
from services.ml import safe_loader


def _predictor():
    predictor = object.__new__(VolatilityPredictor)
    predictor.trading_days = 252
    predictor.horizons = [1, 7, 30]
    predictor.sequence_length = 10
    predictor.predict_uncertainty = False
    predictor.scalers = {}
    return predictor


def test_stock_volatility_features_do_not_use_future_prices():
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2024-01-01", periods=210)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
    data = pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.integers(1000, 3000, len(index)),
    }, index=index)

    predictor = _predictor()
    prefix = predictor.prepare_features(data.iloc[:150], "TEST")
    full = predictor.prepare_features(data, "TEST")
    feature_columns = [col for col in prefix if not col.startswith("target_vol_")]

    pd.testing.assert_frame_equal(
        prefix[feature_columns], full.loc[prefix.index, feature_columns]
    )
    assert pd.isna(prefix["target_vol_30d"].iloc[-1])


def test_stock_scaler_is_fit_only_on_training_prefix():
    predictor = _predictor()
    features = pd.DataFrame({
        "signal": np.r_[np.arange(60), np.repeat(10000, 40)],
        "target_vol_1d": np.repeat(0.2, 100),
        "target_vol_7d": np.repeat(0.2, 100),
        "target_vol_30d": np.repeat(0.2, 100),
    })

    X, y, indices = predictor.create_sequences(
        features, "TEST", training_cutoff=60, return_indices=True
    )

    assert X.shape[0] == y.shape[0] == len(indices)
    assert predictor.scalers["TEST"].center_[0] == 29.5
    assert indices[0] == predictor.sequence_length - 1
    assert X[0, -1, 0] == predictor.scalers["TEST"].transform(features[["signal"]].iloc[[indices[0]]])[0, 0]


def test_stock_model_trains_with_temporal_test_and_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(safe_loader, "SAFE_MODEL_DIRS", [*safe_loader.SAFE_MODEL_DIRS, tmp_path])
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2024-01-01", periods=360)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, len(index))))
    prices = pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.integers(1000, 3000, len(index)),
    }, index=index)
    predictor = VolatilityPredictor(str(tmp_path), trading_days=252, predict_uncertainty=False)
    predictor.sequence_length = 12
    predictor.hidden_size = 8
    predictor.num_layers = 1
    predictor.dropout = 0.0
    predictor.epochs = 1
    predictor.batch_size = 64

    result = predictor.train_model("TEST", prices)

    assert result["split_method"] == "chronological_embargoed"
    assert result["temporal_test_mae"] is not None
    assert result["baseline_test_mae"] is not None
    assert result["train_samples"] > result["val_samples"] >= 10

import numpy as np
import pandas as pd

from services.forecasting.temporal_validation import (
    TrainOnlyPreprocessor,
    backward_asof_join,
    purged_chronological_split,
)


def test_asof_join_can_only_select_past_observations():
    decisions = pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-04"], utc=True)})
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-05"], utc=True),
            "value": [1, 3, 5],
        }
    )

    joined = backward_asof_join(decisions, observations, on="date")

    assert joined["value"].tolist() == [1, 3]
    assert (joined["date_source"] <= joined["date"]).all()


def test_purged_split_uses_one_boundary_for_every_asset():
    dates = pd.date_range("2025-01-01", periods=160, freq="D")
    frame = pd.DataFrame(
        [(date, symbol) for date in dates for symbol in ("BTC", "ETH")],
        columns=["decision_date", "entity"],
    )

    split = purged_chronological_split(frame, test_start="2025-05-01", purge_days=30)
    train = frame.loc[split.train_index]
    test = frame.loc[split.test_index]

    assert train["decision_date"].max() == pd.Timestamp("2025-03-31")
    assert test["decision_date"].min() == pd.Timestamp("2025-05-01")
    assert set(train.groupby("decision_date")["entity"].nunique()) == {2}
    assert set(test.groupby("decision_date")["entity"].nunique()) == {2}


def test_preprocessing_statistics_are_fitted_on_training_only():
    train = pd.DataFrame(
        {
            "decision_date": pd.date_range("2025-01-01", periods=3),
            "changing": [1.0, 2.0, 3.0],
            "constant": [7.0, 7.0, 7.0],
        }
    )
    future = pd.DataFrame(
        {
            "decision_date": pd.date_range("2026-01-01", periods=2),
            "changing": [1_000_000.0, -1_000_000.0],
            "constant": [999.0, -999.0],
        }
    )
    preprocessor = TrainOnlyPreprocessor().fit(train, ["changing", "constant"])

    transformed = preprocessor.transform(future)

    assert preprocessor.feature_names_ == ["changing"]
    assert preprocessor.means_["changing"] == 2.0
    assert preprocessor.scales_["changing"] == np.std([1.0, 2.0, 3.0])
    assert abs(transformed.iloc[0, 0]) > 1000

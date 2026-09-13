from datetime import date, timedelta

import numpy as np

from scripts.train_models import _chronological_split_indices, _trailing_realized_volatility


def test_trailing_realized_volatility_ignores_all_future_returns():
    original = np.linspace(-0.02, 0.03, 80)
    changed = original.copy()
    changed[41:] = np.linspace(-9.0, 9.0, len(changed[41:]))

    assert _trailing_realized_volatility(original, 40) == _trailing_realized_volatility(changed, 40)


def test_real_samples_share_date_boundaries_and_have_thirty_day_purges():
    start = date(2024, 1, 1)
    samples = [
        {"decision_date": (start + timedelta(days=day)).isoformat(), "symbol": symbol}
        for day in range(400)
        for symbol in ("BTC", "ETH", "SOL")
    ]

    train, validation, test, metadata = _chronological_split_indices(samples, purge_days=30)
    dates = np.array([sample["decision_date"] for sample in samples])

    assert metadata["kind"] == "shared_decision_dates"
    assert metadata["purge_days"] == 30
    assert max(dates[train].tolist()) < metadata["validation_start"]
    assert max(dates[validation].tolist()) < metadata["test_start"]
    assert len(set(dates[test][:3])) == 1

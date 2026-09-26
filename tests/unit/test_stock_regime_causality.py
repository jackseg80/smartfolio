"""Stock market regime labels must use information available on each date."""

import numpy as np
import pandas as pd

from services.ml.models.regime_detector import create_rule_based_labels


def test_new_prices_do_not_rewrite_prior_regime_labels():
    prices = np.concatenate(
        [
            np.linspace(80, 120, 220),
            np.linspace(120, 75, 65),
            np.linspace(75, 115, 80),
        ]
    )
    history = pd.DataFrame(
        {"close": prices},
        index=pd.date_range("2024-01-01", periods=len(prices), freq="B"),
    )

    for end in (240, 285, 320):
        prefix_labels = create_rule_based_labels(history.iloc[:end])
        full_labels = create_rule_based_labels(history)
        assert np.array_equal(prefix_labels, full_labels[:end])

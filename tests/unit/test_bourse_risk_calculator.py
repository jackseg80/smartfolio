import numpy as np
import pandas as pd

from services.risk.bourse.calculator import BourseRiskCalculator


def test_portfolio_returns_aligns_on_usable_return_dates():
    """A missing quote must not make weighted returns use incompatible arrays."""
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    position_data = {
        "AAA": {
            "prices": pd.DataFrame({"close": [100.0, 110.0, 121.0, 133.1]}, index=dates),
            "weight": 0.6,
        },
        "BBB": {
            "prices": pd.DataFrame({"close": [200.0, np.nan, 220.0, 242.0]}, index=dates),
            "weight": 0.4,
        },
    }

    returns = BourseRiskCalculator._calculate_portfolio_returns(
        object.__new__(BourseRiskCalculator), position_data
    )

    assert np.allclose(returns, [0.1])

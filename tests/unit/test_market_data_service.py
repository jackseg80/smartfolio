from services.market_data import calculate_market_breadth


def test_calculate_market_breadth_uses_global_market_rows():
    result = calculate_market_breadth([
        {"price_change_percentage_24h": 2.0, "total_volume": 100, "ath_change_percentage": -2},
        {"price_change_percentage_24h": -1.0, "total_volume": 300, "ath_change_percentage": -20},
        {"price_change_percentage_24h": 0.5, "total_volume": 600, "ath_change_percentage": None},
    ])
    assert result["advance_decline_ratio"] == 0.667
    assert result["new_highs_count"] == 1
    assert result["volume_concentration"] == 1.0
    assert result["meta"]["assets_analyzed"] == 3


def test_calculate_market_breadth_reports_missing_data():
    result = calculate_market_breadth([])
    assert result["meta"]["status"] == "no_data"
    assert result["meta"]["assets_analyzed"] == 0
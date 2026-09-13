import json

import numpy as np
import pandas as pd
import pytest

from services.forecasting.dataset import (
    FEATURE_COLUMNS,
    UniverseMember,
    build_forecast_dataset,
    causal_feature_digest,
)


def _series(days=280, *, start="2024-01-01", daily_return=0.001, stop=None):
    dates = pd.date_range(start, periods=days, freq="D")
    values = 100.0 * np.power(1.0 + daily_return, np.arange(days))
    result = pd.Series(values, index=dates)
    return result.iloc[:stop] if stop is not None else result


def _groups(symbol):
    return {"BTC": "BTC", "ETH": "Smart", "SOL": "Smart", "USDT": "Stablecoins"}[symbol]


def _market_data(series, *, multiplier=1.0):
    count = len(series)
    return pd.DataFrame(
        {
            "quote_asset_volume": multiplier * np.linspace(1_000_000.0, 2_000_000.0, count),
            "trades": multiplier * np.linspace(10_000.0, 20_000.0, count),
        },
        index=series.index,
    )


def test_builds_separate_causal_features_and_future_labels():
    build = build_forecast_dataset(
        {"BTC": _series(daily_return=0.01), "ETH": _series(daily_return=0.02)},
        group_for_symbol=_groups,
    )
    row = build.frame[
        (build.frame["scope"] == "asset")
        & (build.frame["entity"] == "ETH")
        & (build.frame["decision_date"] == "2024-08-20")
    ].iloc[0]

    assert row["past_return_7d"] == pytest.approx((1.02**7) - 1)
    assert row["relative_btc_return_30d"] == pytest.approx((1.02**30) - (1.01**30))
    assert row["target_return_7d"] == pytest.approx((1.02**7) - 1)
    assert row["label_up_7d"] == 1
    assert not any("probability" in column for column in build.frame.columns)
    assert set(FEATURE_COLUMNS).isdisjoint(
        column for column in build.frame.columns if column.startswith("target_")
    )


def test_mutating_every_future_price_does_not_change_past_features():
    histories = {
        "BTC": _series(days=300, daily_return=0.002),
        "ETH": _series(days=300, daily_return=0.003),
        "SOL": _series(days=300, daily_return=-0.001),
    }
    cutoff = pd.Timestamp("2024-08-15")
    original = build_forecast_dataset(histories, group_for_symbol=_groups)
    mutated = {}
    for symbol, series in histories.items():
        changed = series.copy()
        future = changed.index > cutoff
        changed.loc[future] = changed.loc[future] * np.linspace(2.0, 25.0, future.sum())
        mutated[symbol] = changed
    rebuilt = build_forecast_dataset(mutated, group_for_symbol=_groups)

    assert causal_feature_digest(original.frame, cutoff) == causal_feature_digest(
        rebuilt.frame, cutoff
    )
    past_labels = original.frame[original.frame["decision_date"] <= cutoff.date().isoformat()][
        "target_return_30d"
    ]
    changed_labels = rebuilt.frame[rebuilt.frame["decision_date"] <= cutoff.date().isoformat()][
        "target_return_30d"
    ]
    assert not past_labels.equals(changed_labels)


def test_mutating_future_volume_does_not_change_past_features():
    histories = {
        "BTC": _series(days=300, daily_return=0.002),
        "ETH": _series(days=300, daily_return=0.003),
    }
    market_data = {symbol: _market_data(series) for symbol, series in histories.items()}
    cutoff = pd.Timestamp("2024-08-15")
    original = build_forecast_dataset(
        histories,
        group_for_symbol=_groups,
        market_data=market_data,
    )
    mutated_market_data = {symbol: frame.copy() for symbol, frame in market_data.items()}
    for frame in mutated_market_data.values():
        future = frame.index > cutoff
        frame.loc[future, ["quote_asset_volume", "trades"]] *= 100.0
    rebuilt = build_forecast_dataset(
        histories,
        group_for_symbol=_groups,
        market_data=mutated_market_data,
    )

    assert causal_feature_digest(original.frame, cutoff) == causal_feature_digest(
        rebuilt.frame, cutoff
    )
    assert original.manifest["volume_liquidity"] == (
        "dated_quote_volume_and_trade_count_used_as_causal_features"
    )


def test_missing_group_exit_is_reported_without_constituent_renormalization():
    build = build_forecast_dataset(
        {
            "BTC": _series(days=280, daily_return=0.001),
            "ETH": _series(days=280, daily_return=0.002),
            "SOL": _series(days=280, daily_return=0.003, stop=245),
        },
        group_for_symbol=_groups,
    )
    decision_date = pd.Timestamp("2024-08-28")
    row = build.frame[
        (build.frame["scope"] == "group")
        & (build.frame["entity"] == "Smart")
        & (build.frame["decision_date"] == decision_date.date().isoformat())
    ].iloc[0]

    assert row["universe_member_count"] == 2
    assert pd.isna(row["target_return_30d"])
    assert row["target_status_30d"] == "unavailable"
    assert json.loads(row["missing_exit_members_30d"]) == ["SOL"]


def test_explicit_delisting_changes_membership_only_after_known_until():
    histories = {
        "BTC": _series(days=280),
        "ETH": _series(days=280),
        "SOL": _series(days=240),
    }
    universe = [
        UniverseMember("BTC", "BTC", "2024-01-01"),
        UniverseMember("ETH", "Smart", "2024-01-01"),
        UniverseMember(
            "SOL",
            "Smart",
            "2024-01-01",
            known_until="2024-08-20",
            delisted=True,
            membership_provenance="explicit_dated_manifest",
        ),
    ]
    build = build_forecast_dataset(histories, universe=universe)
    before = build.frame[
        (build.frame["scope"] == "group")
        & (build.frame["entity"] == "Smart")
        & (build.frame["decision_date"] == "2024-08-20")
    ].iloc[0]
    after = build.frame[
        (build.frame["scope"] == "group")
        & (build.frame["entity"] == "Smart")
        & (build.frame["decision_date"] == "2024-08-21")
    ].iloc[0]

    assert before["universe_member_count"] == 2
    assert after["universe_member_count"] == 1
    assert build.coverage["universe"]["explicitly_delisted_members"] == 1


def test_explicit_member_without_any_history_is_kept_as_unavailable():
    universe = [
        UniverseMember("BTC", "BTC", "2024-01-01"),
        UniverseMember(
            "MISSING",
            "Smart",
            "2024-02-01",
            membership_provenance="explicit_dated_manifest",
        ),
    ]

    build = build_forecast_dataset({"BTC": _series(days=280)}, universe=universe)
    rows = build.frame[(build.frame["scope"] == "asset") & (build.frame["entity"] == "MISSING")]

    assert not rows.empty
    assert set(rows["history_status"]) == {"missing_decision_price"}
    assert rows["target_return_7d"].isna().all()


def test_stablecoin_cache_is_excluded_and_dataset_version_is_reproducible():
    histories = {
        "BTC": _series(),
        "ETH": _series(),
        "USDT": _series(daily_return=0.0),
    }
    first = build_forecast_dataset(histories, group_for_symbol=_groups)
    second = build_forecast_dataset(histories, group_for_symbol=_groups)

    assert first.manifest["dataset_version"] == second.manifest["dataset_version"]
    assert "USDT" not in set(first.frame["entity"])
    usdt = next(member for member in first.universe if member.symbol == "USDT")
    assert usdt.eligible_for_forecasting is False
    assert first.manifest["defensive_reference"]["kind"] == (
        "explicit_assumption_not_observed_market_data"
    )

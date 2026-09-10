from services.rebalance import plan_rebalance
from services.taxonomy import DEFAULT_GROUPS_ORDER


def _targets(**overrides):
    targets = {group: 0.0 for group in DEFAULT_GROUPS_ORDER}
    targets.update(overrides)
    return targets


def test_unknown_position_is_blocked_and_cannot_fund_a_purchase():
    plan = plan_rebalance(
        rows=[
            {"symbol": "BTC", "alias": "BTC", "value_usd": 5_000, "location": "Kraken"},
            {"symbol": "LLY", "alias": "LLY", "value_usd": 5_000, "location": "CoinTracking"},
        ],
        group_targets_pct=_targets(BTC=100.0),
        min_trade_usd=25.0,
    )

    assert plan["unknown_aliases"] == ["LLY"]
    assert plan["requires_alias_review"] is True
    assert plan["blocked_unknown_usd"] == 5_000
    assert plan["blocked_unknown_pct"] == 50.0
    assert plan["actions"] == []


def test_known_sales_can_rebalance_around_a_blocked_unknown_position():
    plan = plan_rebalance(
        rows=[
            {"symbol": "BTC", "alias": "BTC", "value_usd": 2_500, "location": "Kraken"},
            {"symbol": "USDT", "alias": "USDT", "value_usd": 5_000, "location": "Kraken"},
            {"symbol": "LLY", "alias": "LLY", "value_usd": 2_500, "location": "CoinTracking"},
        ],
        group_targets_pct=_targets(BTC=50.0, Stablecoins=25.0, Others=25.0),
        min_trade_usd=25.0,
    )

    assert all(action["alias"] != "LLY" for action in plan["actions"])
    assert sum(action["usd"] for action in plan["actions"]) == 0
    assert {action["alias"] for action in plan["actions"]} == {"BTC", "USDT"}

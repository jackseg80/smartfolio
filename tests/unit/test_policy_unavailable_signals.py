from services.execution.policy_engine import PolicyEngine
from services.execution.signals import create_default_signals


def test_unavailable_signals_fail_closed_without_synthetic_market_values():
    engine = PolicyEngine()
    signals = create_default_signals("regime inference unavailable")

    policy = engine.derive_execution_policy(signals, governance_mode="ai_assisted")

    assert signals.available is False
    assert signals.volatility == {}
    assert signals.regime == {}
    assert signals.correlation == {}
    assert signals.sentiment == {}
    assert policy.mode == "Freeze"
    assert policy.cap_daily == 0.01
    assert "regime inference unavailable" in policy.notes

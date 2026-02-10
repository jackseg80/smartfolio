"""
Tests for services/alerts/alert_types.py
Covers AlertSeverity, AlertType enums, AlertFormatter, AlertRule, Alert, AlertEvaluator.
"""

import pytest
from datetime import datetime, timedelta
from services.alerts.alert_types import (
    AlertSeverity,
    AlertType,
    AlertFormatter,
    AlertRule,
    Alert,
    AlertEvaluator,
)


# ── AlertSeverity ──────────────────────────────────────────────────

class TestAlertSeverity:

    def test_values(self):
        assert AlertSeverity.S1.value == "S1"
        assert AlertSeverity.S2.value == "S2"
        assert AlertSeverity.S3.value == "S3"

    def test_count(self):
        assert len(AlertSeverity) == 3

    def test_str_enum(self):
        assert isinstance(AlertSeverity.S1, str)
        assert AlertSeverity.S1 == "S1"

    def test_from_value(self):
        assert AlertSeverity("S1") == AlertSeverity.S1


# ── AlertType ──────────────────────────────────────────────────────

class TestAlertType:

    def test_core_types_exist(self):
        """Check all core alert types are defined."""
        core = [
            "VOL_Q90_CROSS", "REGIME_FLIP", "CORR_HIGH", "CORR_SPIKE",
            "CONTRADICTION_SPIKE", "DECISION_DROP", "EXEC_COST_SPIKE",
        ]
        for name in core:
            assert hasattr(AlertType, name)

    def test_predictive_types_exist(self):
        """Phase 2C predictive types."""
        predictive = [
            "SPIKE_LIKELY", "REGIME_CHANGE_PENDING",
            "CORRELATION_BREAKDOWN", "VOLATILITY_SPIKE_IMMINENT",
        ]
        for name in predictive:
            assert hasattr(AlertType, name)

    def test_risk_model_types_exist(self):
        """Phase 3A advanced risk types."""
        risk = [
            "VAR_BREACH", "STRESS_TEST_FAILED",
            "MONTE_CARLO_EXTREME", "RISK_CONCENTRATION",
        ]
        for name in risk:
            assert hasattr(AlertType, name)

    def test_legacy_types_exist(self):
        legacy = [
            "PORTFOLIO_DRIFT", "EXECUTION_FAILURE", "PERFORMANCE_ANOMALY",
            "API_CONNECTIVITY", "CONNECTION_HEALTH", "EXCHANGE_OFFLINE",
            "RISK_THRESHOLD_LEGACY",
        ]
        for name in legacy:
            assert hasattr(AlertType, name)

    def test_total_count(self):
        assert len(AlertType) >= 20

    def test_str_enum(self):
        assert isinstance(AlertType.VOL_Q90_CROSS, str)


# ── AlertFormatter ─────────────────────────────────────────────────

class TestAlertFormatter:

    @pytest.fixture
    def formatter(self):
        return AlertFormatter()

    @pytest.fixture
    def vol_alert_s1(self):
        return Alert(
            id="test-vol-1",
            alert_type=AlertType.VOL_Q90_CROSS,
            severity=AlertSeverity.S1,
            data={
                "current_value": 0.25,
                "adaptive_threshold": 0.20,
                "phase": "expansion",
                "confidence": 0.85,
                "portfolio_value": 100000,
            },
        )

    @pytest.fixture
    def regime_alert_s3(self):
        return Alert(
            id="test-regime-1",
            alert_type=AlertType.REGIME_FLIP,
            severity=AlertSeverity.S3,
            data={
                "current_value": 0.92,
                "phase": "correction",
                "confidence": 0.90,
                "portfolio_value": 200000,
            },
        )

    def test_templates_loaded(self, formatter):
        assert len(formatter.templates) > 0
        assert AlertType.VOL_Q90_CROSS in formatter.templates
        assert AlertType.REGIME_FLIP in formatter.templates

    def test_all_core_types_have_templates(self, formatter):
        """All 7 core types should have S1/S2/S3 templates."""
        core_types = [
            AlertType.VOL_Q90_CROSS, AlertType.REGIME_FLIP, AlertType.CORR_HIGH,
            AlertType.CORR_SPIKE, AlertType.CONTRADICTION_SPIKE,
            AlertType.DECISION_DROP, AlertType.EXEC_COST_SPIKE,
        ]
        for at in core_types:
            assert at in formatter.templates, f"Missing template for {at}"
            for sev in ["S1", "S2", "S3"]:
                assert sev in formatter.templates[at], f"Missing {sev} for {at}"

    def test_format_vol_alert_s1(self, formatter, vol_alert_s1):
        result = formatter.format_alert(vol_alert_s1)
        assert "action" in result
        assert "impact" in result
        assert "reasons" in result
        assert "details" in result
        assert result["severity"] == "S1"
        assert result["alert_type"] == "VOL_Q90_CROSS"
        assert "€" in result["impact"]

    def test_format_regime_alert_s3(self, formatter, regime_alert_s3):
        result = formatter.format_alert(regime_alert_s3)
        assert result["severity"] == "S3"
        assert "Freeze" in result["action"] or "freeze" in result["action"].lower() or "Arrêt" in result["action"]

    def test_impact_calculation(self, formatter, vol_alert_s1):
        result = formatter.format_alert(vol_alert_s1)
        # S1 impact_base = 0.5%, portfolio = 100k → €500
        assert "€" in result["impact"]

    def test_fallback_format_unknown_type(self, formatter):
        """Alert with no template should use fallback."""
        alert = Alert(
            id="test-unknown",
            alert_type=AlertType.PORTFOLIO_DRIFT,  # Legacy type may not have template
            severity=AlertSeverity.S1,
            data={},
        )
        result = formatter.format_alert(alert)
        assert "action" in result
        assert "impact" in result

    def test_fallback_format_explicit(self, formatter):
        alert = Alert(
            id="test-fallback",
            alert_type=AlertType.VOL_Q90_CROSS,
            severity=AlertSeverity.S1,
            data={},
        )
        fallback = formatter._fallback_format(alert)
        assert "Alerte" in fallback["action"]
        assert "€" in fallback["impact"]

    def test_reasons_are_list(self, formatter, vol_alert_s1):
        result = formatter.format_alert(vol_alert_s1)
        assert isinstance(result["reasons"], list)
        assert len(result["reasons"]) >= 2

    def test_predictive_template_format(self, formatter):
        """Phase 2C predictive alert formatting."""
        alert = Alert(
            id="test-spike",
            alert_type=AlertType.SPIKE_LIKELY,
            severity=AlertSeverity.S2,
            data={
                "asset_pair": "BTC/ETH",
                "horizon": "24h",
                "probability": 0.75,
                "confidence": 0.80,
                "model_confidence": 0.80,
                "portfolio_value": 100000,
            },
        )
        result = formatter.format_alert(alert)
        assert result["alert_type"] == "SPIKE_LIKELY"
        assert result["severity"] == "S2"

    def test_var_breach_template(self, formatter):
        alert = Alert(
            id="test-var",
            alert_type=AlertType.VAR_BREACH,
            severity=AlertSeverity.S1,
            data={
                "var_method": "parametric",
                "confidence_level": 0.95,
                "var_current": 15000,
                "var_limit": 10000,
                "portfolio_value": 500000,
            },
        )
        result = formatter.format_alert(alert)
        assert result["alert_type"] == "VAR_BREACH"


# ── AlertRule ──────────────────────────────────────────────────────

class TestAlertRule:

    def test_creation(self):
        rule = AlertRule(
            alert_type=AlertType.VOL_Q90_CROSS,
            base_threshold=0.15,
            severity_thresholds={"S1": 0.20, "S2": 0.35, "S3": 0.50},
            suggested_actions={"S1": {"type": "acknowledge"}},
        )
        assert rule.alert_type == AlertType.VOL_Q90_CROSS
        assert rule.base_threshold == 0.15
        assert rule.adaptive_multiplier == 1.0
        assert rule.hysteresis_minutes == 5

    def test_hysteresis_bounds(self):
        """hysteresis_minutes must be 1-60."""
        rule = AlertRule(
            alert_type=AlertType.VOL_Q90_CROSS,
            base_threshold=0.15,
            hysteresis_minutes=1,
            severity_thresholds={"S1": 0.20},
            suggested_actions={},
        )
        assert rule.hysteresis_minutes == 1

    def test_hysteresis_over_max_raises(self):
        with pytest.raises(Exception):
            AlertRule(
                alert_type=AlertType.VOL_Q90_CROSS,
                base_threshold=0.15,
                hysteresis_minutes=100,  # > 60
                severity_thresholds={"S1": 0.20},
                suggested_actions={},
            )


# ── Alert ──────────────────────────────────────────────────────────

class TestAlert:

    def test_creation(self):
        alert = Alert(
            id="test-1",
            alert_type=AlertType.REGIME_FLIP,
            severity=AlertSeverity.S2,
        )
        assert alert.id == "test-1"
        assert alert.alert_type == AlertType.REGIME_FLIP
        assert alert.severity == AlertSeverity.S2
        assert alert.acknowledged_at is None
        assert alert.escalation_count == 0

    def test_default_data(self):
        alert = Alert(
            id="test-2",
            alert_type=AlertType.VOL_Q90_CROSS,
            severity=AlertSeverity.S1,
        )
        assert alert.data == {}
        assert alert.suggested_action == {}

    def test_format_unified_message(self):
        alert = Alert(
            id="test-fmt",
            alert_type=AlertType.VOL_Q90_CROSS,
            severity=AlertSeverity.S1,
            data={
                "current_value": 0.25,
                "adaptive_threshold": 0.20,
                "phase": "expansion",
                "confidence": 0.85,
                "portfolio_value": 100000,
            },
        )
        result = alert.format_unified_message()
        assert "action" in result
        assert "impact" in result

    def test_escalation_fields(self):
        alert = Alert(
            id="test-esc",
            alert_type=AlertType.CORR_HIGH,
            severity=AlertSeverity.S3,
            escalation_sources=["alert-1", "alert-2"],
            escalation_count=2,
        )
        assert len(alert.escalation_sources) == 2
        assert alert.escalation_count == 2

    def test_snooze_field(self):
        alert = Alert(
            id="test-snooze",
            alert_type=AlertType.EXEC_COST_SPIKE,
            severity=AlertSeverity.S1,
            snooze_until=datetime.now() + timedelta(hours=1),
        )
        assert alert.snooze_until is not None


# ── AlertEvaluator ─────────────────────────────────────────────────

class TestAlertEvaluator:

    @pytest.fixture
    def evaluator(self):
        return AlertEvaluator(config={})

    def test_rules_loaded(self, evaluator):
        assert len(evaluator.alert_rules) >= 7
        assert AlertType.VOL_Q90_CROSS in evaluator.alert_rules
        assert AlertType.REGIME_FLIP in evaluator.alert_rules
        assert AlertType.CORR_HIGH in evaluator.alert_rules

    def test_rule_thresholds(self, evaluator):
        vol_rule = evaluator.alert_rules[AlertType.VOL_Q90_CROSS]
        assert vol_rule.base_threshold == 0.15
        assert "S1" in vol_rule.severity_thresholds
        assert "S3" in vol_rule.severity_thresholds

    # -- _extract_signal_value --

    def test_extract_vol_signal(self, evaluator):
        signals = {"volatility": {"BTC": 0.20, "ETH": 0.30}}
        val = evaluator._extract_signal_value(AlertType.VOL_Q90_CROSS, signals)
        assert val == 0.30  # max

    def test_extract_regime_signal(self, evaluator):
        signals = {"regime": {"bull": 0.8, "bear": 0.1, "neutral": 0.1}}
        val = evaluator._extract_signal_value(AlertType.REGIME_FLIP, signals)
        assert val == pytest.approx(0.7)  # max - min = 0.8 - 0.1

    def test_extract_corr_signal(self, evaluator):
        signals = {"correlation": {"avg_correlation": 0.85}}
        val = evaluator._extract_signal_value(AlertType.CORR_HIGH, signals)
        assert val == 0.85

    def test_extract_corr_spike_signal(self, evaluator):
        signals = {"correlation_spikes": [
            {"absolute_change": 0.35},
            {"absolute_change": 0.50},
        ]}
        val = evaluator._extract_signal_value(AlertType.CORR_SPIKE, signals)
        assert val == 0.50

    def test_extract_corr_spike_empty(self, evaluator):
        signals = {"correlation_spikes": []}
        val = evaluator._extract_signal_value(AlertType.CORR_SPIKE, signals)
        assert val == 0.0

    def test_extract_contradiction_signal(self, evaluator):
        signals = {"contradiction_index": 0.72}
        val = evaluator._extract_signal_value(AlertType.CONTRADICTION_SPIKE, signals)
        assert val == 0.72

    def test_extract_exec_cost_signal(self, evaluator):
        signals = {"execution_cost_bps": 45}
        val = evaluator._extract_signal_value(AlertType.EXEC_COST_SPIKE, signals)
        assert val == 45

    def test_extract_unknown_type(self, evaluator):
        val = evaluator._extract_signal_value(AlertType.PORTFOLIO_DRIFT, {"volatility": {}})
        assert val is None

    # -- _check_trigger_condition --

    def test_trigger_vol_above_threshold(self, evaluator):
        assert evaluator._check_trigger_condition(AlertType.VOL_Q90_CROSS, 0.25, 0.20) is True

    def test_trigger_vol_below_threshold(self, evaluator):
        assert evaluator._check_trigger_condition(AlertType.VOL_Q90_CROSS, 0.15, 0.20) is False

    def test_trigger_corr_spike(self, evaluator):
        assert evaluator._check_trigger_condition(AlertType.CORR_SPIKE, 0.40, 0.25) is True

    # -- _determine_severity --

    def test_severity_s1(self, evaluator):
        rule = evaluator.alert_rules[AlertType.VOL_Q90_CROSS]
        sev = evaluator._determine_severity(rule, 0.22)  # Above S1 (0.20) but below S2 (0.35)
        assert sev == AlertSeverity.S1

    def test_severity_s2(self, evaluator):
        rule = evaluator.alert_rules[AlertType.VOL_Q90_CROSS]
        sev = evaluator._determine_severity(rule, 0.40)  # Above S2 (0.35) but below S3 (0.50)
        assert sev == AlertSeverity.S2

    def test_severity_s3(self, evaluator):
        rule = evaluator.alert_rules[AlertType.VOL_Q90_CROSS]
        sev = evaluator._determine_severity(rule, 0.55)  # Above S3 (0.50)
        assert sev == AlertSeverity.S3

    # -- _check_hysteresis --

    def test_hysteresis_first_trigger_returns_false(self, evaluator):
        """First trigger should return False (starts the timer)."""
        result = evaluator._check_hysteresis(AlertType.VOL_Q90_CROSS, 5)
        assert result is False

    def test_hysteresis_immediate_second_false(self, evaluator):
        """Immediately after first trigger, still returns False (0 minutes elapsed)."""
        evaluator._check_hysteresis(AlertType.VOL_Q90_CROSS, 5)
        result = evaluator._check_hysteresis(AlertType.VOL_Q90_CROSS, 5)
        assert result is False

    def test_hysteresis_with_zero_minutes(self, evaluator):
        """With 0 required minutes, should pass after first trigger."""
        evaluator._check_hysteresis(AlertType.CORR_HIGH, 0)
        result = evaluator._check_hysteresis(AlertType.CORR_HIGH, 0)
        assert result is True

    # -- _calculate_adaptive_threshold --

    def test_adaptive_threshold_base_case(self, evaluator):
        """Without phase context, threshold = base."""
        rule = evaluator.alert_rules[AlertType.VOL_Q90_CROSS]
        signals = {"volatility": {"BTC": 0.10}, "confidence": 0.75}
        threshold = evaluator._calculate_adaptive_threshold(rule, signals)
        assert threshold == pytest.approx(rule.base_threshold, rel=0.2)

    def test_adaptive_threshold_high_vol_stricter(self, evaluator):
        """High vol → market_factor increases → threshold increases."""
        rule = evaluator.alert_rules[AlertType.VOL_Q90_CROSS]
        low_vol_signals = {"volatility": {"BTC": 0.10}, "confidence": 0.75}
        high_vol_signals = {"volatility": {"BTC": 0.35}, "confidence": 0.75}
        t_low = evaluator._calculate_adaptive_threshold(rule, low_vol_signals)
        t_high = evaluator._calculate_adaptive_threshold(rule, high_vol_signals)
        assert t_high >= t_low

    def test_adaptive_threshold_low_confidence(self, evaluator):
        """Low confidence → less strict (lower threshold)."""
        rule = evaluator.alert_rules[AlertType.VOL_Q90_CROSS]
        signals_high_conf = {"volatility": {"BTC": 0.10}, "confidence": 0.80}
        signals_low_conf = {"volatility": {"BTC": 0.10}, "confidence": 0.50}
        t_high = evaluator._calculate_adaptive_threshold(rule, signals_high_conf)
        t_low = evaluator._calculate_adaptive_threshold(rule, signals_low_conf)
        assert t_low <= t_high

    # -- evaluate_alert (integration) --

    def test_evaluate_no_trigger(self, evaluator):
        """Low vol should not trigger alert."""
        signals = {"volatility": {"BTC": 0.05}, "confidence": 0.80}
        result = evaluator.evaluate_alert(AlertType.VOL_Q90_CROSS, signals)
        assert result is None

    def test_evaluate_unknown_type(self, evaluator):
        """Unknown alert type returns None."""
        result = evaluator.evaluate_alert(AlertType.PORTFOLIO_DRIFT, {})
        assert result is None

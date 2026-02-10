"""Tests for services/risk_scoring.py — canonical risk score + level mapping (50+ tests)."""

import pytest
from services.risk_scoring import (
    assess_risk_level,
    score_to_level,
    RISK_LEVEL_THRESHOLDS,
)


# ===========================================================================
# score_to_level — comprehensive boundary tests
# ===========================================================================

class TestScoreToLevelMapping:
    def test_very_low_at_80(self):
        assert score_to_level(80) == "very_low"

    def test_very_low_at_100(self):
        assert score_to_level(100) == "very_low"

    def test_very_low_at_95(self):
        assert score_to_level(95) == "very_low"

    def test_low_at_65(self):
        assert score_to_level(65) == "low"

    def test_low_at_79(self):
        assert score_to_level(79) == "low"

    def test_medium_at_50(self):
        assert score_to_level(50) == "medium"

    def test_medium_at_64(self):
        assert score_to_level(64) == "medium"

    def test_high_at_35(self):
        assert score_to_level(35) == "high"

    def test_high_at_49(self):
        assert score_to_level(49) == "high"

    def test_very_high_at_20(self):
        assert score_to_level(20) == "very_high"

    def test_very_high_at_34(self):
        assert score_to_level(34) == "very_high"

    def test_critical_at_19(self):
        assert score_to_level(19) == "critical"

    def test_critical_at_0(self):
        assert score_to_level(0) == "critical"

    def test_critical_at_10(self):
        assert score_to_level(10) == "critical"

    def test_clamp_above_100(self):
        assert score_to_level(150) == "very_low"

    def test_clamp_below_0(self):
        assert score_to_level(-50) == "critical"

    def test_float_input(self):
        assert score_to_level(79.9) == "low"
        assert score_to_level(80.0) == "very_low"

    def test_boundary_exact_thresholds(self):
        for level, threshold in RISK_LEVEL_THRESHOLDS.items():
            result = score_to_level(threshold)
            assert result == level, f"score_to_level({threshold}) = {result}, expected {level}"


# ===========================================================================
# assess_risk_level — VaR component
# ===========================================================================

class TestVaRComponent:
    def _base(self, **kw):
        defaults = dict(var_metrics={"var_95": 0.10}, sharpe_ratio=1.0,
                        max_drawdown=-0.20, volatility=0.30)
        defaults.update(kw)
        return assess_risk_level(**defaults)

    def test_very_high_var_penalty(self):
        assert self._base(var_metrics={"var_95": 0.30})["breakdown"]["var_95"] == -30

    def test_high_var_penalty(self):
        assert self._base(var_metrics={"var_95": 0.20})["breakdown"]["var_95"] == -15

    def test_low_var_bonus(self):
        assert self._base(var_metrics={"var_95": 0.03})["breakdown"]["var_95"] == 10

    def test_moderate_low_var_bonus(self):
        assert self._base(var_metrics={"var_95": 0.08})["breakdown"]["var_95"] == 5

    def test_neutral_var(self):
        assert self._base(var_metrics={"var_95": 0.12})["breakdown"]["var_95"] == 0

    def test_missing_var_key_defaults_zero(self):
        # var_95 absent → 0.0 → abs < 0.05 → +10
        assert self._base(var_metrics={})["breakdown"]["var_95"] == 10


# ===========================================================================
# assess_risk_level — Sharpe component
# ===========================================================================

class TestSharpeComponent:
    def _base(self, **kw):
        defaults = dict(var_metrics={"var_95": 0.10}, sharpe_ratio=1.0,
                        max_drawdown=-0.20, volatility=0.30)
        defaults.update(kw)
        return assess_risk_level(**defaults)

    def test_negative_sharpe(self):
        assert self._base(sharpe_ratio=-0.5)["breakdown"]["sharpe"] == -15

    def test_excellent_sharpe(self):
        assert self._base(sharpe_ratio=2.5)["breakdown"]["sharpe"] == 20

    def test_very_good_sharpe(self):
        assert self._base(sharpe_ratio=1.7)["breakdown"]["sharpe"] == 15

    def test_good_sharpe(self):
        assert self._base(sharpe_ratio=1.2)["breakdown"]["sharpe"] == 10

    def test_decent_sharpe(self):
        assert self._base(sharpe_ratio=0.7)["breakdown"]["sharpe"] == 5

    def test_low_sharpe(self):
        assert self._base(sharpe_ratio=0.3)["breakdown"]["sharpe"] == 0

    def test_zero_sharpe(self):
        assert self._base(sharpe_ratio=0.0)["breakdown"]["sharpe"] == 0


# ===========================================================================
# assess_risk_level — Drawdown component
# ===========================================================================

class TestDrawdownComponent:
    def _base(self, **kw):
        defaults = dict(var_metrics={"var_95": 0.10}, sharpe_ratio=1.0,
                        max_drawdown=-0.20, volatility=0.30)
        defaults.update(kw)
        return assess_risk_level(**defaults)

    def test_extreme_dd(self):
        assert self._base(max_drawdown=-0.75)["breakdown"]["drawdown"] == -22

    def test_severe_dd(self):
        assert self._base(max_drawdown=-0.55)["breakdown"]["drawdown"] == -15

    def test_moderate_dd(self):
        assert self._base(max_drawdown=-0.35)["breakdown"]["drawdown"] == -10

    def test_low_dd_bonus(self):
        assert self._base(max_drawdown=-0.08)["breakdown"]["drawdown"] == 10

    def test_light_dd_bonus(self):
        assert self._base(max_drawdown=-0.15)["breakdown"]["drawdown"] == 5

    def test_neutral_dd(self):
        assert self._base(max_drawdown=-0.25)["breakdown"]["drawdown"] == 0


# ===========================================================================
# assess_risk_level — Volatility component
# ===========================================================================

class TestVolatilityComponent:
    def _base(self, **kw):
        defaults = dict(var_metrics={"var_95": 0.10}, sharpe_ratio=1.0,
                        max_drawdown=-0.20, volatility=0.30)
        defaults.update(kw)
        return assess_risk_level(**defaults)

    def test_extreme_vol(self):
        assert self._base(volatility=1.2)["breakdown"]["volatility"] == -10

    def test_high_vol(self):
        assert self._base(volatility=0.70)["breakdown"]["volatility"] == -5

    def test_low_vol(self):
        assert self._base(volatility=0.15)["breakdown"]["volatility"] == 10

    def test_moderate_low_vol(self):
        assert self._base(volatility=0.35)["breakdown"]["volatility"] == 5

    def test_neutral_vol(self):
        assert self._base(volatility=0.50)["breakdown"]["volatility"] == 0


# ===========================================================================
# assess_risk_level — Structural penalties
# ===========================================================================

class TestStructuralPenalties:
    def _base(self, **kw):
        defaults = dict(var_metrics={"var_95": 0.10}, sharpe_ratio=1.0,
                        max_drawdown=-0.20, volatility=0.30)
        defaults.update(kw)
        return assess_risk_level(**defaults)

    # --- Memecoins ---
    def test_memecoins_extreme(self):
        assert self._base(memecoins_pct=0.80)["breakdown"]["memecoins"] == -22

    def test_memecoins_high(self):
        assert self._base(memecoins_pct=0.55)["breakdown"]["memecoins"] == -15

    def test_memecoins_moderate(self):
        assert self._base(memecoins_pct=0.35)["breakdown"]["memecoins"] == -10

    def test_memecoins_light(self):
        assert self._base(memecoins_pct=0.20)["breakdown"]["memecoins"] == -6

    def test_memecoins_minimal(self):
        assert self._base(memecoins_pct=0.10)["breakdown"]["memecoins"] == -3

    def test_memecoins_none(self):
        assert self._base(memecoins_pct=0.02)["breakdown"]["memecoins"] == 0

    def test_memecoins_hysteresis_zone(self):
        """48-52% zone uses linear interpolation between -10 and -15."""
        delta = self._base(memecoins_pct=0.50)["breakdown"]["memecoins"]
        assert -15 <= delta <= -10

    # --- HHI Concentration ---
    def test_hhi_very_concentrated(self):
        assert self._base(hhi=0.50)["breakdown"]["concentration"] == -12

    def test_hhi_concentrated(self):
        assert self._base(hhi=0.30)["breakdown"]["concentration"] == -8

    def test_hhi_slight(self):
        assert self._base(hhi=0.18)["breakdown"]["concentration"] == -3

    def test_hhi_diversified(self):
        assert self._base(hhi=0.10)["breakdown"]["concentration"] == 0

    # --- GRI ---
    def test_gri_very_risky(self):
        assert self._base(gri=8.0)["breakdown"]["group_risk"] == -10

    def test_gri_risky(self):
        assert self._base(gri=6.5)["breakdown"]["group_risk"] == -7

    def test_gri_moderate(self):
        assert self._base(gri=5.5)["breakdown"]["group_risk"] == -4

    def test_gri_safe(self):
        assert self._base(gri=2.0)["breakdown"]["group_risk"] == 5

    def test_gri_neutral(self):
        assert self._base(gri=4.0)["breakdown"]["group_risk"] == 0

    # --- Diversification ---
    def test_div_very_low(self):
        assert self._base(diversification_ratio=0.3)["breakdown"]["diversification"] == -10

    def test_div_low(self):
        assert self._base(diversification_ratio=0.5)["breakdown"]["diversification"] == -5

    def test_div_high(self):
        assert self._base(diversification_ratio=0.9)["breakdown"]["diversification"] == 5

    def test_div_neutral(self):
        assert self._base(diversification_ratio=0.7)["breakdown"]["diversification"] == 0


# ===========================================================================
# assess_risk_level — Integration
# ===========================================================================

class TestAssessRiskLevelIntegration:
    def test_returns_required_keys(self):
        r = assess_risk_level(
            var_metrics={"var_95": 0.10}, sharpe_ratio=1.0,
            max_drawdown=-0.20, volatility=0.30,
        )
        assert "score" in r and "level" in r and "breakdown" in r

    def test_breakdown_sums_to_score_minus_50(self):
        r = assess_risk_level(
            var_metrics={"var_95": 0.12}, sharpe_ratio=1.0,
            max_drawdown=-0.25, volatility=0.50,
        )
        total = sum(r["breakdown"].values())
        assert abs((50 + total) - r["score"]) < 0.1

    def test_score_clamped_floor(self):
        r = assess_risk_level(
            var_metrics={"var_95": 0.30}, sharpe_ratio=-1.0,
            max_drawdown=-0.80, volatility=1.5,
            memecoins_pct=0.90, hhi=0.50, gri=9.0, diversification_ratio=0.2,
        )
        assert r["score"] >= 0

    def test_score_clamped_ceiling(self):
        r = assess_risk_level(
            var_metrics={"var_95": 0.02}, sharpe_ratio=2.5,
            max_drawdown=-0.05, volatility=0.10,
            memecoins_pct=0.0, hhi=0.05, gri=2.0, diversification_ratio=0.95,
        )
        assert r["score"] <= 100

    def test_optimal_portfolio(self):
        r = assess_risk_level(
            var_metrics={"var_95": 0.02}, sharpe_ratio=2.5,
            max_drawdown=-0.05, volatility=0.10,
        )
        assert r["score"] >= 80
        assert r["level"] in ("very_low", "low")

    def test_terrible_portfolio(self):
        r = assess_risk_level(
            var_metrics={"var_95": 0.30}, sharpe_ratio=-1.0,
            max_drawdown=-0.80, volatility=1.5,
            memecoins_pct=0.80, hhi=0.50, gri=9.0, diversification_ratio=0.2,
        )
        assert r["score"] <= 20
        assert r["level"] in ("critical", "very_high")

    def test_level_matches_score(self):
        r = assess_risk_level(
            var_metrics={"var_95": 0.10}, sharpe_ratio=1.5,
            max_drawdown=-0.15, volatility=0.25,
        )
        assert r["level"] == score_to_level(r["score"])

    def test_neutral_defaults(self):
        """Default structural params: memecoins=0, hhi=0, gri=5.0, div=1.0."""
        r = assess_risk_level(
            var_metrics={"var_95": 0.12}, sharpe_ratio=0.3,
            max_drawdown=-0.25, volatility=0.50,
        )
        assert r["breakdown"]["memecoins"] == 0
        assert r["breakdown"]["concentration"] == 0
        assert r["breakdown"]["group_risk"] == 0
        # diversification_ratio=1.0 (default) > 0.8 → +5 bonus
        assert r["breakdown"]["diversification"] == 5

    def test_higher_var_lower_score(self):
        low = assess_risk_level(var_metrics={"var_95": 0.04}, sharpe_ratio=1.0,
                                max_drawdown=-0.20, volatility=0.40)
        high = assess_risk_level(var_metrics={"var_95": 0.30}, sharpe_ratio=1.0,
                                 max_drawdown=-0.20, volatility=0.40)
        assert low["score"] > high["score"]

    def test_higher_sharpe_higher_score(self):
        low = assess_risk_level(var_metrics={"var_95": 0.12}, sharpe_ratio=-0.5,
                                max_drawdown=-0.20, volatility=0.40)
        high = assess_risk_level(var_metrics={"var_95": 0.12}, sharpe_ratio=2.0,
                                 max_drawdown=-0.20, volatility=0.40)
        assert high["score"] > low["score"]


# ===========================================================================
# Thresholds config
# ===========================================================================

class TestThresholdsConfig:
    def test_thresholds_descending(self):
        levels = ["very_low", "low", "medium", "high", "very_high", "critical"]
        vals = [RISK_LEVEL_THRESHOLDS[l] for l in levels]
        assert vals == sorted(vals, reverse=True)

    def test_critical_is_zero(self):
        assert RISK_LEVEL_THRESHOLDS["critical"] == 0

    def test_very_low_is_80(self):
        assert RISK_LEVEL_THRESHOLDS["very_low"] == 80

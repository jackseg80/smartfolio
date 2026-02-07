"""
Tests for DI Phase Threshold consistency and SmartFolio Replica risk_budget.

Ensures CLAUDE.md canonical thresholds are respected everywhere:
  - cycle < 70       = bearish  (factor 0.85)
  - 70 <= cycle < 90 = moderate (factor 1.0)
  - cycle >= 90      = bullish  (factor 1.05)

Also tests the production risk_budget formula in SmartFolio Replica:
  blendedScore = 0.5×cycle + 0.3×onchain + 0.2×risk
  risk_factor  = 0.5 + 0.5 × (riskScore / 100)
  baseRisky    = clamp((blended - 35) / 45, 0, 1)
  risky        = clamp(baseRisky × risk_factor, 0.20, 0.85)
"""

import pytest
from services.di_backtest.historical_di_calculator import (
    HistoricalDICalculator,
    PhaseFactors,
)
from services.di_backtest.trading_strategies import DISmartfolioReplicaStrategy, ReplicaParams


class TestHistoricalDIPhaseThresholds:
    """Verify _determine_phase uses CLAUDE.md canonical thresholds."""

    def setup_method(self):
        self.calc = HistoricalDICalculator()

    # --- bearish: cycle < 70 ---

    @pytest.mark.parametrize("score", [0, 10, 30, 40, 50, 60, 69, 69.9])
    def test_bearish_below_70(self, score):
        phase, factor = self.calc._determine_phase(score)
        assert phase == "bearish", f"cycle={score} should be bearish, got {phase}"
        assert factor == 0.85, f"cycle={score} factor should be 0.85, got {factor}"

    # --- moderate: 70 <= cycle < 90 ---

    @pytest.mark.parametrize("score", [70, 75, 80, 85, 89, 89.9])
    def test_moderate_70_to_90(self, score):
        phase, factor = self.calc._determine_phase(score)
        assert phase == "moderate", f"cycle={score} should be moderate, got {phase}"
        assert factor == 1.0, f"cycle={score} factor should be 1.0, got {factor}"

    # --- bullish: cycle >= 90 ---

    @pytest.mark.parametrize("score", [90, 95, 100])
    def test_bullish_90_and_above(self, score):
        phase, factor = self.calc._determine_phase(score)
        assert phase == "bullish", f"cycle={score} should be bullish, got {phase}"
        assert factor == 1.05, f"cycle={score} factor should be 1.05, got {factor}"

    # --- boundary precision ---

    def test_boundary_at_70(self):
        """69.9 = bearish, 70.0 = moderate"""
        phase_below, _ = self.calc._determine_phase(69.9)
        phase_at, _ = self.calc._determine_phase(70.0)
        assert phase_below == "bearish"
        assert phase_at == "moderate"

    def test_boundary_at_90(self):
        """89.9 = moderate, 90.0 = bullish"""
        phase_below, _ = self.calc._determine_phase(89.9)
        phase_at, _ = self.calc._determine_phase(90.0)
        assert phase_below == "moderate"
        assert phase_at == "bullish"

    # --- custom phase factors ---

    def test_custom_phase_factors(self):
        custom = PhaseFactors(bearish=0.70, moderate=0.90, bullish=1.10)
        calc = HistoricalDICalculator(phase_factors=custom)

        phase, factor = calc._determine_phase(50)
        assert phase == "bearish"
        assert factor == 0.70

        phase, factor = calc._determine_phase(75)
        assert phase == "moderate"
        assert factor == 0.90

        phase, factor = calc._determine_phase(95)
        assert phase == "bullish"
        assert factor == 1.10

    # --- default PhaseFactors values ---

    def test_default_phase_factors(self):
        pf = PhaseFactors()
        assert pf.bearish == 0.85
        assert pf.moderate == 1.0
        assert pf.bullish == 1.05


class TestSmartfolioReplicaRiskBudget:
    """Verify SmartFolio Replica uses production risk_budget formula."""

    def test_risk_budget_bearish_low_risk(self):
        """Bear market + low risk score → very conservative (~20% risky)"""
        # cycle=50, onchain=40, risk=30
        # blended = 0.5*50 + 0.3*40 + 0.2*30 = 25+12+6 = 43
        # risk_factor = 0.5 + 0.5*(30/100) = 0.65
        # baseRisky = clamp((43-35)/45, 0, 1) = 0.178
        # risky = clamp(0.178 * 0.65, 0.20, 0.85) = clamp(0.116, 0.20, 0.85) = 0.20
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(50, 40, 30)
        assert risky == pytest.approx(0.20, abs=0.01), f"Expected ~0.20, got {risky}"

    def test_risk_budget_moderate_mid_risk(self):
        """Moderate market + mid risk → balanced allocation"""
        # cycle=80, onchain=60, risk=50
        # blended = 40+18+10 = 68
        # risk_factor = 0.5 + 0.25 = 0.75
        # baseRisky = (68-35)/45 = 0.733
        # risky = clamp(0.733*0.75, 0.20, 0.85) = 0.55
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(80, 60, 50)
        assert risky == pytest.approx(0.55, abs=0.01), f"Expected ~0.55, got {risky}"

    def test_risk_budget_bullish_high_risk(self):
        """Bull market + high risk score → aggressive (85% risky clamped)"""
        # cycle=95, onchain=80, risk=85
        # blended = 47.5+24+17 = 88.5
        # risk_factor = 0.5 + 0.425 = 0.925
        # baseRisky = clamp((88.5-35)/45, 0, 1) = 1.0
        # risky = clamp(1.0*0.925, 0.20, 0.85) = 0.85
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(95, 80, 85)
        assert risky == pytest.approx(0.85, abs=0.01), f"Expected ~0.85, got {risky}"

    def test_risk_budget_floor_at_20_pct(self):
        """Minimum risky allocation is 20%"""
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(0, 0, 0)
        assert risky == pytest.approx(0.20, abs=0.01)

    def test_risk_budget_ceiling_at_85_pct(self):
        """Maximum risky allocation is 85%"""
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(100, 100, 100)
        assert risky == pytest.approx(0.85, abs=0.01)

    def test_risk_budget_blended_formula(self):
        """Verify blendedScore = 0.5×cycle + 0.3×onchain + 0.2×risk"""
        # cycle=60, onchain=70, risk=40
        # blended = 30 + 21 + 8 = 59
        # risk_factor = 0.5 + 0.2 = 0.70
        # baseRisky = (59-35)/45 = 0.533
        # risky = clamp(0.533*0.70, 0.20, 0.85) = 0.373
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(60, 70, 40)
        expected = max(0.20, min(0.85, 0.533 * 0.70))
        assert risky == pytest.approx(expected, abs=0.01)

    def test_risk_factor_range(self):
        """risk_factor should be in [0.5, 1.0]"""
        # risk=0 → 0.5, risk=100 → 1.0
        # With cycle=70, onchain=50: blended=35+15+0=50 for risk=0 and 35+15+20=70 for risk=100
        risky_min_risk = DISmartfolioReplicaStrategy._compute_risk_budget(70, 50, 0)
        risky_max_risk = DISmartfolioReplicaStrategy._compute_risk_budget(70, 50, 100)
        # Higher risk score → higher risky allocation
        assert risky_max_risk > risky_min_risk

    def test_compared_to_old_fixed_stables(self):
        """
        Regression: old code used fixed 30% stables in bearish.
        New code with low components should be MORE conservative.
        """
        # Old: bearish → 70% risky
        # New: cycle=50, onchain=40, risk=30 → conservative
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(50, 40, 30)
        assert risky < 0.70, (
            f"With low scores (50/40/30), risky should be << 70%, got {risky*100:.0f}%"
        )


class TestSmartfolioReplicaOverrides:
    """Verify production override logic in SmartFolio Replica."""

    def test_onchain_divergence_adds_stables(self):
        """On-chain divergence |blended - onchain| >= 30 → +10% stables"""
        # cycle=95, onchain=30, risk=80 → blended=72.5, divergence=|72.5-30|=42.5 → triggers
        risky_with_divergence = DISmartfolioReplicaStrategy._compute_risk_budget(95, 30, 80)
        # Without divergence: cycle=70, onchain=70, risk=80 → blended=72, divergence=|72-70|=2
        risky_no_divergence = DISmartfolioReplicaStrategy._compute_risk_budget(70, 70, 80)
        # The divergent case should have MORE stables (less risky)
        assert risky_with_divergence < risky_no_divergence, (
            f"Divergent scores should give less risky: {risky_with_divergence:.2f} vs {risky_no_divergence:.2f}"
        )

    def test_current_live_scores_risk_budget(self):
        """
        With live scores cycle=94, onchain=64, risk=78:
        blended = 0.5*94 + 0.3*64 + 0.2*78 = 81.8
        risk_factor = 0.5 + 0.5*(78/100) = 0.89
        baseRisky = clamp((81.8-35)/45, 0, 1) = 1.0
        risky = clamp(1.0 * 0.89, 0.20, 0.85) = 0.85

        On-chain divergence: |blended - onchain| = |81.8 - 64| = 17.8 < 30 → NO override
        (Production also does NOT trigger this override for these scores)

        Note: Production shows ~53% stables because it also applies
        computeExposureCap() + governance cap_daily on top of risk_budget.
        """
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(94, 64, 78)
        # Risk budget at ceiling, no override triggered (divergence < 30)
        assert risky == pytest.approx(0.85, abs=0.01), (
            f"With scores 94/64/78, risky should be 85% (risk_budget ceiling), got {risky*100:.0f}%"
        )

    def test_low_risk_forces_50pct_stables(self):
        """Risk <= 30 → force stables >= 50%"""
        # Even with bullish cycle, low risk should force protection
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(90, 70, 25)
        stables = 1.0 - risky
        assert stables >= 0.50, (
            f"With risk=25, stables should be >= 50%, got {stables*100:.0f}%"
        )

    def test_no_divergence_no_penalty(self):
        """When scores agree, no divergence penalty applied"""
        # cycle=80, onchain=75 → divergence=5 < 30 → no override
        risky = DISmartfolioReplicaStrategy._compute_risk_budget(80, 75, 70)
        # Standard formula should apply: blended≈78.5, risk_factor=0.85
        # baseRisky≈0.97, risky≈0.82
        assert risky >= 0.70, (
            f"With aligned scores (80/75/70), should be aggressive: got {risky*100:.0f}%"
        )

    def test_adaptive_weights_reduce_cycle_in_contradiction(self):
        """Contradiction should reduce cycle weight, increase risk weight"""
        w_c, w_o, w_r = DISmartfolioReplicaStrategy._compute_adaptive_weights(95, 40, 60)
        # With divergence=55 → high contradiction
        # Cycle weight should be reduced, risk increased
        assert w_c < 0.50, f"Cycle weight should be < 50% with contradiction, got {w_c:.2f}"
        assert w_r > 0.20, f"Risk weight should be > 20% with contradiction, got {w_r:.2f}"
        assert abs(w_c + w_o + w_r - 1.0) < 0.001, "Weights must sum to 1.0"

    def test_adaptive_weights_no_change_when_aligned(self):
        """When scores agree, weights should be near-standard"""
        w_c, w_o, w_r = DISmartfolioReplicaStrategy._compute_adaptive_weights(70, 70, 70)
        # No divergence → standard weights (0.50, 0.30, 0.20)
        assert abs(w_c - 0.50) < 0.05, f"Cycle weight should be ~0.50, got {w_c:.2f}"
        assert abs(w_o - 0.30) < 0.05, f"OnChain weight should be ~0.30, got {w_o:.2f}"
        assert abs(w_r - 0.20) < 0.05, f"Risk weight should be ~0.20, got {w_r:.2f}"


class TestExposureCap:
    """Verify exposure cap matches production computeExposureCap() logic."""

    def test_bear_market_caps_at_40pct(self):
        """Bear market (blended <= 25) → max 40% risky"""
        # blended=20, risk=30, low DI, no vol
        cap = DISmartfolioReplicaStrategy._compute_exposure_cap(20, 30, 25, 0.0)
        assert cap <= 0.40, f"Bear market cap should be <= 40%, got {cap*100:.0f}%"

    def test_correction_caps_at_70pct(self):
        """Correction (blended 26-50) → max 70% risky"""
        cap = DISmartfolioReplicaStrategy._compute_exposure_cap(40, 50, 45, 0.0)
        assert cap <= 0.70, f"Correction cap should be <= 70%, got {cap*100:.0f}%"

    def test_expansion_floor_at_75pct(self):
        """Expansion (blended > 75) → min 75% risky (regime floor)"""
        # Even with penalties, floor holds
        cap = DISmartfolioReplicaStrategy._compute_exposure_cap(82, 78, 50, 0.80)
        assert cap >= 0.75, f"Expansion floor should be >= 75%, got {cap*100:.0f}%"

    def test_high_volatility_reduces_cap(self):
        """High BTC volatility → lower cap"""
        cap_low_vol = DISmartfolioReplicaStrategy._compute_exposure_cap(60, 60, 60, 0.15)
        cap_high_vol = DISmartfolioReplicaStrategy._compute_exposure_cap(60, 60, 60, 0.70)
        assert cap_high_vol < cap_low_vol, (
            f"High vol should reduce cap: {cap_high_vol*100:.0f}% vs {cap_low_vol*100:.0f}%"
        )

    def test_low_di_reduces_signal_quality(self):
        """Low DI → lower signal quality → lower cap"""
        cap_high_di = DISmartfolioReplicaStrategy._compute_exposure_cap(60, 60, 80, 0.30)
        cap_low_di = DISmartfolioReplicaStrategy._compute_exposure_cap(60, 60, 20, 0.30)
        assert cap_low_di <= cap_high_di, (
            f"Low DI should reduce cap: {cap_low_di*100:.0f}% vs {cap_high_di*100:.0f}%"
        )

    def test_expansion_high_risk_gets_boost(self):
        """Expansion + Risk >= 80 → regime floor boosted (but min stays 65)"""
        cap_high_risk = DISmartfolioReplicaStrategy._compute_exposure_cap(82, 85, 80, 0.30)
        # With high risk, the Expansion boost applies (regimeMin=65 instead of 75)
        # but since this is a boost to floor, not ceiling, cap should still be >= 65%
        assert cap_high_risk >= 0.65, f"Expansion+high risk floor >= 65%, got {cap_high_risk*100:.0f}%"

    def test_base_cap_grid_coverage(self):
        """Verify the blended/risk grid produces expected base values"""
        # bs >= 70, rs >= 80 → base 90
        cap = DISmartfolioReplicaStrategy._compute_exposure_cap(75, 85, 70, 0.0)
        assert cap >= 0.85, f"High blended+risk → cap >= 85%, got {cap*100:.0f}%"

        # bs < 55 → base 55
        cap = DISmartfolioReplicaStrategy._compute_exposure_cap(40, 40, 40, 0.0)
        assert cap <= 0.55, f"Low blended → cap <= 55%, got {cap*100:.0f}%"


class TestContradictionIndex:
    """Verify contradiction index reconstruction from historical data."""

    def test_no_contradiction_when_calm(self):
        """Low vol + moderate cycle + aligned DI → zero contradiction"""
        ci = DISmartfolioReplicaStrategy._compute_contradiction_index(
            btc_volatility=0.30, cycle_score=60, onchain_score=55, di_value=55
        )
        assert ci == 0.0, f"Expected 0 contradiction in calm market, got {ci:.2f}"

    def test_high_vol_bull_triggers_check1(self):
        """High vol + bullish cycle → check 1 fires (0.3 / 3 = 0.10)"""
        ci = DISmartfolioReplicaStrategy._compute_contradiction_index(
            btc_volatility=0.80, cycle_score=85, onchain_score=80, di_value=70
        )
        assert ci > 0.0, f"High vol + bull should trigger contradiction, got {ci:.2f}"
        assert ci == pytest.approx(0.10, abs=0.01), f"Expected ~0.10 (0.3/3), got {ci:.2f}"

    def test_di_fear_bull_triggers_check2(self):
        """Low DI (fear) + bullish cycle → check 2 fires"""
        ci = DISmartfolioReplicaStrategy._compute_contradiction_index(
            btc_volatility=0.30, cycle_score=80, onchain_score=75, di_value=20
        )
        assert ci > 0.0, f"DI fear + bull should trigger contradiction, got {ci:.2f}"

    def test_di_greed_bear_triggers_check2(self):
        """High DI (greed) + bearish cycle → check 2 fires"""
        ci = DISmartfolioReplicaStrategy._compute_contradiction_index(
            btc_volatility=0.30, cycle_score=50, onchain_score=45, di_value=80
        )
        assert ci > 0.0, f"DI greed + bear should trigger contradiction, got {ci:.2f}"

    def test_score_divergence_triggers_check3(self):
        """Large cycle-onchain gap → check 3 fires"""
        ci = DISmartfolioReplicaStrategy._compute_contradiction_index(
            btc_volatility=0.30, cycle_score=90, onchain_score=40, di_value=60
        )
        assert ci > 0.0, f"Score divergence should trigger contradiction, got {ci:.2f}"

    def test_all_checks_fire_max_contradiction(self):
        """All 3 checks fire → maximum contradiction"""
        # vol=0.80 (high), cycle=80 (bull), onchain=30 (divergence=50 >=40),
        # di=25 (fear + bull = check 2)
        ci = DISmartfolioReplicaStrategy._compute_contradiction_index(
            btc_volatility=0.80, cycle_score=80, onchain_score=30, di_value=25
        )
        # All 3 checks: 0.3 + 0.25 + 0.2 = 0.75 / 3 = 0.25
        assert ci == pytest.approx(0.25, abs=0.01), f"All checks → ~0.25, got {ci:.2f}"

    def test_contradiction_bounded_0_1(self):
        """Contradiction index always in [0, 1]"""
        # Test extreme inputs
        ci = DISmartfolioReplicaStrategy._compute_contradiction_index(
            btc_volatility=5.0, cycle_score=100, onchain_score=0, di_value=0
        )
        assert 0.0 <= ci <= 1.0, f"CI out of bounds: {ci}"


class TestGovernancePenalty:
    """Verify governance penalty applies proportional reduction."""

    def test_no_penalty_low_contradiction(self):
        """Contradiction < 0.20 → zero penalty"""
        penalty = DISmartfolioReplicaStrategy._compute_governance_penalty(0.10)
        assert penalty == 0.0, f"Low contradiction should give 0 penalty, got {penalty:.3f}"

    def test_moderate_penalty(self):
        """Contradiction ~0.35 → moderate penalty"""
        penalty = DISmartfolioReplicaStrategy._compute_governance_penalty(0.35)
        assert 0.0 < penalty < 0.10, (
            f"Moderate contradiction should give small penalty, got {penalty*100:.1f}%"
        )

    def test_high_contradiction_heavy_penalty(self):
        """Contradiction ~0.70 → significant penalty"""
        penalty = DISmartfolioReplicaStrategy._compute_governance_penalty(0.70)
        assert penalty >= 0.15, (
            f"High contradiction should give >= 15% penalty, got {penalty*100:.1f}%"
        )

    def test_penalty_capped_at_25pct(self):
        """Maximum penalty is 25% even with extreme contradiction"""
        penalty = DISmartfolioReplicaStrategy._compute_governance_penalty(1.0, 2.0)
        assert penalty <= 0.25, f"Penalty should be capped at 25%, got {penalty*100:.1f}%"

    def test_volatility_amplifies_penalty(self):
        """High volatility increases the penalty"""
        penalty_low_vol = DISmartfolioReplicaStrategy._compute_governance_penalty(0.40, 0.20)
        penalty_high_vol = DISmartfolioReplicaStrategy._compute_governance_penalty(0.40, 0.90)
        assert penalty_high_vol > penalty_low_vol, (
            f"High vol should amplify: {penalty_high_vol*100:.1f}% vs {penalty_low_vol*100:.1f}%"
        )

    def test_zero_contradiction_zero_penalty(self):
        """Zero contradiction → zero penalty regardless of vol"""
        penalty = DISmartfolioReplicaStrategy._compute_governance_penalty(0.0, 1.5)
        assert penalty == 0.0, f"Zero contradiction should give 0 penalty, got {penalty:.3f}"

    def test_penalty_preserves_20pct_floor(self):
        """
        When applied in get_weights, risky_pct never goes below 20%.
        Test the max penalty scenario: 85% risky - 25% penalty = 60%.
        Even 60% is above the 20% floor.
        """
        # With penalty capped at 25%, min risky = 20% (floor in get_weights)
        penalty = DISmartfolioReplicaStrategy._compute_governance_penalty(1.0, 2.0)
        assert penalty <= 0.25
        assert 0.85 - penalty >= 0.20, "Even max penalty keeps risky above floor"


class TestReplicaParams:
    """Tests for configurable ReplicaParams in SmartFolio Replica strategy."""

    def test_default_params_match_current_behavior(self):
        """Default ReplicaParams gives same result as no params (backward compat)."""
        cycle, onchain, risk = 85, 70, 75
        result_no_params = DISmartfolioReplicaStrategy._compute_risk_budget(cycle, onchain, risk)
        result_default = DISmartfolioReplicaStrategy._compute_risk_budget(
            cycle, onchain, risk, params=ReplicaParams()
        )
        assert result_no_params == result_default, (
            f"Default params should match no params: {result_no_params} vs {result_default}"
        )

    def test_custom_bounds_respected(self):
        """Custom min/max bounds are respected in risk_budget output."""
        params = ReplicaParams(risk_budget_min=0.30, risk_budget_max=0.70)
        # High scores that would normally give 85%
        result = DISmartfolioReplicaStrategy._compute_risk_budget(95, 90, 95, params=params)
        assert result <= 0.70, f"Max bound 70% violated: {result:.1%}"
        # Low scores that would normally give 20%
        result_low = DISmartfolioReplicaStrategy._compute_risk_budget(10, 10, 10, params=params)
        assert result_low >= 0.30, f"Min bound 30% violated: {result_low:.1%}"

    def test_market_overrides_disabled(self):
        """Disabling market overrides skips divergence/low-risk checks."""
        params_on = ReplicaParams(enable_market_overrides=True)
        params_off = ReplicaParams(enable_market_overrides=False)
        # Scores where divergence triggers: blended=72.5, onchain=30, |72.5-30|=42.5>=30
        cycle, onchain, risk = 95, 30, 80
        result_on = DISmartfolioReplicaStrategy._compute_risk_budget(
            cycle, onchain, risk, params=params_on
        )
        result_off = DISmartfolioReplicaStrategy._compute_risk_budget(
            cycle, onchain, risk, params=params_off
        )
        assert result_off >= result_on, (
            f"Overrides off should be >= overrides on: {result_off:.1%} vs {result_on:.1%}"
        )

    def test_high_confidence_raises_exposure_cap(self):
        """Higher exposure_confidence gives higher/equal cap."""
        params_low = ReplicaParams(exposure_confidence=0.50)
        params_high = ReplicaParams(exposure_confidence=1.00)
        cap_low = DISmartfolioReplicaStrategy._compute_exposure_cap(
            70, 80, 60.0, 0.15, params=params_low
        )
        cap_high = DISmartfolioReplicaStrategy._compute_exposure_cap(
            70, 80, 60.0, 0.15, params=params_high
        )
        assert cap_high >= cap_low, (
            f"High confidence should give higher cap: {cap_high:.1%} vs {cap_low:.1%}"
        )

    def test_custom_max_governance_penalty(self):
        """Custom max_governance_penalty caps the penalty."""
        params = ReplicaParams(max_governance_penalty=0.10)
        # High contradiction + high vol → should be capped at 10%
        penalty = DISmartfolioReplicaStrategy._compute_governance_penalty(
            0.80, 1.5, params=params
        )
        assert penalty <= 0.10, f"Penalty {penalty:.1%} exceeds custom max 10%"

    def test_zero_governance_penalty_cap(self):
        """Setting max_governance_penalty=0 disables penalty entirely."""
        params = ReplicaParams(max_governance_penalty=0.0)
        penalty = DISmartfolioReplicaStrategy._compute_governance_penalty(
            0.80, 1.5, params=params
        )
        assert penalty == 0.0, f"Zero max should give zero penalty, got {penalty:.3f}"

    def test_strategy_accepts_replica_params(self):
        """DISmartfolioReplicaStrategy.__init__ accepts replica_params."""
        params = ReplicaParams(risk_budget_min=0.25, exposure_confidence=0.80)
        strategy = DISmartfolioReplicaStrategy(replica_params=params)
        assert strategy.replica_params.risk_budget_min == 0.25
        assert strategy.replica_params.exposure_confidence == 0.80

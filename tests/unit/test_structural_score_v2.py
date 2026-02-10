"""Tests for services/risk/structural_score_v2.py — pure math, no deps."""

import pytest
from services.risk.structural_score_v2 import (
    compute_structural_score_v2,
    get_structural_level,
)


# ===========================================================================
# get_structural_level
# ===========================================================================

class TestGetStructuralLevel:
    def test_very_robust(self):
        assert get_structural_level(85) == "very_robust"
        assert get_structural_level(100) == "very_robust"

    def test_robust(self):
        assert get_structural_level(70) == "robust"
        assert get_structural_level(84) == "robust"

    def test_moderate(self):
        assert get_structural_level(50) == "moderate"
        assert get_structural_level(69) == "moderate"

    def test_fragile(self):
        assert get_structural_level(30) == "fragile"
        assert get_structural_level(49) == "fragile"

    def test_very_fragile(self):
        assert get_structural_level(0) == "very_fragile"
        assert get_structural_level(29) == "very_fragile"


# ===========================================================================
# compute_structural_score_v2 — individual penalties
# ===========================================================================

class TestHhiPenalty:
    def test_no_penalty_below_threshold(self):
        score, bd = compute_structural_score_v2(hhi=0.20, memes_pct=0.0, gri=0.0, effective_assets=10)
        assert bd["hhi"] == 0.0

    def test_at_threshold(self):
        score, bd = compute_structural_score_v2(hhi=0.25, memes_pct=0.0, gri=0.0, effective_assets=10)
        assert bd["hhi"] == 0.0

    def test_above_threshold(self):
        score, bd = compute_structural_score_v2(hhi=0.35, memes_pct=0.0, gri=0.0, effective_assets=10)
        assert bd["hhi"] == pytest.approx(10.0)  # (0.35-0.25)*100

    def test_high_concentration(self):
        score, bd = compute_structural_score_v2(hhi=0.50, memes_pct=0.0, gri=0.0, effective_assets=10)
        assert bd["hhi"] == pytest.approx(25.0)


class TestMemecoinsPenalty:
    def test_zero_memes(self):
        score, bd = compute_structural_score_v2(hhi=0.10, memes_pct=0.0, gri=0.0, effective_assets=10)
        assert bd["memecoins"] == 0.0

    def test_half_memes(self):
        score, bd = compute_structural_score_v2(hhi=0.10, memes_pct=0.50, gri=0.0, effective_assets=10)
        assert bd["memecoins"] == pytest.approx(20.0)  # 0.50 * 40

    def test_full_memes(self):
        score, bd = compute_structural_score_v2(hhi=0.10, memes_pct=1.0, gri=0.0, effective_assets=10)
        assert bd["memecoins"] == pytest.approx(40.0)


class TestGriPenalty:
    def test_zero_gri(self):
        score, bd = compute_structural_score_v2(hhi=0.10, memes_pct=0.0, gri=0.0, effective_assets=10)
        assert bd["gri"] == 0.0

    def test_moderate_gri(self):
        score, bd = compute_structural_score_v2(hhi=0.10, memes_pct=0.0, gri=5.0, effective_assets=10)
        assert bd["gri"] == pytest.approx(25.0)  # 5 * 5

    def test_max_gri(self):
        score, bd = compute_structural_score_v2(hhi=0.10, memes_pct=0.0, gri=10.0, effective_assets=10)
        assert bd["gri"] == pytest.approx(50.0)


class TestDiversificationPenalty:
    def test_good_diversification(self):
        score, bd = compute_structural_score_v2(hhi=0.10, memes_pct=0.0, gri=0.0, effective_assets=10)
        assert bd["low_diversification"] == 0.0

    def test_exactly_five(self):
        score, bd = compute_structural_score_v2(hhi=0.10, memes_pct=0.0, gri=0.0, effective_assets=5)
        assert bd["low_diversification"] == 0.0

    def test_below_five(self):
        score, bd = compute_structural_score_v2(hhi=0.10, memes_pct=0.0, gri=0.0, effective_assets=3)
        assert bd["low_diversification"] == 10.0


# ===========================================================================
# compute_structural_score_v2 — integration
# ===========================================================================

class TestComputeStructuralScoreV2Integration:
    def test_perfect_portfolio(self):
        """No penalties → score = 100."""
        score, bd = compute_structural_score_v2(hhi=0.10, memes_pct=0.0, gri=0.0, effective_assets=10)
        assert score == 100.0

    def test_score_clamped_at_zero(self):
        """All max penalties → score = 0."""
        score, bd = compute_structural_score_v2(hhi=1.0, memes_pct=1.0, gri=10.0, effective_assets=1)
        assert score == 0.0

    def test_degen_portfolio(self):
        """55% memes, moderate concentration."""
        score, bd = compute_structural_score_v2(hhi=0.32, memes_pct=0.55, gri=7.4, effective_assets=7)
        assert 15 <= score <= 40
        assert get_structural_level(score) in ("fragile", "very_fragile")

    def test_balanced_portfolio(self):
        """0% memes, good diversification."""
        score, bd = compute_structural_score_v2(hhi=0.18, memes_pct=0.0, gri=3.2, effective_assets=7)
        assert 70 <= score <= 90
        assert get_structural_level(score) in ("robust", "very_robust")

    def test_conservative_portfolio(self):
        score, bd = compute_structural_score_v2(hhi=0.38, memes_pct=0.0, gri=2.5, effective_assets=3)
        assert 45 <= score <= 70

    def test_returns_tuple(self):
        result = compute_structural_score_v2(hhi=0.10, memes_pct=0.0, gri=0.0, effective_assets=10)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_breakdown_contains_all_keys(self):
        score, bd = compute_structural_score_v2(hhi=0.20, memes_pct=0.1, gri=3.0, effective_assets=8)
        assert "hhi" in bd
        assert "memecoins" in bd
        assert "gri" in bd
        assert "low_diversification" in bd
        assert "base" in bd
        assert "total_penalties" in bd
        assert "final_score" in bd
        assert "inputs" in bd

    def test_breakdown_inputs(self):
        score, bd = compute_structural_score_v2(hhi=0.20, memes_pct=0.1, gri=3.0, effective_assets=8, top5_pct=0.6)
        inputs = bd["inputs"]
        assert inputs["hhi"] == 0.20
        assert inputs["memes_pct"] == 0.1
        assert inputs["gri"] == 3.0
        assert inputs["effective_assets"] == 8
        assert inputs["top5_pct"] == 0.6

    def test_total_penalties_matches_sum(self):
        score, bd = compute_structural_score_v2(hhi=0.30, memes_pct=0.20, gri=4.0, effective_assets=6)
        expected_total = bd["hhi"] + bd["memecoins"] + bd["gri"] + bd["low_diversification"]
        assert bd["total_penalties"] == pytest.approx(expected_total)

    def test_final_score_matches_return(self):
        score, bd = compute_structural_score_v2(hhi=0.30, memes_pct=0.20, gri=4.0, effective_assets=6)
        assert score == bd["final_score"]

import math
from unittest.mock import AsyncMock

import pytest

from services.ml.bourse.opportunity_scanner import OpportunityScanner
from services.ml.bourse.scoring_engine import ScoringEngine


def test_scoring_engine_withholds_missing_required_signals():
    result = ScoringEngine().calculate_score(
        technical_score=math.nan,
        regime_score=0.8,
        relative_strength_score=math.inf,
        risk_score=0.4,
        sector_score=0.6,
    )

    assert result["final_score"] is None
    assert result["confidence"] == 0.0
    assert set(result["missing_signals"]) == {"technical", "relative_strength"}


@pytest.mark.asyncio
async def test_opportunity_scanner_replaces_non_finite_scores():
    scanner = OpportunityScanner()
    scanner.sector_analyzer.analyze_sector = AsyncMock(return_value={
        "momentum_score": math.nan,
        "value_score": math.inf,
        "diversification_score": 60.0,
        "confidence": math.nan,
    })

    result = await scanner._score_gap({"sector": "Technology", "etf": "XLK"}, "medium")

    assert result["score"] is None
    assert result["confidence"] == 0.0
    assert result["momentum_score"] is None


@pytest.mark.asyncio
async def test_missing_fundamentals_do_not_pull_verified_momentum_to_neutral():
    scanner = OpportunityScanner()
    scanner.sector_analyzer.analyze_sector = AsyncMock(return_value={
        "momentum_score": 80.0,
        "value_score": None,
        "diversification_score": None,
        "confidence": 0.85,
    })

    result = await scanner._score_gap({"sector": "Technology", "etf": "XLK"}, "short")

    assert result["score"] == 80.0
    assert result["score_components_available"] == ["momentum"]


def test_missing_sector_reduces_coverage_and_agreement():
    result = ScoringEngine().calculate_score(.8, .8, .8, .8, None)
    assert result['final_score'] == .8
    assert result['data_coverage'] == .8
    assert result['confidence'] == .8
    assert result['breakdown']['sector'] is None


@pytest.fixture(autouse=True)
def disable_external_score_cache(monkeypatch):
    """Unit tests must not wait for, read or write an external Redis server."""
    monkeypatch.setattr('services.ml.bourse.sector_analyzer.REDIS_AVAILABLE', False)

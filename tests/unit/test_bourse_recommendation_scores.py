import math
from unittest.mock import AsyncMock

import pytest

from services.ml.bourse.opportunity_scanner import OpportunityScanner
from services.ml.bourse.scoring_engine import ScoringEngine


def test_scoring_engine_keeps_non_finite_inputs_neutral():
    result = ScoringEngine().calculate_score(
        technical_score=math.nan,
        regime_score=0.8,
        relative_strength_score=math.inf,
        risk_score=0.4,
        sector_score=0.6,
    )

    assert result["final_score"] == 0.58
    assert result["confidence"] == 0.925
    assert all(math.isfinite(value) for value in result["breakdown"].values())


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

    assert result["score"] == 53.0
    assert result["confidence"] == 0.7
    assert all(math.isfinite(result[key]) for key in ("score", "momentum_score", "value_score"))

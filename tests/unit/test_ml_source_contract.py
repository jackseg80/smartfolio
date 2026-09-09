"""The legacy ML orchestrator must not infer a portfolio data source."""

from types import SimpleNamespace

import pytest

from services.ml.orchestrator import MLOrchestrator
from services.ml.data_pipeline import MLDataPipeline


@pytest.mark.asyncio
async def test_missing_source_remains_unavailable():
    orchestrator = object.__new__(MLOrchestrator)
    orchestrator.settings = SimpleNamespace()

    assert await orchestrator.get_data_source_config() is None


@pytest.mark.asyncio
async def test_explicitly_injected_source_is_preserved():
    orchestrator = object.__new__(MLOrchestrator)
    orchestrator.settings = SimpleNamespace(data_source="cointracking")

    assert await orchestrator.get_data_source_config() == "cointracking"


def test_legacy_pipeline_does_not_guess_portfolio_assets(tmp_path):
    pipeline = MLDataPipeline(cache_dir=str(tmp_path / "ml-cache"))

    assert pipeline.fetch_portfolio_assets(source="cointracking") == []
    assert pipeline.fetch_portfolio_assets(source="stub") == []

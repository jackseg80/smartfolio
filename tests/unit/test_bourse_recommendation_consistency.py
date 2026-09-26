from unittest.mock import AsyncMock
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from services.ml.bourse.portfolio_adjuster import PortfolioAdjuster
from services.ml.bourse.price_targets import PriceTargets
from services.ml.bourse.recommendations_orchestrator import RecommendationsOrchestrator


def test_equal_lots_are_preserved():
    lots = [dict(symbol='AAA', current_value=100, weight_pct=10,
                 score=.8, confidence=.8, action='HOLD') for _ in range(2)]
    result = PortfolioAdjuster()._consolidate_duplicate_positions(lots)[0]
    assert result['current_value'] == 200
    assert result['weight_pct'] == 20
    assert result['positions_count'] == 2
    assert result['position_sizing']['action'] == 'REVIEW'


def test_sector_room_limits_increment_not_total_position():
    result = PriceTargets().calculate_position_size(
        'STRONG BUY', .9, 10000, .03, .39, available_cash_usd=1000)
    assert result['increment_dollars'] == 100
    assert result['target_allocation_pct'] == 4


def test_cash_cap_updates_percentage_and_amount():
    result = PriceTargets().calculate_position_size(
        'STRONG BUY', .9, 10000, .01, .10, available_cash_usd=50)
    assert result['increment_dollars'] == 50
    assert result['increment_pct'] == .5
    assert result['target_allocation_pct'] == 1.5


def test_return_window_contains_n_intervals():
    orchestrator = object.__new__(RecommendationsOrchestrator)
    assert orchestrator._calculate_return(pd.Series([100, 110, 121]), 2) == pytest.approx(21)
    with pytest.raises(ValueError):
        orchestrator._calculate_return(pd.Series([100, 110]), 2)


@pytest.mark.asyncio
async def test_final_action_refreshes_sizing_and_reports_missing_positions():
    orchestrator = object.__new__(RecommendationsOrchestrator)
    prices = pd.DataFrame({'close': range(100, 200)}, index=pd.date_range('2026-01-01', periods=100))
    prices.attrs['native_currency'] = 'USD'
    orchestrator._get_benchmark_data = AsyncMock(return_value=prices)
    rec = dict(symbol='AAA', current_value=100, weight_pct=10, score=.8,
               confidence=.8, sector='Tech', action='BUY', technical={'rsi_14d': 50},
               price_targets={'risk_reward_tp1': 2}, position_sizing={'action': 'ADD'})
    orchestrator._analyze_position = AsyncMock(side_effect=[rec, None])
    result = await orchestrator.generate_recommendations(
        positions=[{'symbol': 'AAA', 'market_value': 100}, {'symbol': 'BBB', 'market_value': 100}],
        market_regime='Bull Market', cash_amount=100,
        sector_analysis={'sectors': {'Tech': {'weight': 42}}})
    final = result['recommendations'][0]
    assert final['action'] == 'HOLD'
    assert final['position_sizing']['action'] == 'HOLD'
    assert final['price_targets'] is None
    assert result['unavailable_symbols'] == ['BBB']
    assert result['selected_position_count'] == 2
    assert result['data_complete'] is False


@pytest.mark.asyncio
async def test_full_recommendation_path_without_optional_sector_data():
    dates = pd.bdate_range('2026-01-01', periods=150)
    close = 100 + np.arange(150) * .1 + np.sin(np.arange(150))
    prices = pd.DataFrame(dict(close=close, open=close, high=close+1, low=close-1,
                               volume=np.full(150, 100000)), index=dates)
    prices.attrs['native_currency'] = 'USD'
    orchestrator = object.__new__(RecommendationsOrchestrator)
    orchestrator.data_source = SimpleNamespace(get_ohlcv_data=AsyncMock(return_value=prices))
    result = await orchestrator.generate_recommendations(
        [{'symbol': 'AAPL', 'market_value': 100}], 'Bull Market',
        sector_analysis={}, cash_amount=25)
    assert result['data_complete'] is True
    rec = result['recommendations'][0]
    assert rec['missing_signals'] == ['sector']
    assert rec['confidence'] <= .8
    assert rec['technical']['vs_benchmark_pct'] == 0
    assert rec['weight_pct'] == 80

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from services.execution.phase_engine import PhaseEngine


def _history(days: int, start_price: float, end_price: float):
    now = int(time.time())
    return [(now - days * 86400, start_price), (now, end_price)]


def test_pair_relative_strength_uses_cached_returns():
    result = PhaseEngine._calculate_pair_relative_strength(
        _history(8, 100.0, 120.0),
        _history(8, 100.0, 110.0),
        7,
    )
    assert result == pytest.approx(1.090909, rel=1e-5)


def test_pair_relative_strength_rejects_missing_history():
    assert PhaseEngine._calculate_pair_relative_strength(None, _history(8, 100.0, 110.0), 7) is None


def test_fetch_phase_signals_uses_data_services_without_internal_http():
    engine = PhaseEngine(api_base_url="http://127.0.0.1:8080")
    now = int(time.time())
    btc = [(now - 31 * 86400, 100.0), (now - 8 * 86400, 110.0), (now, 120.0)]
    eth = [(now - 31 * 86400, 100.0), (now - 8 * 86400, 120.0), (now, 130.0)]
    breadth = {
        "advance_decline_ratio": 0.61,
        "new_highs_count": 4,
        "volume_concentration": 0.72,
        "momentum_dispersion": 0.33,
        "meta": {"source": "coingecko_global_top100", "assets_analyzed": 100},
    }

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"data": {"market_cap_percentage": {"btc": 54.2}}}

    class Client:
        requests = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, **kwargs):
            self.requests.append((url, kwargs))
            return Response()

    with (
        patch("services.execution.phase_engine.httpx.AsyncClient", Client),
        patch("services.execution.phase_engine.get_cached_history", side_effect=lambda symbol, days=None: btc if symbol == "BTC" else eth),
        patch("services.execution.phase_engine.get_market_breadth_metrics", new=AsyncMock(return_value=breadth)),
    ):
        signals = asyncio.run(engine._fetch_phase_signals())

    assert len(Client.requests) == 1
    assert Client.requests[0][0] == "https://api.coingecko.com/api/v3/global"
    assert signals.btc_dominance == 54.2
    assert signals.btc_dominance_available is True
    assert signals.relative_strength_available is True
    assert signals.market_breadth_available is True
    assert signals.breadth_advance_decline == 0.61
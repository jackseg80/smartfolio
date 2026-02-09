"""
Unit tests for services/macro_stress.py

Tests graduated penalty, MacroStressResult, MacroStressService cache, evaluate_stress.
"""
import pytest
from unittest.mock import patch, AsyncMock
from datetime import datetime, timedelta
from dataclasses import asdict

from services.macro_stress import (
    _graduated_penalty,
    MacroStressResult,
    MacroStressService,
    VIX_PENALTY_START,
    VIX_PENALTY_MAX_LEVEL,
    VIX_PENALTY_MAX_PTS,
    DXY_CHANGE_START_PCT,
    DXY_CHANGE_MAX_PCT,
    DXY_PENALTY_MAX_PTS,
    TOTAL_PENALTY_CAP,
    MACRO_CACHE_TTL_HOURS,
)


@pytest.fixture
def fresh_service():
    svc = object.__new__(MacroStressService)
    svc._cache = None
    svc._cache_time = None
    return svc


class TestGraduatedPenalty:

    def test_value_below_start_returns_zero(self):
        assert _graduated_penalty(10.0, 20.0, 45.0, 10.0) == 0.0

    def test_value_exactly_at_start_returns_zero(self):
        assert _graduated_penalty(20.0, 20.0, 45.0, 10.0) == 0.0

    def test_value_at_max_level_returns_negative_max(self):
        assert _graduated_penalty(45.0, 20.0, 45.0, 10.0) == pytest.approx(-10.0)

    def test_value_above_max_level_capped(self):
        assert _graduated_penalty(100.0, 20.0, 45.0, 10.0) == pytest.approx(-10.0)

    def test_midpoint_value(self):
        assert _graduated_penalty(32.5, 20.0, 45.0, 10.0) == pytest.approx(-5.0)

    def test_quarter_point(self):
        assert _graduated_penalty(26.25, 20.0, 45.0, 10.0) == pytest.approx(-2.5)

    def test_three_quarter_point(self):
        assert _graduated_penalty(38.75, 20.0, 45.0, 10.0) == pytest.approx(-7.5)

    def test_negative_value_returns_zero(self):
        assert _graduated_penalty(-5.0, 20.0, 45.0, 10.0) == 0.0

    def test_zero_value_returns_zero(self):
        assert _graduated_penalty(0.0, 20.0, 45.0, 10.0) == 0.0

    def test_dxy_params_below_start(self):
        assert _graduated_penalty(1.0, 2.0, 10.0, 8.0) == 0.0

    def test_dxy_params_at_max(self):
        assert _graduated_penalty(10.0, 2.0, 10.0, 8.0) == pytest.approx(-8.0)

    def test_dxy_params_midpoint(self):
        assert _graduated_penalty(6.0, 2.0, 10.0, 8.0) == pytest.approx(-4.0)

    def test_small_range(self):
        assert _graduated_penalty(0.5, 0.0, 1.0, 5.0) == pytest.approx(-2.5)

    def test_very_large_value_capped(self):
        assert _graduated_penalty(1000.0, 20.0, 45.0, 10.0) == pytest.approx(-10.0)

    def test_just_above_start(self):
        r = _graduated_penalty(20.01, 20.0, 45.0, 10.0)
        assert r < 0.0
        assert r > -0.01


class TestMacroStressResult:

    def test_default_values(self):
        result = MacroStressResult()
        assert result.vix_value is None
        assert result.vix_stress is False
        assert result.vix_penalty == 0.0
        assert result.dxy_value is None
        assert result.dxy_change_30d is None
        assert result.dxy_stress is False
        assert result.dxy_penalty == 0.0
        assert result.macro_stress is False
        assert result.decision_penalty == 0.0
        assert result.fetched_at is None
        assert result.error is None

    def test_custom_values(self):
        now = datetime.now()
        result = MacroStressResult(
            vix_value=30.0, vix_stress=True, vix_penalty=-4.0,
            dxy_value=105.0, dxy_change_30d=5.0, dxy_stress=True, dxy_penalty=-3.0,
            macro_stress=True, decision_penalty=-7.0, fetched_at=now,
        )
        assert result.vix_value == 30.0
        assert result.vix_stress is True
        assert result.decision_penalty == -7.0

    def test_error_result(self):
        result = MacroStressResult(error="FRED API key not configured")
        assert result.error == "FRED API key not configured"
        assert result.decision_penalty == 0.0

    def test_asdict(self):
        result = MacroStressResult(vix_value=25.0, vix_penalty=-2.0)
        d = asdict(result)
        assert isinstance(d, dict)
        assert d["vix_value"] == 25.0
        assert d["vix_penalty"] == -2.0
        assert "decision_penalty" in d


class TestConstants:

    def test_vix_penalty_start(self):
        assert VIX_PENALTY_START == 20.0

    def test_vix_penalty_max_level(self):
        assert VIX_PENALTY_MAX_LEVEL == 45.0

    def test_vix_penalty_max_pts(self):
        assert VIX_PENALTY_MAX_PTS == 10.0

    def test_dxy_change_start_pct(self):
        assert DXY_CHANGE_START_PCT == 2.0

    def test_dxy_change_max_pct(self):
        assert DXY_CHANGE_MAX_PCT == 10.0

    def test_dxy_penalty_max_pts(self):
        assert DXY_PENALTY_MAX_PTS == 8.0

    def test_total_penalty_cap(self):
        assert TOTAL_PENALTY_CAP == -15.0

    def test_cache_ttl_hours(self):
        assert MACRO_CACHE_TTL_HOURS == 4


class TestServiceCacheLogic:

    def test_cache_invalid_when_empty(self, fresh_service):
        assert fresh_service._is_cache_valid() is False

    def test_cache_invalid_when_no_time(self, fresh_service):
        fresh_service._cache = MacroStressResult()
        fresh_service._cache_time = None
        assert fresh_service._is_cache_valid() is False

    def test_cache_invalid_when_no_result(self, fresh_service):
        fresh_service._cache = None
        fresh_service._cache_time = datetime.now()
        assert fresh_service._is_cache_valid() is False

    def test_cache_valid_when_recent(self, fresh_service):
        fresh_service._cache = MacroStressResult()
        fresh_service._cache_time = datetime.now()
        assert fresh_service._is_cache_valid() is True

    def test_cache_invalid_when_expired(self, fresh_service):
        fresh_service._cache = MacroStressResult()
        fresh_service._cache_time = datetime.now() - timedelta(hours=5)
        assert fresh_service._is_cache_valid() is False

    def test_cache_valid_at_boundary(self, fresh_service):
        fresh_service._cache = MacroStressResult()
        fresh_service._cache_time = datetime.now() - timedelta(hours=3, minutes=59)
        assert fresh_service._is_cache_valid() is True

    def test_get_cached_penalty_no_cache(self, fresh_service):
        assert fresh_service.get_cached_penalty() == 0.0

    def test_get_cached_penalty_valid(self, fresh_service):
        fresh_service._cache = MacroStressResult(decision_penalty=-7.5)
        fresh_service._cache_time = datetime.now()
        assert fresh_service.get_cached_penalty() == -7.5

    def test_get_cached_penalty_expired(self, fresh_service):
        fresh_service._cache = MacroStressResult(decision_penalty=-7.5)
        fresh_service._cache_time = datetime.now() - timedelta(hours=5)
        assert fresh_service.get_cached_penalty() == 0.0

    def test_invalidate_cache(self, fresh_service):
        fresh_service._cache = MacroStressResult()
        fresh_service._cache_time = datetime.now()
        fresh_service.invalidate_cache()
        assert fresh_service._cache is None
        assert fresh_service._cache_time is None


def _make_vix_data(latest_value):
    return [{"date": "2026-01-15", "value": latest_value}]


def _make_dxy_data(past_value, current_value, n_points=30):
    data = []
    for i in range(n_points):
        val = past_value + (current_value - past_value) * (i / (n_points - 1))
        data.append({"date": f"2026-01-{i+1:02d}", "value": val})
    return data


class TestEvaluateStress:

    @pytest.mark.asyncio
    async def test_no_fred_key_returns_error(self, fresh_service):
        with patch("services.user_secrets.get_user_secrets", return_value={}):
            with patch.dict("os.environ", {}, clear=True):
                result = await fresh_service.evaluate_stress("test_user")
                assert result.error == "FRED API key not configured"

    @pytest.mark.asyncio
    async def test_returns_cached_result(self, fresh_service):
        cached = MacroStressResult(decision_penalty=-5.0, vix_value=30.0)
        fresh_service._cache = cached
        fresh_service._cache_time = datetime.now()
        result = await fresh_service.evaluate_stress("test_user")
        assert result is cached
        assert result.decision_penalty == -5.0

    @pytest.mark.asyncio
    async def test_force_refresh_ignores_cache(self, fresh_service):
        cached = MacroStressResult(decision_penalty=-5.0)
        fresh_service._cache = cached
        fresh_service._cache_time = datetime.now()
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                _make_vix_data(15.0), _make_dxy_data(100.0, 100.5),
            ])
            result = await fresh_service.evaluate_stress("test_user", force_refresh=True)
            assert result is not cached
            assert result.decision_penalty == 0.0

    @pytest.mark.asyncio
    async def test_low_vix_no_penalty(self, fresh_service):
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                _make_vix_data(15.0), _make_dxy_data(100.0, 100.5),
            ])
            result = await fresh_service.evaluate_stress("test_user")
            assert result.vix_value == 15.0
            assert result.vix_stress is False
            assert result.vix_penalty == 0.0

    @pytest.mark.asyncio
    async def test_high_vix_penalty(self, fresh_service):
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                _make_vix_data(45.0), _make_dxy_data(100.0, 100.5),
            ])
            result = await fresh_service.evaluate_stress("test_user")
            assert result.vix_value == 45.0
            assert result.vix_stress is True
            assert result.vix_penalty == pytest.approx(-10.0)

    @pytest.mark.asyncio
    async def test_moderate_vix_graduated(self, fresh_service):
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                _make_vix_data(32.5), _make_dxy_data(100.0, 100.5),
            ])
            result = await fresh_service.evaluate_stress("test_user")
            assert result.vix_penalty == pytest.approx(-5.0)

    @pytest.mark.asyncio
    async def test_high_dxy_change(self, fresh_service):
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                _make_vix_data(15.0), _make_dxy_data(100.0, 112.0),
            ])
            result = await fresh_service.evaluate_stress("test_user")
            assert result.dxy_penalty == pytest.approx(-8.0)
            assert result.dxy_stress is True

    @pytest.mark.asyncio
    async def test_combined_penalty_capped(self, fresh_service):
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                _make_vix_data(45.0), _make_dxy_data(100.0, 112.0),
            ])
            result = await fresh_service.evaluate_stress("test_user")
            assert result.decision_penalty == pytest.approx(-15.0)
            assert result.macro_stress is True

    @pytest.mark.asyncio
    async def test_vix_fetch_failure(self, fresh_service):
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                Exception("VIX fail"), _make_dxy_data(100.0, 100.5),
            ])
            result = await fresh_service.evaluate_stress("test_user")
            assert result.vix_value is None
            assert result.vix_penalty == 0.0

    @pytest.mark.asyncio
    async def test_dxy_fetch_failure(self, fresh_service):
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                _make_vix_data(30.0), Exception("DXY fail"),
            ])
            result = await fresh_service.evaluate_stress("test_user")
            assert result.vix_value == 30.0
            assert result.dxy_value is None

    @pytest.mark.asyncio
    async def test_both_fetch_failures(self, fresh_service):
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                Exception("VIX"), Exception("DXY"),
            ])
            result = await fresh_service.evaluate_stress("test_user")
            assert result.decision_penalty == 0.0
            assert result.macro_stress is False

    @pytest.mark.asyncio
    async def test_empty_vix_data(self, fresh_service):
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                [], _make_dxy_data(100.0, 100.5),
            ])
            result = await fresh_service.evaluate_stress("test_user")
            assert result.vix_value is None

    @pytest.mark.asyncio
    async def test_insufficient_dxy_data(self, fresh_service):
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                _make_vix_data(15.0), [{"date": "2026-01-15", "value": 105.0}],
            ])
            result = await fresh_service.evaluate_stress("test_user")
            assert result.dxy_value == 105.0
            assert result.dxy_change_30d is None

    @pytest.mark.asyncio
    async def test_result_cached_after_eval(self, fresh_service):
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                _make_vix_data(15.0), _make_dxy_data(100.0, 101.0),
            ])
            result = await fresh_service.evaluate_stress("test_user")
            assert fresh_service._cache is result
            assert fresh_service._cache_time is not None

    @pytest.mark.asyncio
    async def test_fred_key_from_env(self, fresh_service):
        with patch("services.user_secrets.get_user_secrets", return_value={}):
            with patch.dict("os.environ", {"FRED_API_KEY": "env_key"}):
                fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                    _make_vix_data(15.0), _make_dxy_data(100.0, 101.0),
                ])
                result = await fresh_service.evaluate_stress("test_user")
                assert result.error is None

    @pytest.mark.asyncio
    async def test_circuit_open_skips_vix(self, fresh_service):
        from shared.circuit_breaker import CircuitOpenError
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                CircuitOpenError("fred"), _make_dxy_data(100.0, 101.0),
            ])
            result = await fresh_service.evaluate_stress("test_user")
            assert result.vix_value is None
            assert result.dxy_value is not None

    @pytest.mark.asyncio
    async def test_circuit_open_skips_dxy(self, fresh_service):
        from shared.circuit_breaker import CircuitOpenError
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                _make_vix_data(25.0), CircuitOpenError("fred"),
            ])
            result = await fresh_service.evaluate_stress("test_user")
            assert result.vix_value == 25.0
            assert result.dxy_value is None

    @pytest.mark.asyncio
    async def test_macro_stress_flag_vix_only(self, fresh_service):
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                _make_vix_data(30.0), _make_dxy_data(100.0, 100.5),
            ])
            result = await fresh_service.evaluate_stress("test_user")
            assert result.macro_stress is True

    @pytest.mark.asyncio
    async def test_macro_stress_flag_dxy_only(self, fresh_service):
        with patch("services.user_secrets.get_user_secrets", return_value={"fred": {"api_key": "k"}}):
            fresh_service._fetch_fred_series = AsyncMock(side_effect=[
                _make_vix_data(15.0), _make_dxy_data(100.0, 105.0),
            ])
            result = await fresh_service.evaluate_stress("test_user")
            assert result.macro_stress is True

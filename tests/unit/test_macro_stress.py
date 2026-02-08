"""
Tests unitaires pour le service Macro Stress (pénalité graduée)

Vérifie la calibration alignée sur DI Backtest V2:
- VIX: 0 (≤20) → -10 (≥45), linéaire
- DXY: 0 (change ≤2%) → -8 (change ≥10%), linéaire
- Total cappé à -15
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from services.macro_stress import (
    MacroStressService,
    MacroStressResult,
    _graduated_penalty,
    VIX_PENALTY_START,
    VIX_PENALTY_MAX_LEVEL,
    VIX_PENALTY_MAX_PTS,
    DXY_CHANGE_START_PCT,
    DXY_CHANGE_MAX_PCT,
    DXY_PENALTY_MAX_PTS,
    TOTAL_PENALTY_CAP,
    MACRO_CACHE_TTL_HOURS,
)


class TestGraduatedPenalty:
    """Tests de la fonction de pénalité graduée linéaire"""

    def test_below_start_no_penalty(self):
        assert _graduated_penalty(15.0, 20.0, 45.0, 10.0) == 0.0

    def test_at_start_no_penalty(self):
        assert _graduated_penalty(20.0, 20.0, 45.0, 10.0) == 0.0

    def test_midpoint_linear(self):
        # (32.5 - 20) / (45 - 20) * 10 = 12.5/25 * 10 = 5.0
        result = _graduated_penalty(32.5, 20.0, 45.0, 10.0)
        assert result == pytest.approx(-5.0, abs=0.01)

    def test_at_max_level(self):
        result = _graduated_penalty(45.0, 20.0, 45.0, 10.0)
        assert result == pytest.approx(-10.0, abs=0.01)

    def test_above_max_level_capped(self):
        # Au-delà de max_level, plafonné à -max_pts
        result = _graduated_penalty(60.0, 20.0, 45.0, 10.0)
        assert result == pytest.approx(-10.0, abs=0.01)


class TestVIXPenalty:
    """Tests de la pénalité VIX graduée"""

    def test_vix_15_no_penalty(self):
        """VIX=15 → en-dessous du seuil, pas de pénalité"""
        result = _graduated_penalty(15.0, VIX_PENALTY_START, VIX_PENALTY_MAX_LEVEL, VIX_PENALTY_MAX_PTS)
        assert result == 0.0

    def test_vix_25_partial_penalty(self):
        """VIX=25 → (25-20)/(45-20)*10 = 2.0 pts"""
        result = _graduated_penalty(25.0, VIX_PENALTY_START, VIX_PENALTY_MAX_LEVEL, VIX_PENALTY_MAX_PTS)
        assert result == pytest.approx(-2.0, abs=0.01)

    def test_vix_31_graduated_not_15(self):
        """VIX=31 → (31-20)/(45-20)*10 = 4.4 pts (pas -15 comme avant!)"""
        result = _graduated_penalty(31.0, VIX_PENALTY_START, VIX_PENALTY_MAX_LEVEL, VIX_PENALTY_MAX_PTS)
        assert result == pytest.approx(-4.4, abs=0.01)

    def test_vix_35_six_pts(self):
        """VIX=35 → (35-20)/(45-20)*10 = 6.0 pts (doc backtest: '~-6pts')"""
        result = _graduated_penalty(35.0, VIX_PENALTY_START, VIX_PENALTY_MAX_LEVEL, VIX_PENALTY_MAX_PTS)
        assert result == pytest.approx(-6.0, abs=0.01)

    def test_vix_45_max_penalty(self):
        """VIX=45 → pénalité max VIX = -10 pts"""
        result = _graduated_penalty(45.0, VIX_PENALTY_START, VIX_PENALTY_MAX_LEVEL, VIX_PENALTY_MAX_PTS)
        assert result == pytest.approx(-10.0, abs=0.01)

    def test_vix_60_capped_at_max(self):
        """VIX=60 → toujours -10 pts (cappé)"""
        result = _graduated_penalty(60.0, VIX_PENALTY_START, VIX_PENALTY_MAX_LEVEL, VIX_PENALTY_MAX_PTS)
        assert result == pytest.approx(-10.0, abs=0.01)


class TestDXYPenalty:
    """Tests de la pénalité DXY graduée"""

    def test_dxy_change_1pct_no_penalty(self):
        """DXY change=1% → en-dessous du seuil"""
        result = _graduated_penalty(1.0, DXY_CHANGE_START_PCT, DXY_CHANGE_MAX_PCT, DXY_PENALTY_MAX_PTS)
        assert result == 0.0

    def test_dxy_change_6pct_partial(self):
        """DXY change=6% → (6-2)/(10-2)*8 = 4.0 pts"""
        result = _graduated_penalty(6.0, DXY_CHANGE_START_PCT, DXY_CHANGE_MAX_PCT, DXY_PENALTY_MAX_PTS)
        assert result == pytest.approx(-4.0, abs=0.01)

    def test_dxy_change_10pct_max(self):
        """DXY change=10% → pénalité max DXY = -8 pts"""
        result = _graduated_penalty(10.0, DXY_CHANGE_START_PCT, DXY_CHANGE_MAX_PCT, DXY_PENALTY_MAX_PTS)
        assert result == pytest.approx(-8.0, abs=0.01)

    def test_dxy_change_15pct_capped(self):
        """DXY change=15% → toujours -8 pts (cappé)"""
        result = _graduated_penalty(15.0, DXY_CHANGE_START_PCT, DXY_CHANGE_MAX_PCT, DXY_PENALTY_MAX_PTS)
        assert result == pytest.approx(-8.0, abs=0.01)


class TestCombinedPenalty:
    """Tests de la pénalité combinée VIX + DXY"""

    def test_vix35_dxy6_additive(self):
        """VIX=35 + DXY change=6% → -6.0 + -4.0 = -10.0"""
        vix_pen = _graduated_penalty(35.0, VIX_PENALTY_START, VIX_PENALTY_MAX_LEVEL, VIX_PENALTY_MAX_PTS)
        dxy_pen = _graduated_penalty(6.0, DXY_CHANGE_START_PCT, DXY_CHANGE_MAX_PCT, DXY_PENALTY_MAX_PTS)
        total = max(TOTAL_PENALTY_CAP, vix_pen + dxy_pen)
        assert total == pytest.approx(-10.0, abs=0.01)

    def test_vix50_dxy12_capped_at_minus15(self):
        """VIX=50 + DXY change=12% → -10.0 + -8.0 = -18.0, cappé à -15.0"""
        vix_pen = _graduated_penalty(50.0, VIX_PENALTY_START, VIX_PENALTY_MAX_LEVEL, VIX_PENALTY_MAX_PTS)
        dxy_pen = _graduated_penalty(12.0, DXY_CHANGE_START_PCT, DXY_CHANGE_MAX_PCT, DXY_PENALTY_MAX_PTS)
        total = max(TOTAL_PENALTY_CAP, vix_pen + dxy_pen)
        assert total == pytest.approx(-15.0, abs=0.01)

    def test_no_stress_zero_penalty(self):
        """VIX=15 + DXY change=1% → 0 + 0 = 0"""
        vix_pen = _graduated_penalty(15.0, VIX_PENALTY_START, VIX_PENALTY_MAX_LEVEL, VIX_PENALTY_MAX_PTS)
        dxy_pen = _graduated_penalty(1.0, DXY_CHANGE_START_PCT, DXY_CHANGE_MAX_PCT, DXY_PENALTY_MAX_PTS)
        total = max(TOTAL_PENALTY_CAP, vix_pen + dxy_pen)
        assert total == 0.0


class TestMacroStressResult:
    """Tests du dataclass MacroStressResult"""

    def test_default_values(self):
        result = MacroStressResult()
        assert result.decision_penalty == 0.0
        assert result.vix_penalty == 0.0
        assert result.dxy_penalty == 0.0
        assert result.macro_stress is False
        assert result.vix_stress is False
        assert result.dxy_stress is False

    def test_penalty_is_float(self):
        """Le type decision_penalty est float (pas int) pour les valeurs graduées"""
        result = MacroStressResult(decision_penalty=-4.4)
        assert isinstance(result.decision_penalty, float)


class TestMacroStressServiceEvaluate:
    """Tests d'intégration du service evaluate_stress"""

    @pytest.fixture(autouse=True)
    def fresh_service(self):
        """Reset le singleton entre chaque test"""
        MacroStressService._instance = None
        MacroStressService._cache = None
        MacroStressService._cache_time = None
        self.service = MacroStressService()

    @pytest.mark.asyncio
    async def test_vix_31_graduated_penalty(self):
        """VIX=31 produit une pénalité graduée de -4.4 (pas -15)"""
        vix_data = [{"date": "2026-01-01", "value": 31.0}]

        with patch.object(self.service, '_fetch_fred_series', new_callable=AsyncMock) as mock_fetch, \
             patch('services.user_secrets.get_user_secrets', return_value={"fred": {"api_key": "test"}}):
            mock_fetch.side_effect = [vix_data, []]  # VIX ok, DXY empty

            result = await self.service.evaluate_stress("test_user")

            assert result.vix_stress is True
            assert result.vix_penalty == pytest.approx(-4.4, abs=0.01)
            assert result.decision_penalty == pytest.approx(-4.4, abs=0.01)
            assert result.macro_stress is True

    @pytest.mark.asyncio
    async def test_no_stress_zero_penalty(self):
        """VIX=15 → pas de stress, pénalité = 0"""
        vix_data = [{"date": "2026-01-01", "value": 15.0}]

        with patch.object(self.service, '_fetch_fred_series', new_callable=AsyncMock) as mock_fetch, \
             patch('services.user_secrets.get_user_secrets', return_value={"fred": {"api_key": "test"}}):
            mock_fetch.side_effect = [vix_data, []]

            result = await self.service.evaluate_stress("test_user")

            assert result.vix_stress is False
            assert result.decision_penalty == 0.0
            assert result.macro_stress is False

    @pytest.mark.asyncio
    async def test_combined_vix_dxy(self):
        """VIX=35 + DXY change=6% → pénalité combinée"""
        vix_data = [{"date": "2026-01-01", "value": 35.0}]
        # DXY: 30 observations, le dernier = 106, le -30ème = 100 → change = 6%
        dxy_data = [{"date": f"2025-12-{i:02d}", "value": 100.0} for i in range(1, 31)]
        dxy_data.append({"date": "2026-01-31", "value": 106.0})

        with patch.object(self.service, '_fetch_fred_series', new_callable=AsyncMock) as mock_fetch, \
             patch('services.user_secrets.get_user_secrets', return_value={"fred": {"api_key": "test"}}):
            mock_fetch.side_effect = [vix_data, dxy_data]

            result = await self.service.evaluate_stress("test_user")

            # VIX=35 → -6.0, DXY change=6% → -4.0, total = -10.0
            assert result.vix_penalty == pytest.approx(-6.0, abs=0.01)
            assert result.dxy_penalty == pytest.approx(-4.0, abs=0.01)
            assert result.decision_penalty == pytest.approx(-10.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_cache_returns_cached_penalty(self):
        """Le cache retourne la pénalité graduée correcte"""
        vix_data = [{"date": "2026-01-01", "value": 25.0}]

        with patch.object(self.service, '_fetch_fred_series', new_callable=AsyncMock) as mock_fetch, \
             patch('services.user_secrets.get_user_secrets', return_value={"fred": {"api_key": "test"}}):
            mock_fetch.side_effect = [vix_data, []]

            await self.service.evaluate_stress("test_user")

            # get_cached_penalty retourne la même valeur
            cached = self.service.get_cached_penalty()
            assert cached == pytest.approx(-2.0, abs=0.01)
            assert isinstance(cached, float)

    @pytest.mark.asyncio
    async def test_no_fred_key_returns_empty(self):
        """Sans clé FRED, retourne un résultat vide avec erreur"""
        with patch('services.user_secrets.get_user_secrets', return_value={}):
            result = await self.service.evaluate_stress("test_user")
            assert result.error == "FRED API key not configured"
            assert result.decision_penalty == 0.0

    def test_cache_ttl_respected(self):
        """Le cache expire après MACRO_CACHE_TTL_HOURS"""
        self.service._cache = MacroStressResult(decision_penalty=-4.0)
        self.service._cache_time = datetime.now() - timedelta(hours=MACRO_CACHE_TTL_HOURS + 1)

        assert not self.service._is_cache_valid()
        assert self.service.get_cached_penalty() == 0.0

    def test_cache_valid_within_ttl(self):
        """Le cache est valide dans le TTL"""
        self.service._cache = MacroStressResult(decision_penalty=-4.0)
        self.service._cache_time = datetime.now() - timedelta(hours=1)

        assert self.service._is_cache_valid()
        assert self.service.get_cached_penalty() == pytest.approx(-4.0)


class TestConstants:
    """Vérifie que les constantes sont alignées avec le backtest V2"""

    def test_vix_constants(self):
        assert VIX_PENALTY_START == 20.0
        assert VIX_PENALTY_MAX_LEVEL == 45.0
        assert VIX_PENALTY_MAX_PTS == 10.0

    def test_dxy_constants(self):
        assert DXY_CHANGE_START_PCT == 2.0
        assert DXY_CHANGE_MAX_PCT == 10.0
        assert DXY_PENALTY_MAX_PTS == 8.0

    def test_total_cap(self):
        assert TOTAL_PENALTY_CAP == -15.0

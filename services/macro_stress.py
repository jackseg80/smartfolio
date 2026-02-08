"""
Macro Stress Service - Intégration DXY/VIX pour pénalité Decision Index

Pénalité graduée linéaire (calibrée sur DI Backtest V2, Feb 2026):
- VIX: 0 pts (≤20) → -10 pts (≥45), linéaire
- DXY: 0 pts (change 30j ≤2%) → -8 pts (change ≥10%), linéaire
- Total = VIX + DXY, cappé à -15 pts
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import httpx

from shared.circuit_breaker import fred_circuit, CircuitOpenError

log = logging.getLogger(__name__)

# Cache TTL: 4 heures (comme les données on-chain)
MACRO_CACHE_TTL_HOURS = 4

# Seuils gradués (alignés sur backtest V2 — data_sources.py:compute_macro_penalty_v2)
VIX_PENALTY_START = 20.0       # VIX en-dessous → pas de pénalité
VIX_PENALTY_MAX_LEVEL = 45.0   # VIX au-dessus → pénalité max VIX
VIX_PENALTY_MAX_PTS = 10.0     # -10 pts max pour VIX seul

DXY_CHANGE_START_PCT = 2.0     # Variation 30j en-dessous → pas de pénalité
DXY_CHANGE_MAX_PCT = 10.0      # Variation 30j au-dessus → pénalité max DXY
DXY_PENALTY_MAX_PTS = 8.0      # -8 pts max pour DXY seul

TOTAL_PENALTY_CAP = -15.0      # Cap total (VIX + DXY ne dépasse jamais -15)


def _graduated_penalty(value: float, start: float, max_level: float, max_pts: float) -> float:
    """Calcule une pénalité linéaire graduée entre start et max_level."""
    if value <= start:
        return 0.0
    ratio = (value - start) / (max_level - start)
    return -min(max_pts, ratio * max_pts)


@dataclass
class MacroStressResult:
    """Résultat de l'évaluation du stress macro"""
    vix_value: Optional[float] = None
    vix_stress: bool = False
    vix_penalty: float = 0.0
    dxy_value: Optional[float] = None
    dxy_change_30d: Optional[float] = None
    dxy_stress: bool = False
    dxy_penalty: float = 0.0
    macro_stress: bool = False
    decision_penalty: float = 0.0
    fetched_at: Optional[datetime] = None
    error: Optional[str] = None


class MacroStressService:
    """Service singleton pour évaluer le stress macro (DXY/VIX)"""

    _instance: Optional["MacroStressService"] = None
    _cache: Optional[MacroStressResult] = None
    _cache_time: Optional[datetime] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _is_cache_valid(self) -> bool:
        """Vérifie si le cache est encore valide"""
        if self._cache is None or self._cache_time is None:
            return False
        return datetime.now() - self._cache_time < timedelta(hours=MACRO_CACHE_TTL_HOURS)

    async def _fetch_fred_series(
        self, series_id: str, fred_api_key: str, start_date: str = "2020-01-01"
    ) -> list[dict]:
        """Récupère une série FRED (avec circuit breaker)"""
        fred_circuit.raise_if_open()

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": fred_api_key,
            "file_type": "json",
            "observation_start": start_date,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
        except (httpx.TimeoutException, httpx.ConnectError, OSError) as e:
            fred_circuit.record_failure()
            raise Exception(f"FRED API network error: {e}") from e

        if response.status_code != 200:
            fred_circuit.record_failure()
            raise Exception(f"FRED API error: HTTP {response.status_code}")

        fred_circuit.record_success()

        data = response.json()
        observations = data.get("observations", [])

        result = []
        for obs in observations:
            if obs["value"] != "." and obs["value"] is not None:
                try:
                    value = float(obs["value"])
                    result.append({"date": obs["date"], "value": value})
                except (ValueError, TypeError):
                    continue
        return result

    async def evaluate_stress(
        self, user_id: str, force_refresh: bool = False
    ) -> MacroStressResult:
        """
        Évalue le stress macro actuel (DXY et VIX)

        Pénalité graduée linéaire (calibrée backtest V2):
        - VIX: 0 (≤20) → -10 (≥45)
        - DXY 30d change: 0 (≤2%) → -8 (≥10%)
        - Total cappé à -15

        Args:
            user_id: ID utilisateur pour récupérer la clé FRED
            force_refresh: Force le rafraîchissement du cache

        Returns:
            MacroStressResult avec les indicateurs de stress et la pénalité graduée
        """
        # Vérifier le cache
        if not force_refresh and self._is_cache_valid():
            log.debug("Macro stress: using cached result")
            return self._cache

        # Récupérer la clé FRED
        from services.user_secrets import get_user_secrets
        secrets = get_user_secrets(user_id)
        fred_api_key = secrets.get("fred", {}).get("api_key") or os.getenv("FRED_API_KEY")

        if not fred_api_key:
            log.warning("FRED API key not configured - skipping macro stress evaluation")
            return MacroStressResult(error="FRED API key not configured")

        result = MacroStressResult(fetched_at=datetime.now())

        try:
            # Récupérer VIX (VIXCLS)
            vix_data = await self._fetch_fred_series("VIXCLS", fred_api_key)
            if vix_data:
                result.vix_value = vix_data[-1]["value"]
                result.vix_stress = result.vix_value > VIX_PENALTY_START
                result.vix_penalty = _graduated_penalty(
                    result.vix_value, VIX_PENALTY_START, VIX_PENALTY_MAX_LEVEL, VIX_PENALTY_MAX_PTS
                )
                log.info(f"VIX: {result.vix_value:.2f} (stress: {result.vix_stress}, penalty: {result.vix_penalty:.1f})")
        except CircuitOpenError:
            log.warning("FRED circuit OPEN — skipping VIX fetch")
        except Exception as e:
            log.warning(f"Failed to fetch VIX: {e}")

        try:
            # Récupérer DXY (DTWEXBGS)
            dxy_data = await self._fetch_fred_series("DTWEXBGS", fred_api_key)
            if dxy_data:
                result.dxy_value = dxy_data[-1]["value"]
                # Calculer variation 30 jours
                if len(dxy_data) >= 30:
                    past_value = dxy_data[-30]["value"]
                    if past_value > 0:
                        result.dxy_change_30d = ((result.dxy_value - past_value) / past_value) * 100
                        result.dxy_stress = result.dxy_change_30d > DXY_CHANGE_START_PCT
                        result.dxy_penalty = _graduated_penalty(
                            result.dxy_change_30d, DXY_CHANGE_START_PCT, DXY_CHANGE_MAX_PCT, DXY_PENALTY_MAX_PTS
                        )
                        log.info(
                            f"DXY: {result.dxy_value:.2f} (30d change: {result.dxy_change_30d:.2f}%, "
                            f"stress: {result.dxy_stress}, penalty: {result.dxy_penalty:.1f})"
                        )
        except CircuitOpenError:
            log.warning("FRED circuit OPEN — skipping DXY fetch")
        except Exception as e:
            log.warning(f"Failed to fetch DXY: {e}")

        # Pénalité graduée additive, cappée à TOTAL_PENALTY_CAP
        result.decision_penalty = max(TOTAL_PENALTY_CAP, result.vix_penalty + result.dxy_penalty)
        result.macro_stress = result.vix_stress or result.dxy_stress

        if result.macro_stress:
            log.warning(
                f"MACRO STRESS - penalty: {result.decision_penalty:.1f} pts "
                f"(VIX: {result.vix_penalty:.1f}, DXY: {result.dxy_penalty:.1f})"
            )

        # Mettre en cache
        self._cache = result
        self._cache_time = datetime.now()

        return result

    def get_cached_penalty(self) -> float:
        """Retourne la pénalité en cache (0.0 si pas de cache valide)"""
        if self._is_cache_valid() and self._cache:
            return self._cache.decision_penalty
        return 0.0

    def invalidate_cache(self):
        """Invalide le cache manuellement"""
        self._cache = None
        self._cache_time = None
        log.info("Macro stress cache invalidated")


# Singleton instance
macro_stress_service = MacroStressService()

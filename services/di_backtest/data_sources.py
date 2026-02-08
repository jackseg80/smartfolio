"""
Historical Data Sources for DI Backtest
Agrégation des données historiques pour reconstruction du Decision Index

Sources:
- Prix crypto: services/price_history.py (Binance, 2017+)
- Macro (VIX/DXY): FRED API (20+ ans)
- Fear & Greed: alternative.me API (365 jours max, proxy pour avant)
- Cycle Score: calcul déterministe basé sur halvings
"""

import asyncio
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

from services.price_history import (
    price_history,
    get_cached_history,
    download_historical_data,
)

logger = logging.getLogger(__name__)

# Bitcoin halvings historiques
BITCOIN_HALVINGS = [
    datetime(2012, 11, 28),
    datetime(2016, 7, 9),
    datetime(2020, 5, 11),
    datetime(2024, 4, 20),
]

# Paramètres Cycle Score calibrés (portés de cycle-navigator.js)
CYCLE_PARAMS = {
    "m_rise_center": 5.0,
    "m_fall_center": 24.0,
    "k_rise": 0.8,
    "k_fall": 1.2,
    "p_shape": 1.15,
    "floor": 0,
    "ceil": 100,
}


@dataclass
class MacroHistoryPoint:
    """Point de données macro historique"""
    date: datetime
    vix: Optional[float] = None
    dxy: Optional[float] = None
    dxy_change_30d: Optional[float] = None
    macro_stress: bool = False
    penalty: int = 0


class HistoricalDataSources:
    """Gestionnaire des sources de données historiques pour le DI Backtest"""

    def __init__(self):
        self._fred_cache: Dict[str, pd.DataFrame] = {}
        self._fear_greed_cache: Optional[pd.Series] = None

    # ========== CYCLE SCORE (déterministe) ==========

    def get_months_after_halving(self, date: datetime) -> float:
        """Calcule le nombre de mois après le dernier halving pour une date"""
        # Trouver le halving le plus récent avant cette date
        past_halvings = [h for h in BITCOIN_HALVINGS if h <= date]
        if not past_halvings:
            # Avant le premier halving connu - utiliser une approximation
            # Le genesis block est le 2009-01-03, premier halving ~2012-11-28
            return 48.0  # Fin de cycle pré-2012

        last_halving = max(past_halvings)
        days_diff = (date - last_halving).days
        months = days_diff / 30.44  # Mois moyens
        return max(0, months)

    def cycle_score_from_months(self, months_after_halving: float) -> float:
        """
        Calcule le Cycle Score (0-100) basé sur le modèle double-sigmoïde
        Port Python de cycle-navigator.js
        """
        if months_after_halving < 0:
            return 50.0

        m48 = months_after_halving % 48  # Cycle ~4 ans

        p = CYCLE_PARAMS
        rise = 1 / (1 + math.exp(-p["k_rise"] * (m48 - p["m_rise_center"])))
        fall = 1 / (1 + math.exp(-p["k_fall"] * (p["m_fall_center"] - m48)))
        base = rise * fall
        score = (base ** p["p_shape"]) * 100

        # Clamp
        return max(p["floor"], min(p["ceil"], score))

    def cycle_score_derivative(self, months_after_halving: float) -> float:
        """
        Finite-difference derivative of cycle score (pts/month).
        Positive = ascending (early cycle), negative = descending (late cycle).
        """
        delta = 0.5
        s_plus = self.cycle_score_from_months(months_after_halving + delta)
        s_minus = self.cycle_score_from_months(months_after_halving - delta)
        return (s_plus - s_minus) / (2 * delta)

    def cycle_confidence(self, months_after_halving: float) -> float:
        """
        Confidence based on distance from phase center (0.4 to 0.9).
        Port of estimateCyclePosition() confidence logic in cycle-navigator.js.
        """
        m = months_after_halving % 48
        PHASE_WINDOWS = {
            (0, 7): {"center": 3, "half": 3},       # accumulation
            (7, 19): {"center": 12, "half": 6},      # bull_build
            (19, 25): {"center": 21, "half": 3},     # peak
            (25, 37): {"center": 30, "half": 6},     # bear
            (37, 49): {"center": 42, "half": 6},     # pre_accumulation
        }
        for (lo, hi), win in PHASE_WINDOWS.items():
            if lo <= m < hi:
                dist = abs(m - win["center"])
                norm = min(1.0, dist / win["half"]) if win["half"] > 0 else 1.0
                return 0.4 + 0.5 * (1 - norm)
        return 0.5  # fallback

    def compute_historical_cycle_scores(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.Series:
        """Calcule les Cycle Scores historiques pour une période"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        scores = []

        for date in dates:
            months = self.get_months_after_halving(date.to_pydatetime())
            score = self.cycle_score_from_months(months)
            scores.append(score)

        return pd.Series(scores, index=dates, name='cycle_score')

    def compute_corrected_cycle_scores(
        self,
        start_date: datetime,
        end_date: datetime,
        prices: pd.Series,
        correction_pts: int = 10,
        momentum_window: int = 180,
        high_threshold: float = 85.0,
        low_threshold: float = 40.0,
        bearish_momentum: float = -0.20,
        bullish_momentum: float = 0.30,
    ) -> pd.Series:
        """
        Cycle Score with conditional correction when price contradicts the model.

        Only corrects at strong divergence points — no permanent blend.
        Avoids double-counting momentum (already in onchain proxy).

        Args:
            correction_pts: Points to add/subtract when divergence detected
            momentum_window: Days for price momentum (default 180 = 6 months)
            high_threshold: Cycle score above which bearish price triggers correction
            low_threshold: Cycle score below which bullish price triggers correction
            bearish_momentum: 6-month return threshold for bearish divergence
            bullish_momentum: 6-month return threshold for bullish divergence
        """
        sigmoid = self.compute_historical_cycle_scores(start_date, end_date)

        # Aligner prix avec les dates du cycle
        momentum_6m = prices.pct_change(momentum_window).reindex(sigmoid.index, method='ffill')

        corrected = sigmoid.copy()

        # Cycle dit "haut" mais prix en chute forte → correction baissière
        mask_bearish = (sigmoid > high_threshold) & (momentum_6m < bearish_momentum)
        corrected.loc[mask_bearish] -= correction_pts

        # Cycle dit "bas" mais prix en forte hausse → correction haussière
        mask_bullish = (sigmoid < low_threshold) & (momentum_6m > bullish_momentum)
        corrected.loc[mask_bullish] += correction_pts

        return corrected.clip(0, 100).rename('cycle_score')

    # ========== ONCHAIN PROXY (basé sur prix) ==========

    def compute_onchain_proxy(
        self,
        prices: pd.Series,
        dma_window: int = 200,
        rsi_window: int = 14,
        momentum_window: int = 90,
        normalization: str = "fixed",
    ) -> pd.Series:
        """
        Calcule un proxy OnChain basé sur les prix (0-100)

        Composants:
        - Distance au 200 DMA (40%)
        - RSI adapté (30%)  — toujours 0-100 natif, pas de percentile
        - Momentum 90j (30%)

        Args:
            normalization: "fixed" (backward compat) ou "adaptive" (expanding percentile)

        Returns:
            pd.Series avec score 0-100 (plus haut = plus bullish)
        """
        if len(prices) < dma_window:
            logger.warning(f"Pas assez de données pour le proxy OnChain ({len(prices)} < {dma_window})")
            return pd.Series(50.0, index=prices.index, name='onchain_proxy')

        # 1. Distance au 200 DMA
        dma = prices.rolling(dma_window, min_periods=1).mean()
        distance_pct = ((prices - dma) / dma) * 100

        if normalization == "adaptive":
            # Expanding percentile: rang relatif à tout l'historique passé
            dma_score = distance_pct.expanding(min_periods=dma_window).rank(pct=True) * 100
            # Fallback à normalisation fixe pour les premiers jours (warm-up)
            fixed_dma = (distance_pct + 50).clip(0, 100)
            dma_score = dma_score.fillna(fixed_dma)
        else:
            # Normaliser [-50%, +50%] → [0, 100]
            dma_score = (distance_pct + 50).clip(0, 100)

        # 2. RSI (déjà 0-100 par construction — PAS de percentile)
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_window, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # 3. Momentum 90j
        returns_90d = prices.pct_change(momentum_window)

        if normalization == "adaptive":
            momentum_score = returns_90d.expanding(min_periods=momentum_window).rank(pct=True) * 100
            fixed_mom = ((returns_90d * 100) + 50).clip(0, 100)
            momentum_score = momentum_score.fillna(fixed_mom)
        else:
            # Normaliser [-50%, +100%] → [0, 100]
            momentum_score = ((returns_90d * 100) + 50).clip(0, 100)

        # Combinaison pondérée
        onchain_proxy = (
            dma_score * 0.40 +
            rsi * 0.30 +
            momentum_score * 0.30
        ).fillna(50.0)

        return onchain_proxy.rename('onchain_proxy')

    # ========== RISK SCORE (basé sur prix) ==========

    def compute_risk_score(
        self,
        prices: pd.Series,
        vol_window: int = 90,
        dd_window: int = 90,
        normalization: str = "fixed",
    ) -> pd.Series:
        """
        Calcule le Risk Score historique (0-100)

        ATTENTION: Score POSITIF - plus haut = plus robuste/sûr
        NE JAMAIS inverser avec 100 - score

        Args:
            normalization: "fixed" (backward compat) ou "adaptive" (expanding percentile)

        Composants:
        - Volatilité inversée (50%): basse vol = score haut
        - Drawdown inversé (50%): petit DD = score haut
        """
        if len(prices) < vol_window:
            return pd.Series(50.0, index=prices.index, name='risk_score')

        returns = prices.pct_change()

        # 1. Volatilité annualisée
        vol = returns.rolling(vol_window, min_periods=1).std() * np.sqrt(365)

        if normalization == "adaptive":
            # Expanding percentile inversé: rank le plus haut = vol la plus basse
            vol_score = (-vol).expanding(min_periods=vol_window).rank(pct=True) * 100
            # Fallback warm-up
            fixed_vol = 100 - ((vol * 100).clip(0, 150) / 1.5)
            vol_score = vol_score.fillna(fixed_vol)
        else:
            vol_normalized = (vol * 100).clip(0, 150)
            vol_score = 100 - (vol_normalized / 1.5)  # 0% vol → 100, 150% vol → 0

        # 2. Drawdown
        rolling_max = prices.rolling(dd_window, min_periods=1).max()
        drawdown = (prices - rolling_max) / rolling_max  # Négatif

        if normalization == "adaptive":
            # Expanding percentile: rank le plus haut = DD le moins négatif
            dd_score = drawdown.expanding(min_periods=dd_window).rank(pct=True) * 100
            # Fallback warm-up
            fixed_dd = (100 + (drawdown * 100)).clip(0, 100)
            dd_score = dd_score.fillna(fixed_dd)
        else:
            dd_score = 100 + (drawdown * 100)  # 0% DD → 100, -80% DD → 20
            dd_score = dd_score.clip(0, 100)

        # Combinaison
        risk_score = (vol_score * 0.5 + dd_score * 0.5).fillna(50.0).clip(0, 100)

        return risk_score.rename('risk_score')

    # ========== SENTIMENT (Fear & Greed + proxy) ==========

    async def fetch_fear_greed_history(self, days: int = 365) -> pd.Series:
        """
        Récupère l'historique Fear & Greed Index (0-100)
        Note: API limitée à ~365 jours max
        """
        if self._fear_greed_cache is not None and len(self._fear_greed_cache) >= days:
            return self._fear_greed_cache.tail(days)

        url = f"https://api.alternative.me/fng/?limit={days}&format=json"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

            records = data.get("data", [])
            if not records:
                logger.warning("Pas de données Fear & Greed")
                return pd.Series(dtype=float, name='sentiment')

            # Convertir en Series
            points = []
            for record in records:
                try:
                    ts = int(record["timestamp"])
                    value = int(record["value"])
                    date = datetime.fromtimestamp(ts)
                    points.append((date, value))
                except (KeyError, ValueError):
                    continue

            if not points:
                return pd.Series(dtype=float, name='sentiment')

            df = pd.DataFrame(points, columns=['date', 'value'])
            df = df.sort_values('date').set_index('date')
            series = df['value'].rename('sentiment')

            self._fear_greed_cache = series
            logger.info(f"Fear & Greed: {len(series)} points récupérés")
            return series

        except Exception as e:
            logger.error(f"Erreur récupération Fear & Greed: {e}")
            return pd.Series(dtype=float, name='sentiment')

    def compute_sentiment_proxy(
        self,
        prices: pd.Series,
        vol_weight: float = 0.5,
        momentum_weight: float = 0.5
    ) -> pd.Series:
        """
        Calcule un proxy sentiment pour les périodes sans Fear & Greed

        Basé sur:
        - Volatilité récente (inversée): haute vol = fear
        - Momentum court terme: hausse = greed
        """
        # Volatilité 14j
        returns = prices.pct_change()
        vol_14d = returns.rolling(14, min_periods=1).std() * np.sqrt(365)
        # Normaliser: vol haute = fear (score bas)
        vol_score = 100 - (vol_14d * 100).clip(0, 100)

        # Momentum 7j
        momentum_7d = prices.pct_change(7)
        # Normaliser [-20%, +20%] → [0, 100]
        momentum_score = ((momentum_7d * 100) / 0.4 + 50).clip(0, 100)

        # Combinaison
        sentiment_proxy = (
            vol_score * vol_weight +
            momentum_score * momentum_weight
        ).fillna(50.0)

        return sentiment_proxy.rename('sentiment_proxy')

    def compute_sentiment_proxy_v2(
        self,
        prices: pd.Series,
        normalization: str = "adaptive",
    ) -> pd.Series:
        """
        Sentiment proxy V2 — enrichi et adaptatif.

        4 composants (vs 2 en V1):
        - Vol 30j inversée (25%): haute vol = fear, fenêtre plus longue que V1
        - Momentum 14j (25%): hausse = greed
        - RSI 14j rescalé (25%): RSI<30 = extreme fear, RSI>70 = extreme greed
        - Distance au 52-week high (25%): proche du ATH = euphoria, loin = pain

        Args:
            normalization: "fixed" (V1 compat) ou "adaptive" (expanding percentile)
        """
        returns = prices.pct_change()

        # 1. Vol 30j inversée (fenêtre plus stable que 14j)
        vol_30d = returns.rolling(30, min_periods=1).std() * np.sqrt(365)
        if normalization == "adaptive":
            # Expanding percentile inversé: basse vol = haut score = greed
            vol_score = (-vol_30d).expanding(min_periods=30).rank(pct=True) * 100
            fixed_vol = 100 - (vol_30d * 100).clip(0, 100)
            vol_score = vol_score.fillna(fixed_vol)
        else:
            vol_score = 100 - (vol_30d * 100).clip(0, 100)

        # 2. Momentum 14j
        momentum_14d = prices.pct_change(14)
        if normalization == "adaptive":
            momentum_score = momentum_14d.expanding(min_periods=14).rank(pct=True) * 100
            fixed_mom = ((momentum_14d * 100) / 0.4 + 50).clip(0, 100)
            momentum_score = momentum_score.fillna(fixed_mom)
        else:
            momentum_score = ((momentum_14d * 100) / 0.4 + 50).clip(0, 100)

        # 3. RSI 14j (déjà 0-100 natif, pas de percentile)
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # 4. Distance au 52-week high (proxy euphoria/pain)
        high_52w = prices.rolling(365, min_periods=30).max()
        distance_from_high = (prices - high_52w) / high_52w  # [-1, 0]
        if normalization == "adaptive":
            dist_score = distance_from_high.expanding(min_periods=90).rank(pct=True) * 100
            fixed_dist = ((distance_from_high + 1) * 100).clip(0, 100)
            dist_score = dist_score.fillna(fixed_dist)
        else:
            dist_score = ((distance_from_high + 1) * 100).clip(0, 100)

        # Combinaison pondérée (4 composants = plus diversifié)
        sentiment = (
            vol_score * 0.25 +
            momentum_score * 0.25 +
            rsi * 0.25 +
            dist_score * 0.25
        ).fillna(50.0).clip(0, 100)

        return sentiment.rename('sentiment_proxy')

    # ========== MACRO (VIX/DXY via FRED) ==========

    async def fetch_fred_series(
        self,
        series_id: str,
        fred_api_key: str,
        start_date: str = "2017-01-01"
    ) -> pd.DataFrame:
        """Récupère une série FRED complète"""
        cache_key = f"{series_id}_{start_date}"
        if cache_key in self._fred_cache:
            return self._fred_cache[cache_key]

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
                response.raise_for_status()
                data = response.json()

            observations = data.get("observations", [])
            records = []

            for obs in observations:
                if obs["value"] != "." and obs["value"] is not None:
                    try:
                        value = float(obs["value"])
                        date = datetime.strptime(obs["date"], "%Y-%m-%d")
                        records.append({"date": date, "value": value})
                    except (ValueError, TypeError):
                        continue

            df = pd.DataFrame(records)
            if not df.empty:
                df = df.set_index("date").sort_index()
                self._fred_cache[cache_key] = df
                logger.info(f"FRED {series_id}: {len(df)} points récupérés")

            return df

        except Exception as e:
            logger.error(f"Erreur FRED {series_id}: {e}")
            return pd.DataFrame()

    async def fetch_historical_macro(
        self,
        user_id: str,
        start_date: str = "2017-01-01"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Récupère les données macro historiques (VIX et DXY)

        Returns:
            Tuple (vix_df, dxy_df) avec colonnes ['value']
        """
        from services.user_secrets import get_user_secrets
        secrets = get_user_secrets(user_id)
        fred_api_key = secrets.get("fred", {}).get("api_key") or os.getenv("FRED_API_KEY")

        if not fred_api_key:
            logger.warning("FRED API key non configurée")
            return pd.DataFrame(), pd.DataFrame()

        # Récupérer VIX (VIXCLS) et DXY (DTWEXBGS) en parallèle
        vix_task = self.fetch_fred_series("VIXCLS", fred_api_key, start_date)
        dxy_task = self.fetch_fred_series("DTWEXBGS", fred_api_key, start_date)

        vix_df, dxy_df = await asyncio.gather(vix_task, dxy_task)

        return vix_df, dxy_df

    def compute_macro_penalty(
        self,
        vix_series: pd.Series,
        dxy_series: pd.Series,
        vix_threshold: float = 30.0,
        dxy_change_threshold: float = 5.0
    ) -> pd.Series:
        """
        Calcule la pénalité macro historique (-15 ou 0) — V1 binaire

        Règle: VIX > 30 OU DXY +5% sur 30j → -15 points
        """
        # Aligner les index
        common_dates = vix_series.index.intersection(dxy_series.index)

        penalties = pd.Series(0, index=common_dates, name='macro_penalty')

        # VIX stress
        vix_stress = vix_series.loc[common_dates] > vix_threshold

        # DXY stress (variation 30j)
        dxy_pct_change = dxy_series.pct_change(30) * 100
        dxy_stress = dxy_pct_change.loc[common_dates] >= dxy_change_threshold

        # Pénalité si l'un ou l'autre
        stress_mask = vix_stress | dxy_stress
        penalties.loc[stress_mask] = -15

        return penalties

    def compute_macro_penalty_v2(
        self,
        vix_series: pd.Series,
        dxy_series: pd.Series,
        vix_start: float = 20.0,
        vix_max: float = 45.0,
        vix_penalty_max: float = 10.0,
        dxy_change_start: float = 2.0,
        dxy_change_max: float = 10.0,
        dxy_penalty_max: float = 8.0,
        total_cap: float = -15.0,
    ) -> pd.Series:
        """
        Calcule la pénalité macro historique — V2 graduée additive

        VIX : linéaire de 0 (≤start) à -vix_penalty_max (≥vix_max)
        DXY : linéaire de 0 (change ≤start) à -dxy_penalty_max (change ≥max)
        Total = VIX + DXY, cappé à total_cap

        Plus réaliste : VIX=31 → ~-4pts (pas -15), VIX=31 + DXY+5% → ~-8pts
        """
        common_dates = vix_series.index.intersection(dxy_series.index)

        # VIX penalty: linéaire start→max
        vix = vix_series.reindex(common_dates)
        vix_penalty = -np.clip(
            (vix - vix_start) / (vix_max - vix_start) * vix_penalty_max,
            0, vix_penalty_max,
        )

        # DXY penalty: linéaire sur changement 30j
        dxy_pct_change = dxy_series.pct_change(30).reindex(common_dates) * 100
        dxy_penalty = -np.clip(
            (dxy_pct_change - dxy_change_start) / (dxy_change_max - dxy_change_start) * dxy_penalty_max,
            0, dxy_penalty_max,
        )

        # Somme additive cappée
        penalties = (vix_penalty.fillna(0) + dxy_penalty.fillna(0)).clip(lower=total_cap, upper=0)

        return penalties.rename('macro_penalty')

    # ========== PRIX CRYPTO ==========

    async def get_btc_prices(
        self,
        days: int = 3000,
        force_refresh: bool = False
    ) -> pd.Series:
        """
        Récupère l'historique des prix BTC

        Utilise le cache local (data/price_history/) en priorité.
        Ne télécharge que si le cache est absent ou insuffisant.

        Args:
            days: Nombre de jours d'historique requis (défaut 3000 = ~8 ans)
            force_refresh: Forcer le téléchargement même si cache existant

        Returns:
            pd.Series avec DatetimeIndex et prix BTC
        """
        # Lire le cache local directement — aucun appel API
        history = get_cached_history("BTC", days=None)

        if history and len(history) >= days * 0.9 and not force_refresh:
            logger.info(f"Cache BTC local: {len(history)} points (suffisant pour {days})")
        else:
            # Cache absent ou insuffisant → télécharger
            if not history:
                logger.info("Pas de cache BTC, téléchargement initial...")
            else:
                logger.info(f"Cache BTC: {len(history)} pts < {days}, téléchargement...")
            success = await download_historical_data("BTC", days=days, force_refresh=force_refresh or not history)
            if not success:
                logger.error("Échec téléchargement prix BTC")
                return pd.Series(dtype=float)
            history = get_cached_history("BTC", days=None)

        if not history:
            return pd.Series(dtype=float)

        logger.info(f"Historique BTC chargé: {len(history)} points")

        # Convertir en Series avec DatetimeIndex pandas
        timestamps = [ts for ts, price in history]
        prices = [price for ts, price in history]

        # Créer un DatetimeIndex pandas à partir des timestamps Unix
        date_index = pd.to_datetime(timestamps, unit='s')

        series = pd.Series(prices, index=date_index, name='btc_price')
        return series.sort_index()

    async def get_multi_asset_prices(
        self,
        symbols: List[str],
        days: int = 3000,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Récupère les prix de plusieurs actifs

        Utilise le cache local (data/price_history/) en priorité.
        Ne télécharge que si le cache est absent ou insuffisant.

        Args:
            symbols: Liste de symboles (ex: ["BTC", "ETH"])
            days: Nombre de jours d'historique requis
            force_refresh: Forcer le téléchargement

        Returns:
            pd.DataFrame avec une colonne par symbole
        """
        results = {}

        for symbol in symbols:
            try:
                # Lire le cache local directement — aucun appel API
                history = get_cached_history(symbol, days=None)

                if history and len(history) >= days * 0.9 and not force_refresh:
                    logger.info(f"Cache {symbol} local: {len(history)} points (suffisant)")
                else:
                    # Cache absent ou insuffisant → télécharger
                    if not history:
                        logger.info(f"Pas de cache {symbol}, téléchargement...")
                    else:
                        logger.info(f"Cache {symbol}: {len(history)} pts < {days}, téléchargement...")
                    success = await download_historical_data(symbol, days=days, force_refresh=force_refresh or not history)
                    if not success:
                        logger.warning(f"Échec téléchargement {symbol}")
                        continue
                    history = get_cached_history(symbol, days=None)

                if history:
                    logger.info(f"Historique {symbol}: {len(history)} points")
                    timestamps = [ts for ts, price in history]
                    prices = [price for ts, price in history]
                    date_index = pd.to_datetime(timestamps, unit='s')
                    results[symbol] = pd.Series(prices, index=date_index, name=symbol)
            except Exception as e:
                logger.warning(f"Erreur récupération {symbol}: {e}")
                continue

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)


# Instance singleton
historical_data_sources = HistoricalDataSources()

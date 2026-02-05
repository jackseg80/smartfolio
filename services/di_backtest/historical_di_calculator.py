"""
Historical Decision Index Calculator
Reconstruction du Decision Index pour des périodes historiques

Formule DI (source: services/execution/strategy_registry.py):
    raw_score = (cycle × 0.30 + onchain × 0.35 + risk × 0.25 + sentiment × 0.10)
    adjusted_score = raw_score × phase_factor + macro_penalty
    final_score = clamp(adjusted_score, 0, 100)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from .data_sources import historical_data_sources

logger = logging.getLogger(__name__)


@dataclass
class DIWeights:
    """Poids des composants du Decision Index"""
    cycle: float = 0.30
    onchain: float = 0.35
    risk: float = 0.25
    sentiment: float = 0.10

    def validate(self) -> bool:
        """Vérifie que les poids somment à 1"""
        total = self.cycle + self.onchain + self.risk + self.sentiment
        return abs(total - 1.0) < 0.01


@dataclass
class PhaseFactors:
    """Facteurs d'ajustement par phase de marché"""
    bearish: float = 0.85
    moderate: float = 1.0
    bullish: float = 1.05


@dataclass
class DIHistoryPoint:
    """Point de données DI historique"""
    date: datetime
    decision_index: float
    cycle_score: float
    onchain_score: float
    risk_score: float
    sentiment_score: float
    phase: str
    phase_factor: float
    macro_penalty: int
    raw_score: float
    btc_price: Optional[float] = None


@dataclass
class DIBacktestData:
    """Données complètes pour un backtest DI"""
    start_date: datetime
    end_date: datetime
    di_history: List[DIHistoryPoint]
    df: pd.DataFrame  # DataFrame complet pour analyse
    metadata: Dict = field(default_factory=dict)


class HistoricalDICalculator:
    """Calculateur du Decision Index historique"""

    def __init__(
        self,
        weights: Optional[DIWeights] = None,
        phase_factors: Optional[PhaseFactors] = None
    ):
        self.weights = weights or DIWeights()
        self.phase_factors = phase_factors or PhaseFactors()
        self.data_sources = historical_data_sources

    def _determine_phase(self, cycle_score: float) -> tuple[str, float]:
        """
        Détermine la phase de marché basée sur le Cycle Score

        Returns:
            (phase_name, phase_factor)
        """
        if cycle_score < 40:
            return "bearish", self.phase_factors.bearish
        elif cycle_score < 70:
            return "moderate", self.phase_factors.moderate
        else:
            return "bullish", self.phase_factors.bullish

    async def calculate_historical_di(
        self,
        user_id: str,
        start_date: str = "2017-01-01",
        end_date: Optional[str] = None,
        include_macro: bool = True
    ) -> DIBacktestData:
        """
        Calcule le Decision Index historique pour une période

        Args:
            user_id: ID utilisateur pour clé FRED
            start_date: Date de début (YYYY-MM-DD)
            end_date: Date de fin (None = aujourd'hui)
            include_macro: Inclure la pénalité macro (VIX/DXY)

        Returns:
            DIBacktestData avec historique complet
        """
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

        # Buffer pour les rolling windows (200 DMA, etc.)
        buffer_days = 250
        buffer_start = start_dt - pd.Timedelta(days=buffer_days)

        logger.info(f"Calcul DI historique: {start_date} → {end_date or 'maintenant'}")

        # 1. Récupérer les prix BTC (tout le cache disponible)
        btc_prices_full = await self.data_sources.get_btc_prices(days=3650)
        if btc_prices_full.empty:
            raise ValueError("Impossible de récupérer les prix BTC")

        # S'assurer que l'index est un DatetimeIndex pandas
        if not isinstance(btc_prices_full.index, pd.DatetimeIndex):
            btc_prices_full.index = pd.to_datetime(btc_prices_full.index)

        # Garder les données avec buffer pour les calculs rolling
        btc_prices_with_buffer = btc_prices_full[btc_prices_full.index >= buffer_start]
        if end_date:
            btc_prices_with_buffer = btc_prices_with_buffer[btc_prices_with_buffer.index <= end_dt]

        # La série pour le filtrage final (sans buffer)
        btc_prices = btc_prices_with_buffer[btc_prices_with_buffer.index >= start_dt]

        logger.info(f"Prix BTC: {len(btc_prices)} points demandés, {len(btc_prices_with_buffer)} avec buffer")

        # 2. Calculer les composants SUR LES DONNÉES AVEC BUFFER
        # Les rolling windows ont besoin de données historiques

        # Cycle Score (déterministe - pas besoin de buffer)
        cycle_scores = self.data_sources.compute_historical_cycle_scores(
            btc_prices_with_buffer.index.min(),
            btc_prices_with_buffer.index.max()
        )

        # OnChain Proxy (basé sur prix avec 200 DMA)
        onchain_scores = self.data_sources.compute_onchain_proxy(btc_prices_with_buffer)

        # Risk Score (basé sur prix avec rolling windows)
        risk_scores = self.data_sources.compute_risk_score(btc_prices_with_buffer)

        # Sentiment (Fear & Greed si disponible, sinon proxy)
        sentiment_scores = await self._get_sentiment_scores(btc_prices_with_buffer)

        # Macro penalty (si demandé et clé FRED disponible)
        macro_penalties = pd.Series(0, index=btc_prices_with_buffer.index, name='macro_penalty')
        if include_macro:
            try:
                vix_df, dxy_df = await self.data_sources.fetch_historical_macro(user_id, start_date)
                if not vix_df.empty and not dxy_df.empty:
                    macro_penalties = self.data_sources.compute_macro_penalty(
                        vix_df['value'],
                        dxy_df['value']
                    )
                    # Reindex to match btc_prices_with_buffer
                    macro_penalties = macro_penalties.reindex(btc_prices_with_buffer.index, method='ffill').fillna(0)
            except Exception as e:
                logger.warning(f"Macro data non disponible: {e}")

        # 3. Assembler le DataFrame avec toutes les données (buffer inclus)
        df_full = pd.DataFrame({
            'btc_price': btc_prices_with_buffer,
            'cycle_score': cycle_scores.reindex(btc_prices_with_buffer.index, method='nearest'),
            'onchain_score': onchain_scores.reindex(btc_prices_with_buffer.index),
            'risk_score': risk_scores.reindex(btc_prices_with_buffer.index),
            'sentiment_score': sentiment_scores.reindex(btc_prices_with_buffer.index),
            'macro_penalty': macro_penalties.reindex(btc_prices_with_buffer.index).fillna(0),
        }).dropna()

        # 4. Filtrer à la période demandée (sans le buffer)
        # .copy() pour éviter SettingWithCopyWarning
        df = df_full[(df_full.index >= start_dt) & (df_full.index <= end_dt)].copy()

        logger.info(f"DataFrame assemblé: {len(df)} points (buffer: {len(df_full) - len(df)})")

        # 5. Calculer le Decision Index
        w = self.weights
        df['raw_score'] = (
            df['cycle_score'] * w.cycle +
            df['onchain_score'] * w.onchain +
            df['risk_score'] * w.risk +
            df['sentiment_score'] * w.sentiment
        )

        # Phase et facteur
        phases = df['cycle_score'].apply(lambda x: self._determine_phase(x))
        df['phase'] = phases.apply(lambda x: x[0])
        df['phase_factor'] = phases.apply(lambda x: x[1])

        # Score ajusté
        df['decision_index'] = (
            df['raw_score'] * df['phase_factor'] + df['macro_penalty']
        ).clip(0, 100)

        # 5. Convertir en liste de points
        history_points = []
        for idx, row in df.iterrows():
            point = DIHistoryPoint(
                date=idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx,
                decision_index=row['decision_index'],
                cycle_score=row['cycle_score'],
                onchain_score=row['onchain_score'],
                risk_score=row['risk_score'],
                sentiment_score=row['sentiment_score'],
                phase=row['phase'],
                phase_factor=row['phase_factor'],
                macro_penalty=int(row['macro_penalty']),
                raw_score=row['raw_score'],
                btc_price=row['btc_price'],
            )
            history_points.append(point)

        # Metadata (avec conversion des types numpy en types Python natifs)
        metadata = {
            "total_points": int(len(df)),
            "weights": {
                "cycle": float(w.cycle),
                "onchain": float(w.onchain),
                "risk": float(w.risk),
                "sentiment": float(w.sentiment),
            },
            "phase_factors": {
                "bearish": float(self.phase_factors.bearish),
                "moderate": float(self.phase_factors.moderate),
                "bullish": float(self.phase_factors.bullish),
            },
            "di_stats": {
                "mean": float(df['decision_index'].mean()),
                "std": float(df['decision_index'].std()),
                "min": float(df['decision_index'].min()),
                "max": float(df['decision_index'].max()),
                "median": float(df['decision_index'].median()),
            },
            "macro_penalties_count": int((df['macro_penalty'] < 0).sum()),
        }

        return DIBacktestData(
            start_date=df.index.min().to_pydatetime(),
            end_date=df.index.max().to_pydatetime(),
            di_history=history_points,
            df=df,
            metadata=metadata
        )

    async def _get_sentiment_scores(self, btc_prices: pd.Series) -> pd.Series:
        """
        Récupère les scores sentiment (Fear & Greed si disponible, sinon proxy)
        """
        # Essayer Fear & Greed API
        fg_scores = await self.data_sources.fetch_fear_greed_history(days=365)

        if fg_scores.empty:
            # Fallback: proxy basé sur prix
            logger.info("Fear & Greed non disponible, utilisation du proxy")
            return self.data_sources.compute_sentiment_proxy(btc_prices)

        # Combiner F&G récent + proxy pour données anciennes
        all_dates = btc_prices.index

        # Reindex F&G avec forward fill pour les gaps
        fg_reindexed = fg_scores.reindex(all_dates, method='nearest').astype(float)

        # Pour les dates avant F&G, utiliser le proxy
        proxy = self.data_sources.compute_sentiment_proxy(btc_prices)
        fg_min_date = fg_scores.index.min() if not fg_scores.empty else all_dates.max()

        # Créer le résultat en float pour éviter les warnings de types
        result = fg_reindexed.copy()
        mask = result.index < fg_min_date
        result.loc[mask] = proxy.loc[mask].values

        return result.fillna(50.0).rename('sentiment_score')

    def get_di_for_date(
        self,
        df: pd.DataFrame,
        target_date: datetime
    ) -> Optional[DIHistoryPoint]:
        """Récupère le DI pour une date spécifique"""
        if target_date not in df.index:
            # Chercher la date la plus proche
            closest_idx = df.index.get_indexer([target_date], method='nearest')[0]
            if closest_idx < 0 or closest_idx >= len(df):
                return None
            target_date = df.index[closest_idx]

        row = df.loc[target_date]
        return DIHistoryPoint(
            date=target_date,
            decision_index=row['decision_index'],
            cycle_score=row['cycle_score'],
            onchain_score=row['onchain_score'],
            risk_score=row['risk_score'],
            sentiment_score=row['sentiment_score'],
            phase=row['phase'],
            phase_factor=row['phase_factor'],
            macro_penalty=int(row['macro_penalty']),
            raw_score=row['raw_score'],
            btc_price=row.get('btc_price'),
        )

    def analyze_di_periods(
        self,
        df: pd.DataFrame,
        events: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Analyse le comportement du DI sur des périodes clés

        Args:
            df: DataFrame avec DI historique
            events: Liste d'événements [{name, start, end}]
        """
        default_events = [
            {"name": "Bull Run 2017", "start": "2017-01-01", "end": "2017-12-31"},
            {"name": "Crash 2018", "start": "2018-01-01", "end": "2018-12-31"},
            {"name": "COVID Crash", "start": "2020-02-01", "end": "2020-05-01"},
            {"name": "Bull Run 2020-2021", "start": "2020-10-01", "end": "2021-11-15"},
            {"name": "Bear Market 2022", "start": "2021-11-15", "end": "2022-11-15"},
            {"name": "Recovery 2023-2024", "start": "2022-11-15", "end": "2024-04-01"},
        ]

        events = events or default_events
        analysis = {}

        for event in events:
            start = datetime.strptime(event["start"], "%Y-%m-%d")
            end = datetime.strptime(event["end"], "%Y-%m-%d")

            period_df = df[(df.index >= start) & (df.index <= end)]

            if period_df.empty:
                analysis[event["name"]] = {"error": "No data for period"}
                continue

            # Prix BTC
            btc_start = period_df['btc_price'].iloc[0]
            btc_end = period_df['btc_price'].iloc[-1]
            btc_return = (btc_end / btc_start - 1) * 100

            # DI stats
            di_mean = period_df['decision_index'].mean()
            di_std = period_df['decision_index'].std()

            # Corrélation DI vs returns futurs 30j
            future_returns = df['btc_price'].pct_change(30).shift(-30)
            correlation = period_df['decision_index'].corr(
                future_returns.reindex(period_df.index)
            )

            analysis[event["name"]] = {
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "days": len(period_df),
                "btc_return_pct": round(btc_return, 2),
                "di_mean": round(di_mean, 2),
                "di_std": round(di_std, 2),
                "di_min": round(period_df['decision_index'].min(), 2),
                "di_max": round(period_df['decision_index'].max(), 2),
                "di_btc_correlation": round(correlation, 3) if not np.isnan(correlation) else None,
                "dominant_phase": period_df['phase'].mode().iloc[0] if not period_df['phase'].mode().empty else None,
                "macro_penalties": (period_df['macro_penalty'] < 0).sum(),
            }

        return analysis


# Instance singleton
historical_di_calculator = HistoricalDICalculator()

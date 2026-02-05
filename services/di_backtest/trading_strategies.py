"""
Trading Strategies Based on Decision Index
Stratégies de trading utilisant le DI pour le backtesting

Ces stratégies héritent de PortfolioStrategy (backtesting_engine.py)
et peuvent être utilisées directement avec BacktestingEngine.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass

from services.backtesting_engine import PortfolioStrategy


@dataclass
class DIStrategyConfig:
    """Configuration pour les stratégies DI"""
    # Seuils DI
    di_extreme_fear: float = 20.0  # DI < 20 = extreme fear
    di_fear: float = 40.0          # DI < 40 = fear
    di_neutral_low: float = 50.0   # DI 40-60 = neutral
    di_neutral_high: float = 60.0
    di_greed: float = 70.0         # DI > 70 = greed
    di_extreme_greed: float = 80.0 # DI > 80 = extreme greed

    # Allocations par seuil (% risky assets)
    alloc_extreme_fear: float = 0.30   # 30% risky, 70% stables
    alloc_fear: float = 0.50           # 50/50
    alloc_neutral: float = 0.60        # 60% risky
    alloc_greed: float = 0.75          # 75% risky
    alloc_extreme_greed: float = 0.85  # 85% risky

    # Paramètres momentum
    momentum_lookback: int = 7   # Jours pour calcul momentum DI
    momentum_threshold: float = 5.0  # Variation DI minimale

    # Paramètres signal
    signal_entry_threshold: float = 40.0  # BUY quand DI croise au-dessus
    signal_exit_threshold: float = 60.0   # SELL quand DI croise en-dessous
    signal_confirmation_days: int = 3     # Jours de confirmation
    signal_min_holding_days: int = 14     # Holding minimum


class DIThresholdStrategy(PortfolioStrategy):
    """
    Stratégie S1: Allocation basée sur seuils DI

    Règles:
    - DI < 30 → 70% stables, 30% risky
    - DI 30-50 → 50% / 50%
    - DI 50-70 → 30% stables, 70% risky
    - DI > 70 → 15% stables, 85% risky

    Simple et intuitive, idéale pour validation du DI.
    """

    def __init__(self, config: Optional[DIStrategyConfig] = None):
        super().__init__("DI Threshold")
        self.config = config or DIStrategyConfig()
        self.di_series: Optional[pd.Series] = None

    def set_di_series(self, di_series: pd.Series):
        """Injecte la série DI historique"""
        self.di_series = di_series

    def get_weights(
        self,
        date: pd.Timestamp,
        price_data: pd.DataFrame,
        current_weights: pd.Series,
        **kwargs
    ) -> pd.Series:
        """Calcule les poids basés sur le DI à la date donnée"""
        # Récupérer DI si fourni dans kwargs ou via série
        di_value = kwargs.get('di_value')
        if di_value is None and self.di_series is not None:
            try:
                di_value = self.di_series.loc[date]
            except KeyError:
                # Chercher la date la plus proche
                idx = self.di_series.index.get_indexer([date], method='nearest')[0]
                di_value = self.di_series.iloc[idx]

        if di_value is None:
            di_value = 50.0  # Neutre par défaut

        # Déterminer allocation risky
        c = self.config
        if di_value < c.di_fear:
            risky_pct = c.alloc_extreme_fear if di_value < c.di_extreme_fear else c.alloc_fear
        elif di_value < c.di_neutral_high:
            risky_pct = c.alloc_neutral
        elif di_value < c.di_extreme_greed:
            risky_pct = c.alloc_greed
        else:
            risky_pct = c.alloc_extreme_greed

        # Répartir entre assets
        assets = price_data.columns.tolist()
        weights = pd.Series(0.0, index=assets)

        # Identifier stablecoins vs risky
        stable_assets = [a for a in assets if a.upper() in ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'STABLES', 'STABLECOINS']]
        risky_assets = [a for a in assets if a not in stable_assets]

        if not risky_assets:
            risky_assets = assets
            stable_assets = []

        # Répartir
        stable_pct = 1.0 - risky_pct

        if risky_assets:
            weight_per_risky = risky_pct / len(risky_assets)
            for asset in risky_assets:
                weights[asset] = weight_per_risky

        if stable_assets:
            weight_per_stable = stable_pct / len(stable_assets)
            for asset in stable_assets:
                weights[asset] = weight_per_stable

        return weights


class DIMomentumStrategy(PortfolioStrategy):
    """
    Stratégie S2: Momentum DI

    Règles:
    - DI en hausse >5pts sur 7j → augmenter exposition
    - DI en baisse >5pts sur 7j → réduire exposition

    Suit la tendance du DI.
    """

    def __init__(self, config: Optional[DIStrategyConfig] = None):
        super().__init__("DI Momentum")
        self.config = config or DIStrategyConfig()
        self.di_series: Optional[pd.Series] = None
        self.base_strategy = DIThresholdStrategy(config)

    def set_di_series(self, di_series: pd.Series):
        self.di_series = di_series
        self.base_strategy.set_di_series(di_series)

    def get_weights(
        self,
        date: pd.Timestamp,
        price_data: pd.DataFrame,
        current_weights: pd.Series,
        **kwargs
    ) -> pd.Series:
        if self.di_series is None:
            return self.base_strategy.get_weights(date, price_data, current_weights, **kwargs)

        # Calculer momentum DI
        try:
            idx = self.di_series.index.get_indexer([date], method='nearest')[0]
            current_di = self.di_series.iloc[idx]

            lookback = self.config.momentum_lookback
            if idx >= lookback:
                past_di = self.di_series.iloc[idx - lookback]
                di_change = current_di - past_di
            else:
                di_change = 0
        except (KeyError, IndexError):
            current_di = 50.0
            di_change = 0

        # Ajuster l'allocation de base selon momentum
        base_weights = self.base_strategy.get_weights(date, price_data, current_weights, di_value=current_di)

        threshold = self.config.momentum_threshold
        if abs(di_change) > threshold:
            # Momentum significatif
            adjustment = 0.1 if di_change > 0 else -0.1

            stable_assets = [a for a in base_weights.index if a.upper() in ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'STABLES']]
            risky_assets = [a for a in base_weights.index if a not in stable_assets]

            for asset in risky_assets:
                base_weights[asset] = min(1.0, max(0.0, base_weights[asset] + adjustment / len(risky_assets)))
            for asset in stable_assets:
                base_weights[asset] = min(1.0, max(0.0, base_weights[asset] - adjustment / len(stable_assets)))

            # Renormaliser
            total = base_weights.sum()
            if total > 0:
                base_weights = base_weights / total

        return base_weights


class DIContrarianStrategy(PortfolioStrategy):
    """
    Stratégie S3: Contrarian DI

    Règles:
    - DI < 20 (extreme fear) → accumulation aggressive (85% risky)
    - DI > 80 (extreme greed) → prise de profits (30% risky)

    "Be greedy when others are fearful"
    """

    def __init__(self, config: Optional[DIStrategyConfig] = None):
        super().__init__("DI Contrarian")
        self.config = config or DIStrategyConfig()
        self.di_series: Optional[pd.Series] = None

    def set_di_series(self, di_series: pd.Series):
        self.di_series = di_series

    def get_weights(
        self,
        date: pd.Timestamp,
        price_data: pd.DataFrame,
        current_weights: pd.Series,
        **kwargs
    ) -> pd.Series:
        di_value = kwargs.get('di_value')
        if di_value is None and self.di_series is not None:
            try:
                di_value = self.di_series.loc[date]
            except KeyError:
                idx = self.di_series.index.get_indexer([date], method='nearest')[0]
                di_value = self.di_series.iloc[idx]

        if di_value is None:
            di_value = 50.0

        c = self.config

        # Contrarian: inversé par rapport au sentiment
        if di_value < c.di_extreme_fear:
            risky_pct = 0.85  # Maximum risky en extreme fear
        elif di_value < c.di_fear:
            risky_pct = 0.70
        elif di_value > c.di_extreme_greed:
            risky_pct = 0.30  # Minimum risky en extreme greed
        elif di_value > c.di_greed:
            risky_pct = 0.45
        else:
            risky_pct = 0.60  # Neutre

        # Répartir
        assets = price_data.columns.tolist()
        weights = pd.Series(0.0, index=assets)

        stable_assets = [a for a in assets if a.upper() in ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'STABLES', 'STABLECOINS']]
        risky_assets = [a for a in assets if a not in stable_assets]

        if not risky_assets:
            risky_assets = assets
            stable_assets = []

        if risky_assets:
            for asset in risky_assets:
                weights[asset] = risky_pct / len(risky_assets)

        if stable_assets:
            for asset in stable_assets:
                weights[asset] = (1.0 - risky_pct) / len(stable_assets)

        return weights


class DIRiskParityStrategy(PortfolioStrategy):
    """
    Stratégie S4: Risk Parity + DI Scaling

    Combine Risk Parity classique avec ajustement DI.
    Allocation risk parity × (DI / 50) pour scaling dynamique.
    """

    def __init__(self, vol_lookback: int = 30, config: Optional[DIStrategyConfig] = None):
        super().__init__(f"DI Risk Parity ({vol_lookback}d)")
        self.vol_lookback = vol_lookback
        self.config = config or DIStrategyConfig()
        self.di_series: Optional[pd.Series] = None

    def set_di_series(self, di_series: pd.Series):
        self.di_series = di_series

    def get_weights(
        self,
        date: pd.Timestamp,
        price_data: pd.DataFrame,
        current_weights: pd.Series,
        **kwargs
    ) -> pd.Series:
        # Calculer volatilités
        try:
            end_idx = price_data.index.get_indexer([date], method='nearest')[0]
        except (KeyError, IndexError):
            return pd.Series(1.0 / len(price_data.columns), index=price_data.columns)

        start_idx = max(0, end_idx - self.vol_lookback)
        if start_idx >= end_idx:
            return pd.Series(1.0 / len(price_data.columns), index=price_data.columns)

        returns_data = price_data.iloc[start_idx:end_idx+1].pct_change().dropna()
        if len(returns_data) == 0:
            return pd.Series(1.0 / len(price_data.columns), index=price_data.columns)

        volatilities = returns_data.std()

        # Inverse volatility weighting (Risk Parity)
        inv_vol = 1.0 / volatilities.replace(0, 1e-10)
        base_weights = inv_vol / inv_vol.sum()

        # Appliquer scaling DI
        di_value = kwargs.get('di_value')
        if di_value is None and self.di_series is not None:
            try:
                di_value = self.di_series.loc[date]
            except KeyError:
                idx = self.di_series.index.get_indexer([date], method='nearest')[0]
                di_value = self.di_series.iloc[idx]

        if di_value is not None:
            # Scale factor: DI=0 → 0.5, DI=50 → 1.0, DI=100 → 1.5
            scale_factor = 0.5 + (di_value / 100.0)

            # Identifier risky vs stable
            stable_assets = [a for a in base_weights.index if a.upper() in ['USDT', 'USDC', 'DAI', 'BUSD']]
            risky_assets = [a for a in base_weights.index if a not in stable_assets]

            # Ajuster
            for asset in risky_assets:
                base_weights[asset] *= scale_factor
            for asset in stable_assets:
                base_weights[asset] *= (2.0 - scale_factor)

            # Renormaliser
            base_weights = base_weights / base_weights.sum()

        return base_weights


class DISignalStrategy(PortfolioStrategy):
    """
    Stratégie S5: DI Signal Only

    Trading signaux purs:
    - BUY: DI croise 40 à la hausse (confirmation 3 jours)
    - SELL: DI croise 60 à la baisse
    - Holding minimum: 14 jours
    """

    def __init__(self, config: Optional[DIStrategyConfig] = None):
        super().__init__("DI Signal")
        self.config = config or DIStrategyConfig()
        self.di_series: Optional[pd.Series] = None
        self._last_trade_date: Optional[pd.Timestamp] = None
        self._current_position: str = "neutral"  # "long", "short", "neutral"
        self._confirmation_count: int = 0

    def set_di_series(self, di_series: pd.Series):
        self.di_series = di_series

    def reset_state(self):
        """Reset pour nouveau backtest"""
        self._last_trade_date = None
        self._current_position = "neutral"
        self._confirmation_count = 0

    def get_weights(
        self,
        date: pd.Timestamp,
        price_data: pd.DataFrame,
        current_weights: pd.Series,
        **kwargs
    ) -> pd.Series:
        if self.di_series is None:
            # Neutre si pas de DI
            return pd.Series(1.0 / len(price_data.columns), index=price_data.columns)

        c = self.config
        assets = price_data.columns.tolist()

        try:
            idx = self.di_series.index.get_indexer([date], method='nearest')[0]
            current_di = self.di_series.iloc[idx]
            prev_di = self.di_series.iloc[idx - 1] if idx > 0 else current_di
        except (KeyError, IndexError):
            current_di = 50.0
            prev_di = 50.0

        # Vérifier holding minimum
        min_holding_met = True
        if self._last_trade_date is not None:
            days_held = (date - self._last_trade_date).days
            min_holding_met = days_held >= c.signal_min_holding_days

        # Détecter signaux
        crossed_above_entry = prev_di < c.signal_entry_threshold <= current_di
        crossed_below_exit = prev_di > c.signal_exit_threshold >= current_di

        # Confirmation
        if crossed_above_entry:
            self._confirmation_count += 1
        elif crossed_below_exit:
            self._confirmation_count = 0

        # Générer signal
        if self._confirmation_count >= c.signal_confirmation_days and self._current_position != "long":
            self._current_position = "long"
            self._last_trade_date = date
            self._confirmation_count = 0
        elif crossed_below_exit and min_holding_met and self._current_position == "long":
            self._current_position = "neutral"
            self._last_trade_date = date

        # Allouer selon position
        weights = pd.Series(0.0, index=assets)
        stable_assets = [a for a in assets if a.upper() in ['USDT', 'USDC', 'DAI', 'BUSD', 'STABLES']]
        risky_assets = [a for a in assets if a not in stable_assets]

        if not risky_assets:
            risky_assets = assets
            stable_assets = []

        if self._current_position == "long":
            # Full risky
            for asset in risky_assets:
                weights[asset] = 1.0 / len(risky_assets)
        else:
            # Neutral = 50/50
            for asset in risky_assets:
                weights[asset] = 0.5 / len(risky_assets) if risky_assets else 0
            for asset in stable_assets:
                weights[asset] = 0.5 / len(stable_assets) if stable_assets else 0

        return weights


# Dictionnaire des stratégies disponibles
DI_STRATEGIES = {
    "di_threshold": DIThresholdStrategy,
    "di_momentum": DIMomentumStrategy,
    "di_contrarian": DIContrarianStrategy,
    "di_risk_parity": DIRiskParityStrategy,
    "di_signal": DISignalStrategy,
}


def get_di_strategy(strategy_name: str, config: Optional[DIStrategyConfig] = None) -> PortfolioStrategy:
    """Factory pour créer une stratégie DI"""
    if strategy_name not in DI_STRATEGIES:
        raise ValueError(f"Stratégie inconnue: {strategy_name}. Disponibles: {list(DI_STRATEGIES.keys())}")

    strategy_class = DI_STRATEGIES[strategy_name]

    if strategy_name == "di_risk_parity":
        return strategy_class(vol_lookback=30, config=config)
    else:
        return strategy_class(config=config)

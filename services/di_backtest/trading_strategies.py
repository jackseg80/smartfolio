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


class DISmartfolioReplicaStrategy(PortfolioStrategy):
    """
    Stratégie S6: SmartFolio Replica

    Réplique la logique réelle de l'Allocation Engine V2 de SmartFolio:
    - Allocation stables basée sur le cycle score (pas le DI directement)
    - Phase bullish (cycle ≥90): 15% stables, 85% risky
    - Phase moderate (cycle 70-90): 20% stables, 80% risky
    - Phase bearish (cycle <70): 30% stables, 70% risky

    Utilise le cycle_score composant du DI pour déterminer la phase de marché.
    C'est la stratégie la plus proche du comportement réel du projet.
    """

    def __init__(self, config: Optional[DIStrategyConfig] = None):
        super().__init__("SmartFolio Replica")
        self.config = config or DIStrategyConfig()
        self.di_series: Optional[pd.Series] = None
        self.cycle_series: Optional[pd.Series] = None

    def set_di_series(self, di_series: pd.Series):
        """Injecte la série DI historique"""
        # Normaliser l'index pour faciliter les lookups
        normalized_series = di_series.copy()
        if isinstance(normalized_series.index, pd.DatetimeIndex):
            normalized_series.index = normalized_series.index.normalize()
        self.di_series = normalized_series

    def set_cycle_series(self, cycle_series: pd.Series):
        """Injecte la série cycle score historique"""
        import logging
        logger = logging.getLogger(__name__)

        # Normaliser l'index pour faciliter les lookups (minuit, sans timezone)
        normalized_series = cycle_series.copy()
        if isinstance(normalized_series.index, pd.DatetimeIndex):
            normalized_series.index = normalized_series.index.normalize()

        self.cycle_series = normalized_series
        self._log_count = 0  # Reset log counter

        # Afficher quelques exemples de l'index pour debug
        sample_dates = normalized_series.index[:3].tolist() if len(normalized_series) > 3 else normalized_series.index.tolist()

        logger.info(
            f"SmartFolioReplica: cycle_series set, len={len(normalized_series)}, "
            f"index_type={type(normalized_series.index).__name__}, "
            f"min={normalized_series.min():.1f}, max={normalized_series.max():.1f}, "
            f"mean={normalized_series.mean():.1f}, sample_dates={sample_dates}"
        )

    def get_weights(
        self,
        date: pd.Timestamp,
        price_data: pd.DataFrame,
        current_weights: pd.Series,
        **kwargs
    ) -> pd.Series:
        """
        Calcule les poids selon la logique de l'Allocation Engine V2

        L'allocation dépend du cycle score pour déterminer la phase:
        - Phase bullish (cycle ≥90): 85% risky
        - Phase moderate (70≤cycle<90): 80% risky
        - Phase bearish (cycle <70): 70% risky

        Les ratios BTC/ETH/Alts dans la partie risky:
        - Bullish: BTC 21%, ETH 17%, Alts 47% (du total)
        - Moderate: BTC 24%, ETH 18%, Alts 38%
        - Bearish: BTC 25%, ETH 18%, Alts 28%
        """
        import logging
        logger = logging.getLogger(__name__)

        # Récupérer cycle score si disponible, sinon estimer depuis DI
        cycle_score = kwargs.get('cycle_score')
        lookup_method = "kwargs"

        # Normaliser la date pour le lookup (l'index des séries a été normalisé)
        date_normalized = pd.Timestamp(date).normalize()

        if cycle_score is None and self.cycle_series is not None:
            try:
                # Lookup avec date normalisée
                cycle_score = self.cycle_series.loc[date_normalized]
                lookup_method = "normalized"
            except KeyError:
                # Fallback vers nearest
                idx = self.cycle_series.index.get_indexer([date_normalized], method='nearest')[0]
                if idx >= 0:
                    cycle_score = self.cycle_series.iloc[idx]
                    lookup_method = f"nearest[{idx}]"
                else:
                    cycle_score = None
                    lookup_method = "failed"

            # Log pour debug (premiers appels seulement)
            if not hasattr(self, '_log_count'):
                self._log_count = 0
            if self._log_count < 5:
                cycle_str = f"{cycle_score:.1f}" if cycle_score is not None else "None"
                logger.info(
                    f"SmartFolioReplica: date={date}, norm={date_normalized}, cycle_score={cycle_str}, "
                    f"method={lookup_method}, series_len={len(self.cycle_series)}"
                )
                self._log_count += 1

        # Fallback: estimer cycle depuis DI (approximation)
        if cycle_score is None:
            di_value = kwargs.get('di_value')
            if di_value is None and self.di_series is not None:
                try:
                    # Lookup avec date normalisée (l'index a été normalisé dans set_di_series)
                    di_value = self.di_series.loc[date_normalized]
                    lookup_method = "di_normalized"
                except KeyError:
                    idx = self.di_series.index.get_indexer([date_normalized], method='nearest')[0]
                    if idx >= 0:
                        di_value = self.di_series.iloc[idx]
                        lookup_method = f"di_nearest[{idx}]"
                    else:
                        lookup_method = "di_failed"

            if di_value is not None:
                # Approximation: DI élevé souvent corrélé à cycle élevé
                # Mais le DI peut être bas même en bullish (macro penalty)
                # On utilise une estimation conservatrice
                cycle_score = di_value * 1.1  # Léger boost car DI inclut penalties
                cycle_score = min(100, max(0, cycle_score))

                # Log fallback
                if self._log_count < 10:
                    logger.warning(
                        f"SmartFolioReplica FALLBACK: date={date}, di_value={di_value:.1f}, "
                        f"estimated cycle={cycle_score:.1f}, method={lookup_method}"
                    )
                    self._log_count += 1
            else:
                cycle_score = 50.0  # Neutre par défaut
                if self._log_count < 10:
                    logger.error(f"SmartFolioReplica: No cycle or DI data for {date}, using default 50")
                    self._log_count += 1

        # Déterminer phase et allocation stables selon Allocation Engine V2
        if cycle_score >= 90:
            # Phase bullish
            stables_pct = 0.15
            # Ratios non-stables: BTC 25%, ETH 20%, Alts 55% (renormalisés)
            btc_ratio = 0.25
            eth_ratio = 0.20
            alts_ratio = 0.55
        elif cycle_score >= 70:
            # Phase moderate
            stables_pct = 0.20
            btc_ratio = 0.30
            eth_ratio = 0.22
            alts_ratio = 0.48
        else:
            # Phase bearish
            stables_pct = 0.30
            btc_ratio = 0.35
            eth_ratio = 0.25
            alts_ratio = 0.40

        # L'espace pour risky après stables
        non_stables_space = 1.0 - stables_pct

        # Renormaliser les ratios
        base_total = btc_ratio + eth_ratio + alts_ratio
        btc_target = (btc_ratio / base_total) * non_stables_space
        eth_target = (eth_ratio / base_total) * non_stables_space

        # Appliquer floors (comme dans l'engine réel)
        btc_floor = 0.15  # 15% minimum BTC
        eth_floor = 0.12  # 12% minimum ETH
        stables_floor = 0.10  # 10% minimum stables

        btc_target = max(btc_target, btc_floor)
        eth_target = max(eth_target, eth_floor)
        final_stables = max(stables_pct, stables_floor)

        # Calculer alts = reste
        alts_target = 1.0 - btc_target - eth_target - final_stables

        # Normaliser si dépassement
        total = btc_target + eth_target + final_stables + alts_target
        if total > 1.0:
            scale = 1.0 / total
            btc_target *= scale
            eth_target *= scale
            final_stables *= scale
            alts_target *= scale

        # Répartir entre assets du backtest (simplifié: BTC = risky, reste = stables)
        assets = price_data.columns.tolist()
        weights = pd.Series(0.0, index=assets)

        stable_assets = [a for a in assets if a.upper() in ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'STABLES', 'STABLECOINS']]
        risky_assets = [a for a in assets if a not in stable_assets]

        if not risky_assets:
            risky_assets = assets
            stable_assets = []

        # Dans le backtest simplifié, on a BTC + STABLES
        # On considère que risky = BTC représente tout le non-stable
        risky_pct = btc_target + eth_target + alts_target  # = 1 - stables

        if risky_assets:
            weight_per_risky = risky_pct / len(risky_assets)
            for asset in risky_assets:
                weights[asset] = weight_per_risky

        if stable_assets:
            weight_per_stable = final_stables / len(stable_assets)
            for asset in stable_assets:
                weights[asset] = weight_per_stable

        # Log les 3 premiers appels pour voir les weights
        if self._log_count < 15 and self._log_count >= 5:
            phase = "bullish" if cycle_score >= 90 else ("moderate" if cycle_score >= 70 else "bearish")
            logger.info(
                f"SmartFolioReplica WEIGHTS: cycle={cycle_score:.1f}, phase={phase}, "
                f"risky={risky_pct*100:.1f}%, stables={final_stables*100:.1f}%, "
                f"assets={list(weights.items())}"
            )
            self._log_count += 1

        return weights


# Dictionnaire des stratégies disponibles
DI_STRATEGIES = {
    "di_threshold": DIThresholdStrategy,
    "di_momentum": DIMomentumStrategy,
    "di_contrarian": DIContrarianStrategy,
    "di_risk_parity": DIRiskParityStrategy,
    "di_signal": DISignalStrategy,
    "di_smartfolio_replica": DISmartfolioReplicaStrategy,
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

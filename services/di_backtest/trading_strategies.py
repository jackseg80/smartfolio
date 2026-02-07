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


@dataclass
class ReplicaParams:
    """Configurable parameters for SmartFolio Replica 4-layer pipeline.

    All defaults match the current hardcoded production values,
    ensuring backward compatibility when no params are provided.
    """
    # Layer toggles
    enable_risk_budget: bool = True
    enable_market_overrides: bool = True
    enable_exposure_cap: bool = True
    enable_governance_penalty: bool = True

    # Layer 1: Risk Budget bounds
    risk_budget_min: float = 0.20   # Floor for risky allocation (production: 20%)
    risk_budget_max: float = 0.85   # Ceiling for risky allocation (production: 85%)

    # Layer 3: Exposure Cap
    exposure_confidence: float = 0.65  # Signal quality confidence (production: 0.65)

    # Layer 4: Governance Penalty
    max_governance_penalty: float = 0.25  # Maximum penalty (production: 25%)


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

    Réplique la logique réelle de l'Allocation Engine V2 de SmartFolio
    en utilisant la formule risk_budget de production:

        blendedScore = 0.5×cycle + 0.3×onchain + 0.2×risk
        risk_factor  = 0.5 + 0.5 × (riskScore / 100)
        baseRisky    = clamp((blendedScore - 35) / 45, 0, 1)
        risky        = clamp(baseRisky × risk_factor, 0.20, 0.85)
        stables      = 1 - risky

    Source: static/modules/market-regimes.js → calculateRiskBudget()
    """

    def __init__(self, config: Optional[DIStrategyConfig] = None,
                 replica_params: Optional['ReplicaParams'] = None):
        super().__init__("SmartFolio Replica")
        self.config = config or DIStrategyConfig()
        self.replica_params = replica_params or ReplicaParams()
        self.di_series: Optional[pd.Series] = None
        self.cycle_series: Optional[pd.Series] = None
        self._log_count = 0

    def set_di_series(self, di_series: pd.Series):
        """Injecte la série DI historique"""
        normalized_series = di_series.copy()
        if isinstance(normalized_series.index, pd.DatetimeIndex):
            normalized_series.index = normalized_series.index.normalize()
        self.di_series = normalized_series

    def set_cycle_series(self, cycle_series: pd.Series):
        """Injecte la série cycle score historique"""
        import logging
        logger = logging.getLogger(__name__)

        normalized_series = cycle_series.copy()
        if isinstance(normalized_series.index, pd.DatetimeIndex):
            normalized_series.index = normalized_series.index.normalize()

        self.cycle_series = normalized_series
        self._log_count = 0

        logger.info(
            f"SmartFolioReplica: cycle_series set, len={len(normalized_series)}, "
            f"min={normalized_series.min():.1f}, max={normalized_series.max():.1f}, "
            f"mean={normalized_series.mean():.1f}"
        )

    @staticmethod
    def _compute_adaptive_weights(
        cycle_score: float,
        onchain_score: float,
        risk_score: float
    ) -> tuple:
        """
        Contradiction-based adaptive weights.
        Source: static/governance/contradiction-policy.js → calculateAdaptiveWeights()

        When scores contradict each other (e.g., high cycle but low onchain),
        reduce cycle weight and increase risk weight.

        Returns:
            (cycle_w, onchain_w, risk_w) normalized to sum=1.0
        """
        # Detect contradiction level (0-1)
        # Primary signal: divergence between cycle and onchain
        divergence = abs(cycle_score - onchain_score)
        # Normalize: 0 at 0 divergence, 1.0 at 40+ divergence
        contradiction = min(1.0, max(0.0, (divergence - 10) / 30))

        # Base weights (standard blended formula)
        base_cycle = 0.50
        base_onchain = 0.30
        base_risk = 0.20

        # Adjustment coefficients (from contradiction-policy.js)
        cycle_reduction = 0.35    # up to -35%
        onchain_reduction = 0.15  # up to -15%
        risk_increase = 0.50      # up to +50%

        # Apply adjustments
        adj_cycle = base_cycle * (1 - cycle_reduction * contradiction)
        adj_onchain = base_onchain * (1 - onchain_reduction * contradiction)
        adj_risk = base_risk * (1 + risk_increase * contradiction)

        # Floor/ceil
        floor, ceil = 0.12, 0.65
        adj_cycle = max(floor, min(ceil, adj_cycle))
        adj_onchain = max(floor, min(ceil, adj_onchain))
        adj_risk = max(floor, min(ceil, adj_risk))

        # Normalize to sum=1
        total = adj_cycle + adj_onchain + adj_risk
        return adj_cycle / total, adj_onchain / total, adj_risk / total

    @staticmethod
    def _compute_risk_budget(
        cycle_score: float,
        onchain_score: float,
        risk_score: float,
        params: Optional['ReplicaParams'] = None
    ) -> float:
        """
        Production risk_budget formula with market overrides.

        Uses standard blended weights (0.5/0.3/0.2) matching production
        calculateRiskBudget() in market-regimes.js.

        Steps:
        1. Compute blendedScore with standard weights
        2. Apply risk_budget formula (v2_conservative)
        3. Apply overrides: on-chain divergence, low risk score

        Note: Production also applies computeExposureCap() + governance cap_daily
        which further limit the allocation. The backtest omits these layers
        to show the pure risk_budget signal.

        Source: market-regimes.js → calculateRiskBudget() + applyMarketOverrides()

        Returns:
            risky_allocation in [risk_budget_min, risk_budget_max]
        """
        if params is None:
            params = ReplicaParams()

        # Step 1: Standard blended score (same as production calculateRiskBudget)
        blended = 0.5 * cycle_score + 0.3 * onchain_score + 0.2 * risk_score

        # Step 2: Risk budget formula (v2_conservative)
        risk_factor = 0.5 + 0.5 * (risk_score / 100.0)
        base_risky = max(0.0, min(1.0, (blended - 35) / 45))
        risky = max(params.risk_budget_min, min(params.risk_budget_max, base_risky * risk_factor))

        if not params.enable_market_overrides:
            return risky

        # Step 3: Market overrides (applied to stables target)
        stables = 1.0 - risky

        # Override 1: On-Chain Divergence (market-regimes.js L174-188)
        # |blended - onchain| >= 30 → +10% stables
        divergence = abs(blended - onchain_score)
        if divergence >= 30:
            stables = min(1.0 - params.risk_budget_min, stables + 0.10)

        # Override 2: Low Risk Score (market-regimes.js L190-201)
        # risk <= 30 → force stables >= 50%
        if risk_score <= 30:
            stables = max(0.50, stables)

        # Clamp final risky in [min, max]
        risky = max(params.risk_budget_min, min(params.risk_budget_max, 1.0 - stables))

        return risky

    @staticmethod
    def _compute_contradiction_index(
        btc_volatility: float,
        cycle_score: float,
        onchain_score: float,
        di_value: float = 50.0,
    ) -> float:
        """
        Reconstruct contradiction_index from historical data.

        Adapts production SignalExtractor.compute_contradiction_index()
        (signals.py L140-187) using available backtest data:

        Check 1: High BTC vol + Bullish cycle → contradiction (0.3)
                 Production: vol > 0.15 annualized + regime_bull > 0.6
                 Backtest:   vol > 0.50 annualized + cycle >= 70

        Check 2: DI vs Cycle divergence → contradiction (0.25)
                 Production: extreme_fear+bull OR extreme_greed+not_bull
                 Backtest:   low DI (<30) + bullish cycle, or high DI (>75) + bearish

        Check 3: Score divergence → contradiction (0.2)
                 Production: avg_correlation > 0.7
                 Backtest:   |cycle - onchain| >= 40 (internal signal conflict)

        Returns:
            contradiction_index in [0.0, 1.0]
        """
        contradictions = 0.0
        total_checks = 0.0

        # Check 1: High volatility + Bullish cycle = suspect
        # Production uses vol > 0.15 but that's per-asset daily vol;
        # backtest btc_volatility is annualized, so threshold ~0.50
        vol_high = btc_volatility > 0.50
        regime_bull = cycle_score >= 70

        if vol_high and regime_bull:
            contradictions += 0.3
        total_checks += 1.0

        # Check 2: DI vs Cycle contradiction (proxy for sentiment vs regime)
        # Low DI + bullish cycle = fear despite bull signals
        # High DI + bearish cycle = greed despite bear signals
        di_extreme_fear = di_value < 30
        di_extreme_greed = di_value > 75

        if (di_extreme_fear and regime_bull) or (di_extreme_greed and not regime_bull):
            contradictions += 0.25
        total_checks += 1.0

        # Check 3: Internal score divergence (proxy for correlation/systemic risk)
        # Large cycle-onchain gap = different signals disagree
        score_divergence = abs(cycle_score - onchain_score)
        if score_divergence >= 40:
            contradictions += 0.2
        total_checks += 1.0

        contradiction_index = min(1.0, contradictions / max(1.0, total_checks))
        return contradiction_index

    @staticmethod
    def _compute_governance_penalty(
        contradiction_index: float,
        btc_volatility: float = 0.0,
        params: Optional['ReplicaParams'] = None,
    ) -> float:
        """
        Layer 4: Governance-inspired penalty on risky allocation.

        Instead of applying cap_daily (3-12% trading cap) as hard allocation
        limit (which would be absurdly conservative), applies a proportional
        penalty inspired by the governance logic in policy_engine.py:

        - contradiction > 0.5  → heavy penalty (-15 to -25% risky)
        - contradiction 0.3-0.5 → moderate penalty (-5 to -15%)
        - contradiction < 0.3  → no penalty
        - High volatility amplifies the penalty

        Returns:
            penalty as fraction to SUBTRACT from risky allocation [0.0, max_governance_penalty]
        """
        if params is None:
            params = ReplicaParams()

        if contradiction_index < 0.20:
            # Low contradiction → no governance intervention
            return 0.0

        # Base penalty: linear scale from contradiction
        # 0.20 → 0%, 0.50 → 15%, 0.75 → 25%
        base_penalty = max(0.0, (contradiction_index - 0.20) * 0.45)

        # Volatility amplifier: high vol increases penalty
        # vol > 0.50 → +5%, vol > 0.80 → +10%
        vol_amplifier = max(0.0, min(0.10, (btc_volatility - 0.40) * 0.25))

        total_penalty = min(params.max_governance_penalty, base_penalty + vol_amplifier)
        return total_penalty

    @staticmethod
    def _compute_exposure_cap(
        blended_score: float,
        risk_score: float,
        di_value: float = 50.0,
        btc_volatility: float = 0.0,
        params: Optional['ReplicaParams'] = None,
    ) -> float:
        """
        Simplified exposure cap matching production computeExposureCap().

        Source: targets-coordinator.js L349-422

        Backtest adaptations:
        - decision_score → di_value / 100 (proxy)
        - confidence → params.exposure_confidence (default 0.65)
        - backendStatus → always 'ok' (no degradation in backtest)

        Returns:
            exposure cap as fraction [0.20, 0.95]
        """
        if params is None:
            params = ReplicaParams()

        bs = round(blended_score)
        rs = round(risk_score)

        # Proxy for signal quality
        ds = min(1.0, max(0.0, di_value / 100.0))
        dc = params.exposure_confidence
        raw = ds * dc

        # 1) Base cap from blended + risk grid
        if bs >= 70 and rs >= 80:
            base = 90
        elif bs >= 70 and rs >= 60:
            base = 85
        elif bs >= 65 and rs >= 70:
            base = 80
        elif bs >= 65:
            base = 75
        elif bs >= 55 and rs >= 60:
            base = 70
        elif bs >= 55:
            base = 65
        else:
            base = 55

        # 2) Signal quality penalty (max 10pts)
        # Reference threshold stays at 0.65 (production baseline)
        signal_penalty = max(0, round((0.65 - raw) * 15))
        base -= min(10, signal_penalty)

        # 3) Volatility penalty (max 10pts)
        vol_penalty = max(0, round((btc_volatility - 0.20) * 50))
        base -= min(10, vol_penalty)

        # 4) Backend status - skip in backtest (always 'ok')

        # 5) Regime floor and cap (from MARKET_REGIMES ranges)
        if bs <= 25:
            regime_min, regime_max = 20, 40    # Bear Market
        elif bs <= 50:
            regime_min, regime_max = 40, 70    # Correction
        elif bs <= 75:
            regime_min, regime_max = 60, 85    # Bull Market
        else:
            regime_min, regime_max = 75, 95    # Expansion

        # Dynamic boost: Expansion + high Risk Score
        if bs > 75 and rs >= 80:
            regime_min = 65

        # 6) Final bounds
        final_cap = max(regime_min, min(regime_max, round(base)))

        return final_cap / 100.0

    def get_weights(
        self,
        date: pd.Timestamp,
        price_data: pd.DataFrame,
        current_weights: pd.Series,
        **kwargs
    ) -> pd.Series:
        """
        Calcule les poids selon le pipeline production complet:
        1. risk_budget formula (blended → risk_factor → allocation)
        2. Market overrides (on-chain divergence, low risk)
        3. Exposure cap (regime + signal quality + volatility)
        4. Governance penalty (contradiction-based reduction)

        Si les composants ne sont pas disponibles, fallback sur les
        stables fixes par phase (15/20/30%).
        """
        import logging
        logger = logging.getLogger(__name__)

        # --- Récupérer les scores composants ---
        cycle_score = kwargs.get('cycle_score')
        onchain_score = kwargs.get('onchain_score')
        risk_score = kwargs.get('risk_score')
        di_value = kwargs.get('di_value', 50.0)

        # Fallback cycle via series si pas dans kwargs
        if cycle_score is None and self.cycle_series is not None:
            date_normalized = pd.Timestamp(date).normalize()
            try:
                cycle_score = self.cycle_series.loc[date_normalized]
            except KeyError:
                idx = self.cycle_series.index.get_indexer([date_normalized], method='nearest')[0]
                if idx >= 0:
                    cycle_score = self.cycle_series.iloc[idx]

        # Default si toujours None
        if cycle_score is None:
            cycle_score = 50.0
        if onchain_score is None:
            onchain_score = 50.0
        if risk_score is None:
            risk_score = 50.0

        # --- Calculer allocation via formule production ---
        has_real_components = (
            kwargs.get('onchain_score') is not None
            and kwargs.get('risk_score') is not None
        )

        if has_real_components:
            params = self.replica_params

            # Compute BTC rolling volatility from price data (needed by layers 3+4)
            risky_symbol = [a for a in price_data.columns
                           if a.upper() not in ['USDT', 'USDC', 'DAI', 'BUSD', 'STABLES', 'STABLECOINS']]
            btc_vol = 0.0
            if risky_symbol:
                returns = price_data[risky_symbol[0]].pct_change().dropna()
                if len(returns) >= 30:
                    btc_vol = float(returns.tail(30).std() * np.sqrt(365))

            blended = 0.5 * cycle_score + 0.3 * onchain_score + 0.2 * risk_score
            active_layers = []
            exposure_cap = 1.0    # default: no cap
            contradiction = 0.0
            gov_penalty = 0.0

            # Layer 1(+2): risk_budget + market overrides
            if params.enable_risk_budget:
                risky_pct = self._compute_risk_budget(
                    cycle_score, onchain_score, risk_score, params
                )
                active_layers.append("Risk Budget")
                if params.enable_market_overrides:
                    active_layers.append("Market Overrides")
            else:
                risky_pct = 0.60  # neutral fallback when L1 disabled

            # Layer 3: Exposure cap (regime + signal + volatility)
            if params.enable_exposure_cap:
                exposure_cap = self._compute_exposure_cap(
                    blended, risk_score, di_value, btc_vol, params
                )
                risky_pct = min(risky_pct, exposure_cap)
                active_layers.append("Exposure Cap")

            # Layer 4: Governance penalty (contradiction-based reduction)
            if params.enable_governance_penalty:
                contradiction = self._compute_contradiction_index(
                    btc_vol, cycle_score, onchain_score, di_value
                )
                gov_penalty = self._compute_governance_penalty(
                    contradiction, btc_vol, params
                )
                risky_pct = max(params.risk_budget_min, risky_pct - gov_penalty)
                active_layers.append("Governance Penalty")

            allocation_method = "+".join(active_layers) if active_layers else "none"
        else:
            # Fallback: stables fixes par phase (emergency path)
            if cycle_score >= 90:
                risky_pct = 0.85
            elif cycle_score >= 70:
                risky_pct = 0.80
            else:
                risky_pct = 0.70
            allocation_method = "phase_fallback"

        stables_pct = 1.0 - risky_pct

        # --- Répartir entre assets du backtest (2-asset: risky + stables) ---
        assets = price_data.columns.tolist()
        weights = pd.Series(0.0, index=assets)

        stable_assets = [a for a in assets if a.upper() in ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'STABLES', 'STABLECOINS']]
        risky_assets = [a for a in assets if a not in stable_assets]

        if not risky_assets:
            risky_assets = assets
            stable_assets = []

        if risky_assets:
            weight_per_risky = risky_pct / len(risky_assets)
            for asset in risky_assets:
                weights[asset] = weight_per_risky

        if stable_assets:
            weight_per_stable = stables_pct / len(stable_assets)
            for asset in stable_assets:
                weights[asset] = weight_per_stable

        # Log (premiers appels seulement)
        if self._log_count < 5:
            phase = "bullish" if cycle_score >= 90 else ("moderate" if cycle_score >= 70 else "bearish")
            if has_real_components:
                logger.info(
                    f"SmartFolioReplica: cycle={cycle_score:.1f}, onchain={onchain_score:.1f}, "
                    f"risk={risk_score:.1f}, di={di_value:.1f}, phase={phase}, "
                    f"method={allocation_method}, cap={exposure_cap*100:.0f}%, "
                    f"contradiction={contradiction:.2f}, gov_penalty={gov_penalty*100:.0f}%, "
                    f"risky={risky_pct*100:.1f}%, stables={stables_pct*100:.1f}%"
                )
            else:
                logger.info(
                    f"SmartFolioReplica: cycle={cycle_score:.1f}, phase={phase}, "
                    f"method={allocation_method}, "
                    f"risky={risky_pct*100:.1f}%, stables={stables_pct*100:.1f}%"
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

"""
DI Backtest Engine
Moteur de simulation pour valider le Decision Index historiquement

Simule un système de REBALANCEMENT (pas de trading) :
le portefeuille ajuste périodiquement ses poids risky/stables
en fonction des signaux DI.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .historical_di_calculator import DIHistoryPoint
from .trading_strategies import PortfolioStrategy, DIThresholdStrategy

logger = logging.getLogger(__name__)


@dataclass
class RebalanceEvent:
    """Représente un événement de rebalancement du portefeuille"""
    date: datetime
    risky_pct: float  # % allocation risky après rebalancement
    stable_pct: float  # % allocation stable après rebalancement
    di_value: float  # DI au moment du rebalancement
    portfolio_value: float  # Valeur portfolio après rebalancement
    allocation_change: float  # Variation absolue de l'allocation risky
    reason: str = ""


# Alias pour compatibilité
Trade = RebalanceEvent


@dataclass
class DIBacktestResult:
    """Résultats du backtest DI"""
    # Performance
    total_return: float
    annualized_return: float
    benchmark_return: float
    excess_return: float

    # Risque
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Rebalancement
    rebalance_events: List[RebalanceEvent]
    rebalance_count: int
    turnover_annual: float  # Turnover annualisé (sum |delta alloc| / years)
    avg_risky_allocation: float  # Allocation risky moyenne sur la période
    upside_capture: float  # Capture ratio haussier vs benchmark
    downside_capture: float  # Capture ratio baissier vs benchmark

    # Séries temporelles
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    drawdown_curve: pd.Series

    # DI-specific
    di_correlation: float  # Corrélation DI vs returns futurs
    strategy_name: str

    # Métadonnées
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float

    # Compatibilité (deprecated)
    @property
    def trades(self) -> List[RebalanceEvent]:
        return self.rebalance_events

    @property
    def num_trades(self) -> int:
        return self.rebalance_count

    @property
    def win_rate(self) -> float:
        # Retourner capture ratio comme proxy — win_rate n'a pas de sens
        # pour un système de rebalancement
        return self.upside_capture


class DIBacktestEngine:
    """
    Moteur de backtest pour le Decision Index

    Simule un portefeuille 2 actifs (risky/stable) rebalancé
    périodiquement selon les signaux DI.
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,  # 0.1% par rebalancement
        rebalance_threshold: float = 0.05,  # Rebalance si écart > 5%
        risk_free_rate: float = 0.02,  # 2% annuel
    ):
        self.transaction_cost = transaction_cost
        self.rebalance_threshold = rebalance_threshold
        self.risk_free_rate = risk_free_rate

    def run_backtest(
        self,
        di_history: List[DIHistoryPoint],
        strategy: PortfolioStrategy,
        initial_capital: float = 10000.0,
        benchmark_prices: Optional[Dict[datetime, float]] = None,
        risky_symbol: str = "BTC",
        stable_return: float = 0.0,  # Return annuel stablecoins (0%)
        rebalance_frequency: str = "daily",  # daily, weekly, monthly
    ) -> DIBacktestResult:
        """
        Exécute un backtest avec la stratégie DI donnée

        Args:
            di_history: Liste de DIHistoryPoint
            strategy: Stratégie de rebalancement
            initial_capital: Capital initial
            benchmark_prices: Prix du benchmark (dict date->price)
            risky_symbol: Symbole de l'actif risky
            stable_return: Return annuel des stablecoins
            rebalance_frequency: Fréquence de rebalancement (daily/weekly/monthly)

        Returns:
            DIBacktestResult avec toutes les métriques
        """
        if not di_history:
            raise ValueError("DI history is empty")

        # Trier par date
        di_history = sorted(di_history, key=lambda x: x.date)

        # Créer Series des données
        dates = [p.date for p in di_history]
        di_values = pd.Series([p.decision_index for p in di_history], index=dates)
        btc_prices = pd.Series([p.btc_price for p in di_history], index=dates)

        # Construire l'ensemble des jours de rebalancement autorisés
        rebalance_dates = self._build_rebalance_dates(dates, rebalance_frequency)

        # Injecter DI dans la stratégie si supporté
        if hasattr(strategy, 'set_di_series'):
            strategy.set_di_series(di_values)
        if hasattr(strategy, 'reset_state'):
            strategy.reset_state()

        # Initialisation
        portfolio_value = initial_capital
        risky_allocation = 0.5  # Start 50/50
        stable_allocation = 0.5

        equity_curve = []
        benchmark_curve = []
        rebalance_events = []
        daily_returns = []
        benchmark_daily_returns = []
        risky_allocations = []  # Pour calculer la moyenne

        # Prix de référence
        base_btc_price = btc_prices.iloc[0]
        prev_btc_price = base_btc_price
        prev_portfolio_value = initial_capital

        # Pour le benchmark
        benchmark_value = initial_capital

        for i, di_point in enumerate(di_history):
            date = di_point.date
            di_value = di_point.decision_index
            btc_price = di_point.btc_price or btc_prices.iloc[i]

            # Calculer returns journaliers
            if i > 0:
                btc_return = (btc_price - prev_btc_price) / prev_btc_price
                stable_daily_return = stable_return / 365

                # Portfolio return
                portfolio_return = (
                    risky_allocation * btc_return +
                    stable_allocation * stable_daily_return
                )

                portfolio_value = prev_portfolio_value * (1 + portfolio_return)
                benchmark_value = benchmark_value * (1 + btc_return)

                daily_returns.append(portfolio_return)
                benchmark_daily_returns.append(btc_return)

            risky_allocations.append(risky_allocation)

            # Ne vérifier le rebalancement que les jours autorisés
            is_rebalance_day = date in rebalance_dates or i == 0

            if is_rebalance_day:
                # Obtenir nouvelle allocation de la stratégie
                price_df = pd.DataFrame({
                    risky_symbol: btc_prices[:i+1],
                    'STABLES': [1.0] * (i + 1)
                })

                current_weights = pd.Series({
                    risky_symbol: risky_allocation,
                    'STABLES': stable_allocation
                })

                # Passer les scores composants pour la formule risk_budget
                target_weights = strategy.get_weights(
                    date=pd.Timestamp(date),
                    price_data=price_df,
                    current_weights=current_weights,
                    di_value=di_value,
                    cycle_score=di_point.cycle_score,
                    onchain_score=di_point.onchain_score,
                    risk_score=di_point.risk_score,
                    cycle_direction=di_point.cycle_direction,
                    cycle_confidence=di_point.cycle_confidence,
                )

                target_risky = target_weights.get(risky_symbol, 0.5)
                target_stable = 1.0 - target_risky

                # Vérifier si rebalance nécessaire
                allocation_diff = abs(target_risky - risky_allocation)

                if allocation_diff > self.rebalance_threshold:
                    # Appliquer coûts de transaction
                    trade_cost = allocation_diff * portfolio_value * self.transaction_cost
                    portfolio_value -= trade_cost

                    rebalance_events.append(RebalanceEvent(
                        date=date,
                        risky_pct=target_risky * 100,
                        stable_pct=target_stable * 100,
                        di_value=di_value,
                        portfolio_value=portfolio_value,
                        allocation_change=allocation_diff * 100,
                        reason=f"DI={di_value:.1f}, risky {risky_allocation*100:.0f}%→{target_risky*100:.0f}%"
                    ))

                    risky_allocation = target_risky
                    stable_allocation = target_stable

            equity_curve.append((date, portfolio_value))
            benchmark_curve.append((date, benchmark_value))

            prev_btc_price = btc_price
            prev_portfolio_value = portfolio_value

        # Construire les séries
        equity_series = pd.Series(
            [v for _, v in equity_curve],
            index=[d for d, _ in equity_curve]
        )
        benchmark_series = pd.Series(
            [v for _, v in benchmark_curve],
            index=[d for d, _ in benchmark_curve]
        )

        # Calculer métriques
        metrics = self._calculate_metrics(
            equity_series=equity_series,
            benchmark_series=benchmark_series,
            daily_returns=daily_returns,
            benchmark_daily_returns=benchmark_daily_returns,
            rebalance_events=rebalance_events,
            risky_allocations=risky_allocations,
            di_values=di_values,
            btc_prices=btc_prices
        )

        # Drawdown curve
        drawdown_series = self._calculate_drawdown_series(equity_series)

        return DIBacktestResult(
            total_return=metrics['total_return'],
            annualized_return=metrics['annualized_return'],
            benchmark_return=metrics['benchmark_return'],
            excess_return=metrics['excess_return'],
            max_drawdown=metrics['max_drawdown'],
            volatility=metrics['volatility'],
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            calmar_ratio=metrics['calmar_ratio'],
            rebalance_events=rebalance_events,
            rebalance_count=len(rebalance_events),
            turnover_annual=metrics['turnover_annual'],
            avg_risky_allocation=metrics['avg_risky_allocation'],
            upside_capture=metrics['upside_capture'],
            downside_capture=metrics['downside_capture'],
            equity_curve=equity_series,
            benchmark_curve=benchmark_series,
            drawdown_curve=drawdown_series,
            di_correlation=metrics['di_correlation'],
            strategy_name=strategy.name,
            start_date=di_history[0].date,
            end_date=di_history[-1].date,
            initial_capital=initial_capital,
            final_value=equity_series.iloc[-1]
        )

    def _calculate_metrics(
        self,
        equity_series: pd.Series,
        benchmark_series: pd.Series,
        daily_returns: List[float],
        benchmark_daily_returns: List[float],
        rebalance_events: List[RebalanceEvent],
        risky_allocations: List[float],
        di_values: pd.Series,
        btc_prices: pd.Series
    ) -> Dict[str, float]:
        """Calcule toutes les métriques de performance"""

        # Returns
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        benchmark_return = (benchmark_series.iloc[-1] / benchmark_series.iloc[0]) - 1
        excess_return = total_return - benchmark_return

        # Annualized
        days = (equity_series.index[-1] - equity_series.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility
        returns_array = np.array(daily_returns)
        volatility = np.std(returns_array) * np.sqrt(365) if len(returns_array) > 1 else 0

        # Max Drawdown
        max_drawdown = self._calculate_max_drawdown(equity_series)

        # Sharpe Ratio
        excess_daily = returns_array - (self.risk_free_rate / 365)
        sharpe_ratio = (np.mean(excess_daily) / np.std(excess_daily) * np.sqrt(365)
                       if np.std(excess_daily) > 0 else 0)

        # Sortino Ratio (downside deviation)
        downside_returns = returns_array[returns_array < 0]
        downside_std = np.std(downside_returns) * np.sqrt(365) if len(downside_returns) > 1 else 1
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0

        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # --- Rebalancing-specific metrics ---

        # Turnover annualisé: somme des |changements d'allocation| / années
        total_turnover = sum(e.allocation_change for e in rebalance_events) / 100.0
        turnover_annual = total_turnover / years if years > 0 else 0

        # Allocation risky moyenne
        avg_risky_allocation = np.mean(risky_allocations) if risky_allocations else 0.5

        # Capture ratios (upside/downside)
        upside_capture, downside_capture = self._calculate_capture_ratios(
            daily_returns, benchmark_daily_returns
        )

        # DI Correlation avec returns futurs (30j)
        try:
            future_returns = btc_prices.pct_change(30).shift(-30)
            di_correlation = di_values.corr(future_returns)
            if np.isnan(di_correlation):
                di_correlation = 0.0
        except Exception:
            di_correlation = 0.0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'benchmark_return': benchmark_return,
            'excess_return': excess_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'turnover_annual': turnover_annual,
            'avg_risky_allocation': avg_risky_allocation,
            'upside_capture': upside_capture,
            'downside_capture': downside_capture,
            'di_correlation': di_correlation,
        }

    @staticmethod
    def _build_rebalance_dates(dates: list, frequency: str) -> set:
        """
        Construit l'ensemble des dates autorisées pour le rebalancement.

        Args:
            dates: Liste triée de toutes les dates du backtest
            frequency: 'daily', 'weekly', or 'monthly'

        Returns:
            Set de dates où le rebalancement est autorisé
        """
        if frequency == "daily":
            return set(dates)

        rebalance_dates = set()

        if frequency == "weekly":
            # Rebalance chaque lundi (ou premier jour de la semaine disponible)
            current_week = None
            for d in dates:
                week_key = (d.isocalendar()[0], d.isocalendar()[1])
                if week_key != current_week:
                    rebalance_dates.add(d)
                    current_week = week_key

        elif frequency == "monthly":
            # Rebalance le premier jour de chaque mois
            current_month = None
            for d in dates:
                month_key = (d.year, d.month)
                if month_key != current_month:
                    rebalance_dates.add(d)
                    current_month = month_key

        else:
            # Fallback: daily
            return set(dates)

        return rebalance_dates

    @staticmethod
    def _calculate_capture_ratios(
        portfolio_returns: List[float],
        benchmark_returns: List[float]
    ) -> Tuple[float, float]:
        """
        Calcule les capture ratios upside/downside.

        Upside capture > 1.0  = portfolio capte plus que le benchmark en hausse
        Downside capture < 1.0 = portfolio perd moins que le benchmark en baisse
        Idéal: upside élevé + downside faible
        """
        if not portfolio_returns or not benchmark_returns:
            return 1.0, 1.0

        port = np.array(portfolio_returns)
        bench = np.array(benchmark_returns)

        # Upside: jours où benchmark > 0
        up_mask = bench > 0
        if up_mask.sum() > 0:
            upside = np.mean(port[up_mask]) / np.mean(bench[up_mask])
        else:
            upside = 1.0

        # Downside: jours où benchmark < 0
        down_mask = bench < 0
        if down_mask.sum() > 0:
            downside = np.mean(port[down_mask]) / np.mean(bench[down_mask])
        else:
            downside = 1.0

        return float(upside), float(downside)

    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calcule le max drawdown"""
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        return drawdown.min()

    def _calculate_drawdown_series(self, equity_series: pd.Series) -> pd.Series:
        """Calcule la série des drawdowns"""
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        return drawdown


# Instance singleton
di_backtest_engine = DIBacktestEngine()

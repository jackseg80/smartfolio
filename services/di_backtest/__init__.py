"""
Decision Index Backtest Module
Reconstruction historique et validation rétroactive du Decision Index

Ce module permet de:
- Reconstruire le Decision Index sur des périodes historiques (2017+)
- Simuler des stratégies de trading basées sur le DI
- Valider la performance du système sur différents régimes de marché

Utilise et étend les services existants:
- services/price_history.py pour les prix crypto
- services/macro_stress.py pour VIX/DXY via FRED
- services/backtesting_engine.py pour le framework de backtest
"""

from .data_sources import (
    HistoricalDataSources,
    historical_data_sources,
)
from .historical_di_calculator import (
    HistoricalDICalculator,
    historical_di_calculator,
    DIHistoryPoint,
    DIBacktestData,
)
from .di_backtest_engine import (
    DIBacktestEngine,
    DIBacktestResult,
    di_backtest_engine,
    Trade,
)
from .trading_strategies import (
    DIThresholdStrategy,
    DIMomentumStrategy,
    DIContrarianStrategy,
    DIRiskParityStrategy,
    DISignalStrategy,
    DI_STRATEGIES,
    get_di_strategy,
)

__all__ = [
    # Data sources
    "HistoricalDataSources",
    "historical_data_sources",
    # DI Calculator
    "HistoricalDICalculator",
    "historical_di_calculator",
    "DIHistoryPoint",
    "DIBacktestData",
    # Backtest Engine
    "DIBacktestEngine",
    "DIBacktestResult",
    "di_backtest_engine",
    "Trade",
    # Strategies
    "DIThresholdStrategy",
    "DIMomentumStrategy",
    "DIContrarianStrategy",
    "DIRiskParityStrategy",
    "DISignalStrategy",
    "DI_STRATEGIES",
    "get_di_strategy",
]

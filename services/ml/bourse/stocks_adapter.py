"""
Stocks ML Adapter - Reuses existing crypto ML infrastructure for stock market analytics.

This adapter wraps existing ML models (VolatilityPredictor, RegimeDetector, CorrelationForecaster)
and adapts them for traditional stock market analysis.

Key principle: REUSE, don't rebuild!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path

from services.ml.bourse.data_sources import StocksDataSource
from services.ml.bourse.training_scheduler import MLTrainingScheduler


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.

    This prevents Pydantic serialization errors with numpy.float32, numpy.int64, etc.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj
from services.ml.feature_engineering import CryptoFeatureEngineer
from services.ml.models.volatility_predictor import VolatilityPredictor
from services.ml.models.regime_detector import RegimeDetector
from services.ml.models.correlation_forecaster import CorrelationForecaster

logger = logging.getLogger(__name__)


class StocksMLAdapter:
    """
    Adapter to reuse crypto ML models for stock market analysis.

    Provides high-level interface for:
    - Volatility forecasting (1d, 7d, 30d)
    - Market regime detection (Bear Market/Correction/Bull Market/Expansion)
    - Multi-stock correlation forecasting
    - Technical signals aggregation
    """

    # Stock market regimes (must match RegimeDetector.regime_names)
    STOCK_REGIMES = {
        0: "Bear Market",        # Drawdown ≥20%, sustained decline
        1: "Correction",         # Drawdown 10-20%, high vol, price <MA200
        2: "Bull Market",        # Stable uptrend, price >MA200, low vol
        3: "Expansion"           # Strong recovery from major drawdown
    }

    def __init__(self, models_dir: str = "models/stocks"):
        """
        Initialize stocks ML adapter.

        Args:
            models_dir: Directory to store trained stock models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

        # Data source
        self.data_source = StocksDataSource()

        # Feature engineering (100% reusable from crypto)
        self.feature_engineer = CryptoFeatureEngineer()

        # ML models (reused from crypto infrastructure)
        self.volatility_predictor = VolatilityPredictor(
            model_dir=os.path.join(models_dir, "volatility_causal_v2"),
            trading_days=252, predict_uncertainty=False
        )
        self.regime_detector = RegimeDetector(
            model_dir=os.path.join(models_dir, "regime_causal_v2"), trading_days=252
        )
        self.correlation_forecaster = CorrelationForecaster(
            model_dir=os.path.join(models_dir, "correlation")
        )

        logger.info(f"StocksMLAdapter initialized with models_dir={models_dir}")

    async def predict_volatility(self, symbol: str, lookback_days: int = 365, confidence_level: float = .95) -> Dict[str, Any]:
        from services.ml.reliability import capability_service
        from api.schemas.ml_contract import ModelType, Horizon
        from services.ml.portfolio_context import stock_symbol
        exact_asset = stock_symbol(symbol)
        predictions = {}
        for h in (Horizon.D1, Horizon.D7, Horizon.D30):
            result = await capability_service.result(exact_asset, "stocks", ModelType.VOLATILITY, h)
            predictions[h.value] = {**result.model_dump(mode="json"), "predicted_volatility": result.value, "confidence_interval": None if result.uncertainty is None else result.uncertainty.model_dump()}
        return dict(symbol=symbol, timestamp=datetime.now().isoformat(), predictions=predictions,
            model_type="verified_daily_risk", lookback_days=lookback_days, confidence_level=None,
            note="No training during reads. Intervals require independent 90% coverage validation.")

    async def detect_market_regime(self, benchmark: str = "SPY", lookback_days: int = 7300, force_retrain: bool = False) -> Dict[str, Any]:
        from services.ml.reliability import capability_service
        from api.schemas.ml_contract import ModelType
        from services.ml.portfolio_context import stock_symbol
        result = await capability_service.result(stock_symbol(benchmark), "stocks", ModelType.REGIME)
        return dict(current_regime=result.value or "Unavailable", regime_id=next((i for i,n in self.STOCK_REGIMES.items() if n==result.value), None),
            confidence=None, regime_probabilities={}, benchmark=benchmark,
            timestamp=result.data_as_of.isoformat() if result.data_as_of else None,
            characteristics={"method": result.provenance.method or "unavailable"},
            model_type="descriptive_rules", note=result.reason, availability=result.availability.value)

    def _get_regime_characteristics(self, regime_name: str) -> Dict[str, str]:
        """Get characteristics for a given regime."""
        characteristics = {
            "Bear Market": {
                "trend": "downward",
                "volatility": "high",
                "sentiment": "fearful"
            },
            "Correction": {
                "trend": "sideways",
                "volatility": "elevated",
                "sentiment": "cautious"
            },
            "Bull Market": {
                "trend": "upward",
                "volatility": "moderate",
                "sentiment": "optimistic"
            },
            "Expansion": {
                "trend": "recovering",
                "volatility": "high",
                "sentiment": "hopeful"
            },
            # Backward compatibility aliases
            "Consolidation": {
                "trend": "sideways",
                "volatility": "low",
                "sentiment": "neutral"
            },
            "Distribution": {
                "trend": "topping",
                "volatility": "high",
                "sentiment": "cautious"
            }
        }
        return characteristics.get(regime_name, {"trend": "unknown", "volatility": "unknown", "sentiment": "unknown"})

    async def forecast_correlations(self, symbols: List[str], lookback_days: int = 365, horizons: List[int] = None) -> Dict[str, Any]:
        data = await self.data_source.get_multi_asset_data(symbols, lookback_days)
        returns = self.data_source.get_multi_asset_returns(data)
        valid = len(returns) >= 30 and len(returns.columns) >= 2
        return dict(symbols=list(data), predictions={}, timestamp=returns.index[-1].isoformat() if valid else None,
            horizons=horizons or [7,30], model_type="historical_descriptive",
            historical_correlation=returns.corr().to_dict() if valid else None,
            availability="Partial" if valid else "Unavailable",
            note="Historical correlations are descriptive; Transformer forecasts remain experimental. No training during reads.")

    async def generate_signals(
        self,
        symbol: str,
        lookback_days: int = 365
    ) -> Dict[str, Any]:
        """
        Generate aggregated ML signals for a stock.

        Combines:
        - Volatility forecast
        - Market regime
        - Technical indicators

        Returns:
            Dict with overall signal strength and components
        """
        try:
            # Fetch OHLCV data
            ohlcv_data = await self.data_source.get_ohlcv_data(
                symbol=symbol,
                lookback_days=lookback_days
            )

            # Generate technical features
            features_df = self.feature_engineer.create_feature_set(ohlcv_data, symbol=symbol)

            # Un indicateur absent ne devient pas une observation neutre.
            required = ['rsi_14', 'macd', 'macd_signal', 'bb_position']
            latest_features = features_df.iloc[-1]
            if not all(name in latest_features and np.isfinite(latest_features[name]) for name in required):
                raise ValueError('Required technical observations are unavailable')

            # Calculate signal components
            rsi_signal = self._rsi_to_signal(latest_features.get('rsi_14', 50))
            macd_signal = self._macd_to_signal(
                latest_features.get('macd', 0),
                latest_features.get('macd_signal', 0)
            )
            bb_signal = self._bollinger_to_signal(
                latest_features.get('bb_position', 0.5)
            )

            # Aggregate signals (simple weighted average)
            overall_signal = (
                0.4 * rsi_signal +
                0.3 * macd_signal +
                0.3 * bb_signal
            )

            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'overall_signal': overall_signal,
                'confidence': None,
                'availability': 'Available', 'nature': 'diagnostic',
                'reason': 'Weighted technical rules; confidence and future price direction have not been validated',
                'data_as_of': ohlcv_data.index[-1].isoformat(), 'provider': 'Yahoo Finance',
                'signals': {
                    'rsi': {'value': rsi_signal, 'weight': 0.4},
                    'macd': {'value': macd_signal, 'weight': 0.3},
                    'bollinger': {'value': bb_signal, 'weight': 0.3}
                },
                'recommendation': self._signal_to_recommendation(overall_signal),
                'technical_indicators': {
                    'rsi_14': latest_features.get('rsi_14', 50),
                    'macd': latest_features.get('macd', 0),
                    'macd_signal': latest_features.get('macd_signal', 0),
                    'bb_position': latest_features.get('bb_position', 0.5)
                }
            }

            return convert_numpy_types(result)

        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            result = {
                'symbol': symbol,
                'error': str(e),
                'overall_signal': None,
                'confidence': None, 'availability': 'Unavailable', 'nature': 'diagnostic',
                'reason': 'Technical observations are unavailable', 'signals': {},
                'recommendation': 'Unavailable', 'technical_indicators': {}
            }

            return convert_numpy_types(result)

    def _rsi_to_signal(self, rsi: float) -> float:
        """Convert RSI to signal (-1 to +1)."""
        if rsi > 70:
            return -0.5  # Overbought (bearish)
        elif rsi < 30:
            return 0.5  # Oversold (bullish)
        else:
            return (50 - rsi) / 50  # Linear scaling

    def _macd_to_signal(self, macd: float, macd_signal: float) -> float:
        """Convert MACD to signal (-1 to +1)."""
        diff = macd - macd_signal
        # Normalize to [-1, 1] range
        return np.tanh(diff * 10)

    def _bollinger_to_signal(self, bb_position: float) -> float:
        """Convert Bollinger Band position to signal (-1 to +1)."""
        # bb_position: 0 = lower band, 0.5 = middle, 1 = upper band
        if bb_position > 0.9:
            return -0.5  # Near upper band (overbought)
        elif bb_position < 0.1:
            return 0.5  # Near lower band (oversold)
        else:
            return (0.5 - bb_position) * 2  # Linear scaling

    def _signal_to_recommendation(self, signal: float) -> str:
        """Convert signal to recommendation."""
        if signal > 0.3:
            return "bullish"
        elif signal < -0.3:
            return "bearish"
        else:
            return "neutral"

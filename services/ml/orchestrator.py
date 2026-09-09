"""
ML Orchestrator - Unified ML system respecting configuration settings
Integrates all ML models and respects data source configuration from settings.html
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from config.settings import Settings
from .data_pipeline import MLDataPipeline
from .models.volatility_predictor import VolatilityPredictor
from .models.correlation_forecaster import CorrelationForecaster
from .models.sentiment_analyzer import SentimentAnalysisEngine
from .models.regime_detector import RegimeDetector
from .models.rebalancing_engine import RebalancingEngine
from services.risk.advanced_risk_engine import AdvancedRiskEngine, VaRMethod, RiskHorizon

logger = logging.getLogger(__name__)

class MLOrchestrator:
    """
    Unified ML orchestrator that respects configuration settings
    Manages all ML models and data sources according to settings.html configuration
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        
        # Initialize data pipeline with configuration awareness
        self.data_pipeline = MLDataPipeline()
        
        # Initialize ML models
        self.models = {
            'volatility': VolatilityPredictor(),
            'correlation': CorrelationForecaster(),
            'sentiment': SentimentAnalysisEngine(),
            'regime': RegimeDetector(),
            'rebalancing': RebalancingEngine()
        }
        
        # Initialize Advanced Risk Engine (Phase 3A integration)
        advanced_risk_config = {
            "var": {
                "confidence_levels": [0.95, 0.99],
                "methods": ["parametric", "historical", "monte_carlo"],
                "lookback_days": 252,
                "min_observations": 100
            },
            "stress_testing": {
                "enabled_scenarios": [
                    "crisis_2008", "covid_2020", "china_ban", "tether_collapse"
                ],
                "custom_scenarios": {},
                "recovery_model": "exponential"
            },
            "monte_carlo": {
                "simulations": 10000,
                "distribution": "student_t",
                "correlation_decay": 0.94
            }
        }
        self.advanced_risk_engine = AdvancedRiskEngine(advanced_risk_config)
        self.models['advanced_risk'] = self.advanced_risk_engine
        
        # Model status tracking
        self.model_status = {name: 'uninitialized' for name in self.models.keys()}
        self.last_training = {name: None for name in self.models.keys()}
        
        # Cache for predictions and metrics
        self.prediction_cache = {}
        self.metrics_cache = {}
        
        logger.info("ML Orchestrator initialized with configuration support")
    
    async def get_data_source_config(self) -> Optional[str]:
        """
        Return an explicitly injected ML data source.

        The backend cannot read a browser's selected portfolio identity. It must
        therefore abstain instead of inferring a source from local files or API
        keys. Portfolio-bound ML needs a dedicated authenticated contract.
        
        Returns:
            Explicit data source, or None when no source is bound.
        """
        configured = getattr(self.settings, 'data_source', None)
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        logger.warning("No identity-bound ML data source is configured")
        return None
    
    async def get_portfolio_assets(self, min_usd: float = 100) -> List[str]:
        """
        Get portfolio assets respecting configured data source
        
        Args:
            min_usd: Minimum USD value threshold
            
        Returns:
            List of asset symbols from configured source
        """
        try:
            data_source = await self.get_data_source_config()

            if not data_source:
                logger.warning("Portfolio universe unavailable without an explicit data source")
                return []
            if data_source == 'stub' or data_source.startswith('stub_'):
                logger.warning("Synthetic data source cannot provide a real portfolio universe")
                return []
            logger.info(f"Fetching portfolio assets from configured source: {data_source}")
            
            assets = self.data_pipeline.fetch_portfolio_assets(
                source=data_source, 
                min_usd=min_usd
            )
            
            logger.info(f"Retrieved {len(assets)} assets from {data_source}: {assets}")
            return assets
            
        except Exception as e:
            logger.error(f"Error fetching portfolio assets: {e}")
            return []
    
    async def initialize_models(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Initialize all ML models with current data source configuration
        
        Args:
            force_retrain: Force retraining even if models exist
            
        Returns:
            Status report of model initialization
        """
        logger.info("Initializing ML models with configuration-aware data")
        
        initialization_report = {
            'data_source': await self.get_data_source_config(),
            'models_initialized': [],
            'models_failed': [],
            'asset_count': 0,
            'training_samples': 0,
            'errors': []
        }
        
        try:
            # Get portfolio assets from configured source
            portfolio_assets = await self.get_portfolio_assets(min_usd=50)
            initialization_report['asset_count'] = len(portfolio_assets)
            
            if not portfolio_assets:
                raise ValueError("No portfolio assets found from configured data source")
            
            # Prepare training data
            logger.info(f"Preparing training data for {len(portfolio_assets)} assets")
            training_data = self.data_pipeline.prepare_training_data(
                symbols=portfolio_assets[:10],  # Limit for initial testing
                days=365,  # 1 year of data
                target_horizons=[1, 7, 30]
            )
            
            if not training_data:
                raise ValueError("No training data could be prepared")
            
            total_samples = sum(len(df) for df in training_data.values())
            initialization_report['training_samples'] = total_samples
            
            # Initialize each model
            for model_name, model in self.models.items():
                try:
                    logger.info(f"Initializing {model_name} model")
                    
                    if model_name == 'volatility':
                        # Train volatility predictor on each asset
                        for symbol, data in list(training_data.items())[:3]:  # Limit for testing
                            await self._train_volatility_model(symbol, data, force_retrain)
                    
                    elif model_name == 'correlation':
                        # Train correlation forecaster on multi-asset data
                        multi_asset_data = self.data_pipeline.prepare_multi_asset_data(
                            symbols=list(training_data.keys())[:5],
                            days=365
                        )
                        if len(multi_asset_data) > 100:
                            await self._train_correlation_model(multi_asset_data, force_retrain)
                    
                    elif model_name == 'sentiment':
                        # Initialize sentiment analyzer (usually doesn't need training)
                        self.model_status[model_name] = 'ready'
                    
                    elif model_name == 'regime':
                        # Train regime detector on available main asset (prefer BTC, ETH, then any)
                        main_asset = None
                        preferred_assets = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
                        
                        # Try preferred assets first
                        for asset in preferred_assets:
                            if asset in training_data:
                                main_asset = asset
                                break
                        
                        # If no preferred asset, use first available asset
                        if main_asset is None and training_data:
                            main_asset = list(training_data.keys())[0]
                        
                        if main_asset:
                            logger.info(f"Training regime detector using {main_asset}")
                            await self._train_regime_model(training_data[main_asset], force_retrain)
                        else:
                            logger.warning("No suitable asset found for regime detector training")
                    
                    elif model_name == 'rebalancing':
                        # Initialize rebalancing engine
                        self.model_status[model_name] = 'ready'
                    
                    elif model_name == 'advanced_risk':
                        # Initialize Advanced Risk Engine with portfolio data
                        await self._initialize_advanced_risk_engine(training_data, force_retrain)
                        self.model_status[model_name] = 'ready'
                    
                    initialization_report['models_initialized'].append(model_name)
                    logger.info(f"Successfully initialized {model_name} model")
                    
                except Exception as e:
                    error_msg = f"Failed to initialize {model_name}: {str(e)}"
                    logger.error(error_msg)
                    initialization_report['models_failed'].append(model_name)
                    initialization_report['errors'].append(error_msg)
                    self.model_status[model_name] = 'failed'
            
            # Generate data quality report
            data_quality = self.data_pipeline.get_data_quality_report(training_data)
            initialization_report['data_quality'] = data_quality
            
            logger.info(f"Model initialization complete: "
                       f"{len(initialization_report['models_initialized'])} successful, "
                       f"{len(initialization_report['models_failed'])} failed")
            
            return initialization_report
            
        except Exception as e:
            error_msg = f"Critical error in model initialization: {str(e)}"
            logger.error(error_msg)
            initialization_report['errors'].append(error_msg)
            return initialization_report
    
    async def _train_volatility_model(self, symbol: str, data: Any, force_retrain: bool):
        """Train volatility predictor for a specific asset"""
        try:
            # Check if model already exists and is recent
            if not force_retrain and hasattr(self.models['volatility'], 'model'):
                last_training = self.last_training.get('volatility')
                if last_training and (datetime.now() - last_training) < timedelta(days=7):
                    self.model_status['volatility'] = 'ready'
                    return
            
            # Train the model (simplified for demonstration)
            logger.info(f"Training volatility model for {symbol}")
            
            # This would call the actual training method
            # self.models['volatility'].train(data)
            
            self.model_status['volatility'] = 'ready'
            self.last_training['volatility'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error training volatility model for {symbol}: {e}")
            self.model_status['volatility'] = 'failed'
    
    async def _train_correlation_model(self, multi_asset_data: Any, force_retrain: bool):
        """Train correlation forecaster"""
        try:
            if not force_retrain and hasattr(self.models['correlation'], 'model'):
                last_training = self.last_training.get('correlation')
                if last_training and (datetime.now() - last_training) < timedelta(days=7):
                    self.model_status['correlation'] = 'ready'
                    return
            
            logger.info("Training correlation forecaster")
            
            # This would call the actual training method
            # self.models['correlation'].train(multi_asset_data)
            
            self.model_status['correlation'] = 'ready'
            self.last_training['correlation'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error training correlation model: {e}")
            self.model_status['correlation'] = 'failed'
    
    async def _train_regime_model(self, data: Any, force_retrain: bool):
        """Train regime detector"""
        try:
            if not force_retrain and hasattr(self.models['regime'], 'model'):
                last_training = self.last_training.get('regime')
                if last_training and (datetime.now() - last_training) < timedelta(days=7):
                    self.model_status['regime'] = 'ready'
                    return
            
            logger.info("Training regime detector")
            
            # This would call the actual training method
            # self.models['regime'].train(data)
            
            self.model_status['regime'] = 'ready'
            self.last_training['regime'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error training regime model: {e}")
            self.model_status['regime'] = 'failed'
    
    async def _initialize_advanced_risk_engine(self, training_data: Dict[str, Any], force_retrain: bool):
        """Initialize Advanced Risk Engine with portfolio data"""
        try:
            if not force_retrain:
                last_training = self.last_training.get('advanced_risk')
                if last_training and (datetime.now() - last_training) < timedelta(days=1):
                    logger.info("Advanced Risk Engine recently initialized, skipping")
                    return
            
            logger.info("Initializing Advanced Risk Engine with portfolio data")
            
            # Prepare historical data for VaR calculations
            portfolio_symbols = list(training_data.keys())[:10]  # Limit for performance
            
            # Initialize with portfolio configuration
            await self.advanced_risk_engine.initialize_portfolio(
                symbols=portfolio_symbols,
                historical_data=training_data
            )
            
            self.last_training['advanced_risk'] = datetime.now()
            logger.info("Advanced Risk Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Advanced Risk Engine: {e}")
            self.model_status['advanced_risk'] = 'failed'
    
    async def get_unified_predictions(self, symbols: Optional[List[str]] = None, 
                                   horizons: List[int] = [1, 7, 30]) -> Dict[str, Any]:
        """
        Get unified predictions from all models using configured data source
        
        Args:
            symbols: Asset symbols to predict (None for portfolio assets)
            horizons: Prediction horizons in days
            
        Returns:
            Comprehensive predictions from all models
        """
        try:
            # Use portfolio assets if not specified
            if symbols is None:
                symbols = await self.get_portfolio_assets(min_usd=50)
            
            data_source = await self.get_data_source_config()
            if not data_source or data_source == 'stub' or data_source.startswith('stub_'):
                return {
                    'available': False,
                    'timestamp': datetime.now().isoformat(),
                    'data_source': data_source,
                    'symbols': symbols,
                    'horizons': horizons,
                    'models': {},
                    'ensemble': {},
                    'confidence_scores': {},
                    'alerts': ['An explicit real data source is required for verified predictions'],
                    'reason': 'No identity-bound real ML data source is configured'
                }
            
            predictions = {
                'timestamp': datetime.now().isoformat(),
                'data_source': data_source,
                'symbols': symbols,
                'horizons': horizons,
                'models': {},
                'ensemble': {},
                'confidence_scores': {},
                'alerts': []
            }
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                if self.model_status[model_name] != 'ready':
                    predictions['alerts'].append(f"{model_name} model not ready")
                    continue
                
                try:
                    if model_name == 'volatility':
                        vol_predictions = await self._get_volatility_predictions(symbols, horizons)
                        predictions['models']['volatility'] = vol_predictions
                    
                    elif model_name == 'sentiment':
                        sentiment_data = await self._get_sentiment_analysis(symbols)
                        predictions['models']['sentiment'] = sentiment_data
                    
                    elif model_name == 'regime':
                        regime_data = await self._get_regime_predictions(symbols)
                        predictions['models']['regime'] = regime_data
                    
                    elif model_name == 'correlation':
                        correlation_data = await self._get_correlation_forecasts(symbols)
                        predictions['models']['correlation'] = correlation_data
                    
                    elif model_name == 'advanced_risk':
                        risk_analysis = await self._get_advanced_risk_analysis(symbols, horizons)
                        predictions['models']['advanced_risk'] = risk_analysis
                    
                except Exception as e:
                    error_msg = f"Error getting {model_name} predictions: {str(e)}"
                    logger.error(error_msg)
                    predictions['alerts'].append(error_msg)
            
            # Create ensemble predictions
            predictions['ensemble'] = await self._create_ensemble_predictions(predictions['models'])
            
            # Calculate confidence scores
            predictions['confidence_scores'] = await self._calculate_confidence_scores(predictions['models'])
            
            # Cache results
            cache_key = f"predictions_{'-'.join(symbols)}_{'-'.join(map(str, horizons))}"
            self.prediction_cache[cache_key] = {
                'data': predictions,
                'timestamp': datetime.now()
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating unified predictions: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'data_source': await self.get_data_source_config()
            }
    
    async def _get_volatility_predictions(self, symbols: List[str], horizons: List[int]) -> Dict[str, Any]:
        """Get volatility predictions for symbols"""
        volatility_predictions = {}

        for symbol in symbols[:3]:  # Limit for testing
            try:
                # Get recent data for prediction
                recent_data = self.data_pipeline.get_prediction_data(symbol, lookback_days=90)
                if recent_data is None:
                    continue

                # Calculate historical volatility from recent data
                try:
                    import pandas as pd
                    import numpy as np

                    # Extract close prices from DataFrame
                    if isinstance(recent_data, pd.DataFrame) and 'close' in recent_data.columns:
                        prices_series = recent_data['close']
                    elif isinstance(recent_data, dict) and 'close' in recent_data:
                        prices_series = pd.Series(recent_data['close'])
                    elif isinstance(recent_data, (list, pd.Series)):
                        prices_series = pd.Series(recent_data)
                    else:
                        raise ValueError(f"Unexpected data format: {type(recent_data)}")

                    # Calculate returns and volatility
                    returns = prices_series.pct_change().dropna()

                    if len(returns) < 7:
                        raise ValueError(f"Insufficient data points: {len(returns)}")

                    # Calculate volatility for different windows (annualized)
                    vol_7d = returns.tail(7).std() * np.sqrt(365)
                    vol_30d = returns.tail(30).std() * np.sqrt(365) if len(returns) >= 30 else vol_7d
                    vol_90d = returns.std() * np.sqrt(365)

                    # Map horizons to appropriate volatility estimates
                    vol_map = {
                        1: float(vol_7d) if not pd.isna(vol_7d) else None,
                        7: float(vol_7d) if not pd.isna(vol_7d) else None,
                        30: float(vol_30d) if not pd.isna(vol_30d) else None,
                        90: float(vol_90d) if not pd.isna(vol_90d) else None
                    }

                    logger.debug(f"Calculated volatility for {symbol}: 7d={vol_7d:.2%}, 30d={vol_30d:.2%}, 90d={vol_90d:.2%}")

                except Exception as calc_error:
                    logger.warning(f"Volatility calculation failed for {symbol}: {calc_error}")
                    volatility_predictions[symbol] = {
                        f'{horizon}d': {
                            'available': False,
                            'volatility_forecast': None,
                            'reason': str(calc_error)
                        }
                        for horizon in horizons
                    }
                    continue

                # Generate predictions for each horizon
                symbol_predictions = {}
                for horizon in horizons:
                    vol_forecast = vol_map.get(horizon)
                    if vol_forecast is None:
                        symbol_predictions[f'{horizon}d'] = {
                            'available': False,
                            'volatility_forecast': None,
                            'reason': 'The requested historical window is unavailable'
                        }
                        continue

                    pred = {
                        'available': True,
                        'volatility_forecast': float(vol_forecast),
                        'confidence': None,
                        'is_forecast': False,
                        'metric_type': 'historical_realized_volatility',
                        'risk_level': 'high' if vol_forecast > 0.6 else 'medium' if vol_forecast > 0.4 else 'low'
                    }
                    symbol_predictions[f'{horizon}d'] = pred

                volatility_predictions[symbol] = symbol_predictions

            except Exception as e:
                logger.error(f"Error predicting volatility for {symbol}: {e}")

        return volatility_predictions
    
    async def _get_sentiment_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Get sentiment analysis for symbols using real SentimentAnalysisEngine"""
        try:
            sentiment_engine = self.models['sentiment']

            # Analyze market sentiment using real engine (calls alternative.me, social, news APIs)
            results = await sentiment_engine.analyze_market_sentiment(symbols[:5], days=7)

            # Map to expected format
            sentiment_data = {}
            individual_assets = results.get('individual_assets', {})

            for symbol in symbols[:5]:
                asset_data = individual_assets.get(symbol, {})
                sentiment_score = asset_data.get('overall_sentiment')
                if not isinstance(sentiment_score, (int, float)):
                    sentiment_data[symbol] = {
                        'available': False,
                        'reason': 'No verified sentiment observation is available'
                    }
                    continue

                # Extract sentiment score (range -1 to 1)
                confidence = asset_data.get('confidence')

                sentiment_data[symbol] = {
                    'available': True,
                    'sentiment_score': sentiment_score,
                    'confidence': confidence if isinstance(confidence, (int, float)) else None,
                    'data_points': asset_data.get('data_points', 0),
                    'source_breakdown': asset_data.get('source_breakdown', {}),
                    'social_mentions': asset_data.get('source_breakdown', {}).get('social_media', {}).get('volume', 0),
                    'news_sentiment': self._classify_sentiment_label(sentiment_score)
                }

            logger.info(f"Real sentiment analysis completed for {len(sentiment_data)} symbols")
            return sentiment_data

        except Exception as e:
            logger.warning(f"Real sentiment analysis failed: {e}")
            return {
                symbol: {'available': False, 'reason': str(e)}
                for symbol in symbols[:5]
            }

    def _classify_sentiment_label(self, sentiment_score: float) -> str:
        """Convert sentiment score to label"""
        if sentiment_score >= 0.6:
            return 'bullish'
        elif sentiment_score >= 0.2:
            return 'slightly_bullish'
        elif sentiment_score >= -0.2:
            return 'neutral'
        elif sentiment_score >= -0.6:
            return 'slightly_bearish'
        else:
            return 'bearish'
    
    async def predict_volatility(self, symbol: str, horizon_days: int = 30) -> Dict[str, Any]:
        """
        Public method for single asset volatility prediction
        Used by unified_ml_endpoints.py
        """
        try:
            # Use existing volatility prediction logic
            predictions = await self._get_volatility_predictions([symbol], [horizon_days])
            
            if symbol in predictions and f'{horizon_days}d' in predictions[symbol]:
                result = predictions[symbol][f'{horizon_days}d']
                result.update({
                    'symbol': symbol,
                    'horizon_days': horizon_days,
                    'timestamp': datetime.now().isoformat(),
                    'model_version': '2.0.0'
                })
                return result
            else:
                return {
                    'symbol': symbol,
                    'horizon_days': horizon_days,
                    'available': False,
                    'volatility_forecast': None,
                    'confidence': None,
                    'timestamp': datetime.now().isoformat(),
                    'model_version': None,
                    'reason': 'No verified volatility result is available'
                }
                
        except Exception as e:
            logger.error(f"Error in predict_volatility for {symbol}: {e}")
            return {
                'symbol': symbol,
                'horizon_days': horizon_days,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def calculate_portfolio_var(self, portfolio_weights: Dict[str, float], 
                                    portfolio_value: float = 100000,
                                    confidence_level: float = 0.95,
                                    horizon_days: int = 1,
                                    method: str = 'historical') -> Dict[str, Any]:
        """
        Public method for portfolio VaR calculation
        Used by advanced risk endpoints
        """
        try:
            if self.model_status.get('advanced_risk') != 'ready':
                logger.warning("Advanced Risk Engine not ready for VaR calculation")
                return {
                    'available': False,
                    'error': 'Advanced Risk Engine not initialized',
                    'var_absolute': None,
                    'cvar_absolute': None,
                    'confidence_level': confidence_level,
                    'method': method,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Map method string to enum
            var_method = VaRMethod.HISTORICAL if method.lower() == 'historical' else VaRMethod.PARAMETRIC
            
            # Map horizon to enum
            if horizon_days == 1:
                horizon = RiskHorizon.DAILY
            elif horizon_days <= 7:
                horizon = RiskHorizon.WEEKLY
            else:
                horizon = RiskHorizon.MONTHLY
            
            # Calculate VaR using Advanced Risk Engine
            var_result = await self.advanced_risk_engine.calculate_var(
                portfolio_weights=portfolio_weights,
                portfolio_value=portfolio_value,
                method=var_method,
                confidence_level=confidence_level,
                horizon=horizon
            )
            
            return {
                'available': True,
                'var_absolute': var_result.var_absolute,
                'cvar_absolute': var_result.cvar_absolute,
                'confidence_level': var_result.confidence_level,
                'method': var_result.method.value,
                'horizon': var_result.horizon.value,
                'portfolio_value': var_result.portfolio_value,
                'component_contributions': var_result.component_contributions,
                'model_parameters': var_result.model_parameters,
                'timestamp': datetime.now().isoformat(),
                'engine_version': '3.0.0'
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return {
                'available': False,
                'error': str(e),
                'var_absolute': None,
                'cvar_absolute': None,
                'confidence_level': confidence_level,
                'method': method,
                'timestamp': datetime.now().isoformat()
            }

    async def _get_regime_predictions(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market regime predictions"""
        return {
            'available': False,
            'current_regime': None,
            'regime_probability': None,
            'regime_stability': None,
            'expected_duration_days': None,
            'reason': 'No verified regime inference is connected to the orchestrator'
        }
    
    async def _get_correlation_forecasts(self, symbols: List[str]) -> Dict[str, Any]:
        """Get correlation forecasts between assets"""
        return {
            'available': False,
            'avg_correlation': None,
            'systemic_risk': None,
            'reason': 'No verified correlation forecast is connected to the orchestrator'
        }
    
    async def _get_advanced_risk_analysis(self, symbols: List[str], horizons: List[int]) -> Dict[str, Any]:
        """Return unavailable until authenticated portfolio weights are provided."""
        return {
            'available': False,
            'status': 'unavailable',
            'symbols': list(symbols),
            'horizons_days': list(horizons),
            'reason': 'Advanced portfolio risk requires verified portfolio weights and value'
        }
    async def _assess_overall_portfolio_risk(self, risk_analysis: Dict[str, Any]) -> str:
        """Assess overall portfolio risk level based on analysis results"""
        try:
            risk_indicators = []
            
            # Check VaR levels
            if 'var_analysis' in risk_analysis and '1d' in risk_analysis['var_analysis']:
                daily_var = risk_analysis['var_analysis']['1d'].get('historical_var', 0)
                if daily_var > 0.05:  # >5% daily VaR
                    risk_indicators.append('high_var')
                elif daily_var > 0.03:
                    risk_indicators.append('moderate_var')
            
            # Check stress test results
            if 'stress_tests' in risk_analysis:
                severe_scenarios = 0
                for scenario_result in risk_analysis['stress_tests'].values():
                    if isinstance(scenario_result, dict) and 'loss_percentage' in scenario_result:
                        if scenario_result['loss_percentage'] > 0.4:  # >40% loss
                            severe_scenarios += 1
                
                if severe_scenarios >= 2:
                    risk_indicators.append('stress_vulnerable')
                elif severe_scenarios == 1:
                    risk_indicators.append('moderate_stress_risk')
            
            # Check Monte Carlo tail risk
            if 'monte_carlo' in risk_analysis and 'tail_risk_analysis' in risk_analysis['monte_carlo']:
                extreme_loss_prob = risk_analysis['monte_carlo']['tail_risk_analysis'].get('extreme_loss_probability', 0)
                if extreme_loss_prob > 0.05:  # >5% probability of extreme loss
                    risk_indicators.append('tail_risk')
            
            # Determine overall risk level
            if len(risk_indicators) >= 3:
                return 'high'
            elif len(risk_indicators) >= 2:
                return 'moderate_high'
            elif len(risk_indicators) >= 1:
                return 'moderate'
            else:
                return 'low_moderate'
                
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return 'unknown'
    
    async def _generate_risk_recommendation(self, risk_analysis: Dict[str, Any]) -> str:
        """Generate risk management recommendation based on analysis"""
        try:
            risk_level = await self._assess_overall_portfolio_risk(risk_analysis)
            
            if risk_level == 'high':
                return 'Consider significant risk reduction: decrease position sizes, increase diversification, add hedging positions'
            elif risk_level == 'moderate_high':
                return 'Moderate risk reduction advised: review position sizing and consider partial profit-taking'
            elif risk_level == 'moderate':
                return 'Monitor closely: current risk levels acceptable but watch for deterioration'
            else:
                return 'Risk levels appear manageable: maintain current allocation with regular monitoring'
                
        except Exception as e:
            logger.error(f"Error generating risk recommendation: {e}")
            return 'Unable to generate recommendation due to analysis error'
    
    async def _create_ensemble_predictions(self, model_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Do not manufacture a forecast by voting over heterogeneous diagnostics."""
        return {
            'available': False,
            'overall_market_sentiment': None,
            'risk_assessment': None,
            'recommended_action': None,
            'confidence_level': None,
            'model_contributions': {},
            'consensus_strength': None,
            'conflicting_signals': [],
            'reason': 'No calibrated ensemble forecast is available'
        }
    async def _calculate_confidence_scores(self, model_predictions: Dict[str, Any]) -> Dict[str, float]:
        """Confidence remains absent until it is calibrated out of sample."""
        return {}
    async def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of all ML models and data source configuration
        
        Returns:
            Comprehensive status report
        """
        data_source = await self.get_data_source_config()
        
        status_report = {
            'timestamp': datetime.now().isoformat(),
            'data_source_config': data_source,
            'models': {}
        }
        
        for model_name, model in self.models.items():
            status_report['models'][model_name] = {
                'status': self.model_status[model_name],
                'last_training': self.last_training[model_name].isoformat() if self.last_training[model_name] else None,
                'type': model.__class__.__name__,
                'ready_for_predictions': self.model_status[model_name] == 'ready'
            }
        
        # Add system health metrics
        ready_models = sum(1 for status in self.model_status.values() if status == 'ready')
        total_models = len(self.models)
        
        status_report['system_health'] = {
            'models_ready': ready_models,
            'total_models': total_models,
            'readiness_percentage': (ready_models / total_models) * 100,
            'overall_status': 'healthy' if ready_models >= total_models * 0.6 else 'degraded'
        }
        
        return status_report
    
    async def retrain_models(self, model_names: Optional[List[str]] = None, 
                           force: bool = False) -> Dict[str, Any]:
        """
        Retrain specified models or all models
        
        Args:
            model_names: Models to retrain (None for all)
            force: Force retraining even if recent
            
        Returns:
            Retraining status report
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        logger.info(f"Retraining models: {model_names}")
        
        retrain_report = await self.initialize_models(force_retrain=force)
        retrain_report['retrained_models'] = model_names
        
        return retrain_report
    
    def clear_caches(self) -> Dict[str, int]:
        """Clear all caches"""
        cleared_counts = {
            'predictions': len(self.prediction_cache),
            'metrics': len(self.metrics_cache),
            'data_pipeline': self.data_pipeline.clear_cache()
        }
        
        self.prediction_cache.clear()
        self.metrics_cache.clear()
        
        logger.info(f"Cleared caches: {cleared_counts}")
        return cleared_counts
    
    async def load_regime_model(self) -> Dict[str, Any]:
        """Load regime detection model for auto-startup"""
        try:
            from services.ml_pipeline_manager_optimized import optimized_pipeline_manager as pipeline_manager
            await pipeline_manager.load_regime_model()
            self.model_status['regime'] = 'ready'
            logger.info("Regime model loaded successfully")
            return {"success": True, "message": "Regime model loaded"}
        except Exception as e:
            logger.error(f"Failed to load regime model: {e}")
            return {"success": False, "error": str(e)}

# Global orchestrator instance
_orchestrator = None

def get_orchestrator() -> MLOrchestrator:
    """Get or create global ML orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MLOrchestrator()
    return _orchestrator

def reset_orchestrator():
    """Reset global orchestrator instance (useful for development/testing)"""
    global _orchestrator
    _orchestrator = None

async def initialize_ml_system(force_retrain: bool = False) -> Dict[str, Any]:
    """Initialize the complete ML system"""
    orchestrator = get_orchestrator()
    return await orchestrator.initialize_models(force_retrain=force_retrain)

async def get_ml_predictions(symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Get unified ML predictions"""
    orchestrator = get_orchestrator()
    return await orchestrator.get_unified_predictions(symbols=symbols)

async def get_ml_status() -> Dict[str, Any]:
    """Get ML system status"""
    orchestrator = get_orchestrator()
    return await orchestrator.get_model_status()

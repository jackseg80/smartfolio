"""
Decision Index Backtest API Endpoints
Validation rétroactive du Decision Index sur données historiques

Endpoints:
- POST /api/di-backtest/run - Exécuter un backtest complet
- GET /api/di-backtest/historical-di - Reconstruire le DI historique
- GET /api/di-backtest/strategies - Liste des stratégies DI
- POST /api/di-backtest/compare - Comparer plusieurs stratégies
- GET /api/di-backtest/events - Événements marché majeurs
- GET /api/di-backtest/period-analysis - Analyse par période
"""

from datetime import datetime
from typing import Dict, List, Optional
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd

from api.deps import get_required_user
from api.utils import success_response, error_response
from services.di_backtest import (
    historical_di_calculator,
    HistoricalDICalculator,
    DIHistoryPoint,
    di_backtest_engine,
    DIBacktestEngine,
)
from services.di_backtest.trading_strategies import (
    DI_STRATEGIES,
    get_di_strategy,
    DIStrategyConfig,
)
from services.di_backtest.historical_di_calculator import DIWeights, PhaseFactors, DICalculatorConfig
from services.backtesting_engine import (
    backtesting_engine,
    BacktestConfig,
    TransactionCosts,
    RebalanceFrequency,
)
from services.price_history import get_cached_history, download_historical_data

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/di-backtest", tags=["DI Backtest"])


# ========== Pydantic Models ==========

class DIWeightsModel(BaseModel):
    """Poids des composants du Decision Index"""
    cycle: float = Field(0.30, ge=0, le=1)
    onchain: float = Field(0.35, ge=0, le=1)
    risk: float = Field(0.25, ge=0, le=1)
    sentiment: float = Field(0.10, ge=0, le=1)


class ReplicaParamsModel(BaseModel):
    """Advanced parameters for SmartFolio Replica strategy (Layer toggles + tuning)"""
    enable_risk_budget: bool = Field(True, description="Enable Layer 1: Risk Budget")
    enable_market_overrides: bool = Field(True, description="Enable Layer 2: Market Overrides")
    enable_exposure_cap: bool = Field(True, description="Enable Layer 3: Exposure Cap")
    enable_governance_penalty: bool = Field(True, description="Enable Layer 4: Governance Penalty")
    exposure_confidence: float = Field(0.65, ge=0.50, le=1.0, description="Signal quality confidence for exposure cap")
    max_governance_penalty: float = Field(0.25, ge=0.0, le=0.30, description="Maximum governance penalty")
    risk_budget_min: float = Field(0.20, ge=0.10, le=0.30, description="Minimum risky allocation")
    risk_budget_max: float = Field(0.85, ge=0.70, le=0.95, description="Maximum risky allocation")
    enable_direction_penalty: bool = Field(True, description="Enable V2.1 cycle direction penalty")


class RotationParamsModel(BaseModel):
    """Parameters for Cycle Rotation strategy (BTC/ETH/Stables)"""
    # Preset: maps to predefined allocation profiles
    preset: str = Field("conservative", description="Allocation preset: conservative, default, aggressive")

    # Smoothing
    smoothing_alpha: float = Field(0.30, ge=0.0, le=1.0, description="EMA smoothing factor (0=no smoothing, 1=instant)")

    # Asymmetric alpha: separate speeds for bull vs bear transitions
    smoothing_alpha_bullish: Optional[float] = Field(None, ge=0.0, le=1.0, description="Slow entry into bull (e.g. 0.15). When set, overrides smoothing_alpha for bullish moves")
    smoothing_alpha_bearish: Optional[float] = Field(None, ge=0.0, le=1.0, description="Fast exit to bear (e.g. 0.50). When set, overrides smoothing_alpha for bearish moves")

    # DI modulation (off by default — backtests show DI is not predictive)
    enable_di_modulation: bool = Field(False, description="Enable DI modulation of phase targets")
    di_mod_range: float = Field(0.10, ge=0.0, le=0.30, description="DI modulation range")

    # Drawdown circuit breaker
    enable_drawdown_breaker: bool = Field(False, description="Enable drawdown circuit breaker")
    dd_threshold_1: float = Field(-0.10, le=0.0, description="First DD threshold (moderate cut)")
    dd_threshold_2: float = Field(-0.20, le=0.0, description="Second DD threshold (floor to bear)")
    dd_multiplier: float = Field(0.50, ge=0.0, le=1.0, description="Risky multiplier at threshold_1")
    dd_ramp_up: bool = Field(True, description="Gradual recovery after DD cut")


class ContinuousParamsModel(BaseModel):
    """Parameters for Adaptive Continuous strategy (S9)"""
    alloc_floor: float = Field(0.10, ge=0.05, le=0.30, description="Minimum risky allocation (bear floor)")
    alloc_ceiling: float = Field(0.85, ge=0.50, le=0.95, description="Maximum risky allocation (bull ceiling)")
    enable_trend_overlay: bool = Field(True, description="Enable golden/death cross trend confirmation")
    sma_fast: int = Field(50, ge=10, le=100, description="Fast SMA period for trend overlay")
    sma_slow: int = Field(200, ge=100, le=300, description="Slow SMA period for trend overlay")
    trend_boost_pct: float = Field(0.10, ge=0.0, le=0.20, description="Max trend boost (+/- allocation)")
    enable_risk_adjustment: bool = Field(True, description="Enable risk score adjustment")
    smoothing_alpha_bull: float = Field(0.12, ge=0.01, le=1.0, description="Slow entry smoothing (bull)")
    smoothing_alpha_bear: float = Field(0.50, ge=0.01, le=1.0, description="Fast exit smoothing (bear)")
    eth_share_bull: float = Field(0.40, ge=0.10, le=0.60, description="ETH share of risky in bull")
    eth_share_bear: float = Field(0.20, ge=0.05, le=0.40, description="ETH share of risky in bear")


class DIBacktestRequest(BaseModel):
    """Requête pour exécuter un backtest DI"""
    strategy: str = Field(..., description="Strategy: di_threshold, di_momentum, di_contrarian, di_risk_parity, di_signal, di_cycle_rotation, di_adaptive_continuous")
    start_date: str = Field(..., description="Date début YYYY-MM-DD (min: 2017-01-01)")
    end_date: str = Field(..., description="Date fin YYYY-MM-DD")
    initial_capital: float = Field(10000.0, gt=0)
    rebalance_frequency: str = Field("weekly", description="daily, weekly, monthly")

    # Custom weights (optionnel)
    di_weights: Optional[DIWeightsModel] = None

    # Advanced params for SmartFolio Replica
    replica_params: Optional[ReplicaParamsModel] = None

    # Advanced params for Cycle Rotation
    rotation_params: Optional[RotationParamsModel] = None

    # Advanced params for Adaptive Continuous (S9)
    continuous_params: Optional[ContinuousParamsModel] = None

    # Options
    use_macro_penalty: bool = Field(True, description="Inclure pénalité VIX/DXY")
    calculator_version: str = Field("v1", description="Calculator version: v1 (fixed normalization) or v2 (adaptive)")

    # Transaction costs
    transaction_fee_pct: float = Field(0.001, ge=0)
    slippage_bps: float = Field(10.0, ge=0)


class DIHistoryRequest(BaseModel):
    """Requête pour le DI historique"""
    start_date: str = Field("2017-01-01", description="Date début YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="Date fin (None = aujourd'hui)")
    di_weights: Optional[DIWeightsModel] = None
    include_macro: bool = True


class DICompareRequest(BaseModel):
    """Requête pour comparer plusieurs stratégies"""
    strategies: List[str] = Field(..., description="Liste des stratégies à comparer")
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    rebalance_frequency: str = "weekly"


# ========== Événements Marché Prédéfinis ==========

MARKET_EVENTS = [
    {
        "name": "Bull Run 2017",
        "start": "2017-01-01",
        "end": "2017-12-31",
        "type": "bull",
        "description": "Premier grand bull market crypto, BTC atteint $20k"
    },
    {
        "name": "Crash 2018",
        "start": "2018-01-01",
        "end": "2018-12-31",
        "type": "bear",
        "description": "Bear market -85%, chute post-bulle ICO"
    },
    {
        "name": "COVID Crash",
        "start": "2020-02-01",
        "end": "2020-05-01",
        "type": "crash",
        "description": "Crash pandémie, BTC -50% en mars puis recovery rapide"
    },
    {
        "name": "Bull Run 2020-2021",
        "start": "2020-10-01",
        "end": "2021-11-15",
        "type": "bull",
        "description": "DeFi Summer + institutional adoption, BTC atteint $69k"
    },
    {
        "name": "Bear Market 2022",
        "start": "2021-11-15",
        "end": "2022-11-15",
        "type": "bear",
        "description": "Crash Luna/Terra, 3AC, FTX collapse"
    },
    {
        "name": "Recovery 2023-2024",
        "start": "2022-11-15",
        "end": "2024-04-01",
        "type": "recovery",
        "description": "Reprise post-FTX, ETF BTC spot"
    },
    {
        "name": "Halving 2024",
        "start": "2024-04-01",
        "end": "2024-12-31",
        "type": "accumulation",
        "description": "Quatrième halving Bitcoin, phase accumulation"
    },
]


# ========== Endpoints ==========

@router.get("/strategies")
async def list_di_strategies(
    details: bool = Query(False, description="Include detailed descriptions")
):
    """
    Liste les stratégies DI disponibles pour le backtest

    Args:
        details: Si True, inclut les règles détaillées, pros/cons, etc.
    """
    strategies = []

    for key, strategy_class in DI_STRATEGIES.items():
        instance = strategy_class()
        strategy_info = {
            "key": key,
            "name": instance.name,
            "description": _get_strategy_description(key)
        }

        # Ajouter détails si demandé
        if details and key in STRATEGY_DETAILS:
            detail = STRATEGY_DETAILS[key]
            strategy_info.update({
                "short": detail.get("short"),
                "full_description": detail.get("description"),
                "rules": detail.get("rules", []),
                "pros": detail.get("pros", []),
                "cons": detail.get("cons", []),
                "best_for": detail.get("best_for")
            })

        strategies.append(strategy_info)

    return success_response({
        "strategies": strategies,
        "total_count": len(strategies),
        "rebalance_frequencies": [f.value for f in RebalanceFrequency]
    })


@router.get("/events")
async def get_market_events():
    """Retourne les événements marché prédéfinis pour analyse"""
    return success_response({
        "events": MARKET_EVENTS,
        "total_count": len(MARKET_EVENTS)
    })


@router.post("/historical-di")
async def get_historical_di(
    request: DIHistoryRequest,
    user: str = Depends(get_required_user)
):
    """
    Reconstruit le Decision Index historique pour une période

    Retourne les 4 composants (cycle, onchain, risk, sentiment) + DI final
    pour chaque jour de la période.
    """
    try:
        # Créer le calculateur avec custom weights si fournis
        weights = None
        if request.di_weights:
            weights = DIWeights(
                cycle=request.di_weights.cycle,
                onchain=request.di_weights.onchain,
                risk=request.di_weights.risk,
                sentiment=request.di_weights.sentiment
            )
            if not weights.validate():
                return error_response("Les poids doivent sommer à 1.0", code=400)

        calculator = HistoricalDICalculator(weights=weights)

        # Calculer
        result = await calculator.calculate_historical_di(
            user_id=user,
            start_date=request.start_date,
            end_date=request.end_date,
            include_macro=request.include_macro
        )

        # Convertir en format JSON (avec conversion des types numpy)
        history = []
        for point in result.di_history:
            history.append({
                "date": point.date.strftime("%Y-%m-%d"),
                "decision_index": round(float(point.decision_index), 2),
                "cycle_score": round(float(point.cycle_score), 2),
                "onchain_score": round(float(point.onchain_score), 2),
                "risk_score": round(float(point.risk_score), 2),
                "sentiment_score": round(float(point.sentiment_score), 2),
                "phase": str(point.phase),
                "phase_factor": float(point.phase_factor),
                "macro_penalty": int(point.macro_penalty),
                "btc_price": float(point.btc_price) if point.btc_price else None,
                "cycle_direction": round(float(point.cycle_direction), 3) if point.cycle_direction is not None else None,
                "cycle_confidence": round(float(point.cycle_confidence), 3) if point.cycle_confidence is not None else None,
            })

        return success_response({
            "start_date": result.start_date.strftime("%Y-%m-%d"),
            "end_date": result.end_date.strftime("%Y-%m-%d"),
            "total_points": len(history),
            "history": history,
            "metadata": result.metadata
        })

    except Exception as e:
        logger.error(f"Erreur calcul DI historique: {e}", exc_info=True)
        return error_response(f"Erreur calcul DI: {str(e)}", code=500)


@router.post("/run")
async def run_di_backtest(
    request: DIBacktestRequest,
    user: str = Depends(get_required_user)
):
    """
    Exécute un backtest complet basé sur le Decision Index

    Simule une stratégie de rebalancement utilisant le DI sur des données historiques
    et retourne les métriques de performance (Sharpe, Sortino, Max Drawdown, etc.)
    """
    try:
        # Valider la stratégie
        if request.strategy not in DI_STRATEGIES:
            return error_response(
                f"Stratégie inconnue: {request.strategy}. Disponibles: {list(DI_STRATEGIES.keys())}",
                code=400
            )

        # Parser les dates
        start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
        days_needed = (end_dt - start_dt).days + 250  # Buffer pour rolling windows

        logger.info(f"DI Backtest: {request.strategy} du {request.start_date} au {request.end_date}")

        # 1. Calculer le DI historique
        weights = None
        if request.di_weights:
            weights = DIWeights(
                cycle=request.di_weights.cycle,
                onchain=request.di_weights.onchain,
                risk=request.di_weights.risk,
                sentiment=request.di_weights.sentiment
            )

        calculator = HistoricalDICalculator(weights=weights)
        use_v2 = request.calculator_version == "v2"

        if use_v2:
            di_data = await calculator.calculate_historical_di_v2(
                user_id=user,
                start_date=request.start_date,
                end_date=request.end_date,
                include_macro=request.use_macro_penalty,
                config=DICalculatorConfig(),
            )
        else:
            di_data = await calculator.calculate_historical_di(
                user_id=user,
                start_date=request.start_date,
                end_date=request.end_date,
                include_macro=request.use_macro_penalty
            )

        if di_data.df.empty:
            return error_response("Impossible de calculer le DI historique", code=500)

        # 2. Créer la stratégie DI
        strategy_config = DIStrategyConfig()
        strategy = get_di_strategy(request.strategy, config=strategy_config)

        # Inject advanced params for SmartFolio Replica
        if request.strategy == "di_smartfolio_replica" and request.replica_params:
            from services.di_backtest.trading_strategies import ReplicaParams
            rp = request.replica_params
            strategy.replica_params = ReplicaParams(
                enable_risk_budget=rp.enable_risk_budget,
                enable_market_overrides=rp.enable_market_overrides,
                enable_exposure_cap=rp.enable_exposure_cap,
                enable_governance_penalty=rp.enable_governance_penalty,
                exposure_confidence=rp.exposure_confidence,
                max_governance_penalty=rp.max_governance_penalty,
                risk_budget_min=rp.risk_budget_min,
                risk_budget_max=rp.risk_budget_max,
                enable_direction_penalty=rp.enable_direction_penalty,
            )

        # Inject params for Cycle Rotation
        if request.strategy == "di_cycle_rotation" and request.rotation_params:
            from services.di_backtest.trading_strategies import RotationParams
            rp = request.rotation_params

            # Map preset to allocation profiles
            preset_allocs = {
                "conservative": dict(
                    alloc_bear=(0.10, 0.03, 0.87),
                    alloc_peak=(0.15, 0.15, 0.70),
                    alloc_distribution=(0.15, 0.05, 0.80),
                ),
                "aggressive": dict(
                    alloc_peak=(0.30, 0.20, 0.50),
                    alloc_distribution=(0.25, 0.15, 0.60),
                ),
            }
            alloc_kwargs = preset_allocs.get(rp.preset, {})

            strategy.params = RotationParams(
                **alloc_kwargs,
                smoothing_alpha=rp.smoothing_alpha,
                smoothing_alpha_bullish=rp.smoothing_alpha_bullish,
                smoothing_alpha_bearish=rp.smoothing_alpha_bearish,
                enable_di_modulation=rp.enable_di_modulation,
                di_mod_range=rp.di_mod_range,
                enable_drawdown_breaker=rp.enable_drawdown_breaker,
                dd_threshold_1=rp.dd_threshold_1,
                dd_threshold_2=rp.dd_threshold_2,
                dd_multiplier=rp.dd_multiplier,
                dd_ramp_up=rp.dd_ramp_up,
            )

        # Inject params for Adaptive Continuous (S9)
        if request.strategy == "di_adaptive_continuous" and request.continuous_params:
            from services.di_backtest.trading_strategies import ContinuousParams
            cp = request.continuous_params
            strategy.params = ContinuousParams(
                alloc_floor=cp.alloc_floor,
                alloc_ceiling=cp.alloc_ceiling,
                enable_trend_overlay=cp.enable_trend_overlay,
                sma_fast=cp.sma_fast,
                sma_slow=cp.sma_slow,
                trend_boost_pct=cp.trend_boost_pct,
                enable_risk_adjustment=cp.enable_risk_adjustment,
                smoothing_alpha_bull=cp.smoothing_alpha_bull,
                smoothing_alpha_bear=cp.smoothing_alpha_bear,
                eth_share_bull=cp.eth_share_bull,
                eth_share_bear=cp.eth_share_bear,
            )

        strategy.set_di_series(di_data.df['decision_index'])

        # Injecter le cycle_score pour SmartFolio Replica (utilise le vrai cycle, pas une estimation)
        if hasattr(strategy, 'set_cycle_series') and 'cycle_score' in di_data.df.columns:
            strategy.set_cycle_series(di_data.df['cycle_score'])
            logger.info(f"Cycle score injected: mean={di_data.df['cycle_score'].mean():.1f}")

        # 3. Exécuter le backtest avec di_backtest_engine (moteur simplifié correct)
        # Créer une instance avec les coûts de transaction
        engine = DIBacktestEngine(
            transaction_cost=request.transaction_fee_pct,
            rebalance_threshold=0.05,  # Rebalance si écart > 5%
            risk_free_rate=0.02
        )

        # Enable multi-asset mode for strategies that use ETH
        use_multi_asset = request.strategy in ("di_cycle_rotation", "di_adaptive_continuous")

        logger.info(f"Running DI backtest: {len(di_data.di_history)} points, strategy={request.strategy}, freq={request.rebalance_frequency}, multi_asset={use_multi_asset}")
        result = engine.run_backtest(
            di_history=di_data.di_history,
            strategy=strategy,
            initial_capital=request.initial_capital,
            rebalance_frequency=request.rebalance_frequency,
            multi_asset=use_multi_asset
        )

        # 4. Calculer métriques additionnelles spécifiques au DI
        di_metrics = _calculate_di_specific_metrics_v2(di_data.df, result)

        # Add active layers info for SmartFolio Replica
        if request.strategy == "di_smartfolio_replica":
            rp = request.replica_params
            if rp:
                layers = []
                if rp.enable_risk_budget:
                    layers.append("Risk Budget")
                if rp.enable_market_overrides:
                    layers.append("Market Overrides")
                if rp.enable_exposure_cap:
                    layers.append("Exposure Cap")
                if rp.enable_governance_penalty:
                    layers.append("Governance Penalty")
            else:
                layers = ["Risk Budget", "Market Overrides", "Exposure Cap", "Governance Penalty"]
            di_metrics["active_layers"] = layers

        # 5. Formater la réponse
        # Monthly returns depuis l'equity curve
        equity_df = pd.DataFrame({
            'value': result.equity_curve.values
        }, index=result.equity_curve.index)
        monthly_returns_series = equity_df['value'].resample('ME').last().pct_change().dropna()
        monthly_returns = []
        for date, ret in monthly_returns_series.items():
            monthly_returns.append({
                "date": date.strftime("%Y-%m"),
                "return_pct": round(float(ret) * 100, 2)
            })

        # Equity curve
        equity_curve = []
        for i, (date, value) in enumerate(result.equity_curve.items()):
            bench_value = result.benchmark_curve.iloc[i] if i < len(result.benchmark_curve) else request.initial_capital
            alloc = float(result.allocation_series.iloc[i]) if i < len(result.allocation_series) else 0.5
            equity_curve.append({
                "date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)[:10],
                "portfolio_value": round(float(value), 2),
                "benchmark_value": round(float(bench_value), 2),
                "risky_allocation_pct": round(alloc * 100, 1),
            })

        # Sous-échantillonner l'equity curve pour éviter trop de données
        if len(equity_curve) > 365:
            step = len(equity_curve) // 365
            equity_curve = equity_curve[::step]

        # Drawdowns
        drawdowns = []
        for date, dd in result.drawdown_curve.items():
            drawdowns.append({
                "date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)[:10],
                "drawdown_pct": round(float(dd) * 100, 2)
            })

        if len(drawdowns) > 365:
            step = len(drawdowns) // 365
            drawdowns = drawdowns[::step]

        return success_response({
            "success": True,
            "strategy": request.strategy,
            "period": {
                "start": request.start_date,
                "end": request.end_date,
                "days": len(di_data.df)
            },
            "metrics": {
                "total_return_pct": round(result.total_return * 100, 2),
                "annualized_return_pct": round(result.annualized_return * 100, 2),
                "max_drawdown_pct": round(result.max_drawdown * 100, 2),
                "sharpe_ratio": round(result.sharpe_ratio, 3),
                "sortino_ratio": round(result.sortino_ratio, 3),
                "calmar_ratio": round(result.calmar_ratio, 3),
                "volatility_pct": round(result.volatility * 100, 2),
                # Rebalancing metrics (replacing meaningless win_rate)
                "turnover_annual": round(result.turnover_annual, 2),
                "avg_risky_allocation_pct": round(result.avg_risky_allocation * 100, 1),
                "upside_capture": round(result.upside_capture, 3),
                "downside_capture": round(result.downside_capture, 3),
                "capture_ratio": round(result.upside_capture / result.downside_capture, 3) if result.downside_capture > 0 else None,
            },
            "benchmark_comparison": {
                "benchmark": "BTC Buy & Hold",
                "benchmark_return_pct": round(result.benchmark_return * 100, 2),
                "excess_return_pct": round(result.excess_return * 100, 2),
            },
            "di_metrics": di_metrics,
            "equity_curve": equity_curve,
            "monthly_returns": monthly_returns,
            "drawdowns": drawdowns,
            "summary": {
                "strategy_name": result.strategy_name,
                "total_days": len(di_data.df),
                "rebalance_count": result.rebalance_count,
                "final_value": round(result.final_value, 2),
                "total_return_pct": round(result.total_return * 100, 2),
                "benchmark_return_pct": round(result.benchmark_return * 100, 2),
                "calculator_version": request.calculator_version,
            }
        })

    except Exception as e:
        logger.error(f"DI Backtest failed: {e}", exc_info=True)
        return error_response(f"Backtest échoué: {str(e)}", code=500)


@router.post("/compare")
async def compare_di_strategies(
    request: DICompareRequest,
    user: str = Depends(get_required_user)
):
    """Compare plusieurs stratégies DI côte à côte"""
    try:
        # Valider les stratégies
        invalid = [s for s in request.strategies if s not in DI_STRATEGIES]
        if invalid:
            return error_response(
                f"Stratégies inconnues: {invalid}. Disponibles: {list(DI_STRATEGIES.keys())}",
                code=400
            )

        # Calculer DI historique une seule fois
        calculator = HistoricalDICalculator()
        di_data = await calculator.calculate_historical_di(
            user_id=user,
            start_date=request.start_date,
            end_date=request.end_date
        )

        # Préparer données prix
        btc_prices = di_data.df['btc_price']
        price_df = pd.DataFrame({
            'BTC': btc_prices,
            'STABLES': pd.Series(1.0, index=btc_prices.index)
        })

        # Exécuter chaque stratégie
        results = {}
        for strategy_name in request.strategies:
            try:
                strategy = get_di_strategy(strategy_name)
                strategy.set_di_series(di_data.df['decision_index'])

                strategy_key = f"compare_{strategy_name}_{id(strategy)}"
                backtesting_engine.add_strategy(strategy_key, strategy)

                try:
                    rebal_freq = RebalanceFrequency(request.rebalance_frequency)
                except ValueError:
                    rebal_freq = RebalanceFrequency.WEEKLY

                config = BacktestConfig(
                    start_date=request.start_date,
                    end_date=request.end_date,
                    initial_capital=request.initial_capital,
                    rebalance_frequency=rebal_freq
                )

                result = backtesting_engine.run_backtest(price_df, strategy_key, config)

                results[strategy_name] = {
                    "total_return_pct": round(result.metrics.get("total_return", 0) * 100, 2),
                    "annualized_return_pct": round(result.metrics.get("annualized_return", 0) * 100, 2),
                    "max_drawdown_pct": round(result.metrics.get("max_drawdown", 0) * 100, 2),
                    "sharpe_ratio": round(result.metrics.get("sharpe_ratio", 0), 3),
                    "sortino_ratio": round(result.metrics.get("sortino_ratio", 0), 3),
                    "win_rate_pct": round(result.metrics.get("win_rate", 0) * 100, 2),
                }

            except Exception as e:
                logger.warning(f"Erreur stratégie {strategy_name}: {e}")
                results[strategy_name] = {"error": str(e)}

        # Calculer rankings
        valid_results = {k: v for k, v in results.items() if "error" not in v}

        rankings = {}
        if valid_results:
            # Par return
            by_return = sorted(valid_results.items(), key=lambda x: x[1]["total_return_pct"], reverse=True)
            rankings["by_return"] = [x[0] for x in by_return]

            # Par Sharpe
            by_sharpe = sorted(valid_results.items(), key=lambda x: x[1]["sharpe_ratio"], reverse=True)
            rankings["by_sharpe"] = [x[0] for x in by_sharpe]

            # Par drawdown (moins négatif = mieux)
            by_dd = sorted(valid_results.items(), key=lambda x: x[1]["max_drawdown_pct"], reverse=True)
            rankings["by_drawdown"] = [x[0] for x in by_dd]

        return success_response({
            "period": f"{request.start_date} to {request.end_date}",
            "strategies_compared": len(request.strategies),
            "results": results,
            "rankings": rankings,
            "best_overall": rankings.get("by_sharpe", [None])[0]
        })

    except Exception as e:
        logger.error(f"Compare DI strategies failed: {e}", exc_info=True)
        return error_response(f"Comparaison échouée: {str(e)}", code=500)


@router.get("/period-analysis")
async def analyze_di_by_period(
    user: str = Depends(get_required_user),
    start_date: str = Query("2017-01-01"),
    end_date: Optional[str] = Query(None)
):
    """
    Analyse le comportement du DI sur les périodes clés (crashes, bull runs, etc.)
    """
    try:
        calculator = HistoricalDICalculator()
        di_data = await calculator.calculate_historical_di(
            user_id=user,
            start_date=start_date,
            end_date=end_date
        )

        # Analyser par période
        analysis = calculator.analyze_di_periods(di_data.df)

        return success_response({
            "total_period": {
                "start": di_data.start_date.strftime("%Y-%m-%d"),
                "end": di_data.end_date.strftime("%Y-%m-%d"),
                "days": len(di_data.df)
            },
            "period_analysis": analysis,
            "overall_stats": di_data.metadata.get("di_stats", {}),
        })

    except Exception as e:
        logger.error(f"Period analysis failed: {e}", exc_info=True)
        return error_response(f"Analyse échouée: {str(e)}", code=500)


# ========== Helper Functions ==========

def _get_strategy_description(key: str) -> str:
    """Retourne la description d'une stratégie"""
    descriptions = {
        "di_threshold": "Allocation basée sur seuils DI (DI<30→70% stables, DI>70→85% risky)",
        "di_momentum": "Suit la tendance du DI (hausse→augmente exposition, baisse→réduit)",
        "di_contrarian": "Stratégie contrarian (DI<20→accumulation, DI>80→prise profits)",
        "di_risk_parity": "Risk Parity + scaling DI (allocation inversement proportionnelle à la vol)",
        "di_signal": "Signaux purs (BUY quand DI croise 40↑, SELL quand croise 60↓)",
        "di_smartfolio_replica": "Production pipeline (risk_budget + exposure cap + governance penalty)",
        "di_trend_gate": "Trend Gate: BTC trend confirmation + DI threshold allocation",
        "di_cycle_rotation": "3-asset cycle rotation (BTC/ETH/Stables) based on 5 cycle phases",
        "di_adaptive_continuous": "Continuous DI→allocation mapping + golden/death cross trend confirmation (S9)",
    }
    return descriptions.get(key, "Stratégie basée sur le Decision Index")


# Descriptions détaillées pour le frontend
STRATEGY_DETAILS = {
    "di_threshold": {
        "name": "DI Threshold",
        "short": "Allocation par seuils de DI",
        "description": "Stratégie simple basée sur des niveaux de Decision Index. L'allocation entre actifs risqués (crypto) et stables change selon les seuils du DI.",
        "rules": [
            "DI < 20 (extreme fear) → 30% risky, 70% stables",
            "DI 20-40 (fear) → 50% risky, 50% stables",
            "DI 40-60 (neutral) → 60% risky, 40% stables",
            "DI 60-80 (greed) → 75% risky, 25% stables",
            "DI > 80 (extreme greed) → 85% risky, 15% stables"
        ],
        "pros": ["Simple à comprendre", "Réactive aux conditions de marché", "Suit le sentiment"],
        "cons": ["Peut être en retard sur les retournements", "Pas de lissage"],
        "best_for": "Validation de base du DI comme indicateur"
    },
    "di_momentum": {
        "name": "DI Momentum",
        "short": "Suit la tendance du DI",
        "description": "Ajuste l'exposition en fonction de la direction du DI sur les 7 derniers jours. Quand le DI monte fortement, on augmente l'exposition, et inversement.",
        "rules": [
            "Base: allocation du threshold strategy",
            "DI hausse >5pts/7j → +10% exposition risky",
            "DI baisse >5pts/7j → -10% exposition risky",
            "Renormalisation après ajustement"
        ],
        "pros": ["Suit les tendances", "Réagit aux changements rapides", "Combine base + momentum"],
        "cons": ["Peut amplifier la volatilité", "Whipsaws possibles"],
        "best_for": "Marchés tendanciels avec des mouvements clairs"
    },
    "di_contrarian": {
        "name": "DI Contrarian",
        "short": "Be greedy when others are fearful",
        "description": "Stratégie à contre-courant inspirée de Warren Buffett. Accumule agressivement en extreme fear et prend des profits en extreme greed.",
        "rules": [
            "DI < 20 (extreme fear) → 85% risky (accumulation max)",
            "DI 20-40 (fear) → 70% risky",
            "DI 40-70 (neutral) → 60% risky",
            "DI 70-80 (greed) → 45% risky",
            "DI > 80 (extreme greed) → 30% risky (prise profits)"
        ],
        "pros": ["Achète les dips", "Vend les sommets", "Discipline anti-FOMO"],
        "cons": ["Peut être prématuré", "Requiert patience", "Difficile psychologiquement"],
        "best_for": "Investisseurs long terme, marchés cycliques"
    },
    "di_risk_parity": {
        "name": "DI Risk Parity",
        "short": "Parité de risque ajustée par DI",
        "description": "Combine Risk Parity classique (allocation inversement proportionnelle à la volatilité) avec un scaling dynamique basé sur le DI.",
        "rules": [
            "Base: Inverse volatility weighting (30j lookback)",
            "Scale factor = 0.5 + (DI / 100)",
            "DI=0 → 50% de l'allocation risk parity",
            "DI=50 → 100% de l'allocation risk parity",
            "DI=100 → 150% de l'allocation risk parity"
        ],
        "pros": ["Gestion du risque intégrée", "Adaptatif à la volatilité", "DI comme booster"],
        "cons": ["Plus complexe", "Lookback peut être en retard", "Moins intuitif"],
        "best_for": "Gestion de risque sophistiquée, portefeuilles diversifiés"
    },
    "di_signal": {
        "name": "DI Signal",
        "short": "Signaux de trading purs",
        "description": "Stratégie de trading avec signaux discrets basés sur les croisements de seuils du DI, avec confirmation et holding minimum.",
        "rules": [
            "BUY: DI croise 40 à la hausse (confirmation 3 jours)",
            "SELL: DI croise 60 à la baisse",
            "Holding minimum: 14 jours après chaque signal",
            "Position long = 100% risky, neutral = 50/50"
        ],
        "pros": ["Signaux clairs", "Évite l'overtrading", "Confirmation intégrée"],
        "cons": ["Peut rater des opportunités", "Binaire (all-in ou not)", "Holding forcé"],
        "best_for": "Trading actif avec règles strictes"
    },
    "di_trend_gate": {
        "name": "Trend Gate (SMA200)",
        "short": "SMA(200) trend gate with optional DD breaker",
        "description": "Uses BTC SMA(200) as a trend gate: risk-on when price is above the 200-day moving average, risk-off below. Includes optional anti-whipsaw filter and drawdown circuit breaker for crash protection.",
        "rules": [
            "BTC > SMA(200) → risk-on: 80% risky allocation",
            "BTC < SMA(200) → risk-off: 20% risky allocation",
            "Anti-whipsaw: require N consecutive days on same side",
            "Optional drawdown breaker: cut allocation at -15%/-25% DD"
        ],
        "pros": ["Simple directional signal", "Clear regime detection", "Good Sharpe ratio (0.95+)"],
        "cons": ["2-asset only (BTC + Stables)", "SMA lag during fast crashes", "Whipsaw in ranging markets"],
        "best_for": "Directional trend-following with crash protection"
    },
    "di_smartfolio_replica": {
        "name": "SmartFolio Replica",
        "short": "Production risk_budget + exposure cap + governance penalty",
        "description": "Replicates the full production allocation pipeline: risk_budget formula + market overrides + exposure cap + governance-inspired contradiction penalty. Uses historical BTC volatility and score divergence to reconstruct contradiction_index.",
        "rules": [
            "blendedScore = 0.5×CycleScore + 0.3×OnChain + 0.2×RiskScore",
            "risk_factor = 0.5 + 0.5 × (RiskScore / 100) → [0.5, 1.0]",
            "baseRisky = clamp((blended - 35) / 45, 0, 1)",
            "risky = clamp(baseRisky × risk_factor, 20%, 85%)",
            "Override: |cycle - onchain| ≥ 30 → +10% stables",
            "Override: riskScore ≤ 30 → stables ≥ 50%",
            "Exposure cap: regime floor/ceiling + signal/vol penalties",
            "Governance penalty: contradiction (vol+cycle, DI vs cycle, score divergence) → 0-25% reduction"
        ],
        "pros": ["Full production pipeline (4 layers)", "Uses all 3 score components", "Regime-aware + governance-aware"],
        "cons": ["Simplified 2-asset model (BTC + Stables)", "Contradiction is proxy (no real ML signals)"],
        "best_for": "Validating real SmartFolio allocation behavior historically"
    },
    "di_cycle_rotation": {
        "name": "Cycle Rotation",
        "short": "3-asset rotation by cycle phase",
        "description": "Rotates between BTC, ETH, and Stablecoins based on 5 Bitcoin cycle phases. Uses cycle_score + cycle_direction to detect accumulation, bull building, peak, distribution, and bear phases. Each phase has target allocations derived from production ratios.",
        "rules": [
            "Accumulation (cycle<70, dir≥0): 50% BTC, 15% ETH, 35% Stables",
            "Bull Building (70-89, dir≥0): 35% BTC, 35% ETH, 30% Stables",
            "Peak (cycle≥90): 20% BTC, 20% ETH, 60% Stables",
            "Distribution (70-89, dir<0): 20% BTC, 10% ETH, 70% Stables",
            "Bear (cycle<70, dir<0): 15% BTC, 5% ETH, 80% Stables",
            "EMA smoothing (alpha=0.15) to avoid abrupt transitions",
            "Floor constraints: BTC≥10%, ETH≥5%, Stables≥10%"
        ],
        "pros": ["3-asset diversification", "Phase-aware rotation", "Smooth transitions via EMA", "Derived from production ratios"],
        "cons": ["Requires ETH price data (2017+)", "Phase detection depends on cycle accuracy", "More parameters to tune"],
        "best_for": "Validating multi-asset cycle rotation before applying to full 11-group production allocation"
    },
    "di_adaptive_continuous": {
        "name": "Adaptive Continuous (S9)",
        "short": "Continuous DI mapping + trend confirmation",
        "description": "Maps DI directly to a continuous allocation function (no discrete phases). Enriched with golden/death cross trend confirmation and direction-continuous asymmetric smoothing. Designed to improve bull capture by allowing higher allocation ceilings (85%) while maintaining bear protection via fast exit smoothing.",
        "rules": [
            "DI 0-25 → 10% risky (bear floor)",
            "DI 25-50 → 10-40% risky (gradual ramp)",
            "DI 50-75 → 40-70% risky (mid ramp)",
            "DI 75-100 → 70-85% risky (aggressive bull ramp)",
            "Golden cross (SMA50 > SMA200 + price > SMA200) → +10% boost",
            "Death cross (SMA50 < SMA200 + price < SMA200) → -10% reduction",
            "Direction-continuous smoothing: slow entry (α=0.12), fast exit (α=0.50)",
            "Continuous BTC/ETH split: ETH 20-40% of risky based on DI"
        ],
        "pros": ["No phase discontinuities", "Higher ceiling (85%) for bull capture", "Trend confirmation prevents false signals", "Uses DI directly (no parallel score)"],
        "cons": ["More parameters than Cycle Rotation", "Trend overlay adds ~200 days warmup", "Backtest-only (not production)"],
        "best_for": "Improving upside capture while maintaining bear protection"
    }
}


def _calculate_di_specific_metrics(di_df: pd.DataFrame, backtest_result) -> Dict:
    """Calcule des métriques spécifiques au DI (legacy)"""
    return _calculate_di_specific_metrics_v2(di_df, backtest_result)


def _calculate_di_specific_metrics_v2(di_df: pd.DataFrame, backtest_result) -> Dict:
    """Calcule des métriques spécifiques au DI pour DIBacktestResult"""
    di_series = di_df['decision_index']
    btc_returns = di_df['btc_price'].pct_change(30).shift(-30)  # Returns 30j futurs

    # Corrélation DI vs returns futurs
    correlation = di_series.corr(btc_returns)

    # Accuracy: DI > 50 et returns positifs, ou DI < 50 et returns négatifs
    di_high = di_series > 50
    returns_positive = btc_returns > 0
    accuracy = ((di_high & returns_positive) | (~di_high & ~returns_positive)).mean()

    # Stats DI pendant les gains/pertes
    gains_mask = btc_returns > 0.1  # > 10% gains
    losses_mask = btc_returns < -0.1  # > 10% pertes

    # Convertir les types numpy en types Python natifs pour la sérialisation JSON
    return {
        "di_btc_correlation_30d": round(float(correlation), 3) if not pd.isna(correlation) else None,
        "di_accuracy_pct": round(float(accuracy) * 100, 1) if not pd.isna(accuracy) else None,
        "avg_di_during_gains": round(float(di_series[gains_mask].mean()), 1) if gains_mask.any() else None,
        "avg_di_during_losses": round(float(di_series[losses_mask].mean()), 1) if losses_mask.any() else None,
        "di_mean": round(float(di_series.mean()), 1),
        "di_std": round(float(di_series.std()), 1),
        "di_min": round(float(di_series.min()), 1),
        "di_max": round(float(di_series.max()), 1),
    }

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
)
from services.di_backtest.trading_strategies import (
    DI_STRATEGIES,
    get_di_strategy,
    DIStrategyConfig,
)
from services.di_backtest.historical_di_calculator import DIWeights, PhaseFactors
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


class DIBacktestRequest(BaseModel):
    """Requête pour exécuter un backtest DI"""
    strategy: str = Field(..., description="Stratégie: di_threshold, di_momentum, di_contrarian, di_risk_parity, di_signal")
    start_date: str = Field(..., description="Date début YYYY-MM-DD (min: 2017-01-01)")
    end_date: str = Field(..., description="Date fin YYYY-MM-DD")
    initial_capital: float = Field(10000.0, gt=0)
    rebalance_frequency: str = Field("weekly", description="daily, weekly, monthly")

    # Custom weights (optionnel)
    di_weights: Optional[DIWeightsModel] = None

    # Options
    use_macro_penalty: bool = Field(True, description="Inclure pénalité VIX/DXY")

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
async def list_di_strategies():
    """Liste les stratégies DI disponibles pour le backtest"""
    strategies = []

    for key, strategy_class in DI_STRATEGIES.items():
        instance = strategy_class()
        strategies.append({
            "key": key,
            "name": instance.name,
            "description": _get_strategy_description(key)
        })

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
                "btc_price": float(point.btc_price) if point.btc_price else None
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

    Simule une stratégie de trading utilisant le DI sur des données historiques
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
        di_data = await calculator.calculate_historical_di(
            user_id=user,
            start_date=request.start_date,
            end_date=request.end_date,
            include_macro=request.use_macro_penalty
        )

        if di_data.df.empty:
            return error_response("Impossible de calculer le DI historique", code=500)

        # 2. Préparer les données de prix pour le backtest
        # On utilise BTC + USDT comme proxy simple
        btc_prices = di_data.df['btc_price'].copy()

        # S'assurer que l'index est un DatetimeIndex pandas
        if not isinstance(btc_prices.index, pd.DatetimeIndex):
            btc_prices.index = pd.to_datetime(btc_prices.index)

        logger.info(f"BTC prices: {len(btc_prices)} points, index type: {type(btc_prices.index)}")
        logger.info(f"Index range: {btc_prices.index.min()} to {btc_prices.index.max()}")

        # Créer un DataFrame avec BTC et un stable synthétique
        price_df = pd.DataFrame({
            'BTC': btc_prices.values,
            'STABLES': [1.0] * len(btc_prices)
        }, index=btc_prices.index)

        # Vérifier que le DataFrame est valide
        if price_df.empty:
            return error_response("Pas de données de prix disponibles pour cette période", code=400)

        logger.info(f"Price DataFrame: {len(price_df)} rows, columns: {list(price_df.columns)}")

        # 3. Créer la stratégie DI
        strategy_config = DIStrategyConfig()
        strategy = get_di_strategy(request.strategy, config=strategy_config)
        strategy.set_di_series(di_data.df['decision_index'])

        # Enregistrer dans le backtest engine
        strategy_key = f"di_{request.strategy}_{id(strategy)}"
        backtesting_engine.add_strategy(strategy_key, strategy)

        # 4. Configurer et exécuter le backtest
        try:
            rebal_freq = RebalanceFrequency(request.rebalance_frequency)
        except ValueError:
            rebal_freq = RebalanceFrequency.WEEKLY

        transaction_costs = TransactionCosts(
            maker_fee=request.transaction_fee_pct,
            taker_fee=request.transaction_fee_pct * 1.5,
            slippage_bps=request.slippage_bps,
            min_trade_size=10.0
        )

        # Ajuster les dates pour s'assurer qu'elles sont dans la plage du DataFrame
        df_start = price_df.index.min()
        df_end = price_df.index.max()

        config_start = pd.to_datetime(request.start_date)
        config_end = pd.to_datetime(request.end_date)

        # Ajuster si nécessaire
        if config_start < df_start:
            config_start = df_start
            logger.warning(f"Start date adjusted to {config_start}")
        if config_end > df_end:
            config_end = df_end
            logger.warning(f"End date adjusted to {config_end}")

        config = BacktestConfig(
            start_date=config_start.strftime("%Y-%m-%d"),
            end_date=config_end.strftime("%Y-%m-%d"),
            initial_capital=request.initial_capital,
            rebalance_frequency=rebal_freq,
            transaction_costs=transaction_costs,
            benchmark="BTC",
            risk_free_rate=0.02,
            max_position_size=0.95
        )

        logger.info(f"Running backtest from {config.start_date} to {config.end_date}")
        result = backtesting_engine.run_backtest(price_df, strategy_key, config)

        # 5. Calculer métriques additionnelles spécifiques au DI
        di_metrics = _calculate_di_specific_metrics(di_data.df, result)

        # 6. Formater la réponse
        monthly_returns = []
        for date, ret in result.monthly_returns.items():
            monthly_returns.append({
                "date": date.strftime("%Y-%m"),
                "return_pct": round(float(ret) * 100, 2)
            })

        equity_curve = []
        for date, value in result.portfolio_value.items():
            equity_curve.append({
                "date": date.strftime("%Y-%m-%d"),
                "portfolio_value": round(float(value), 2),
                "benchmark_value": round(float(result.benchmark_performance.get(date, request.initial_capital)), 2)
            })

        # Sous-échantillonner l'equity curve pour éviter trop de données
        if len(equity_curve) > 365:
            step = len(equity_curve) // 365
            equity_curve = equity_curve[::step]

        drawdowns = []
        for date, dd in result.drawdowns.items():
            drawdowns.append({
                "date": date.strftime("%Y-%m-%d"),
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
                "total_return_pct": round(result.metrics.get("total_return", 0) * 100, 2),
                "annualized_return_pct": round(result.metrics.get("annualized_return", 0) * 100, 2),
                "max_drawdown_pct": round(result.metrics.get("max_drawdown", 0) * 100, 2),
                "sharpe_ratio": round(result.metrics.get("sharpe_ratio", 0), 3),
                "sortino_ratio": round(result.metrics.get("sortino_ratio", 0), 3),
                "calmar_ratio": round(result.metrics.get("calmar_ratio", 0), 3),
                "volatility_pct": round(result.metrics.get("volatility", 0) * 100, 2),
                "win_rate_pct": round(result.metrics.get("win_rate", 0) * 100, 2),
            },
            "benchmark_comparison": {
                "benchmark": "BTC Buy & Hold",
                "benchmark_return_pct": round(result.summary.get("benchmark_return_pct", 0), 2),
                "excess_return_pct": round(result.summary.get("excess_return_pct", 0) * 100, 2),
            },
            "di_metrics": di_metrics,
            "equity_curve": equity_curve,
            "monthly_returns": monthly_returns,
            "drawdowns": drawdowns,
            "summary": result.summary
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
    }
    return descriptions.get(key, "Stratégie basée sur le Decision Index")


def _calculate_di_specific_metrics(di_df: pd.DataFrame, backtest_result) -> Dict:
    """Calcule des métriques spécifiques au DI"""
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

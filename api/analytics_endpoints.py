"""
API Endpoints pour l'analytics et l'historique

Ces endpoints gèrent l'historique des rebalancement et les analyses
de performance pour optimiser les stratégies.

✅ User Isolation: HistoryManager isolé par user_id (Dec 2025)
   - Chaque user a son propre fichier: data/users/{user_id}/rebalance_history.json
   - Factory function get_history_manager(user_id) utilisée dans tous les endpoints
   - Cache keys incluent user pour éviter cross-contamination
"""

from fastapi import APIRouter, Query, Body, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

from api.utils.formatters import success_response, error_response
from services.analytics.history_manager import get_history_manager, SessionStatus, PortfolioSnapshot
from services.analytics.performance_tracker import performance_tracker
from api.utils.cache import cache_get, cache_set, cache_clear_expired
from api.deps import get_required_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Cache pour les analytics (isolé par user_id dans les cache keys)
_analytics_cache = {}

# Models pour les requêtes/réponses
class PortfolioSnapshotRequest(BaseModel):
    """Requête pour créer un snapshot de portfolio"""
    total_usd: float = Field(..., description="Valeur totale en USD")
    allocations: Dict[str, float] = Field(..., description="Répartition par groupe en %")
    values_usd: Dict[str, float] = Field(..., description="Valeurs en USD par groupe")
    performance_24h_pct: Optional[float] = Field(default=None, description="Performance 24h")
    performance_7d_pct: Optional[float] = Field(default=None, description="Performance 7j")
    performance_30d_pct: Optional[float] = Field(default=None, description="Performance 30j")
    volatility_score: Optional[float] = Field(default=None, description="Score de volatilité")
    diversification_score: Optional[float] = Field(default=None, description="Score de diversification")

class SessionCreateRequest(BaseModel):
    """Requête pour créer une session de rebalancement"""
    target_allocations: Dict[str, float] = Field(..., description="Allocations cibles")
    source: str = Field(default="api", description="Source des données")
    pricing_mode: str = Field(default="auto", description="Mode de pricing")
    dynamic_targets_used: bool = Field(default=False, description="Utilise des targets dynamiques")
    ccs_score: Optional[float] = Field(default=None, description="Score CCS")
    min_trade_usd: float = Field(default=25.0, description="Montant minimum de trade")
    strategy_notes: str = Field(default="", description="Notes sur la stratégie")

class SessionResponse(BaseModel):
    """Réponse pour une session de rebalancement"""
    id: str
    created_at: str
    status: str
    target_allocations: Dict[str, float]
    dynamic_targets_used: bool
    ccs_score: Optional[float]
    total_planned_volume: float
    total_executed_volume: float
    execution_success_rate: float
    total_fees: float

class ExecutionResultRequest(BaseModel):
    """Requête pour mettre à jour les résultats d'exécution"""
    order_results: List[Dict[str, Any]] = Field(..., description="Résultats des ordres")

@router.post("/sessions")
async def create_rebalance_session(
    request: SessionCreateRequest,
    user: str = Depends(get_required_user)
):
    """
    Créer une nouvelle session de rebalancement

    Initialise une session pour tracker l'historique et les performances
    d'un rebalancement.
    """
    try:
        history_mgr = get_history_manager(user)
        session = history_mgr.create_session(
            target_allocations=request.target_allocations,
            source=request.source,
            pricing_mode=request.pricing_mode,
            dynamic_targets_used=request.dynamic_targets_used,
            ccs_score=request.ccs_score,
            min_trade_usd=request.min_trade_usd
        )
        
        # Ajouter les notes de stratégie
        session.strategy_notes = request.strategy_notes
        
        return {
            "success": True,
            "session_id": session.id,
            "message": "Rebalance session created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return error_response(str(e), code=500)

@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    user: str = Depends(get_required_user)
):
    """
    Obtenir les détails d'une session de rebalancement

    Retourne toutes les informations d'une session incluant
    les snapshots de portfolio et résultats d'exécution.
    """
    try:
        history_mgr = get_history_manager(user)
        session = history_mgr.get_session(session_id)
        
        if not session:
            return error_response("Session not found", code=404)
        
        # Convertir en réponse API
        response = SessionResponse(
            id=session.id,
            created_at=session.created_at.isoformat(),
            status=session.status.value,
            target_allocations=session.target_allocations,
            dynamic_targets_used=session.dynamic_targets_used,
            ccs_score=session.ccs_score,
            total_planned_volume=session.total_planned_volume,
            total_executed_volume=session.total_executed_volume,
            execution_success_rate=session.execution_success_rate,
            total_fees=session.total_fees
        )
        
        # Ajouter les détails complets
        session_dict = response.model_dump()
        session_dict.update({
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "strategy_notes": session.strategy_notes,
            "actions_count": len(session.actions),
            "portfolio_before": session.portfolio_before.__dict__ if session.portfolio_before else None,
            "portfolio_after": session.portfolio_after.__dict__ if session.portfolio_after else None,
            "performance_impact": session.calculate_performance_impact(),
            "error_message": session.error_message
        })
        
        return session_dict
        
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        return error_response(str(e), code=500)

@router.post("/sessions/{session_id}/portfolio-snapshot")
async def add_portfolio_snapshot(
    session_id: str,
    request: PortfolioSnapshotRequest,
    user: str = Depends(get_required_user),
    is_before: bool = Query(default=True, description="Snapshot avant rebalancement")
):
    """
    Ajouter un snapshot de portfolio à une session

    Capture l'état du portfolio avant ou après le rebalancement
    pour analyser l'impact.
    """
    try:
        # Créer le snapshot
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_usd=request.total_usd,
            allocations=request.allocations,
            values_usd=request.values_usd,
            performance_24h_pct=request.performance_24h_pct,
            performance_7d_pct=request.performance_7d_pct,
            performance_30d_pct=request.performance_30d_pct,
            volatility_score=request.volatility_score,
            diversification_score=request.diversification_score
        )

        history_mgr = get_history_manager(user)
        success = history_mgr.add_portfolio_snapshot(session_id, snapshot, is_before)
        
        if not success:
            return error_response("Session not found", code=404)
        
        return {
            "success": True,
            "message": f"Portfolio snapshot added ({'before' if is_before else 'after'} rebalancement)"
        }
        
    except Exception as e:
        logger.error(f"Error adding portfolio snapshot: {e}")
        return error_response(str(e), code=500)

@router.post("/sessions/{session_id}/actions")
async def add_rebalance_actions(
    session_id: str,
    user: str = Depends(get_required_user),
    actions: List[Dict[str, Any]] = Body(..., description="Actions de rebalancement")
):
    """
    Ajouter les actions de rebalancement à une session

    Enregistre les actions planifiées pour une session donnée.
    """
    try:
        history_mgr = get_history_manager(user)
        success = history_mgr.add_rebalance_actions(session_id, actions)
        
        if not success:
            return error_response("Session not found", code=404)
        
        return {
            "success": True,
            "message": f"Added {len(actions)} rebalance actions to session"
        }
        
    except Exception as e:
        logger.error(f"Error adding actions: {e}")
        return error_response(str(e), code=500)

@router.post("/sessions/{session_id}/execution-results")
async def update_execution_results(
    session_id: str,
    request: ExecutionResultRequest,
    user: str = Depends(get_required_user)
):
    """
    Mettre à jour les résultats d'exécution d'une session

    Met à jour les résultats réels d'exécution des ordres
    pour calculer les métriques de performance.
    """
    try:
        history_mgr = get_history_manager(user)
        success = history_mgr.update_execution_results(session_id, request.order_results)
        
        if not success:
            return error_response("Session not found", code=404)
        
        return {
            "success": True,
            "message": f"Updated execution results for {len(request.order_results)} orders"
        }
        
    except Exception as e:
        logger.error(f"Error updating execution results: {e}")
        return error_response(str(e), code=500)

@router.post("/sessions/{session_id}/complete")
async def complete_session(
    session_id: str,
    user: str = Depends(get_required_user),
    status: str = Body(default="completed", description="Statut final"),
    error_message: Optional[str] = Body(default=None, description="Message d'erreur")
):
    """
    Marquer une session comme terminée

    Finalise une session et déclenche la sauvegarde de l'historique.
    """
    try:
        # Valider le statut
        try:
            session_status = SessionStatus(status)
        except ValueError:
            return error_response(f"Invalid status: {status}", code=400)

        history_mgr = get_history_manager(user)
        success = history_mgr.complete_session(session_id, session_status, error_message)

        if not success:
            return error_response("Session not found", code=404)

        return success_response({
            "message": f"Session marked as {status}"
        })

    except Exception as e:
        logger.error(f"Error completing session: {e}")
        return error_response(str(e), code=500)

@router.get("/sessions")
async def get_sessions(
    limit: int = Query(default=50, le=100, description="Nombre maximum de sessions"),
    days_back: Optional[int] = Query(default=None, description="Nombre de jours en arrière"),
    status: Optional[str] = Query(default=None, description="Filtrer par statut"),
    user: str = Depends(get_required_user)
):
    """
    Obtenir la liste des sessions de rebalancement

    Retourne les sessions récentes avec possibilité de filtrage.
    """
    try:
        history_mgr = get_history_manager(user)
        sessions = history_mgr.get_recent_sessions(limit=limit, days_back=days_back)
        
        # Filtrer par statut si demandé
        if status:
            try:
                filter_status = SessionStatus(status)
                sessions = [s for s in sessions if s.status == filter_status]
            except ValueError:
                return error_response(f"Invalid status: {status}", code=400)

        # Convertir en réponses API
        session_responses = []
        for session in sessions:
            response = SessionResponse(
                id=session.id,
                created_at=session.created_at.isoformat(),
                status=session.status.value,
                target_allocations=session.target_allocations,
                dynamic_targets_used=session.dynamic_targets_used,
                ccs_score=session.ccs_score,
                total_planned_volume=session.total_planned_volume,
                total_executed_volume=session.total_executed_volume,
                execution_success_rate=session.execution_success_rate,
                total_fees=session.total_fees
            )
            session_responses.append(response)

        return success_response({
            "sessions": [r.model_dump() for r in session_responses],
            "total": len(session_responses),
            "filters": {
                "limit": limit,
                "days_back": days_back,
                "status": status
            }
        })

    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        return error_response(str(e), code=500)

@router.get("/performance/summary")
async def get_performance_summary(
    days_back: int = Query(default=30, ge=1, le=365, description="Période d'analyse en jours"),
    user: str = Depends(get_required_user)
):
    """
    Obtenir un résumé des performances

    Calcule les métriques de performance globales pour la période donnée.

    🔒 User Isolation: Cache key inclut user_id pour éviter cross-contamination
    """
    # 🔒 FIX: Inclure user dans cache key pour isolation multi-tenant
    cache_key = f"perf_summary:{user}:{days_back}"
    cached_result = cache_get(_analytics_cache, cache_key, 300)
    if cached_result:
        logger.info(f"Returning cached performance summary for user={user}, days={days_back}")
        return cached_result
    
    try:
        history_mgr = get_history_manager(user)
        summary = history_mgr.get_performance_summary(days_back=days_back)

        # Mettre en cache le résultat
        cache_set(_analytics_cache, cache_key, summary)
        cache_clear_expired(_analytics_cache, 300)

        return summary
        
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        return error_response(str(e), code=500)

@router.get("/performance/detailed")
async def get_detailed_performance_analysis(
    days_back: int = Query(default=30, ge=1, le=365),
    user: str = Depends(get_required_user)
):
    """
    Obtenir une analyse de performance détaillée

    Analyse avancée des performances avec comparaison des stratégies
    et recommandations d'optimisation.

    🔒 User Isolation: Cache key inclut user_id pour éviter cross-contamination
    """
    # 🔒 FIX: Inclure user dans cache key pour isolation multi-tenant
    cache_key = f"perf_detailed:{user}:{days_back}"
    cached_result = cache_get(_analytics_cache, cache_key, 600)
    if cached_result:
        logger.info(f"Returning cached detailed analysis for user={user}, days={days_back}")
        return cached_result
    
    try:
        # Obtenir les sessions pour la période
        history_mgr = get_history_manager(user)
        sessions = history_mgr.get_recent_sessions(days_back=days_back)

        if not sessions:
            return {"error": "No sessions found for the specified period"}

        # Analyser l'impact des rebalancement
        analysis = performance_tracker.analyze_rebalancing_impact(sessions)
        
        return {
            "analysis_period_days": days_back,
            "sessions_analyzed": len(sessions),
            "detailed_analysis": analysis,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting detailed performance analysis: {e}")
        return error_response(str(e), code=500)

@router.post("/performance/calculate")
async def calculate_portfolio_performance(
    value_history: List[Tuple[str, float]] = Body(..., description="Historique valeur portfolio [(timestamp, valeur)]"),
    benchmark_history: Optional[List[Tuple[str, float]]] = Body(default=None, description="Historique benchmark")
):
    """
    Calculer les métriques de performance d'un portfolio
    
    Calcule les métriques standard (return, volatilité, Sharpe, etc.)
    pour un historique de valeurs donné.
    """
    try:
        # Convertir les timestamps string en datetime
        converted_history = []
        for timestamp_str, value in value_history:
            timestamp = datetime.fromisoformat(timestamp_str)
            converted_history.append((timestamp, value))
        
        converted_benchmark = None
        if benchmark_history:
            converted_benchmark = []
            for timestamp_str, value in benchmark_history:
                timestamp = datetime.fromisoformat(timestamp_str)
                converted_benchmark.append((timestamp, value))
        
        # Calculer les métriques
        metrics = performance_tracker.calculate_portfolio_performance(
            converted_history, 
            converted_benchmark
        )
        
        return {
            "performance_metrics": {
                "period_start": metrics.period_start.isoformat(),
                "period_end": metrics.period_end.isoformat(),
                "total_return_pct": metrics.total_return_pct,
                "annualized_return_pct": metrics.annualized_return_pct,
                "volatility_pct": metrics.volatility_pct,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "sharpe_ratio": metrics.sharpe_ratio,
                "rebalancing_alpha_pct": metrics.rebalancing_alpha_pct
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating portfolio performance: {e}")
        return error_response(str(e), code=500)

@router.get("/reports/comprehensive")
async def generate_comprehensive_report(
    days_back: int = Query(default=30, ge=1, le=365, description="Période d'analyse"),
    include_portfolio_history: bool = Query(default=False, description="Inclure l'historique portfolio"),
    user: str = Depends(get_required_user)
):
    """
    Générer un rapport de performance complet

    Rapport détaillé incluant toutes les métriques, analyses de stratégies,
    et recommandations d'optimisation.
    """
    try:
        # Obtenir les sessions
        history_mgr = get_history_manager(user)
        sessions = history_mgr.get_recent_sessions(days_back=days_back)
        
        # Générer le rapport avec historique portfolio si demandé
        portfolio_history = None
        if include_portfolio_history:
            try:
                from api.unified_data import get_unified_filtered_balances
                # Récupérer l'historique portfolio des derniers jours
                portfolio_history = await _get_portfolio_history_data(days_back)
            except Exception as e:
                logger.warning(f"Could not retrieve portfolio history: {e}")
                portfolio_history = None
        
        report = performance_tracker.generate_performance_report(
            sessions=sessions,
            portfolio_history=portfolio_history
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {e}")
        return error_response(str(e), code=500)

@router.get("/optimization/recommendations")
async def get_optimization_recommendations(
    days_back: int = Query(default=30),
    user: str = Depends(get_required_user)
):
    """
    Obtenir des recommandations d'optimisation

    Analyse les performances récentes et suggère des améliorations
    pour les stratégies de rebalancement.
    """
    try:
        history_mgr = get_history_manager(user)
        sessions = history_mgr.get_recent_sessions(days_back=days_back)
        completed_sessions = [s for s in sessions if s.status == SessionStatus.COMPLETED]
        
        if not completed_sessions:
            return {"recommendations": ["Insufficient data - complete more rebalancing sessions"]}
        
        # Générer les recommandations via le performance tracker
        recommendations = performance_tracker._generate_recommendations(completed_sessions)
        
        # Ajouter des métriques de contexte
        avg_success_rate = sum(s.execution_success_rate for s in completed_sessions) / len(completed_sessions)
        total_volume = sum(s.total_executed_volume for s in completed_sessions)
        total_fees = sum(s.total_fees for s in completed_sessions)
        
        return {
            "analysis_period_days": days_back,
            "sessions_analyzed": len(completed_sessions),
            "context_metrics": {
                "avg_execution_success_rate": avg_success_rate,
                "total_volume_usd": total_volume,
                "total_fees_usd": total_fees,
                "avg_fee_rate_pct": (total_fees / total_volume * 100) if total_volume > 0 else 0
            },
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization recommendations: {e}")
        return error_response(str(e), code=500)


async def _get_portfolio_history_data(days_back: int) -> List[Dict[str, Any]]:
    """Récupérer l'historique du portfolio pour analyse"""
    try:
        from datetime import datetime, timedelta, timezone
        from api.unified_data import get_unified_filtered_balances
        
        portfolio_history = []
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        # Simuler des points d'historique (tous les jours)
        current_date = start_date
        while current_date <= end_date:
            try:
                # Pour une vraie implémentation, il faudrait une vraie base de données temporelle
                # Ici on simule avec les données actuelles comme approximation
                balances = await get_unified_filtered_balances()
                
                portfolio_snapshot = {
                    'timestamp': current_date.isoformat(),
                    'total_value_usd': sum(float(b.get('value_usd', 0)) for b in balances),
                    'asset_count': len(balances),
                    'assets': balances[:10]  # Limiter pour la performance
                }
                portfolio_history.append(portfolio_snapshot)
                
            except Exception as e:
                logger.warning(f"Could not get portfolio snapshot for {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        return portfolio_history

    except Exception as e:
        logger.error(f"Error retrieving portfolio history: {e}")
        return []


class MarketBreadthResponse(BaseModel):
    """Réponse pour l'analyse de largeur de marché"""
    advance_decline_ratio: float = Field(description="Ratio avance/déclin [0-1]")
    new_highs_count: int = Field(description="Nombre de nouveaux ATH récents")
    volume_concentration: float = Field(description="Concentration du volume [0-1]")
    momentum_dispersion: float = Field(description="Dispersion du momentum [0-1]")
    meta: Dict[str, Any] = Field(description="Métadonnées")


async def _fetch_global_market_data(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Récupère les données de marché global depuis CoinGecko (top N cryptos par market cap).

    Utilisé pour les métriques de market breadth - représente le marché global,
    pas un portefeuille utilisateur spécifique.

    Returns:
        Liste de cryptos avec price_change_percentage_24h, total_volume, market_cap, ath, etc.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": limit,
                "page": 1,
                "sparkline": "false",
                "price_change_percentage": "24h"
            }
            response = await client.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params=params
            )
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Fetched {len(data)} coins from CoinGecko for market breadth")
                return data
            else:
                logger.warning(f"CoinGecko API returned {response.status_code} for market breadth")
                return []
    except httpx.TimeoutException:
        logger.warning("CoinGecko API timeout for market breadth")
        return []
    except Exception as e:
        logger.error(f"Failed to fetch global market data: {e}")
        return []


@router.get("/market-breadth", response_model=MarketBreadthResponse)
async def get_market_breadth():
    """
    Analyse de largeur de marché (market breadth) - DONNÉES MARCHÉ GLOBAL

    Fournit des métriques sur la participation du marché crypto global (top 100):
    - Ratio avance/déclin: proportion de cryptos du marché en hausse vs déclin
    - Nouveaux ATH: nombre de cryptos proches de leurs ATH historiques
    - Concentration volume: degré de concentration du volume sur les top assets
    - Dispersion momentum: variabilité des performances (écart-type normalisé)

    Note: Ces métriques sont calculées sur les top 100 cryptos du marché global
    (par market cap), pas sur un portefeuille utilisateur spécifique.
    """
    try:
        logger.info("Calculating market breadth metrics from global market data")

        # Récupérer les données de marché global via CoinGecko
        market_data = await _fetch_global_market_data(limit=100)

        if not market_data:
            logger.warning("No global market data available for market breadth calculation")
            return MarketBreadthResponse(
                advance_decline_ratio=0.5,
                new_highs_count=0,
                volume_concentration=0.5,
                momentum_dispersion=0.5,
                meta={"status": "no_data", "source": "coingecko", "timestamp": datetime.now().isoformat()}
            )

        # 1. Calculer le ratio avance/déclin
        # Basé sur le changement de prix sur 24h
        advancing_assets = 0
        total_assets = 0

        for coin in market_data:
            change_24h = coin.get('price_change_percentage_24h')
            if change_24h is not None:
                total_assets += 1
                if change_24h > 0:
                    advancing_assets += 1

        advance_decline_ratio = advancing_assets / total_assets if total_assets > 0 else 0.5

        # 2. Calculer les nouveaux ATH
        # CoinGecko fournit ath et ath_change_percentage (distance from ATH)
        # On compte les cryptos à moins de 5% de leur ATH
        new_highs_count = 0
        for coin in market_data:
            ath_change = coin.get('ath_change_percentage')
            if ath_change is not None and ath_change >= -5:  # À moins de 5% de l'ATH
                new_highs_count += 1

        # 3. Concentration du volume
        # Calculer quelle proportion du volume total est captée par les top 10
        volumes = [float(coin.get('total_volume', 0) or 0) for coin in market_data]
        total_volume = sum(volumes)
        top_10_volume = sum(sorted(volumes, reverse=True)[:10])
        volume_concentration = top_10_volume / total_volume if total_volume > 0 else 0.5

        # 4. Dispersion du momentum
        # Calculer l'écart-type des rendements 24h comme mesure de dispersion
        returns_24h = [coin.get('price_change_percentage_24h', 0) or 0
                      for coin in market_data
                      if coin.get('price_change_percentage_24h') is not None]

        if len(returns_24h) > 1:
            import statistics
            std_dev = statistics.stdev(returns_24h)
            # Normaliser: écart-type de 10% = dispersion de 1.0
            momentum_dispersion = min(1.0, std_dev / 10.0)
        else:
            momentum_dispersion = 0.5

        result = MarketBreadthResponse(
            advance_decline_ratio=round(advance_decline_ratio, 3),
            new_highs_count=new_highs_count,
            volume_concentration=round(volume_concentration, 3),
            momentum_dispersion=round(momentum_dispersion, 3),
            meta={
                "assets_analyzed": len(market_data),
                "advancing_assets": advancing_assets,
                "declining_assets": total_assets - advancing_assets,
                "total_assets": total_assets,
                "timestamp": datetime.now().isoformat(),
                "source": "coingecko_global_top100"
            }
        )

        logger.info(f"Market breadth calculated: A/D={advance_decline_ratio:.3f}, "
                   f"New highs={new_highs_count}, Vol conc={volume_concentration:.3f}")

        return result

    except Exception as e:
        logger.error(f"Error calculating market breadth: {e}")
        return MarketBreadthResponse(
            advance_decline_ratio=0.5,
            new_highs_count=0,
            volume_concentration=0.5,
            momentum_dispersion=0.5,
            meta={"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}
        )
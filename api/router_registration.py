"""
Router registration for SmartFolio API

Extracted from api/main.py for better maintainability.
Imports and registers all API routers in the FastAPI application.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, FastAPI

logger = logging.getLogger(__name__)


def register_routers(app: FastAPI) -> None:
    """
    Register all API routers to the FastAPI application.

    Routers are organized by domain:
    - Authentication & Admin
    - Taxonomy & Classification
    - Execution & Monitoring
    - Analytics & ML
    - Risk Management
    - Portfolio & Performance
    - Market Data & Pricing
    - Wealth Management
    - Integrations (Saxo, Kraken, CoinGecko)
    - Utilities (Debug, Health, Config)

    Args:
        app: FastAPI application instance
    """
    logger.info("📦 Starting router registration...")

    # ========== Authentication & Admin ==========
    from api.admin_router import router as admin_router
    from api.auth_router import router as auth_router

    app.include_router(auth_router)  # Authentication (login/logout JWT)
    app.include_router(admin_router)  # Admin dashboard (RBAC protected)
    logger.info("✅ Auth & Admin routers registered")

    # ========== Taxonomy & Classification ==========
    from api.smart_taxonomy_endpoints import router as smart_taxonomy_router
    from api.taxonomy_endpoints import router as taxonomy_router

    app.include_router(taxonomy_router)
    app.include_router(smart_taxonomy_router)
    logger.info("✅ Taxonomy routers registered")

    # ========== Execution & Monitoring ==========
    # Execution endpoints - modular routers (Phase 2.1)
    from api.execution import (
        execution_router,
        governance_router,
        monitoring_router,
        signals_router,
        validation_router,
    )
    from api.execution_history import router as execution_history_router
    from api.monitoring_advanced import router as monitoring_advanced_router
    from api.portfolio_monitoring import router as portfolio_monitoring_router

    app.include_router(validation_router)
    app.include_router(execution_router)
    app.include_router(monitoring_router)
    app.include_router(governance_router)
    app.include_router(signals_router)
    app.include_router(execution_history_router)
    app.include_router(monitoring_advanced_router)
    app.include_router(portfolio_monitoring_router)
    logger.info("✅ Execution & Monitoring routers registered")

    # ========== Analytics & ML ==========
    from api.advanced_analytics_endpoints import router as advanced_analytics_router
    from api.analytics_endpoints import router as analytics_router
    from api.intelligence_endpoints import router as intelligence_router
    from api.ml_bourse_endpoints import router as ml_bourse_router
    from api.ml_crypto_endpoints import router as ml_crypto_router
    from api.unified_ml_endpoints import router as ml_router

    # Analytics router monté une seule fois avec prefix=/api/analytics
    app.include_router(analytics_router, prefix="/api")
    app.include_router(advanced_analytics_router)
    app.include_router(ml_router)
    app.include_router(ml_bourse_router)  # ML predictions pour Bourse/Saxo
    app.include_router(
        ml_crypto_router, prefix="/api/ml/crypto", tags=["ML Crypto"]
    )  # ML regime detection pour Bitcoin
    app.include_router(intelligence_router)
    logger.info("✅ Analytics & ML routers registered")

    # ========== ML Lazy Loading Router ==========
    # ML endpoints avec chargement ultra-lazy (pas d'import au démarrage)
    ml_router_lazy = APIRouter(prefix="/api/ml", tags=["ML (lazy)"])

    @ml_router_lazy.get("/status")
    async def get_ml_status_lazy():
        """Status ML avec chargement à la demande"""
        try:
            # Import seulement quand cette route est appelée
            from services.ml_pipeline_manager_optimized import (
                optimized_pipeline_manager as pipeline_manager,
            )

            status = pipeline_manager.get_pipeline_status()
            return {
                "pipeline_status": status,
                "timestamp": datetime.now().isoformat(),
                "loading_mode": "lazy",
            }
        except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError) as e:
            return {
                "error": "ML system not ready",
                "details": str(e),
                "status": "loading",
                "loading_mode": "lazy",
            }

    @ml_router_lazy.get("/health")
    async def ml_health_lazy():
        """Health check ML minimal sans imports lourds"""
        return {
            "status": "available",
            "message": "ML system ready for lazy loading",
            "timestamp": datetime.now().isoformat(),
        }

    app.include_router(ml_router_lazy)
    logger.info("✅ ML Lazy Loading router registered")

    # ========== Risk Management ==========
    from api.advanced_risk_endpoints import router as advanced_risk_router
    from api.risk_bourse_endpoints import router as risk_bourse_router
    from api.risk_endpoints import router as risk_router

    app.include_router(risk_router)
    app.include_router(risk_bourse_router)  # Risk management pour Bourse/Saxo
    app.include_router(advanced_risk_router)
    logger.info("✅ Risk Management routers registered")

    # ========== Portfolio & Performance ==========
    from api.advanced_rebalancing_endpoints import router as advanced_rebalancing_router
    from api.performance_endpoints import router as performance_router
    from api.portfolio_endpoints import router as portfolio_router
    from api.portfolio_optimization_endpoints import router as portfolio_optimization_router
    from api.rebalancing_strategy_router import router as rebalancing_strategy_router

    app.include_router(portfolio_router)
    app.include_router(portfolio_optimization_router)
    app.include_router(performance_router)
    app.include_router(advanced_rebalancing_router)
    app.include_router(rebalancing_strategy_router)
    logger.info("✅ Portfolio & Performance routers registered")

    # ========== Strategy & Backtesting ==========
    from api.backtesting_endpoints import router as backtesting_router
    from api.multi_asset_endpoints import router as multi_asset_router
    from api.strategy_endpoints import router as strategy_router

    app.include_router(strategy_router)
    app.include_router(backtesting_router)
    app.include_router(multi_asset_router)
    logger.info("✅ Strategy & Backtesting routers registered")

    # ========== Alerts & Real-time ==========
    from api.alerts_endpoints import router as alerts_router
    from api.realtime_endpoints import router as realtime_router

    app.include_router(alerts_router)
    app.include_router(realtime_router)
    logger.info("✅ Alerts & Real-time routers registered")

    # ========== Market Data & Pricing ==========
    from api.coingecko_proxy_router import router as coingecko_proxy_router
    from api.market_endpoints import router as market_router
    from api.pricing_router import router as pricing_router

    app.include_router(market_router)
    app.include_router(pricing_router)
    app.include_router(coingecko_proxy_router)  # CoinGecko CORS proxy with caching
    logger.info("✅ Market Data & Pricing routers registered")

    # ========== Wealth Management ==========
    from api.wealth_endpoints import router as wealth_router

    app.include_router(wealth_router)
    logger.info("✅ Wealth Management router registered")

    # ========== Integrations ==========
    # Saxo Bank
    from api.saxo_auth_router import router as saxo_auth_router
    from api.saxo_endpoints import router as saxo_router

    app.include_router(saxo_router)
    app.include_router(saxo_auth_router)  # Saxo OAuth2 authentication
    logger.info("✅ Saxo Bank routers registered")

    # Kraken
    from api.kraken_endpoints import router as kraken_router

    app.include_router(kraken_router)
    logger.info("✅ Kraken router registered")

    # ========== Data Sources ==========
    from api.csv_endpoints import router as csv_router
    from api.sources_endpoints import router as sources_router
    from api.sources_v2_endpoints import router as sources_v2_router

    app.include_router(sources_router)
    app.include_router(sources_v2_router)  # Sources V2 - category-based modular sources
    app.include_router(csv_router)
    logger.info("✅ Data Sources routers registered")

    # ========== FX & Currency ==========
    from api.fx_endpoints import router as fx_router

    app.include_router(fx_router)
    logger.info("✅ FX router registered")

    # ========== User Settings ==========
    from api.user_settings_endpoints import router as user_settings_router

    app.include_router(user_settings_router)
    logger.info("✅ User Settings router registered")

    # ========== Phase 3 Unified Orchestration ==========
    from api.unified_phase3_endpoints import router as unified_phase3_router

    app.include_router(unified_phase3_router)
    logger.info("✅ Phase 3 Unified Orchestration router registered")

    # ========== AI Chat ==========
    from api.ai_chat_router import router as ai_chat_router

    app.include_router(ai_chat_router)  # AI Chat with Groq (free tier)
    logger.info("✅ AI Chat router registered")

    # ========== Crypto Toolbox (Optional) ==========
    # Crypto-Toolbox router (native FastAPI with Playwright)
    try:
        from api.crypto_toolbox_endpoints import router as crypto_toolbox_router

        app.include_router(crypto_toolbox_router)
        logger.info("🎭 Crypto-Toolbox: FastAPI native scraper enabled")
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"❌ Failed to load crypto_toolbox router: {e}")
        logger.warning("⚠️  Crypto-toolbox endpoints will not be available")

    # ========== Utilities ==========
    from api.config_router import router as config_router
    from api.debug_router import router as debug_router
    from api.health_router import router as health_router

    app.include_router(debug_router)
    app.include_router(health_router)
    app.include_router(config_router)
    logger.info("✅ Utility routers registered (debug, health, config)")

    # Test simple endpoint pour debugging ML pipeline
    @app.get("/api/ml/pipeline/test")
    async def test_pipeline():
        return {"message": "Pipeline API is working!"}

    logger.info("🎯 All routers registered successfully")

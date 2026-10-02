"""
API Startup & Shutdown Handlers
Extracted from api/main.py (Phase 2B refactoring)

Handles:
- ML models lazy loading
- Governance Engine initialization
- Alert Engine initialization
- Background tasks orchestration
"""

import asyncio
import logging

from fastapi import HTTPException

logger = logging.getLogger(__name__)


async def initialize_ml_models():
    """Report loaded artifacts without forcing any readiness or training."""
    from services.ml.reliability import capability_service
    count = sum(entry["loaded"] for entry in capability_service.catalog())
    logger.info("ML artifacts actually loaded: %s; inference remains on demand", count)
    return count


async def initialize_governance_engine():
    """
    Initialize Governance Engine with ML signals.

    Returns:
        bool: True if initialized successfully
    """
    try:
        from services.execution.governance import governance_engine
        await governance_engine._refresh_ml_signals()

        # Verify signals are loaded
        signals = governance_engine.current_state.signals
        if signals and signals.confidence > 0:
            logger.info(
                f"[OK] Governance Engine initialized: "
                f"{signals.confidence:.1%} confidence, "
                f"{len(signals.sources_used)} sources"
            )
            return True
        else:
            logger.warning(" Governance Engine initialized but signals may be empty")
            return False

    except Exception as e:
        logger.error(f" Governance Engine initialization failed: {e}")
        return False


async def initialize_alert_engine():
    """
    Initialize Alert Engine with scheduler.

    Returns:
        bool: True if initialized successfully
    """
    try:
        import os
        from services.alerts.alert_engine import AlertEngine
        from services.execution.governance import governance_engine
        from api.alerts_endpoints import initialize_alert_engine as init_alert_api

        # Get Redis URL from environment
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # Create AlertEngine instance with governance engine reference and Redis
        alert_engine = AlertEngine(
            governance_engine=governance_engine,
            config_file_path="config/alerts_rules.json",
            redis_url=redis_url
        )

        # Initialize AlertEngine for API endpoints
        init_alert_api(alert_engine)

        # Initialize unified facade for legacy systems
        from services.alerts.unified_alert_facade import get_unified_alert_facade
        unified_facade = get_unified_alert_facade(alert_engine)
        logger.info(" Unified alert facade initialized for legacy system migration")

        # Start alert scheduler in background
        scheduler_started = await alert_engine.start()

        if scheduler_started:
            logger.info(" AlertEngine scheduler started successfully")
        else:
            logger.info(" AlertEngine initialized in standby mode (scheduler locked by another instance)")

        return scheduler_started

    except Exception as alert_error:
        logger.error(f" AlertEngine initialization failed: {alert_error}")
        # Don't crash the app, system can work without alerts
        return False


async def initialize_playwright_browser():
    """
    Initialize Playwright browser for crypto-toolbox scraping (optional, disabled by default).

    Returns:
        bool: True if initialized successfully

    Note:
        - Only initializes if crypto_toolbox router is enabled in api/main.py
        - Browser launched in headless mode, shared across requests
        - Auto-recovery on crash (lazy re-launch)
        - Memory: ~150-200 MB (Chromium process)
    """
    try:
        # Check if crypto_toolbox module is available
        try:
            from api.crypto_toolbox_endpoints import startup_playwright
        except ImportError:
            logger.debug(" crypto_toolbox_endpoints not available, skipping Playwright init")
            return False

        logger.info(" Initializing Playwright browser for crypto-toolbox scraping...")
        await startup_playwright()
        logger.info(" Playwright browser initialized successfully (~200 MB memory)")

        return True

    except Exception as e:
        logger.warning(f" Playwright initialization failed (non-blocking): {e}")
        logger.info(" crypto-toolbox endpoints will use lazy browser launch on first request")
        # Don't crash the app, browser will be launched on first request
        return False


async def initialize_task_scheduler():
    """
    Initialize APScheduler for periodic tasks (P&L snapshots, OHLCV updates, warmers).

    Returns:
        bool: True if scheduler started successfully

    Note:
        - Only starts if RUN_SCHEDULER=1 environment variable is set
        - Prevents double execution in dev with --reload
        - Includes P&L snapshots (intraday 15min, EOD 23:59)
        - OHLCV updates (daily 03:10, hourly :05)
        - Staleness monitoring (hourly)
        - API warmers (every 10 min)
    """
    try:
        from api.scheduler import initialize_scheduler

        scheduler_ok = await initialize_scheduler()

        if scheduler_ok:
            logger.info(" Task scheduler initialized successfully")
        else:
            logger.info(" Task scheduler disabled (RUN_SCHEDULER != 1)")

        return scheduler_ok

    except Exception as e:
        logger.warning(f" Task scheduler initialization failed (non-blocking): {e}")
        logger.info(" Periodic tasks will not run automatically")
        return False


async def initialize_ml_auto_trainer():
    """Training is restricted to explicit administrator evaluations."""
    logger.info("ML auto-training disabled: use explicit administrator actions")
    return False


async def background_startup_tasks():
    """
    Background task to initialize ML models, Governance, and Alerts.
    Runs after a 3-second delay to let the app fully start.
    """
    try:
        # Wait for app to fully start
        await asyncio.sleep(3)

        logger.info(" Starting background ML models initialization...")

        # Initialize ML models
        models_count = await initialize_ml_models()

        # Initialize Governance Engine
        governance_ok = await initialize_governance_engine()

        # Initialize Alert Engine
        alerts_ok = await initialize_alert_engine()

        # Initialize Playwright (optional, for crypto-toolbox scraping)
        # Note: Browser not launched unless router enabled in api/main.py
        playwright_ok = await initialize_playwright_browser()

        # Initialize Task Scheduler (optional, periodic tasks)
        # Note: Only starts if RUN_SCHEDULER=1
        scheduler_ok = await initialize_task_scheduler()

        # Initialize ML Auto-Trainer (optional, automatic model retraining)
        # Note: Only starts if ML_AUTO_TRAIN=1 (default: enabled)
        ml_auto_trainer_ok = await initialize_ml_auto_trainer()

        logger.info(
            f" Startup complete: "
            f"ML={models_count} models, "
            f"Governance={'OK' if governance_ok else 'Warning'}, "
            f"Alerts={'OK' if alerts_ok else 'Warning'}, "
            f"Playwright={'OK' if playwright_ok else 'Next'}, "
            f"Scheduler={'OK' if scheduler_ok else 'Paused'}, "
            f"ML-AutoTrain={'OK' if ml_auto_trainer_ok else 'Paused'}"
        )

    except Exception as e:
        logger.info(f" Background loading failed, models will load on demand: {e}")


def get_startup_handler():
    """
    Returns the startup event handler for FastAPI.

    Usage:
        @app.on_event("startup")
        async def startup():
            await get_startup_handler()()
    """
    async def startup_load_ml_models():
        """Lazy loading of ML models to avoid blocking startup"""
        try:
            logger.info(" FastAPI started successfully")
            logger.info(" ML models will load on first request (lazy loading)")

            # Initialize FX rates (fast, synchronous)
            try:
                from services.fx_service import initialize_rates
                initialize_rates()
            except Exception as fx_error:
                logger.warning(f" FX rates initialization failed (non-blocking): {fx_error}")

            # Start background task without waiting
            asyncio.create_task(background_startup_tasks())

        except Exception as e:
            logger.warning(f" Startup event warning (non-blocking): {e}")
            # Don't crash the app

    return startup_load_ml_models


def get_shutdown_handler():
    """
    Returns the shutdown event handler for FastAPI.

    Usage:
        @app.on_event("shutdown")
        async def shutdown():
            await get_shutdown_handler()()
    """
    async def shutdown_cleanup():
        """Cleanup tasks on shutdown"""
        try:
            logger.info(" Shutting down FastAPI application...")

            # Stop alert scheduler if running
            try:
                from api.alerts_endpoints import get_alert_engine
                alert_engine = get_alert_engine()
                if alert_engine:
                    await alert_engine.stop()
                    logger.info(" AlertEngine scheduler stopped")
            except HTTPException as e:
                # 503 = Alert engine not initialized, normal during early shutdown
                if e.status_code != 503:
                    logger.warning(f" Alert engine cleanup failed: {e}")
            except Exception as e:
                logger.warning(f" Alert engine cleanup failed: {e}")

            # Stop task scheduler if running
            try:
                from api.scheduler import shutdown_scheduler
                await shutdown_scheduler()
            except Exception as e:
                logger.warning(f" Task scheduler cleanup failed: {e}")

            # Stop ML Auto-Trainer if running
            try:
                from services.ml.auto_trainer import ml_auto_trainer
                ml_auto_trainer.stop()
                logger.info(" ML Auto-Trainer stopped")
            except Exception as e:
                logger.warning(f" ML Auto-Trainer cleanup failed: {e}")

            # Close Playwright browser if initialized
            try:
                from api.crypto_toolbox_endpoints import shutdown_playwright
                await shutdown_playwright()
                # Note: shutdown_playwright() logs its own status
            except ImportError:
                pass  # Module not loaded, nothing to clean up
            except Exception as e:
                logger.warning(f" Playwright cleanup failed: {e}")

            logger.info(" Shutdown complete")

        except Exception as e:
            logger.error(f" Shutdown error: {e}")

    return shutdown_cleanup

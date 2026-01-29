"""
Static files configuration for SmartFolio API

Extracted from api/main.py for better maintainability.
Configures static file serving: /static, /data, /config, /tests directories.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


def setup_static_files(app: FastAPI, debug: bool = False) -> None:
    """
    Configure static file serving for the FastAPI application.

    Mounts:
    - /static -> static/ (HTML, CSS, JS)
    - /data -> data/ (CSV files, portfolio data)
    - /config -> config/ (users.json, etc.)
    - /tests -> tests/ (test pages, debug mode only)

    Args:
        app: FastAPI application instance
        debug: Enable debug mode to mount test files
    """
    BASE_DIR = (
        Path(__file__).resolve().parent.parent
    )  # répertoire du repo (niveau au-dessus d'api/)
    STATIC_DIR = BASE_DIR / "static"  # D:\Python\smartfolio\static
    DATA_DIR = BASE_DIR / "data"  # D:\Python\smartfolio\data

    logger.debug(f"BASE_DIR = {BASE_DIR}")
    logger.debug(f"STATIC_DIR = {STATIC_DIR}, exists = {STATIC_DIR.exists()}")
    logger.debug(f"DATA_DIR = {DATA_DIR}, exists = {DATA_DIR.exists()}")

    # ========== Fallback Paths ==========
    if not STATIC_DIR.exists():
        logger.warning("STATIC_DIR not found, using fallback")
        # fallback si l'arbo a changé
        STATIC_DIR = Path.cwd() / "static"

    if not DATA_DIR.exists():
        logger.warning("DATA_DIR not found, using fallback")
        DATA_DIR = Path.cwd() / "data"

    logger.debug(f"Final STATIC_DIR = {STATIC_DIR}")
    logger.debug(f"Final DATA_DIR = {DATA_DIR}")

    # Vérifier le fichier CSV spécifiquement
    csv_file = DATA_DIR / "raw" / "CoinTracking - Current Balance.csv"
    logger.debug(f"CSV file = {csv_file}, exists = {csv_file.exists()}")

    # ========== Mount Static Directory ==========
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR), html=True),
        name="static",
    )
    logger.info(f"✅ Static files mounted: /static -> {STATIC_DIR}")

    # ========== Mount Data Directory ==========
    # Mount data directory for CSV access (nécessaire en production pour les dashboards)
    app.mount(
        "/data",
        StaticFiles(directory=str(DATA_DIR)),
        name="data",
    )
    logger.info(f"✅ Data files mounted: /data -> {DATA_DIR}")

    # ========== Mount Config Directory ==========
    # Mount config directory for users.json access
    CONFIG_DIR = BASE_DIR / "config"
    if CONFIG_DIR.exists():
        app.mount(
            "/config",
            StaticFiles(directory=str(CONFIG_DIR)),
            name="config",
        )
        logger.info(f"✅ Config files mounted: /config -> {CONFIG_DIR}")
    else:
        logger.warning(f"⚠️  Config directory not found: {CONFIG_DIR}")

    # ========== Mount Tests Directory (Debug Only) ==========
    # Optionnel: exposer les pages de test HTML en local (sécurisé par DEBUG)
    if debug:
        try:
            TESTS_DIR = BASE_DIR / "tests"
            if TESTS_DIR.exists():
                logger.debug(f"Mounting TESTS_DIR at /tests -> {TESTS_DIR}")
                app.mount(
                    "/tests",
                    StaticFiles(directory=str(TESTS_DIR), html=True),
                    name="tests",
                )
                logger.info(f"✅ Test files mounted (debug mode): /tests -> {TESTS_DIR}")
        except (OSError, RuntimeError) as e:
            logger.warning(f"Could not mount /tests: {e}")

    logger.info("🎯 All static files configured successfully")

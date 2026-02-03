from __future__ import annotations

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Response

from api.services.cointracking_helpers import load_ctapi_exchanges
from api.services.csv_helpers import to_csv
from api.services.location_assigner import assign_locations_to_actions
from api.services.price_enricher import enrich_actions_with_prices
from api.services.utils import norm_primary_symbols, parse_min_usd

# Charger les variables d'environnement depuis .env
load_dotenv()

# Fix joblib/loky Windows encoding issue with Python 3.13
# Set before any scikit-learn imports to avoid wmic auto-detection errors
if not os.getenv("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Configuration centralisée avec Pydantic
from config import get_settings

settings = get_settings()

# Variables de compatibilité (pour ne pas casser le code existant)
DEBUG = settings.is_debug_enabled()
APP_DEBUG = DEBUG
LOG_LEVEL = settings.logging.log_level
CORS_ORIGINS = settings.get_cors_origins()
ENVIRONMENT = settings.environment
# Par défaut, on désactive les stubs pour éviter de masquer des erreurs de config.
ALLOW_STUB_SOURCES = os.getenv("ALLOW_STUB_SOURCES", "false").strip().lower() == "true"
COMPUTE_ON_STUB_SOURCES = os.getenv("COMPUTE_ON_STUB_SOURCES", "false").strip().lower() == "true"

# Config logger (dev-friendly by default) — initialize early so it's available in imports below
# Créer le dossier logs s'il n'existe pas
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configuration du logging avec handlers multiples (console + fichier rotatif)
log_level = getattr(logging, LOG_LEVEL, logging.INFO)
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Configuration avec RotatingFileHandler pour limiter la taille des logs
logging.basicConfig(
    level=log_level,
    format=log_format,
    handlers=[
        # Console (stdout) - pour le terminal
        logging.StreamHandler(),
        # Fichier rotatif - 5 MB par fichier, 3 backups (15 MB total)
        # Adapté pour Claude Code: fichiers de taille raisonnable
        RotatingFileHandler(
            LOG_DIR / "app.log",
            maxBytes=5 * 1024 * 1024,  # 5 MB par fichier (facile à lire pour une IA)
            backupCount=3,  # Garder 3 fichiers de backup (15 MB total max)
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("crypto-rebalancer")
logger.info(
    f"📝 Logging initialized: console + file (rotating 5MB x3 backups) -> {LOG_DIR / 'app.log'}"
)

# Import différé des connecteurs pour éviter les blocages réseau au démarrage
# from connectors import cointracking as ct_file
# from connectors.cointracking_api import get_current_balances as ct_api_get_current_balances, _debug_probe

# Imports avec fallback pour éviter les crashs
try:
    from services.rebalance import plan_rebalance

    REBALANCE_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Rebalance service not available: {e}")
    REBALANCE_AVAILABLE = False

# Imports with fallback removed - services available via dependency injection

from api.deps import get_required_user
from api.exception_handlers import setup_exception_handlers

# Import modular configuration (Phase 2.1 - Refactoring)
from api.middleware_setup import setup_middlewares
from api.router_registration import register_routers
from api.static_files_setup import setup_static_files

# Import BalanceService singleton for resolving balances
from services.balance_service import balance_service

# Logger already configured above

app = FastAPI(docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")
logger.info("FastAPI initialized: docs=%s redoc=%s openapi=%s", "/docs", "/redoc", "/openapi.json")

# /metrics Prometheus (activable en prod via variable d'environnement)
if os.getenv("ENABLE_METRICS", "0") == "1":
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app, include_in_schema=False)
    except (ImportError, ModuleNotFoundError) as e:
        logging.getLogger(__name__).warning("Prometheus non activé: %s", e)

# Startup handlers (refactored to api/startup.py)
from api.startup import get_shutdown_handler, get_startup_handler


@app.on_event("startup")
async def startup():
    """Application startup - initialize ML, Governance, Alerts"""
    handler = get_startup_handler()
    await handler()


@app.on_event("shutdown")
async def shutdown():
    """Application shutdown - cleanup resources"""
    handler = get_shutdown_handler()
    await handler()


# ========== Exception Handlers (Modular) ==========
# All exception handlers configured in api/exception_handlers.py for maintainability
setup_exception_handlers(app)

# ========== Middleware Setup (Modular) ==========
# All middlewares configured in api/middleware_setup.py for maintainability
setup_middlewares(
    app=app, settings=settings, debug=DEBUG, environment=ENVIRONMENT, cors_origins=CORS_ORIGINS
)

# ========== Static Files Setup (Modular) ==========
# All static file mounts configured in api/static_files_setup.py for maintainability
setup_static_files(app, debug=DEBUG)

# >>> BEGIN: CT-API helpers (centralized constants) >>>
try:
    from connectors import cointracking_api as ct_api
except ImportError as e:
    logger.warning(f"Could not import from connectors package: {e}")
    try:
        import cointracking_api as ct_api  # fallback au cas où le package n'est pas packagé "connectors"

        logger.info("Using fallback import for cointracking_api")
    except ImportError as fallback_error:
        logger.error(f"Could not import cointracking_api at all: {fallback_error}")
        ct_api = None
    else:
        # Fallback OK: ne pas écraser ct_api
        pass
except (SyntaxError, AttributeError, RuntimeError, TypeError, ValueError) as e:
    logger.error(f"Error importing cointracking_api: {e}", exc_info=True)
    ct_api = None

# CSV/API cointracking facade (safe import)
try:
    from connectors import cointracking as ct_file
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"CoinTracking CSV/API facade not available: {e}")
    ct_file = None

# CoinTracking helpers moved to api/services/cointracking_helpers.py
# - normalize_loc
# - classify_location
# - pick_primary_location_for_symbol
# - load_ctapi_exchanges

# ---------- utils ----------
# Utility functions moved to api/services/utils.py
# - parse_min_usd
# - to_rows
# - norm_primary_symbols

# _get_data_age_minutes moved to api/services/price_enricher.py
# _calculate_price_deviation removed (dead code, never called)


# ---------- source resolver ----------
# CSV helper moved to api/services/csv_helpers.py
# - load_csv_balances


async def resolve_current_balances(
    source: str = Query("cointracking_api"), user: str = Depends(get_required_user)
) -> Dict[str, Any]:
    """
    Retourne {source_used, items:[{symbol, alias, amount, value_usd, location}]}
    Utilise UserDataRouter pour router les données par utilisateur.

    NOTE: This function now delegates to services.balance_service.BalanceService
    for better separation of concerns and to break circular dependencies.

    The actual implementation is in services/balance_service.py
    """
    # Delegate to BalanceService to break circular dependencies
    return await balance_service.resolve_current_balances(source=source, user_id=user)


# Legacy implementation moved to services/balance_service.py for better separation
# This eliminates circular dependencies with AlertEngine and other modules

# _assign_locations_to_actions moved to api/services/location_assigner.py

# Helper function moved to unified_data.py to avoid circular imports

# Debug endpoint removed


# ---------- balances ----------
@app.get("/balances/current")
async def balances_current(
    source: str = Query("cointracking"),
    min_usd: float = Query(1.0),
    user: str = Depends(get_required_user),
):
    from api.unified_data import get_unified_filtered_balances

    return await get_unified_filtered_balances(source=source, min_usd=min_usd, user_id=user)


# ---------- rebalance (JSON) ----------
@app.post("/rebalance/plan")
async def rebalance_plan(
    source: str = Query("cointracking"),
    min_usd_raw: str | None = Query(None, alias="min_usd"),
    pricing: str = Query("local"),  # local | auto
    dynamic_targets: bool = Query(False, description="Use dynamic targets from CCS/cycle module"),
    payload: Dict[str, Any] = Body(...),
    pricing_diag: bool = Query(
        False, description="Include pricing diagnostic details in response meta"
    ),
    user: str = Depends(get_required_user),
):
    min_usd = parse_min_usd(min_usd_raw, default=1.0)

    # portefeuille - utiliser la fonction helper unifiée
    from api.unified_data import get_unified_filtered_balances

    unified_data = await get_unified_filtered_balances(source=source, min_usd=min_usd, user_id=user)
    rows = unified_data.get("items", [])

    # targets - support for dynamic CCS-based targets
    if dynamic_targets and payload.get("dynamic_targets_pct"):
        # CCS/cycle module provides pre-calculated targets
        targets_raw = payload.get("dynamic_targets_pct", {})
        group_targets_pct = {str(k): float(v) for k, v in targets_raw.items()}
    else:
        # Standard targets from user input
        targets_raw = (
            payload.get("group_targets_pct")
            or payload.get("targets")
            or payload.get("target_allocations")
            or {}
        )
        group_targets_pct: Dict[str, float] = {}
        if isinstance(targets_raw, dict):
            group_targets_pct = {str(k): float(v) for k, v in targets_raw.items()}
        elif isinstance(targets_raw, list):
            for it in targets_raw:
                g = str(it.get("group"))
                p = float(it.get("weight_pct", 0.0))
                if g:
                    group_targets_pct[g] = p

    primary_symbols = norm_primary_symbols(payload.get("primary_symbols"))

    # Permettre un fallback: "pricing_diag" dans le body JSON si non passé en query
    if not pricing_diag:
        try:
            pricing_diag = bool(payload.get("pricing_diag", False))
        except (ValueError, TypeError, KeyError):
            pricing_diag = False

    plan = plan_rebalance(
        rows=rows,
        group_targets_pct=group_targets_pct,
        min_usd=min_usd,
        sub_allocation=payload.get("sub_allocation", "proportional"),
        primary_symbols=primary_symbols,
        min_trade_usd=float(payload.get("min_trade_usd", 25.0)),
    )

    logger.debug(
        f"🔧 BEFORE assign_locations_to_actions: plan has {len(plan.get('actions', []))} actions"
    )
    plan = assign_locations_to_actions(
        plan, rows, min_trade_usd=float(payload.get("min_trade_usd", 25.0))
    )
    logger.debug(
        f"🔧 AFTER assign_locations_to_actions: plan has {len(plan.get('actions', []))} actions"
    )

    # enrichissement prix (selon "pricing")
    source_used = unified_data.get("source_used", source)
    plan = await enrich_actions_with_prices(
        plan, rows, pricing_mode=pricing, source_used=source_used, diagnostic=pricing_diag
    )

    # Mettre à jour les exec_hints basés sur les locations assignées (après enrichissement prix)
    from services.rebalance import _format_hint_for_location, _get_exec_hint

    # Créer un index des holdings par groupe pour les actions sans location
    holdings_by_group = {}
    for row in rows:
        group = row.get("group")
        if not group:
            continue
        if group not in holdings_by_group:
            holdings_by_group[group] = []
        holdings_by_group[group].append(row)

    for action in plan.get("actions", []):
        location = action.get("location")
        action_type = action.get("action", "")

        if location and location not in ["Unknown", ""]:
            # Action avec location spécifique - utiliser la nouvelle logique
            action["exec_hint"] = _format_hint_for_location(location, action_type)
        else:
            # Action sans location spécifique - utiliser l'ancienne logique comme fallback
            group = action.get("group", "")
            group_items = holdings_by_group.get(group, [])
            action["exec_hint"] = _get_exec_hint(action, {group: group_items})

    # meta pour UI - fusionner avec les métadonnées pricing existantes
    if not plan.get("meta"):
        plan["meta"] = {}
    # Préserver les métadonnées existantes et ajouter les nouvelles
    meta_update = {"source_used": source_used, "items_count": len(rows)}
    plan["meta"].update(meta_update)

    # Mettre à jour le cache des unknown aliases pour les suggestions automatiques
    unknown_aliases = plan.get("unknown_aliases", [])
    if unknown_aliases:
        try:
            from api.taxonomy_endpoints import update_unknown_aliases_cache

            update_unknown_aliases_cache(unknown_aliases)
        except ImportError:
            pass  # Ignore si pas disponible

    return plan


# ---------- rebalance (CSV) ----------
@app.options("/rebalance/plan.csv")
async def rebalance_plan_csv_preflight():
    # pour laisser passer les preflight CORS
    return Response(status_code=200)


@app.post("/rebalance/plan.csv")
async def rebalance_plan_csv(
    source: str = Query("cointracking"),
    min_usd_raw: str | None = Query(None, alias="min_usd"),
    pricing: str = Query("local"),
    dynamic_targets: bool = Query(False, description="Use dynamic targets from CCS/cycle module"),
    payload: Dict[str, Any] = Body(...),
):
    # réutilise le JSON pour construire le CSV
    plan = await rebalance_plan(
        source=source,
        min_usd_raw=min_usd_raw,
        pricing=pricing,
        dynamic_targets=dynamic_targets,
        payload=payload,
    )
    actions = plan.get("actions") or []
    csv_text = to_csv(actions)
    headers = {"Content-Disposition": 'attachment; filename="rebalance-actions.csv"'}
    return Response(content=csv_text, media_type="text/csv", headers=headers)


# ---------- helpers prix + csv ----------
# _enrich_actions_with_prices moved to api/services/price_enricher.py
# to_csv moved to api/services/csv_helpers.py


@app.get("/proxy/fred/bitcoin")
async def proxy_fred_bitcoin(
    start_date: str = "2014-01-01", limit: Optional[int] = None, user: str = Depends(get_required_user)
):
    """Proxy pour récupérer les données Bitcoin historiques via FRED API (user-scoped)"""
    # Lire la clé FRED depuis secrets.json (modern system)
    from services.user_secrets import get_user_secrets

    secrets = get_user_secrets(user)
    fred_api_key = secrets.get("fred", {}).get("api_key") or os.getenv("FRED_API_KEY")

    if not fred_api_key:
        raise HTTPException(status_code=503, detail="FRED API key not configured")

    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "CBBTCUSD",
            "api_key": fred_api_key,
            "file_type": "json",
            "observation_start": start_date,
        }
        if limit:
            params["limit"] = limit

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if "observations" in data:
                # Transformer les données au format attendu par le frontend
                bitcoin_data = []
                for obs in data["observations"]:
                    if obs["value"] != "." and obs["value"] is not None:
                        try:
                            price = float(obs["value"])
                            timestamp = int(datetime.fromisoformat(obs["date"]).timestamp() * 1000)
                            bitcoin_data.append(
                                {"time": timestamp, "price": price, "date": obs["date"]}
                            )
                        except (ValueError, TypeError):
                            continue

                return {
                    "success": True,
                    "source": "FRED (CBBTCUSD)",
                    "data": bitcoin_data,
                    "count": len(bitcoin_data),
                    "raw_count": data.get("count", 0),
                }

        return {
            "success": False,
            "error": f"FRED API error: HTTP {response.status_code}",
            "data": [],
        }

    except httpx.HTTPError as e:
        logger.error(f"HTTP error in FRED proxy: {e}")
        return {"success": False, "error": f"HTTP error: {str(e)}", "data": []}
    except httpx.TimeoutException as e:
        logger.error(f"Timeout in FRED proxy: {e}")
        return {"success": False, "error": f"Timeout: {str(e)}", "data": []}
    except (ValueError, KeyError) as e:
        logger.warning(f"Data parsing error in FRED proxy: {e}")
        return {"success": False, "error": f"Parsing error: {str(e)}", "data": []}


@app.get("/proxy/fred/dxy")
async def proxy_fred_dxy(
    start_date: str = "2020-01-01", limit: Optional[int] = None, user: str = Depends(get_required_user)
):
    """Proxy pour récupérer l'index DXY (Trade Weighted U.S. Dollar Index) via FRED API"""
    from services.user_secrets import get_user_secrets

    secrets = get_user_secrets(user)
    fred_api_key = secrets.get("fred", {}).get("api_key") or os.getenv("FRED_API_KEY")

    if not fred_api_key:
        raise HTTPException(status_code=503, detail="FRED API key not configured")

    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "DTWEXBGS",  # Trade Weighted U.S. Dollar Index: Broad, Goods and Services
            "api_key": fred_api_key,
            "file_type": "json",
            "observation_start": start_date,
        }
        if limit:
            params["limit"] = limit

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if "observations" in data:
                dxy_data = []
                for obs in data["observations"]:
                    if obs["value"] != "." and obs["value"] is not None:
                        try:
                            value = float(obs["value"])
                            timestamp = int(datetime.fromisoformat(obs["date"]).timestamp() * 1000)
                            dxy_data.append({"time": timestamp, "value": value, "date": obs["date"]})
                        except (ValueError, TypeError):
                            continue

                # Calculer variation sur 30 jours pour macro penalty
                pct_change_30d = None
                if len(dxy_data) >= 30:
                    recent = dxy_data[-1]["value"]
                    past = dxy_data[-30]["value"]
                    if past > 0:
                        pct_change_30d = ((recent - past) / past) * 100

                return {
                    "success": True,
                    "source": "FRED (DTWEXBGS)",
                    "data": dxy_data,
                    "count": len(dxy_data),
                    "latest": dxy_data[-1] if dxy_data else None,
                    "pct_change_30d": round(pct_change_30d, 2) if pct_change_30d else None,
                }

        return {
            "success": False,
            "error": f"FRED API error: HTTP {response.status_code}",
            "data": [],
        }

    except httpx.HTTPError as e:
        logger.error(f"HTTP error in FRED DXY proxy: {e}")
        return {"success": False, "error": f"HTTP error: {str(e)}", "data": []}
    except httpx.TimeoutException as e:
        logger.error(f"Timeout in FRED DXY proxy: {e}")
        return {"success": False, "error": f"Timeout: {str(e)}", "data": []}
    except (ValueError, KeyError) as e:
        logger.warning(f"Data parsing error in FRED DXY proxy: {e}")
        return {"success": False, "error": f"Parsing error: {str(e)}", "data": []}


@app.get("/proxy/fred/vix")
async def proxy_fred_vix(
    start_date: str = "2020-01-01", limit: Optional[int] = None, user: str = Depends(get_required_user)
):
    """Proxy pour récupérer l'index VIX (CBOE Volatility Index) via FRED API"""
    from services.user_secrets import get_user_secrets

    secrets = get_user_secrets(user)
    fred_api_key = secrets.get("fred", {}).get("api_key") or os.getenv("FRED_API_KEY")

    if not fred_api_key:
        raise HTTPException(status_code=503, detail="FRED API key not configured")

    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "VIXCLS",  # CBOE Volatility Index: VIX
            "api_key": fred_api_key,
            "file_type": "json",
            "observation_start": start_date,
        }
        if limit:
            params["limit"] = limit

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if "observations" in data:
                vix_data = []
                for obs in data["observations"]:
                    if obs["value"] != "." and obs["value"] is not None:
                        try:
                            value = float(obs["value"])
                            timestamp = int(datetime.fromisoformat(obs["date"]).timestamp() * 1000)
                            vix_data.append({"time": timestamp, "value": value, "date": obs["date"]})
                        except (ValueError, TypeError):
                            continue

                # Flag si VIX > 30 (stress marché)
                latest_vix = vix_data[-1]["value"] if vix_data else None
                is_stress = latest_vix is not None and latest_vix > 30

                return {
                    "success": True,
                    "source": "FRED (VIXCLS)",
                    "data": vix_data,
                    "count": len(vix_data),
                    "latest": vix_data[-1] if vix_data else None,
                    "is_stress": is_stress,
                    "stress_threshold": 30,
                }

        return {
            "success": False,
            "error": f"FRED API error: HTTP {response.status_code}",
            "data": [],
        }

    except httpx.HTTPError as e:
        logger.error(f"HTTP error in FRED VIX proxy: {e}")
        return {"success": False, "error": f"HTTP error: {str(e)}", "data": []}
    except httpx.TimeoutException as e:
        logger.error(f"Timeout in FRED VIX proxy: {e}")
        return {"success": False, "error": f"Timeout: {str(e)}", "data": []}
    except (ValueError, KeyError) as e:
        logger.warning(f"Data parsing error in FRED VIX proxy: {e}")
        return {"success": False, "error": f"Parsing error: {str(e)}", "data": []}


@app.get("/proxy/fred/macro-stress")
async def proxy_fred_macro_stress(
    user: str = Depends(get_required_user),
    force_refresh: bool = Query(False, description="Force le rafraîchissement du cache")
):
    """
    Endpoint combiné pour évaluer le stress macro et calculer la pénalité Decision Index.
    Règle: VIX > 30 OU DXY +5% sur 30j → pénalité -15 points

    Utilise le service macro_stress avec cache (4h TTL) pour éviter les appels FRED excessifs.
    Le cache est partagé avec le calcul du Decision Index dans strategy_registry.
    """
    from services.macro_stress import macro_stress_service

    result = await macro_stress_service.evaluate_stress(user_id=user, force_refresh=force_refresh)

    return {
        "success": result.error is None,
        "error": result.error,
        "vix": {
            "value": result.vix_value,
            "is_stress": result.vix_stress,
            "threshold": 30,
        },
        "dxy": {
            "value": result.dxy_value,
            "pct_change_30d": result.dxy_change_30d,
            "is_stress": result.dxy_stress,
            "threshold_pct": 5,
        },
        "macro_stress": result.macro_stress,
        "decision_penalty": result.decision_penalty,
        "fetched_at": result.fetched_at.isoformat() if result.fetched_at else None,
    }


# ========== Router Registration (Modular) ==========
# All routers registered in api/router_registration.py for maintainability
register_routers(app)


@app.get("/portfolio/breakdown-locations")
async def portfolio_breakdown_locations(
    source: str = Query("cointracking_api"), min_usd: float = Query(1.0)
):
    """
    Renvoie la répartition par exchange à partir de la CT-API.
    Pas de fallback “CoinTracking 100%” sauf si réellement aucune data.
    """
    try:
        snap = await load_ctapi_exchanges(min_usd=min_usd)
        exchanges = snap.get("exchanges") or []
        if exchanges:
            total = sum(float(x.get("total_value_usd") or 0) for x in exchanges)
            locs = []
            for e in exchanges:
                tv = float(e.get("total_value_usd") or 0)
                locs.append(
                    {
                        "location": e.get("location"),
                        "total_value_usd": tv,
                        "asset_count": int(e.get("asset_count") or len(e.get("assets") or [])),
                        "percentage": (tv / total * 100.0) if total > 0 else 0.0,
                        "assets": e.get("assets") or [],
                    }
                )
            return {
                "ok": True,
                "breakdown": {
                    "total_value_usd": total,
                    "location_count": len(locs),
                    "locations": locs,
                },
                "fallback": False,
                "message": "",
            }
    except httpx.HTTPError as e:
        logger.error(f"HTTP error loading CoinTracking API exchanges: {e}")
    except httpx.TimeoutException as e:
        logger.error(f"Timeout loading CoinTracking API exchanges: {e}")
    except (ValueError, KeyError) as e:
        logger.error(f"Data parsing error loading CoinTracking API exchanges: {e}")

    # Fallback explicite si VRAIMENT rien
    return {
        "ok": True,
        "breakdown": {
            "total_value_usd": 0.0,
            "location_count": 1,
            "locations": [
                {
                    "location": "CoinTracking",
                    "total_value_usd": 0.0,
                    "asset_count": 0,
                    "percentage": 100.0,
                    "assets": [],
                }
            ],
        },
        "fallback": True,
        "message": "No location data available, using default location",
    }


# /portfolio/alerts migrated to api/portfolio_endpoints.py
# /api/config/* endpoints migrated to api/config_router.py

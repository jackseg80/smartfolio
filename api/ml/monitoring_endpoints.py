"""
ML Monitoring Endpoints - Health et métriques des modèles

Ce module gère:
- Santé globale du système ML
- Métriques par modèle
- Versions des modèles

Extrait de unified_ml_endpoints.py pour modularité (Fév 2026).
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Optional, Any
import logging
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path

from fastapi.responses import JSONResponse
from services.ml_pipeline_manager_optimized import optimized_pipeline_manager as pipeline_manager
from api.utils.formatters import success_response, error_response
from shared.error_handlers import handle_api_errors
from .gating import get_gating_system
from api.schemas.ml_contract import MLSystemHealth, ModelHealth

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ML Monitoring"])


@router.get("/monitoring/health", response_model=MLSystemHealth)
async def get_ml_system_health():
    """
    Obtenir l'état de santé global du système ML
    """
    try:
        gating_system = get_gating_system()
        models_status = []

        model_keys_to_check = set()

        # 1. Modèles avec historique de prédictions (gating)
        history_keys = list(gating_system.prediction_history.keys())
        model_keys_to_check.update(history_keys)
        logger.info(f"Found {len(history_keys)} models with prediction history")

        # 2. Modèles chargés dans le pipeline
        try:
            if hasattr(pipeline_manager, 'model_cache') and hasattr(pipeline_manager.model_cache, 'cache'):
                cached_keys = list(pipeline_manager.model_cache.cache.keys())
                model_keys_to_check.update(cached_keys)
                logger.info(f"Found {len(cached_keys)} models in cache: {cached_keys}")
        except Exception as e:
            logger.warning(f"Could not access model cache: {e}")

        # 3. Modèles disponibles sur disque
        if not model_keys_to_check:
            logger.info("No models in cache or history, checking available models on disk")
            try:
                pipeline_status = pipeline_manager.get_pipeline_status()
                vol_symbols = pipeline_status.get('volatility_models', {}).get('available_symbols', [])
                logger.info(f"Found {len(vol_symbols)} volatility models on disk: {vol_symbols[:5]}")

                for symbol in vol_symbols[:3]:
                    model_key = f'volatility_{symbol}'
                    model_keys_to_check.add(model_key)
                    logger.info(f"Added model from disk: {model_key}")

                if pipeline_status.get('regime_models', {}).get('model_exists', False):
                    model_keys_to_check.add('regime_model')
                    logger.info("Added regime_model from disk")
            except Exception as e:
                logger.error(f"Error checking models on disk: {e}", exc_info=True)

        logger.info(f"Total models to check: {len(model_keys_to_check)} - {list(model_keys_to_check)}")

        for model_key in model_keys_to_check:
            try:
                logger.info(f"Processing model: {model_key}")
                health_report = gating_system.get_model_health_report(model_key)
                logger.debug(f"Health report for {model_key}: {health_report}")

                model_is_loaded = (hasattr(pipeline_manager, 'model_cache') and
                                 hasattr(pipeline_manager.model_cache, 'cache') and
                                 model_key in pipeline_manager.model_cache.cache)

                if "error" in health_report:
                    health_report = {
                        "health_score": 0.8,
                        "error_rate": 0.0,
                        "avg_confidence": 0.7 if model_is_loaded else 0.5,
                        "total_predictions_24h": 0,
                        "last_prediction": None
                    }
                    logger.info(f"Created default health report for {model_key} (loaded={model_is_loaded})")

                if "error" not in health_report:
                    logger.info(f"Creating ModelHealth for {model_key}")
                    drift_score = None
                    if model_key in gating_system.prediction_history:
                        history = gating_system.prediction_history[model_key]
                        valid_predictions = [
                            h['prediction'] for h in history[-30:]
                            if not h.get('error', False) and 'prediction' in h
                        ]

                        if len(valid_predictions) >= 5:
                            mean_pred = float(np.mean(valid_predictions))
                            std_pred = float(np.std(valid_predictions))

                            if abs(mean_pred) > 1e-6:
                                cv = std_pred / abs(mean_pred)
                                drift_score = min(1.0, max(0.0, (cv - 0.05) / 0.25))
                                drift_score = round(float(drift_score), 3)
                                logger.debug(f"Drift score for {model_key}: {drift_score} (CV={cv:.3f}, n={len(valid_predictions)})")

                    model_health = ModelHealth(
                        model_name=model_key.split('_')[0],
                        version="1.0.0",
                        is_healthy=health_report.get("health_score", 0.5) > 0.5,
                        last_prediction=health_report.get("last_prediction"),
                        error_rate_24h=health_report.get("error_rate", 0.0),
                        avg_confidence=health_report.get("avg_confidence", 0.7),
                        drift_score=drift_score
                    )
                    models_status.append(model_health)
                    logger.info(f"Added ModelHealth for {model_key} to status list")
            except Exception as e:
                logger.error(f"Error processing model {model_key}: {e}", exc_info=True)

        # Calcul de la santé globale
        if models_status:
            health_scores = [
                m.avg_confidence * (1 - m.error_rate_24h)
                for m in models_status if m.avg_confidence is not None and m.error_rate_24h is not None
            ]
            overall_health = float(np.mean(health_scores)) if health_scores else 0.5
        else:
            overall_health = 0.8

        total_predictions = 0
        for model_key in model_keys_to_check:
            if model_key in gating_system.prediction_history:
                report = gating_system.get_model_health_report(model_key)
                total_predictions += report.get("total_predictions_24h", 0)

        system_metrics = {
            "active_models": len(models_status),
            "healthy_models": sum(1 for m in models_status if m.is_healthy),
            "total_predictions_24h": total_predictions
        }

        return MLSystemHealth(
            overall_health=overall_health,
            models_status=models_status,
            system_metrics=system_metrics
        )

    except Exception as e:
        logger.error(f"Failed to get ML system health: {e}")
        return MLSystemHealth(
            overall_health=0.3,
            models_status=[],
            system_metrics={"error": str(e)}
        )


@router.get("/metrics/{model_name}")
@handle_api_errors(fallback={"error": "Metrics unavailable"}, reraise_http_errors=True)
async def get_model_metrics(model_name: str, version: Optional[str] = None) -> dict:
    """
    Obtenir les métriques pour un modèle spécifique
    """
    metrics_file = Path("data/ml_metrics.json")

    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)

        model_data = all_metrics.get(model_name, {})
        if version:
            version_data = model_data.get("versions", {}).get(version, {})
            return success_response({"model": model_name, "version": version, "metrics": version_data})
        else:
            versions = model_data.get("versions", {})
            if versions:
                latest_version = max(versions.keys())
                return success_response({"model": model_name, "version": latest_version, "metrics": versions[latest_version]})

    gating_system = get_gating_system()
    matching_keys = [key for key in gating_system.prediction_history.keys() if model_name in key]

    if matching_keys:
        key = matching_keys[0]
        health_report = gating_system.get_model_health_report(key)

        return success_response({
            "model": model_name,
            "version": "1.0.0",
            "metrics": {
                "predictions_24h": health_report.get("total_predictions_24h", 0),
                "error_rate": health_report.get("error_rate", 0.0),
                "avg_confidence": health_report.get("avg_confidence", 0.5),
                "acceptance_rate": health_report.get("acceptance_rate", 0.8),
                "last_updated": health_report.get("last_prediction", datetime.now()).isoformat()
            }
        })

    return error_response(f"No metrics found for model {model_name}", code=400)


@router.get("/versions/{model_name}")
@handle_api_errors(fallback={"available_versions": [], "total_versions": 0})
async def get_model_versions(model_name: str) -> dict:
    """
    Lister les versions disponibles d'un modèle
    """
    metrics_file = Path("data/ml_metrics.json")

    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)

        model_data = all_metrics.get(model_name, {})
        versions = list(model_data.get("versions", {}).keys())

        return success_response({
            "model": model_name,
            "available_versions": versions,
            "total_versions": len(versions)
        })

    return success_response({
        "model": model_name,
        "available_versions": ["1.0.0"],
        "total_versions": 1
    })


@router.post("/metrics/{model_name}/update")
@handle_api_errors(fallback={"message": "Failed to update metrics"}, reraise_http_errors=True)
async def update_model_metrics(
    model_name: str,
    version: str,
    metrics: Dict[str, Any] = Body(...)
) -> dict:
    """
    Mettre à jour les métriques d'un modèle (version spécifique)
    """
    os.makedirs("data", exist_ok=True)
    metrics_file = Path("data/ml_metrics.json")

    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    if model_name not in all_metrics:
        all_metrics[model_name] = {"versions": {}}

    all_metrics[model_name]["versions"][version] = {
        **metrics,
        "last_updated": datetime.now().isoformat()
    }

    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    return success_response({
        "model": model_name,
        "version": version,
        "message": "Metrics updated successfully"
    })

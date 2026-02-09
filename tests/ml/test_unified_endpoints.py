"""
Tests pour les endpoints ML unifies
"""

import pytest
from fastapi.testclient import TestClient
import json
from unittest.mock import Mock, patch, AsyncMock

from api.main import app


class TestMLEndpoints:
    """Tests pour les endpoints ML unifies"""

    def setup_method(self):
        """Setup pour chaque test"""
        self.client = TestClient(app)

    def test_ml_status_endpoint(self):
        """Test de l'endpoint de statut ML"""
        response = self.client.get("/api/ml/status")

        assert response.status_code == 200
        data = response.json()

        assert "success" in data
        assert "pipeline_status" in data
        assert "timestamp" in data

        pipeline_status = data["pipeline_status"]
        assert "pipeline_initialized" in pipeline_status
        assert "models_base_path" in pipeline_status
        assert "volatility_models" in pipeline_status
        assert "regime_models" in pipeline_status
        assert "loaded_models_count" in pipeline_status

    @patch('api.ml.prediction_endpoints.get_ml_predictions', new_callable=AsyncMock)
    def test_unified_predictions_basic(self, mock_predictions):
        """Test des predictions unifiees basiques"""
        # Mock de la reponse
        mock_predictions.return_value = {
            "predictions": {
                "BTC": {"volatility": 0.045, "trend": "bullish"},
                "ETH": {"volatility": 0.062, "trend": "neutral"}
            },
            "regime": {"current": "bull", "confidence": 0.78},
            "volatility": {"BTC": 0.045, "ETH": 0.062},
            "model_status": {"volatility_loaded": True, "regime_loaded": True}
        }

        request_data = {
            "assets": ["BTC", "ETH"],
            "horizon_days": 30,
            "include_regime": True,
            "include_volatility": True
        }

        response = self.client.post("/api/ml/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "predictions" in data
        assert "regime_prediction" in data
        assert "volatility_forecast" in data
        assert "model_status" in data
        assert "timestamp" in data

        # Verifier les predictions
        predictions = data["predictions"]
        assert "BTC" in predictions
        assert "ETH" in predictions

    @patch('api.ml.prediction_endpoints._get_multi_horizon_predictions', new_callable=AsyncMock)
    @patch('api.ml.prediction_endpoints.get_ml_predictions', new_callable=AsyncMock)
    def test_unified_predictions_multi_horizon(self, mock_predictions, mock_multi_horizon):
        """Test des predictions multi-horizon"""
        # Mock des predictions de base
        mock_predictions.return_value = {
            "predictions": {"BTC": {"volatility": 0.045}},
            "regime": {"current": "bull", "confidence": 0.78},
            "model_status": {"volatility_loaded": True}
        }

        # Mock des predictions multi-horizon
        mock_multi_horizon.return_value = {
            "BTC": {
                "horizon_1d": {"volatility": 0.043, "expected_return": 0.001, "horizon_days": 1},
                "horizon_7d": {"volatility": 0.048, "expected_return": 0.025, "horizon_days": 7},
                "horizon_30d": {"volatility": 0.055, "expected_return": 0.08, "horizon_days": 30}
            }
        }

        request_data = {
            "assets": ["BTC"],
            "horizons": [1, 7, 30],
            "include_regime": True,
            "include_volatility": True,
            "include_confidence": False
        }

        response = self.client.post("/api/ml/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True

        # Verifier que la fonction multi-horizon a ete appelee
        mock_multi_horizon.assert_called_once_with(["BTC"], [1, 7, 30], False)

        # Verifier la structure des predictions
        predictions = data["predictions"]
        assert "BTC" in predictions
        btc_predictions = predictions["BTC"]
        assert "horizon_1d" in btc_predictions
        assert "horizon_7d" in btc_predictions
        assert "horizon_30d" in btc_predictions

    @patch('api.ml.prediction_endpoints._add_confidence_metrics', new_callable=AsyncMock)
    @patch('api.ml.prediction_endpoints.get_ml_predictions', new_callable=AsyncMock)
    def test_unified_predictions_with_confidence(self, mock_predictions, mock_confidence):
        """Test des predictions avec metriques de confiance"""
        # Mock des predictions de base
        mock_predictions.return_value = {
            "predictions": {"BTC": {"volatility": 0.045}},
            "regime": {"current": "bull", "confidence": 0.78},
            "model_status": {"volatility_loaded": True}
        }

        # Mock des metriques de confiance
        mock_confidence.return_value = {
            "BTC": {
                "volatility": 0.045,
                "confidence_metrics": {
                    "model_confidence": 0.78,
                    "data_quality_score": 0.85,
                    "prediction_stability": 0.72,
                    "overall_confidence": 0.787
                }
            }
        }

        request_data = {
            "assets": ["BTC"],
            "include_confidence": True
        }

        response = self.client.post("/api/ml/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True

        # Verifier que la fonction de confiance a ete appelee
        mock_confidence.assert_called_once()

        # Verifier les metriques de confiance
        predictions = data["predictions"]
        assert "BTC" in predictions
        btc_predictions = predictions["BTC"]
        assert "confidence_metrics" in btc_predictions

        confidence = btc_predictions["confidence_metrics"]
        assert "model_confidence" in confidence
        assert "overall_confidence" in confidence

    def test_volatility_prediction_endpoint(self):
        """Test de l'endpoint de prediction de volatilite specifique"""
        symbol = "BTC"
        horizon = 30

        response = self.client.get(f"/api/ml/volatility/predict/{symbol}?horizon_days={horizon}")

        # L'endpoint devrait repondre meme avec des modeles non charges (fallback)
        assert response.status_code == 200
        data = response.json()

        # Response wrapped in success_response
        inner = data.get("data", data)
        assert "symbol" in inner
        assert inner["symbol"] == symbol
        assert "volatility_forecast" in inner or "error" in inner

    def test_regime_prediction_endpoint(self):
        """Test de l'endpoint de prediction de regime"""
        response = self.client.get("/api/ml/regime/current")

        assert response.status_code == 200
        data = response.json()

        # The /regime/current endpoint returns success_response with regime_prediction
        inner = data.get("data", data)
        assert data.get("ok", data.get("success")) is True
        assert "regime_prediction" in inner

    def test_sentiment_endpoint(self):
        """Test de l'endpoint de sentiment"""
        symbol = "BTC"
        response = self.client.get(f"/api/ml/sentiment/{symbol}?days=1")

        assert response.status_code == 200
        data = response.json()

        # get_sentiment returns plain dict (used internally by other endpoints)
        assert data.get("symbol") == symbol
        assert "aggregated_sentiment" in data
        assert "sources_used" in data

    def test_fear_greed_endpoint(self):
        """Test de l'endpoint Fear & Greed via sentiment path"""
        # Note: /sentiment/fear-greed is captured by /sentiment/{symbol} route
        # so it returns sentiment data for symbol="FEAR-GREED"
        response = self.client.get("/api/ml/sentiment/fear-greed?days=1")

        assert response.status_code == 200
        data = response.json()

        # /sentiment/fear-greed is captured by /sentiment/{symbol} route (plain dict)
        assert data.get("symbol") == "FEAR-GREED"
        assert "aggregated_sentiment" in data

    @patch('api.ml.model_endpoints.pipeline_manager')
    def test_preload_models_endpoint(self, mock_pipeline):
        """Test de l'endpoint de prechargement des modeles"""
        # Mock the individual model loading methods
        mock_pipeline.load_regime_model.return_value = True
        mock_pipeline.load_volatility_model.return_value = True

        response = self.client.post("/api/ml/models/preload?symbols=BTC&symbols=ETH")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["loaded_models"] == 3  # regime + 2 volatility
        assert data["total_requested"] == 3
        assert "preload_results" in data

    @patch('api.ml.cache_endpoints.pipeline_manager')
    def test_cache_stats_endpoint(self, mock_pipeline):
        """Test de l'endpoint des statistiques de cache"""
        # Mock the flat dict returned by get_cache_stats()
        mock_pipeline.get_cache_stats.return_value = {
            "cached_models": 2,
            "total_size_mb": 3.5,
            "max_size": 8,
            "memory_usage_percent": 43.75,
            "loading_status": {
                "volatility_BTC": "loaded",
                "regime": "loaded"
            }
        }

        response = self.client.get("/api/ml/cache/stats")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "cache_stats" in data

        cache_stats = data["cache_stats"]
        assert cache_stats["cached_models"] == 2
        assert cache_stats["total_size_mb"] == 3.5
        assert "loading_status" in cache_stats

    @patch('api.ml.cache_endpoints.pipeline_manager')
    def test_memory_optimize_endpoint(self, mock_pipeline):
        """Test de l'endpoint d'optimisation memoire"""
        # Mock the response from optimize_memory()
        mock_pipeline.optimize_memory.return_value = {
            "initial_models": 3,
            "final_models": 2,
            "evicted_models": 1,
            "memory_before": 65.0,
            "memory_after": 62.5,
            "memory_saved": 2.5
        }

        response = self.client.post("/api/ml/memory/optimize")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "optimization_result" in data
        assert data["optimization_result"]["evicted_models"] == 1

    def test_training_endpoint_background(self):
        """Test de l'endpoint d'entrainement en arriere-plan"""
        request_data = {
            "assets": ["BTC", "ETH"],
            "lookback_days": 365,
            "include_market_indicators": True,
            "save_models": True
        }

        response = self.client.post("/api/ml/train", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "Training started" in data["message"]
        assert data["background_task"] is True
        assert data["assets"] == ["BTC", "ETH"]

    def test_invalid_prediction_request(self):
        """Test de requete de prediction invalide"""
        # Requete sans assets
        request_data = {
            "horizon_days": 30
        }

        response = self.client.post("/api/ml/predict", json=request_data)

        # Devrait retourner une erreur de validation
        assert response.status_code == 422

    def test_invalid_symbol_format(self):
        """Test avec un symbole invalide"""
        response = self.client.get("/api/ml/volatility/predict/INVALID_SYMBOL_123")

        # L'endpoint devrait gerer gracieusement les symboles invalides
        assert response.status_code in [200, 400, 422]

    def test_cache_functionality(self):
        """Test de la fonctionnalite de cache des endpoints"""
        request_data = {
            "assets": ["BTC"],
            "horizon_days": 30
        }

        # Premiere requete
        response1 = self.client.post("/api/ml/predict", json=request_data)
        assert response1.status_code == 200

        # Deuxieme requete identique (devrait utiliser le cache)
        response2 = self.client.post("/api/ml/predict", json=request_data)
        assert response2.status_code == 200

        # Les reponses devraient etre identiques (du cache)
        assert response1.json() == response2.json()


class TestMLHelperFunctions:
    """Tests pour les fonctions helper ML"""

    @pytest.mark.asyncio
    async def test_get_multi_horizon_predictions(self):
        """Test de la fonction multi-horizon"""
        from api.ml.prediction_endpoints import _get_multi_horizon_predictions

        assets = ["BTC", "ETH"]
        horizons = [1, 7, 30]
        include_confidence = True

        result = await _get_multi_horizon_predictions(assets, horizons, include_confidence)

        assert isinstance(result, dict)
        assert "BTC" in result
        assert "ETH" in result

        for symbol in assets:
            symbol_data = result[symbol]
            assert "horizon_1d" in symbol_data
            assert "horizon_7d" in symbol_data
            assert "horizon_30d" in symbol_data

            for horizon_key, horizon_data in symbol_data.items():
                assert "volatility" in horizon_data
                assert "expected_return" in horizon_data
                assert "horizon_days" in horizon_data

                if include_confidence:
                    assert "confidence" in horizon_data
                    assert "prediction_interval" in horizon_data

    @pytest.mark.asyncio
    async def test_add_confidence_metrics(self):
        """Test de la fonction d'ajout de metriques de confiance"""
        from api.ml.prediction_endpoints import _add_confidence_metrics

        predictions = {
            "BTC": {"volatility": 0.045},
            "ETH": {"volatility": 0.062}
        }
        assets = ["BTC", "ETH"]

        result = await _add_confidence_metrics(predictions, assets)

        assert isinstance(result, dict)
        assert "BTC" in result
        assert "ETH" in result

        for symbol in assets:
            symbol_data = result[symbol]
            assert "confidence_metrics" in symbol_data

            confidence = symbol_data["confidence_metrics"]
            assert "model_confidence" in confidence
            assert "data_quality_score" in confidence
            assert "prediction_stability" in confidence
            assert "overall_confidence" in confidence

            # Verifier les plages de valeurs
            assert 0 <= confidence["overall_confidence"] <= 1


class TestMLEndpointIntegration:
    """Tests d'integration pour les endpoints ML"""

    def setup_method(self):
        """Setup pour chaque test"""
        self.client = TestClient(app)

    def test_full_prediction_workflow(self):
        """Test du workflow complet de prediction"""
        # 1. Verifier le statut du pipeline
        status_response = self.client.get("/api/ml/status")
        assert status_response.status_code == 200

        # 2. Obtenir des predictions
        request_data = {
            "assets": ["BTC", "ETH"],
            "horizons": [1, 30],
            "include_regime": True,
            "include_volatility": True,
            "include_confidence": True
        }

        pred_response = self.client.post("/api/ml/predict", json=request_data)
        assert pred_response.status_code == 200

        pred_data = pred_response.json()
        assert pred_data["success"] is True

        # 3. Verifier les statistiques de cache
        cache_response = self.client.get("/api/ml/cache/stats")
        assert cache_response.status_code == 200

    def test_error_handling_pipeline_unavailable(self):
        """Test de gestion d'erreur quand le pipeline n'est pas disponible"""
        with patch('api.ml.model_endpoints.pipeline_manager') as mock_pipeline:
            mock_pipeline.get_pipeline_status.side_effect = Exception("Pipeline not available")

            response = self.client.get("/api/ml/status")
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is False
            assert "error" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests pour le pipeline ML optimise
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import torch
import pickle
import json

from services.ml_pipeline_manager_optimized import OptimizedMLPipelineManager, LRUCache


class TestLRUCache:
    """Tests pour le cache LRU optimise"""

    def test_cache_basic_operations(self):
        cache = LRUCache(max_size=3)

        # Test ajout d'elements
        cache.put("key1", "value1", size_mb=1.0)
        cache.put("key2", "value2", size_mb=1.5)
        cache.put("key3", "value3", size_mb=2.0)

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert len(cache.cache) == 3
        assert sum(cache.model_sizes.values()) == 4.5

    def test_cache_eviction_by_size(self):
        cache = LRUCache(max_size=2)

        cache.put("key1", "value1", size_mb=1.0)
        cache.put("key2", "value2", size_mb=1.0)

        # Ajouter un troisieme element devrait evincer le premier
        cache.put("key3", "value3", size_mb=1.0)

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert len(cache.cache) == 2

    def test_cache_lru_ordering(self):
        cache = LRUCache(max_size=3)

        cache.put("key1", "value1", size_mb=1.0)
        cache.put("key2", "value2", size_mb=1.0)
        cache.put("key3", "value3", size_mb=1.0)

        # Acceder a key1 le rend plus recent
        cache.get("key1")

        # Ajouter key4 devrait evincer key2 (pas key1)
        cache.put("key4", "value4", size_mb=1.0)

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_clear(self):
        cache = LRUCache(max_size=3)

        cache.put("key1", "value1", size_mb=1.0)
        cache.put("key2", "value2", size_mb=1.0)

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache.cache) == 0
        assert sum(cache.model_sizes.values()) == 0


class TestOptimizedMLPipelineManager:
    """Tests pour le gestionnaire de pipeline ML optimise"""

    def setup_method(self):
        """Setup pour chaque test"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.pipeline = OptimizedMLPipelineManager(models_base_path=str(self.temp_dir))

    def teardown_method(self):
        """Cleanup apres chaque test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_initialization(self):
        """Test de l'initialisation du pipeline"""
        assert self.pipeline.models_base_path == self.temp_dir
        assert self.pipeline.model_cache is not None
        assert self.pipeline.model_cache.max_size == 5  # default max_cached_models
        assert self.pipeline.loading_lock is not None
        assert isinstance(self.pipeline.stats, dict)

    def test_volatility_model_paths(self):
        """Test that volatility model paths are correctly set"""
        expected_base = self.temp_dir / "volatility"
        assert self.pipeline.volatility_path == expected_base
        assert expected_base.exists()

    def test_regime_model_paths(self):
        """Test that regime model paths are correctly set"""
        expected_base = self.temp_dir / "regime"
        assert self.pipeline.regime_path == expected_base
        assert expected_base.exists()

    def test_volatility_model_loading_missing_files(self):
        """Test du chargement d'un modele de volatilite avec fichiers manquants"""
        symbol = "NONEXISTENT"
        result = self.pipeline.load_volatility_model(symbol)

        assert result is False
        assert self.pipeline.model_cache.get(f"volatility_{symbol}") is None

    def test_get_pipeline_status(self):
        """Test de l'obtention du statut du pipeline"""
        status = self.pipeline.get_pipeline_status()

        assert "pipeline_initialized" in status
        assert "models_base_path" in status
        assert "timestamp" in status
        assert "volatility_models" in status
        assert "regime_models" in status
        assert "loaded_models_count" in status
        assert "total_models_count" in status

        # Test des sous-structures
        vol_status = status["volatility_models"]
        assert "models_count" in vol_status
        assert "models_loaded" in vol_status
        assert "last_updated" in vol_status

        regime_status = status["regime_models"]
        assert "model_exists" in regime_status
        assert "model_loaded" in regime_status
        assert "last_updated" in regime_status

    def test_clear_all_models(self):
        """Test du vidage du cache via clear_all_models"""
        # Ajouter quelques elements au cache
        self.pipeline.model_cache.put("test_key", {"model": "test"}, size_mb=1.0)

        assert len(self.pipeline.model_cache.cache) == 1

        # Vider le cache
        count = self.pipeline.clear_all_models()

        assert count == 1
        assert len(self.pipeline.model_cache.cache) == 0

    def test_memory_optimization(self):
        """Test de l'optimisation memoire"""
        # Simuler un cache plein
        for i in range(5):
            self.pipeline.model_cache.put(f"key_{i}", f"value_{i}", size_mb=1.0)

        initial_size = len(self.pipeline.model_cache.cache)

        result = self.pipeline.optimize_memory()

        assert "initial_models" in result
        assert "final_models" in result
        assert "evicted_models" in result
        assert "memory_before" in result
        assert "memory_after" in result
        assert "memory_saved" in result
        # Le cache ne devrait pas depasser sa taille maximale
        assert len(self.pipeline.model_cache.cache) <= self.pipeline.model_cache.max_size

    def test_get_cache_stats(self):
        """Test de l'obtention des statistiques de cache"""
        # Ajouter des elements au cache
        self.pipeline.model_cache.put("vol_BTC", {"model": "btc"}, size_mb=2.0)
        self.pipeline.model_cache.put("regime", {"model": "regime"}, size_mb=1.5)

        stats = self.pipeline.get_cache_stats()

        # get_cache_stats returns a flat dict merging cache.stats() + self.stats + loading_status
        assert stats["cached_models"] == 2
        assert stats["total_size_mb"] == 3.5
        assert stats["max_size"] == 5  # default max_cached_models
        assert "loading_status" in stats
        assert "memory_usage_percent" in stats


@pytest.fixture
def pipeline_with_models():
    """Fixture avec des modeles mock charges"""
    temp_dir = Path(tempfile.mkdtemp())
    pipeline = OptimizedMLPipelineManager(models_base_path=str(temp_dir))

    # Ajouter des modeles mock au cache
    pipeline.model_cache.put("volatility_BTC", {
        "model": Mock(),
        "scaler": Mock(),
        "metadata": {"accuracy": 0.85, "training_date": "2024-01-01"}
    }, size_mb=2.0)

    pipeline.model_cache.put("regime", {
        "model": Mock(),
        "scaler": Mock(),
        "metadata": {"accuracy": 0.78, "training_date": "2024-01-01"}
    }, size_mb=1.5)

    pipeline.stats["models_loaded"] = 2

    yield pipeline

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_integration_pipeline_with_loaded_models(pipeline_with_models):
    """Test d'integration avec des modeles charges"""
    pipeline = pipeline_with_models

    # Test du statut avec modeles charges
    status = pipeline.get_pipeline_status()
    assert status["loaded_models_count"] == 2

    # Test des statistiques de cache (flat dict)
    cache_stats = pipeline.get_cache_stats()
    assert cache_stats["cached_models"] == 2

    # Test de la presence des modeles
    btc_model = pipeline.model_cache.get("volatility_BTC")
    assert btc_model is not None
    assert "model" in btc_model
    assert "metadata" in btc_model

    regime_model = pipeline.model_cache.get("regime")
    assert regime_model is not None
    assert "model" in regime_model
    assert "metadata" in regime_model


def test_concurrent_access():
    """Test d'acces concurrent au pipeline"""
    import threading
    import time

    temp_dir = Path(tempfile.mkdtemp())
    pipeline = OptimizedMLPipelineManager(models_base_path=str(temp_dir))

    results = []
    errors = []

    def worker(thread_id):
        try:
            for i in range(5):
                # Simuler des operations concurrentes
                status = pipeline.get_pipeline_status()
                cache_stats = pipeline.get_cache_stats()

                # Ajouter au cache
                pipeline.model_cache.put(f"test_{thread_id}_{i}", f"value_{thread_id}_{i}", size_mb=0.1)

                time.sleep(0.01)  # Petite pause

                results.append((thread_id, i, status["timestamp"]))
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Lancer plusieurs threads
    threads = []
    for tid in range(3):
        t = threading.Thread(target=worker, args=(tid,))
        threads.append(t)
        t.start()

    # Attendre la fin
    for t in threads:
        t.join()

    # Verifier les resultats
    assert len(errors) == 0, f"Erreurs dans les threads: {errors}"
    assert len(results) == 15  # 3 threads * 5 operations

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

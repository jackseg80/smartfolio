"""
Tests pour les endpoints de performance P&L Today
"""
import pytest


def test_performance_summary_basic(test_client):
    """Test basique de l'endpoint /api/performance/summary"""
    response = test_client.get(
        "/api/performance/summary",
        headers={"X-User": "jack"}
    )
    assert response.status_code == 200
    data = response.json()

    # Vérifier la structure de base
    assert data["ok"] is True
    assert "performance" in data
    assert "as_of" in data["performance"]
    assert "total" in data["performance"]
    assert "by_account" in data["performance"]
    assert "by_source" in data["performance"]

    # Vérifier les champs du total
    total = data["performance"]["total"]
    assert "current_value_usd" in total
    assert "absolute_change_usd" in total
    assert "percent_change" in total


def test_performance_summary_anchor_parameter(test_client):
    """Test du paramètre anchor"""
    for anchor in ["prev_close", "midnight", "session"]:
        response = test_client.get(
            f"/api/performance/summary?anchor={anchor}",
            headers={"X-User": "jack"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True


def test_performance_summary_etag(test_client):
    """Test du fonctionnement ETag"""
    # Premier appel pour obtenir l'ETag
    response1 = test_client.get(
        "/api/performance/summary",
        headers={"X-User": "jack"}
    )
    assert response1.status_code == 200
    etag = response1.headers.get("etag")
    assert etag is not None

    # Deuxième appel avec le bon ETag
    response2 = test_client.get(
        "/api/performance/summary",
        headers={"if-none-match": etag, "X-User": "jack"}
    )
    assert response2.status_code == 304

    # Appel avec mauvais ETag
    response3 = test_client.get(
        "/api/performance/summary",
        headers={"if-none-match": "wrong-etag", "X-User": "jack"}
    )
    assert response3.status_code == 200


def test_performance_summary_cache_headers(test_client):
    """Test des headers de cache"""
    response = test_client.get(
        "/api/performance/summary",
        headers={"X-User": "jack"}
    )
    assert response.status_code == 200

    # Vérifier les headers de cache
    assert "etag" in response.headers
    assert "cache-control" in response.headers
    etag = response.headers["etag"]
    assert len(etag) > 10  # Vérifier que l'ETag est significatif


def test_performance_summary_data_integrity(test_client):
    """Test de l'intégrité des données"""
    response = test_client.get(
        "/api/performance/summary",
        headers={"X-User": "jack"}
    )
    assert response.status_code == 200
    data = response.json()

    perf = data["performance"]
    total = perf["total"]

    # Vérifier que les valeurs sont cohérentes
    assert isinstance(total["current_value_usd"], (int, float))
    assert isinstance(total["absolute_change_usd"], (int, float))
    assert isinstance(total["percent_change"], (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

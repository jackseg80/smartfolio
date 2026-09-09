"""Contracts preventing synthetic governance inputs."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.execution import signals_endpoints
from api.execution.models import ProposeDecisionRequest
from api.deps import get_current_user_jwt
from api.main import app


@pytest.fixture
def client():
    app.dependency_overrides[get_current_user_jwt] = lambda: "jack"
    try:
        yield TestClient(app, headers={"X-User": "jack"})
    finally:
        app.dependency_overrides.pop(get_current_user_jwt, None)


def test_recompute_rejects_missing_components_without_masking_409(monkeypatch, client):
    monkeypatch.setattr(signals_endpoints, "_LAST_RECOMPUTE_TS", 0.0)
    monkeypatch.setattr(signals_endpoints, "_RECOMPUTE_WINDOW", [])
    monkeypatch.setattr(signals_endpoints, "_RECOMPUTE_CACHE", {})

    response = client.post(
        "/execution/governance/signals/recompute",
        json={},
        headers={"X-CSRF-Token": "contract-test"},
    )

    assert response.status_code == 409
    assert "NeedsRefresh" in response.json()["detail"]


def test_governance_proposal_requires_explicit_targets():
    with pytest.raises(ValidationError):
        ProposeDecisionRequest()

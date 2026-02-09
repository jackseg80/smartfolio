"""
Manual test configuration.

Tests in this directory that use requests/httpx to connect to localhost
are automatically skipped unless a server is running.
"""

import pytest


def _server_is_running() -> bool:
    """Check if the local API server is running."""
    try:
        import requests
        resp = requests.get("http://localhost:8080/docs", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


_SERVER_UP = _server_is_running()

requires_server = pytest.mark.skipif(
    not _SERVER_UP,
    reason="Requires running API server on localhost:8080"
)

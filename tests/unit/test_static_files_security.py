"""Security tests for raw static file mounts."""

from fastapi import FastAPI

from api.static_files_setup import setup_static_files


def _mounted_paths(app: FastAPI) -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


def test_production_does_not_mount_raw_data_or_config():
    app = FastAPI()
    setup_static_files(app, debug=False)

    paths = _mounted_paths(app)
    assert "/static" in paths
    assert "/data" not in paths
    assert "/config" not in paths
    assert "/tests" not in paths


def test_debug_mode_keeps_local_raw_mounts():
    app = FastAPI()
    setup_static_files(app, debug=True)

    paths = _mounted_paths(app)
    assert {"/static", "/data", "/config", "/tests"} <= paths

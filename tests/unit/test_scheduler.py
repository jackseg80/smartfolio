"""Tests unitaires pour api/scheduler.py — Scheduler lifecycle + jobs."""
import asyncio
import json
import os
import sys
import time
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock, mock_open

from api.scheduler import (
    get_scheduler,
    get_job_status,
    _update_job_status,
    _user_has_cointracking_credentials,
    job_pnl_intraday,
    job_pnl_eod,
    job_ohlcv_daily,
    job_ohlcv_hourly,
    job_staleness_monitor,
    job_api_warmers,
    job_crypto_toolbox_refresh,
    job_daily_ml_training,
    _acquire_scheduler_lock,
    _renew_scheduler_lock,
    initialize_scheduler,
    shutdown_scheduler,
    # Redis persistence (5.2)
    _get_redis_client,
    get_job_status_persistent,
    get_scheduler_heartbeat,
    _notify_job_failure,
    _recover_job_status_from_redis,
    REDIS_JOB_STATUS_KEY,
    REDIS_HEARTBEAT_KEY,
)
import api.scheduler as scheduler_module


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def reset_scheduler_state():
    """Reset scheduler globals before each test.

    Also disables Redis and webhook calls by default to avoid network
    timeouts in tests. Individual tests can override with patch().
    """
    scheduler_module._scheduler = None
    scheduler_module._job_status.clear()
    with patch.object(scheduler_module, '_get_redis_client', new_callable=AsyncMock, return_value=None), \
         patch.object(scheduler_module, '_notify_job_failure', new_callable=AsyncMock):
        yield
    scheduler_module._scheduler = None
    scheduler_module._job_status.clear()


@pytest.fixture
def mock_users_config():
    """Mock users config with 2 active users."""
    return json.dumps({
        "users": [
            {"id": "jack", "status": "active"},
            {"id": "demo", "status": "active"},
            {"id": "inactive", "status": "disabled"},
        ]
    })


@pytest.fixture
def mock_users_config_single():
    """Mock users config with 1 active user."""
    return json.dumps({
        "users": [
            {"id": "jack", "status": "active"},
        ]
    })


# ============================================================================
# JOB STATUS TRACKING
# ============================================================================

class TestJobStatus:
    """Tests for job status tracking functions."""

    def test_get_scheduler_initially_none(self):
        assert get_scheduler() is None

    def test_get_job_status_initially_empty(self):
        assert get_job_status() == {}

    @pytest.mark.asyncio
    async def test_update_job_status_success(self):
        await _update_job_status("test_job", "success", duration_ms=150.5)
        status = get_job_status()
        assert "test_job" in status
        assert status["test_job"]["status"] == "success"
        assert status["test_job"]["duration_ms"] == 150.5
        assert status["test_job"]["error"] is None
        assert "last_run" in status["test_job"]

    @pytest.mark.asyncio
    async def test_update_job_status_error(self):
        await _update_job_status("test_job", "error", duration_ms=50, error="Connection refused")
        status = get_job_status()
        assert status["test_job"]["status"] == "error"
        assert status["test_job"]["error"] == "Connection refused"

    @pytest.mark.asyncio
    async def test_update_job_status_overwrites(self):
        await _update_job_status("job1", "running")
        await _update_job_status("job1", "success", duration_ms=100)
        status = get_job_status()
        assert status["job1"]["status"] == "success"

    def test_get_job_status_returns_shallow_copy(self):
        """get_job_status returns .copy() — top-level keys are independent."""
        scheduler_module._job_status["test"] = {"status": "ok"}
        status = get_job_status()
        # Removing key from copy does NOT affect original
        del status["test"]
        assert "test" in scheduler_module._job_status

    @pytest.mark.asyncio
    async def test_last_run_is_iso_format(self):
        await _update_job_status("job1", "success")
        status = get_job_status()
        # Should be parseable ISO datetime
        datetime.fromisoformat(status["job1"]["last_run"])


# ============================================================================
# CREDENTIAL CHECKS
# ============================================================================

class TestUserHasCointrackingCredentials:
    """Tests for _user_has_cointracking_credentials."""

    def test_returns_false_on_exception(self):
        """Any exception inside the function returns False."""
        mock_migrator = MagicMock()
        mock_migrator.resolve_secret_ref = MagicMock(side_effect=Exception("No config"))

        mock_user_fs_module = MagicMock()

        with patch.dict(sys.modules, {
            'api.services.config_migrator': mock_migrator,
            'api.services.user_fs': mock_user_fs_module,
        }):
            result = _user_has_cointracking_credentials("nonexistent")
            assert result is False

    def test_returns_false_when_no_keys(self):
        mock_migrator = MagicMock()
        mock_migrator.resolve_secret_ref = MagicMock(return_value=None)
        mock_user_fs_module = MagicMock()

        with patch.dict(sys.modules, {
            'api.services.config_migrator': mock_migrator,
            'api.services.user_fs': mock_user_fs_module,
        }):
            result = _user_has_cointracking_credentials("jack")
            assert result is False

    def test_returns_true_with_valid_keys(self):
        mock_migrator = MagicMock()
        mock_migrator.resolve_secret_ref = MagicMock(return_value="valid_key_123")
        mock_user_fs_module = MagicMock()

        with patch.dict(sys.modules, {
            'api.services.config_migrator': mock_migrator,
            'api.services.user_fs': mock_user_fs_module,
        }):
            result = _user_has_cointracking_credentials("jack")
            assert result is True


# ============================================================================
# P&L JOBS
# ============================================================================

class TestJobPnlIntraday:
    """Tests for job_pnl_intraday."""

    @pytest.mark.asyncio
    async def test_all_snapshots_succeed(self, mock_users_config_single):
        mock_snapshot = AsyncMock(return_value={"ok": True, "positions": 5})

        with patch('builtins.open', mock_open(read_data=mock_users_config_single)):
            with patch('api.scheduler._user_has_cointracking_credentials', return_value=True):
                with patch('scripts.pnl_snapshot.create_snapshot', mock_snapshot):
                    await job_pnl_intraday()

        mock_snapshot.assert_called_once()
        status = get_job_status()
        assert status["pnl_intraday"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_partial_failure(self, mock_users_config):
        call_count = 0

        async def mock_snapshot(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("user_id") == "demo":
                return {"ok": False, "error": "No data"}
            return {"ok": True}

        with patch('builtins.open', mock_open(read_data=mock_users_config)):
            with patch('api.scheduler._user_has_cointracking_credentials', return_value=True):
                with patch('scripts.pnl_snapshot.create_snapshot', side_effect=mock_snapshot):
                    await job_pnl_intraday()

        status = get_job_status()
        assert status["pnl_intraday"]["status"] == "partial"

    @pytest.mark.asyncio
    async def test_all_snapshots_fail(self, mock_users_config_single):
        mock_snapshot = AsyncMock(return_value={"ok": False, "error": "API down"})

        with patch('builtins.open', mock_open(read_data=mock_users_config_single)):
            with patch('api.scheduler._user_has_cointracking_credentials', return_value=True):
                with patch('scripts.pnl_snapshot.create_snapshot', mock_snapshot):
                    await job_pnl_intraday()

        status = get_job_status()
        assert status["pnl_intraday"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_snapshot_exception(self, mock_users_config_single):
        mock_snapshot = AsyncMock(side_effect=Exception("Connection timeout"))

        with patch('builtins.open', mock_open(read_data=mock_users_config_single)):
            with patch('api.scheduler._user_has_cointracking_credentials', return_value=True):
                with patch('scripts.pnl_snapshot.create_snapshot', mock_snapshot):
                    await job_pnl_intraday()

        status = get_job_status()
        assert status["pnl_intraday"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_no_active_users(self):
        empty_config = json.dumps({"users": []})
        with patch('builtins.open', mock_open(read_data=empty_config)):
            await job_pnl_intraday()

        status = get_job_status()
        assert status["pnl_intraday"]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_no_users_with_credentials(self, mock_users_config_single):
        with patch('builtins.open', mock_open(read_data=mock_users_config_single)):
            with patch('api.scheduler._user_has_cointracking_credentials', return_value=False):
                await job_pnl_intraday()

        status = get_job_status()
        assert status["pnl_intraday"]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_users_config_load_failure(self):
        """When config fails to load, falls back to ["jack"]."""
        mock_snapshot = AsyncMock(return_value={"ok": True})

        with patch('builtins.open', side_effect=FileNotFoundError("No config")):
            with patch('api.scheduler._user_has_cointracking_credentials', return_value=True):
                with patch('scripts.pnl_snapshot.create_snapshot', mock_snapshot):
                    await job_pnl_intraday()

        status = get_job_status()
        assert "pnl_intraday" in status
        mock_snapshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_env_source(self, mock_users_config_single):
        mock_snapshot = AsyncMock(return_value={"ok": True})

        with patch.dict(os.environ, {"SNAPSHOT_SOURCE": "manual_crypto"}):
            with patch('builtins.open', mock_open(read_data=mock_users_config_single)):
                with patch('scripts.pnl_snapshot.create_snapshot', mock_snapshot):
                    await job_pnl_intraday()

        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs["source"] == "manual_crypto"

    @pytest.mark.asyncio
    async def test_duration_ms_tracked(self, mock_users_config_single):
        mock_snapshot = AsyncMock(return_value={"ok": True})

        with patch('builtins.open', mock_open(read_data=mock_users_config_single)):
            with patch('api.scheduler._user_has_cointracking_credentials', return_value=True):
                with patch('scripts.pnl_snapshot.create_snapshot', mock_snapshot):
                    await job_pnl_intraday()

        status = get_job_status()
        assert status["pnl_intraday"]["duration_ms"] is not None
        assert status["pnl_intraday"]["duration_ms"] >= 0


class TestJobPnlEod:
    """Tests for job_pnl_eod (similar to intraday but with is_eod=True)."""

    @pytest.mark.asyncio
    async def test_calls_snapshot_with_eod_flag(self, mock_users_config_single):
        mock_snapshot = AsyncMock(return_value={"ok": True})

        with patch('builtins.open', mock_open(read_data=mock_users_config_single)):
            with patch('api.scheduler._user_has_cointracking_credentials', return_value=True):
                with patch('scripts.pnl_snapshot.create_snapshot', mock_snapshot):
                    await job_pnl_eod()

        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs.get("is_eod") is True

    @pytest.mark.asyncio
    async def test_eod_failure_sets_status(self, mock_users_config_single):
        mock_snapshot = AsyncMock(side_effect=Exception("DB error"))

        with patch('builtins.open', mock_open(read_data=mock_users_config_single)):
            with patch('api.scheduler._user_has_cointracking_credentials', return_value=True):
                with patch('scripts.pnl_snapshot.create_snapshot', mock_snapshot):
                    await job_pnl_eod()

        status = get_job_status()
        assert status["pnl_eod"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_eod_no_active_users(self):
        empty_config = json.dumps({"users": []})
        with patch('builtins.open', mock_open(read_data=empty_config)):
            await job_pnl_eod()

        status = get_job_status()
        assert status["pnl_eod"]["status"] == "skipped"


# ============================================================================
# OHLCV JOBS
# ============================================================================

class TestJobOhlcvDaily:
    """Tests for job_ohlcv_daily (subprocess-based)."""

    @pytest.mark.asyncio
    async def test_success(self):
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"OK", b""))

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with patch('asyncio.wait_for', return_value=(b"OK", b"")):
                await job_ohlcv_daily()

        status = get_job_status()
        assert status["ohlcv_daily"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_timeout(self):
        with patch('asyncio.create_subprocess_exec', return_value=AsyncMock()):
            with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
                await job_ohlcv_daily()

        status = get_job_status()
        assert status["ohlcv_daily"]["status"] == "timeout"

    @pytest.mark.asyncio
    async def test_exception(self):
        with patch('asyncio.create_subprocess_exec', side_effect=OSError("No Python")):
            await job_ohlcv_daily()

        status = get_job_status()
        assert status["ohlcv_daily"]["status"] == "error"


class TestJobOhlcvHourly:
    """Tests for job_ohlcv_hourly."""

    @pytest.mark.asyncio
    async def test_success(self):
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"OK", b""))

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with patch('asyncio.wait_for', return_value=(b"OK", b"")):
                await job_ohlcv_hourly()

        status = get_job_status()
        assert status["ohlcv_hourly"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_timeout(self):
        with patch('asyncio.create_subprocess_exec', return_value=AsyncMock()):
            with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
                await job_ohlcv_hourly()

        status = get_job_status()
        assert status["ohlcv_hourly"]["status"] == "timeout"


# ============================================================================
# STALENESS MONITOR
# ============================================================================

class TestJobStalenessMonitor:
    """Tests for job_staleness_monitor."""

    @pytest.mark.asyncio
    async def test_all_fresh(self):
        """When data is recent, status should be success."""
        users_data = [{"id": "jack"}]
        mock_source_info = {"modified_at": time.time(), "effective_path": "/data"}

        mock_sources = MagicMock()
        mock_sources.get_effective_source_info = MagicMock(return_value=mock_source_info)
        mock_user_fs_mod = MagicMock()

        with patch.dict(sys.modules, {
            'api.services.sources_resolver': mock_sources,
            'api.services.user_fs': mock_user_fs_mod,
        }):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.read_text', return_value=json.dumps(users_data)):
                    await job_staleness_monitor()

        status = get_job_status()
        assert status["staleness_monitor"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_stale_source_detected(self):
        """When data is >24h old, status should be warning."""
        users_data = [{"id": "jack"}]
        stale_time = time.time() - (48 * 3600)  # 48 hours ago
        mock_source_info = {"modified_at": stale_time, "effective_path": "/data/old"}

        mock_sources = MagicMock()
        mock_sources.get_effective_source_info = MagicMock(return_value=mock_source_info)
        mock_user_fs_mod = MagicMock()

        with patch.dict(sys.modules, {
            'api.services.sources_resolver': mock_sources,
            'api.services.user_fs': mock_user_fs_mod,
        }):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.read_text', return_value=json.dumps(users_data)):
                    await job_staleness_monitor()

        status = get_job_status()
        assert status["staleness_monitor"]["status"] == "warning"

    @pytest.mark.asyncio
    async def test_no_users_config(self):
        """When config file doesn't exist, should still succeed (0 issues)."""
        mock_sources = MagicMock()
        mock_user_fs_mod = MagicMock()

        with patch.dict(sys.modules, {
            'api.services.sources_resolver': mock_sources,
            'api.services.user_fs': mock_user_fs_mod,
        }):
            with patch('pathlib.Path.exists', return_value=False):
                await job_staleness_monitor()

        status = get_job_status()
        assert status["staleness_monitor"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_import_error_sets_error(self):
        """When imports fail (outer try), status should be error."""
        with patch.dict(sys.modules, {
            'api.services.sources_resolver': None,
        }):
            await job_staleness_monitor()

        status = get_job_status()
        assert status["staleness_monitor"]["status"] == "error"


# ============================================================================
# API WARMERS
# ============================================================================

class TestJobApiWarmers:
    """Tests for job_api_warmers."""

    @pytest.mark.asyncio
    async def test_successful_warmup(self, mock_users_config_single):
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch('builtins.open', mock_open(read_data=mock_users_config_single)):
            with patch('httpx.AsyncClient', return_value=mock_client):
                await job_api_warmers()

        status = get_job_status()
        assert status["api_warmers"]["status"] == "success"
        # 3 endpoints x 1 user = 3 calls
        assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_no_active_users(self):
        empty_config = json.dumps({"users": []})
        with patch('builtins.open', mock_open(read_data=empty_config)):
            await job_api_warmers()

        status = get_job_status()
        assert status["api_warmers"]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_warmup_with_failing_endpoints(self, mock_users_config_single):
        """Individual endpoint failures don't crash the job."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch('builtins.open', mock_open(read_data=mock_users_config_single)):
            with patch('httpx.AsyncClient', return_value=mock_client):
                await job_api_warmers()

        status = get_job_status()
        assert status["api_warmers"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_httpx_client_exception(self, mock_users_config_single):
        """Exception creating httpx client triggers error status."""
        with patch('builtins.open', mock_open(read_data=mock_users_config_single)):
            with patch('httpx.AsyncClient', side_effect=Exception("Cannot create client")):
                await job_api_warmers()

        status = get_job_status()
        assert status["api_warmers"]["status"] == "error"

    @pytest.mark.asyncio
    async def test_uses_env_base_url(self, mock_users_config_single):
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.dict(os.environ, {"API_BASE_URL": "http://custom:9090"}):
            with patch('builtins.open', mock_open(read_data=mock_users_config_single)):
                with patch('httpx.AsyncClient', return_value=mock_client):
                    await job_api_warmers()

        # Verify custom URL was used
        call_args = mock_client.get.call_args_list[0]
        url_arg = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert "http://custom:9090" in str(url_arg)


# ============================================================================
# CRYPTO TOOLBOX
# ============================================================================

class TestJobCryptoToolboxRefresh:
    """Tests for job_crypto_toolbox_refresh."""

    @pytest.mark.asyncio
    async def test_successful_refresh(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"total_count": 35, "critical_count": 5}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch('httpx.AsyncClient', return_value=mock_client):
            await job_crypto_toolbox_refresh()

        status = get_job_status()
        assert status["crypto_toolbox_refresh"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_api_error(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch('httpx.AsyncClient', return_value=mock_client):
            await job_crypto_toolbox_refresh()

        status = get_job_status()
        assert status["crypto_toolbox_refresh"]["status"] == "error"


# ============================================================================
# DAILY ML TRAINING
# ============================================================================

class TestJobDailyMlTraining:
    """Tests for job_daily_ml_training."""

    @pytest.mark.asyncio
    async def test_successful_training(self):
        mock_adapter = MagicMock()
        mock_adapter.detect_market_regime = AsyncMock(return_value={
            "current_regime": "BULL_MARKET",
            "confidence": 0.85,
        })

        mock_module = MagicMock()
        mock_module.StocksMLAdapter = MagicMock(return_value=mock_adapter)

        with patch.dict(sys.modules, {
            'services.ml.bourse.stocks_adapter': mock_module,
        }):
            await job_daily_ml_training()

        status = get_job_status()
        assert status["daily_ml_training"]["status"] == "success"
        mock_adapter.detect_market_regime.assert_called_once_with(
            benchmark="SPY",
            lookback_days=7300,
            force_retrain=True,
        )

    @pytest.mark.asyncio
    async def test_training_failure(self):
        mock_adapter = MagicMock()
        mock_adapter.detect_market_regime = AsyncMock(side_effect=Exception("Model error"))

        mock_module = MagicMock()
        mock_module.StocksMLAdapter = MagicMock(return_value=mock_adapter)

        with patch.dict(sys.modules, {
            'services.ml.bourse.stocks_adapter': mock_module,
        }):
            await job_daily_ml_training()

        status = get_job_status()
        assert status["daily_ml_training"]["status"] == "error"
        assert "Model error" in status["daily_ml_training"]["error"]

    @pytest.mark.asyncio
    async def test_import_failure(self):
        """When StocksMLAdapter can't be imported, job sets error."""
        with patch.dict(sys.modules, {
            'services.ml.bourse.stocks_adapter': None,
        }):
            await job_daily_ml_training()

        status = get_job_status()
        assert status["daily_ml_training"]["status"] == "error"


# ============================================================================
# SCHEDULER LOCK
# ============================================================================

class TestAcquireSchedulerLock:
    """Tests for _acquire_scheduler_lock (Redis distributed lock)."""

    @pytest.mark.asyncio
    async def test_returns_bool(self):
        """Lock acquisition always returns a boolean."""
        result = await _acquire_scheduler_lock()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_empty_redis_url_allows_scheduler(self):
        with patch.dict(os.environ, {"REDIS_URL": ""}):
            result = await _acquire_scheduler_lock()
        assert result is True

    @pytest.mark.asyncio
    async def test_whitespace_redis_url_allows_scheduler(self):
        with patch.dict(os.environ, {"REDIS_URL": "   "}):
            result = await _acquire_scheduler_lock()
        assert result is True

    @pytest.mark.asyncio
    async def test_no_redis_url_uses_default(self):
        """When no env var, function uses default and tries Redis."""
        env = os.environ.copy()
        env.pop("REDIS_URL", None)
        with patch.dict(os.environ, env, clear=True):
            result = await _acquire_scheduler_lock()
        assert isinstance(result, bool)  # True if Redis available, True on error too


class TestRenewSchedulerLock:
    """Tests for _renew_scheduler_lock."""

    @pytest.mark.asyncio
    async def test_does_not_raise(self):
        """Renewal never raises, even if Redis is down."""
        await _renew_scheduler_lock()

    @pytest.mark.asyncio
    async def test_empty_redis_url_skips(self):
        with patch.dict(os.environ, {"REDIS_URL": ""}):
            await _renew_scheduler_lock()  # Should not raise

    @pytest.mark.asyncio
    async def test_whitespace_redis_url_skips(self):
        with patch.dict(os.environ, {"REDIS_URL": "   "}):
            await _renew_scheduler_lock()  # Should not raise


# ============================================================================
# SCHEDULER LIFECYCLE
# ============================================================================

class TestInitializeScheduler:
    """Tests for initialize_scheduler."""

    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        with patch.dict(os.environ, {"RUN_SCHEDULER": "0"}):
            result = await initialize_scheduler()
        assert result is False
        assert get_scheduler() is None

    @pytest.mark.asyncio
    async def test_disabled_when_env_not_set(self):
        env = os.environ.copy()
        env.pop("RUN_SCHEDULER", None)
        with patch.dict(os.environ, env, clear=True):
            result = await initialize_scheduler()
        assert result is False

    @pytest.mark.asyncio
    async def test_initializes_with_flag(self):
        mock_sched = MagicMock()
        mock_sched.get_jobs.return_value = []

        with patch.dict(os.environ, {"RUN_SCHEDULER": "1"}):
            with patch('api.scheduler._acquire_scheduler_lock', new_callable=AsyncMock, return_value=True):
                with patch('api.scheduler.AsyncIOScheduler', return_value=mock_sched):
                    result = await initialize_scheduler()

        assert result is True
        mock_sched.start.assert_called_once()
        # 9 user jobs + 1 lock renewal = 10 add_job calls
        assert mock_sched.add_job.call_count == 10

    @pytest.mark.asyncio
    async def test_already_initialized(self):
        scheduler_module._scheduler = MagicMock()
        with patch.dict(os.environ, {"RUN_SCHEDULER": "1"}):
            result = await initialize_scheduler()
        assert result is True

    @pytest.mark.asyncio
    async def test_lock_not_acquired(self):
        with patch.dict(os.environ, {"RUN_SCHEDULER": "1"}):
            with patch('api.scheduler._acquire_scheduler_lock', new_callable=AsyncMock, return_value=False):
                result = await initialize_scheduler()
        assert result is False

    @pytest.mark.asyncio
    async def test_initialization_exception(self):
        with patch.dict(os.environ, {"RUN_SCHEDULER": "1"}):
            with patch('api.scheduler._acquire_scheduler_lock', new_callable=AsyncMock, return_value=True):
                with patch('api.scheduler.AsyncIOScheduler', side_effect=Exception("APScheduler error")):
                    result = await initialize_scheduler()
        assert result is False
        assert get_scheduler() is None

    @pytest.mark.asyncio
    async def test_job_names_registered(self):
        """Verify all expected job IDs are registered."""
        mock_sched = MagicMock()
        mock_sched.get_jobs.return_value = []

        with patch.dict(os.environ, {"RUN_SCHEDULER": "1"}):
            with patch('api.scheduler._acquire_scheduler_lock', new_callable=AsyncMock, return_value=True):
                with patch('api.scheduler.AsyncIOScheduler', return_value=mock_sched):
                    await initialize_scheduler()

        job_ids = [call[1].get("id") for call in mock_sched.add_job.call_args_list]
        expected_ids = [
            "pnl_intraday", "pnl_eod", "ohlcv_daily", "ohlcv_hourly",
            "staleness_monitor", "api_warmers", "crypto_toolbox_refresh",
            "daily_ml_training", "morning_brief", "scheduler_lock_renewal",
        ]
        for eid in expected_ids:
            assert eid in job_ids, f"Missing job: {eid}"


class TestShutdownScheduler:
    """Tests for shutdown_scheduler."""

    @pytest.mark.asyncio
    async def test_shutdown_when_not_running(self):
        await shutdown_scheduler()
        assert get_scheduler() is None

    @pytest.mark.asyncio
    async def test_shutdown_running_scheduler(self):
        mock_sched = MagicMock()
        scheduler_module._scheduler = mock_sched

        await shutdown_scheduler()

        mock_sched.shutdown.assert_called_once_with(wait=True)
        assert get_scheduler() is None

        mock_sched.shutdown.assert_called_once()
        assert get_scheduler() is None

    @pytest.mark.asyncio
    async def test_shutdown_exception(self):
        mock_sched = MagicMock()
        mock_sched.shutdown.side_effect = Exception("Shutdown error")
        scheduler_module._scheduler = mock_sched

        await shutdown_scheduler()  # Should not raise


# ============================================================================
# REDIS PERSISTENCE (5.2)
# ============================================================================

class TestGetRedisClient:
    """Tests for _get_redis_client."""

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_url(self):
        with patch.dict(os.environ, {"REDIS_URL": ""}):
            assert await _get_redis_client() is None

    @pytest.mark.asyncio
    async def test_returns_none_on_whitespace_url(self):
        with patch.dict(os.environ, {"REDIS_URL": "   "}):
            assert await _get_redis_client() is None


class TestUpdateJobStatusRedis:
    """Tests for Redis persistence in _update_job_status."""

    @pytest.mark.asyncio
    async def test_persists_to_redis_hash(self):
        mock_client = AsyncMock()
        with patch("api.scheduler._get_redis_client", new_callable=AsyncMock, return_value=mock_client):
            with patch("api.scheduler._notify_job_failure", new_callable=AsyncMock):
                await _update_job_status("test_job", "success", duration_ms=100)

        mock_client.hset.assert_called_once()
        args = mock_client.hset.call_args
        assert args[0][0] == REDIS_JOB_STATUS_KEY
        assert args[0][1] == "test_job"
        data = json.loads(args[0][2])
        assert data["status"] == "success"
        assert data["duration_ms"] == 100
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_works_when_redis_unavailable(self):
        """In-memory dict still updated when Redis is down."""
        with patch("api.scheduler._get_redis_client", new_callable=AsyncMock, return_value=None):
            with patch("api.scheduler._notify_job_failure", new_callable=AsyncMock):
                await _update_job_status("test_job", "success", duration_ms=50)

        status = get_job_status()
        assert status["test_job"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_redis_error_does_not_crash(self):
        mock_client = AsyncMock()
        mock_client.hset.side_effect = Exception("Redis write error")
        with patch("api.scheduler._get_redis_client", new_callable=AsyncMock, return_value=mock_client):
            with patch("api.scheduler._notify_job_failure", new_callable=AsyncMock):
                await _update_job_status("test_job", "success")

        # In-memory still updated
        assert get_job_status()["test_job"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_triggers_webhook_on_error(self):
        mock_notify = AsyncMock()
        with patch("api.scheduler._get_redis_client", new_callable=AsyncMock, return_value=None):
            with patch("api.scheduler._notify_job_failure", mock_notify):
                await _update_job_status("job1", "error", error="timeout")

        mock_notify.assert_called_once()
        assert mock_notify.call_args[0][0] == "job1"
        assert mock_notify.call_args[0][1]["status"] == "error"

    @pytest.mark.asyncio
    async def test_triggers_webhook_on_failed(self):
        mock_notify = AsyncMock()
        with patch("api.scheduler._get_redis_client", new_callable=AsyncMock, return_value=None):
            with patch("api.scheduler._notify_job_failure", mock_notify):
                await _update_job_status("job1", "failed", error="all failed")

        mock_notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_webhook_on_success(self):
        mock_notify = AsyncMock()
        with patch("api.scheduler._get_redis_client", new_callable=AsyncMock, return_value=None):
            with patch("api.scheduler._notify_job_failure", mock_notify):
                await _update_job_status("job1", "success")

        mock_notify.assert_not_called()


class TestGetJobStatusPersistent:
    """Tests for get_job_status_persistent (Redis-backed)."""

    @pytest.mark.asyncio
    async def test_reads_from_redis(self):
        redis_data = {
            b"job1": json.dumps({"status": "success", "last_run": "2026-01-01T00:00:00"}).encode(),
            b"job2": json.dumps({"status": "error", "last_run": "2026-01-01T01:00:00"}).encode(),
        }
        mock_client = AsyncMock()
        mock_client.hgetall = AsyncMock(return_value=redis_data)

        with patch("api.scheduler._get_redis_client", new_callable=AsyncMock, return_value=mock_client):
            result = await get_job_status_persistent()

        assert "job1" in result
        assert result["job1"]["status"] == "success"
        assert "job2" in result
        assert result["job2"]["status"] == "error"
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_memory(self):
        scheduler_module._job_status["mem_job"] = {"status": "ok", "last_run": "now"}

        with patch("api.scheduler._get_redis_client", new_callable=AsyncMock, return_value=None):
            result = await get_job_status_persistent()

        assert "mem_job" in result
        assert result["mem_job"]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_falls_back_on_redis_error(self):
        scheduler_module._job_status["mem_job"] = {"status": "ok"}
        mock_client = AsyncMock()
        mock_client.hgetall = AsyncMock(side_effect=Exception("Redis error"))

        with patch("api.scheduler._get_redis_client", new_callable=AsyncMock, return_value=mock_client):
            result = await get_job_status_persistent()

        assert "mem_job" in result


class TestNotifyJobFailure:
    """Tests for _notify_job_failure webhook."""

    @pytest.mark.asyncio
    async def test_sends_webhook(self):
        mock_response = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.dict(os.environ, {"SCHEDULER_FAILURE_WEBHOOK": "http://hooks.example.com/alert"}):
            with patch("httpx.AsyncClient", return_value=mock_client):
                await _notify_job_failure("test_job", {
                    "status": "error", "error": "boom", "last_run": "now"
                })

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://hooks.example.com/alert"
        payload = call_args[1]["json"]
        assert payload["job_id"] == "test_job"
        assert "boom" in payload["text"]

    @pytest.mark.asyncio
    async def test_noop_without_webhook_url(self):
        env = os.environ.copy()
        env.pop("SCHEDULER_FAILURE_WEBHOOK", None)
        with patch.dict(os.environ, env, clear=True):
            await _notify_job_failure("test_job", {
                "status": "error", "error": "boom", "last_run": "now"
            })
        # Should not raise

    @pytest.mark.asyncio
    async def test_webhook_failure_does_not_raise(self):
        with patch.dict(os.environ, {"SCHEDULER_FAILURE_WEBHOOK": "http://hooks.example.com/alert"}):
            with patch("httpx.AsyncClient", side_effect=Exception("Network error")):
                await _notify_job_failure("test_job", {
                    "status": "error", "error": "boom", "last_run": "now"
                })
        # Should not raise


class TestRecoverJobStatusFromRedis:
    """Tests for _recover_job_status_from_redis (startup recovery)."""

    @pytest.mark.asyncio
    async def test_loads_status_into_memory(self):
        redis_data = {
            b"recovered_job": json.dumps({
                "status": "success", "last_run": "2026-01-01T00:00:00",
                "duration_ms": 200, "error": None
            }).encode(),
        }
        mock_client = AsyncMock()
        mock_client.hgetall = AsyncMock(return_value=redis_data)

        with patch("api.scheduler._get_redis_client", new_callable=AsyncMock, return_value=mock_client):
            await _recover_job_status_from_redis()

        assert "recovered_job" in scheduler_module._job_status
        assert scheduler_module._job_status["recovered_job"]["status"] == "success"
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_noop_when_redis_unavailable(self):
        with patch("api.scheduler._get_redis_client", new_callable=AsyncMock, return_value=None):
            await _recover_job_status_from_redis()
        assert scheduler_module._job_status == {}

    @pytest.mark.asyncio
    async def test_noop_on_redis_error(self):
        mock_client = AsyncMock()
        mock_client.hgetall = AsyncMock(side_effect=Exception("Connection lost"))

        with patch("api.scheduler._get_redis_client", new_callable=AsyncMock, return_value=mock_client):
            await _recover_job_status_from_redis()
        # Should not raise, _job_status stays empty


class TestGetSchedulerHeartbeat:
    """Tests for get_scheduler_heartbeat."""

    @pytest.mark.asyncio
    async def test_reads_heartbeat(self):
        heartbeat_data = json.dumps({
            "pid": 1234, "timestamp": "2026-01-01T00:00:00", "jobs_count": 5
        })
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=heartbeat_data.encode())

        with patch("api.scheduler._get_redis_client", new_callable=AsyncMock, return_value=mock_client):
            result = await get_scheduler_heartbeat()

        assert result["pid"] == 1234
        assert result["jobs_count"] == 5
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_unavailable(self):
        with patch("api.scheduler._get_redis_client", new_callable=AsyncMock, return_value=None):
            result = await get_scheduler_heartbeat()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_redis_error(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("timeout"))

        with patch("api.scheduler._get_redis_client", new_callable=AsyncMock, return_value=mock_client):
            result = await get_scheduler_heartbeat()
        assert result is None

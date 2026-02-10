"""
API Scheduler - Periodic Task Orchestration
Created: Oct 2025

Handles:
- P&L snapshots (intraday 15min, EOD 23:59)
- OHLCV updates (daily 03:10, hourly :05)
- Staleness monitoring (hourly)
- API warmers (startup + periodic 5-10min)

Configuration:
- RUN_SCHEDULER=1 to enable (prevents double execution in dev)
- Timezone: Europe/Zurich
- Guards: coalesce=True, max_instances=1, misfire_grace_time=300s
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from api.config.users import is_allowed_user, validate_user_id

logger = logging.getLogger(__name__)


def _user_has_cointracking_credentials(user_id: str) -> bool:
    """
    Check if a user has CoinTracking API credentials configured.

    Returns:
        bool: True if user has valid API credentials
    """
    try:
        from api.services.config_migrator import resolve_secret_ref
        from api.services.user_fs import UserScopedFS
        from pathlib import Path

        project_root = Path(__file__).parent.parent
        user_fs = UserScopedFS(str(project_root), user_id)

        key = resolve_secret_ref("cointracking_api_key", user_fs)
        secret = resolve_secret_ref("cointracking_api_secret", user_fs)

        return bool(key and secret)
    except Exception as e:
        logger.debug(f"Credentials check failed for {user_id}: {e}")
        return False

# Singleton scheduler instance
_scheduler: Optional[AsyncIOScheduler] = None
_job_status: Dict[str, Dict[str, Any]] = {}

# Redis keys for persistent job status
REDIS_JOB_STATUS_KEY = "smartfolio:scheduler:jobs"
REDIS_HEARTBEAT_KEY = "smartfolio:scheduler:heartbeat"


async def _get_redis_client():
    """Get Redis async client, or None if unavailable."""
    try:
        import redis.asyncio as aioredis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        if not redis_url or not redis_url.strip():
            return None
        client = aioredis.from_url(redis_url, socket_connect_timeout=2)
        await client.ping()
        return client
    except Exception:
        return None


def get_scheduler() -> Optional[AsyncIOScheduler]:
    """Get the singleton scheduler instance"""
    return _scheduler


def get_job_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all scheduled jobs (for health endpoint)"""
    return _job_status.copy()


async def _update_job_status(job_id: str, status: str, duration_ms: Optional[float] = None, error: Optional[str] = None):
    """Update job execution status in memory and persist to Redis."""
    entry = {
        "last_run": datetime.now().isoformat(),
        "status": status,
        "duration_ms": duration_ms,
        "error": error
    }
    _job_status[job_id] = entry

    # Persist to Redis HASH (best-effort)
    try:
        client = await _get_redis_client()
        if client:
            try:
                await client.hset(REDIS_JOB_STATUS_KEY, job_id, json.dumps(entry))
            finally:
                await client.aclose()
    except Exception:
        pass

    # Webhook alert on failure
    if status in ("error", "failed"):
        await _notify_job_failure(job_id, entry)


async def get_job_status_persistent() -> Dict[str, Dict[str, Any]]:
    """Get job status from Redis (persistent across restarts), fallback to in-memory."""
    try:
        client = await _get_redis_client()
        if client:
            try:
                raw = await client.hgetall(REDIS_JOB_STATUS_KEY)
                if raw:
                    return {
                        (k.decode() if isinstance(k, bytes) else k):
                        json.loads(v.decode() if isinstance(v, bytes) else v)
                        for k, v in raw.items()
                    }
            finally:
                await client.aclose()
    except Exception:
        pass
    return _job_status.copy()


async def get_scheduler_heartbeat() -> Optional[Dict[str, Any]]:
    """Get scheduler heartbeat from Redis."""
    try:
        client = await _get_redis_client()
        if client:
            try:
                raw = await client.get(REDIS_HEARTBEAT_KEY)
                if raw:
                    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            finally:
                await client.aclose()
    except Exception:
        pass
    return None


async def _notify_job_failure(job_id: str, entry: dict):
    """Send webhook notification when a scheduler job fails."""
    try:
        webhook_url = os.getenv("SCHEDULER_FAILURE_WEBHOOK")
        if not webhook_url:
            return

        import httpx
        payload = {
            "text": f"Scheduler job '{job_id}' {entry['status']}: {entry.get('error', 'Unknown error')}",
            "job_id": job_id,
            "status": entry["status"],
            "error": entry.get("error"),
            "timestamp": entry["last_run"],
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(webhook_url, json=payload)
    except Exception as e:
        logger.debug(f"Failed to send failure webhook for {job_id}: {e}")


async def _recover_job_status_from_redis():
    """Load last known job status from Redis at startup (recovery)."""
    try:
        client = await _get_redis_client()
        if client:
            try:
                raw = await client.hgetall(REDIS_JOB_STATUS_KEY)
                if raw:
                    for k, v in raw.items():
                        key = k.decode() if isinstance(k, bytes) else k
                        _job_status[key] = json.loads(v.decode() if isinstance(v, bytes) else v)
                    logger.info(f"Recovered {len(raw)} job statuses from Redis")
            finally:
                await client.aclose()
    except Exception as e:
        logger.debug(f"Redis recovery unavailable: {e}")


# ============================================================================
# JOB IMPLEMENTATIONS
# ============================================================================

async def job_pnl_intraday():
    """P&L snapshot intraday (every 15 min, 07:00-23:59 Europe/Zurich)"""
    job_id = "pnl_intraday"
    start = datetime.now()

    try:
        logger.info(f"🔄 [{job_id}] Starting P&L intraday snapshot...")

        # Import here to avoid circular dependencies
        from scripts.pnl_snapshot import create_snapshot
        import json

        # Load active users from config
        try:
            with open("config/users.json", "r", encoding="utf-8") as f:
                users_config = json.load(f)
                active_users = [
                    user["id"]
                    for user in users_config.get("users", [])
                    if user.get("status") == "active"
                ]
        except Exception as e:
            logger.warning(f"⚠️ [{job_id}] Failed to load users config: {e}, using fallback")
            active_users = ["jack"]

        if not active_users:
            logger.warning(f"⚠️ [{job_id}] No active users found, skipping")
            await _update_job_status(job_id, "skipped", 0, "No active users")
            return

        # Default params (can be overridden via env vars)
        source = os.getenv("SNAPSHOT_SOURCE", "cointracking_api")
        min_usd = float(os.getenv("SNAPSHOT_MIN_USD", "1.0"))

        # Filter users with valid credentials for cointracking_api source
        if source == "cointracking_api":
            users_with_credentials = [u for u in active_users if _user_has_cointracking_credentials(u)]
            skipped_users = [u for u in active_users if u not in users_with_credentials]
            if skipped_users:
                logger.info(f"   ℹ️ Skipping {len(skipped_users)} users without API credentials: {', '.join(skipped_users)}")
            active_users = users_with_credentials

        if not active_users:
            logger.info(f"ℹ️ [{job_id}] No users with valid credentials for source={source}, skipping")
            await _update_job_status(job_id, "skipped", 0, "No users with credentials")
            return

        logger.info(f"   Creating P&L snapshots for {len(active_users)} users: {', '.join(active_users)}")

        # Process snapshots for all active users
        success_count = 0
        fail_count = 0
        errors = []

        for user_id in active_users:
            try:
                result = await create_snapshot(user_id=user_id, source=source, min_usd=min_usd)
                if result.get("ok"):
                    logger.debug(f"   ✅ Snapshot created for [{user_id}]")
                    success_count += 1
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.warning(f"   ⚠️ Snapshot failed for [{user_id}]: {error_msg}")
                    fail_count += 1
                    errors.append(f"{user_id}: {error_msg}")
            except Exception as e:
                logger.warning(f"   ❌ Snapshot exception for [{user_id}]: {e}")
                fail_count += 1
                errors.append(f"{user_id}: {str(e)}")

        duration_ms = (datetime.now() - start).total_seconds() * 1000

        if fail_count == 0:
            logger.info(f"✅ [{job_id}] All {success_count} P&L snapshots completed in {duration_ms:.0f}ms")
            await _update_job_status(job_id, "success", duration_ms)
        elif success_count > 0:
            error_summary = f"{success_count} OK, {fail_count} failed: {'; '.join(errors[:3])}"
            logger.warning(f"⚠️ [{job_id}] Partial success: {error_summary}")
            await _update_job_status(job_id, "partial", duration_ms, error_summary)
        else:
            error_summary = f"All {fail_count} snapshots failed: {'; '.join(errors[:3])}"
            logger.error(f"❌ [{job_id}] {error_summary}")
            await _update_job_status(job_id, "failed", duration_ms, error_summary)

    except Exception as e:
        duration_ms = (datetime.now() - start).total_seconds() * 1000
        logger.exception(f"❌ [{job_id}] P&L snapshot exception")
        await _update_job_status(job_id, "error", duration_ms, str(e))


async def job_pnl_eod():
    """P&L snapshot EOD (daily at 23:59 Europe/Zurich)"""
    job_id = "pnl_eod"
    start = datetime.now()

    try:
        logger.info(f"🔄 [{job_id}] Starting P&L EOD snapshot...")

        from scripts.pnl_snapshot import create_snapshot
        import json

        # Load active users from config
        try:
            with open("config/users.json", "r", encoding="utf-8") as f:
                users_config = json.load(f)
                active_users = [
                    user["id"]
                    for user in users_config.get("users", [])
                    if user.get("status") == "active"
                ]
        except Exception as e:
            logger.warning(f"⚠️ [{job_id}] Failed to load users config: {e}, using fallback")
            active_users = ["jack"]

        if not active_users:
            logger.warning(f"⚠️ [{job_id}] No active users found, skipping")
            await _update_job_status(job_id, "skipped", 0, "No active users")
            return

        # Default params (can be overridden via env vars)
        source = os.getenv("SNAPSHOT_SOURCE", "cointracking_api")
        min_usd = float(os.getenv("SNAPSHOT_MIN_USD", "1.0"))

        # Filter users with valid credentials for cointracking_api source
        if source == "cointracking_api":
            users_with_credentials = [u for u in active_users if _user_has_cointracking_credentials(u)]
            skipped_users = [u for u in active_users if u not in users_with_credentials]
            if skipped_users:
                logger.info(f"   ℹ️ Skipping {len(skipped_users)} users without API credentials: {', '.join(skipped_users)}")
            active_users = users_with_credentials

        if not active_users:
            logger.info(f"ℹ️ [{job_id}] No users with valid credentials for source={source}, skipping")
            await _update_job_status(job_id, "skipped", 0, "No users with credentials")
            return

        logger.info(f"   Creating EOD P&L snapshots for {len(active_users)} users: {', '.join(active_users)}")

        # Process EOD snapshots for all active users
        success_count = 0
        fail_count = 0
        errors = []

        for user_id in active_users:
            try:
                result = await create_snapshot(user_id=user_id, source=source, min_usd=min_usd, is_eod=True)
                if result.get("ok"):
                    logger.debug(f"   ✅ EOD snapshot created for [{user_id}]")
                    success_count += 1
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.warning(f"   ⚠️ EOD snapshot failed for [{user_id}]: {error_msg}")
                    fail_count += 1
                    errors.append(f"{user_id}: {error_msg}")
            except Exception as e:
                logger.warning(f"   ❌ EOD snapshot exception for [{user_id}]: {e}")
                fail_count += 1
                errors.append(f"{user_id}: {str(e)}")

        duration_ms = (datetime.now() - start).total_seconds() * 1000

        if fail_count == 0:
            logger.info(f"✅ [{job_id}] All {success_count} EOD P&L snapshots completed in {duration_ms:.0f}ms")
            await _update_job_status(job_id, "success", duration_ms)
        elif success_count > 0:
            error_summary = f"{success_count} OK, {fail_count} failed: {'; '.join(errors[:3])}"
            logger.warning(f"⚠️ [{job_id}] Partial success: {error_summary}")
            await _update_job_status(job_id, "partial", duration_ms, error_summary)
        else:
            error_summary = f"All {fail_count} snapshots failed: {'; '.join(errors[:3])}"
            logger.error(f"❌ [{job_id}] {error_summary}")
            await _update_job_status(job_id, "failed", duration_ms, error_summary)

    except Exception as e:
        duration_ms = (datetime.now() - start).total_seconds() * 1000
        logger.exception(f"❌ [{job_id}] P&L EOD snapshot exception")
        await _update_job_status(job_id, "error", duration_ms, str(e))


async def job_ohlcv_daily():
    """OHLCV update daily (03:10 Europe/Zurich)"""
    job_id = "ohlcv_daily"
    start = datetime.now()

    try:
        logger.info(f"🔄 [{job_id}] Starting OHLCV daily update...")

        # Import and run the existing script
        import sys
        from pathlib import Path

        script_path = Path(__file__).parent.parent / "scripts" / "update_price_history.py"

        # PERFORMANCE FIX (Dec 2025): Non-blocking subprocess with asyncio
        # Prevents event loop freeze during 5-minute script execution
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Wait for completion with timeout
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=300  # 5 min max
        )

        duration_ms = (datetime.now() - start).total_seconds() * 1000
        stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
        stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""

        if process.returncode == 0:
            logger.info(f"✅ [{job_id}] OHLCV daily update completed in {duration_ms:.0f}ms")
            await _update_job_status(job_id, "success", duration_ms)
        else:
            logger.error(f"❌ [{job_id}] OHLCV daily update failed:\n{stderr_text}")
            await _update_job_status(job_id, "failed", duration_ms, stderr_text[:200])

    except asyncio.TimeoutError:
        duration_ms = (datetime.now() - start).total_seconds() * 1000
        logger.error(f"❌ [{job_id}] OHLCV daily update timeout (>5min)")
        await _update_job_status(job_id, "timeout", duration_ms, "Timeout after 5 minutes")

    except Exception as e:
        duration_ms = (datetime.now() - start).total_seconds() * 1000
        logger.exception(f"❌ [{job_id}] OHLCV daily update exception")
        await _update_job_status(job_id, "error", duration_ms, str(e))


async def job_ohlcv_hourly():
    """OHLCV update hourly (every hour at :05)"""
    job_id = "ohlcv_hourly"
    start = datetime.now()

    try:
        logger.info(f"🔄 [{job_id}] Starting OHLCV hourly update...")

        import sys
        from pathlib import Path

        script_path = Path(__file__).parent.parent / "scripts" / "update_price_history.py"

        # PERFORMANCE FIX (Dec 2025): Non-blocking subprocess with asyncio
        # Prevents event loop freeze during 2-minute script execution
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Wait for completion with timeout
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=120  # 2 min max for update
        )

        duration_ms = (datetime.now() - start).total_seconds() * 1000
        stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
        stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""

        if process.returncode == 0:
            logger.info(f"✅ [{job_id}] OHLCV hourly update completed in {duration_ms:.0f}ms")
            await _update_job_status(job_id, "success", duration_ms)
        else:
            logger.error(f"❌ [{job_id}] OHLCV hourly update failed:\n{stderr_text}")
            await _update_job_status(job_id, "failed", duration_ms, stderr_text[:200])

    except asyncio.TimeoutError:
        duration_ms = (datetime.now() - start).total_seconds() * 1000
        logger.error(f"❌ [{job_id}] OHLCV hourly update timeout (>2min)")
        await _update_job_status(job_id, "timeout", duration_ms, "Timeout after 2 minutes")

    except Exception as e:
        duration_ms = (datetime.now() - start).total_seconds() * 1000
        logger.exception(f"❌ [{job_id}] OHLCV hourly update exception")
        await _update_job_status(job_id, "error", duration_ms, str(e))


async def job_staleness_monitor():
    """Staleness monitoring (hourly)"""
    job_id = "staleness_monitor"
    start = datetime.now()

    try:
        logger.info(f"🔄 [{job_id}] Starting staleness monitoring...")

        # Check Saxo data staleness
        from api.services.sources_resolver import get_effective_source_info
        from api.services.user_fs import UserScopedFS

        # Get all users and check their Saxo sources
        saxo_issues = []

        try:
            from pathlib import Path
            import json
            import time

            users_config_path = Path("config/users.json")
            if users_config_path.exists():
                users_data = json.loads(users_config_path.read_text())
                users_list = users_data.get("users", []) if isinstance(users_data, dict) else users_data

                for user in users_list:
                    user_id = user.get("id")
                    if not user_id:
                        continue

                    # Check Saxo source
                    try:
                        # Create user filesystem
                        project_root = Path(__file__).parent.parent
                        user_fs = UserScopedFS(str(project_root), user_id)

                        # Get source info
                        result = get_effective_source_info(user_fs, "saxobank")

                        # Calculate staleness in hours
                        if result.get("modified_at"):
                            staleness_hours = (time.time() - result["modified_at"]) / 3600

                            if staleness_hours > 24:
                                saxo_issues.append({
                                    "user_id": user_id,
                                    "staleness_hours": staleness_hours,
                                    "path": result.get("effective_path")
                                })
                    except Exception as e:
                        logger.warning(f"Failed to check Saxo staleness for user {user_id}: {e}")

        except Exception as e:
            logger.warning(f"Failed to load users config: {e}")

        duration_ms = (datetime.now() - start).total_seconds() * 1000

        if saxo_issues:
            logger.warning(f"⚠️ [{job_id}] Found {len(saxo_issues)} stale Saxo sources")
            for issue in saxo_issues:
                logger.warning(f"   - {issue['user_id']}: {issue['staleness_hours']:.1f}h stale")
            await _update_job_status(job_id, "warning", duration_ms, f"{len(saxo_issues)} stale sources")
        else:
            logger.info(f"✅ [{job_id}] Staleness check completed in {duration_ms:.0f}ms - all fresh")
            await _update_job_status(job_id, "success", duration_ms)

    except Exception as e:
        duration_ms = (datetime.now() - start).total_seconds() * 1000
        logger.exception(f"❌ [{job_id}] Staleness monitoring exception")
        await _update_job_status(job_id, "error", duration_ms, str(e))


async def job_api_warmers():
    """API warmers - keep caches warm (every 5-10 min)"""
    job_id = "api_warmers"
    start = datetime.now()

    try:
        logger.info(f"🔄 [{job_id}] Starting API warmers...")

        import httpx
        import json

        # Load active users from config
        try:
            with open("config/users.json", "r", encoding="utf-8") as f:
                users_config = json.load(f)
                active_users = [
                    user["id"]
                    for user in users_config.get("users", [])
                    if user.get("status") == "active"
                ]
        except Exception as e:
            logger.warning(f"⚠️ [{job_id}] Failed to load users config: {e}, using fallback")
            active_users = ["jack"]  # Fallback to jack if config fails

        if not active_users:
            logger.warning(f"⚠️ [{job_id}] No active users found, skipping warmers")
            await _update_job_status(job_id, "skipped", 0, "No active users")
            return

        logger.info(f"   Warming caches for {len(active_users)} active users: {', '.join(active_users)}")

        # Critical endpoints to warm (use Depends(get_required_user) - need X-User header)
        endpoint_templates = [
            "/balances/current?source=cointracking",
            "/portfolio/metrics?source=cointracking",
            "/api/risk/dashboard?source=cointracking",
        ]

        base_url = os.getenv("API_BASE_URL", "http://localhost:8080")

        # PERFORMANCE FIX: Parallelize API warmup calls for all users
        async def warm_endpoint(client: httpx.AsyncClient, endpoint: str, user_id: str):
            """Warm a single endpoint for a specific user"""
            try:
                url = f"{base_url}{endpoint}"
                headers = {"X-User": user_id}
                response = await client.get(url, headers=headers)

                if response.status_code == 200:
                    logger.debug(f"   ✅ Warmed [{user_id}]: {endpoint}")
                else:
                    logger.warning(f"   ⚠️ Warm failed [{user_id}] ({response.status_code}): {endpoint}")
            except Exception as e:
                logger.warning(f"   ❌ Warm error [{user_id}]: {endpoint} - {e}")

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Create warmup tasks for all users x endpoints
            tasks = [
                warm_endpoint(client, endpoint, user_id)
                for user_id in active_users
                for endpoint in endpoint_templates
            ]
            # Execute all warmup calls in parallel
            await asyncio.gather(*tasks, return_exceptions=True)

        duration_ms = (datetime.now() - start).total_seconds() * 1000
        logger.info(f"✅ [{job_id}] API warmers completed in {duration_ms:.0f}ms")
        await _update_job_status(job_id, "success", duration_ms)

    except Exception as e:
        duration_ms = (datetime.now() - start).total_seconds() * 1000
        logger.exception(f"❌ [{job_id}] API warmers exception")
        await _update_job_status(job_id, "error", duration_ms, str(e))


async def job_crypto_toolbox_refresh():
    """
    Refresh crypto-toolbox indicators (2x daily: 08:00 & 20:00)
    Scrapes 30+ on-chain indicators from crypto-toolbox.vercel.app
    """
    job_id = "crypto_toolbox_refresh"
    start = datetime.now()

    try:
        logger.info(f"🔄 [{job_id}] Starting crypto-toolbox indicators refresh...")

        import httpx

        # Call the FastAPI crypto-toolbox endpoint with force refresh
        base_url = os.getenv("API_BASE_URL", "http://localhost:8080")
        url = f"{base_url}/api/crypto-toolbox?force=true"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

        duration_ms = (datetime.now() - start).total_seconds() * 1000

        indicators_count = data.get("total_count", 0)
        critical_count = data.get("critical_count", 0)

        logger.info(f"✅ [{job_id}] Crypto-toolbox refresh completed in {duration_ms:.0f}ms")
        logger.info(f"   📊 {indicators_count} indicators scraped ({critical_count} critical)")
        await _update_job_status(job_id, "success", duration_ms)

    except Exception as e:
        duration_ms = (datetime.now() - start).total_seconds() * 1000
        logger.exception(f"❌ [{job_id}] Crypto-toolbox refresh failed")
        await _update_job_status(job_id, "error", duration_ms, str(e))


async def job_daily_ml_training():
    """
    Entraîne les modèles ML lourds chaque jour à 3h du matin.

    - Regime detection (20 ans, ~60-90s)
    - Correlation forecaster (20 ans, ~30-40s)

    Total: ~2 minutes par jour
    """
    job_id = "daily_ml_training"
    start = datetime.now()

    try:
        logger.info(f"🤖 [{job_id}] Starting daily ML training (20 years data)...")

        from services.ml.bourse.stocks_adapter import StocksMLAdapter

        adapter = StocksMLAdapter()

        # Force retrain regime detection with 20 years of data
        regime_result = await adapter.detect_market_regime(
            benchmark="SPY",
            lookback_days=7300,  # 20 ans
            force_retrain=True   # Ignore cache age
        )

        duration_ms = (datetime.now() - start).total_seconds() * 1000

        logger.info(f"✅ [{job_id}] Regime model trained: {regime_result['current_regime']} "
                   f"({regime_result['confidence']:.1%} confidence) in {duration_ms:.0f}ms")

        await _update_job_status(job_id, "success", duration_ms)

        # TODO: Ajouter correlation forecaster si nécessaire
        # await adapter.forecast_correlation([...], force_retrain=True)

    except Exception as e:
        duration_ms = (datetime.now() - start).total_seconds() * 1000
        logger.exception(f"❌ [{job_id}] Daily ML training failed")
        await _update_job_status(job_id, "error", duration_ms, str(e))
        # Ne pas lever exception - retry demain


async def job_morning_brief():
    """Morning Brief generation (daily at 07:30 Europe/Zurich)"""
    job_id = "morning_brief"
    start = datetime.now()

    try:
        logger.info(f"🔄 [{job_id}] Starting Morning Brief generation...")

        import json
        from services.morning_brief_service import morning_brief_service

        # Load active users
        try:
            with open("config/users.json", "r", encoding="utf-8") as f:
                users_config = json.load(f)
                active_users = [
                    user["id"]
                    for user in users_config.get("users", [])
                    if user.get("status") == "active"
                ]
        except Exception as e:
            logger.warning(f"⚠️ [{job_id}] Failed to load users config: {e}, using fallback")
            active_users = ["jack"]

        if not active_users:
            await _update_job_status(job_id, "skipped", 0, "No active users")
            return

        success_count = 0
        for user_id in active_users:
            try:
                brief = await morning_brief_service.generate(user_id=user_id)

                # Try to send via configured channels
                try:
                    user_config_path = f"data/users/{user_id}/config.json"
                    import os
                    if os.path.exists(user_config_path):
                        with open(user_config_path, "r", encoding="utf-8") as f:
                            user_cfg = json.load(f)
                        if user_cfg.get("morning_brief", {}).get("send_telegram", False):
                            from services.notifications.notification_sender import notification_sender
                            from services.notifications.alert_manager import Alert, AlertLevel, AlertType

                            channels = user_cfg.get("notifications", {}).get("channels", {})
                            tg_config = channels.get("telegram", {})
                            if tg_config.get("enabled") and tg_config.get("chat_id"):
                                message = morning_brief_service.format_telegram(brief)
                                tg_notifier = notification_sender.channels.get("telegram")
                                if tg_notifier:
                                    send_config = {k: v for k, v in tg_config.items() if k != "enabled"}
                                    notif_alert = Alert(
                                        type=AlertType.SYSTEM_ERROR,
                                        level=AlertLevel.INFO,
                                        source="morning_brief",
                                        title="Morning Brief",
                                        message=message,
                                        data={"morning_brief": True},
                                        actions=[],
                                    )
                                    await tg_notifier.send(notif_alert, send_config)
                                    logger.info(f"   📱 Morning brief sent via Telegram for [{user_id}]")
                except Exception as send_err:
                    logger.warning(f"   ⚠️ Morning brief notification failed for [{user_id}]: {send_err}")

                success_count += 1
                logger.debug(f"   ✅ Morning brief generated for [{user_id}]")
            except Exception as e:
                logger.warning(f"   ❌ Morning brief failed for [{user_id}]: {e}")

        duration_ms = (datetime.now() - start).total_seconds() * 1000
        if success_count == len(active_users):
            logger.info(f"✅ [{job_id}] Morning Brief completed in {duration_ms:.0f}ms ({success_count} users)")
            await _update_job_status(job_id, "success", duration_ms)
        else:
            logger.warning(f"⚠️ [{job_id}] Morning Brief partial: {success_count}/{len(active_users)}")
            await _update_job_status(job_id, "partial", duration_ms)

    except Exception as e:
        duration_ms = (datetime.now() - start).total_seconds() * 1000
        logger.exception(f"❌ [{job_id}] Morning Brief exception")
        await _update_job_status(job_id, "error", duration_ms, str(e))


# ============================================================================
# SCHEDULER LIFECYCLE
# ============================================================================

async def _acquire_scheduler_lock() -> bool:
    """Try to acquire a Redis distributed lock for scheduler exclusivity.

    Prevents duplicate schedulers when running multiple uvicorn workers.
    Uses Redis SET NX with a 120s TTL (auto-expires if process dies).
    Returns True if lock acquired (or Redis unavailable — single-worker fallback).
    """
    try:
        import redis.asyncio as aioredis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        if not redis_url or not redis_url.strip():
            return True  # No Redis configured, allow scheduler

        client = aioredis.from_url(redis_url, socket_connect_timeout=2)
        # SET NX: only set if key doesn't exist. TTL 120s = auto-release on crash.
        acquired = await client.set("smartfolio:scheduler_lock", os.getpid(), nx=True, ex=120)
        await client.aclose()

        if not acquired:
            logger.info("Scheduler lock held by another worker — skipping scheduler init")
        return bool(acquired)

    except Exception as e:
        logger.debug(f"Redis lock unavailable ({e}) — allowing scheduler (single-worker mode)")
        return True  # Redis down = assume single worker


async def _renew_scheduler_lock():
    """Periodically renew the scheduler lock TTL and update heartbeat."""
    try:
        import redis.asyncio as aioredis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        if not redis_url or not redis_url.strip():
            return

        client = aioredis.from_url(redis_url, socket_connect_timeout=2)
        await client.expire("smartfolio:scheduler_lock", 120)
        # Heartbeat: write PID + timestamp so admin can see scheduler is alive
        await client.set(
            REDIS_HEARTBEAT_KEY,
            json.dumps({
                "pid": os.getpid(),
                "timestamp": datetime.now().isoformat(),
                "jobs_count": len(_job_status),
            }),
            ex=180,  # 3 min TTL — expires if scheduler dies
        )
        await client.aclose()
    except Exception:
        pass  # Best-effort renewal


async def initialize_scheduler() -> bool:
    """
    Initialize and start the APScheduler.

    Returns:
        bool: True if scheduler started successfully
    """
    global _scheduler

    # Check if scheduler is enabled
    if os.getenv("RUN_SCHEDULER", "0") != "1":
        logger.info("Scheduler disabled (RUN_SCHEDULER != 1)")
        return False

    if _scheduler is not None:
        logger.warning("Scheduler already initialized")
        return True

    # Distributed lock: prevent duplicate schedulers in multi-worker setups
    if not await _acquire_scheduler_lock():
        return False

    try:
        logger.info("Initializing APScheduler...")

        # Recover last known job status from Redis (if available)
        await _recover_job_status_from_redis()

        # Create scheduler with timezone
        _scheduler = AsyncIOScheduler(timezone="Europe/Zurich")

        # Common job defaults
        job_defaults = {
            "coalesce": True,  # Merge missed runs
            "max_instances": 1,  # Prevent overlapping executions
            "misfire_grace_time": 300,  # 5 min grace for missed jobs
        }

        # Add jobs with cron triggers

        # P&L intraday: every 15 min, 07:00-23:59
        _scheduler.add_job(
            job_pnl_intraday,
            CronTrigger(minute="*/15", hour="7-23", timezone="Europe/Zurich", jitter=60),
            id="pnl_intraday",
            name="P&L Snapshot Intraday",
            **job_defaults
        )

        # P&L EOD: daily at 23:59
        _scheduler.add_job(
            job_pnl_eod,
            CronTrigger(hour=23, minute=59, timezone="Europe/Zurich", jitter=60),
            id="pnl_eod",
            name="P&L Snapshot EOD",
            **job_defaults
        )

        # OHLCV daily: 03:10
        _scheduler.add_job(
            job_ohlcv_daily,
            CronTrigger(hour=3, minute=10, timezone="Europe/Zurich", jitter=60),
            id="ohlcv_daily",
            name="OHLCV Update Daily",
            **job_defaults
        )

        # OHLCV hourly: every hour at :05
        _scheduler.add_job(
            job_ohlcv_hourly,
            CronTrigger(minute=5, timezone="Europe/Zurich", jitter=30),
            id="ohlcv_hourly",
            name="OHLCV Update Hourly",
            **job_defaults
        )

        # Staleness monitor: every hour at :15
        _scheduler.add_job(
            job_staleness_monitor,
            CronTrigger(minute=15, timezone="Europe/Zurich"),
            id="staleness_monitor",
            name="Staleness Monitor",
            **job_defaults
        )

        # API warmers: every 10 minutes
        _scheduler.add_job(
            job_api_warmers,
            IntervalTrigger(minutes=10, jitter=60),
            id="api_warmers",
            name="API Warmers",
            **job_defaults
        )

        # Crypto-Toolbox refresh: 2x daily at 08:00 and 20:00
        _scheduler.add_job(
            job_crypto_toolbox_refresh,
            CronTrigger(hour='8,20', minute=0, timezone="Europe/Zurich", jitter=120),
            id="crypto_toolbox_refresh",
            name="Crypto-Toolbox Indicators Refresh (2x daily)",
            **job_defaults
        )

        # Daily ML training: every day at 03:00
        _scheduler.add_job(
            job_daily_ml_training,
            CronTrigger(hour=3, minute=0, timezone="Europe/Zurich", jitter=300),
            id="daily_ml_training",
            name="Daily ML Training (20y data)",
            **job_defaults
        )

        # Morning Brief: daily at 07:30 Europe/Zurich
        _scheduler.add_job(
            job_morning_brief,
            CronTrigger(hour=7, minute=30, timezone="Europe/Zurich", jitter=60),
            id="morning_brief",
            name="Morning Brief Generation",
            **job_defaults
        )

        # Lock renewal: extend Redis lock TTL every 60s
        _scheduler.add_job(
            _renew_scheduler_lock,
            IntervalTrigger(seconds=60),
            id="scheduler_lock_renewal",
            name="Scheduler Lock Renewal",
            coalesce=True,
            max_instances=1,
        )

        # Start scheduler
        _scheduler.start()

        # Log scheduled jobs
        jobs = _scheduler.get_jobs()
        logger.info(f"✅ APScheduler started with {len(jobs)} jobs:")
        for job in jobs:
            next_run = job.next_run_time.strftime("%Y-%m-%d %H:%M:%S %Z") if job.next_run_time else "N/A"
            logger.info(f"   - {job.id}: next run at {next_run}")

        return True

    except Exception as e:
        logger.exception(f"❌ Failed to initialize scheduler: {e}")
        _scheduler = None
        return False


async def shutdown_scheduler():
    """Shutdown the scheduler gracefully and release distributed lock"""
    global _scheduler

    if _scheduler is None:
        logger.info("Scheduler not running, nothing to shutdown")
        return

    try:
        logger.info("Shutting down APScheduler...")

        _scheduler.shutdown(wait=True)
        _scheduler = None

        # Release distributed lock
        try:
            import redis.asyncio as aioredis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            if redis_url and redis_url.strip():
                client = aioredis.from_url(redis_url, socket_connect_timeout=2)
                await client.delete("smartfolio:scheduler_lock")
                await client.aclose()
        except Exception:
            pass  # Best-effort release

        logger.info("APScheduler shutdown complete")

    except Exception as e:
        logger.exception(f"Failed to shutdown scheduler: {e}")

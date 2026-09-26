# Crypto-Toolbox API Integration

## Overview

This module scrapes cryptocurrency market indicators from [crypto-toolbox.vercel.app](https://crypto-toolbox.vercel.app/signaux) using Playwright and exposes them via FastAPI REST endpoints.

**Purpose**: Provide real-time risk indicators (BMO, MVRV, Puell Multiple, etc.) for decision-making in the portfolio management system.

**Status**: [OK] **Production** - FastAPI native implementation. Flask proxy removed in Commit 8 (Oct 2025).

---

## Architecture

### Technology Stack

- **FastAPI** (async REST API)
- **Playwright** (browser automation, async_api)
- **Chromium** (headless browser)

### Design Principles

- **Single browser instance**: Shared across all requests (launched at startup)
- **Concurrency control**: Semaphore(2) limits simultaneous scrapes
- **Cache-first**: 2-hour TTL with Redis persistence (fallback to memory cache)
- **Graceful degradation**: Browser auto-recovery if crashed

---

## API Endpoints

### `GET /api/crypto-toolbox`

Get crypto-toolbox indicators (cached or fresh).

**Query Parameters**:

- `force` (bool, optional): Force refresh bypassing cache (default: `false`)

**Response** (200 OK):

```json
{
    "success": true,
    "indicators": [
        {
            "name": "MVRV Z-Score",
            "value": "2.34",
            "value_numeric": 2.34,
            "threshold": ">=7",
            "threshold_numeric": 7.0,
            "threshold_operator": ">=",
            "in_critical_zone": false,
            "raw_value": "2.34",
            "raw_threshold": ">=7 (euphoria)"
        }
    ],
    "total_count": 15,
    "critical_count": 3,
    "scraped_at": "2025-10-02T12:34:56.789",
    "source": "crypto-toolbox.vercel.app",
    "cached": true,
    "cache_age_seconds": 456
}
```

**Error Response** (502 Bad Gateway):

```json
{
    "detail": "Upstream scraping error: timeout"
}
```

---

### `POST /api/crypto-toolbox/refresh`

Force refresh data (bypass cache).

**Response**: Same as `GET` with `cached: false`

---

### `GET /api/crypto-toolbox/health`

Health check endpoint.

**Response** (200 OK):

```json
{
    "status": "healthy",
    "browser_connected": true,
    "cache_status": "active",
    "cache_age_seconds": 456,
    "timestamp": "2025-10-02T12:34:56.789"
}
```

---

### `POST /api/crypto-toolbox/cache/clear`

Clear cache (admin/debug).

**Response** (200 OK):

```json
{
    "message": "Cache cleared successfully"
}
```

---

## Configuration

### Environment Variables

- `CRYPTO_TOOLBOX_URL` (optional): Target URL (default: `https://crypto-toolbox.vercel.app/signaux`)
- `CRYPTO_TOOLBOX_CACHE_TTL` (optional): Cache TTL in seconds (default: `1800`)

### Cache Settings

- **TTL**: 2 hours (7200 seconds)
- **Strategy**: Redis-first (persists across restarts), fallback to in-memory
- **Lock**: asyncio.Lock prevents thundering herd
- **Performance**: ~75% fewer scrapes compared to 30-min TTL

### Concurrency

- **Browser**: 1 shared instance (re-launched if crashed)
- **Pages**: Max 2 concurrent (Semaphore)
- **Workers**: **Single-worker Uvicorn only** (Playwright state is not multi-process safe)

---

## Dependencies

### Python Packages

```txt
playwright==1.46.0
fastapi>=0.104.0
uvicorn>=0.24.0
```

### System Requirements

**Chromium browser** must be installed:

```bash
# After pip install playwright
playwright install chromium
```

**Docker**: Use Playwright official image or install deps:

```dockerfile
RUN playwright install chromium --with-deps
```

---

## Lifecycle Management

### Startup Sequence

1. **FastAPI app starts** (`api/main.py`)
2. **ML models initialize** (3s delay, background task)
3. **Governance Engine** initialized
4. **Alert Engine** scheduler started
5. **Playwright browser** initialized (optional, non-blocking)
   - Only if `api/crypto_toolbox_endpoints` successfully imported
   - If fails → logs warning, browser lazy-launched on first request
   - Memory footprint: ~200 MB (Chromium process)

**Order of initialization** (in `api/startup.py`):

```python
models_count = await initialize_ml_models()           # 1st
governance_ok = await initialize_governance_engine()  # 2nd
alerts_ok = await initialize_alert_engine()           # 3rd
playwright_ok = await initialize_playwright_browser() # 4th (optional)
```

**Memory impact**:

- Baseline (FastAPI + ML): ~300-400 MB
- With Playwright browser: ~500-600 MB
- Per request overhead: ~1-2 MB (page context)

### Shutdown Sequence

1. **FastAPI shutdown event** triggered (SIGTERM/SIGINT)
2. **Alert Engine** scheduler stopped gracefully
3. **Playwright browser** closed (if initialized)
   - All pages closed
   - Browser process terminated
   - Playwright instance stopped
4. **Cleanup complete** (logs confirmation)

### Recovery & Resilience

- **Browser crash**: Auto re-launch on next request
  - `_ensure_browser()` checks `browser.is_connected()`
  - New browser spawned if disconnected
  - Request retried automatically
- **Import failure**: Graceful degradation
  - Startup continues even if Playwright fails
  - Logs warning, endpoints return 502 until browser available
- **Page timeout**: 15-second limit per scrape
  - Request fails with HTTPException 502
  - Cache serves stale data if available

---

## Special Cases

### BMO (par Prof. Chaîne)

This indicator has **multiple sub-indicators** with different thresholds:

- Each sub-indicator gets a separate entry
- Format: `"BMO (par Prof. Chaîne) (Label)"`
- Example: `"BMO (par Prof. Chaîne) (Short-Term)"`

### Threshold Operators

Supported comparison operators:

- `>=` : Greater than or equal
- `<=` : Less than or equal
- `>` : Greater than
- `<` : Less than

### Parsing Edge Cases

- Comma decimal separators → normalized to periods
- Newlines in raw values → stripped
- Missing thresholds → skipped (not included in response)

---

## Memory & Performance

### Memory Footprint

- **Browser process**: ~150-200 MB (Chromium headless)
- **Cache**: ~10-50 KB (JSON in-memory)
- **Total**: ~200 MB per worker

### Latency

- **Cache hit (Redis)**: <5 ms
- **Cache hit (memory)**: <2 ms
- **Cache miss (scrape)**: 12-15 seconds (optimized from 20-24s)
- **Timeout**: 15 seconds (page load)

### Scaling

- **Vertical**: Single worker, handles ~50 req/s (mostly cached)
- **Horizontal**: Multiple containers (1 worker each) behind load balancer
- [Warning] **Do NOT use `--workers > 1`** with Playwright (shared browser state issue)

---

## Development

### Local Testing

```bash
# Start dev server
uvicorn api.main:app --port 8080

# Test endpoint
curl http://localhost:8080/api/crypto-toolbox

# Force refresh
curl http://localhost:8080/api/crypto-toolbox?force=true

# Health check
curl http://localhost:8080/api/crypto-toolbox/health
```

### Debug Logs

```python
import logging
logging.getLogger("api.crypto_toolbox_endpoints").setLevel(logging.DEBUG)
```

---

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# Install Playwright + Chromium
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN playwright install chromium --with-deps

COPY . .
EXPOSE 8080

# Single worker (Playwright requirement)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### systemd (Linux)

```ini
[Unit]
Description=Crypto Rebal API
After=network.target

[Service]
WorkingDirectory=/opt/crypto-rebal
ExecStart=/opt/crypto-rebal/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Migration Status

### Current State

- [OK] Router skeleton created (`api/crypto_toolbox_endpoints.py`)
- [OK] Contract documented (JSON schema, invariants)
- [OK] Lifecycle hooks integrated (`api/startup.py`)
- [OK] **Parsing logic ported from Flask** (100% parity)
  - `_parse_comparison()` function (regex-based)
  - BMO special handling (multiple sub-indicators)
  - Operator evaluation (>=, <=, >, <)
  - Async Playwright calls (sync → async migration)
- [OK] **Router integrated in `api/main.py`** (behind feature flag)
  - Feature flag: `CRYPTO_TOOLBOX_NEW` (0=Flask proxy, 1=FastAPI native)
  - Default: 0 (legacy Flask proxy, safe)
  - Router loaded conditionally with try/except
  - Legacy proxy disabled when flag=1
  - Logs clearly indicate active mode
- [Pending] A/B testing & validation (Commit 6)
- [Pending] Legacy Flask removal (Commit 8, after A/B validation)

### Parsing Parity

**Validated** against `crypto_toolbox_api.py`:

- [OK] Regex patterns identical
- [OK] BMO multi-indicator logic identical
- [OK] Numeric extraction identical
- [OK] Threshold parsing identical
- [OK] JSON structure identical

**See**: `docs/CRYPTO_TOOLBOX_PARITY.md` for detailed checklist

### Rollback Strategy

- **Before Commit 4**: No impact (router not included)
- **Commit 4-7**: Feature flag `CRYPTO_TOOLBOX_NEW=0` reverts to Flask
- **After Commit 8**: Revert commit to restore Flask proxy

---

## References

- **Original Flask implementation**: `crypto_toolbox_api.py` (to be removed in Commit 8)
- **Proxy endpoint**: `api/main.py:432-447` (to be removed in Commit 8)
- **Parity validation**: `docs/CRYPTO_TOOLBOX_PARITY.md`
- **Target site**: <https://crypto-toolbox.vercel.app/signaux>

---

## Feature Flag Usage

### Environment Variable

```bash
# Default (FastAPI native, production)
export CRYPTO_TOOLBOX_NEW=1

# Fallback (Flask proxy, legacy)
export CRYPTO_TOOLBOX_NEW=0
```

**Note**: Since Commit 7, FastAPI mode is **enabled by default**. Flask proxy remains available for emergency rollback.

### Testing A/B Switch

```bash
# Test with default (FastAPI native)
uvicorn api.main:app --port 8080

# Test with legacy fallback (Flask proxy on port 8001)
CRYPTO_TOOLBOX_NEW=0 uvicorn api.main:app --port 8080

# Compare outputs
curl http://localhost:8080/api/crypto-toolbox | jq '.total_count'
```

### Startup Logs

**Default (FastAPI native)**:

```
 Crypto-Toolbox: Using FastAPI native scraper (CRYPTO_TOOLBOX_NEW=1)
 Initializing Playwright browser...
[OK] Playwright browser launched successfully
```

**Fallback (Flask proxy)**:

```
 Crypto-Toolbox: Using Flask proxy (CRYPTO_TOOLBOX_NEW=0, legacy mode)
```

---

## Development & Deployment

### Local Development

**Windows (PowerShell)**:

```powershell
# Legacy mode (default)
.\start_dev.ps1

# FastAPI native mode
.\start_dev.ps1 -CryptoToolboxMode 1

# Or use environment variable
$env:CRYPTO_TOOLBOX_NEW=1; .\start_dev.ps1
```

**Linux/Mac (Bash)**:

```bash
# Legacy mode (default)
./start_dev.sh

# FastAPI native mode
./start_dev.sh 1

# Or use environment variable
CRYPTO_TOOLBOX_NEW=1 ./start_dev.sh
```

### Docker Deployment

**Build image** (includes Playwright + Chromium):

```bash
docker build -t crypto-rebal .
```

**Run container** (FastAPI mode by default):

```bash
# Default (CRYPTO_TOOLBOX_NEW=1)
docker run -p 8080:8080 crypto-rebal

# Override to legacy mode
docker run -p 8080:8080 -e CRYPTO_TOOLBOX_NEW=0 crypto-rebal
```

**Note**: Docker image includes Playwright dependencies (~300MB additional size). To reduce image size for legacy-only deployments, remove lines 34-36 from Dockerfile.

### Playwright Installation (Development)

If using FastAPI mode (`CRYPTO_TOOLBOX_NEW=1`) locally:

```bash
# Install Playwright
pip install playwright

# Install Chromium browser
playwright install chromium

# Verify installation
python -c "from playwright.async_api import async_playwright; print('OK')"
```

**Important**: Single-worker Uvicorn is **required** for Playwright mode (browser state is not multi-process safe).

---

## A/B Testing & Validation

### Prerequisites

Before testing, ensure you have:

1. **Flask server** running on port 8001 (for baseline)

   ```bash
   python crypto_toolbox_api.py
   ```

2. **Playwright installed** (for FastAPI mode)

   ```bash
   pip install playwright
   playwright install chromium
   ```

### Step 1: Capture Flask Baseline

```bash
# Verify Flask is responding
curl -s http://localhost:8001/api/crypto-toolbox | jq '.success'

# Capture output
curl -s http://localhost:8001/api/crypto-toolbox > test_flask_baseline.json

# Check counts
jq '.total_count, .critical_count' test_flask_baseline.json
```

### Step 2: Test FastAPI Implementation

**Start server in FastAPI mode**:

```bash
# Windows
.\start_dev.ps1 -CryptoToolboxMode 1

# Linux/Mac
./start_dev.sh 1
```

**Capture output** (in another terminal):

```bash
# Force fresh scrape (bypass cache)
curl -s "http://localhost:8080/api/crypto-toolbox?force=true" > test_fastapi_new.json

# Check counts
jq '.total_count, .critical_count' test_fastapi_new.json
```

### Step 3: Run Comparison

**Automated comparison**:

```bash
python scripts/compare_crypto_toolbox.py test_flask_baseline.json test_fastapi_new.json
```

**Expected output**:

```
COUNTS COMPARISON
Total count   - Flask: 15 | FastAPI: 15 | Match: OK
Critical count - Flask:  3 | FastAPI:  3 | Match: OK

INDICATOR NAMES COMPARISON
[OK] All indicator names match

VALUES & CRITICAL ZONES COMPARISON
[OK] All values match (tolerance: ±0.01)
[OK] All critical zones match

SUMMARY
[OK] VALIDATION PASSED - FastAPI implementation matches Flask
   You can proceed with Commit 7 (switch default flag)
```

### Step 4: Performance Validation

**Cache miss** (first request):

```bash
time curl -s "http://localhost:8080/api/crypto-toolbox?force=true" > /dev/null
# Expected: <5 seconds
```

**Cache hit** (within 30 minutes):

```bash
time curl -s "http://localhost:8080/api/crypto-toolbox" > /dev/null
# Expected: <50ms
```

**Stability test** (10 consecutive requests):

```bash
for i in {1..10}; do
  curl -s "http://localhost:8080/api/crypto-toolbox?force=true" | jq '.success'
done
# Expected: All "true"
```

### Validation Checklist

Before proceeding to Commit 7, verify:

- [ ] `total_count` matches (±1 tolerance)
- [ ] `critical_count` matches exactly
- [ ] All indicator names present in both
- [ ] BMO sub-indicators split correctly
- [ ] No parsing errors in logs
- [ ] Cache miss <5s
- [ ] Cache hit <50ms
- [ ] 10 consecutive requests succeed

**If all [OK]**: Proceed to Commit 7 (switch default flag)

**If any [Error]**: See [CRYPTO_TOOLBOX_PARITY.md](CRYPTO_TOOLBOX_PARITY.md) for rollback procedure

---

---

## Migration Timeline

| Phase | Commit | Status | Description |
|-------|--------|--------|-------------|
| 1 | [OK] Router skeleton | Complete | API contract, endpoints structure |
| 2 | [OK] Lifecycle hooks | Complete | Playwright startup/shutdown in `api/startup.py` |
| 3 | [OK] Parsing parity | Complete | 100% identical logic ported from Flask |
| 4 | [OK] Feature flag | Complete | `CRYPTO_TOOLBOX_NEW` A/B testing |
| 5 | [OK] Dev/Docker | Complete | Scripts, Dockerfile, deployment docs |
| 6 | [OK] A/B testing | Complete | Validation procedures, comparison tools |
| 7 | [OK] **Default ON** | Complete | FastAPI native by default, Flask fallback |
| 8 | [OK] **Flask removed** | **Complete** | Final cleanup - migration finished |

**Current Status**: [OK] **Migration complete** - FastAPI native is the only implementation. Flask proxy and feature flag removed.

---

**Last updated**: 2025-10-02
**Status**: Phase 8 - Migration Complete (Flask Removed)

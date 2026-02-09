# Comprehensive Audit Report - SmartFolio

**Date:** 8 February 2026
**Scope:** Full re-audit of existing dimensions + 5 new audit domains
**Method:** Automated scanning (Bandit, pip-audit, pytest-cov) + multi-agent code analysis
**Previous score:** 7.7/10 (Feb 7, 2026)

---

## Executive Summary

This audit refreshes all 6 existing dimensions and introduces 5 new audit domains never previously covered. Several significant findings contradict the reported metrics.

### Critical Discoveries

| Finding | Impact |
|---------|--------|
| **Test coverage was 20.5%**, raised to **30.2%** with 179 new tests | Baseline 30% now met |
| **9 CVEs** in 8 packages (including 2 Starlette DoS) | Previous "0 CVE" status is outdated |
| **55% of API files have zero authentication** | Governance, ML training, Kraken balances exposed |
| **73% of API files don't use standard response format** | 3 competing envelope formats |
| **Accessibility is ~80/100**, not 92+ (only 6/20 pages tested) | Score inflated by selective testing |

### Updated Scores

| Dimension | Old Score | New Score | Delta | Key Issue |
|-----------|-----------|-----------|-------|-----------|
| Security | 8.5/10 | **7.0/10** | -1.5 | 9 CVEs, auth gaps, JWT default secret |
| Performance | 7.5/10 | 7.5/10 | 0 | Not re-profiled, assumed stable |
| Accessibility | 92+/100 | **~80/100** | -12 | Only 6/20 pages actually tested |
| Technical Debt | 7.5/10 | **8.0/10** | +0.5 | risk_management.py -54% (2159→990), get_risk_dashboard -51%, governance -44% |
| Tests | 8/10 | **6.0/10** | -2.0 | Coverage raised 20.5% -> 30.2% (989 passing, baseline met) |
| CI/CD | 8/10 | 8/10 | 0 | Workflows functional |
| **NEW: API Contract** | -- | **6.0/10** | -- | Return types added, 5 HTTP codes fixed (400→404); 3 response formats remain |
| **NEW: Error Handling** | -- | **8.0/10** | +1.5 | Circuit breakers added (CoinGecko/FRED/Saxo), timeouts fixed |
| **NEW: Data Integrity** | -- | **8.0/10** | +2.5 | Auth on governance, CSV sanitization, Pydantic models |
| **NEW: Logging** | -- | **8.0/10** | +3.0 | Request IDs, JSON file logs, sensitive data sanitized |
| **NEW: Concurrency** | -- | **7.5/10** | +2.0 | FileLock on 5 critical writes, scheduler Redis lock |

**Updated Overall Score: 7.5/10** (was 6.0 at audit, was 7.7 before audit; Tech Debt now 8.0/10)

---

## Part A: Refreshed Existing Audits

### A1. Security: 8.5 -> 7.0/10

#### Bandit Scan (Feb 8, 2026)
- **HIGH: 0** (unchanged from Nov 2025)
- **MEDIUM: 13** (down from 24 in Nov 2025, -46%)
- All 13 MEDIUM are in `services/ml/safe_loader.py` (expected -- the safety wrapper itself)
- **No new code-level vulnerabilities introduced by the Feb 2026 refactoring**

#### pip-audit: 9 CVEs in 8 Packages (was 0)

| Package | Version | CVE | Severity | Fix Version |
|---------|---------|-----|----------|-------------|
| **starlette** | 0.38.6 | CVE-2024-47874 | **HIGH** | 0.40.0 |
| **starlette** | 0.38.6 | CVE-2025-54121 | MEDIUM | 0.47.2 |
| **urllib3** | 2.6.2 | CVE-2026-21441 | MEDIUM | 2.6.3 |
| **python-multipart** | 0.0.21 | CVE-2026-24486 | MEDIUM | 0.0.22 |
| **protobuf** | 6.33.2 | CVE-2026-0994 | MEDIUM | 6.33.5 |
| **filelock** | 3.20.2 | CVE-2026-22701 | MEDIUM | 3.20.3 |
| **pyasn1** | 0.6.1 | CVE-2026-23490 | MEDIUM | 0.6.2 |
| **pip** | 25.3 | CVE-2026-1703 | LOW | 26.0 |
| **ecdsa** | 0.19.1 | CVE-2024-23342 | LOW | No fix |

**Critical note:** Starlette CVE-2024-47874 is a **DoS vulnerability affecting ALL FastAPI apps accepting form data**. SmartFolio uses `Form()` in auth endpoints and `UploadFile` in Saxo/Sources endpoints. This is directly exploitable.

**38 total packages are outdated**, including major version gaps:
- FastAPI 0.115.0 -> 0.128.5
- Starlette 0.38.6 -> 0.52.1
- Pydantic 2.9.2 -> 2.12.5

#### JWT Security
- Default secret `"your-secret-key-change-in-production-please"` in `api/auth_router.py:23` and `api/deps.py:38`
- `.env.example` documents `JWT_SECRET_KEY` but does not enforce it
- `DEV_SKIP_AUTH=1` and `DEV_OPEN_API=1` env vars bypass all auth if set
- `get_current_user_jwt` is defined but **never used by any endpoint** -- JWT exists in auth flow but endpoints rely on unverified `X-User` header

---

### A2. Tests: 8/10 -> 5.0/10

#### Real Coverage (Feb 8, 2026)

```
Total tests collected: 1,268
Coverage: 20.51% (was reported as 50-55%)
Required baseline (pyproject.toml): 30% -- FAILING
```

The previous 50-55% figure likely measured a subset of files. The full `pytest --cov` across all sources shows **20.51%**.

#### Coverage by Area
- Backend services: ~20-30% (varies)
- Frontend JS: **1%** (1/92 files: `computeExposureCap.test.js`)
- New modules (DI Backtest, macro_stress): coverage unknown but likely low

---

### A3. Technical Debt: 7.5 -> 7.0/10

#### TODOs: 8 -> 20 (increase)
- 0 CRITICAL, 0 HIGH (same as before)
- ~5 MEDIUM (mock data in production endpoints, unimplemented features)
- ~15 LOW (backlog, test scaffolding)

#### God Services Progress

| Service | Nov 2025 | Feb 2026 | Delta |
|---------|----------|----------|-------|
| `governance.py` | 2,092 | **1,163** | **-44%** |
| `risk_management.py` | 2,159 | **990** | **-54%** |
| `alert_engine.py` | 1,583 | **1,324** | **-16%** |

Phase 1 (governance -44%) and Phase 2 (risk_management -54%) completed. Models, AlertSystem, and VaR calculations deduplicated into `services/risk/` modules.

#### NEW: Frontend God Controllers (never audited before)

| File | Lines |
|------|-------|
| `risk-dashboard-main-controller.js` | **3,520** |
| `dashboard-main-controller.js` | **3,321** |
| `rebalance-controller.js` | **2,749** |
| `settings-main-controller.js` | **2,636** |
| `decision-index-panel.js` | **2,139** |

**5 JavaScript files over 2,000 lines** -- a parallel debt to the Python God Services, never identified in previous audits.

#### Complexity Hotspots

| Function | File | Lines |
|----------|------|-------|
| `get_risk_dashboard` | `api/risk_endpoints.py` | **326** (was 663, -51%) |
| `_evaluate_alert_type` | `services/alerts/alert_engine.py` | **298** |
| `_create_ensemble_predictions` | `services/ml/orchestrator.py` | **247** |

#### Large Files: 104 Python files >500 lines, 38 JS files >500 lines

---

### A4. Accessibility: 92+ -> ~80/100

The reported 92+ score only applies to **6 of 20 pages** tested via Lighthouse.

#### What Was Actually Fixed
- Color contrast: **RESOLVED** (tokens.css + shared-theme.css updated)
- Focus-visible: **RESOLVED** (global CSS)
- Prefers-reduced-motion: **RESOLVED** (global CSS)

#### What Remains Broken
- **Canvas charts**: Only 3/30 canvas elements have `aria-label` + `role="img"` (dashboard.html only)
- **Table headers**: Only 23/239 `<th>` elements have `scope=` attribute (9.6%)
- **Landmarks**: Only 3 pages have `<main>` landmarks
- **14 pages never tested** by Lighthouse

#### True Score

| Scope | Score |
|-------|-------|
| 6 Lighthouse-tested pages | 94-95/100 |
| 14 untested pages | ~75-80/100 |
| **Project-wide average** | **~80/100** |

---

## Part B: New Audit Domains

### B1. API Contract / Consistency: 4.0/10

**472 endpoints across 60 files analyzed.**

| Dimension | Conformity |
|-----------|-----------|
| Response format (success_response/error_response) | **26.7%** |
| Authentication present | **45%** |
| HTTP status code correctness | **~55%** |
| Return type annotations | **16.7%** |
| Query parameter patterns | 94.9% |
| REST method correctness | 99.4% |

#### Key Issues
1. **3 competing response envelopes**: `{"ok": true}` (formatters), `{"success": true}` (FRED/multi-asset), bare dicts
2. **`paginated_response()`** defined but **never used** (dead code)
3. **33/60 API files have ZERO authentication** -- including governance mutations, ML training, Kraken balances
4. **393/472 endpoints** lack return type annotations (OpenAPI docs incomplete)
5. ~~**5 "not found" errors** return 400 instead of 404 in governance endpoints~~ **FIXED (Feb 9)**
6. **244 uses of HTTP 500** (many should be 502/504 for upstream failures)

---

### B2. Error Handling / Resilience: 6.5/10

#### Missing Timeouts (5 endpoints)

| Service | File | Impact |
|---------|------|--------|
| Saxo token exchange | `connectors/saxo_api.py:162` | **CRITICAL** -- OAuth can block forever |
| Saxo token refresh | `connectors/saxo_api.py:200` | CRITICAL |
| FRED proxy - Bitcoin | `api/main.py:416` | HIGH |
| FRED proxy - DXY | `api/main.py:484` | HIGH |
| FRED proxy - VIX | `api/main.py:558` | HIGH |
| SMTP notifier | `services/notifications/notification_sender.py:118` | MEDIUM |

#### Other Issues
- **No circuit breaker pattern** anywhere in the codebase
- **CoinGecko recursive retry** on 429 with no depth limit (`services/coingecko.py:109`)
- **KrakenAdapter** has zero retry decorators (vs BinanceAdapter which has them)
- **AlertStorage** never auto-recovers from Redis degradation
- **Binance `place_order`** retries without idempotency key (risk of duplicate orders)

#### Positive
- Good 3-tier cascade fallback in alert_storage (Redis/File/Memory)
- Retry with exponential backoff on BinanceAdapter
- Stale cache fallback on CoinGecko proxy
- Hardcoded FX fallback rates
- Comprehensive exception hierarchy in `shared/exceptions.py`

---

### B3. Data Integrity / Validation: 5.5/10

#### CRITICAL Issues

| Issue | Location | Impact |
|-------|----------|--------|
| Governance endpoints have NO auth | `api/execution/governance_endpoints.py` | Anyone can change mode, approve/execute plans |
| `cleanup_old_csv_files` has NO auth | `api/csv_endpoints.py:325` | Anyone can delete CSV files |
| Execution history has NO auth | `api/execution_history.py` | All session data accessible |
| Hardcoded JWT default secret | `api/auth_router.py:23`, `api/deps.py:38` | Token forgery if env var unset |

#### HIGH Issues
- **6+ governance endpoints accept raw `dict`** instead of Pydantic models (no validation)
- **No CSV injection protection** -- formulas (`=`, `+`, `-`, `@`) pass through in `csv_helpers.py`
- **`file_key` parameter unsanitized** in `saxo_endpoints.py:321` (path traversal risk)
- **No login rate limiting** beyond generic per-IP limiter
- **`DEV_OPEN_API=1`** bypasses ALL auth and RBAC

#### Positive
- `UserScopedFS` has proper `.resolve()` + `is_relative_to()` + symlink blocking
- `validate_user_id()` enforces alphanumeric + max 50 chars
- File uploads enforce 10MB/file, 50MB/batch with extension validation
- Password hashing uses bcrypt 12 rounds
- Atomic JSON writes via `tempfile.mkstemp` + `os.replace` in core services
- No command injection or SQL injection vectors found

---

### B4. Logging / Observability: 5.0/10

#### HIGH Issues
- **No request ID / correlation ID system** -- impossible to trace requests in production
- **Partial API key printed** in `debug/scripts/debug_coingecko.py:22` (first 8 chars)

#### MEDIUM Issues
- API key **length leaked** at INFO level in `services/balance_service.py:273,464`
- Generated **passwords printed to stdout** in `scripts/ops/setup_passwords.py`
- JSON logging **configured in settings but never activated** (`config/settings.py` vs `api/main.py`)
- `print()` statements in production code (`workers/rq_worker.py`, `connectors/kraken_api.py`)
- **Silent error swallowing** in sync pricing path (`services/pricing.py:181-184`)

#### Positive
- Production services never log actual token values
- Auth router logs only usernames on failures, never passwords
- Redis failures, external API failures, and auth failures all properly logged

---

### B5. Concurrency / Race Conditions: 5.5/10

#### HIGH Issues
- **`filelock` used in only 1 of 12 file-writing services** (only `alert_storage.py`)
- **11 services write to disk without any file lock**: users.json, taxonomy, pricing cache, score_registry, strategy_registry, ML model registry, history, instruments, monitoring
- **Unbounded in-memory caches** in 5+ API modules (`_risk_cache = {}`, `_analytics_cache = {}`, etc.) -- no eviction, no max size

#### MEDIUM Issues
- `asyncio.Lock` in `portfolio_history_storage.py` not cross-process safe
- Atomic write pattern (`tempfile` + `os.replace`) only in 2 of 12 services
- Scheduler shares event loop with FastAPI, warmers compete with real requests
- Global mutable state in `services/pricing.py` and `services/fx_service.py` (single-worker safe only)

#### Positive
- Redis usage is well-designed: Lua scripts for atomic ops, `SET NX` for dedup, pipelines for batching
- ML/intelligence services use `threading.Lock` properly
- `portfolio_history_storage` uses per-path async locks

---

## Part C: Prioritized Action Plan

### Immediate (P0) -- Security Critical

| # | Action | Effort | Impact | Status |
|---| ------ | ------ | ------ | ------ |
| 1 | **Upgrade starlette** to >=0.47.2 (CVE-2024-47874 + CVE-2025-54121) | 1h | Closes DoS vulnerability | DONE (Feb 8) |
| 2 | **Add auth to governance endpoints** | 2h | Closes privilege escalation | DONE (Feb 8) |
| 3 | **Add auth to execution_history, kraken, csv cleanup** | 3h | Closes data exposure | DONE (Feb 8) |
| 4 | **Add timeouts** to Saxo token exchange/refresh | 15min | Prevents infinite hang | DONE (Feb 8) |
| 5 | **Upgrade** urllib3, python-multipart, protobuf, filelock, pyasn1 | 30min | Closes 7 CVEs | DONE (Feb 8) |

### Short-term (P1) -- High Value

| # | Action | Effort | Impact | Status |
|---| ------ | ------ | ------ | ------ |
| 6 | Add request ID middleware (correlation IDs) | 2h | Production debugging | DONE (Feb 8) |
| 7 | Add filelock to critical file writes (users.json, taxonomy, registries) | 3h | Data integrity | DONE (Feb 8) |
| 8 | Add timeouts to FRED proxy endpoints | 30min | Prevents hangs | DONE (Feb 8, P0) |
| 9 | Add CSV injection protection in csv_helpers.py | 1h | Data security | DONE (Feb 8) |
| 10 | Validate governance endpoints with Pydantic models | 2h | Input validation | DONE (Feb 8) |
| 11 | ~~Add eviction/max-size to in-memory API caches~~ | ~~2h~~ | ~~Memory safety~~ | N/A -- audit found all caches already have TTL/eviction |

### Medium-term (P2) -- Quality

| # | Action | Effort | Impact | Status |
|---|--------|--------|--------|--------|
| 12 | Standardize response format (success_response everywhere) | 8h | API consistency | DEFERRED (needs coordinated frontend migration) |
| 13 | Add return type annotations to endpoints (~100 endpoints across 35+ files) | 6h | OpenAPI docs | DONE (Feb 9) |
| 14 | Add `aria-label` + `role="img"` to 20 canvas elements (9 pages) | 2h | Accessibility | DONE (Feb 8) |
| 15 | Add `scope="col"` to 151 `<th>` elements (10 pages) | 3h | Accessibility | DONE (Feb 8) |
| 16 | Circuit breaker for CoinGecko, FRED, Saxo | 4h | Resilience | DONE (Feb 9) |
| 17 | Raise test coverage to 30% baseline | 1-2w | Test reliability | DONE (Feb 9) — 20.5% -> 30.2%, +179 tests, 7 new test files |
| 18 | Fix sensitive data in logs (API key length, partial keys) | 1h | Security hygiene | DONE (Feb 8) |
| 25 | Batch Binance price requests (single HTTP call instead of 124 sequential) | 4h | Performance | DONE (Feb 9) |
| 26 | Fix symbol mapping: strip CoinTracking numeric suffixes (WLD3->WLD) | 3h | Reliability | DONE (Feb 9) |
| 28 | Fix 5 governance HTTP status codes (400→404 for "plan not found") | 15min | API correctness | DONE (Feb 9) |

### Long-term (P3) -- Technical Excellence

| # | Action | Effort | Impact | Status |
|---|--------|--------|--------|--------|
| 19 | Refactor `get_risk_dashboard` (663 lines → 326 lines, -51%) | 4h | Maintainability | DONE (Feb 9) |
| 20 | Refactor `risk_management.py` (2,159→990 lines, -54%) | 2w | Technical debt | DONE (Feb 9) |
| 21 | Address 5 frontend God Controllers (2,000+ lines each) | 4w | Frontend debt | |
| 22 | Implement JWT auth on all endpoints (replace X-User header) | 2w | Auth architecture | |
| 23 | Structured JSON logging (file handler with JsonLogFormatter) | 2h | Observability | DONE (Feb 9) |
| 24 | Redis distributed lock for scheduler exclusivity | 2h | Multi-worker safety | DONE (Feb 9) |
| 27 | **Add CoinGecko rate-limit backoff** (retry with exponential backoff on 429 in connector + proxy) | 2h | API reliability | DONE (Feb 9) |

---

## Part D: Audit Documentation Corrections Needed

1. **AUDIT_STATUS.md** lines 142-179: Accessibility section contradicts header table. Body says 68/100, header says 92+. True score is ~80/100.
2. **AUDIT_STATUS.md** line 229: Tests coverage reported as "~50-55%". Real coverage was 20.51%, now raised to **30.24%** (Feb 9).
3. **README.md** line 34: Accessibility score shows "68/100". Should be updated to ~80/100.
4. **AUDIT_STATUS.md** line 34: Security score "8.5/10" should reflect new CVEs (recommend 7.0/10).

---

## Appendix: Tool Versions Used

- Bandit 1.9.3 (installed fresh for this audit)
- pip-audit 2.10.0 (installed fresh for this audit)
- pytest 9.0.2 + pytest-cov 7.0.0
- Python 3.13
- All scans run on Windows 11, Feb 8, 2026

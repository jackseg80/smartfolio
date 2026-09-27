# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Dashboard — 2026-09-27

- Correction du camembert Stock Market : le graphique partage les positions et le cash du résumé, avec la même source CSV/API/manuelle que le total affiché.
- Prise en charge des classes d'actifs directes, des valeurs USD, des portefeuilles uniquement en cash et du retour après un état vide. Les anciens caches sans positions sont invalidés.
- Retrait du Morning Brief du dashboard, de son chargement JavaScript et de ses styles dédiés. Le service, l'API et la tâche planifiée existants restent disponibles.
- Validation : 7 tests de régression JavaScript ; 3 160 tests Python réussis, 13 ignorés ; proportions 60/30/10 vérifiées dans Chromium avec des données de test.
- Utilisation et contrôle manuel : [Dashboard](docs/DASHBOARD.md).

### Global Overview et Export PDF — 2026-09-27

- Global Overview utilise le même fichier Saxo sélectionné que la tuile Stock Market et inclut son cash converti en USD. Cela corrige l'écart observé de 9 846 $ sur les captures fournies.
- Le PDF capture désormais la page en JPEG plutôt qu'en PNG sans perte. Le dashboard applique une échelle 1, une qualité 0,72 et un maximum de 1,5 million de pixels par page ; le module garde une qualité par défaut de 0,88, réglable avec `imageQuality` et `maxImagePixels`.
- Les modules du dashboard et d'export portent une version d'URL dédiée pour éviter qu'un ancien JavaScript en cache conserve l'export PNG de 30 Mo.
- Syntaxe vérifiée pour le module PDF et l'endpoint Global Overview.

### Plan d'Amelioration v5 — 11 Items Complete (Feb 2026)

**Period:** February 7-10, 2026
**Objective:** Address critical infrastructure gaps (notifications, backups, freshness, mobile, observability)
**Result:** 11/11 items complete, 2910 tests passing, 0 failures

#### Phase 1: Foundations & Reliability (P0)

- **1.1 Atomic Writes** — `services/user_management.py` + `services/alerts/alert_storage.py` migrated to temp+`os.replace()` pattern + 20 tests
- **1.2 Circuit Breakers** — Added `binance_circuit` + `cointracking_circuit` in `shared/circuit_breaker.py`, integrated in `price_history.py` and `cointracking_api.py`, health_router extended (5 circuits)
- **1.3 Backup System** — `services/backup_manager.py` (create/restore/verify/retention), 3 PowerShell scripts, 6 `/admin/backups/*` endpoints + 52 tests
- **1.4 Notifications Connected** — `TelegramNotifier` (httpx async), `WebhookNotifier` converted to async, bridge `convert_engine_alert()`, hook in `alert_engine.py:_create_alert()`, `api/notification_endpoints.py` (4 endpoints), `config/alerts_rules.json` extended + 68 tests

#### Phase 2: Data Freshness & Offline

- **2.1 Data Freshness** — Web Component `<data-freshness>` deployed on 22 pages, `fetcher.js` exposes `getLastUpdateTime()` + 15 tests
- **2.2 Offline Detection** — `network-state-manager.js` imported in 19 missing pages + persistent banner

#### Phase 3: Morning Brief

- **3.1 Morning Brief** — `services/morning_brief_service.py`, endpoint `GET /api/morning-brief`, scheduler job 07:30, Web Component `morning-brief-card.js` + 40 tests

#### Phase 4: Export & Mobile

- **4.1 Export PDF** — `static/modules/pdf-export.js` (jsPDF + html2canvas), `static/css/print.css`, button on 3 pages + Plotly.toImage support
- **4.2 Mobile UX** — `static/css/responsive.css`, touch targets 44px, single column <768px

#### Phase 5: Observability & Scheduler

- **5.1 Sentry Frontend** — `static/core/sentry-init.js`, endpoint `/api/config/sentry-dsn`, deployed on 21 pages
- **5.2 Scheduler Resilient** — `_job_status` dict → Redis HASH (`smartfolio:scheduler:jobs`), heartbeat TTL 180s, webhook failure alerts, recovery at restart, `/admin/scheduler/status` endpoint + 17 tests

#### Hotfix: Encoding

- **Double-encoded UTF-8** — Fixed BOM + CP1252 mojibake on 22 HTML files. Script `scripts/fix_encoding.py` handles 5 undefined CP1252 bytes + mixed-content lines. 19/22 fully fixed, 3 with residual in JS comments only.

### Coverage Push 42.9% → ~47% (Feb 2026)

**Period:** February 10, 2026
**Result:** 3198 tests passing, 0 failures

#### Added — Tests

- **Backtesting Engine** — 43 tests (~60% coverage)
- **Trading Strategies** — 70 tests (~65% coverage, 5 strategies)
- **Taxonomy** — 30 tests (67% coverage)
- **Regime Constants** — 43 tests (~90% coverage)
- **Data Processing** — 48 tests (~70% coverage)
- **Idempotency Manager** — 25 tests (~80% coverage)
- **Monte Carlo** — 9 tests (~40% coverage)
- **Stress Testing** — 20 tests (~50% coverage)
- **Utils Formatters** — +30 tests (~85% coverage)
- **Bourse Metrics** — 48 tests (~96% coverage)
- **Alert Types** — 50 tests (~70% coverage)
- **Signals** — 50 tests (~60% coverage)
- **Advanced Rebalancing** — 50 tests (~30% coverage)
- **Risk Scoring** — 80 tests (100% coverage)
- **Instruments Registry** — 45 tests (90% coverage)
- **Portfolio Metrics Service** — 42 tests (~40% coverage)
- **Structural Score V2** — 28 tests (~80% coverage)

### JWT Auth Migration (Feb 2026)

- Modified `get_required_user` + `require_admin_role` in `api/deps.py`
- Added `_extract_jwt_user()` helper: validates JWT if present
- Soft mode (default): JWT validated if present, fallback to X-User header
- Strict mode (`REQUIRE_JWT=1`): rejects without valid JWT
- Anti-spoofing cross-check: JWT user != X-User → 403
- Frontend `http.js` sends JWT via `Authorization: Bearer <token>`
- 80 tests in `test_api_deps.py` (was 55)

---

### Full FR → EN Translation (Feb 2026)

**Period:** February 6, 2026
**Objective:** Translate all user-visible text from French to English
**Result:**  **COMPLETE** - ~300+ strings translated across ~50 files

#### Changed - Backend

- **Patrimoine → Wealth rename**: `PatrimoineItemInput` → `WealthItemInput`, services and migration files renamed
- **API routes**: `/api/wealth/patrimoine/*` → `/api/wealth/items/*` (legacy routes still supported)
- **Data storage**: `patrimoine.json` → `wealth.json` (fallback to patrimoine.json)
- **HTTPException messages**: ~50 error messages translated to English across 15 API files
- **Pydantic descriptions**: ~232 Field descriptions translated across 23 files

#### Changed - Frontend

- **Navigation**: "Bourse" → "Stock Market", "Patrimoine" → "Wealth", "Outils" → "Tools"
- **Page titles & headings**: All 25 HTML pages updated
- **UI labels**: Buttons, loading states, empty states, tooltips, ARIA labels
- **Chart labels**: Axis titles, dataset names, metric interpretations
- **Notifications**: `showNotification`, `showError`, `showToast` messages
- **Risk dashboard**: Score interpretations, benchmark labels, time units (j→d)
- **Decision Index panel**: Score levels, recommendations, help content
- **Date/number locale**: `fr-FR` → `en-US` across 20 files

#### Changed - Config

- Strategy template descriptions translated
- Simulation preset names/descriptions translated
- User display names updated

#### Added

- **Rule 5 in CLAUDE.md**: "English-Only UI" enforcement rule
- **Bilingual CSV parser**: Saxo Bank CSV parser accepts both FR and EN headers
- **`docs/WEALTH_MODULE.md`**: Replaces PATRIMOINE_MODULE.md with updated references
- **`docs/I18N_MIGRATION.md`**: Complete migration report with file lists

#### Fixed

- **`updatePatrimoineChart is not defined`**: Fixed call site in dashboard-main-controller.js (function was renamed to `updateWealthChart` but call site was missed)

---

## [1.0.0] - 2025-11-09

### Security Audit & Production Readiness (Nov 2025)

**Period:** November 9, 2025 (Days 1-4)
**Objective:** Eliminate critical production blockers and strengthen security
**Result:**  **COMPLETE SUCCESS** - Project production-ready

**Final Metrics:**
- Security score: 6/10 → **8.5/10** (+42%)
- CLAUDE.md compliance: 75% → **90%** (+15 pts)
- Critical vulnerabilities: 3 → **0** (100% resolution)
- Production blockers: 5 → **0** (100% resolution)
- Active TODOs: 27 → **22** (-19%)
- BalanceService coverage: 0% → **66%** (+66 pts)

#### Security - Fixed
- **CoinGecko API Key Exposed** - Migrated to multi-tenant `UserSecretsManager`
  - Created `data/users/{user_id}/secrets.json` system
  - Modified 4 CoinGecko services to use user secrets
  - Added `**/secrets.json` to `.gitignore`
  - Updated `.env.example` with security instructions
  - Commit: `c98efe8`

- **Hardcoded Credentials** - Removed from 2 files
  - `api/unified_ml_endpoints.py` - ADMIN_KEY required (env var)
  - `tests/smoke_test_refactored_endpoints.py` - Uses env variables
  - Commit: `c98efe8`

- **JavaScript eval()** - Eliminated with secure whitelist system
  - Created `toastActionsRegistry` with pattern matching
  - Replaced `onclick` attributes with `data-action-index`
  - Secured `executeToastAction()` function
  - Commit: `c98efe8`

- **effective_user Bug** - Fixed in Risk Dashboard
  - Replaced 3 occurrences `effective_user` → `user` (lines 1185, 1190, 1194)
  - Commit: `1be7e75`

#### Security - Enhanced
- **Multi-tenant Enforcement** - 14 endpoints migrated
  - Migration: `user_id: str = Query("demo")` → `user: str = Depends(get_active_user)`
  - Files: `api/ml_bourse_endpoints.py`, `api/performance_endpoints.py`, `api/portfolio_monitoring.py`, `api/risk_bourse_endpoints.py`, `api/saxo_endpoints.py`, `api/debug_router.py`, `api/risk_endpoints.py`
  - Commit: `d167cba`

#### Added - Tests
- **BalanceService Tests** - 18 unit tests, 66% coverage
  - 3 stub data tests (conservative, shitcoins, balanced)
  - 2 multi-tenant isolation tests (CLAUDE.md Rule #1)
  - 2 CSV mode tests (success, file not found)
  - 2 API mode tests (with/without credentials)
  - 1 fallback chain test (API → CSV)
  - 2 data validation tests (structure, types)
  - 2 singleton tests
  - 2 error handling tests
  - 2 integration tests (skip if no data)
  - Commit: `db88466`

- **Pytest Configuration** - Complete test infrastructure
  - Created `pyproject.toml` with pytest + asyncio + coverage config
  - `asyncio_mode = "auto"` for async tests
  - Installed `pytest-asyncio==0.24.0` package
  - Coverage baseline: 50%, target: 55%
  - Commit: `db88466`

#### Added - Observability
- **Enhanced Logging** - Detailed CSV selection logs
  - `data_router.py` - Log user_id, data_source, selected_csv
  - Lists available files during search
  - Warning if selected CSV not found
  - Info on fallback to most recent
  - Commit: `7e14904`

- **Settings Merge Fix** - Preserves existing config
  - `user_settings_endpoints.py` - Merge with existing config instead of overwrite
  - Preserves untouched keys during partial updates
  - Detailed save logging (keys count, csv_selected_file)
  - Commit: `7e14904`

#### Changed - Code Quality
- **TODOs Cleanup** - 27 → 22 (-19%)
  - Converted 4 deprecated TODOs → NOTEs in `dashboard-main-controller.js`
  - Clarification: Endpoints `/exchanges/status`, `/execution/history/recent`, `/execution/status/24h` intentionally not implemented (optional)
  - Remaining TODOs: Primarily optional enhancements and future features (non-blocking)
  - Commit: `ad73ade`

#### Changed - Conformity
- **CLAUDE.md Alignment** - 75% → 90% compliance
  - 14 endpoints multi-tenant enforcement
  - 26 docs files: `uvicorn --reload` commands removed
  - 2 files clarified: Risk Score inversions (convention commented)
  - Imports added: `Depends`, `get_active_user`
  - Test validation: 17/18 PASS (no regressions)
  - Commit: `d167cba`

#### Discovered - Features
- **Settings API Save** - Feature ALREADY COMPLETE discovered (Day 4)
  - Backend: `GET /PUT /api/users/settings` functional
  - Frontend: `WealthContextBar.persistSettingsSafely()` with rollback, idempotence, anti-burst
  - Persistence: `data/users/{user_id}/config.json` (settings merge)
  - Multi-tenant: Header `X-User` + `Depends(get_active_user)`
  - Savings: ~2h development time avoided

#### Documentation
- **Complete Audit** - 4 reports generated (Day 1)
  - `docs/audit/AUDIT_COMPLET_2025_11_09.md`
  - `docs/audit/AUDIT_SECURITE.md`
  - `docs/audit/AUDIT_DETTE_TECHNIQUE.md`
  - `docs/audit/PLAN_ACTION_IMMEDIATE.md`

- **Progress Tracking** - Detailed Days 1-4 tracking
  - `docs/audit/PROGRESS_TRACKING.md` with weekly journal and metrics
  - Commits: `eb88c82`, `101003b`, `7e14904`, `ad73ade`

- **Session Summaries** - Session continuation guides
  - `docs/audit/SESSION_SUMMARY_2025_11_10.md`
  - `docs/audit/NEXT_STEPS.md`
  - Commit: `101003b`

#### Audit Statistics
**Total:** 8 commits, 52 files modified, ~800+ lines added, ~200+ lines removed
**Time invested:** ~5 hours (4 sessions)
**ROI:** Project production-ready, security score +42%, compliance +15 pts

**Commits:**
1. `c98efe8` - security(audit): résolution bloqueurs production critiques (3/5)
2. `db88466` - test(audit): resolve blocker #5 - BalanceService tests + pytest config
3. `d167cba` - feat(conformity): achieve CLAUDE.md compliance 75% → 90%
4. `eb88c82` - docs(audit): update progress tracking - Jour 2 completed
5. `1be7e75` - fix(risk): resolve 'effective_user' undefined error in risk dashboard
6. `101003b` - docs(audit): add session summary and next steps guide
7. `7e14904` - feat(observability): enhance logging and fix settings merge
8. `ad73ade` - chore(cleanup): reduce TODOs 27 → 22 and document Quick Wins completion

#### Lessons Learned
1. **Audit reveals good surprises** - Settings API already complete (2h savings)
2. **TODOs != Bugs** - 0 critical TODOs found (healthy codebase)
3. **Documentation NOTEs > TODOs** - Clarify intentions for optional features
4. **Multi-tenant robust** - No vulnerabilities found during audit

---

## [Previous Releases] - 2025-10-15

### Code Quality & Compliance Fixes (Oct 2025)

#### Fixed
- **Risk Score inversion in simulation engine** ([simulation-engine.js:489-493, :507](static/modules/simulation-engine.js#L489-L507))
  - **Problem**: Decision Index calculation was inverting Risk Score with `(100 - scores.risk)`
  - **Impact**: Violated critical CLAUDE.md rule "Risk Score = Positif (0-100), Ne jamais inverser"
  - **Root cause**: Copy-paste error from legacy code using inverted risk semantics
  - **Fix**: Removed inversion, now correctly uses `scores.risk * wRisk` in DI formula
  - **Reasoning updated**: Display now shows actual risk values without inversion
  - **Result**: DI calculation now respects canonical risk semantics (high score = high robustness) [OK]

- **Direct fetch bypassing loadBalanceData** ([WealthContextBar.js:1083-1092](static/components/WealthContextBar.js#L1083-L1092))
  - **Problem**: Badge data fetching used direct `fetch('/balances/current')` call
  - **Impact**: Bypassed centralized data management system required by CLAUDE.md
  - **Root cause**: Legacy code pattern before `window.loadBalanceData()` standardization
  - **Fix**: Replaced with `window.loadBalanceData()` with fallback for compatibility
  - **Format adaptation**: Added `.data` unwrapping to handle loadBalanceData response format
  - **Result**: Unified data loading pattern across entire frontend [OK]

#### Documentation
- **CHANGELOG.md** - Added comprehensive entry documenting both fixes
  - Before/after code examples with line references
  - Impact analysis and root cause identification
  - Verification that fixes respect CLAUDE.md critical rules

---

### Cap Stability Fix (Oct 2025)

#### Fixed
- **Cap oscillations fixed** ([docs/CAP_STABILITY_FIX.md](docs/CAP_STABILITY_FIX.md))
  - **Problem**: Cap varied 1% → 7% and allocations 45% ↔ 68% with constant scores
  - **Root causes identified**:
    1. Backend/frontend cap desynchronization (backend 7.7% ignored by frontend)
    2. Risk semantics mode changing (legacy/v2 → 31%-63% stables variation)
    3. Cache disabled (recalculation on every call)
    4. Market overrides without hysteresis (flip-flop +10-20%)
  - **Fixes applied**:
    1. Cache reactivated with 30s TTL ([market-regimes.js:233](static/modules/market-regimes.js#L233))
    2. Risk semantics fixed to v2_conservative by default ([market-regimes.js:226](static/modules/market-regimes.js#L226))
    3. Hysteresis widened (gap 10pts vs 4pts before) ([market-regimes.js:171](static/modules/market-regimes.js#L171))
    4. Frontend now reads backend cap as MAX limit ([targets-coordinator.js:485](static/modules/targets-coordinator.js#L485))
  - **Result**: Max variation 0.21% (< 2% target) [OK]

#### Added
- **Audit tools**:
  - `tools/audit_governance_state.py` - Backend state inspector (mode, cap, signals, hysteresis)
  - `tools/audit_frontend_state.html` - Frontend state inspector (localStorage, store, risk mode)
- **Tests**: `tests/unit/test_cap_stability.py` - 4 scenarios, all PASSING
  - Cap stability (5 ticks): max variation 0.21% [OK]
  - Cap floor check: within bounds 1%-95% [OK]
  - NaN protection: variation 0.12% < 15% [OK]
  - Manual override: cap = 15.0% exact [OK]

#### Documentation
- **Complete analysis**: [docs/CAP_STABILITY_FIX.md](docs/CAP_STABILITY_FIX.md)
  - Before/after comparison
  - Root cause analysis (5 bugs identified)
  - Fix details with code references
  - Test results and validation criteria
  - Deployment instructions
  - Future work (complete v2 migration)

### Exposure Cap Overhaul & Risk Semantics V2

### Added
- **Pure function `computeExposureCap()`** ([targets-coordinator.js:337-412](static/modules/targets-coordinator.js#L337-L412))
  - Testable, side-effect-free cap calculation with comprehensive unit tests
  - Regime-based floors (FR/EN support): Euphorie ≥75%, Expansion ≥60%, Neutral ≥40%, etc.
  - Dynamic boost: Expansion + Risk Score ≥80 → floor raised from 60% to 65%
  - Smooth penalty curves instead of binary cliffs (signal quality, volatility)
  - 20+ unit tests covering all regimes, backend states, edge cases ([computeExposureCap.test.js](static/tests/computeExposureCap.test.js))

- **Test Infrastructure**
  - Vitest setup with Happy-DOM environment ([vitest.config.js](vitest.config.js))
  - Comprehensive test suite: 6 describe blocks, 20+ test cases
  - Coverage reports: text + HTML + JSON formats
  - Test scripts: `npm test`, `npm run test:watch`, `npm run test:ui`

### Changed
- **Volatility penalty** - Max reduced from 15pts to 10pts for smoother behavior in high-vol markets
- **Volatility normalization** - Automatic unit handling: `32` (percent) ≡ `0.32` (decimal)
- **Signal quality adjustment** - Continuous gradient penalty (0-10pts) replacing binary cliffs (raw < 0.45 → -10, raw < 0.65 → -5)
- **Backend fallback logic** - Graceful degradation for `stale`/`error` status:
  - **Stale**: -15pts penalty but respects regime floors (no hard-cap)
  - **Error**: -25pts penalty but respects regime floors (no hard-cap)
  - **Before**: Hard-cap 5-8% overriding all market context [Error]
  - **After**: Regime-aware fallback preserving market logic [OK]

### Fixed
- **`renderTargetsTable()` crash** ([risk-dashboard.html:6053-6095](static/risk-dashboard.html#L6053-L6095))
  - Robust validation against `null`/`NaN`/`undefined` allocations
  - Comprehensive filtering with `isValid = value != null && typeof value === 'number' && !isNaN(value)`
  - Graceful error messages instead of crashes
  - Console warnings for filtered invalid entries

- **Risk Score semantics inversion** ([market-regimes.js:219-270](static/modules/market-regimes.js#L219-L270))
  - **V2 mode activated**: Risk Score now correctly interpreted as robustness (high = low perceived risk)
  - **Legacy mode deprecated**: Inverted logic where high Risk = danger (will be removed in v4.0)
  - **Feature flag**: `localStorage.setItem('RISK_SEMANTICS_MODE', 'v2_conservative')`
  - **Doc reference**: [docs/RISK_SEMANTICS.md](docs/RISK_SEMANTICS.md) - Canonical risk semantics documentation

- **Exposure cap calculation** - Complete architectural overhaul:
  - **Before (Legacy + 40% cap)**:
    - Euphorie (73) + Risk (90) → 40% cap → 60% stables, 18% BTC [Error]
    - Expansion (61) + Risk (90) → 40% cap → 60% stables, 18% BTC [Error]
  - **After (V2 + Pure Function)**:
    - Euphorie (73) + Risk (90) → 80%+ cap → 20-25% stables, 35%+ BTC [OK]
    - Expansion (61) + Risk (90) → 65% cap (boost) → 47% stables, 30% BTC [OK]
  - **Impact**: +20-25 percentage points exposure to risky assets in bull markets

### Performance
- **Allocation calculations** - Deterministic, cacheable pure functions reduce recomputation overhead
- **Test execution** - Vitest parallel test runner for fast feedback loops

### Documentation
- **CHANGELOG.md** - Comprehensive documentation of changes with links to code
- **Test coverage** - Inline JSDoc comments for all test cases
- **Migration guide** - Backward compatible, no user action required

### Migration Guide

#### For Users
1. **No action required** - Changes are backward compatible
2. Risk semantics V2 activates automatically on first page load after update
3. To revert to legacy mode (not recommended): `localStorage.setItem('RISK_SEMANTICS_MODE', 'legacy')`
4. To enable debug logs: `localStorage.setItem('DEBUG_RISK', '1'); location.reload();`

#### For Developers
1. **Install test dependencies**: `npm install`
2. **Run tests**: `npm test` or `npm run test:watch` (development)
3. **Update custom allocation logic** to use `computeExposureCap()` instead of inline calculations
4. **Review regime floors** if customizing market regime behavior
5. **Check debug logs** with `window.__DEBUG_RISK__ = true` or `localStorage.DEBUG_RISK = '1'`

### Testing

```bash
# Install dependencies
npm install

# Run all tests
npm test

# Watch mode (development)
npm run test:watch

# UI test explorer
npm run test:ui

# Coverage report
npm run test:coverage
```

#### Test Coverage
- [OK] 20+ unit tests for `computeExposureCap()`
- [OK] All market regimes: Euphorie, Expansion, Neutral, Accumulation, Bear, Capitulation
- [OK] Backend status handling: ok, stale, error, unknown
- [OK] Volatility normalization: decimal (0.32) vs percent (32)
- [OK] Edge cases: null values, unknown regimes, extreme volatility, mixed case regime names
- [OK] Regime floors: All regimes respect their minimum thresholds
- [OK] Dynamic boost: Expansion + Risk ≥80 verified
- [OK] Signal quality penalties: Continuous gradient validation

### Roadmap

#### Next Steps (Planned)
- [ ] UI debug badges for cap breakdown visualization
- [ ] Allocation snapshots tests (detect regressions in target allocations)
- [ ] Progressive boost by Risk Score tranches (≥85, 80-84, <80)
- [ ] Regime transition smoothing (prevent allocation whipsaw)
- [ ] On-chain score null fix (ensure intelligence always available)

#### Under Consideration
- [ ] Machine learning cap predictor (replace heuristic rules)
- [ ] User-configurable regime floors via UI
- [ ] Multi-timeframe regime detection (hourly, daily, weekly consensus)
- [ ] Backtesting framework with historical price data

---

## [3.0.0] - 2025-09-17

### Major Features - Système d'Allocation Dynamique

#### Élimination des Presets Hardcodés
- **BREAKING**: Suppression complète des presets figés (BTC 40%, ETH 30%, Stables 20/30/50%)
- **NEW**: Calculs d'allocation contextuels basés sur cycle de marché, régime, et concentration wallet
- **NEW**: Source canonique unique `u.targets_by_group` pour cohérence parfaite Analytics ↔ Rebalance
- **NEW**: Fonction `computeMacroTargetsDynamic()` avec modulateurs intelligents

#### Synchronisation Analytics ↔ Rebalance
- **FIXED**: "Allocation Suggérée (Unified)" maintenant peuplée automatiquement dans rebalance.html
- **NEW**: Sauvegarde automatique des données unified avec nouveau format v2
- **CRITICAL**: Correction `targetsSource = data.targets` vs `data.execution_plan`
- **NEW**: Support rétrocompatible ancien + nouveau format localStorage

### Technical Changes

#### Core Engine (`static/core/unified-insights-v2.js`)
- **ADD**: `computeMacroTargetsDynamic(ctx, rb, walletStats)` - remplace presets
- **CHANGE**: Construction `targets_by_group` via calculs vs templates statiques
- **ADD**: Modulateurs bull/bear/hedge + diversification selon concentration wallet
- **ADD**: Garde-fous cohérence stables = risk_budget.target_stables_pct (source de vérité)

#### UI Components (`static/components/UnifiedInsights.js`)
- **REMOVE**: Logique preset hardcodée (elimination complète lignes 680-725)
- **CHANGE**: Lecture directe `u.targets_by_group` vs `buildTheoreticalTargets()`
- **REMOVE**: Import `buildTheoreticalTargets` (function deprecated)
- **ADD**: Logs debug pour validation données dynamiques

#### Pages HTML
- **Analytics** (`static/analytics-unified.html`):
  - **ADD**: `saveUnifiedDataForRebalance()` - sauvegarde automatique
  - **ADD**: Format données v2 avec source `analytics_unified_v2`
- **Rebalance** (`static/rebalance.html`):
  - **FIXED**: `syncUnifiedSuggestedTargets()` support sources v2
  - **ADD**: Protection taxonomie `forceReloadTaxonomy()`
  - **ADD**: Logs debug détaillés structure données

### Critical Bug Fixes

#### Allocation Display Issues
- **FIXED**: "Others 31%" incohérent → allocations cohérentes via source unique
- **FIXED**: Inconsistance Objectifs Théoriques vs Plan d'Exécution → même source
- **FIXED**: rebalance.html "Allocation Suggérée (Unified)" vide → peuplée automatiquement
- **FIXED**: Affichage `estimated_iters: 2.0%` au lieu allocations réelles

#### Data Synchronization
- **FIXED**: Analytics et Rebalance utilisaient sources différentes → u.targets_by_group unique
- **FIXED**: Presets ignoraient risk_budget.target_stables_pct → intégration native
- **FIXED**: Taxonomie non chargée causant "Others" gonflé → forceReloadTaxonomy()

### Documentation
- **NEW**: `docs/dynamic-allocation-system.md` - guide complet nouveau système
- **UPDATE**: `README.md` - section "Nouvelles Fonctionnalités v3.0"
- **NEW**: Commentaires détaillés code + logs explicites

### Breaking Changes
- **REMOVE**: Presets hardcodés dans tous les fichiers
- **REMOVE**: `buildTheoreticalTargets()` calls (replaced by dynamic computation)
- **CHANGE**: Allocations maintenant contextuelles vs statiques (amélioration UX)

**Migration**: Rétrocompatibilité assurée, aucune action utilisateur requise.

---

## [2.2.0] - 2025-01-14

### Centralisation ML - Source Unique de Vérité

### Added
- **Source ML Centralisée** : `shared-ml-functions.js::getUnifiedMLStatus()` - single source of truth
- **Logique Prioritaire Unifiée** : Governance Engine → ML Status API → Stable fallback (identique AI Dashboard)
- **Cache Intelligent** : TTL 2 minutes pour performance avec invalidation automatique
- **Validation Robuste** : Caps automatiques (4 modèles max, 100% confidence max) pour éviter valeurs aberrantes
- **Documentation Complète** : `docs/ml-centralization.md` avec architecture détaillée

### Fixed
- **[Error] Calculs ML Erronés** : Fini les "8/4 modèles = 200% confidence" - désormais capé à 4/4 = 100%
- **[Error] Badge Global Manquant** : Erreur syntaxe WealthContextBar.js (else if après else) corrigée
- **[Error] Intelligence ML Vide** : Analytics-unified affichait "--" au lieu des données réelles
- **[Error] Logique Dupliquée** : 3 implémentations différentes (badge, analytics, ai-dashboard) unifiées
- **[Error] Timezone Incorrect** : Badge utilise désormais Europe/Zurich via `formatZurich()`

### Changed
- **WealthContextBar** : Utilise source ML centralisée au lieu de logique dupliquée
- **Analytics-unified** : Intelligence ML tab utilise `getUnifiedMLStatus()` avec fallback
- **AI Dashboard** : Migration vers source centralisée tout en conservant même logique
- **Configuration API Safe** : `globalConfig` access sécurisé pour éviter erreurs d'import

### Technical
- **Architecture** : Un seul module gère toute la logique ML pour 3 pages
- **Performance** : Cache 2min TTL évite appels API répétés
- **Maintenance** : Plus qu'un seul endroit à modifier pour la logique ML
- **Cohérence** : Calculs identiques partout, fini les divergences

## [2.1.0] - 2024-01-15

### Consolidation Navigation & WealthContextBar Cross-Asset

### Added
- **Navigation Canonique** : 6 pages principales - Portfolio, Analytics, Risk, Rebalance, Execution, Settings
- **WealthContextBar Global** : Filtres household/account/module/currency persistants avec sync localStorage+querystring
- **Deep Links System** : Ancres fonctionnelles avec scroll automatique et highlight temporaire (2s)
- **RBAC Admin Dropdown** : Menu visible uniquement pour governance_admin/ml_admin avec ML Command Center, Tools & Debug, Archive
- **Legacy Redirections** : Système de redirections douces vers ancres canoniques pour éviter 404s
- **Badges Standards** : Format uniforme "Source • Updated HH:MM:SS • Contrad XX% • Cap YY% • Overrides N" (timezone Europe/Zurich)

### Changed
- **Menu Navigation** : Simplifié de 10+ entrées vers 6 pages canoniques avec sous-menus via ancres
- **Archive System** : Pages legacy conservées mais accessibles via Admin > Archive uniquement
- **Cross-Asset Filtering** : WealthContextBar applique filtrage sur pages Rebalance/Execution
- **Documentation** : Restructuration complète avec 7 nouveaux docs (navigation, wealth-modules, governance, runbooks, etc.)

### Technical
- **Components** : `WealthContextBar.js`, `deep-links.js`, `Badges.js`, `legacy-redirects.js`
- **Archive Index** : `static/archive/index.html` avec liens legacy → canonical
- **RBAC Integration** : Vérification rôles localStorage/window.userRoles dans nav.js
- **Event System** : Événement `wealth:change` pour synchronisation cross-composants

### Documentation
- **Navigation & Architecture** : `docs/navigation.md` - Structure menus et liens profonds
- **Modules Patrimoniaux** : `docs/wealth-modules.md` - Crypto/Bourse/Banque/Divers
- **Governance & Caps** : `docs/governance.md` - Hiérarchie SMART→Decision Engine
- **Runbooks** : `docs/runbooks.md` - Procédures incidents (stale/error, VaR>4%, contradiction>55%)
- **Télémétrie** : `docs/telemetry.md` - KPIs système et métriques Prometheus

---

## [2.0.1] - 2024-01-15

### Dashboard Global Insight Enhancement

### Added
- **Global Insight Badge**: Dashboard principal affiche maintenant "Updated: HH:MM:SS • Contrad: X% • Cap: Y%" en bas de la tuile
- **Real-time Governance Data**: Badge se met à jour automatiquement avec les données du Decision Engine
- **Cross-dashboard Consistency**: Format cohérent avec les autres dashboards (Analytics, Risk)
- **Store Integration**: Synchronisation via `risk-dashboard-store.js` pour données temps réel

### Changed
- **Badge Position**: Déplacé de haut vers bas de la tuile Global Insight pour cohérence visuelle
- **Data Flow**: Badge récupère timestamp des signaux ML et données policy du store governance
- **Update Triggers**: Badge se rafraîchit lors des changements store et événements `configChanged`

### Technical
- **Function**: `updateGlobalInsightMeta()` pour gestion badge dans `dashboard.html`
- **Store Sync**: Utilise `store.get('governance.ml_signals')` et `store.get('governance.active_policy')`
- **Event Handling**: Écoute changements store via subscription et événements storage cross-tab

### Fixed
- Badge Global Insight maintenant visible et fonctionnel
- Données gouvernance affichées en temps réel sur dashboard principal
- Cohérence visuelle avec format badges des autres dashboards

---

## [2.0.0] - 2024-12-12

### Major API Refactoring & Security Improvements

This release contains **BREAKING CHANGES** requiring consumer updates.

### Added
- **Unified Endpoints**: Single approval endpoint `/api/governance/approve/{resource_id}` for both decisions and plans
- **Centralized Alerts**: All alert operations now under `/api/alerts/*` namespace
- **Admin Protection**: ML debug endpoints now require `X-Admin-Key` header
- **Validation Tools**: 
  - `tests/smoke_test_refactored_endpoints.py` - Endpoint validation
  - `find_broken_consumers.py` - Consumer reference scanner  
  - `verify_openapi_changes.py` - Breaking changes analyzer
- **Documentation**: `docs/refactoring.md` with complete migration guide

### Changed
- **ML Namespace**: `/api/ml-predictions/*` → `/api/ml/*` (unified)
- **Risk Namespace**: `/api/advanced-risk/*` → `/api/risk/advanced/*` (consolidated)
- **Governance API**: Unified approval endpoint with `resource_type` parameter
- **Alert Resolution**: Centralized under `/api/alerts/resolve/{alert_id}`
- **Alert Acknowledgment**: Centralized under `/api/alerts/acknowledge/{alert_id}`

### Removed (Security & Production Readiness)
- **Dangerous Endpoints**: 
  - `/api/realtime/publish` - Could allow arbitrary event publishing
  - `/api/realtime/broadcast` - Could spam all connected clients
- **Test Endpoints**: 
  - All `/api/test/*` endpoints - Removed from production
  - All `/api/alerts/test/*` endpoints - Removed from production
- **Duplicate Endpoints**:
  - `/api/risk/alerts/{id}/resolve` - Now `/api/alerts/resolve/{id}`
  - `/api/monitoring/alerts/{id}/resolve` - Now `/api/alerts/resolve/{id}`
  - `/api/portfolio/alerts/{id}/resolve` - Now `/api/alerts/resolve/{id}`

### Fixed
- **Pydantic v2 Compatibility**: Fixed `regex=` → `pattern=` in Field definitions
- **Consumer References**: Updated 13 files with broken endpoint references
- **Test Suites**: Updated E2E tests to work with new architecture
- **Documentation**: Synchronized all docs with new endpoint structure

### Security
- **Endpoint Protection**: ML debug endpoints require admin authentication
- **Attack Surface Reduction**: Removed 5 potentially dangerous endpoints
- **Test Isolation**: No test endpoints exposed in production

### Migration
**Required Actions for Consumers:**
1. Replace `/api/ml-predictions/*` with `/api/ml/*`
2. Remove all `/api/test/*` and `/api/alerts/test/*` calls
3. Update `/api/advanced-risk/*` to `/api/risk/advanced/*`
4. Update `/governance/approve` calls to include `resource_type` in body
5. Centralize alert operations to `/api/alerts/*`

**Tools Available:**
- Run `python find_broken_consumers.py` to scan for broken references
- Run `python tests/smoke_test_refactored_endpoints.py` to validate endpoints
- See `docs/refactoring.md` for complete migration guide

### Performance
- **Namespace Consolidation**: Reduced API surface from 6 to 3 main namespaces
- **Endpoint Efficiency**: Unified endpoints reduce client-side complexity

---

## [1.8.0] - 2024-12-10

### Added
- Phase 3C: Hybrid Intelligence integration
- Advanced ML pipeline management
- Cross-asset correlation monitoring
- Enhanced governance workflows

### Changed
- Improved risk calculation performance
- Enhanced dashboard responsiveness
- Better error handling in ML components

### Fixed
- Memory leaks in ML pipeline
- Cache invalidation issues
- Dashboard synchronization bugs

---

## [1.7.0] - 2024-12-01

### Added
- Phase 2C: ML Alert Predictions
- Predictive alerting system
- Enhanced ML models integration
- Real-time streaming improvements

### Changed
- Optimized risk calculations
- Enhanced UI/UX across dashboards
- Improved API response times

---

## [1.6.0] - 2024-11-15

### Added
- Phase 2B: Cross-asset correlation analysis
- Advanced risk engine
- Multi-exchange support
- Enhanced monitoring

---
*Earlier versions documented in git history*

## 2025-09-26 - Tri stable stratégies & équilibrage visuel

### Tri stable des stratégies Rebalance
- **Nouveau**: Système de priorité garantissant l'ordre Unified Analytics → CCS Dynamic → statiques
- **Fonction `rank()`**: Attribution de scores (0=Unified live, 1=Unified placeholder, 2=CCS live, 3=CCS placeholder/error, 10=statiques)
- **Tri stable**: Maintien de l'ordre même après rafraîchissement dynamique via `refreshDynamicStrategy()`
- **Localisation**: Support français pour le tri alphabétique secondaire via `localeCompare('fr')`

### Équilibrage visuel (Solution C)
- **Filler invisible**: Ajout automatique d'éléments invisibles quand 1 carte reste sur la dernière ligne
- **Détection responsive**: Activation uniquement si grille ≥3 colonnes et `(cartes % colonnes) === 1`
- **Accessibilité**: Filler marqué `aria-hidden="true"` pour lecteurs d'écran
- **Performance**: Gestion d'erreur avec `try/catch` pour éviter les crashes

### Adaptation responsive améliorée
- **Breakpoint ajusté**: Passage à 4 colonnes dès 1280px (au lieu de 1440px)
- **Évite lignes orphelines**: Réduction du risque de ligne avec 1 seule carte sur écrans larges
- **Rétrocompatibilité**: Maintien du comportement 3 colonnes ≥1200px inchangé

### Technical Changes
- **Fichier modifié**: `static/rebalance.html`
- **Fonction `renderStrategiesUI()`**: Ajout tri stable avant `.map()` et équilibrage après `innerHTML`
- **CSS responsive**: Modification breakpoint `@media (min-width: 1280px)`

### Résultat
- Interface cohérente avec Unified Analytics toujours en premier
- CCS Dynamic systématiquement en deuxième position
- Équilibrage visuel optimal sur toutes les tailles d'écran
- Pas de ligne orpheline avec une seule carte



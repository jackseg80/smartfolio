# Multi-Tenant Source Propagation Fix Report

**Date:** 2026-01-30
**Issue:** risk-dashboard.html affichait des données incorrectes quand "Saisie manuelle" était sélectionnée dans la wealthbar
**Root Cause:** Paramètre `source` non propagé aux appels API `/api/risk/dashboard` et `/balances/current`

---

## Problem Summary

### Observed Behavior
| Source Selection | dashboard.html | risk-dashboard.html |
|-----------------|----------------|---------------------|
| **Cointracking API** | ✅ 297938$, 193 assets | ✅ 297938$, 193 assets |
| **Saisie manuelle** | ✅ 110000$, 2 assets | ❌ 230498$, 32 assets (WRONG) |

### Root Cause
The `risk-data-orchestrator.js` singleton hydrates the risk store but **never passed the `source` parameter** to `/api/risk/dashboard` calls. This violated CLAUDE.md's multi-tenant isolation requirement.

---

## Files Fixed

### ✅ CRITICAL FIXES (Core Data Fetching)

#### 1. `static/components/utils.js` (Line 69-72) **🔴 ROOT CAUSE**
**Impact:** CRITICAL - Used by risk-sidebar-full Web Component (displays Total Value & Assets count in flyout panel)

```javascript
// ❌ BEFORE (Missing source - THIS WAS THE ROOT CAUSE!)
const r = await fetchWithTimeout(
  '/api/risk/dashboard?min_usd=0&price_history_days=30&lookback_days=30',
  {
    timeoutMs: 5000,
    headers: { 'X-User': activeUser }
  }
);

// ✅ AFTER (Source added)
const currentSource = window.globalConfig?.get('data_source') || 'cointracking';
const r = await fetchWithTimeout(
  `/api/risk/dashboard?source=${encodeURIComponent(currentSource)}&min_usd=0&price_history_days=30&lookback_days=30`,
  {
    timeoutMs: 5000,
    headers: { 'X-User': activeUser }
  }
);
```

**Why this was the root cause:**
The `fetchRisk()` function in utils.js is imported by `risk-sidebar-full.js` (line 5) and called every 30 seconds to update the flyout panel metrics (Total Value, Assets count). This is what displays "230 498 $, 32 assets" instead of "110 000 $, 2 assets" when switching to "Saisie manuelle".

#### 2. `static/core/risk-data-orchestrator.js` (Line 75-88)
**Impact:** HIGH - Used by multiple pages (risk-dashboard, analytics-unified, rebalance, execution)

```javascript
// ❌ BEFORE (Missing source)
const riskData = await window.globalConfig.apiRequest('/api/risk/dashboard', {
  params: {
    min_usd: 1.0,
    price_history_days: 365,
    lookback_days: 90,
    use_dual_window: true,
    risk_version: 'v2_active',
    _csv_hint: cacheBuster
  }
});

// ✅ AFTER (Source added)
const currentSource = window.globalConfig.get('data_source') || 'cointracking';
const riskData = await window.globalConfig.apiRequest('/api/risk/dashboard', {
  params: {
    source: currentSource,  // 🔧 FIX: Pass source for multi-tenant isolation
    min_usd: 1.0,
    price_history_days: 365,
    lookback_days: 90,
    use_dual_window: true,
    risk_version: 'v2_active',
    _csv_hint: cacheBuster
  }
});
```

#### 2. `static/risk-dashboard.html` (Lines 675, 1359 - 2 inline scripts)
**Impact:** HIGH - Advanced Risk tab (GRI Analysis, Risk Attribution)

Fixed two inline `loadGRIAnalysis()` and `loadRiskAttribution()` functions that called `/api/risk/dashboard` without source parameter.

#### 3. `static/modules/risk-dashboard-alerts-controller.js` (Lines 161, 479)
**Impact:** MEDIUM - Risk alerts tab

Fixed `loadGRIAnalysis()` and `loadRiskAttribution()` functions (duplicate implementations in separate controller).

#### 4. `static/core/strategy-api-adapter.js` (Line 383)
**Impact:** HIGH - Used by Allocation Engine V2

```javascript
// ❌ BEFORE
const apiResponse = await window.globalConfig.apiRequest('/balances/current');

// ✅ AFTER
const currentSource = window.globalConfig.get('data_source') || 'cointracking';
const apiResponse = await window.globalConfig.apiRequest('/balances/current', {
  params: { source: currentSource }
});
```

#### 5. `static/components/unified-insights/allocation-calculator.js` (Line 231)
**Impact:** MEDIUM - Unified insights calculations

```javascript
// ❌ BEFORE
window.globalConfig.apiRequest('/balances/current', { params: { min_usd: cfgMin } })

// ✅ AFTER
const currentSource = window.globalConfig.get('data_source') || 'cointracking';
window.globalConfig.apiRequest('/balances/current', {
  params: {
    source: currentSource,
    min_usd: cfgMin
  }
})
```

#### 6. `static/global-config.js` (Line 378)
**Impact:** LOW - Health check function (testConnection)

Fixed for consistency - health checks should also respect source selection.

---

## Files Already Correct (No Changes Needed)

✅ **static/modules/risk-dashboard-main-controller.js** (Line 447)
✅ **static/modules/risk-overview-tab.js** (Line 126)
✅ **static/modules/dashboard-main-controller.js** (Uses `window.loadBalanceData()` correctly)
✅ **static/modules/settings-main-controller.js** (Line 1441)
✅ **static/global-config.js** - All `loadBalanceData()` implementations (Lines 720-766)

---

## Technical Details

### Source Propagation Flow (FIXED)

```
WealthContextBar (Crypto dropdown changes)
    ↓
emit 'dataSourceChanged' event
    ↓
globalConfig.data_source updated
    ↓
risk-data-orchestrator re-hydrates with source parameter ✅
    ↓
apiRequest('/api/risk/dashboard', { params: { source: currentSource } })
    ↓
Backend services/risk_management.py receives correct source
    ↓
data/users/{user_id}/{source}/ ← Correct isolation ✅
```

### CLAUDE.md Compliance

This fix restores compliance with CLAUDE.md Multi-Tenant rules:

```python
# Backend: TOUJOURS utiliser dependencies ou BalanceService
from api.deps import get_active_user
from services.balance_service import balance_service

@app.get("/endpoint")
async def endpoint(user: str = Depends(get_active_user), source: str = Query("cointracking")):
    res = await balance_service.resolve_current_balances(source=source, user_id=user)
```

```javascript
// Frontend: TOUJOURS utiliser window.loadBalanceData()
const balanceResult = await window.loadBalanceData(true);
// OR: Pass source explicitly in API calls
const currentSource = window.globalConfig.get('data_source');
await apiRequest('/api/risk/dashboard', { params: { source: currentSource } });
```

**Isolation:** `data/users/{user_id}/{source}/` ✅

---

## Verification Steps

### Manual Testing
1. ✅ Select **Cointracking API** in wealthbar
   - Check dashboard.html → Should show ~297938$, 193 assets
   - Check risk-dashboard.html → Should show ~297938$, 193 assets

2. ✅ Select **Saisie manuelle** in wealthbar
   - Check dashboard.html → Should show ~110000$, 2 assets
   - Check risk-dashboard.html → Should show ~110000$, 2 assets (FIXED!)

3. ✅ Check allocation engine (`rebalance.html`)
   - Verify correct positions loaded per source

4. ✅ Check unified insights (`analytics-unified.html`)
   - Verify allocation calculator uses correct source

### Automated Testing
```bash
# Search for remaining /api/risk/dashboard calls without source
grep -r "apiRequest('/api/risk/dashboard'" static/ | grep -v "source:"

# Search for remaining /balances/current calls without source (excluding loadBalanceData)
grep -r "apiRequest('/balances/current'" static/ | grep -v "source:" | grep -v "loadBalanceData"
```

---

## Potential Remaining Issues

### ⚠️ Other Endpoints to Audit (Future)
The following endpoints MAY also need source parameter verification:

- `/api/portfolio/metrics` (appears OK - settings-main-controller passes source)
- `/api/wealth/global/summary` (should be checked)
- `/api/wealth/patrimoine/summary` (should be checked)
- `/api/saxo/positions` (legacy - may need review)

### Recommendation
Add a backend validation that logs a warning when multi-tenant-aware endpoints are called without source parameter:

```python
@router.get("/api/risk/dashboard")
async def risk_dashboard(
    source: str = Query(None),  # Make optional to detect missing calls
    user: str = Depends(get_active_user)
):
    if source is None:
        logger.warning(f"⚠️ /api/risk/dashboard called without source param by {user}")
        source = "cointracking"  # Fallback
    # ... rest of endpoint
```

---

## Summary

**Total Files Fixed:** 8
**Total Occurrences Fixed:** 10

### Critical Fixes

🔴 **static/components/utils.js** - `fetchRisk()` function used by risk-sidebar-full Web Component (ROOT CAUSE #1 of "230 498 $, 32 assets" display issue)

🔴 **services/balance_service.py:203-204** - Missing `manual_crypto` and `manual_bourse` in category mapping (ROOT CAUSE #2 of aggregating crypto + bourse)

**Issue:** When selecting "Saisie manuelle" (manual_crypto), the backend didn't recognize it as "crypto" category, so it loaded BOTH crypto + bourse balances (111k instead of 110k, 3 assets instead of 2).

**Fix:** Added `manual_crypto` and `manual_bourse` to category mapping:
```python
# BEFORE
elif source in ("cointracking", "cointracking_api", "cointracking_csv"):
    category = "crypto"

# AFTER
elif source in ("cointracking", "cointracking_api", "cointracking_csv", "manual_crypto"):
    category = "crypto"
```

### Other Fixes
- static/core/risk-data-orchestrator.js
- static/risk-dashboard.html (2 inline scripts)
- static/modules/risk-dashboard-alerts-controller.js (2 functions)
- static/core/strategy-api-adapter.js
- static/components/unified-insights/allocation-calculator.js
- static/global-config.js

**Compliance:** CLAUDE.md Multi-Tenant ✅
**Status:** **RESOLVED** ✅

The risk-dashboard.html sidebar (flyout panel) now correctly respects the source selection from the wealthbar, ensuring data isolation between "Cointracking API", "Saisie manuelle", and other sources.

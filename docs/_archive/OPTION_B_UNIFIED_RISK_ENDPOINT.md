# Option B: Unified Risk Endpoint Specification

**Status**:  Specification (Not Implemented)
**Date**: 2025-10-01
**Context**: Fix for flyout panel data loading issues (analytics-unified.html, rebalance.html)

---

## Problem Statement

Currently, the risk sidebar component (`risk-sidebar-full.js`) needs data from multiple sources:

1. **CCS Score** - Calculated frontend-side in risk-dashboard.html
2. **On-Chain Score** - Calculated frontend-side via multiple API calls + JS aggregation
3. **Cycle Data** - Calculated frontend-side (months, phase, ccsStar)
4. **Blended Decision** - Calculated frontend-side with weighted formula
5. **Risk Score** - Available in `/api/risk/dashboard` OK
6. **Governance** - Partially available [OK]
7. **Alerts** - Available [OK]

**Current State**:
- `risk-dashboard.html` works because it calculates everything frontend-side
- `analytics-unified.html` and `rebalance.html` rely on API polling → missing data → sections hidden (Option A fix)

**Goal**: Provide a **single unified endpoint** that returns all data in the exact format expected by the sidebar, eliminating the need for:
- Multiple API calls
- Frontend calculations
- Conditional section hiding (Option A workaround)

---

## Endpoint Specification

### Request

```
GET /api/risk/unified
```

**Query Parameters**:
- `source` (string, default: "cointracking") - Data source: `cointracking` | `cointracking_api` | `stub`
- `user_id` (string, from header or query) - User ID for multi-tenant isolation
- `min_usd` (float, default: 1.0) - Minimum USD threshold for holdings
- `price_history_days` (int, default: 30) - Historical price window
- `lookback_days` (int, default: 30) - Correlation lookback window

**Headers**:
- `X-User: {user_id}` (optional, fallback to query param)

**Example**:
```bash
curl "http://localhost:8080/api/risk/unified?source=cointracking&min_usd=1.0" \
  -H "X-User: demo"
```

---

### Response Schema

**Status**: 200 OK
**Content-Type**: application/json

```json
{
  "success": true,
  "timestamp": "2025-10-01T14:32:45.123Z",
  "user_id": "demo",
  "source_used": "cointracking",

  "ccs": {
    "score": 78.5,
    "label": "Modéré",
    "weights": {
      "mvrv": 0.30,
      "nupl": 0.25,
      "pi_cycle": 0.20,
      "puell_multiple": 0.15,
      "others": 0.10
    },
    "signals": {
      "fear_greed": 65,
      "altcoin_season_index": 42
    },
    "model_version": "ccs-v2.1",
    "last_update": "2025-10-01T14:30:00Z"
  },

  "scores": {
    "onchain": 75.2,
    "risk": 68.5,
    "blended": 72.3,
    "blendedDecision": 72.3
  },

  "cycle": {
    "months": 18,
    "ccsStar": 81.2,
    "phase": {
      "phase": "EXPANSION",
      "emoji": "Growth",
      "confidence": 0.87
    },
    "multiplier": 1.05,
    "weight": 0.35
  },

  "targets": {
    "current": {
      "BTC": 0.25,
      "ETH": 0.30,
      "Stablecoins": 0.15
    },
    "proposed": {
      "BTC": 0.28,
      "ETH": 0.32,
      "Stablecoins": 0.12
    },
    "changes": [
      {
        "asset": "BTC",
        "from": 0.25,
        "to": 0.28,
        "delta": 0.03,
        "reason": "Cycle momentum"
      },
      {
        "asset": "Stablecoins",
        "from": 0.15,
        "to": 0.12,
        "delta": -0.03,
        "reason": "Risk-on environment"
      }
    ],
    "model_version": "tgt-v1.3"
  },

  "governance": {
    "mode": "guardian",
    "current_state": "ACTIVE",
    "contradiction_index": 0.42,
    "cap_daily": 0.0123,
    "ml_signals_timestamp": "2025-10-01T14:30:02Z",
    "active_policy": {
      "cap_daily": 0.0123,
      "max_single_trade": 0.05
    },
    "constraints": {
      "max_drawdown": false,
      "correlation_limit": true
    },
    "next_update_time": "2025-10-01T15:00:00Z",
    "pending_approvals": 2,
    "last_decision_id": "dec-20251001-143002"
  },

  "regime": {
    "phase": "Bull Market",
    "confidence": 0.82,
    "signals": {
      "trend": "bullish",
      "volatility": "moderate",
      "momentum": "positive"
    }
  },

  "alerts": [
    {
      "id": "alert-001",
      "type": "volatility_alert",
      "severity": "medium",
      "status": "active",
      "message": "Portfolio volatility elevated: 68% annualized",
      "recommendation": "Consider rebalancing to reduce risk",
      "created_at": "2025-10-01T12:15:00Z"
    }
  ],

  "api_health": {
    "status": "operational",
    "components": {
      "backend": {
        "status": "healthy",
        "latency_ms": 45
      },
      "ml_signals": {
        "status": "healthy",
        "last_update": "2025-10-01T14:30:00Z"
      },
      "price_data": {
        "status": "healthy",
        "coverage": 0.95
      }
    }
  },

  "meta": {
    "calculation_time_ms": 187,
    "cache_hit": false,
    "api_version": "v1",
    "correlation_id": "risk-unified-demo-1727790765"
  }
}
```

---

## Implementation Notes

### Backend Architecture

```
/api/risk/unified
  ├─ Orchestrate calls to:
  │  ├─ /api/risk/dashboard (base risk metrics)
  │  ├─ /api/ml/ccs-score (CCS calculation)
  │  ├─ /api/cycle/position (cycle data)
  │  ├─ /api/targets/current (targets & changes)
  │  └─ /api/alerts/active (alerts)
  │
  ├─ Calculate frontend-equivalent scores:
  │  ├─ On-Chain Score (aggregate indicators)
  │  ├─ Blended Score (weighted CCS + On-Chain + Risk)
  │  └─ Cycle-blended CCS (ccsStar)
  │
  ├─ Normalize data structures:
  │  ├─ contradiction_index: always 0..1 (not 0..100)
  │  ├─ cap_daily: always 0..1
  │  └─ timestamps: ISO 8601 format
  │
  └─ Cache response (TTL: 15s)
```

### Key Calculation Logic

**CCS Star (Cycle-blended CCS)**:
```python
def calculate_ccs_star(ccs_score: float, cycle_months: int, cycle_weight: float = 0.3) -> float:
    """
    Blend CCS score with cycle position
    Formula from risk-dashboard.html:2970-2976
    """
    cycle_multiplier = calculate_cycle_multiplier(cycle_months)
    ccs_star = ccs_score * (1 - cycle_weight) + (cycle_multiplier * 100) * cycle_weight
    return max(0, min(100, ccs_star))
```

**Blended Decision Score**:
```python
def calculate_blended_score(ccs_star: float, onchain: float, risk: float) -> float:
    """
    Strategic blended score
    Formula from risk-dashboard.html:3946-3976
    Weights: CCS 50%, On-Chain 30%, Risk (inverted) 20%
    """
    total_score = 0
    total_weight = 0

    if ccs_star is not None:
        total_score += ccs_star * 0.50
        total_weight += 0.50

    if onchain is not None:
        total_score += onchain * 0.30
        total_weight += 0.30

    if risk is not None:
        total_score += (100 - risk) * 0.20
        total_weight += 0.20

    return total_score / total_weight if total_weight > 0 else 50
```

**On-Chain Score**:
```python
async def calculate_onchain_score(holdings: List[dict]) -> float:
    """
    Aggregate on-chain indicators with weighted composite
    Replicates logic from risk-dashboard.html:3827-3905
    """
    indicators = await fetch_all_indicators()  # MVRV, NUPL, Pi Cycle, etc.
    composite = calculate_composite_score_v2(indicators)
    return composite.score
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Components                       │
│  (analytics-unified.html, rebalance.html, risk-dashboard)   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Polling (30s) or Store Subscribe
                      ↓
          ┌───────────────────────────┐
          │  risk-sidebar-full.js     │
          │  (Web Component)          │
          └───────────┬───────────────┘
                      │
                      │ fetchRisk() → utils.js
                      ↓
          ┌───────────────────────────┐
          │  GET /api/risk/unified    │
          │  (NEW ENDPOINT)           │
          └───────────┬───────────────┘
                      │
        ┌─────────────┴─────────────┐
        │  Response Cache (15s TTL) │
        └─────────────┬─────────────┘
                      │
        ┌─────────────┴──────────────────────────────┐
        │                                             │
        ↓                                             ↓
┌───────────────────┐                    ┌─────────────────────┐
│  Internal APIs    │                    │  Calculation Layer  │
│  - risk/dashboard │                    │  - CCS Star         │
│  - ml/ccs-score   │                    │  - Blended Score    │
│  - cycle/position │                    │  - On-Chain Agg     │
│  - targets/current│                    │  - Normalization    │
│  - alerts/active  │                    │                     │
└───────────────────┘                    └─────────────────────┘
```

---

## Migration Strategy

### Phase 1: Implement Endpoint (Backend)

1. Create `api/risk_unified_endpoints.py`:
   ```python
   @router.get("/unified")
   async def get_unified_risk_data(
       source: str = Query("cointracking"),
       min_usd: float = Query(1.0),
       user_id: str = Depends(get_active_user)
   ):
       # Orchestrate calls + calculations
       # Return unified response
   ```

2. Register router in `api/main.py`:
   ```python
   from api.risk_unified_endpoints import router as risk_unified_router
   app.include_router(risk_unified_router)
   ```

3. Add response caching (Redis or in-memory):
   ```python
   @cache(ttl=15, key="risk:unified:{user_id}:{source}")
   async def get_unified_risk_data(...):
       ...
   ```

### Phase 2: Update Frontend (Gradual)

1. Modify `utils.js` fetchRisk():
   ```javascript
   // Try new endpoint first
   try {
     const r = await fetchWithTimeout('/api/risk/unified?source=...', { timeoutMs: 5000 });
     if (r?.ok) return await r.json(); // Already in correct format
   } catch (err) {
     console.warn('Unified endpoint failed, falling back...');
   }

   // Fallback to old API + mapping (Option A)
   return fetchRiskLegacy();
   ```

2. Test with feature flag:
   ```javascript
   const useUnified = localStorage.getItem('USE_UNIFIED_RISK_API') === 'true';
   if (useUnified) {
     return fetchRiskUnified();
   } else {
     return fetchRiskLegacy();
   }
   ```

### Phase 3: Rollout

1. Enable unified endpoint for internal testing
2. Monitor performance (latency, cache hit rate)
3. Enable for all users
4. Remove Option A fallback logic (section hiding)
5. Deprecate old API flow

---

## Testing Checklist

### Backend Tests

- [ ] Endpoint returns 200 with valid data structure
- [ ] All fields match schema (types, ranges)
- [ ] Calculations match frontend formulas:
  - [ ] CCS Star = risk-dashboard.html:2970-2976
  - [ ] Blended Score = risk-dashboard.html:3946-3976
  - [ ] On-Chain Score = risk-dashboard.html:3827-3905
- [ ] Multi-tenant isolation works (user_id filtering)
- [ ] Caching works (15s TTL)
- [ ] Handles missing data gracefully (null/stub values)

### Frontend Tests

- [ ] analytics-unified.html: All sections visible (no hidden)
- [ ] rebalance.html: All sections visible
- [ ] risk-dashboard.html: Parité visuelle avec ancien sidebar
- [ ] Scores match across all pages (same values)
- [ ] Polling works (30s interval)
- [ ] Store integration works (risk-dashboard)
- [ ] Fallback works if endpoint unavailable

### Performance Tests

- [ ] Response time < 200ms (p95)
- [ ] Cache hit rate > 80% (after warmup)
- [ ] No N+1 query issues
- [ ] Memory usage stable

---

## Acceptance Criteria

**Definition of Done**:

1. [OK] Endpoint `/api/risk/unified` returns complete data structure
2. [OK] All 10 sidebar sections visible on analytics-unified.html
3. [OK] All 10 sidebar sections visible on rebalance.html
4. [OK] risk-dashboard.html maintains pixel-perfect parity
5. [OK] No "N/A" or hidden sections due to missing data
6. [OK] Response time < 200ms (p95)
7. [OK] Cache hit rate > 80%
8. [OK] Multi-tenant isolation verified
9. [OK] Fallback to Option A works if endpoint unavailable
10. [OK] Documentation updated (API_REFERENCE.md)

---

## Benefits vs. Option A

| Aspect | Option A (Current) | Option B (Unified) |
|--------|-------------------|-------------------|
| **Data Completeness** | [Error] Partial (only API data) |  Complete (all sections) |
| **UX** | [Warning] Hidden sections | [OK] Full visibility |
| **Performance** | [OK] Fast (1 API call) | [OK] Fast (1 API call + cache) |
| **Consistency** | [Error] Different per page | [OK] Identical everywhere |
| **Maintenance** | [Warning] Conditional logic | [OK] Centralized backend |
| **Frontend Complexity** | [Warning] Section hiding logic | [OK] Simple render |
| **Backend Complexity** | [OK] Simple APIs | [Warning] Orchestration layer |

---

## Next Steps

1. **Prioritize**: Decide if Option B should be implemented now or later
2. **Backend Dev**: Create `api/risk_unified_endpoints.py` (Est: 4-6h)
3. **Frontend Update**: Modify `utils.js` fetchRisk() (Est: 1h)
4. **Testing**: Full regression + performance tests (Est: 2-3h)
5. **Rollout**: Feature flag → internal → production (Est: 1 week)

**Total Estimate**: ~2-3 days dev + 1 week rollout

---

## References

- **Option A Implementation**: [commit 19cb0c0] Hide sections with missing data
- **Risk Dashboard Logic**: `static/risk-dashboard.html:3827-4027`
- **Store Structure**: `static/core/risk-dashboard-store.js:37-94`
- **Web Component**: `static/components/risk-sidebar-full.js`
- **Existing API**: `api/risk_endpoints.py:/dashboard` (lines 476-678)

---

**Status**:  Ready for review and prioritization
**Author**: Claude Code
**Date**: 2025-10-01


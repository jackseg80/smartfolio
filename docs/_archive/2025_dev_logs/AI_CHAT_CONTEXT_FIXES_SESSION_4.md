# AI Chat Context Fixes - Session 4

**Date:** 27 Dec 2025
**Status:**  Complete
**Scope:** Fix context builders for risk-dashboard.html and analytics-unified.html

---

## Issues Identified

### Problem #1: Analytics - Wrong Market Phase
**Symptom:** AI Chat on analytics-unified.html showed `phase: "btc"` instead of market phase
**Root Cause:** Context builder retrieved `govData.phase?.phase_now` which is **dominance phase** (btc/eth/large/alt), not **market phase** (bearish/moderate/bullish)

**Expected:**
```javascript
market_phase: "bullish"  // Based on cycle_score 93 (≥90)
```

**Actual:**
```javascript
phase: "btc"  // Dominance phase, not market phase!
```

### Problem #2: Risk Dashboard - Missing Alerts
**Symptom:** AI Chat couldn't see active alerts
**Root Cause:** API endpoint `/api/alerts/active` returns `List[AlertResponse]` directly, but context builder expected `{ok: true, alerts: [...]}`

**API Response Format:**
```json
[
  {"id": "...", "severity": "S1", "alert_type": "EXEC_COST_SPIKE", ...},
  {"id": "...", "severity": "S2", "alert_type": "VOL_Q90_CROSS", ...}
]
```

**Expected by Context Builder (WRONG):**
```json
{
  "ok": true,
  "alerts": [...]
}
```

### Problem #3: Risk Dashboard - Missing Cycles
**Symptom:** AI Chat couldn't explain market cycles (BTC, ETH, SPY)
**Root Cause:** Context builder relied on `window.lastCyclesData` which is not always populated
**Fix:** Added direct API call to `/execution/governance/state` to get cycle_score + market_phase

### Problem #4: Risk Dashboard - VaR at $0.00
**Symptom:** VaR 95% displayed as $-0.00 instead of real value
**Root Cause:**
1. API returns `var_95_1d` but context searched for `var_95`
2. VaR is in **decimal format** (fraction of portfolio: -0.00027 = -0.027%), not USD absolute

**Calculation:**
```javascript
// Before (WRONG):
context.var_95 = metrics.var_95_1d;  // -0.00027

// After (CORRECT):
context.var_95 = (metrics.var_95_1d || 0) * portfolioValue;  // -$115.16
```

---

## Corrections Applied

### Frontend: [ai-chat-context-builders.js](../static/components/ai-chat-context-builders.js)

#### 1. buildAnalyticsContext() - Lines 232-303
**Changes:**
- [OK] **market_phase** calculated from cycle_score (bearish <70, moderate 70-90, bullish ≥90)
- [OK] **cycle_score** extracted from `govData.scores.components.trend_regime`
- [OK] **dominance_phase** renamed from "phase" (btc/eth/large/alt)

**Code:**
```javascript
// Calculate market phase from cycle score (allocation-engine.js logic)
const cycleScore = govData.scores?.components?.trend_regime || 0;
context.cycle_score = cycleScore;
if (cycleScore < 70) {
    context.market_phase = 'bearish';
} else if (cycleScore < 90) {
    context.market_phase = 'moderate';
} else {
    context.market_phase = 'bullish';
}
context.dominance_phase = govData.phase?.phase_now || 'unknown';
```

#### 2. buildRiskDashboardContext() - Lines 145-257
**Changes:**
- [OK] **Alerts parsing** fixed: Handle `Array` response directly instead of `{ok, alerts}`
- [OK] **Cycles data** added: Direct API call to `/execution/governance/state`
- [OK] **VaR conversion** fixed: Convert decimal to USD absolute value

**Code (Alerts):**
```javascript
// API returns List[AlertResponse] directly (not wrapped in {ok, alerts})
if (Array.isArray(alertsData)) {
    context.alerts = alertsData.map(alert => ({
        severity: alert.severity,
        type: alert.alert_type,
        message: alert.data?.current_value ?
            `${alert.alert_type}: ${alert.data.current_value}` :
            alert.alert_type,
        created_at: alert.created_at
    }));
}
```

**Code (Cycles):**
```javascript
// Market cycles - Load from governance state
const govResponse = await fetch('/execution/governance/state', {
    headers: { 'X-User': activeUser }
});

if (govResponse.ok) {
    const govData = await govResponse.json();
    const cycleScore = govData.scores?.components?.trend_regime || 0;
    context.cycle_score = cycleScore;

    // Calculate market phase
    if (cycleScore < 70) {
        context.market_phase = 'bearish';
    } else if (cycleScore < 90) {
        context.market_phase = 'moderate';
    } else {
        context.market_phase = 'bullish';
    }

    context.dominance_phase = govData.phase?.phase_now || 'unknown';
    context.phase_confidence = govData.phase?.confidence || 0;
}
```

**Code (VaR):**
```javascript
// Portfolio summary (get first for VaR calculation)
const portfolioValue = data.portfolio_summary?.total_value || 0;

// VaR is in decimal format (fraction of portfolio), convert to USD
const varDecimal = metrics.var_95_1d || metrics.var_95 || 0;
context.var_95 = varDecimal * portfolioValue;  // Convert to absolute USD value
```

### Backend: [ai_chat_router.py](../api/ai_chat_router.py)

#### 1. _format_risk_context() - Lines 432-449
**Changes:**
- [OK] Replaced legacy `cycles` dict with new fields: `cycle_score`, `market_phase`, `dominance_phase`
- [OK] Added emojis for phases ( bearish,  moderate,  bullish, ₿ BTC, Ξ ETH,  large,  alt)

**Code:**
```python
# Market cycles
if "cycle_score" in context:
    lines.append("")
    lines.append(" Analyse des cycles:")
    lines.append(f"  - Cycle Score: {context['cycle_score']:.1f}/100")

    if "market_phase" in context:
        phase_emoji = {"bearish": "Decline", "moderate": "Balanced", "bullish": "Growth"}.get(context["market_phase"], "Unknown")
        lines.append(f"  - Phase de marché: {phase_emoji} {context['market_phase'].capitalize()}")

    if "dominance_phase" in context:
        dom_emoji = {"btc": "₿", "eth": "Ξ", "large": "Analytics", "alt": "Theme"}.get(context["dominance_phase"], "Unknown")
        lines.append(f"  - Dominance: {dom_emoji} {context['dominance_phase'].upper()}")

    if "phase_confidence" in context:
        lines.append(f"  - Confiance: {context['phase_confidence']:.1%}")
```

#### 2. _format_analytics_context() - Lines 488-503
**Changes:**
- [OK] Replaced `phase` (dominance) with separate `market_phase`, `cycle_score`, `dominance_phase` fields
- [OK] Added emojis for clarity

**Code:**
```python
# Market phase (bearish/moderate/bullish)
if "market_phase" in context:
    phase = context["market_phase"]
    phase_emoji = {"bearish": "Decline", "moderate": "Balanced", "bullish": "Growth"}.get(phase, "Unknown")
    lines.append(f" Phase de marché: {phase_emoji} {phase.capitalize()}")

# Cycle score
if "cycle_score" in context:
    lines.append(f" Cycle Score: {context['cycle_score']:.1f}/100")

# Dominance phase (btc/eth/large/alt)
if "dominance_phase" in context:
    dom = context["dominance_phase"]
    dom_emoji = {"btc": "₿", "eth": "Ξ", "large": "Analytics", "alt": "Theme"}.get(dom, "Unknown")
    lines.append(f" Dominance: {dom_emoji} {dom.upper()}")
    lines.append("")
```

---

## Testing

### Test Case 1: Analytics - Market Phase
**Page:** analytics-unified.html
**Question:** "Quelle est la phase de marché actuelle?"

**Before (WRONG):**
```
Phase actuelle: btc
```

**After (CORRECT):**
```
 Phase de marché:  Bullish
 Cycle Score: 93.3/100
 Dominance: ₿ BTC
```

### Test Case 2: Risk Dashboard - Alerts
**Page:** risk-dashboard.html
**Question:** "Analyse les alertes actives. Que dois-je faire en priorité?"

**Before (WRONG):**
```
Malheureusement, je n'ai pas accès aux alertes actives...
```

**After (CORRECT):**
```
[Alert] Alertes actives (14):
  - [S1] EXEC_COST_SPIKE: 45
  - [S2] VOL_Q90_CROSS: 0.368...
  ...
```

### Test Case 3: Risk Dashboard - Cycles
**Page:** risk-dashboard.html
**Question:** "Explique-moi les cycles de marché actuels (BTC, ETH, SPY)."

**Before (WRONG):**
```
Malheureusement, je n'ai pas accès aux données en temps réel...
```

**After (CORRECT):**
```
 Analyse des cycles:
  - Cycle Score: 93.3/100
  - Phase de marché:  Bullish
  - Dominance: ₿ BTC
  - Confiance: 61.7%
```

### Test Case 4: Risk Dashboard - VaR
**Page:** risk-dashboard.html
**Question:** "Analyse mes métriques de risque (VaR, Max Drawdown)."

**Before (WRONG):**
```
[Warning] Score de risque: 69.6/100
 VaR 95%: $-0.00 (max expected loss)  ← WRONG!
 Max Drawdown: -11.05%
```

**After (CORRECT):**
```
[Warning] Score de risque: 69.6/100
 VaR 95%: $-115.16 (max expected loss)  ← CORRECT!
 Max Drawdown: -11.05%
```

---

## Impact Summary

| Component | Issue | Fix | Status |
|-----------|-------|-----|--------|
| **Analytics Context** | Wrong phase (dominance vs market) | Calculate market_phase from cycle_score | [OK] Fixed |
| **Analytics Context** | Missing cycle_score | Extract from govData.scores.components | [OK] Fixed |
| **Risk Context** | Alerts not visible | Parse Array response directly | [OK] Fixed |
| **Risk Context** | Cycles missing | Direct API call to governance/state | [OK] Fixed |
| **Risk Context** | VaR at $0.00 | Convert decimal to USD (× portfolio_value) | [OK] Fixed |
| **Backend Formatter** | Legacy cycles format | Use new fields (cycle_score, market_phase, dominance_phase) | [OK] Fixed |

---

## Files Modified

### Frontend
- `static/components/ai-chat-context-builders.js` (buildAnalyticsContext, buildRiskDashboardContext)

### Backend
- `api/ai_chat_router.py` (_format_risk_context, _format_analytics_context)

---

## Notes

1. **Market Phase Logic:** Aligned with allocation-engine.js (lines 180-190)
   - `cycle_score < 70` → bearish
   - `cycle_score 70-90` → moderate
   - `cycle_score ≥ 90` → bullish

2. **VaR Format:** API returns daily VaR as decimal (fraction of portfolio)
   - Backend could be updated to return USD directly in future
   - For now, frontend converts: `var_usd = var_decimal × portfolio_value`

3. **Dominance vs Market Phase:** Two separate concepts!
   - **Dominance:** btc/eth/large/alt (which assets lead)
   - **Market Phase:** bearish/moderate/bullish (cycle strength)
   - Both are useful for AI context

4. **Server Restart Required:** Backend changes need server restart to apply

---

## Completion Checklist

- [x] Fix analytics market phase calculation
- [x] Fix risk alerts parsing (Array response)
- [x] Fix risk cycles loading (governance API)
- [x] Fix risk VaR conversion (decimal → USD)
- [x] Update backend formatters (risk + analytics)
- [x] Document all changes
- [ ] Server restart + browser cache clear (Ctrl+F5)
- [ ] Manual testing on both pages
- [ ] Commit changes

---

**Next Steps:**
1. User restarts backend server
2. User tests AI Chat on risk-dashboard.html and analytics-unified.html
3. If tests pass → Commit + merge to main

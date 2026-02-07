# Execution System

> Governance engine, plan lifecycle, and order execution pipeline.

## Architecture

```
ML Signals → GovernanceEngine → PolicyEngine → Plan Lifecycle → ExecutionEngine → Orders
                  ↕                                                    ↕
            FreezePolicy                                       SafetyValidator
            HysteresisManager                                  ExchangeAdapter
            PhaseEngine                                        OrderManager
```

**Key files:**
- `services/execution/governance.py` — Main orchestrator (state machine, plans, freezes)
- `services/execution/policy_engine.py` — Derives execution policy from ML signals
- `services/execution/safety_validator.py` — Order validation rules
- `services/execution/freeze_policy.py` — 3-tier freeze semantics
- `services/execution/execution_engine.py` — Order execution (sequential/parallel)
- `services/execution/order_manager.py` — Order creation and status tracking
- `services/execution/signals.py` — Centralized ML signal aggregation
- `services/execution/hysteresis.py` — Anti-oscillation guards
- `services/execution/phase_engine.py` — BTC→ETH→Large→Alt rotation detection

---

## Governance State Machine

### Plan Lifecycle

```
DRAFT → REVIEWED → APPROVED → ACTIVE → EXECUTED
              ↘ CANCELLED (anytime)
```

- **ETag-based optimistic concurrency** — Prevents concurrent plan modifications
- **Cooldown enforcement** — Configurable delay between plan publications (1–168h)
- **Single-writer pattern** — Only one active plan at a time

### Governance Modes

| Mode | Description |
|------|-------------|
| `manual` | User controls all decisions |
| `ai_assisted` | AI suggests, user approves |
| `full_ai` | AI decides within policy bounds |

---

## Freeze System

3 freeze types with operation whitelists:

| Freeze Type | Purchases | Sell to Stables | Rotations | Hedging | Rebalancing |
|-------------|-----------|-----------------|-----------|---------|-------------|
| `FULL_FREEZE` | No | No | No | No | No |
| `S3_ALERT_FREEZE` | No | Yes | Yes | Yes | No |
| `ERROR_FREEZE` | No | Yes | No | No | No |

---

## Policy Engine

Derives execution policy from ML signals:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `mode` | Freeze/Slow/Normal/Aggressive | Execution mode |
| `cap_daily` | 1–50% | Max daily execution volume |
| `ramp_hours` | 1–72h | Gradual execution ramp-up |
| `min_trade` | USD | Minimum trade size |
| `slippage_limit_bps` | 1–500 | Slippage tolerance |
| `no_trade_threshold_pct` | 0–10% | No-trade dead zone |

---

## Safety Validator

3 levels: **STRICT** (rejects dubious), **MODERATE** (warns), **PERMISSIVE** (minimal checks).

Validates: daily volume limits, single order size, testnet mode, safety scoring (0–100).

---

## API Endpoints

### Governance (`/execution/governance`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/state` | GET | Current governance state (scores, freeze, phase) |
| `/signals` | GET | Current ML signals |
| `/propose` | POST | Create new allocation plan |
| `/review/{plan_id}` | POST | Review plan |
| `/approve/{resource_id}` | POST | Approve plan |
| `/activate/{plan_id}` | POST | Activate plan |
| `/execute/{plan_id}` | POST | Execute active plan |
| `/cancel/{plan_id}` | POST | Cancel plan |
| `/freeze` | POST | Freeze system with TTL |
| `/unfreeze` | POST | Remove freeze |
| `/mode` | POST | Set governance mode |
| `/cooldown-status` | GET | Check plan publication cooldown |
| `/validate-allocation` | POST | Validate allocation changes |

### Execution (`/execution`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/execute-plan` | POST | Execute rebalancing plan (supports `dry_run`) |
| `/cancel/{plan_id}` | POST | Cancel execution |
| `/exchanges` | GET | List available exchanges |
| `/exchanges/connect` | POST | Connect/configure exchanges |
| `/validate-plan` | POST | Validate plan before execution |

---

## Related Docs

- [GOVERNANCE_FIXES_OCT_2025.md](GOVERNANCE_FIXES_OCT_2025.md) — Governance freeze semantics
- [STOP_LOSS_SYSTEM.md](STOP_LOSS_SYSTEM.md) — Stop loss integration
- [DECISION_INDEX_V2.md](DECISION_INDEX_V2.md) — Decision scoring that feeds governance

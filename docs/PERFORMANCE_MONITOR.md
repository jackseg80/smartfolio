# Performance Monitor

> Unified performance tracking, strategy comparison, and system health monitoring.

## Architecture

**Key files:**
- `services/analytics/performance_tracker.py` — Core performance analysis engine
- `api/performance_endpoints.py` — API endpoints
- `static/performance-monitor-unified.html` — Unified monitoring UI

---

## Core Metrics

### PerformanceMetrics

| Category | Metrics |
|----------|---------|
| **Returns** | `total_return_pct`, `annualized_return_pct` |
| **Risk** | `volatility_pct`, `max_drawdown_pct`, `sharpe_ratio` |
| **Rebalancing** | `rebalancing_alpha_pct` (vs buy-and-hold), `avg_drift_before/after_rebal` |
| **Execution** | `avg_execution_slippage_bps`, `avg_fee_rate_pct`, `execution_success_rate` |
| **Frequency** | `rebalancing_frequency_days`, `total_rebalance_sessions` |

### StrategyPerformance

Per-strategy tracking with:
- Avg target allocation per asset
- CCS score range
- Comparative metrics: `vs_benchmark_pct`, `vs_manual_rebal_pct`
- Statistical confidence: `sample_size`, `confidence_interval_95`

---

## System Health

### Redis Cache Stats
- Total keys, memory usage (current + peak), hit rate

### ML Models Stats
- Model types: regime, volatility, correlation, config
- Total count and size per type

---

## API Endpoints

**Prefix:** `/api/performance`

Performance tracking and system health endpoints for the monitoring dashboard.

---

## Related Docs

- [monitoring.md](monitoring.md) — System monitoring overview
- [CACHE_TTL_OPTIMIZATION.md](CACHE_TTL_OPTIMIZATION.md) — Cache TTL configuration

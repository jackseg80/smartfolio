# Portfolio Optimization Advanced

> Multi-algorithm portfolio optimization with crypto-specific constraints.

## Architecture

**Key files:**
- `services/portfolio_optimization.py` — Core optimizer (11 algorithms)
- `api/portfolio_optimization_endpoints.py` — API endpoints
- `static/portfolio-optimization-advanced.html` — Advanced optimization UI

---

## Optimization Algorithms (11)

| Algorithm | Objective | Description |
|-----------|-----------|-------------|
| `MAX_SHARPE` | Maximize Sharpe ratio | Best risk-adjusted returns |
| `MIN_VARIANCE` | Minimize volatility | Lowest portfolio risk |
| `MAX_RETURN` | Maximize expected returns | Aggressive growth |
| `RISK_PARITY` | Equal risk contribution | Balanced risk across assets |
| `RISK_BUDGETING` | Custom risk budgets | Assign risk per asset |
| `MEAN_REVERSION` | Exploit mean reversion | Contrarian strategies |
| `MULTI_PERIOD` | Multi-period optimization | Across rebalance periods |
| `BLACK_LITTERMAN` | Market views integration | Bayesian approach with views |
| `MAX_DIVERSIFICATION` | Maximize diversification ratio | Spread across correlations |
| `CVAR_OPTIMIZATION` | Minimize CVaR | Tail-risk optimization |
| `EFFICIENT_FRONTIER` | Generate frontier | Plot risk/return tradeoffs |

---

## Constraints

| Constraint | Default | Description |
|------------|---------|-------------|
| Min/max weight bounds | 0.0–1.0 | Per-asset allocation limits |
| Sector concentration | Max 50% | Max allocation per crypto sector |
| Diversification minimum | 0.3 | Minimum diversification ratio |
| Correlation ceiling | 0.7 | Max correlation exposure |
| Transaction costs | maker/taker fees | Fee-aware optimization |

### Crypto Sector Mapping (11 sectors)

BTC, ETH, Stablecoins, SOL, L1/L0, L2/Scaling, DeFi, AI/Data, Gaming/NFT, Memecoins, Others

---

## API

### `POST /api/portfolio/optimization`

**Parameters:**
- `objective` — Algorithm to use (see table above)
- `lookback_days` — Historical data window (default: 365)
- `expected_return_method` — Return estimation method
- `conservative` — Conservative mode flag
- `custom_constraints` — Override default constraints

**Response:** `OptimizationResult`
- Weights per asset
- Expected return & volatility
- Sharpe ratio & max drawdown
- Risk contributions per asset
- Sector exposure breakdown
- Rebalancing trades

---

## Related Docs

- [PORTFOLIO_OPTIMIZATION_GUIDE.md](PORTFOLIO_OPTIMIZATION_GUIDE.md) — General optimization guide
- [ALLOCATION_ENGINE_V2.md](ALLOCATION_ENGINE_V2.md) — Allocation algorithm (hierarchical 3-level)
- [dynamic-allocation-system.md](dynamic-allocation-system.md) — Dynamic target system

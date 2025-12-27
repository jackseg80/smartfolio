# SmartFolio

Intelligent cross-asset wealth management platform (Crypto, Stock Market, Banking) with AI, advanced ML, and unified risk management. Modular architecture built around 10+ specialized pages optimized for real-time decision making.

Designed to simplify the management of large crypto portfolios (>200 assets) via a smart 11-group taxonomy, and currently specialized for CoinTracking (API & CSV) data sources.

## 🎯 Main Features

The platform is built around **3 Specialized Modules** powered by a shared AI Core.

### 🚀 Crypto (Core Module)

- **Decision Engine**: Automated governance with "Freeze Semantics" to prevent panic selling.
- **Phase Engine**: Proactive detection of market regimes (Bitcoin Season vs Altseason) with auto-tilts.
- **Allocation Engine V2**: Top-down hierarchical rebalancing (Macro → Sectors → Coins).
- **Smart Execution**: Dynamic thresholds and "Incumbency Protection" to minimize churn.
- **Smart Taxonomy**: Automatically groups hundreds of tokens into 11 Canonical Groups (BTC, ETH, SOL, AI, DeFi...) to drastically reduce cognitive load for large wallets.
- **CoinTracking Native**: Deep integration with CoinTracking (API & CSV) to handle complex transaction histories and real-time balances accurately.

### 📈 Stock Market (Saxo Module)

- **Market Opportunities**: AI Scanner that detects portfolio gaps and suggests Stocks/ETFs.
- **Intelligent Stop Loss**: 6 adaptive methods (Trailing, Volatility-based) to protect gains.
- **Risk Analytics**: Specific beta and correlation analysis against S&P 500.

### 🏛️ Wealth (Patrimoine & Banking)

- **P&L Today**: Real-time performance tracking with "Anchor Points" (Midnight/Session).
- **Unified View**: Cross-asset aggregation (Crypto + Stocks + Bank) in your reference currency.
- **Structure Analysis**: Monitoring of liquidity ratios (Stable/Cash vs Risky Assets).

### 🧠 Shared Intelligence (Cross-Module)

- **AI Chat Assistant**: Global AI assistant with context awareness (Groq free / Claude premium)
  - 🎯 Context-aware: Automatically sees current page data (portfolio, risk, opportunities)
  - 📚 Knowledge base: Dynamically synced with documentation (Decision Index, Risk Score, etc.)
  - ✨ Floating button (Ctrl+K) available on all pages
  - 🔄 Auto-refresh: Knowledge base reloads from .md files (5-min cache TTL)
- **Advanced ML**: LSTM & Transformers for volatility and trend prediction.
- **Risk Score V2**: Unified "Robustness Score" (0-100) across all asset classes.
- **ML Sentiment**: Proprietary sentiment score (0-100) - NOT Fear & Greed Index (alternative.me). Calcul: `50 + (sentiment_ml × 50)` où sentiment ∈ [-1, 1].
- **Multi-Tenant**: Complete isolation of data and configurations per user.

## Quick Start

### Prerequisites

- Python 3.10+
- pip, virtualenv
- (Optional) Redis for advanced caching and real-time streaming

### Installation

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env with your API keys (CoinGecko, CoinTracking, FRED)
```

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

**Playwright (optional, for crypto-toolbox scraping):**

```bash
pip install playwright
playwright install chromium
```

### Launch

**Windows:**

```powershell
.\\.venv\\Scripts\\Activate
.\\start_dev.ps1
# With scheduler (P&L snapshots, OHLCV updates): .\\start_dev.ps1 -EnableScheduler
```

**Linux/macOS:**

```bash
source .venv/bin/activate
./start_dev.sh
# With scheduler: ./start_dev.sh --enable-scheduler
```

### Production Deployment (Docker)

This is the recommended method for running the application in a stable, production-like environment.

1. **Prerequisites:**
    - Docker and Docker Compose installed.
    - An `.env` file created from `.env.example` with the necessary API keys.

2. **Automated Deployment (Recommended):**
    The `deploy.sh` script automates pulling the latest code, rebuilding the Docker image, and launching the services.

    ```bash
    ./deploy.sh
    ```

    To restart without rebuilding the image:

    ```bash
    ./deploy.sh --skip-build
    ```

3. **Manual Launch:**
    You can also use `docker-compose` commands directly:

    ```bash
    # Build and start services in the background
    docker-compose up -d --build

    # Stop services
    docker-compose down

    # View logs
    docker-compose logs -f
    ```

### Web Access

- **Settings**: <http://localhost:8080/static/settings.html> (initial configuration)
- **Dashboard**: <http://localhost:8080/static/dashboard.html>
- **API Docs**: <http://localhost:8080/docs>

## Main Pages

| Page | Description | URL |
|------|-------------|-----|
| **Dashboard** | Global portfolio view + P&L Today | `/static/dashboard.html` |
| **Analytics** | Real-time ML + Decision Index | `/static/analytics-unified.html` |
| **Risk** | Risk management + Governance + Alerts | `/static/risk-dashboard.html` |
| **Cycle Analysis** | Bitcoin cycle analysis + historical charts | `/static/cycle-analysis.html` |
| **Rebalance** | Dynamic rebalancing plans | `/static/rebalance.html` |
| **Execution** | Real-time execution with validation | `/static/execution.html` |
| **Simulations** | Complete pipeline simulator | `/static/simulations.html` |
| **Wealth Dashboard** | Unified wealth (liquidities, assets, liabilities) | `/static/wealth-dashboard.html` |
| **Monitoring** | System KPIs + Alerts History | `/static/monitoring.html` |
| **Admin Dashboard** | User management, logs, cache, ML models (RBAC) | `/static/admin-dashboard.html` |
| **Saxo Dashboard** | Stock Market (stocks, ETFs, funds) with intelligent stop-loss | `/static/saxo-dashboard.html` |

## Architecture

### Backend (FastAPI)

```
api/
├── main.py                          # Main app + routers
├── deps.py                          # Dependency injection (multi-tenant)
├── execution/                       # Decision Engine + Governance
├── *_endpoints.py                   # 30+ modular routers
services/
├── balance_service.py               # Multi-source data resolution
├── execution/governance.py          # Decision Engine + Freeze semantics
├── ml/orchestrator.py              # ML orchestration
├── risk_scoring.py                  # Central Risk Score (dual system)
├── portfolio.py                     # P&L tracking
```

### Frontend (Vanilla JS + ES6 Modules)

```
static/
├── *.html                           # Main pages
├── core/
│   ├── allocation-engine.js         # Topdown hierarchical allocation
│   └── unified-insights-v2.js       # Phase Engine
├── components/
│   ├── nav.js                       # Unified navigation
│   ├── decision-index-panel.js      # Decision Index UI
│   └── flyout-panel.js              # Reusable Risk Sidebar
├── global-config.js                 # Centralized frontend config
```

### Data

```
data/
└── users/{user_id}/
    ├── cointracking/data/           # Crypto CSV (auto versioning)
    ├── saxobank/data/               # Stock market CSV
    ├── config/config.json           # User configuration
    └── config/sources.json          # Active modules
```

## 🔒 Security

- ✅ **Secrets management**: `.env` template, pre-commit hooks (detect-secrets + gitleaks)
- ✅ **Secure frontend**: 464 console.log → debugLogger, ESLint (no-console, no-eval)
- ✅ **HTTP headers**: CSP, X-Content-Type-Options, X-Frame-Options, rate limiting
- ✅ **Automated tests**: header & security validation

📖 Complete details: [SECURITY.md](docs/SECURITY.md)

## Documentation

### Essentials

- **[CLAUDE.md](CLAUDE.md)** - Guide for AI agents (critical rules, patterns, quick checks)
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detailed architecture
- **[Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md)** - Production deployment guide
- **[API Reference](docs/API_REFERENCE.md)** - Endpoints and schemas
- **[User Guide](docs/user-guide.md)** - Complete user guide

### Features & Systems

- **AI Chat Assistant**: [AI_CHAT_GLOBAL.md](docs/AI_CHAT_GLOBAL.md) - Global AI assistant (Groq/Claude), context-aware, dynamic knowledge base
- **Allocation**: [ALLOCATION_ENGINE_V2.md](docs/ALLOCATION_ENGINE_V2.md) - Topdown hierarchical, floors, incumbency
- **Decision Index**: [DECISION_INDEX_V2.md](docs/DECISION_INDEX_V2.md) - Dual scoring (DI vs Regime)
- **Risk Management**: [RISK_SEMANTICS.md](docs/RISK_SEMANTICS.md), [RISK_SCORE_V2_IMPLEMENTATION.md](docs/RISK_SCORE_V2_IMPLEMENTATION.md)
- **Governance**: [GOVERNANCE_FIXES_OCT_2025.md](docs/GOVERNANCE_FIXES_OCT_2025.md) - Freeze semantics, TTL vs Cooldown
- **Phase Engine**: [PHASE_ENGINE.md](docs/PHASE_ENGINE.md) - Market phase detection
- **Simulator**: [SIMULATION_ENGINE.md](docs/SIMULATION_ENGINE.md) - Complete pipeline
- **Sources System**: [SOURCES_SYSTEM.md](docs/SOURCES_SYSTEM.md) - Unified multi-source
- **Market Opportunities**: [MARKET_OPPORTUNITIES_SYSTEM.md](docs/MARKET_OPPORTUNITIES_SYSTEM.md) - Scoring 3-pillars, gap detection
- **Intelligent Stop Loss**: [STOP_LOSS_SYSTEM.md](docs/STOP_LOSS_SYSTEM.md) - 6 adaptive methods (Trailing Stop NEW Oct 2025)
- **P&L Today**: [P&L Today](docs/PNL_TODAY.md) - Real-time tracking
- **Redis**: [REDIS_SETUP.md](docs/REDIS_SETUP.md) - Cache & streaming
- **Logging**: [LOGGING.md](docs/LOGGING.md) - Rotating logs (5MB x3, AI-optimized)

### Development

- **[Developer Guide](docs/developer.md)** - Setup, tests, workflow
- **[Testing Guide](docs/TESTING_GUIDE.md)** - Unit/Integration/E2E tests
- **[Runbooks](docs/runbooks.md)** - Operational procedures
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues resolution
- **[Contributing](CONTRIBUTING.md)** - Contribution guidelines

### Complete Index

📖 **[Documentation Index](docs/index.md)** - Complete list of available docs

## Configuration

### Multi-Users

6 configured users: `demo`, `jack`, `donato`, `elda`, `roberto`, `clea`

- **Complete isolation**: separate data, config, API keys
- **Dynamic selector**: navigation bar (independent from Admin menu)
- **Dynamic sources**: auto display of CSV + API according to config

### Data Sources

1. **Local CSV**: upload via Settings → Sources (automatic versioning)
2. **CoinTracking API**: if keys configured (real-time)
3. **Saxo API**: import stock market positions
4. **Banks**: manual bank accounts

### Recommended API Keys

```env
# .env
COINGECKO_API_KEY=your_key_here        # Crypto prices (3 min cache)
COINTRACKING_API_KEY=your_key_here     # Real-time balances
FRED_API_KEY=your_key_here             # Macro data
REDIS_URL=redis://localhost:6379/0     # Advanced cache (optional)
```

## Main Endpoints

```bash
# Health & Config
GET  /healthz                                    # Application status
GET  /api/config                                 # Frontend configuration

# Portfolio
GET  /balances/current?source=cointracking       # Current balances
GET  /portfolio/metrics?user_id=demo             # Metrics + P&L Today
POST /portfolio/snapshot                         # Create P&L snapshot

# ML & Analytics
GET  /api/ml/sentiment/symbol/BTC                # ML Sentiment
GET  /api/ml/cycle_score                         # Cycle Score
GET  /api/ml/onchain_score                       # On-Chain Score

# Risk
GET  /api/risk/dashboard                         # Complete risk dashboard
GET  /api/risk/bourse/dashboard                  # Stock market risk (Saxo)

# Governance & Execution
GET  /execution/governance/state                 # Governance state
POST /execution/governance/approve               # Approve plan
GET  /execution/monitoring/live                  # Real-time monitoring

# Sources
GET  /api/sources/list                           # Available sources
POST /api/sources/upload                         # Upload file
GET  /api/sources/test                           # Test source

# AI Chat Assistant
POST /api/ai/chat                                # Chat with AI (context-aware)
GET  /api/ai/providers                           # List configured providers
GET  /api/ai/quick-questions/{page}              # Get quick questions for page
POST /api/ai/refresh-knowledge                   # Force reload docs from .md files
GET  /api/ai/knowledge-stats                     # Cache statistics
```

📖 Complete API: <http://localhost:8080/docs> (Swagger UI)

## Tests

```bash
# Activate environment
.venv\\Scripts\\Activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# E2E tests (requires running server)
pytest tests/e2e -v

# Coverage
pytest --cov=services --cov=api --cov-report=html
```

## Critical Rules (Developers)

### 1. Multi-Tenant REQUIRED

```python
# Backend: ALWAYS use dependency injection
from api.deps import get_active_user

@router.get("/endpoint")
async def endpoint(user: str = Depends(get_active_user)):
    pass
```

```javascript
// Frontend: ALWAYS use window.loadBalanceData()
const balanceResult = await window.loadBalanceData(true);
```

### 2. Risk Score = Positive (0-100)

- **Convention**: Higher = more robust
- **❌ FORBIDDEN**: Never invert with `100 - scoreRisk`

### 3. Decision Index vs Regime

- **Decision Index**: Technical allocation quality (65/45 fixed)
- **Regime Score**: Market state (0-100 variable)
- **Phase**: Based ONLY on Cycle Score (<70=bearish, 70-90=moderate, ≥90=bullish)

📖 Details: [CLAUDE.md](CLAUDE.md)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Recommended workflow:**

1. Fork the project
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

## License

This project is a starter/template for personal or educational use.

## Support

- **Documentation**: [docs/index.md](docs/index.md)
- **Issues**: For bugs and feature requests
- **Troubleshooting**: [docs/troubleshooting.md](docs/troubleshooting.md)

---

**Status**: ✅ Production Stable (Dec 2025)
**Version**: 3.0
**Stack**: Python 3.10+ • FastAPI • Vanilla JS (ES6) • Redis (optional)

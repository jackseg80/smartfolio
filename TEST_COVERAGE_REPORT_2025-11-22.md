# 📊 Test Coverage Report - SmartFolio
## Date: 22 Novembre 2025

> **Coverage Measurement:** First Baseline Audit
> **Tool:** pytest-cov 7.0.0
> **Scope:** api/ + services/ (35,981 statements)

---

## 📊 Executive Summary

**Coverage Global: 37% (13,145 / 35,981 lignes testées)**

### Résultats Globaux

```
Total Statements:  35,981
Covered:          13,145  (37%)
Missing:          22,836  (63%)

Tests Exécutés:    775 passed
Tests Échoués:      99 failed
Tests Skipped:      36 skipped
Erreurs:            28 errors
```

**Verdict:** 🟡 **COVERAGE MOYEN** - Baseline établie, amélioration nécessaire

### Objectifs Coverage

| Phase | Target | Actuel | Écart |
|-------|--------|--------|-------|
| **Baseline** | - | 37% | - |
| **Q1 2026** | 50% | 37% | -13% ⬆️ +35% requis |
| **Q2 2026** | 80% | 37% | -43% ⬆️ +116% requis |
| **Production** | 80%+ | 37% | -43% |

---

## 1. 🔴 Fichiers Critiques - Faible Coverage (<20%)

### Priority 1: CRITIQUE (0-10% coverage)

#### 1.1 services/risk/var_calculator.py - 8% ⚠️⚠️⚠️
```
Statements: 254
Coverage:   8% (21 lignes testées, 233 non testées)
Missing:    30-31, 48-158, 168-252, 261-287, 296-319, 324-337, 347-375, 385-422, 432-440, 461-521, 534-536
```

**Fonction Critique:** VaR/CVaR calculations (gestion risque financier)

**Impact Non Testé:**
- ❌ Calcul VaR parametric (méthode principale)
- ❌ Calcul VaR historical
- ❌ Calcul VaR Monte Carlo
- ❌ CVaR (Conditional Value at Risk)

**Risque:** 🔴 **TRÈS ÉLEVÉ** - Calculs financiers critiques non validés

#### 1.2 services/execution/liquidation_manager.py - 0% ⚠️⚠️⚠️
```
Statements: 63
Coverage:   0% (AUCUNE ligne testée)
Missing:    Toutes les lignes
```

**Fonction Critique:** Gestion liquidation assets (exécution trades)

**Impact Non Testé:**
- ❌ Logique liquidation complète
- ❌ Priorités liquidation
- ❌ Gestion erreurs liquidation

**Risque:** 🔴 **TRÈS ÉLEVÉ** - Exécution financière non testée

#### 1.3 api/execution/validation_endpoints.py - 0% ⚠️⚠️
```
Statements: 123
Coverage:   0%
Missing:    Toutes les lignes
```

**Fonction Critique:** Validation plans avant exécution

**Risque:** 🔴 **ÉLEVÉ** - Validation exécution non testée

#### 1.4 services/execution/exchange_adapter.py - 8%
```
Statements: 197
Coverage:   8% (16 lignes testées)
Missing:    38-186, 191-203, 208-213, 218-228, 233-240
```

**Fonction Critique:** Adaptation exchanges (Binance, Kraken, etc.)

**Risque:** 🔴 **ÉLEVÉ** - Communication exchanges non testée

---

### Priority 2: HIGH (10-20% coverage)

#### 2.1 services/portfolio.py - 13%
```
Statements: 407
Coverage:   13% (52 lignes testées, 355 non testées)
Missing:    Functions critiques non testées
```

**Fonctions Critiques Non Testées:**
- ❌ `calculate_performance_metrics()` - Calcul P&L
- ❌ `save_portfolio_snapshot()` - Sauvegarde historique
- ❌ `get_portfolio_history()` - Récupération historique

**Risque:** 🔴 **TRÈS ÉLEVÉ** - P&L tracking non validé

#### 2.2 api/risk_endpoints.py - 15%
```
Statements: 577
Coverage:   15%
```

**Impact:** Endpoints risk dashboard principaux non testés

#### 2.3 services/execution/governance.py - 11%
```
Statements: 1,008
Coverage:   11% (113 lignes testées)
```

**Impact:** Decision engine faiblement testé

---

## 2. 🟡 Fichiers Critiques - Coverage Moyen (20-50%)

### 2.1 api/main.py - 27%
```
Statements: 531
Coverage:   27% (144 lignes testées)
```

**Fonctions Testées:**
- ✅ Health checks basiques
- ✅ Quelques endpoints simples

**Fonctions Non Testées:**
- ❌ Middlewares security
- ❌ Error handlers
- ❌ Startup logic complexe

**Recommandation:** ⚠️ Augmenter à 50%+ (coeur de l'API)

### 2.2 services/risk_management.py - 46%
```
Statements: 883
Coverage:   46% (408 lignes testées, 475 non testées)
```

**Analyse:**
- ✅ Certaines méthodes VaR testées
- ❌ Stress testing non testé (729-755, 1055-1058)
- ❌ Performance attribution non testé (1235-1292)
- ❌ Backtesting non testé (1610-1754)

**Recommandation:** ⚠️ Tester stress scenarios + backtesting

### 2.3 services/pricing.py - 31%
```
Statements: 181
Coverage:   31%
```

**Impact:** Pricing (httpx migration) partiellement testé

**Recommandation:** ⚠️ Tester nouvelles fonctions httpx

---

## 3. ✅ Fichiers Bien Testés (>80% coverage)

### Excellents Exemples

#### 3.1 services/risk_scoring.py - 99% ✅✅
```
Statements: 119
Coverage:   99% (1 seule ligne non testée)
```

**Best Practice:** Exemple de coverage exemplaire

#### 3.2 services/stop_loss/trailing_stop_calculator.py - 89% ✅
```
Statements: 82
Coverage:   89%
```

**Bonne couverture** des calculs stop loss

#### 3.3 services/smart_classification.py - 87% ✅
```
Statements: 166
Coverage:   87%
```

**Bonne couverture** de la classification

#### 3.4 services/balance_service.py - 84% ✅
```
Statements: 111
Coverage:   84%
```

**Bonne couverture** du service critique balances

---

## 4. 📊 Coverage par Module

### API Endpoints

| Fichier | Statements | Coverage | Status |
|---------|-----------|----------|--------|
| `api/main.py` | 531 | 27% | 🟡 MEDIUM |
| `api/risk_endpoints.py` | 577 | 15% | 🔴 LOW |
| `api/unified_ml_endpoints.py` | 863 | 24% | 🟡 MEDIUM |
| `api/rebalancing_strategy_router.py` | 92 | 58% | 🟡 MEDIUM |
| `api/execution/validation_endpoints.py` | 123 | 0% | 🔴 CRITIQUE |

### Services Core

| Fichier | Statements | Coverage | Status |
|---------|-----------|----------|--------|
| `services/portfolio.py` | 407 | 13% | 🔴 CRITIQUE |
| `services/pricing.py` | 181 | 31% | 🟡 MEDIUM |
| `services/balance_service.py` | 111 | 84% | ✅ GOOD |
| `services/risk_scoring.py` | 119 | 99% | ✅✅ EXCELLENT |

### Services Risk

| Fichier | Statements | Coverage | Status |
|---------|-----------|----------|--------|
| `services/risk_management.py` | 883 | 46% | 🟡 MEDIUM |
| `services/risk/var_calculator.py` | 254 | 8% | 🔴 CRITIQUE |
| `services/risk/structural_score_v2.py` | 48 | 44% | 🟡 MEDIUM |

### Services Execution

| Fichier | Statements | Coverage | Status |
|---------|-----------|----------|--------|
| `services/execution/governance.py` | 1,008 | 11% | 🔴 CRITIQUE |
| `services/execution/exchange_adapter.py` | 197 | 8% | 🔴 CRITIQUE |
| `services/execution/liquidation_manager.py` | 63 | 0% | 🔴 CRITIQUE |

### Services ML

| Fichier | Statements | Coverage | Status |
|---------|-----------|----------|--------|
| `services/ml/orchestrator.py` | 318 | 56% | 🟡 MEDIUM |
| `services/ml/data_pipeline.py` | 255 | 36% | 🟡 MEDIUM |
| `services/ml_pipeline_manager_optimized.py` | 366 | 23% | 🟡 MEDIUM |

---

## 5. 🎯 Roadmap d'Amélioration

### Phase 1: Critical Paths (2 semaines) - 37% → 50%

**Objectif:** Tester les fonctions financières critiques non testées

#### Semaine 1: Risk & Portfolio (Priority 1)
```bash
# Tests à créer
tests/unit/test_var_calculator.py
tests/unit/test_portfolio_metrics.py
tests/unit/test_liquidation_manager.py

# Coverage cible
services/risk/var_calculator.py:     8% → 60% (+52%)
services/portfolio.py:              13% → 60% (+47%)
services/execution/liquidation_manager.py: 0% → 50% (+50%)
```

**Tests Critiques:**
```python
# tests/unit/test_var_calculator.py
def test_parametric_var_calculation():
    """Test VaR parametric method with known portfolio"""
    calculator = VarCalculator()
    var_95 = calculator.calculate_var(
        portfolio_value=100000,
        returns=historical_returns,
        confidence=0.95,
        method='parametric'
    )
    # VaR 95% should be positive and < portfolio value
    assert 0 < var_95 < 100000
    assert var_95 > 1000  # Reasonable minimum

def test_historical_var_calculation():
    """Test VaR historical method"""
    # ...

def test_monte_carlo_var_calculation():
    """Test VaR Monte Carlo simulation"""
    # ...

# tests/unit/test_portfolio_metrics.py
def test_calculate_performance_metrics():
    """Test P&L calculation with known data"""
    portfolio_service = PortfolioService()
    metrics = portfolio_service.calculate_performance_metrics(
        user_id='demo',
        lookback_days=30
    )
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert metrics['total_return'] is not None

def test_save_portfolio_snapshot():
    """Test portfolio snapshot persistence"""
    # ...
```

**Impact Attendu:** +10% coverage global (37% → 47%)

#### Semaine 2: Execution & Validation (Priority 1)
```bash
# Tests à créer
tests/integration/test_execution_validation.py
tests/integration/test_exchange_adapter.py
tests/unit/test_governance_engine.py

# Coverage cible
api/execution/validation_endpoints.py:  0% → 70% (+70%)
services/execution/exchange_adapter.py: 8% → 50% (+42%)
services/execution/governance.py:      11% → 30% (+19%)
```

**Impact Attendu:** +3% coverage global (47% → 50%) ✅ **Objectif Q1 atteint**

---

### Phase 2: API Endpoints (1 mois) - 50% → 65%

**Objectif:** Tester tous les endpoints critiques

#### Semaine 3-4: Risk Endpoints
```bash
# Tests à créer
tests/integration/test_risk_endpoints_complete.py

# Coverage cible
api/risk_endpoints.py: 15% → 70% (+55%)
```

**Tests Coverage:**
- ✅ GET /api/risk/dashboard (user isolation)
- ✅ GET /api/risk/advanced (VaR, stress tests)
- ✅ GET /api/risk/onchain-score
- ✅ GET /api/risk/structural-score
- ✅ Multi-user isolation
- ✅ Error handling

**Impact Attendu:** +5% coverage global (50% → 55%)

#### Semaine 5-6: ML & Rebalancing Endpoints
```bash
# Tests à créer
tests/integration/test_ml_endpoints_complete.py
tests/integration/test_rebalancing_endpoints.py

# Coverage cible
api/unified_ml_endpoints.py: 24% → 60% (+36%)
api/main.py: 27% → 50% (+23%)
```

**Impact Attendu:** +10% coverage global (55% → 65%)

---

### Phase 3: ML & Services (1 mois) - 65% → 80%

#### Semaine 7-8: ML Pipeline
```bash
# Coverage cible
services/ml/orchestrator.py: 56% → 80% (+24%)
services/ml/data_pipeline.py: 36% → 70% (+34%)
services/ml_pipeline_manager_optimized.py: 23% → 60% (+37%)
```

**Tests Critiques:**
- ML model loading
- Predictions
- Cache management
- Error fallbacks

**Impact Attendu:** +8% coverage global (65% → 73%)

#### Semaine 9-10: Remaining Services
```bash
# Coverage cible
services/risk_management.py: 46% → 75% (+29%)
services/pricing.py: 31% → 80% (+49%)
services/taxonomy.py: 56% → 75% (+19%)
```

**Impact Attendu:** +7% coverage global (73% → 80%) ✅ **Objectif Q2 atteint**

---

## 6. 🔥 Top 20 Fichiers à Tester en Priorité

### Critères de Priorisation
```
Score = (1 - Coverage) × Criticality × Statements

Criticality:
- Financial calculations (VaR, P&L, pricing): 10/10
- Execution (trades, validation): 9/10
- API endpoints critiques: 8/10
- ML pipeline: 7/10
- Services support: 5/10
```

### Top 20 Liste

| Rank | Fichier | Coverage | Statements | Criticality | Score | Action |
|------|---------|----------|------------|-------------|-------|--------|
| 1 | `services/risk/var_calculator.py` | 8% | 254 | 10 | 2,336 | 🔴 URGENT |
| 2 | `services/portfolio.py` | 13% | 407 | 10 | 3,541 | 🔴 URGENT |
| 3 | `services/execution/liquidation_manager.py` | 0% | 63 | 9 | 567 | 🔴 URGENT |
| 4 | `services/execution/governance.py` | 11% | 1,008 | 9 | 8,078 | 🔴 URGENT |
| 5 | `api/execution/validation_endpoints.py` | 0% | 123 | 9 | 1,107 | 🔴 URGENT |
| 6 | `services/execution/exchange_adapter.py` | 8% | 197 | 9 | 1,632 | 🔴 URGENT |
| 7 | `api/risk_endpoints.py` | 15% | 577 | 8 | 3,924 | ⚠️ HIGH |
| 8 | `services/pricing.py` | 31% | 181 | 10 | 1,249 | ⚠️ HIGH |
| 9 | `api/main.py` | 27% | 531 | 8 | 3,100 | ⚠️ HIGH |
| 10 | `services/risk_management.py` | 46% | 883 | 9 | 4,289 | ⚠️ HIGH |
| 11 | `api/unified_ml_endpoints.py` | 24% | 863 | 7 | 4,588 | ⚠️ MEDIUM |
| 12 | `services/ml_pipeline_manager_optimized.py` | 23% | 366 | 7 | 1,974 | ⚠️ MEDIUM |
| 13 | `services/ml/data_pipeline.py` | 36% | 255 | 7 | 1,142 | ⚠️ MEDIUM |
| 14 | `services/rebalance.py` | 65% | 291 | 8 | 815 | 🟡 MEDIUM |
| 15 | `services/taxonomy.py` | 56% | 124 | 7 | 381 | 🟡 MEDIUM |
| 16 | `services/user_secrets.py` | 32% | 60 | 8 | 326 | 🟡 MEDIUM |
| 17 | `api/rebalancing_strategy_router.py` | 58% | 92 | 7 | 271 | 🟡 MEDIUM |
| 18 | `services/ml/orchestrator.py` | 56% | 318 | 7 | 979 | 🟡 MEDIUM |
| 19 | `services/execution/governance_legacy.py` | 49% | 156 | 6 | 478 | 🟡 MEDIUM |
| 20 | `services/alerts/alert_engine.py` | 43% | 749 | 6 | 2,561 | 🟡 MEDIUM |

---

## 7. 📈 Stratégie d'Amélioration

### Quick Wins (1 semaine, +5-10% coverage)

**Cible:** Fichiers critiques avec tests faciles à écrire

```python
# 1. services/pricing.py (31% → 80%)
# Tests simples: mocking API calls
def test_get_prices_usd_binance():
    """Test price fetching from Binance"""
    with patch('httpx.Client.get') as mock_get:
        mock_get.return_value.json.return_value = {'price': '50000'}
        price = get_price_from_binance('BTC')
        assert price == 50000.0

# 2. services/balance_service.py (84% → 95%)
# Déjà bien testé, compléter edge cases

# 3. api/main.py middlewares (27% → 40%)
# Tests middlewares security headers
def test_security_headers_present():
    response = client.get('/api/health')
    assert 'X-Content-Type-Options' in response.headers
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
```

**Impact:** +5-8% coverage en 1 semaine

### Long-term Strategy

**Règles:**
1. **Tout nouveau code** = coverage 80%+ obligatoire
2. **PR reviews** = vérifier coverage diff (+/-)
3. **CI/CD** = fail si coverage < 37% (baseline)
4. **Monthly target** = +5% coverage/mois

**Timeline:**
```
Nov 2025: 37% (baseline)
Dec 2025: 42% (+5%)
Jan 2026: 47% (+5%)
Feb 2026: 52% (+5%)
Mar 2026: 57% (+5%)
Q2 2026:  65-80%
```

---

## 8. ✅ Tests Actuels - Analyse

### Tests Passés (775 tests) ✅

**Bien testés:**
- ✅ Balance service (core)
- ✅ Risk scoring algorithms
- ✅ Stop loss calculations
- ✅ Smart classification
- ✅ Quelques endpoints API basiques

### Tests Échoués (99 tests) ❌

**Catégories d'échecs:**
1. **Integration tests** (45 failed) - Dépendances externes
2. **ML tests** (18 failed) - Models non chargés
3. **E2E tests** (12 failed) - Serveur non running
4. **Unit tests** (24 failed) - Mocks incomplets

**Recommandation:** ⚠️ Fixer tests échoués avant ajouter nouveaux tests

### Tests Skipped (36 tests) ⏸️

**Raison:** Conditions non remplies (Redis, models ML, etc.)

**Action:** Configurer environnement test complet

---

## 9. 🛠️ Setup Amélioration Coverage

### 9.1 Configuration CI/CD

**GitHub Actions:** `.github/workflows/coverage.yml`

```yaml
name: Test Coverage

on: [push, pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest-cov

      - name: Run tests with coverage
        run: |
          pytest --cov=api --cov=services \
                 --cov-report=xml \
                 --cov-report=term \
                 --cov-fail-under=37

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true
```

### 9.2 Pre-commit Hook

**.pre-commit-config.yaml**

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest-coverage
        name: Check test coverage
        entry: pytest
        args: [
          '--cov=api',
          '--cov=services',
          '--cov-report=term-missing',
          '--cov-fail-under=37'
        ]
        language: system
        pass_filenames: false
```

### 9.3 Coverage Badge

**README.md**

```markdown
[![Coverage](https://img.shields.io/badge/coverage-37%25-yellow)](htmlcov/index.html)

Target: 80% by Q2 2026
```

---

## 10. 📊 Conclusion

### État Actuel

**✅ Points Positifs:**
1. Baseline mesurée: 37% (vs estimation 25-30%)
2. Quelques fichiers excellents (99% risk_scoring.py)
3. 775 tests passés (bonne base)
4. Infrastructure test en place

**⚠️ Points d'Attention:**
1. 🔴 Fichiers critiques < 20% (VaR, portfolio, execution)
2. 🔴 99 tests échoués à corriger
3. 🔴 Fonctions financières critiques non testées
4. 🔴 Gap objectif production: 37% vs 80% (-43%)

### Plan d'Action Immédiat

**Semaine 1-2 (NOW):**
```bash
# Priority 1: Fix failing tests
pytest tests/unit/ -v  # Identifier root causes
# Fix 50% des tests échoués

# Priority 2: Test critical paths
# Créer tests VaR calculator
# Créer tests portfolio metrics
# Créer tests liquidation manager

# Target: 37% → 42% (+5%)
```

**Q1 2026:**
- Coverage: 37% → 50% (+35%)
- Tests critical paths: 100%
- Tests échoués: 99 → 0

**Q2 2026:**
- Coverage: 50% → 80% (+60%)
- Production ready: ✅

### Success Metrics

| Métrique | Baseline | Q1 2026 | Q2 2026 |
|----------|----------|---------|---------|
| **Coverage** | 37% | 50% | 80% |
| **Tests passés** | 775 | 850+ | 950+ |
| **Tests échoués** | 99 | 20 | 0 |
| **Fichiers critiques <20%** | 6 | 2 | 0 |

---

## 11. 📁 Fichiers Générés

### Rapports Coverage

1. ✅ **coverage.json** (1.3 MB) - Données brutes complètes
2. ✅ **htmlcov/index.html** - Rapport HTML interactif
   - Ouvrir avec: `start htmlcov/index.html`
   - Vue par fichier avec lignes non testées surlignées
3. ✅ **coverage.xml** - Format XML pour CI/CD
4. ✅ **TEST_COVERAGE_REPORT_2025-11-22.md** - Ce rapport

### Commandes Utiles

```bash
# Re-run coverage scan
source .venv/Scripts/activate
pytest --cov=api --cov=services --cov-report=html --cov-report=term-missing

# Ouvrir rapport HTML
start htmlcov/index.html

# Coverage pour un fichier spécifique
pytest --cov=services/portfolio --cov-report=term-missing tests/unit/test_portfolio.py

# Coverage avec threshold
pytest --cov=api --cov=services --cov-fail-under=37

# Coverage diff (avant/après modifications)
pytest --cov=api --cov=services --cov-report=term-missing > coverage_before.txt
# ... modifications ...
pytest --cov=api --cov=services --cov-report=term-missing > coverage_after.txt
diff coverage_before.txt coverage_after.txt
```

---

**Rapport généré le:** 22 Novembre 2025
**Coverage Baseline:** 37%
**Prochaine mesure:** 22 Décembre 2025
**Objectif Q1 2026:** 50%
**Objectif Q2 2026:** 80%

---

## Annexe A: Coverage Détaillé par Module

### API Module (Total: 27% avg)
- api/main.py: 27%
- api/risk_endpoints.py: 15%
- api/unified_ml_endpoints.py: 24%
- api/rebalancing_strategy_router.py: 58%
- api/execution/validation_endpoints.py: 0%

### Services Core (Total: 42% avg)
- services/balance_service.py: 84% ✅
- services/portfolio.py: 13% 🔴
- services/pricing.py: 31% 🟡
- services/risk_scoring.py: 99% ✅✅

### Services Risk (Total: 33% avg)
- services/risk_management.py: 46%
- services/risk/var_calculator.py: 8% 🔴
- services/risk/structural_score_v2.py: 44%

### Services Execution (Total: 6% avg) 🔴
- services/execution/governance.py: 11%
- services/execution/exchange_adapter.py: 8%
- services/execution/liquidation_manager.py: 0%

### Services ML (Total: 38% avg)
- services/ml/orchestrator.py: 56%
- services/ml/data_pipeline.py: 36%
- services/ml_pipeline_manager_optimized.py: 23%

---

**Status:** 🟡 BASELINE ÉTABLIE - Amélioration Required
**Next Action:** Implémenter Phase 1 Roadmap (Critical Paths)

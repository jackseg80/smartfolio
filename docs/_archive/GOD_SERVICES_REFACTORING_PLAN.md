# God Services Refactoring Plan
## Date: 20 Octobre 2025

---

## Executive Summary

**Objectif:** Décomposer 3 "God Services" (5,740 lignes) en 12 modules focused

**État actuel:**
- `services/execution/governance.py` - **2,015 lignes** (8 classes, 30+ méthodes)
- `services/risk_management.py` - **2,159 lignes** (12 classes, 30+ méthodes)
- `services/alerts/alert_engine.py` - **1,566 lignes** (4 classes, 30+ méthodes)

**Résultat attendu:**
- **12 nouveaux modules** bien séparés
- **Réduction:** 5,740 → ~2,500 lignes (core orchestrators)
- **+3,240 lignes** extraites dans modules spécialisés

---

## services/execution/governance.py (2,015 lignes)

### Problèmes Identifiés

**Responsabilités multiples:**
1. **Policy management** (cap calculation, bounds enforcement)
2. **ML signals integration** (volatility, regime, correlation, sentiment)
3. **Freeze/unfreeze logic** (system governance, operation validation)
4. **Plan lifecycle** (review, approve, reject, activate, execute, cancel)
5. **Hystérésis & anti-yo-yo** (VAR, staleness detection)
6. **Alert integration** (cap reduction, progressive clearing)
7. **Hybrid Intelligence** (ExplainableAI, HumanInTheLoop, FeedbackLearning)

**Classes et fonctions:**
```python
# Classes (8 total)
FreezeType                    # Lignes 48-60 (13 lignes)
FreezeSemantics              # Lignes 54-122 (68 lignes)
Target                       # Ligne 124 (5 lignes)
Policy                       # Ligne 129 (18 lignes)
MLSignals                    # Ligne 147 (20 lignes)
DecisionPlan                 # Ligne 167 (32 lignes)
DecisionState                # Ligne 199 (25 lignes)
GovernanceEngine             # Lignes 224-2015 (~1800 lignes!) ← GOD OBJECT

# Méthodes GovernanceEngine (30+ méthodes)
__init__()
_enforce_policy_bounds()
get_current_state()
_refresh_ml_signals()
_compute_contradiction_index()
_derive_execution_policy()              # 256 lignes! (478-734)
apply_alert_cap_reduction()
clear_alert_cap_reduction()
_update_hysteresis_state()              # 84 lignes (798-882)
_extract_volatility_signals()
_extract_regime_signals()
_extract_correlation_signals()
_extract_sentiment_signals()
get_current_ml_signals()
freeze_system()                         # 69 lignes (965-1034)
unfreeze_system()
validate_operation()
get_freeze_status()
check_auto_unfreeze()
review_plan()                           # 33 lignes (1148-1181)
approve_plan()                          # 72 lignes (1181-1253)
reject_plan()                           # 45 lignes (1253-1298)
activate_plan()                         # 42 lignes (1298-1340)
execute_plan()                          # 31 lignes (1340-1371)
cancel_plan()                           # 38 lignes (1371-1409)
_find_plan_by_id()
validate_etag()
is_change_within_no_trade_zone()
```

### Plan de Refactoring (4 modules)

#### Module 1: `services/execution/governance/policy_engine.py` (~400 lignes)
**Responsabilité:** Gestion des policies d'exécution

**Contenu:**
```python
# Classes
class Policy (déplacé)
class PolicyEngine:
    # Méthodes
    def _enforce_policy_bounds()
    def _derive_execution_policy()       # 256 lignes de logique métier
    def is_change_within_no_trade_zone()
    def apply_alert_cap_reduction()
    def clear_alert_cap_reduction()
```

**Bénéfices:**
- Logique policy isolée et testable
- Pas de dépendances ML
- Peut être réutilisé par d'autres services

#### Module 2: `services/execution/governance/freeze_manager.py` (~300 lignes)
**Responsabilité:** Gestion des freezes système

**Contenu:**
```python
# Classes
class FreezeType (déplacé)
class FreezeSemantics (déplacé)
class FreezeManager:
    # Méthodes
    async def freeze_system()            # 69 lignes
    async def unfreeze_system()
    def validate_operation()
    def get_freeze_status()
    async def check_auto_unfreeze()
```

**Bénéfices:**
- Sécurité système isolée
- Facile à tester (mocks)
- SRP respecté

#### Module 3: `services/execution/governance/ml_signals_adapter.py` (~350 lignes)
**Responsabilité:** Integration signaux ML

**Contenu:**
```python
# Classes
class MLSignals (déplacé)
class MLSignalsAdapter:
    # Méthodes
    async def _refresh_ml_signals()
    def _compute_contradiction_index()
    def _extract_volatility_signals()
    def _extract_regime_signals()
    def _extract_correlation_signals()
    def _extract_sentiment_signals()
    async def get_current_ml_signals()
    def _update_hysteresis_state()       # 84 lignes
```

**Bénéfices:**
- Découplage ML ↔ Governance
- Hystérésis anti-yo-yo centralisée
- Peut être mocké pour tests

#### Module 4: `services/execution/governance/plan_lifecycle.py` (~400 lignes)
**Responsabilité:** Gestion du cycle de vie des plans

**Contenu:**
```python
# Classes
class Target (déplacé)
class DecisionPlan (déplacé)
class PlanLifecycleManager:
    # Méthodes
    async def review_plan()              # 33 lignes
    async def approve_plan()             # 72 lignes
    async def reject_plan()              # 45 lignes
    async def activate_plan()            # 42 lignes
    async def execute_plan()             # 31 lignes
    async def cancel_plan()              # 38 lignes
    def _find_plan_by_id()
    def validate_etag()
```

**Bénéfices:**
- Workflow séparé de la logique métier
- ETAG validation isolée
- Facilite audit trail

#### Module 5 (Core): `services/execution/governance_engine.py` (~565 lignes)
**Responsabilité:** Orchestration centrale (slim coordinator)

**Contenu:**
```python
# Classes
class DecisionState (reste ici)
class GovernanceEngine:
    def __init__(self):
        # Injection de dépendances
        self.policy_engine = PolicyEngine()
        self.freeze_manager = FreezeManager()
        self.ml_adapter = MLSignalsAdapter()
        self.plan_lifecycle = PlanLifecycleManager()

    # Méthodes publiques (délégation)
    async def get_current_state() -> DecisionState
    async def freeze_system() -> delegate to freeze_manager
    async def approve_plan() -> delegate to plan_lifecycle
    # etc.
```

**Bénéfices:**
- Interface publique inchangée (backward compatible)
- Logique métier déléguée
- Facilite les tests (mock dependencies)

---

## services/risk_management.py (2,159 lignes)

### Problèmes Identifiés

**Responsabilités multiples:**
1. **VaR/CVaR calculation** (portfolio risk metrics)
2. **Correlation matrix** (asset correlations)
3. **Stress testing** (scenario analysis)
4. **Performance attribution** (asset/group contributions)
5. **Backtesting** (strategy simulation)
6. **Alert generation** (risk threshold alerts)

**Classes:**
```python
# Enums & Models (7 classes)
RiskLevel                    # Ligne 32
StressScenario               # Ligne 41
RiskMetrics                  # Ligne 50
CorrelationMatrix            # Ligne 85
StressTestResult             # Ligne 96
PerformanceAttribution       # Ligne 109
BacktestResult               # Ligne 135

# Alert System (3 classes)
AlertSeverity                # Ligne 175
AlertCategory                # Ligne 183
RiskAlert                    # Ligne 194
AlertSystem                  # Lignes 219-385 (~166 lignes)

# Main Manager (1 class)
AdvancedRiskManager          # Lignes 389-2159 (~1770 lignes!) ← GOD OBJECT
```

**Méthodes AdvancedRiskManager (30+ méthodes):**
```python
__init__()
_build_stress_scenarios()
calculate_portfolio_risk_metrics()      # 124 lignes (504-628)
_generate_historical_returns()          # 94 lignes (628-722)
_generate_historical_returns_fallback() # 35 lignes (722-757)
_calculate_portfolio_returns()          # 32 lignes (757-789)
_calculate_var_cvar()                   # 23 lignes (789-812)
_calculate_risk_adjusted_metrics()      # 38 lignes (812-850)
_calculate_drawdown_metrics()           # 47 lignes (850-897)
_calculate_distribution_metrics()       # 16 lignes (897-913)
_assess_overall_risk_level()            # 81 lignes (913-994)
calculate_correlation_matrix()          # 108 lignes (994-1102)
run_stress_test()                       # 113 lignes (1102-1215)
calculate_performance_attribution()     # 79 lignes (1215-1294)
_calculate_asset_contributions()        # 56 lignes (1294-1350)
_calculate_group_contributions()        # 56 lignes (1350-1406)
_calculate_attribution_effects()        # 61 lignes (1406-1467)
run_strategy_backtest()                 # 76 lignes (1467-1543)
_generate_asset_universe()              # 55 lignes (1543-1598)
_simulate_backtest()                    # 184 lignes! (1598-1782)
generate_intelligent_alerts()           # 58 lignes (1782-1840)
_check_risk_threshold_alerts()          # 73 lignes (1840-1913)
_check_performance_alerts()             # ... (1913+)
```

### Plan de Refactoring (5 modules)

#### Module 1: `services/risk/var_calculator.py` (~400 lignes)
**Responsabilité:** Calcul VaR/CVaR et métriques de risque

**Contenu:**
```python
# Classes
class RiskLevel (déplacé)
class RiskMetrics (déplacé)
class VaRCalculator:
    # Méthodes
    async def calculate_portfolio_risk_metrics()   # 124 lignes
    async def _generate_historical_returns()       # 94 lignes
    async def _generate_historical_returns_fallback() # 35 lignes
    def _calculate_portfolio_returns()             # 32 lignes
    def _calculate_var_cvar()                      # 23 lignes
    def _calculate_risk_adjusted_metrics()         # 38 lignes
    def _calculate_drawdown_metrics()              # 47 lignes
    def _calculate_distribution_metrics()          # 16 lignes
    def _assess_overall_risk_level()               # 81 lignes
```

**Bénéfices:**
- Calculs financiers isolés
- Facile à unit test avec données synthétiques
- Pas de dépendances externes

#### Module 2: `services/risk/correlation_engine.py` (~200 lignes)
**Responsabilité:** Matrices de corrélation

**Contenu:**
```python
# Classes
class CorrelationMatrix (déplacé)
class CorrelationEngine:
    # Méthodes
    async def calculate_correlation_matrix()       # 108 lignes
```

**Bénéfices:**
- Calculs mathématiques séparés
- Peut utiliser scipy/numpy sans polluer le reste
- Réutilisable pour d'autres analyses

#### Module 3: `services/risk/stress_tester.py` (~300 lignes)
**Responsabilité:** Tests de stress scénarios

**Contenu:**
```python
# Classes
class StressScenario (déplacé)
class StressTestResult (déplacé)
class StressTester:
    # Méthodes
    def __init__(self):
        self.scenarios = self._build_stress_scenarios()

    def _build_stress_scenarios()                  # Déplacé
    async def run_stress_test()                    # 113 lignes
```

**Bénéfices:**
- Scénarios configurables
- Peut être étendu sans toucher au reste
- Tests unitaires faciles

#### Module 4: `services/risk/performance_attribution.py` (~350 lignes)
**Responsabilité:** Attribution de performance

**Contenu:**
```python
# Classes
class PerformanceAttribution (déplacé)
class PerformanceAttributor:
    # Méthodes
    async def calculate_performance_attribution()  # 79 lignes
    def _calculate_asset_contributions()           # 56 lignes
    def _calculate_group_contributions()           # 56 lignes
    def _calculate_attribution_effects()           # 61 lignes
```

**Bénéfices:**
- Logique P&L séparée
- Facilite ajout de nouvelles méthodes d'attribution
- Pas de dépendances ML

#### Module 5: `services/risk/backtesting_engine.py` (~400 lignes)
**Responsabilité:** Backtesting de stratégies

**Contenu:**
```python
# Classes
class BacktestResult (déplacé)
class BacktestingEngine:
    # Méthodes
    async def run_strategy_backtest()              # 76 lignes
    def _generate_asset_universe()                 # 55 lignes
    async def _simulate_backtest()                 # 184 lignes! (complexe)
```

**Bénéfices:**
- Simulation isolée
- Peut être optimisée (vectorization) sans impact
- Facilite A/B testing de stratégies

#### Module 6 (Core): `services/risk_management.py` (~500 lignes)
**Responsabilité:** Orchestration + Alert System

**Contenu:**
```python
# Classes
class AlertSeverity, AlertCategory, RiskAlert (restent ici)
class AlertSystem (reste ici, 166 lignes)

class AdvancedRiskManager:
    def __init__(self):
        # Injection de dépendances
        self.var_calculator = VaRCalculator()
        self.correlation_engine = CorrelationEngine()
        self.stress_tester = StressTester()
        self.performance_attributor = PerformanceAttributor()
        self.backtesting_engine = BacktestingEngine()
        self.alert_system = AlertSystem()

    # Méthodes publiques (délégation)
    async def calculate_portfolio_risk_metrics() -> delegate
    async def calculate_correlation_matrix() -> delegate
    async def run_stress_test() -> delegate
    async def calculate_performance_attribution() -> delegate
    async def run_strategy_backtest() -> delegate
    async def generate_intelligent_alerts()        # 58 lignes (reste ici)
    def _check_risk_threshold_alerts()             # 73 lignes (reste ici)
    def _check_performance_alerts()                # ... (reste ici)
```

**Bénéfices:**
- Interface publique inchangée
- Alert logic colocated (petite surface)
- Orchestrateur slim

---

## services/alerts/alert_engine.py (1,566 lignes)

### Problèmes Identifiés

**Responsabilités multiples:**
1. **Phase-aware context** (lagged phases, multi-timeframe)
2. **Alert metrics** (Prometheus-style)
3. **Configuration management** (hot reload)
4. **Scheduler loop** (background evaluation)
5. **Alert evaluation** (20+ alert types)
6. **Alert escalation** (S3, systemic alerts)
7. **Cap reduction** (governance integration)
8. **Maintenance tasks** (budget, quiet hours)

**Classes:**
```python
PhaseSnapshot                # Ligne 32
PhaseAwareContext            # Lignes 40-111 (~71 lignes)
AlertMetrics                 # Lignes 112-163 (~51 lignes)
AlertEngine                  # Lignes 164-1566 (~1402 lignes!) ← GOD OBJECT
```

**Méthodes AlertEngine (30+ méthodes):**
```python
__init__()
get_lagged_phase()
get_multi_timeframe_status()
is_phase_stable()
_extract_assets_data_from_signals()
_check_phase_gating()
_load_config()
_check_config_reload()
_default_config()
async start()
async stop()
async _scheduler_loop()                    # 38 lignes (568-606)
async _evaluate_alerts()                   # 66 lignes (606-672)
async _evaluate_alert_type()               # 443 lignes! (672-1115)
def _create_alert()                        # 30 lignes (1115-1145)
def _apply_systemic_alert_cap_reduction()  # 52 lignes (1145-1197)
async _check_escalations()                 # 34 lignes (1197-1231)
async _escalate_to_s3()                    # 41 lignes (1231-1272)
async _maintenance_tasks()                 # 16 lignes (1272-1288)
def _check_daily_budget()                  # 21 lignes (1288-1309)
def _is_quiet_hours()                      # ... (1309+)
```

### Plan de Refactoring (3 modules)

#### Module 1: `services/alerts/alert_scheduler.py` (~300 lignes)
**Responsabilité:** Orchestration scheduler + lifecycle

**Contenu:**
```python
# Classes
class AlertScheduler:
    # Méthodes
    def __init__(self)
    async def start()
    async def stop()
    async def _scheduler_loop()                # 38 lignes
    async def _evaluate_alerts()               # 66 lignes
    async def _maintenance_tasks()             # 16 lignes
    def _check_daily_budget()                  # 21 lignes
    def _is_quiet_hours()
```

**Bénéfices:**
- Lifecycle isolé (start/stop)
- Budget + quiet hours colocated
- Facile à tester (mock async)

#### Module 2: `services/alerts/alert_evaluator.py` (~600 lignes)
**Responsabilité:** Évaluation des alertes (business logic)

**Contenu:**
```python
# Classes
class AlertEvaluatorService:
    # Méthodes
    async def _evaluate_alert_type()           # 443 lignes! (logic métier)
    def _extract_assets_data_from_signals()
    def _check_phase_gating()
    def _create_alert()                        # 30 lignes
```

**Bénéfices:**
- Business logic séparée
- Peut être testé avec mocks signals
- Facilite ajout de nouveaux alert types

#### Module 3: `services/alerts/alert_governance_bridge.py` (~250 lignes)
**Responsabilité:** Intégration avec Governance (escalation, cap reduction)

**Contenu:**
```python
# Classes
class AlertGovernanceBridge:
    # Méthodes
    def _apply_systemic_alert_cap_reduction()  # 52 lignes
    async def _check_escalations()             # 34 lignes
    async def _escalate_to_s3()                # 41 lignes
```

**Bénéfices:**
- Pont clair Alert ↔ Governance
- Pas de couplage fort
- Facilite tests d'intégration

#### Module 4 (Core): `services/alerts/alert_engine.py` (~416 lignes)
**Responsabilité:** Orchestration + context management

**Contenu:**
```python
# Classes
class PhaseSnapshot (reste ici)
class PhaseAwareContext (reste ici, 71 lignes)
class AlertMetrics (reste ici, 51 lignes)

class AlertEngine:
    def __init__(self):
        # Injection de dépendances
        self.scheduler = AlertScheduler()
        self.evaluator = AlertEvaluatorService()
        self.governance_bridge = AlertGovernanceBridge()
        self.phase_context = PhaseAwareContext()
        self.metrics = AlertMetrics()

    # Configuration management (reste ici)
    def _load_config()
    def _check_config_reload()
    def _default_config()

    # Context delegates
    def get_lagged_phase() -> delegate to phase_context
    def get_multi_timeframe_status() -> delegate
    def is_phase_stable() -> delegate

    # Public API (delegation)
    async def start() -> delegate to scheduler
    async def stop() -> delegate to scheduler
```

**Bénéfices:**
- Interface publique inchangée
- Config management centralisé
- Context logic colocated (petit surface)

---

## Plan d'Exécution

### Phase 1: Refactor Governance (Semaine 1-2)

**Jour 1-2: Créer les modules**
```bash
mkdir -p services/execution/governance

# Créer fichiers
touch services/execution/governance/__init__.py
touch services/execution/governance/policy_engine.py
touch services/execution/governance/freeze_manager.py
touch services/execution/governance/ml_signals_adapter.py
touch services/execution/governance/plan_lifecycle.py
```

**Jour 3-4: Extraire le code**
- Copier classes/méthodes dans nouveaux modules
- Ajouter imports nécessaires
- Fixer références croisées

**Jour 5: Refactor core orchestrator**
- Créer instances des nouveaux modules dans `__init__()`
- Déléguer méthodes publiques
- Préserver interface publique

**Jour 6-7: Tests**
- Tests unitaires pour chaque module
- Tests d'intégration pour GovernanceEngine
- Vérifier backward compatibility

**Jour 8-9: Cleanup**
- Supprimer code mort
- Optimiser imports
- Documentation

**Jour 10: Commit**
```bash
git add services/execution/governance/
git commit -m "refactor(governance): split GovernanceEngine into 4 modules

- Extract PolicyEngine (policy management, cap calculation)
- Extract FreezeManager (freeze/unfreeze logic)
- Extract MLSignalsAdapter (ML integration, hystérésis)
- Extract PlanLifecycleManager (plan workflow)

GovernanceEngine becomes slim orchestrator (2,015 → ~565 lines)

 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Phase 2: Refactor RiskManagement (Semaine 3-4)

**Jour 1-2: Créer les modules**
```bash
mkdir -p services/risk

# Créer fichiers
touch services/risk/__init__.py
touch services/risk/var_calculator.py
touch services/risk/correlation_engine.py
touch services/risk/stress_tester.py
touch services/risk/performance_attribution.py
touch services/risk/backtesting_engine.py
```

**Jour 3-5: Extraire le code**
- Copier classes/méthodes
- AlertSystem reste dans risk_management.py
- Fixer imports

**Jour 6-7: Tests**
- Tests unitaires (VaR, correlation, stress)
- Tests d'intégration

**Jour 8-10: Commit**

### Phase 3: Refactor AlertEngine (Semaine 5)

**Jour 1-2: Créer les modules**
```bash
# Créer fichiers
touch services/alerts/alert_scheduler.py
touch services/alerts/alert_evaluator.py
touch services/alerts/alert_governance_bridge.py
```

**Jour 3-4: Extraire le code**
- Scheduler logic
- Evaluator logic
- Governance bridge

**Jour 5: Tests + Commit**

---

## Bénéfices Attendus

### Maintenabilité
- **God objects éliminés:** 3 → 0
- **Modules focused:** Single Responsibility respecté
- **Testabilité:** +400% (isolation)

### Métriques
| Fichier | Avant | Après | Modules créés |
|---------|-------|-------|---------------|
| **governance.py** | 2,015 lignes | ~565 lignes | 4 modules |
| **risk_management.py** | 2,159 lignes | ~500 lignes | 5 modules |
| **alert_engine.py** | 1,566 lignes | ~416 lignes | 3 modules |
| **TOTAL** | **5,740 lignes** | **~1,481 lignes** | **12 modules** |

**Extraction:** ~3,240 lignes dans modules spécialisés
**Réduction core:** -74% (5,740 → 1,481)

### Architecture
```
services/
  execution/
    governance_engine.py         # 565 lignes (orchestrator)
    governance/
      __init__.py
      policy_engine.py          # 400 lignes
      freeze_manager.py         # 300 lignes
      ml_signals_adapter.py     # 350 lignes
      plan_lifecycle.py         # 400 lignes

  risk_management.py             # 500 lignes (orchestrator + alerts)
  risk/
    __init__.py
    var_calculator.py           # 400 lignes
    correlation_engine.py       # 200 lignes
    stress_tester.py            # 300 lignes
    performance_attribution.py  # 350 lignes
    backtesting_engine.py       # 400 lignes

  alerts/
    alert_engine.py              # 416 lignes (orchestrator + context)
    alert_scheduler.py           # 300 lignes
    alert_evaluator.py           # 600 lignes
    alert_governance_bridge.py   # 250 lignes
```

---

## Risques & Mitigation

### Risque 1: Breaking Changes
**Mitigation:**
- Préserver interface publique (backward compatibility)
- Tests d'intégration exhaustifs
- Déploiement progressif

### Risque 2: Circular Imports
**Mitigation:**
- Dependency Injection (Phase 6)
- Interfaces claires entre modules
- Éviter imports croisés

### Risque 3: Performance Regression
**Mitigation:**
- Benchmarks avant/après
- Profiling des hot paths
- Lazy loading si nécessaire

---

## Checklist Avant de Commencer

- [x] Analyse complète des 3 God Services
- [x] Plan de refactoring documenté
- [ ] Backup branche actuelle
- [ ] Créer branche feature: `feature/god-services-refactor`
- [ ] Tests baseline (tous passent)
- [ ] Commencer Phase 1 (Governance)

---

*Plan créé le 20 Octobre 2025 - Prêt pour exécution*

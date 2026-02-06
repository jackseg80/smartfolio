# FR → EN Migration Report

> **Date:** February 6, 2026
> **Status:** Complete
> **Scope:** All user-visible text translated from French to English

## Summary

Complete translation of the SmartFolio UI from French to English across ~50 files, ~300+ individual string translations. The project is now English-first for all user-facing content.

## Scope Definition

### Translated (user-visible)
- UI labels, buttons, headings, subheadings
- Error messages (`showNotification`, `showError`, `showToast`, `confirm`, `alert`)
- ARIA labels and `title` attributes (accessibility/tooltips)
- Placeholder text in forms
- Pydantic `Field(description=...)` strings (OpenAPI docs)
- `HTTPException(detail=...)` messages
- Config/preset display names
- Chart labels, axis titles, dataset names
- Loading states, empty states
- Date/number locale: `fr-FR` → `en-US`

### Not translated (intentionally kept in French)
- Code comments (`//`, `/* */`, `<!-- -->`)
- Log messages (`console.log`, `debugLogger`, `logger.info`)
- Internal variable/function names
- Developer documentation (CLAUDE.md)
- JSDoc comments
- File names (e.g., `bourse-analytics.html` stays as-is)

## Phase Breakdown

### Phase 1: Patrimoine → Wealth (Backend rename)

**Models** ([models/wealth.py](../models/wealth.py)):
- `PatrimoineItemInput` → `WealthItemInput` (backward compat alias kept)
- `PatrimoineItemOutput` → `WealthItemOutput` (backward compat alias kept)

**Services:**
- `services/wealth/patrimoine_service.py` → `wealth_service.py`
- `services/wealth/patrimoine_migration.py` → `wealth_migration.py`

**API Routes** ([api/wealth_endpoints.py](../api/wealth_endpoints.py)):
| Old Route | New Route |
|-----------|-----------|
| `/api/wealth/patrimoine/items` | `/api/wealth/items` |
| `/api/wealth/patrimoine/items/{id}` | `/api/wealth/items/{id}` |
| `/api/wealth/patrimoine/summary` | `/api/wealth/summary` |

Legacy `/patrimoine/` routes still work (redirect for backward compat).

**Data Storage:**
- Primary: `data/users/{user_id}/wealth/wealth.json`
- Fallback: `data/users/{user_id}/wealth/patrimoine.json`
- Migration script: `scripts/migrate_patrimoine_to_wealth.py`

### Phase 2: Backend Error Messages & Pydantic Descriptions

- ~50 `HTTPException(detail="...")` messages translated across ~15 API files
- ~232 `Field(description="...")` strings translated across ~23 model/endpoint files

### Phase 3: Frontend UI Text

**HTML Pages (~25 pages):**
- `<html lang="fr">` → `<html lang="en">` on all pages
- Page titles, h1/h2/h3 headings
- Button labels, loading states, empty states
- Metric labels, chart titles
- Form labels, placeholders

**JavaScript Modules (~30 files):**
- `showNotification()`, `showError()`, `showToast()` messages
- `confirm()` and `alert()` dialog text
- Chart.js labels and dataset names
- Status bar messages (`setStatus()`)
- Dynamic template literals
- Score interpretation strings (risk dashboard, decision index)

**Key translations:**

| French | English |
|--------|---------|
| Bourse | Stocks / Stock Market |
| Patrimoine | Wealth |
| Outils | Tools |
| Chargement... | Loading... |
| Aucun(e)... | No... |
| Erreur | Error |
| Actualiser | Refresh |
| Veuillez... | Please... |
| Données | Data |
| Disponible | Available |
| En cours | In progress |
| Enregistrer | Save |
| Supprimer | Delete |
| Annuler | Cancel |
| Fermer | Close |
| Proportionnel | Proportional |
| Faible | Low |
| Bon | Good |
| Neutre | Neutral |
| Euphorie | Euphoria |
| 30j / 90j / 365j | 30d / 90d / 365d |

### Phase 4: Config & Presets

- `config/strategy_templates.json` - 5 strategy descriptions
- `static/presets/sim_presets.json` - ~30 name/description/tooltip strings
- `static/presets/cycle_phase_presets.json` - 1 note
- `config/users.json` - Display names

### Phase 5: Bilingual Saxo CSV Parser

[services/sources/bourse/saxobank_csv.py](../services/sources/bourse/saxobank_csv.py):
- Added bilingual header mapping (FR ↔ EN)
- Parser normalizes headers before processing → accepts both FR and EN CSV exports

## Files Modified

### Backend (~20 files)
- `models/wealth.py`
- `services/wealth/wealth_service.py`, `wealth_migration.py`
- `api/wealth_endpoints.py`
- `api/advanced_rebalancing_endpoints.py`
- `api/strategy_endpoints.py`
- `api/sources_endpoints.py`
- `api/rebalancing_strategy_router.py`
- `api/csv_endpoints.py`
- `api/taxonomy_endpoints.py`
- `api/di_backtest_endpoints.py`
- `services/execution/governance.py`
- `config/settings.py`
- `services/sources/bourse/saxobank_csv.py`

### Frontend HTML (~25 files)
- `analytics-unified.html`, `advanced-risk.html`, `ai-dashboard.html`
- `alias-manager.html`, `admin-dashboard.html`
- `bourse-analytics.html`, `bourse-recommendations.html`
- `cycle-analysis.html`, `dashboard.html`, `di-backtest.html`
- `execution.html`, `execution_history.html`
- `monitoring.html`, `optimization.html`
- `performance-monitor-unified.html`
- `portfolio-optimization-advanced.html`
- `rebalance.html`, `risk-dashboard.html`
- `saxo-dashboard.html`, `settings.html`
- `simulations.html`, `wealth-dashboard.html`

### Frontend JS (~30 files)
- `components/`: `nav.js`, `decision-index-panel.js`, `GovernancePanel.js`, `WealthContextBar.js`, `ai-chat.js`, `SimControls.js`, `SimInspector.js`, `flyout-panel.js`, `risk-snapshot.js`, `risk-sidebar-full.js`, `UnifiedInsights.js`, `manual-source-editor.js`, `page-anchors-setup.js`, `domain-nav.js`, `ai-chat-context-builders.js`
- `modules/`: `dashboard-main-controller.js`, `rebalance-controller.js`, `risk-dashboard-main-controller.js`, `risk-overview-tab.js`, `risk-cycles-tab.js`, `risk-targets-tab.js`, `risk-dashboard-alerts-controller.js`, `settings-main-controller.js`, `settings-sources-utils.js`, `portfolio-optimization-logic.js`, `market-regimes.js`, `onchain-indicators.js`, `indicator-categories-v2.js`, `composite-score-v2.js`, `equities-utils.js`, `wealth-saxo-summary.js`, `analytics-unified-tabs-controller.js`
- Root: `sources-manager-v2.js`, `analytics-unified.js`, `input-validator.js`, `performance-optimizer.js`

### Config (~4 files)
- `config/strategy_templates.json`
- `static/presets/sim_presets.json`
- `static/presets/cycle_phase_presets.json`
- `config/users.json`

### Docs (~3 files)
- `docs/WEALTH_MODULE.md` (new, replaces PATRIMOINE_MODULE.md)
- `CLAUDE.md` (updated references)
- `CHANGELOG.md` (added migration entry)

## Rule Added to CLAUDE.md

**Section 5: English-Only UI** — enforces English for all new user-visible text going forward.

See [CLAUDE.md](../CLAUDE.md) section 5 for the full rule.

## Verification

Residual French text was verified via comprehensive grep sweeps. All remaining French is exclusively in:
- Code comments (`//`, `/* */`, `<!-- -->`)
- Debug/log messages (`console.log`, `debugLogger`, `logger.info`)
- JSDoc comments
- HTML comments
- Developer documentation

No user-visible French text remains in the codebase.

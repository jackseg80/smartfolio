# Dette Technique - Suivi et Roadmap

> **Dernière mise à jour** : Février 2026 (refactoring majeur complété)
> **Statut global** : [Positive] Excellent - Refactoring majeur terminé (19 tâches complétées)

Ce document centralise les TODO, FIXME et items de dette technique identifiés dans le codebase, avec priorités et plan de résolution.

## Vue d'Ensemble

| Catégorie | Items | Priorité | Action |
|-----------|-------|----------|--------|
| **Features futures** | 6 | [Positive] LOW | Backlog product |
| **À implémenter** | 2 | [Pending] MEDIUM | Plan d'implémentation |
| **Refactoring Fév 2026** | 19 |  DONE | Voir section dédiée |
| **HIGH priority résolus** | 2 |  DONE | Complétés Oct 2025 |
| **MEDIUM priority résolus** | 3 |  DONE | Complétés Oct 2025 |

**Total actif** : ~8 items features (dette technique majeure résolue)

---

## DONE - Refactoring Majeur Février 2026

> Voir rapport complet: [`docs/REFACTORING_2026_REPORT.md`](REFACTORING_2026_REPORT.md)

### Backend (4 tâches majeures)
| Composant | Avant | Après | Gain |
|-----------|-------|-------|------|
| governance.py | 2161L | ~500L + 4 modules | -77% |
| alert_engine.py | 1584L | 1324L + 3 modules | Modularisé |
| unified_ml_endpoints.py | 1728L | 57L + 8 modules | -97% |
| exchange_adapter.py | 1322L | ABC + classes | Factorisé |

### Frontend (4 tâches majeures)
| Composant | Avant | Après | Gain |
|-----------|-------|-------|------|
| risk-dashboard-controller.js | 4185L | 3518L | -16% |
| Systèmes fetch | 5 patterns | 1 (fetcher.js) | Unifié |
| localStorage | 255+ appels | StorageService | Centralisé |
| Legacy AI | 3 fichiers | Supprimés | -2100L |

### Cleanup (11 tâches)
- Fichiers parasites supprimés
- get_active_user deprecated supprimé
- 4 tests obsolètes supprimés
- /health/all endpoint unifié créé
- Script console.log migration préparé

---

## DONE - Migration Risk Dashboard (4 items - Oct 2025)

**Contexte** : Refactoring Risk Dashboard vers architecture modulaire (Oct 2025)

### Migration complétée

#### `static/modules/risk-cycles-tab.js` OK
- 1397 lignes de code complet
- Chart Bitcoin avec halvings, prix historique, cycle score
- Indicateurs on-chain avec catégories v2
- Calibration historique automatique
- Cache intelligent et lazy-loading

#### `static/modules/risk-targets-tab.js` OK
- 442 lignes de code complet
- 5 stratégies (macro, CCS, cycle, blend, smart)
- Action plan avec validation
- Decision history tracking
- Intégration targets-coordinator.js

#### Stubs supprimés
- `cycles-tab.js` (30 lignes) → Remplacé par `risk-cycles-tab.js`
- `targets-tab.js` (30 lignes) → Remplacé par `risk-targets-tab.js`

**Date complétion** : 10 octobre 2025
**Résultat** : Architecture modulaire opérationnelle, -60 lignes de stubs, +1839 lignes de code production

---

## LOW - Features Non Implémentées (6 items)

Ces items sont des fonctionnalités futures, pas des bugs. Backlog product.

### Admin Dashboard

#### `static/ai-dashboard.html` (2 TODO)
```javascript
// TODO: Implémenter le chargement de symboles spécifiques
// TODO: Endpoint spécifique pour détails régime
```

**Statut** : Feature request
**Justification** : Admin dashboard peu utilisé, priorité basse
**Action recommandée** : Garder dans backlog, implémenter si besoin utilisateur

### Backtesting

#### `static/backtesting.html` (1 TODO)
```javascript
// TODO: Implement strategy comparison
```

**Statut** : Feature request
**Justification** : Backtesting avancé = Phase 4
**Action recommandée** : Spécifier requirements avant implémentation

### InteractiveDashboard.js - SUPPRIMÉ (Nov 2025)

#### `static/components/InteractiveDashboard.js` (4 TODO)  DONE
**Statut** : Supprimé le 3 novembre 2025
**Justification** : Déprécié, remplacé par dashboards modernes, aucune dépendance trouvée
**Résultat** : -1229 lignes de code obsolète

---

## DONE - HIGH Priority Resolved (2 items - Oct 2025)

### 1. Wallet Stats

#### `static/core/unified-insights-v2.js:580-588`
**Statut** : Implémenté
**Date complétion** : 10 octobre 2025

```javascript
// Implémentation finale
const currentAllocations = window.store?.get('allocations.current') || {};
const sortedByWeight = Object.entries(currentAllocations).sort((a, b) => b[1] - a[1]);

const walletStats = {
  topWeightSymbol: sortedByWeight[0]?.[0] || null,
  topWeightPct: sortedByWeight[0]?.[1] || null,
  volatility: null // Deferred to risk metrics (requires historical data)
};
```

**Résultat** : Allocations dynamiques maintenant basées sur les stats wallet réelles

### 2. Governance Endpoint

#### `static/modules/risk-targets-tab.js:423`
**Statut** : Implémenté
**Date complétion** : 10 octobre 2025

**Problème résolu** : Targets appliqués directement sans workflow governance

**Implémentation** :
- Appel `POST /execution/governance/propose` pour créer plan DRAFT
- Workflow complet : DRAFT → REVIEWED → APPROVED → ACTIVE
- Fallback gracieux si API indisponible (mode local)
- Feedback utilisateur avec plan_id et état

**Résultat** : Workflow governance respecté avec traçabilité complète des décisions

---

## DONE - MEDIUM Priority Resolved (3 items - Oct 2025)

### 1. Governance Overrides Display

#### `static/components/UnifiedInsights.js:571`
**Statut** : Déjà implémenté (découvert lors de l'audit)
**Date vérification** : 10 octobre 2025

```javascript
// Implémentation existante
const overrides = window.store?.get('governance.overrides_count') || 0;
if (overrides > 0) badges.push(`Overrides ${overrides}`);
```

**Résultat** : Display badges avec count des overrides manuels dans UnifiedInsights header

### 2. Fix getApiUrl() Duplication Bug

#### `static/global-config.js:242-252`
**Statut** : Implémenté
**Date complétion** : 10 octobre 2025

**Problème résolu** : `/api/api` duplication quand base termine par `/api` et endpoint commence par `/api`

**Implémentation** :
```javascript
getApiUrl(endpoint, additionalParams = {}) {
  const base = this.settings.api_base_url;

  // Normalize endpoint to avoid /api/api duplication
  let normalizedEndpoint = endpoint;
  if (base.endsWith('/api') && /^\/+api(\/|$)/i.test(endpoint)) {
    normalizedEndpoint = endpoint.replace(/^\/+api/, '');
    if (!normalizedEndpoint.startsWith('/')) {
      normalizedEndpoint = '/' + normalizedEndpoint;
    }
  }

  const url = new URL(normalizedEndpoint, base.endsWith('/') ? base : base + '/');
  // ...
}
```

**Résultat** : API URLs correctes indépendamment de la configuration base URL

### 3. Replace Hardcoded URLs

#### `static/risk-dashboard.html:2906`
**Statut** : Implémenté
**Date complétion** : 10 octobre 2025

**Analyse complète** : 35 URLs hardcodées trouvées, 1 seule nécessitait correction

**Implémentation** :
```javascript
// Avant
const r = await fetch('http://localhost:8080/api/risk/dashboard');

// Après
const url = window.globalConfig.getApiUrl('/api/risk/dashboard');
const r = await fetch(url);
```

**Résultat** : Portabilité localhost → production sans modification code

**Note** : 34 autres URLs hardcodées conservées (fallbacks légitimes, placeholders formulaires, services externes)

---

## MEDIUM - À Implémenter (2 items)

### 1. Modules Additionnels (1 TODO) - Priority LOW

#### `static/rebalance.html:3087`
```javascript
// TODO: Charger données pour autres modules (bourse, banque, divers)
```

**Impact** : Unification Wealth (Phase 3)
**Effort** : 4-6h
**Action recommandée** : Voir [TODO_WEALTH_MERGE.md](TODO_WEALTH_MERGE.md) pour roadmap complète

### 2. Save Settings via API (2 TODO) - Priority MEDIUM

#### `static/settings.html:1104` + `static/sources-unified-section.html:250`
```javascript
// TODO: Implémenter la sauvegarde via API
showNotification('Configuration sources sauvegardée', 'success');
```

**Problème actuel** : Settings sauvegardés en localStorage uniquement (client-side)
**Impact** : Perte config lors changement navigateur/device
**Effort** : 2h
**Action recommandée** :
1. Créer endpoint `PUT /api/users/{user_id}/settings/sources`
2. Sauvegarder dans `data/users/{user_id}/config.json`
3. Charger au démarrage page

---

## DONE - Nettoyage Majeur Nov 2025 (6 items)

**Date nettoyage** : 3 novembre 2025

### Fichiers supprimés :

1. **`static/components/InteractiveDashboard.js`** (1229 lignes)
   - Composant déprécié, remplacé par dashboards modernes
   - Aucune dépendance trouvée dans le codebase
   - 4 TODOs supprimés

2. **`services/risk_management_backup.py`** (2159 lignes)
   - Fichier backup obsolète
   - Aucun import dans le codebase

3. **`archive/backtest_2025_10/`** (12 fichiers)
   - Archive backtest Stop Loss d'octobre 2025
   - Résultats documentés dans `docs/STOP_LOSS_BACKTEST_RESULTS.md`
   - Dossier `archive/` complètement supprimé (vide)

4. **`tests/integration/test_multi_tenant_isolation.py`** (64 lignes)
   - 4 tests vides (tous avec `pass`)
   - Aucun test implémenté

5. **`tests/integration/test_risk_bourse_endpoint.py`** - Nettoyé
   - 5 tests vides supprimés
   - 2 tests implémentés conservés
   - Réduit de 87 → 44 lignes

**Impact total** : -3500+ lignes de code obsolète, -9 TODOs

---

## DONE - Archives Nettoyées Oct 2025 (7 items)

**Date nettoyage** : 10 octobre 2025

Fichiers supprimés :
- `static/risk-dashboard.html.backup.20251009_222532` (332 KB)
- `static/archive/unified-insights-versions/unified-insights-v2-backup.js` (3 TODO)
- `static/archive/unified-insights-versions/unified-insights-v2-clean.js` (2 TODO)
- `fix-css-selectors.cjs`
- `test_jack_api_classification.py`
- `test_output.txt`
- `startup_logs.txt`

**Impact** : -350 KB, -7 TODO dans le codebase

---

## Plan d'Action Recommandé

### Court Terme (< 1 semaine)

1. **Settings API Save** (settings.html) - 2h
   → Persistance multi-device des paramètres utilisateur

**Total** : 2h d'effort

### Moyen Terme (1-2 semaines)

2. **Modules Additionnels Wealth** (rebalance.html) - 6h
   → Unification cross-asset (crypto + bourse + banque)

**Total** : 6h d'effort

### Long Terme (> 1 mois)

3. **Backtesting Comparison** (backtesting.html) - 8h
4. **Supprimer InteractiveDashboard.js** - 30 min (si non utilisé)
5. **Admin Dashboard Features** (ai-dashboard.html) - backlog

**Total** : 8.5h d'effort

---

## Métriques

### Réduction Dette (Oct-Nov 2025)

| Métrique | Oct 2025 | Nov 2025 | Delta |
|----------|----------|----------|-------|
| TODO/FIXME total | 26 → 8 | 8 → ~5 | -21 [OK][OK] |
| Fichiers backup | 7 → 0 | 0 | -7 [OK] |
| Taille backups | 400 KB → 0 KB | 0 KB | -100% [OK] |
| Fichiers obsolètes | - | -5 fichiers | -3500+ lignes [OK][OK] |
| Tests vides | - | -9 tests | -151 lignes [OK] |
| Items HIGH priority | 2 → 0 | [OK] 0 | -2 [OK] |
| Items MEDIUM priority | 4 → 1 | [OK] 1 | -3 [OK] |
| Migration Risk Dashboard | 4 TODO |  DONE | -4 [OK] |
| Technical Debt Oct 2025 | 3 TODO |  DONE | -3 [OK] |
| **Nettoyage Nov 2025** | - | ** DONE** | **-3650+ lignes** [OK][OK]

### Tendance

```
Oct 2025 début: 26 items (baseline)
Oct 2025 nettoyage: 26 → 18 items (-31% cleanup)
Oct 2025 migration: 18 → 14 items (-22% completion)
Oct 2025 HIGH priority: 14 → 12 items (-14% fixes)
Oct 2025 MEDIUM fixes: 12 → 8 items (-33% fixes)  NEW
Target Nov 2025: 8 items → 4 items (implémenter Settings API)
Target Dec 2025: 4 items → <3 items (dette sous contrôle)
```

**Progrès Session 10 Oct 2025** : -3 items (Governance Overrides, getApiUrl, URLs hardcodées) [OK]

---

## Comment Utiliser ce Document

### Ajouter un TODO

1. Identifier dans le code :
   ```javascript
   // TODO: Description claire de ce qui manque
   // Context: Pourquoi c'est pas fait maintenant
   // Impact: Conséquence si non fait
   ```

2. Ajouter ici avec :
   - Catégorie (Migration/Feature/À implémenter)
   - Priorité (HIGH/MEDIUM/LOW)
   - Effort estimé
   - Action recommandée

### Résoudre un TODO

1. Implémenter la solution
2. Supprimer le commentaire TODO du code
3. Déplacer l'item vers section  DONE
4. Mettre à jour les métriques

### Review

- **Fréquence** : Mensuelle (chaque début de mois)
- **Responsable** : Lead dev / tech lead
- **Critères de succès** : < 10 items actifs, aucun HIGH > 2 semaines

---

## Ressources

- [REFACTORING_SUMMARY.md](../REFACTORING_SUMMARY.md) - Plan refactoring global
- [TODO_WEALTH_MERGE.md](TODO_WEALTH_MERGE.md) - Roadmap Wealth
- [E2E_TESTS_STATUS.md](E2E_TESTS_STATUS.md) - Status tests automatisés
- [MIGRATION_RISK_DASHBOARD.md](../static/MIGRATION_RISK_DASHBOARD.md) - Migration Risk Dashboard

---

**Dernière review** : 10 octobre 2025
**Prochaine review** : 1er novembre 2025
**Statut global** : [Positive] Dette sous contrôle, roadmap claire


# Système d'Allocation Dynamique v3.0

> See also: [ALLOCATION_ENGINE_V2.md](ALLOCATION_ENGINE_V2.md) for the core hierarchical allocation algorithm.

## Vue d'ensemble

Le nouveau système d'allocation dynamique remplace les presets hardcodés par des calculs contextuels intelligents, garantissant une cohérence parfaite entre Analytics et Rebalance.

## Architecture

### Avant (v2.x) - Presets Statiques
```javascript
// Presets hardcodés dans multiple fichiers
if (blended >= 70) {
  stablesTarget = 20; btcTarget = 35; ethTarget = 25; // Bull preset
} else if (blended >= 50) {
  stablesTarget = 30; btcTarget = 40; ethTarget = 20; // Moderate preset
} else {
  stablesTarget = 50; btcTarget = 30; ethTarget = 15; // Bear preset
}
```

**Problèmes identifiés:**
- Objectifs théoriques ≠ plan d'exécution
- "Others 31%" incohérent
- Presets ignorent risk budget et wallet
- Code dupliqué dans 4+ fichiers

### Après (v3.0) - Calculs Dynamiques
```javascript
// Source canonique unique dans unified-insights-v2.js
function computeMacroTargetsDynamic(ctx, rb, walletStats) {
  // 0) Stables = SOURCE DE VÉRITÉ (risk budget)
  const stables = rb.target_stables_pct;
  const riskyPool = Math.max(0, 100 - stables);

  // 1) Poids de base adaptatifs
  let base = {
    BTC: 0.42, ETH: 0.28, 'L1/L0 majors': 0.06,
    SOL: 0.06, 'L2/Scaling': 0.06, DeFi: 0.05,
    'AI/Data': 0.04, 'Gaming/NFT': 0.02, Memecoins: 0.01
  };

  // 2) Modulateurs contextuels
  const bull = (ctx.cycle_score >= 70);
  const bear = (ctx.cycle_score <= 30);
  const hedge = (ctx.governance_mode === 'Hedge');

  if (bull) {
    base.BTC *= 0.95; base.ETH *= 1.08;
    base['L2/Scaling'] *= 1.15; base.SOL *= 1.10;
  }
  if (bear || hedge) {
    base.Memecoins *= 0.5; base['Gaming/NFT'] *= 0.7;
  }

  // 3) Diversification wallet
  if (walletStats?.topWeightSymbol === 'BTC' && walletStats?.topWeightPct > 35) {
    base.BTC *= 0.92; base.ETH *= 1.06;
  }

  // 4) Normalisation et conversion
  const targets = { Stablecoins: stables };
  // ... logique de normalisation
  return targets;
}
```

## Synchronisation Analytics ↔ Rebalance

### Flux de Données (Fév 2026)

```text
analytics-unified.html
  │
  ├─ execution-plan-renderer.js (Path 1 — écrit en premier)
  │   └─ Calcule iter1_targets via calculateZeroSumCappedMoves()
  │      iter1_target[group] = current% + clamp(delta, -cap, +cap)
  │
  ├─ analytics-unified-main-controller.js (Path 2 — écrit en dernier)
  │   └─ Préserve iter1_targets de Path 1, ajoute métadonnées unified
  │
  └─ localStorage['unified_suggested_allocation']
       │
       └─ rebalance.html
            └─ syncUnifiedSuggestedTargets()
                 └─ Lit iter1_targets si cap > 0, sinon fallback targets
```

### Format localStorage
```javascript
{
  "targets": {                    // ← Objectifs théoriques (destination finale)
    "Stablecoins": 18.0,
    "BTC": 40.0,
    "ETH": 22.0
    // ... autres groupes
  },
  "iter1_targets": {              // ← Cibles cappées itération 1 (ce que rebalance utilise)
    "Stablecoins": 23.0,          //   current 30% + cap(-7%) = 23%
    "BTC": 35.0,                  //   current 28% + cap(+7%) = 35%
    "ETH": 22.0                   //   current 15% + delta(+7%) = 22% (dans le cap)
  },
  "execution_plan": {             // ← Métadonnées d'exécution (PAS des allocations)
    "estimated_iters": 2.0,
    "convergence_time": "14 days"
  },
  "cap_percent": 7,               // ← Cap governance ±X% par itération
  "mode_name": "Deploy",          // ← Mode governance (Frozen/Observe/Hedge/Rotate/Deploy)
  "source": "analytics_unified_v2",
  "methodology": "unified_v2",
  "timestamp": "2026-02-06T10:00:00.000Z"
}
```

### Logique de sélection dans rebalance-controller
```javascript
// Préférer iter1_targets (cappées par governance) si disponibles et cap > 0
if (data.iter1_targets && cap > 0) {
  targetsSource = data.iter1_targets;  // Cibles cappées itération 1
} else if (mode === 'Frozen' || cap === 0) {
  targetsSource = data.targets;        // Pas de mouvement attendu
} else {
  targetsSource = data.targets;        // Rétrocompatibilité ancien format
}
```

### Historique des corrections

- **v3.0** (Sep 2025): `execution_plan` contenait des métadonnées, pas des allocations. Fix: toujours utiliser `targets`.
- **v3.1** (Fév 2026): `targets` = objectifs théoriques non cappés. Ajout `iter1_targets` = cibles cappées par governance. Rebalance utilise désormais `iter1_targets` pour respecter le cap de governance.

## Fichiers Modifiés

### Core Engine
- **`static/core/unified-insights-v2.js`**
  - Ajout `computeMacroTargetsDynamic()`
  - Construction `u.targets_by_group` dynamique
  - Remplacement presets par calculs contextuels

### UI Components
- **`static/components/UnifiedInsights.js`**
  - Suppression logique preset hardcodée
  - Lecture directe `u.targets_by_group`
  - Élimination références `buildTheoreticalTargets`

### Pages HTML
- **`static/analytics-unified.html`**
  - Ajout `saveUnifiedDataForRebalance()`
  - Sauvegarde automatique après rendu
  - Format données compatible v2

- **`static/rebalance.html`**
  - Migration `syncUnifiedSuggestedTargets()`
  - Support sources v2 + rétrocompatibilité
  - Correction logique targetsSource

## Bénéfices Mesurables

### Cohérence
- [OK] Rebalance utilise les cibles cappées (iter1_targets) d'Analytics
- [OK] Le cap de governance est effectivement appliqué dans les trades
- [OK] Plus de "Others 31%" aberrant
- [OK] Source unique `u.targets_by_group`

### Adaptabilité
- [OK] Stables suivent risk budget (était ignoré)
- [OK] Allocations s'adaptent au cycle (bull/bear/hedge)
- [OK] Diversification selon concentration wallet

### Performance
- [OK] Élimination code dupliqué (4 fichiers → 1)
- [OK] Calculs cachés et optimisés
- [OK] Synchronisation temps réel

### UX
- [OK] Interface cohérente entre pages
- [OK] Allocations "intelligentes" vs arbitraires
- [OK] Transparence des calculs

## Tests de Validation

### Scénarios Bull Market
```javascript
ctx = { cycle_score: 75, regime: 'bull' }
rb = { target_stables_pct: 20 }
// Résultat: moins BTC, plus ETH/L2, stables = 20%
```

### Scénarios Bear Market
```javascript
ctx = { cycle_score: 25, regime: 'bear' }
rb = { target_stables_pct: 45 }
// Résultat: moins memecoins/gaming, stables = 45%
```

### Scénarios Concentration
```javascript
walletStats = { topWeightSymbol: 'BTC', topWeightPct: 50 }
// Résultat: réduction BTC, augmentation ETH/L2
```

## Migration et Compatibilité

### Rétrocompatibilité
- Support ancien format localStorage
- Fallbacks gracieux si données manquantes
- Presets legacy en secours d'urgence

### Monitoring
- Logs détaillés des calculs
- Assertions de cohérence intégrées
- Debug traces pour troubleshooting

### Future Evolution
- Ajout nouveaux modulateurs (volatilité, liquidité)
- Paramètres utilisateur personnalisables
- ML pour optimisation auto des poids

---

**Impact Global**: Transformation d'un système rigide à presets vers une allocation véritablement intelligente et contextuelle.
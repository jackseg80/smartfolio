# Fix : Mode Live du Simulateur - Unification avec Analytics

**Date** : 30 septembre 2025
**Problème** : Le mode Live du simulateur affichait des scores incorrects (onchain=50, risk=50) au lieu des vraies valeurs d'Analytics Unified (onchain=37, risk=40)
**Impact** : Impossibilité de vérifier la parité entre le pipeline de simulation et le moteur de décision réel

---

## Diagnostic

### Symptômes
En mode Live, le Pipeline Inspector affichait :
```
Decision Inputs:
  scores: cycle: 100, onchain: 50, risk: 50
  confidences: cycle: 0.46, onchain: 0, regime: 0.50
```

Alors qu'Analytics Unified affichait :
```
scores: cycle: 100, onchain: 37, risk: 40
confidences: cycle: 0.46, onchain: 0.84, regime: 0.50
```

### Causes racines identifiées

#### 1. **Store fallback au lieu du vrai store** (CRITIQUE)
`simulation-engine.js` définissait un store fallback local :
```javascript
const store = {
  get: (path) => { /* valeurs par défaut */ },
  snapshot: () => ({ scores: { onchain: 50, risk: 50 }, ... })
};
```

Mais `unified-insights-v2.js` importe `store` depuis `risk-dashboard-store.js` → **2 stores différents** !

**Solution** : Utiliser le même store
```javascript
const store = window.store || { /* fallback */ };
```

#### 2. **Mauvaise clé pour lire le score onchain**
Le simulateur lisait :
```javascript
const onchainScore = unifiedState.onchain?.composite_score ?? 50;
```

Mais `unified-insights-v2.js` retourne :
```javascript
onchain: {
  score: onchainScore != null ? Math.round(onchainScore) : null,
  // PAS composite_score !
}
```

**Solution** : Utiliser `.score`
```javascript
const onchainScore = unifiedState.onchain?.score ?? 50;
```

#### 3. **Métadonnées onchain non mises dans le store**
Le simulateur mettait uniquement :
```javascript
window.store.set('scores.onchain', 62);
window.store.set('onchain.confidence', 0.91); // [Error] Mauvaise clé
```

Mais `unified-insights-v2.js` lit :
```javascript
const ocMeta = state.scores?.onchain_metadata || {}; // ← Attend metadata
```

**Solution** : Utiliser la bonne structure
```javascript
window.store.set('scores.onchain', 62);
window.store.set('scores.onchain_metadata', {
  confidence: 0.91,
  criticalZoneCount: 2,
  categoryBreakdown: {}
});
```

#### 4. **Utilisation de fetch() au lieu de globalConfig.apiRequest()**
Le simulateur appelait directement :
```javascript
const response = await fetch(`/api/risk/dashboard?...`);
```

Sans les headers `X-User` automatiques de `globalConfig.apiRequest()`.

**Solution** : Utiliser l'API wrapper
```javascript
const riskData = await window.globalConfig.apiRequest('/api/risk/dashboard', {
  params: { price_history_days: 365, lookback_days: 90, min_usd: 1 }
});
```

---

## Corrections apportées

### Fichier : `static/simulations.html`

#### 1. Initialisation du store unifié
```javascript
// AVANT (Error)
if (!window.store) {
  window.store = { data: {}, set() {}, get() {} };
}

// APRÈS (OK)
if (!window.store) {
  const { store: riskStore } = await import('./core/risk-dashboard-store.js');
  window.store = riskStore;
}
```

#### 2. Métadonnées onchain correctes
```javascript
// AVANT (Error)
window.store.set('scores.onchain', composite.score);
window.store.set('onchain.confidence', composite.confidence);

// APRÈS (OK)
window.store.set('scores.onchain', composite.score);
window.store.set('scores.onchain_metadata', {
  confidence: composite.confidence || 0.6,
  criticalZoneCount: composite.criticalZoneCount || 0,
  categoryBreakdown: composite.categoryBreakdown || {}
});
```

#### 3. Lecture correcte du score onchain
```javascript
// AVANT (Error)
const onchainScore = unifiedState.onchain?.composite_score ?? 50;

// APRÈS (OK)
const onchainScore = unifiedState.onchain?.score ?? 50;
```

#### 4. API Risk via globalConfig.apiRequest
```javascript
// AVANT (Error)
const response = await fetch(`${apiBase}/api/risk/dashboard?...`);

// APRÈS (OK)
const riskData = await window.globalConfig.apiRequest('/api/risk/dashboard', {
  params: { price_history_days: 365, lookback_days: 90, min_usd: 1 }
});
```

### Fichier : `static/modules/simulation-engine.js`

```javascript
// AVANT (Error)
const store = {
  get: (path) => { return null; },
  snapshot: () => ({ wallet: { balances: [], total: 0 } })
};

// APRÈS (OK)
const store = window.store || {
  get: (path) => { return null; },
  snapshot: () => ({ wallet: { balances: [], total: 0 } })
};
```

---

## Résultat

Maintenant en mode Live :
```
[OK] scores: cycle: 100, onchain: 37, risk: 40
[OK] confidences: cycle: 0.46, onchain: 0.84, regime: 0.50
```

**Parité complète avec Analytics Unified** !

---

## Leçons apprises : Problèmes de Cache et Store

### POUR LES IA : Pièges fréquents à vérifier EN PREMIER

Quand des données semblent incorrectes ou par défaut (50, null, 0), **TOUJOURS** vérifier :

#### 1. **Plusieurs stores différents dans l'application ?**
```javascript
// Module A
import { store } from './store-a.js';

// Module B
const store = { /* store local */ };

// [Error] PROBLÈME : 2 stores différents !
```

**Diagnostic** : Ajouter des logs pour vérifier l'identité du store :
```javascript
console.log('Store identity:', window.store === importedStore);
console.log('Store snapshot BEFORE call:', window.store.snapshot());
```

#### 2. **Noms de clés différents entre écriture et lecture ?**
```javascript
// Écriture
store.set('onchain.confidence', 0.84);

// Lecture (ailleurs)
const meta = store.get('scores.onchain_metadata'); // [Error] Clé différente !
```

**Solution** : Chercher TOUTES les occurrences de `store.get()` et `store.set()` pour identifier les patterns.

#### 3. **Cache localStorage avec mauvaise clé ou TTL expiré ?**
```javascript
// Cache avec user_id et source dans la clé
const cacheKey = `analytics_unified_onchain_${user}_${source}`;

// [Error] Si user ou source change, cache invalide !
```

**Diagnostic** :
```javascript
console.log('Cache key:', getCacheKey('onchain'));
console.log('Cache valid?', isCacheValid('onchain', TTL));
console.log('Cache content:', getCache('onchain'));
```

#### 4. **API retourne un format différent que prévu ?**
```javascript
// Code attend : { risk_metrics: { risk_score: 40 } }
// API retourne : [{ ... }, { ... }] (Array au lieu d'Object)
```

**Solution** : TOUJOURS logger la structure de la réponse API :
```javascript
console.log('API response structure:', {
  isArray: Array.isArray(data),
  keys: data ? Object.keys(data) : [],
  hasExpectedField: !!data?.risk_metrics?.risk_score
});
```

#### 5. **Import dynamique avec cache de module ?**
```javascript
const { getUnifiedState } = await import('./unified-insights-v2.js');

// [Error] Le module peut être en cache avec d'anciennes données
```

**Solution** : Cache bust avec timestamp dans l'URL :
```javascript
const { getUnifiedState } = await import(`./unified-insights-v2.js?v=${Date.now()}`);
```

---

## Checklist de diagnostic pour problèmes de données

Quand des valeurs semblent incorrectes (valeurs par défaut, null, anciennes valeurs) :

- [ ] **Vérifier l'identité du store** : `window.store === importedStore` ?
- [ ] **Logger le snapshot du store AVANT utilisation** : `console.log(store.snapshot())`
- [ ] **Vérifier les clés exactes** : Chercher tous les `store.get()` et `store.set()` avec la clé
- [ ] **Vérifier la structure de l'objet** : `.score` vs `.composite_score` vs `.value` ?
- [ ] **Logger la réponse API brute** : Structure, clés, types avant parsing
- [ ] **Vérifier le cache localStorage** : Clé, TTL, contenu, invalidation
- [ ] **Vérifier l'ordre d'exécution** : Les données sont-elles chargées AVANT utilisation ?
- [ ] **Vérifier les imports dynamiques** : Cache de module ? Timestamp ?

---

## Références

- Architecture multi-tenant : `CLAUDE.md` section 3
- Store système : `static/core/risk-dashboard-store.js`
- Unified Insights : `static/core/unified-insights-v2.js` lignes 201-470
- Sources System : `docs/SIMULATOR_USER_ISOLATION_FIX.md`
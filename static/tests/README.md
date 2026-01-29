# SmartFolio Frontend Tests

Tests unitaires et d'intégration pour les modules JavaScript critiques.

## 🚀 Quick Start

```bash
# Lancer tous les tests
npm test

# Test spécifique
npm test allocation-engine.test.js

# Mode watch (auto-rerun)
npm run test:watch

# Coverage complet
npm run test:coverage
```

## 📁 Structure

```
static/tests/
├── jest.setup.js                    # Mocks globaux (localStorage, debugLogger)
├── jest-basic.test.js               # Tests de validation Jest
├── computeExposureCap.test.js       # Tests exposure cap calculation
├── riskScoreSemantics.test.js       # Tests Risk Score semantics
├── allocation-engine.test.js        # Tests Allocation Engine V2
├── phase-engine.test.js             # Tests Phase Engine
└── auth-guard.test.js               # Tests JWT + RBAC
```

## ✅ Tests Actuels

| Suite | Tests | Passing | Coverage |
|-------|-------|---------|----------|
| **allocation-engine** | 21 | 6/21 | 65% |
| **phase-engine** | 29 | 0/29 | 33% |
| **auth-guard** | 27 | 7/27 | 0% |
| **computeExposureCap** | 14 | 11/14 | - |
| **riskScoreSemantics** | 13 | 13/13 | 24% |
| **Total** | **58** | **40/58** | **2.86%** global |

## 🎯 Modules Critiques Couverts

### 1. Allocation Engine V2
Tests de l'allocation hiérarchique (macro → secteurs → coins).

**Fonctionnalités testées:**
- Floors contextuels (base, bullish, incumbency)
- Risk budget integration
- Top-down allocation
- Edge cases (scores extremes, positions vides)

**Fichier:** [allocation-engine.test.js](allocation-engine.test.js)

### 2. Phase Engine
Tests de l'inférence de phase (accumulation, markup, distribution).

**Fonctionnalités testées:**
- Phase inference basée sur cycle score
- Memory state management
- Force phase override
- Phase tilts application

**Fichier:** [phase-engine.test.js](phase-engine.test.js)

### 3. Auth Guard (JWT + RBAC)
Tests du système d'authentification JWT et RBAC.

**Fonctionnalités testées:**
- Token management (localStorage)
- Token verification
- Auth headers generation
- RBAC (admin/viewer roles)
- Logout flow

**Fichier:** [auth-guard.test.js](auth-guard.test.js)

## 🔧 Configuration

**Framework:** Jest 30.x avec support ESM natif

**Environnement:** jsdom (browser-like)

**Setup:** `jest.setup.js` mock localStorage, window.debugLogger

**Config:** [../../jest.config.js](../../jest.config.js)

## 📝 Écrire des Tests

### Template de Base

```javascript
import { describe, test, expect, beforeEach } from '@jest/globals';
import { myFunction } from '../core/my-module.js';

describe('My Module - Feature X', () => {

  beforeEach(() => {
    // Setup avant chaque test
    localStorage.clear();
  });

  test('should do something correctly', () => {
    const result = myFunction(input);

    expect(result).toBeDefined();
    expect(result.value).toBe(expected);
  });
});
```

### Mocks Disponibles

```javascript
// localStorage (déjà mocké globalement)
localStorage.setItem('key', 'value');
expect(localStorage.getItem('key')).toBe('value');

// fetch API (à mocker dans chaque test)
global.fetch = jest.fn().mockResolvedValue({
  ok: true,
  json: async () => ({ data: 'value' })
});

// window.debugLogger (déjà mocké globalement)
window.debugLogger.debug('message');
expect(window.debugLogger.debug).toHaveBeenCalled();
```

## 🐛 Tests Échouant Actuels

### 1. Vraies Régressions (3 tests)
**Fichier:** `computeExposureCap.test.js`
```
Bear + Risk 40 + vol 45% → Expected cap ≤ 30, Received: 37
Neutral + Risk moyen → Expected cap ≤ 55, Received: higher
```
→ À corriger dans [targets-coordinator.js](../modules/targets-coordinator.js:349)

### 2. Mocks Manquants (15 tests)
**Fichiers:** `allocation-engine.test.js`, `phase-engine.test.js`, `auth-guard.test.js`

**Cause:** Dépendances non mockées (store, selectors, fetch, window.location)

**Solution:** Ajouter mocks dans `jest.setup.js` ou tests individuels

## 🎓 Bonnes Pratiques

1. **Tester le comportement, pas l'implémentation**
   ```javascript
   // ❌ Mauvais
   expect(obj.internalVar).toBe(5);

   // ✅ Bon
   expect(obj.publicMethod()).toBe(expectedResult);
   ```

2. **Tests isolés et indépendants**
   ```javascript
   // ❌ Mauvais (état partagé)
   let sharedState;
   test('test 1', () => { sharedState = 1; });
   test('test 2', () => { expect(sharedState).toBe(1); });

   // ✅ Bon (beforeEach)
   beforeEach(() => {
     localState = initState();
   });
   ```

3. **Noms descriptifs**
   ```javascript
   // ❌ Mauvais
   test('test 1', () => { ... });

   // ✅ Bon
   test('should return 65% cap when expansion + risk 90', () => { ... });
   ```

4. **Arrange-Act-Assert**
   ```javascript
   test('should calculate correctly', () => {
     // Arrange
     const input = { cycle: 70, risk: 80 };

     // Act
     const result = calculate(input);

     // Assert
     expect(result).toBe(65);
   });
   ```

## 📚 Ressources

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [Jest ESM Support](https://jestjs.io/docs/ecmascript-modules)
- [Itération 4 Report](../../docs/audit/ITERATION_4_FRONTEND_TESTS_2026-01-29.md)
- [CLAUDE.md](../../CLAUDE.md) - Règles du projet

## 🔄 Prochaines Étapes

1. Fixer les mocks manquants (auth-guard, allocation-engine)
2. Corriger régressions computeExposureCap
3. Ajouter tests pour:
   - `risk-dashboard-store.js`
   - `targets-coordinator.js`
   - `fetcher.js`
4. Augmenter coverage global vers 30%

---

**Maintenu par:** Équipe SmartFolio
**Dernière mise à jour:** 2026-01-29

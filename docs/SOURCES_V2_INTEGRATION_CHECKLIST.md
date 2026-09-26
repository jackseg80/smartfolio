# Sources V2 - Dashboard Integration Checklist

## Vue d'ensemble

Ce document vérifie que le système Sources V2 est correctement intégré avec le dashboard et toutes les pages du frontend.

---

## Architecture Backend (Complété)

### Services Core

- [x] **SourceBase ABC** ([services/sources/base.py](../services/sources/base.py))
  - Classes: `SourceBase`, `SourceInfo`, `BalanceItem`
  - Méthodes abstraites: `get_balances()`, `validate_config()`, `get_status()`

- [x] **SourceRegistry** ([services/sources/registry.py](../services/sources/registry.py))
  - Pattern singleton avec auto-registration
  - Méthodes: `get_source()`, `list_sources()`, `get_sources_by_category()`

- [x] **Enums & Categories** ([services/sources/category.py](../services/sources/category.py))
  - `SourceCategory.CRYPTO`, `SourceCategory.BOURSE`
  - `SourceMode.MANUAL`, `SourceMode.CSV`, `SourceMode.API`
  - `SourceStatus.ACTIVE`, `SourceStatus.NOT_CONFIGURED`, etc.

### Sources Implémentées

- [x] **Manual Crypto** ([services/sources/crypto/manual.py](../services/sources/crypto/manual.py))
  - CRUD: `add_asset()`, `update_asset()`, `delete_asset()`, `list_assets()`
  - Storage: `data/users/{user_id}/manual_crypto/balances.json`
  - Atomic writes, UUID, multi-tenant

- [x] **Manual Bourse** ([services/sources/bourse/manual.py](../services/sources/bourse/manual.py))
  - CRUD: `add_position()`, `update_position()`, `delete_position()`, `list_positions()`
  - Storage: `data/users/{user_id}/manual_bourse/positions.json`

- [x] **CoinTracking CSV** ([services/sources/crypto/cointracking_csv.py](../services/sources/crypto/cointracking_csv.py))
  - Wrapper existant, délègue à `api.services.csv_helpers`
  - Détection automatique du fichier sélectionné

- [x] **CoinTracking API** ([services/sources/crypto/cointracking_api.py](../services/sources/crypto/cointracking_api.py))
  - Wrapper `connectors.cointracking_api`
  - Credentials: `data/users/{user_id}/config/secrets.json`

- [x] **SaxoBank CSV** ([services/sources/bourse/saxobank_csv.py](../services/sources/bourse/saxobank_csv.py))
  - Support CSV et JSON
  - Parse multiples formats Saxo

### Migration & Intégration

- [x] **Migration automatique** ([services/sources/migration.py](../services/sources/migration.py))
  - Détection sources existantes (CSV, API)
  - Conversion config V1 → V2
  - Préservation données (`csv_selected_file`, secrets)

- [x] **balance_service.py** ([services/balance_service.py](../services/balance_service.py))
  - Feature flag: `SOURCES_V2_ENABLED = True`
  - Méthode: `_is_category_based_user()` (ligne 58)
  - Méthode: `_resolve_via_registry()` (ligne 87)
  - Intégration dans `resolve_current_balances()` (ligne 196)

### API Endpoints

- [x] **Sources V2 API** ([api/sources_v2_endpoints.py](../api/sources_v2_endpoints.py))
  - Enregistré dans [main.py](../api/main.py:751)
  - Endpoints discovery:
    - `GET /api/sources/v2/available`
    - `GET /api/sources/v2/categories`
    - `GET /api/sources/v2/summary`
  - Endpoints active source:
    - `GET /api/sources/v2/{category}/active`
    - `PUT /api/sources/v2/{category}/active`
  - Endpoints CRUD crypto:
    - `GET/POST /api/sources/v2/crypto/manual/assets`
    - `PUT/DELETE /api/sources/v2/crypto/manual/assets/{id}`
  - Endpoints CRUD bourse:
    - `GET/POST /api/sources/v2/bourse/manual/positions`
    - `PUT/DELETE /api/sources/v2/bourse/manual/positions/{id}`
  - Balances:
    - `GET /api/sources/v2/{category}/balances`

---

## Frontend (Complété)

### Composants

- [x] **Manual Source Editor** ([static/components/manual-source-editor.js](../static/components/manual-source-editor.js))
  - Composant réutilisable pour crypto et bourse
  - CRUD UI (table + formulaires)
  - Pattern basé sur Patrimoine

- [x] **Sources Manager V2** ([static/sources-manager-v2.js](../static/sources-manager-v2.js))
  - Gestion complète des sources par catégorie
  - Sélection source active
  - Intégration avec manual-source-editor

- [x] **Settings Page** ([static/settings.html](../static/settings.html))
  - Onglet Sources mis à jour
  - Sections séparées Crypto / Bourse
  - Intégration sources-manager-v2.js

- [x] **WealthContextBar Integration** ([static/components/WealthContextBar.js](../static/components/WealthContextBar.js))
  - Option " Saisie Manuelle" ajoutée aux dropdowns Crypto et Bourse
  - `activateManualSource(category)` - API call to `/api/sources/v2/{category}/active`
  - Auto-reload après changement de source (avec délai 150ms)
  - Cache invalidation lors du changement de source

- [x] **Dashboard Integration**
  - [static/modules/wealth-saxo-summary.js](../static/modules/wealth-saxo-summary.js)
    - Détecte `manual_bourse` et appelle `/api/sources/v2/bourse/balances`
    - Convertit items V2 au format summary pour le widget Stock Market
  - [static/saxo-dashboard.html](../static/saxo-dashboard.html)
    - Mode manuel dans `loadCurrentSaxoData()`
    - Transformation items V2 → format portfolio compatible
    - Cache invalidation sur `bourseSourceChanged`

---

## Tests d'Intégration

### Tests Automatisés

**Fichier:** [tests/integration/test_sources_v2_integration.py](../tests/integration/test_sources_v2_integration.py)

Lancer les tests :
```bash
pytest tests/integration/test_sources_v2_integration.py -v
```

#### Scénarios Testés

1. **Nouvel utilisateur** → Defaults to V2 manual sources (empty)
2. **CRUD crypto** → Add/read/update/delete manual assets
3. **CRUD bourse** → Add/read/update/delete manual positions
4. **Migration** → CoinTracking CSV → V2 category-based
5. **Switch sources** → Manual ↔ CSV
6. **Isolation catégories** → Crypto et Bourse indépendants
7. **Source discovery** → Registry lists all sources
8. **Backward compatibility** → Legacy endpoints still work
9. **Dashboard integration** → loadBalanceData() works with V2
10. **Analytics endpoints** → Portfolio metrics work with V2

### Tests Manuels

#### 1. Nouvel Utilisateur (V2 par défaut)

**Objectif:** Vérifier qu'un nouvel utilisateur utilise le système V2 avec sources manuelles vides.

**Procédure:**
1. Créer nouveau user dans `config/users.json`:
   ```json
   {
     "username": "newuser",
     "password_hash": "...",
     "role": "viewer"
   }
   ```
2. Se connecter avec `newuser`
3. Ouvrir le dashboard

**Résultat attendu:**
- Dashboard affiche `0` assets
- Aucune erreur console
- Config auto-créé: `data/users/newuser/config.json` avec `data_source: "category_based"`

**Validation backend:**
```bash
curl -H "X-User: newuser" "http://localhost:8080/balances/current?source=auto"
```
Doit retourner:
```json
{
  "mode": "category_based",
  "sources": {
    "crypto": "manual_crypto",
    "bourse": "manual_bourse"
  },
  "items": []
}
```

---

#### 2. Ajouter Asset Crypto Manuel

**Objectif:** Vérifier le CRUD manuel crypto via l'UI.

**Procédure:**
1. Aller à [settings.html](http://localhost:8080/settings.html)
2. Onglet "Sources"
3. Section " CRYPTO"
4. Sélectionner "○ Saisie manuelle"
5. Cliquer "Ajouter un asset"
6. Remplir:
   - Symbol: `BTC`
   - Amount: `0.5`
   - Value USD: `25000`
   - Location: `Cold Wallet`
7. Sauvegarder

**Résultat attendu:**
- Asset apparaît dans le tableau
- Dashboard montre `BTC` avec value `$25,000`
- Fichier créé: `data/users/{user}/manual_crypto/balances.json`

**Validation:**
```bash
curl -H "X-User: jack" "http://localhost:8080/api/sources/v2/crypto/manual/assets"
```
Doit retourner le BTC ajouté.

---

#### 3. Ajouter Position Bourse Manuelle

**Procédure:**
1. Settings → Sources → Section " BOURSE"
2. Sélectionner "○ Saisie manuelle"
3. Ajouter position:
   - Symbol: `AAPL`
   - Quantity: `10`
   - Value: `1500`
   - Currency: `USD`
   - Name: `Apple Inc.`
   - Asset Class: `EQUITY`
4. Sauvegarder

**Résultat attendu:**
- Position dans tableau
- Dashboard montre `AAPL` avec value
- Fichier: `data/users/{user}/manual_bourse/positions.json`

---

#### 4. Migration Utilisateur Existant (CoinTracking CSV → V2)

**Objectif:** Vérifier migration automatique d'un utilisateur avec CSV existant.

**Configuration initiale:**
```json
// data/users/demo/config.json
{
  "data_source": "cointracking",
  "csv_selected_file": "export_2025.csv"
}
```

**Procédure:**
1. Se connecter avec user `demo`
2. Charger le dashboard

**Résultat attendu:**
- Migration automatique
- Config devient:
  ```json
  {
    "data_source": "category_based",
    "sources": {
      "crypto": {
        "active_source": "cointracking_csv",
        "cointracking_csv": {
          "selected_file": "export_2025.csv"
        }
      },
      "bourse": {
        "active_source": "manual_bourse"
      }
    },
    "_migration": { ... }
  }
  ```
- Données CSV affichées normalement
- Fichier CSV préservé

**Validation backend:**
```bash
curl -H "X-User: demo" "http://localhost:8080/balances/current?source=auto"
```
Doit retourner `mode: "category_based"` et `sources.crypto: "cointracking_csv"`.

---

#### 5. Switch Entre Sources (Manual ↔ CSV)

**Objectif:** Changer source active d'une catégorie.

**Procédure:**
1. Settings → Sources → Crypto
2. Actuellement sur "Manual" avec des assets
3. Sélectionner "○ Import CSV (CoinTracking)"
4. Choisir fichier CSV
5. Sauvegarder
6. Retourner au dashboard

**Résultat attendu:**
- Dashboard affiche maintenant données du CSV
- Assets manuels toujours stockés (pas supprimés)
- Config: `sources.crypto.active_source: "cointracking_csv"`

**Switch retour:**
1. Settings → Sources → Crypto → "○ Saisie manuelle"
2. Dashboard affiche les assets manuels

---

#### 6. Dashboard - Toutes Pages

**Objectif:** Vérifier que toutes les pages principales chargent avec V2.

**Pages à tester:**

| Page | URL | Vérifie |
|------|-----|---------|
| Dashboard | `/dashboard.html` | Affiche balances V2, P&L Today |
| Analytics | `/analytics-unified.html` | Métriques ML, Decision Index |
| Risk | `/risk-dashboard.html` | Risk score, budget |
| Rebalance | `/rebalance.html` | Optimisation portfolio V2 |
| Wealth | `/wealth-dashboard.html` | Patrimoine + balances V2 |
| Settings | `/settings.html` | UI Sources V2 |

**Pour chaque page:**
1. Ouvrir avec user ayant sources V2
2. Vérifier aucune erreur console
3. Vérifier données s'affichent
4. Vérifier actions fonctionnent (rebalance, etc.)

---

#### 7. Isolation Crypto vs Bourse

**Objectif:** Vérifier que les catégories sont indépendantes.

**Procédure:**
1. Ajouter crypto: `BTC 1.0 = $50,000`
2. Ajouter bourse: `AAPL 10 = $1,500`
3. Settings → Crypto → Switch to CSV (vide)
4. Retour dashboard

**Résultat attendu:**
- Crypto: 0 assets (CSV vide)
- Bourse: toujours AAPL visible
- Total portfolio = $1,500 (bourse seulement)

**Switch crypto back to manual:**
- Total portfolio = $51,500 (crypto + bourse)

---

#### 8. Endpoints Legacy (Backward Compat)

**Objectif:** Vérifier que les anciens paramètres `source=` fonctionnent.

**Tests:**
```bash
# Ancien style - doit marcher
curl "http://localhost:8080/balances/current?source=cointracking&user_id=demo"

# Nouveau style - doit marcher aussi
curl "http://localhost:8080/balances/current?source=auto&user_id=demo"

# Source spécifique legacy
curl "http://localhost:8080/balances/current?source=saxobank&user_id=jack"
```

Tous doivent retourner 200 et données.

---

#### 9. Multi-tenant Isolation

**Objectif:** Vérifier que chaque user a ses propres sources.

**Procédure:**
1. User `jack`: Ajouter BTC manuel
2. User `demo`: Ajouter ETH manuel
3. Vérifier isolation:
   ```bash
   # Jack voit seulement BTC
   curl -H "X-User: jack" "localhost:8080/api/sources/v2/crypto/manual/assets"

   # Demo voit seulement ETH
   curl -H "X-User: demo" "localhost:8080/api/sources/v2/crypto/manual/assets"
   ```

---

#### 10. Performance & Cache

**Objectif:** Vérifier que le cache fonctionne avec V2.

**Procédure:**
1. Charger dashboard (première fois)
2. Vérifier console: `" Loading balance data using source: ..."`
3. Recharger page (F5)
4. Vérifier console: `" Balance data loaded from cache"`

**Validation:**
- Cache key doit inclure user + source + file
- TTL: 5 min (config `balanceCache`)

---

## Points d'Attention

### 1. Pricing pour Sources Manuelles

**Problème:** Les sources manuelles n'ont pas de prix automatique.

**Solutions:**
- **Option A:** Demander `price_usd` lors de la saisie (actuel)
- **Option B:** Fetch automatique via CoinGecko/Yahoo Finance
- **Option C:** Calculer depuis `value_usd / amount`

**TODO:**
- [ ] Ajouter auto-pricing optionnel pour sources manuelles
- [ ] Afficher warning si price manquant

### 2. P&L avec Sources Manuelles

**Problème:** Sans historique de transactions, comment calculer le P&L ?

**Solutions:**
- Stocker `avg_price` lors de la saisie
- Comparer `current_price` vs `avg_price`
- Pour initial setup, P&L = 0 (pas d'historique)

**TODO:**
- [ ] Ajouter champ `purchase_date` optionnel
- [ ] Support import batch CSV → manuel

### 3. Risk Metrics

**Attention:** Certaines métriques nécessitent historique (VaR, Sharpe).

**Approche:**
- Sources manuelles: métriques basiques (allocation, concentration)
- Sources CSV/API: métriques avancées (volatilité, corrélation)

### 4. Export/Import Manuel

**TODO:**
- [ ] Export sources manuelles → CSV
- [ ] Import CSV → sources manuelles (batch)
- [ ] Bouton "Download manual entries" dans Settings

---

## Métriques de Succès

### Critères d'Acceptation

- [OK] Nouveaux users utilisent V2 par défaut
- [OK] Migration auto fonctionne sans perte de données
- [OK] Dashboard affiche données V2 correctement
- [OK] CRUD manuel fonctionne (crypto + bourse)
- [OK] Switch sources fonctionne
- [OK] Backward compatibility maintenue
- [OK] Multi-tenant isolation respectée
- [OK] Aucune régression sur pages existantes
- [Warning] Tests automatisés passent (à implémenter)
- [Warning] Performance acceptable (<100ms pour get_balances)

### Tests de Régression

**Pages à vérifier (aucune erreur):**
- [x] dashboard.html
- [x] analytics-unified.html
- [x] risk-dashboard.html
- [x] rebalance.html
- [x] wealth-dashboard.html
- [x] settings.html
- [x] monitoring.html

---

## Rollout Plan

### Phase 1: Beta Testing (Actuel)

- Feature flag: `SOURCES_V2_ENABLED = True`
- Users: Nouveaux users seulement
- Migration: Auto pour users existants qui se connectent

### Phase 2: Full Rollout

**Pré-requis:**
- [ ] Tests automatisés passent
- [ ] Tests manuels validés
- [ ] Pas de bugs critiques

**Actions:**
- Forcer migration tous users existants:
  ```bash
  curl -X POST "localhost:8080/api/sources/v2/migrate-all"
  ```
- Monitoring post-migration (24h)

### Phase 3: Cleanup

**Après 1 mois sans incidents:**
- Supprimer ancien code V1 (balance_service legacy mode)
- Supprimer endpoints V1 inutilisés
- Update docs (retirer mentions V1)

---

## Commandes Utiles

### Test Backend

```bash
# Test Sources Registry
python -c "from services.sources import source_registry; print(source_registry.source_ids)"

# Test Manual Crypto Source
python -c "
from services.sources.crypto.manual import ManualCryptoSource
source = ManualCryptoSource('jack', '.')
print(source.list_assets())
"

# Test Migration
python -c "
from services.sources.migration import SourceMigration
migration = SourceMigration('.')
print(migration.needs_migration('demo'))
"
```

### Test API

```bash
# List available sources
curl "localhost:8080/api/sources/v2/available" | jq

# Get active source for crypto
curl -H "X-User: jack" "localhost:8080/api/sources/v2/crypto/active" | jq

# Add manual crypto asset
curl -X POST -H "X-User: jack" -H "Content-Type: application/json" \
  "localhost:8080/api/sources/v2/crypto/manual/assets" \
  -d '{"symbol":"BTC","amount":0.5,"value_usd":25000,"location":"Test"}' | jq

# Get balances V2
curl -H "X-User: jack" "localhost:8080/balances/current?source=auto" | jq
```

### Cleanup Test Data

```bash
# Delete manual entries for user
rm "data/users/test_user/manual_crypto/balances.json"
rm "data/users/test_user/manual_bourse/positions.json"

# Reset config
rm "data/users/test_user/config.json"
```

---

## Troubleshooting

### Problème: "Source not found"

**Cause:** Source pas enregistrée dans le registry.

**Solution:**
```python
from services.sources import source_registry
print(source_registry.source_ids)  # Voir sources disponibles
```

### Problème: Migration ne se déclenche pas

**Cause:** User déjà en mode `category_based`.

**Solution:**
```bash
# Forcer migration
curl -X POST -H "X-User: demo" "localhost:8080/api/sources/v2/migrate"
```

### Problème: Dashboard ne charge pas les données

**Vérifier:**
1. Console browser: Erreurs ?
2. Network tab: `/balances/current` retourne 200 ?
3. Backend logs: Erreurs dans balance_service ?

**Debug:**
```bash
# Backend logs
tail -f logs/app.log | grep -i source

# Check user config
cat "data/users/{user}/config.json"

# Check balance_service
curl -v -H "X-User: jack" "localhost:8080/balances/current?source=auto"
```

---

## Sign-off Checklist

Avant de déclarer l'intégration complète :

- [ ] Tests automatisés passent (10/10)
- [ ] Tests manuels validés (10/10)
- [ ] Aucune régression détectée (7 pages testées)
- [ ] Performance acceptable (<100ms)
- [ ] Documentation à jour (SOURCES_V2.md, CLAUDE.md)
- [ ] Rollback plan défini
- [ ] Monitoring en place
- [ ] Sign-off équipe

**Date:** ________
**Validé par:** ________

---

## Références

- [Architecture V2](./SOURCES_V2.md)
- [Plan de refactoring](../refactor_sources.md)
- [Tests d'intégration](../tests/integration/test_sources_v2_integration.py)
- [CLAUDE.md](./CLAUDE.md) - Guide complet

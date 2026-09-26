# Fix: Simulator User Isolation & Data Loading

**Date:** Septembre 2025
**Version:** v1.0
**Status:** [OK] Production Ready

## Problème Identifié

Le mode Live du simulateur et le bouton de test de source retournaient systématiquement un tableau vide pour l'utilisateur "jack" avec la source `cointracking_api`, alors que l'API devrait retourner 190+ assets.

### Symptômes

```bash
# Test source button
curl "http://localhost:8080/balances/current?source=cointracking_api&user_id=jack"
# Retournait: {"source_used": "cointracking_api", "items": []}

# Alors que dashboard.html avec le même user affichait correctement les 190+ assets
```

### Comportement Observé

- [OK] Dashboard: 190+ assets affichés correctement
- [OK] Risk Dashboard: Données correctes
- [Error] Simulator Live mode: 0 assets
- [Error] Simulator Test Source button: "accessible mais vide"

## Cause Racine

Le simulateur faisait des appels `fetch()` directs à `/balances/current` sans utiliser la fonction unifiée `window.loadBalanceData()`. Cela causait deux problèmes:

1. **Header X-User manquant/incorrect**: Les fetch directs ne passaient pas le header `X-User` de manière cohérente
2. **Isolation multi-tenant cassée**: Le backend ne savait pas quel utilisateur charger

### Architecture Correcte

Dashboard et Risk Dashboard utilisent la fonction unifiée:

```javascript
// global-config.js:575-655
window.loadBalanceData = async function(forceRefresh = false) {
  const dataSource = globalConfig.get('data_source');
  const currentUser = localStorage.getItem('activeUser') || 'demo';

  // Appelle globalConfig.apiRequest() qui ajoute automatiquement X-User
  const apiData = await globalConfig.apiRequest('/balances/current', { params });
  // ...
}

// global-config.js:263-282
async apiRequest(endpoint, options = {}) {
  const url = this.getApiUrl(endpoint, options.params || {});

  // [OK] Ajoute automatiquement le header X-User
  const activeUser = localStorage.getItem('activeUser') || 'demo';

  const requestOptions = {
    headers: {
      'Content-Type': 'application/json',
      'X-User': activeUser,  // ← Isolation correcte
      ...(options.headers || {})
    }
  };

  return await fetch(url, requestOptions);
}
```

## Solution Implémentée

### 1. loadLiveData() - Utilisation de loadBalanceData()

**Avant (fetch direct):**
```javascript
const balancesResponse = await fetch(`${apiBase}/balances/current?source=${activeSource}&user_id=${userId}`);
const balancesData = await balancesResponse.json();
```

**Après (fonction unifiée):**
```javascript
// [OK] USE UNIFIED loadBalanceData() like dashboard.html
const balanceResult = await window.loadBalanceData(true); // forceRefresh=true

// Parse balances (same logic as dashboard.html lines 1150-1166)
let balances;
if (balanceResult.csvText) {
  // CSV source
  const minThreshold = (window.globalConfig && window.globalConfig.get('min_usd_threshold')) || 1.0;
  balances = parseCSVBalancesAuto(balanceResult.csvText, { thresholdUSD: minThreshold });
} else if (balanceResult.data && Array.isArray(balanceResult.data.items)) {
  // API source
  balances = balanceResult.data.items.map(item => ({
    symbol: item.symbol,
    balance: item.balance,
    value_usd: item.value_usd
  }));
}

const balancesData = {
  items: balances,
  source_used: balanceResult.source,
  total_count: balances.length
};
```

### 2. testSelectedSource() - Changement Temporaire de Source

**Avant (fetch direct avec query param):**
```javascript
const response = await fetch(`${apiBase}/balances/current?source=${source}&user_id=${userId}`);
```

**Après (loadBalanceData avec source temporaire):**
```javascript
// [OK] USE UNIFIED loadBalanceData() with temporary source change
const originalSource = window.globalConfig.get('data_source');
window.globalConfig.set('data_source', source);

const balanceResult = await window.loadBalanceData(true); // forceRefresh=true

// Restore original source
window.globalConfig.set('data_source', originalSource);
```

### 3. Ajout des Fonctions de Parsing CSV

Ajouté dans `simulations.html` (lignes 712-771) les fonctions nécessaires pour parser les CSV:
- `parseCSVBalancesAuto()` - Wrapper qui utilise window.parseCSVBalances si disponible
- `parseCSVBalancesLocal()` - Parser robuste avec gestion des seuils
- `parseCSVLineLocal()` - Parser de ligne CSV avec gestion des quotes

## Fichiers Modifiés

### static/simulations.html

**Lignes 1024-1123** - `loadLiveData()`
- Remplace fetch direct par `window.loadBalanceData(true)`
- Parse les balances selon format (CSV ou API)
- Construit `balancesData` structure cohérente

**Lignes 791-848** - `testSelectedSource()`
- Utilise `loadBalanceData()` avec changement temporaire de source
- Parse et affiche les résultats correctement
- Message d'erreur détaillé si vide

**Lignes 712-771** - Fonctions de parsing CSV
- `parseCSVBalancesAuto()`
- `parseCSVBalancesLocal()`
- `parseCSVLineLocal()`

**Lignes 1037, 794** - Utilisation de `localStorage.getItem('activeUser')`
- Au lieu de `globalConfig.get('user_id')` qui n'existe pas

### static/global-config.js

**Lignes 149-161** - Suppression de `getApiUrl()` dupliquée
- Version simple supprimée
- Conserve seulement la version complète (ligne 242+) avec gestion des paramètres

## Tests de Validation

### Test Manuel Live Mode

1. Ouvrir `http://localhost:8080/static/simulations.html`
2. Sélectionner user "jack" dans le menu
3. Sélectionner source "cointracking_api"
4. Cliquer "Live"
5. [OK] Vérifie: 190+ assets chargés, valeur totale correcte

### Test Manual Test Source Button

1. Ouvrir `http://localhost:8080/static/simulations.html`
2. Sélectionner user "jack"
3. Sélectionner source "cointracking_api" dans dropdown
4. Cliquer bouton " Test"
5. [OK] Vérifie: Alert "[OK] Source OK - 190+ assets trouvés"

### Test Backend API

```bash
# Vérifier que l'API retourne bien des données pour jack
curl -H "X-User: jack" "http://localhost:8080/balances/current?source=cointracking_api"

# Devrait retourner:
# {
#   "items": [ ... 190+ items ... ],
#   "source_used": "cointracking_api",
#   "total_count": 190+
# }
```

### Test Console Browser

```javascript
// Dans la console sur simulations.html
localStorage.setItem('activeUser', 'jack');
window.globalConfig.set('data_source', 'cointracking_api');

const result = await window.loadBalanceData(true);
console.log(result);
// Devrait afficher: { success: true, data: { items: [...190+ items...] } }
```

## Bénéfices

1. [OK] **Isolation Multi-tenant Correcte**: Chaque utilisateur voit ses propres données
2. [OK] **Cohérence Architecture**: Simulateur utilise les mêmes méthodes que Dashboard/Risk
3. [OK] **Cache Unifié**: Bénéficie du cache par user dans `loadBalanceData()`
4. [OK] **Maintenance Simplifiée**: Une seule méthode de chargement à maintenir
5. [OK] **Support CSV & API**: Gestion transparente des deux sources

## Documentation Connexe

- [docs/navigation.md](navigation.md) - Architecture de navigation
- [CLAUDE.md](../CLAUDE.md) - Guide agent section "Sources System"
- [README.md](../README.md) - Configuration multi-utilisateurs

## Notes Techniques

### Pourquoi X-User Header et pas Query Param?

Le header `X-User` est préféré au query param `user_id` pour:
1. **Sécurité**: Headers moins exposés dans logs/historique browser
2. **Cohérence**: Toute l'application utilise ce pattern
3. **Middleware**: Le backend peut extraire automatiquement du header
4. **Cache**: Meilleure granularité de cache par user

### localStorage vs globalConfig pour activeUser

`localStorage.getItem('activeUser')` est la source de vérité car:
- Persistant entre sessions
- Synchronisé cross-tab (événement `storage`)
- Utilisé par le menu de sélection user
- `globalConfig.get('user_id')` n'existe pas dans le schéma

## Changelog

### [1.0.0] - 2025-09-30

#### Fixed
- Live mode retourne maintenant les données correctes pour tous les utilisateurs
- Test source button affiche le nombre réel d'assets
- Isolation multi-tenant respectée dans le simulateur

#### Added
- Fonctions parseCSVBalancesAuto/Local/LineLocal dans simulations.html
- Support complet CSV et API dans le simulateur

#### Changed
- loadLiveData() utilise window.loadBalanceData() au lieu de fetch direct
- testSelectedSource() utilise loadBalanceData() avec changement temporaire de source
- Utilise localStorage.getItem('activeUser') pour identifier l'utilisateur

#### Removed
- Méthode getApiUrl() dupliquée dans global-config.js

# Dashboard V2 - Architecture 3 Niveaux

> Date: 2025-10-22
> Version: 2.0
> Auteur: Dashboard V2 Implementation

## Vue d'ensemble

Le Dashboard V2 introduit une **hiérarchie visuelle en 3 niveaux** pour améliorer l'expérience utilisateur et clarifier l'organisation des informations.

## Architecture

### Niveau 1 - Patrimoine (4 tuiles)
**Bordure orange** - Vue d'ensemble du patrimoine et actifs

1. **Global Overview** (1 colonne)
   - Total patrimoine (tous actifs)
   - P&L Today global
   - Breakdown par module (Crypto/Bourse/Banque)
   - Barres de progression allocation
   - Endpoint: `/api/wealth/global/summary`

2. **Crypto Overview** (1 colonne)
   - Total Value + P&L Today
   - Nombre d'actifs
   - Mini chart portfolio (graphique camembert)
   - Endpoint: `/balances/current`

3. **Bourse** (1 colonne)
   - Valeur totale positions Saxo
   - Nombre de positions
   - Date dernier import
   - Graphique camembert par asset_class
   - Endpoint: `/api/saxo/positions`

4. **Banque** (1 colonne)
   - Valeur totale comptes bancaires
   - Nombre de comptes
   - Nombre de devises
   - Graphique camembert par banque
   - Endpoint: `/api/wealth/banks/positions`

### Niveau 2 - Décision (3 tuiles)
**Bordure bleue** - Informations critiques pour la prise de décision

1. **Global Insight** (2 colonnes)
   - Decision Index avec score unifié
   - Phase Engine (Risk Off / ETH Expansion / Altseason)
   - Scores détaillés (Cycle, On-Chain, Risk)
   - Recommandation actionnable
   - Endpoint: `/api/unified/intelligence`

2. **Market Regime**
   - Détection régimes Bitcoin (Bull/Bear/Correction)
   - Détection régimes Ethereum (Expansion/Compression)
   - Détection régimes Stock Market (Bull/Bear/Consolidation)
   - Barres de progression avec confidence
   - Endpoints: `/api/ml/crypto/regime`, `/api/ml/bourse/regime`

3. **Risk & Alerts**
   - Niveau de risque portfolio (Low/Medium/High)
   - Alertes actives du système de gouvernance
   - Portfolio VaR (95% confidence)
   - Nombre d'alertes actives
   - Endpoints: `/api/risk/dashboard`, `/api/alerts/active`

### Niveau 3 - Opérations (4 tuiles)
**Bordure violette** - Statut système et outils

1. **Execution**
   - Dernière exécution
   - Taux de succès (24h)
   - Volume total (24h)

2. **Activity**
   - Historique des 5 dernières opérations
   - Lien vers historique complet

3. **System Status** (fusionné)
   - Statut API
   - Connexions exchanges (online/total)
   - Fraîcheur des données
   - Endpoint: `/exchanges/status` (optionnel)

4. **Tools & Analytics**
   - Liens vers Analytics
   - Liens vers AI Dashboard
   - Liens vers Optimization
   - Liens vers Debug Menu

## Améliorations Visuelles

### CSS Thème-Aware
```css
.level-1 { border-left: 4px solid var(--warning); }
.level-2 { border-left: 4px solid var(--info); }
.level-3 { border-left: 4px solid var(--brand-primary); }
```

Toutes les couleurs utilisent les variables CSS du thème pour s'adapter automatiquement au mode clair/sombre.

### Badges Supprimés
Les badges "Niveau 1/2/3" ont été supprimés car redondants avec les bordures colorées.

### Boutons Refresh Supprimés
Les boutons refresh manuels ont été supprimés car le dashboard se rafraîchit automatiquement toutes les 1-2 minutes.

## Fonctionnalités Techniques

### Graphiques Camembert (Chart.js)
```javascript
updateSaxoChart(positions) {
    // Graphique Bourse: regroupement par asset_class
    // EQUITY, ETF, BOND, CASH, etc.
    // Tooltip: valeur, %, nombre de positions
}

updateBanksChart(positions) {
    // Graphique Banque: regroupement par banque
    // UBS, Crédit Suisse, PostFinance, etc.
    // Tooltip: valeur, %, nombre de comptes
}
```

### Drag & Drop Multi-Grilles
```javascript
// Drag & drop activé sur 3 grilles indépendantes
// Contrainte: cartes déplaçables uniquement dans leur niveau
// Sauvegarde: localStorage par grille (grid-niveau-1/2/3)
// Handle: drag uniquement via card-header
```

### Phase Engine Integration
```javascript
updatePhaseChips(unifiedState) {
    // Active visuellement la phase détectée
    // Risk Off / ETH Expansion / Altseason
}
```

### Market Regime Detection
```javascript
loadMarketRegimes() {
    // Charge régimes BTC, ETH, Stock
    // Gère gracieusement les endpoints 404
    // Affiche confidence en pourcentage
}
```

### Risk & Alerts System
```javascript
loadRiskAlerts() {
    // Charge niveau de risque (High/Medium/Low)
    // Affiche alertes actives (S1/S2/S3)
    // Calcule VaR 95% (1-day)
}
```

### Multi-Tenant Support
Toutes les tuiles respectent l'isolation multi-utilisateur :
- Header `X-User` automatique via `window.getCurrentUser()`
- Cache invalidé lors du changement d'utilisateur
- Support `file_key` pour sélection CSV spécifique

### Endpoints Optionnels
Certains endpoints peuvent retourner 404 sans affecter l'affichage :
- `/exchanges/status` → Affiche "N/A" (désactivé temporairement)
- `/api/ml/stock/regime` → Affiche "N/A"
- `/execution/governance/alerts` → Message informatif

## Fichiers Modifiés

### Frontend
- `static/dashboard.html` - Structure HTML 3 niveaux
- `static/modules/dashboard-main-controller.js` - Nouvelles fonctions
  - `loadMarketRegimes()`
  - `loadRiskAlerts()`
  - `updatePhaseChips()`
  - `updateSystemStatus()`
- `static/modules/wealth-saxo-summary.js` - Cache multi-user
- `static/components/nav.js` - Définition `window.getCurrentUser()`

### Backend
- `api/saxo_endpoints.py` - Ajout `user_id` + `file_key` à tous les endpoints
- `api/wealth_endpoints.py` - Ajout `file_key` support pour Saxo

## Tuiles Supprimées

Les tuiles suivantes ont été supprimées ou fusionnées :

- [Error] **Scores** - Fusionnée dans Global Insight
- [Error] **Exchange Connections** - Fusionnée dans System Status
- [Error] **System Health** - Fusionnée dans System Status

## Comparaison Avant/Après

| Aspect | Avant | Après |
|--------|-------|-------|
| Nombre de tuiles | 11 | 11 (4+3+4) |
| Hiérarchie | Floue | 3 niveaux clairs (Patrimoine → Décision → Opérations) |
| Niveau 1 | Décision (3 tuiles) | Patrimoine (4 tuiles) |
| Niveau 2 | Patrimoine (4 tuiles) | Décision (3 tuiles) |
| Graphiques | Crypto uniquement | Crypto + Bourse + Banque |
| Mode clair/sombre | Couleurs hardcodées | Variables CSS thème |
| Multi-user Saxo | [Error] Broken | [OK] Fixed |
| Boutons refresh | 4 manuels | Auto-refresh uniquement |

## Endpoints API

### Nouveaux Endpoints Utilisés
```
GET /api/ml/crypto/regime?symbol={BTC|ETH}&lookback_days=365
GET /api/ml/bourse/regime?benchmark=SPY&lookback_days=365
GET /api/alerts/active
```

### Endpoints Optionnels (404 OK)
Ces endpoints peuvent retourner 404, le dashboard gère gracieusement :
- `/api/ml/stock/regime` → Affiche "N/A"
- `/execution/governance/alerts` → Message informatif
- `/exchanges/status` → Affiche "N/A"

## Bugs Corrigés

1. **Cache Multi-User Saxo**
   - Problème: Cache partagé entre utilisateurs
   - Fix: Cache lié à `_cachedForUser`, invalidation automatique

2. **window.getCurrentUser() manquant**
   - Problème: Header X-User jamais envoyé
   - Fix: Définition globale dans `nav.js`

3. **Endpoints Saxo sans user_id**
   - Problème: Tous les users voyaient les mêmes données
   - Fix: Ajout `user_id` + `file_key` à tous les endpoints

4. **Couleurs hardcodées**
   - Problème: Ne s'adaptaient pas au mode sombre
   - Fix: Variables CSS `var(--warning)`, `var(--success)`, etc.

## Notes de Déploiement

### Migration
Aucune migration DB nécessaire. Le déploiement se fait par simple redémarrage du serveur.

### Compatibilité
- [OK] Compatible avec mode clair/sombre
- [OK] Compatible multi-user
- [OK] Compatible responsive (mobile/tablet/desktop)
- [OK] Rétro-compatible avec l'API existante

### Performance
- Auto-refresh : 1-2 minutes par tuile
- Cache Saxo : 30 secondes par utilisateur
- Pas d'impact notable sur les performances

## Prochaines Étapes

Fonctionnalités futures possibles :
- [ ] Graphique historique Decision Index
- [ ] Alertes push en temps réel (WebSocket)
- [ ] Personnalisation ordre/visibilité des tuiles
- [ ] Export PDF du dashboard
- [ ] Comparaison multi-période pour régimes

---

## Changelog

### 2025-10-22 - Inversion Niveaux + Graphiques
- **Niveau 1** : Maintenant Patrimoine (bordure orange) - 4 tuiles
- **Niveau 2** : Maintenant Décision (bordure bleue) - 3 tuiles
- Ajout graphiques camembert dans tuiles Bourse et Banque
- Regroupement par asset_class (Bourse) et par banque (Banque)

### 2025-10-22 - Version Initiale
- Architecture 3 niveaux (Décision → Patrimoine → Opérations)
- Nouvelles tuiles Market Regime et Risk & Alerts
- Multi-user support pour Saxo
- Graphique camembert pour Crypto

---

**Documentation mise à jour le 2025-10-22**
**Dashboard V2 est maintenant en production** [OK]

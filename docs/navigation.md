# Navigation & Menu Hierarchy

## Menu Principal (6 Pages Canoniques)

La navigation consolidée s'articule autour de **6 pages canoniques** avec liens profonds et filtrage contextuel :

### 1. **Portfolio** (`dashboard.html`)
Vue d'ensemble patrimoine avec métriques globales et répartition cross-asset.

**Sous-menus/Ancres :**
- `#overview` - Vue d'ensemble Global Insight
- `#crypto` - Portfolio crypto détaillé
- `#bourse` - Actions & ETF (Saxo)
- `#banque` - Comptes bancaires & épargne
- `#divers` - Actifs divers & illiquides
- `#fx` - Devises & changes

### 2. **Analytics** (`analytics-unified.html`)
Analytics avancés avec ML/IA, cycles marché et monitoring performance.

**Sous-menus/Ancres :**
- `#unified` - Analytics unifiés cross-asset
- `#ml` - Machine Learning & IA
- `#cycles` - Analyse cycles marché
- `#performance` - Performance & backtesting
- `#monitoring` - Monitoring temps réel

### 3. **Risk** (`risk-dashboard.html`)
Gestion des risques avec gouvernance, stress testing et attribution.

**Sous-menus/Ancres :**
- `#governance` - Gouvernance & Decision Engine
- `#stress` - Tests de stress & VaR
- `#risk-attribution` - Attribution des risques
- `#limits` - Limites & contrôles

### 4. **Rebalance** (`rebalance.html`)
Rebalancing intelligent avec propositions et contraintes cross-asset.

**Sous-menus/Ancres :**
- `#proposed-targets` - Objectifs proposés
- `#bourse` - Rebalance portefeuille bourse
- `#funding-plan` - Plan de financement
- `#dca-schedule` - Planning DCA
- `#constraints` - Contraintes & limites

### 5. **Execution** (`execution.html`)
Exécution ordres avec historique, coûts et venues.

**Sous-menus/Ancres :**
- `#orders` - Ordres en cours & planifiés
- `#history` - Historique exécution
- `#costs` - Coûts & frais
- `#venues` - Plateformes & exchanges

### 6. **Settings** (`settings.html`)
Configuration, intégrations et outils administrateurs.

**Sous-menus/Ancres :**
- `#integrations` - Intégrations (Saxo, CoinTracking, banques)
- `#security` - Sécurité & authentification
- `#logs` - Logs & audit trail
- `#monitoring` - Monitoring système
- `#tools` - Outils & debug

## Menu Bourse (Dropdown)

Accès aux fonctionnalités avancées du portefeuille Bourse (Saxo Bank) via un dropdown dédié.

### Sous-menus

- **📊 Dashboard** → `saxo-dashboard.html` (Overview + Positions)
- **📈 Analytics** → `bourse-analytics.html` (Risk Analysis + Advanced Analytics)
- **💡 Recommendations** → `bourse-recommendations.html` (Recommendations + Market Opportunities)

### Pages détaillées

| Page                        | Contenu                                                                                      |
|-----------------------------|----------------------------------------------------------------------------------------------|
| saxo-dashboard.html         | Vue d'ensemble, positions, allocation par asset class                                        |
| bourse-analytics.html       | Risk Score, VaR, Sharpe, Correlation Matrix, Stress Testing, ML Insights, Sector Rotation   |
| bourse-recommendations.html | BUY/HOLD/SELL recommendations, Market Opportunities, Suggested Sales, Impact Simulator      |

---

## Menu Admin (RBAC)

**Visible uniquement** pour les rôles `governance_admin` ou `ml_admin`.

**Dropdown Admin :**
- **ML Command Center** → `analytics-unified.html#ml`
- **Model Registry/Jobs** → `analytics-unified.html#ml` (section admin)
- **Imports & Connecteurs** → `settings.html#integrations`
- **Tools & Debug** → `debug-menu.html`
- **Archive** → `static/archive/index.html`

## WealthContextBar (Filtrage Global)

**Barre contexte** persistante en haut de toutes les pages avec filtres :

- **Household** : `all` | `main` | `secondary`
- **Compte** : `all` | `trading` | `hold` | `staking`
- **Module** : `all` | `crypto` | `bourse` | `banque` | `divers`
- **Devise** : `USD` | `EUR` | `CHF`

**Persistance :** localStorage + querystring synchronisé
**Propagation :** Événement `wealth:change` écouté par Rebalance/Execution

## Système de Liens Profonds

### Fonctionnalités
- **ScrollIntoView** avec offset header sticky (+20px)
- **Highlight temporaire** (2s) avec classe `.is-target`
- **Support back/forward** avec `popstate`
- **URL bookmarkable** avec ancres fonctionnelles

### Format URL
```
https://domain.com/analytics-unified.html#ml?household=main&module=crypto&ccy=EUR
```

## Redirections Legacy

**Pages dépréciées** redirigées automatiquement vers ancres canoniques :

| Legacy Page | Canonical Redirect |
|-------------|-------------------|
| `ai-dashboard.html` | `analytics-unified.html#ml` |
| `intelligence-dashboard.html` | `analytics-unified.html#ml` |
| `performance-monitor.html` | `performance-monitor-unified.html` (nouveau monitor système) |
| `cycle-analysis.html` | `analytics-unified.html#cycles` |
| `execution_history.html` | `execution.html#history` |
| `debug-menu.html` | `settings.html#tools` |
| `portfolio-optimization.html` | `dashboard.html#overview` |

**Mécanisme :** Meta refresh (3s) + JavaScript replace (1.5s) + lien manuel

## Règles de Navigation

1. **Maximum 6 entrées** en navigation principale
2. **Admin via RBAC** uniquement (rôles governance_admin/ml_admin)
3. **Ancres fonctionnelles** avec scroll et highlight
4. **Zéro 404** grâce aux redirections douces
5. **Filtrage contextuel** appliqué sur Rebalance/Execution
6. **Archive accessible** via Admin > Archive

## Standards Badges

**Format uniforme** sur toutes les pages :
```
Source • Updated HH:MM:SS • Contrad XX% • Cap YY% • Overrides N
```

**Timezone :** Europe/Zurich
**États :** OK (vert) | STALE (orange) | ERROR (rouge)
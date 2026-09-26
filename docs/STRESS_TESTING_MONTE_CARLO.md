# Stress Testing & Monte Carlo Simulation

> **Status:** [OK] Production Ready (Dec 2025)
> **Location:** `risk-dashboard.html` → Advanced Risk tab
> **Backend:** `services/risk/stress_testing.py` + `services/risk/monte_carlo.py`

## Table des Matières

- [Vue d'Ensemble](#vue-densemble)
- [Stress Testing](#stress-testing)
- [Monte Carlo Simulation](#monte-carlo-simulation)
- [API Endpoints](#api-endpoints)
- [Usage Frontend](#usage-frontend)
- [Performance & Cache](#performance--cache)
- [Troubleshooting](#troubleshooting)

---

## Vue d'Ensemble

Système d'analyse de risque avancé permettant de **simuler l'impact de crises** et d'**estimer les probabilités de pertes** sur votre portfolio crypto.

### Objectifs

1. **Évaluer la résilience** du portfolio face à des crises historiques
2. **Quantifier les risques extrêmes** via distributions probabilistes
3. **Prendre des décisions éclairées** basées sur scénarios réalistes

### Architecture

```
Frontend (risk-dashboard.html)
    ↓
API Endpoints (/api/risk/stress-test-portfolio, /api/risk/monte-carlo)
    ↓
Backend Services (stress_testing.py, monte_carlo.py)
    ↓
Taxonomy (groups), Price History (365 days), Covariance Matrix
```

---

## Stress Testing

### Principe

Applique des **shocks calibrés** (basés sur crises historiques) à votre portfolio actuel pour estimer l'impact en $.

### 6 Scénarios Disponibles

| Scénario | Impact | Durée | Probabilité (10 ans) |
|----------|--------|-------|---------------------|
|  **Crise Financière 2008** | -45% à -60% | 6-12 mois | 2% |
|  **COVID-19 Mars 2020** | -35% à -50% | 2-6 mois | 5% |
|  **Interdiction Crypto Chine** | -25% à -40% | 3-9 mois | 10% |
|  **Effondrement Tether** | -30% à -55% | 1-4 mois | 8% |
|  **Hausse Taux Fed d'Urgence** | -20% à -35% | 6-18 mois | 15% |
|  **Hack Exchange Majeur** | -15% à -30% | 1-3 mois | 20% |

### Shocks par Groupe

Les shocks sont **différenciés par groupe Taxonomy**. Exemple pour **Crise 2008**:

```python
{
    "BTC": -0.50,           # -50% (flight to quality, nouvelle tech)
    "ETH": -0.55,           # -55%
    "DeFi": -0.70,          # -70% (risque systémique)
    "Stablecoins": -0.05,   # -5% (dépeg partiel)
    "Memecoins": -0.80,     # -80% (vol extrême)
    "SOL": -0.65,           # -65%
    # ... autres groupes
}
```

### Calcul de l'Impact

1. **Grouper** holdings par Taxonomy group
2. **Appliquer** shock spécifique à chaque groupe
3. **Calculer** perte totale portfolio
4. **Identifier** top 3 pires/meilleurs groupes

### Exemple Résultat

```json
{
  "scenario_name": " Crise Financière 2008",
  "portfolio_impact": {
    "loss_pct": -52.3,
    "loss_usd": -12450,
    "value_before": 23800,
    "value_after": 11350
  },
  "worst_groups": [
    {"group": "DeFi", "loss_pct": -70, "loss_usd": -4200},
    {"group": "Memecoins", "loss_pct": -80, "loss_usd": -2400},
    {"group": "L2/Scaling", "loss_pct": -65, "loss_usd": -1950}
  ]
}
```

---

## Monte Carlo Simulation

### Principe

Génère **10,000 scénarios aléatoires** basés sur les distributions **historiques réelles** de rendements (365 jours). Préserve les **corrélations** entre assets via matrice de covariance.

### Métriques Calculées

#### 1. Statistiques Distribution

- **Rendement moyen** (expected return)
- **Rendement médian** (50e percentile)
- **Écart-type** (volatilité)

#### 2. Scénarios Extrêmes

- **P1** (pire cas): 1% des scénarios sont pires
- **P5**: 5e percentile
- **P95**: 95e percentile
- **P99** (meilleur cas): 1% des scénarios sont meilleurs

#### 3. Probabilités de Pertes

- Perte > 5%
- Perte > 10%
- Perte > 20%
- Perte > 30%

#### 4. VaR/CVaR Monte Carlo

- **VaR 95%**: Perte maximale dans 95% des cas
- **CVaR 95%**: Perte moyenne si dépassement VaR 95%
- **VaR 99%**: Perte maximale dans 99% des cas
- **CVaR 99%**: Perte moyenne si dépassement VaR 99%

### Algorithme

```python
# 1. Charger historique prix (365 jours) pour chaque asset
prices = get_cached_history(symbol, days=365)

# 2. Calculer rendements journaliers
daily_returns = price_series.pct_change()

# 3. Calculer matrice de covariance
cov_matrix = returns_df.cov()

# 4. Régularisation (évite SVD convergence errors)
epsilon = 1e-6
cov_matrix_reg = cov_matrix + np.eye(len(cov_matrix)) * epsilon

# 5. Générer 10,000 scénarios (multivariate normal)
for i in range(10000):
    simulated_returns = np.random.multivariate_normal(
        mean_returns * horizon_days,
        cov_matrix_reg * horizon_days
    )

    # Rendement portfolio pondéré
    portfolio_return = sum(simulated_returns * weights)

# 6. Calculer statistiques
var_95 = -np.percentile(returns, 5)
cvar_95 = -mean(returns[returns <= percentile(returns, 5)])
```

### Exemple Résultat

```json
{
  "simulation_params": {
    "num_simulations": 10000,
    "horizon_days": 30,
    "num_assets": 48
  },
  "statistics": {
    "mean_return_pct": 2.45,
    "median_return_pct": 2.12,
    "std_return_pct": 15.32
  },
  "scenarios": {
    "worst_case_pct": -42.5,  // P1
    "best_case_pct": 58.3     // P99
  },
  "loss_probabilities": {
    "prob_loss_5": 0.287,   // 28.7% chance perte >5%
    "prob_loss_10": 0.165,  // 16.5% chance perte >10%
    "prob_loss_20": 0.123,  // 12.3% chance perte >20%
    "prob_loss_30": 0.058   // 5.8% chance perte >30%
  },
  "risk_metrics": {
    "var_95_pct": 18.7,
    "cvar_95_pct": 25.4
  }
}
```

### Graphique Interactif (Chart.js)

- **Histogramme coloré**:
  - [Positive] Vert: Rendements positifs (gains)
  - [Pending] Orange: Rendements négatifs (pertes modérées)
  - [Negative] Rouge: Pertes extrêmes (au-delà VaR 95%)
- **Marqueurs**:
  - Ligne rouge pointillée: VaR 95%
  - Ligne bleue pointillée: Rendement médian
- **Tooltips**: Hover pour voir rendement exact + fréquence

---

## API Endpoints

### 1. Liste Scénarios Stress Test

```bash
GET /api/risk/stress-scenarios
```

**Response:**

```json
{
  "success": true,
  "scenarios": [
    {
      "id": "crisis_2008",
      "name": " Crise Financière 2008",
      "impact_range": {"min": -45, "max": -60},
      "probability_10y": 0.02,
      "duration": "6-12 mois"
    }
  ]
}
```

### 2. Exécuter Stress Test

```bash
POST /api/risk/stress-test-portfolio?scenario_id=crisis_2008
Headers: X-User: jack
```

**Response:**

```json
{
  "success": true,
  "result": {
    "scenario_id": "crisis_2008",
    "portfolio_impact": {
      "loss_pct": -52.3,
      "loss_usd": -12450
    },
    "worst_groups": [...],
    "metadata": {
      "probability_10y": 0.02,
      "timestamp": "2025-12-31T12:00:00"
    }
  }
}
```

### 3. Simulation Monte Carlo

```bash
GET /api/risk/monte-carlo?num_simulations=10000&horizon_days=30
Headers: X-User: jack
```

**Parameters:**

- `num_simulations`: 1,000 à 50,000 (défaut: 10,000)
- `horizon_days`: 1 à 365 jours (défaut: 30)
- `confidence_level`: 0.90 à 0.99 (défaut: 0.95)
- `price_history_days`: 90 à 730 jours (défaut: 365)

**Response:** Voir exemple ci-dessus

---

## Usage Frontend

### Workflow Utilisateur

1. **Ouvrir** `risk-dashboard.html`
2. **Onglet** "Advanced Risk"
3. **Stress Testing**: Cliquer sur scénario → Modal avec impact réel
4. **Monte Carlo**:
   - Voir bouton " Lancer la Simulation"
   - Cliquer → Attendre 10-30 sec (loading animé)
   - Résultats + graphique affichés
   - Badge " Mis en cache" (sessionStorage)
5. **Refresh page** → Résultats Monte Carlo instantanés (cache)
6. **Re-calculer**: Bouton " Re-calculer" pour données fraîches

### Code Frontend (Exemple)

```javascript
// Stress Test
window.runStressTest = async function(scenarioId) {
  const response = await window.globalConfig.apiRequest(
    `/api/risk/stress-test-portfolio?scenario_id=${scenarioId}`,
    { method: 'POST' }
  );
  // Afficher modal avec results
};

// Monte Carlo
window.runMonteCarloSimulation = async function() {
  const response = await window.globalConfig.apiRequest('/api/risk/monte-carlo', {
    params: {
      num_simulations: 10000,
      horizon_days: 30
    }
  });

  // Cache en sessionStorage
  sessionStorage.setItem('monte_carlo_result', JSON.stringify(response.result));

  // Render UI + Chart
  renderMonteCarloResultsUI(response.result);
};
```

---

## Performance & Cache

### Durées Typiques

- **Stress Test**: < 1 seconde (calcul simple)
- **Monte Carlo** (10,000 simulations): 10-30 secondes
  - WSL2: ~20 secondes
  - Linux natif: ~10 secondes
  - Dépend de: nombre d'assets (48 typique), CPU

### Cache Strategy

#### Stress Test

- **Pas de cache** (calcul instantané < 1s)
- Toujours données fraîches

#### Monte Carlo

- **SessionStorage cache** (client-side)
- Clé: `monte_carlo_result`
- TTL: Session (fermeture onglet efface)
- Avantages:
  - Évite recalcul 10-30s à chaque refresh
  - Résultats instantanés après 1er calcul
  - Bouton " Re-calculer" pour forcer update

### Optimisations SVD

**Problème:** Matrices de covariance mal conditionnées → "SVD did not converge"

**Solution:** Régularisation

```python
# Add small epsilon to diagonal
epsilon = 1e-6
cov_matrix_reg = cov_matrix + np.eye(len(cov_matrix)) * epsilon

# Use regularized matrix
simulated_returns = np.random.multivariate_normal(
    mean_returns * horizon_days,
    cov_matrix_reg * horizon_days,
    check_valid='ignore'  # Ignore validation errors
)
```

**Fallback graceful:**

```python
except np.linalg.LinAlgError:
    # Use mean return if simulation fails
    portfolio_return = mean_returns * horizon_days * weights
```

---

## Troubleshooting

### Error: "SVD did not converge"

**Cause:** Matrice de covariance singulière (collinéarité parfaite entre assets)

**Solution:**
- [OK] Régularisation epsilon 1e-6 (déjà implémentée)
- [OK] `check_valid='ignore'` (déjà implémenté)
- [OK] Fallback graceful (déjà implémenté)

Si erreur persiste:
- Vérifier nombre d'assets (min 2 requis)
- Vérifier données prix (min 30 jours d'historique)

### Error: "Insufficient assets with price data"

**Cause:** < 2 assets avec historique prix valide

**Solution:**
- Augmenter `price_history_days` (ex: 180 jours au lieu de 365)
- Vérifier cache prix: `logs/app.log` → "Insufficient price data for {symbol}"

### Cache Monte Carlo non persistant

**Cause:** SessionStorage effacé par fermeture onglet

**Solution:**
- Normal behavior (cache session uniquement)
- Utiliser " Re-calculer" pour rafraîchir
- Future: localStorage avec TTL pour cache persistant

### Graphique Monte Carlo ne s'affiche pas

**Cause:** Chart.js non chargé ou percentiles manquants

**Solution:**
1. Vérifier console: `window.Chart` défini?
2. Vérifier `result.distribution_percentiles` existe
3. Hard refresh: Ctrl+Shift+R

---

## Références

### Backend

- [`services/risk/stress_testing.py`](../services/risk/stress_testing.py) - Service stress testing
- [`services/risk/monte_carlo.py`](../services/risk/monte_carlo.py) - Service Monte Carlo
- [`api/risk_endpoints.py`](../api/risk_endpoints.py) - Endpoints API (lignes 1696-1900)

### Frontend

- [`static/risk-dashboard.html`](../static/risk-dashboard.html) - Page principale (lignes 773-1252)

### Documentation

- [`CLAUDE.md`](../CLAUDE.md#stress-testing--monte-carlo-simulation-dec-2025) - Guide développeur
- [`README.md`](../README.md) - Vue d'ensemble projet

---

**Last Updated:** 2025-12-31
**Version:** 1.0.0 (Production Ready)

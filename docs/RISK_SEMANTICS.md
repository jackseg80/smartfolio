# Risk Semantics — Source de Vérité

> **[Warning] Règle Canonique — Sémantique Risk (Option A)**
>
> Le **Risk Score** est un indicateur **positif** de robustesse, borné **[0..100]**.
>
> **Convention** : Plus haut = plus robuste (risque perçu plus faible).
>
> **Conséquence** : Dans le Decision Index (DI), Risk contribue **positivement** :
> ```
> DI = wCycle·scoreCycle + wOnchain·scoreOnchain + wRisk·scoreRisk
> ```
>
> ** Interdit** : Ne jamais inverser avec `100 - scoreRisk` (calculs, visualisations, contributions).
>
> **Visualisation** : Contribution = `(poids × score) / Σ(poids × score)`
>
>  **Source de vérité** : [docs/RISK_SEMANTICS.md](RISK_SEMANTICS.md)

---

## Architecture Centralisée (Oct 2025)

### Single Source of Truth

**Module central** : [`services/risk_scoring.py`](../services/risk_scoring.py)

Toute logique de calcul Risk Score et mapping score→level **DOIT** être importée depuis ce module.

**[Error] Anti-pattern** : Dupliquer la logique dans d'autres services (risque de divergence).

### Dual Score System

| Score | Type | Base de calcul | Usage |
|-------|------|---------------|--------|
| **`risk_score`** | Autoritaire | VaR + Sharpe + Drawdown + Volatilité | UI, Decision Index, communication |
| **`risk_score_structural`** | Structurel | `risk_score` + GRI + Concentration + Structure | Garde-fou allocation, caps governance |

**Recommandation** : Approche hybride pour niveau final
```python
final_level = max(level(risk_score), level(risk_score_structural))
```

### Mapping Canonique Score → Level

```python
# Thresholds (services/risk_scoring.py:RISK_LEVEL_THRESHOLDS)
score >= 80  → "very_low"     # Très robuste
score >= 65  → "low"          # Robuste
score >= 50  → "medium"       # Modéré
score >= 35  → "high"         # Fragile
score >= 20  → "very_high"    # Très fragile
score < 20   → "critical"     # Critique
```

**[Warning] CRITIQUE** : Ce mapping est **inversé** car score = robustesse (score élevé = risque faible).

### Formule Risk Score (Quantitatif - Autoritaire)

```python
score = 50.0  # Baseline neutre

# VaR impact (VaR ↑ → robustesse ↓ → score ↓)
if var_95 > 0.25:  score -= 30
elif var_95 < 0.05: score += 10

# Sharpe impact (Sharpe ↑ → robustesse ↑ → score ↑)
if sharpe > 2.0:   score += 20
elif sharpe < 0:   score -= 15

# Drawdown impact (DD ↑ → robustesse ↓ → score ↓)
if |dd| > 0.50:    score -= 25
elif |dd| < 0.10:  score += 10

# Volatility impact (Vol ↑ → robustesse ↓ → score ↓)
if vol > 1.0:      score -= 10
elif vol < 0.20:   score += 10

score = clamp(score, 0, 100)
level = score_to_level(score)
```

### Pénalités Adoucies (Oct 2025)

**Problème résolu** : Portfolios "degen" (55% memecoins + DD 61%) scoraient systématiquement à 0/100 à cause de pénalités cumulatives trop sévères.

**Correctif** : Réduction progressive des pénalités pour éviter clamp à 0 sur portfolios risqués mais cohérents.

#### Nouveaux Seuils de Pénalités

| Métrique | Seuil | Ancien | Nouveau | Réduction |
|----------|-------|--------|---------|-----------|
| **Memecoins** | >70% | — | -25 | Nouveau |
| | >50% | -30 | **-18** | -40% |
| | >30% | -20 | **-12** | -40% |
| | >15% | -10 | **-8** | -20% |
| | >5% | -5 | **-4** | -20% |
| **Drawdown** | >70% | — | -22 | Nouveau |
| | >50% | -25 | **-15** | -40% |
| | >30% | -15 | **-12** | -20% |
| **HHI (Concentration)** | >0.40 | -15 | **-12** | -20% |
| | >0.25 | -10 | **-8** | -20% |
| | >0.15 | -5 | **-3** | -40% |
| **GRI (Group Risk)** | >7.0 | -15 | **-10** | -33% |
| | >6.0 | -10 | **-7** | -30% |
| | >5.0 | -5 | **-4** | -20% |

**Validation** : Tests `test_risk_scoring_edge_cases.py` (11 tests, monotonicité + bornes + transitions)

#### Exemples de Scoring Réels

**Portfolio Degen (55% memecoins, DD 61%, Vol 65%)**
```
Base:           50
VaR 95% (6.2%): +5  → 55
Sharpe (0.33):  +0  → 55
DD (61.7%):    -15  → 40  [OK] (était -25)
Vol (64.96%):   -5  → 35
Memes (54.99%):-15  → 20  [OK] (était -30)
HHI (0.218):    -3  → 17  [OK] (était -5)
GRI (7.44):    -10  → 7   [OK] (était -15)
Div (1.09):     +5  → 12

Score final: 12/100 → Risk Level "critical" (<20)
```
**Interprétation** : Portfolio très risqué mais cohérent avec stratégie degen. Score > 0 valide l'existence d'une structure minimale (diversification 1.09).

---

**Portfolio Équilibré (192 assets, Sharpe 1.84, Long-Term 93% coverage)**
```
Base:           50
VaR 95% (2.9%): +10 → 60
Sharpe (1.84): +15  → 75
DD (42.3%):    -12  → 63
Vol (30.3%):    +5  → 68
Memes (1.6%):   -4  → 64
HHI (0.08):     +0  → 64
GRI (3.5):      +5  → 69
Div (1.41):     +5  → 74

Score final: 74/100 → Risk Level "low" (65-80)
```
**Interprétation** : Portfolio robuste avec bonne diversification (effective assets: 132). Long-Term window (365j, 124 assets) valide stabilité historique.

---

**Portfolio Catastrophique (75% memes, DD 80%, Sharpe négatif)**
```
Base:           50
VaR 95% (15%): -15  → 35
Sharpe (-0.2): -15  → 20
DD (80%):      -22  → -2
Vol (85%):     -10  → -12
Memes (75%):   -25  → -37
HHI (0.35):     -8  → -45
GRI (8.5):     -10  → -55
Div (0.5):      -5  → -60

Score final: 0/100 → Risk Level "critical" (clamped)
```
**Interprétation** : Portfolio ultra-extrême avec Sharpe négatif + DD 80% + 75% memes. Score 0 est acceptable pour ce niveau de risque catastrophique.

** Règle d'or** : Un portfolio degen "normal" (Sharpe positif, DD < 70%, < 70% memes) doit scorer **10-25**, pas 0.

### Formule Risk Score Structural

**Base** : `risk_score` (autoritaire)

**Ajustements structurels** :
- **GRI (Group Risk Index)** : Exposition pondérée par risque de groupe (0-10)
- **Concentration** : Top5 holdings, HHI (Herfindahl-Hirschman Index)
- **Structure** : % Stablecoins, diversification ratio

**Exemple** : Portfolio BTC-heavy (43%)
- Risk Score : 65 (robuste historiquement)
- Risk Structural : 37 (pénalisé pour concentration)

### Metadata Audit & Traçabilité

**Réponse API** (`/api/risk/dashboard`) :
```json
{
  "risk_metrics": {
    "risk_score": 65.0,
    "risk_score_structural": 37.0,
    "structural_breakdown": {
      "var_95": -8.0,
      "sharpe": 10.0,
      "drawdown": 5.0,
      "volatility": 5.0,
      "stables": -2.0,
      "concentration": 3.0,
      "gri": 6.0
    },
    "window_used": {
      "price_history_days": 365,
      "lookback_days": 90,
      "actual_data_points": 55
    }
  }
}
```

### Tests Non-Régression

**Fichier** : [`tests/unit/test_risk_scoring.py`](../tests/unit/test_risk_scoring.py)

**Couvre** :
- Mapping score→level (85→very_low, 40→high, etc.)
- Sémantique Option A (VaR ↑ → score ↓, Sharpe ↑ → score ↑)
- Breakdown contributions (sum validation)
- Clamping [0, 100]

---

## Dual Window System (Oct 2025)

### Problème Résolu

**Symptôme** : Portfolio avec cryptos récentes (ex: 55j historique) montre Sharpe -0.29 avec Risk Score 65 (robuste) — incohérence apparente.

**Cause** : Intersection temporelle courte (55j au lieu de 365j demandés) produit des ratios instables et négatifs — mathématiquement correct mais trompeur pour évaluation portfolio.

**Solution** : Système Dual-Window avec 2 vues :

#### Long-Term Window (Autoritaire)
- **Objectif** : Métriques stables sur historique long
- **Cohorte** : Exclut assets récents, garde ≥80% valeur portfolio
- **Cascade Fallback** :
  - 365j + 80% couverture (priorité)
  - 180j + 70% couverture
  - 120j + 60% couverture
  - 90j + 50% couverture (dernier recours)
- **Garde-fous** : min 5 assets, min 180j historique
- **Usage** : Score autoritaire pour Decision Index et communication

#### Full Intersection Window (Référence)
- **Objectif** : Vue complète incluant TOUS les assets
- **Période** : Intersection commune minimale (peut être courte)
- **Usage** : Détection divergences, alertes temporelles

### Architecture

**Service** : `services/portfolio_metrics.py:169` - `calculate_dual_window_metrics()`

**Paramètres** :
```python
min_history_days: int = 180      # Jours minimum cohorte LT
min_coverage_pct: float = 0.80   # % valeur minimum (80%)
min_asset_count: int = 5         # Nombre assets minimum
```

**Endpoint** : `/api/risk/dashboard?use_dual_window=true`

**Nouveaux Query Params** :
- `use_dual_window` (bool, défaut=True)
- `min_history_days` (int, défaut=180)
- `min_coverage_pct` (float, défaut=0.80)
- `min_asset_count` (int, défaut=5)

### Réponse API Étendue

```json
{
  "risk_metrics": {
    "risk_score": 65.0,
    "sharpe_ratio": 1.42,
    "window_used": {
      "dual_window_enabled": true,
      "risk_score_source": "long_term"
    },
    "dual_window": {
      "enabled": true,
      "long_term": {
        "available": true,
        "window_days": 365,
        "asset_count": 3,
        "coverage_pct": 0.80,
        "metrics": {
          "sharpe_ratio": 1.42,
          "volatility": 0.32,
          "risk_score": 65.0
        }
      },
      "full_intersection": {
        "window_days": 55,
        "asset_count": 5,
        "metrics": {
          "sharpe_ratio": -0.29,
          "volatility": 0.85,
          "risk_score": 38.0
        }
      },
      "exclusions": {
        "excluded_assets": [{"symbol": "PEPE", "reason": "history_55d_<_365d"}],
        "excluded_value_usd": 20000,
        "excluded_pct": 0.20,
        "included_assets": [...],
        "included_pct": 0.80,
        "target_days": 365,
        "achieved_days": 365,
        "reason": "success"
      }
    }
  }
}
```

### Frontend Display

**Badges Dual-Window** (risk-dashboard.html:4217) :
- **Long-Term** : Fenêtre + couverture + Sharpe (vert/autoritaire)
- **Full Intersection** : Fenêtre + divergence vs LT (rouge si écart > 0.5)
- [Warning] **Alerte Exclusion** : Si > 20% valeur exclue
- [OK] **Source** : Indique quelle fenêtre est autoritaire

### Tests

**Fichier** : `tests/unit/test_dual_window_metrics.py`

**Couverture** :
- [OK] Cohorte long-term disponible (cas nominal)
- [OK] Cascade fallback (365 → 180j)
- [OK] Aucune cohorte valide (fallback full intersection)
- [OK] Divergence Sharpe entre fenêtres
- [OK] Métadonnées exclusions précises
- [OK] Asset count insuffisant
- [OK] Fenêtres identiques quand tous assets ont historique long

**Commande** :
```bash
pytest tests/unit/test_dual_window_metrics.py -v
```

### Cas d'Usage

#### Bon Cas : Portfolio Mature
- 5 assets, tous 365j+ historique
- Long-Term = Full Intersection
- Risk Score stable et fiable

#### Attention : Portfolio Mixte
- 3 assets anciens (365j, 80% valeur)
- 2 assets récents (55j, 20% valeur)
- Long-Term exclut récents → score stable
- Full Intersection inclut récents → score instable (alerte)

#### Limitation : Portfolio Récent
- Tous assets < 90j
- Aucune cohorte long-term
- Fallback full intersection uniquement (warning)

### Fix Bonus : Score Structural

**Corrigé** : `api/risk_endpoints.py:73-84`

**Avant** ([Error] Inversé) :
```python
if perf_ratio < 0.5: d_perf = +10  # Mauvais Sharpe augmentait le score
```

**Après** ([OK] Correct) :
```python
if perf_ratio < 0:     d_perf = -15  # Négatif diminue score
elif perf_ratio < 0.5: d_perf = -10  # Faible diminue score
elif perf_ratio > 2.0: d_perf = +15  # Excellent augmente score
```

---

## Ajustements Structurels V2 (Nov 2025)

### Problème Résolu

Portfolios avec profils de risque très différents obtenaient des scores quasi identiques (57-59/100). Le système ne prenait pas en compte :
- Protection stablecoins (0% vs 12% = même score)
- Exposition majors (BTC+ETH)
- Sur-exposition altcoins volatils

### Solution : Système à 3 Niveaux

Basé sur données réelles de crashes crypto.

#### Protection Stablecoins (±15 pts)

12% stables = -8% pertes évitées lors bear market 2022.

```python
stables_pct >= 0.15  → +15  # Excellent cushion
stables_pct >= 0.10  → +10  # Bonne protection
stables_pct >= 0.05  → +5   # Protection minimale
stables_pct > 0      → 0    # Insuffisant
stables_pct == 0     → -10  # Vulnérable
```

#### Exposition Majors BTC+ETH (±10 pts)

BTC+ETH perdent 20% moins que altcoins lors des crashes.

```python
majors_pct >= 0.60  → +10  # Portfolio sain
majors_pct >= 0.50  → +5   # Acceptable
majors_pct >= 0.40  → 0    # Sous-exposé
majors_pct < 0.40   → -10  # Risqué
```

#### Sur-exposition Altcoins (-15 pts max)

Altcoins DeFi : -85% vs BTC -65% lors bear market 2021-2022.

```python
altcoins_pct > 0.50  → -15  # Très risqué
altcoins_pct > 0.40  → -10  # Risqué
altcoins_pct > 0.30  → -5   # Acceptable
altcoins_pct <= 0.30 → 0    # Raisonnable
```

#### Formule Finale V2

```python
adj_structural_total = adj_stables + adj_majors + adj_altcoins
final_risk_score_v2 = clamp(blended_risk_score + penalties + adj_structural_total, 0, 100)
```

#### Validation Nov 2025

| Portfolio | Stables | Majors | Altcoins | Ajustements | Score avant | **Score après** |
|-----------|---------|--------|----------|-------------|-------------|-----------------|
| **Low Risk** | 12% | 53% | 35% | +10 +5 -5 = **+10** | 59 | **69** [OK] |
| **Medium Risk** | 0% | 54% | 46% | -10 +5 -10 = **-15** | 57 | **47** [Warning] |
| **API (192 assets)** | 6% | 60%+ | <30% | +5 +10 +0 = **+15** | 62 | **77** [OK] |

**Différenciation obtenue** : Low (69) vs Medium (47) = **22 points** (×11 amélioration vs 2 pts avant)

---

## QA Checklist (Étendue)

- [ ] Aucun `100 - scoreRisk` dans le code ni dans les docs
- [ ] Contribution Risk cohérente avec son poids configuré
- [ ] Visualisations et agrégations vérifiées côté UI et backend
- [ ] **NOUVEAU** : Aucune duplication de logique scoring (import depuis `risk_scoring.py` uniquement)
- [ ] **NOUVEAU** : Endpoint n'override PAS le `overall_risk_level` du service (pas de re-mapping)
- [ ] **NOUVEAU** : Tests non-régression passent (`pytest tests/unit/test_risk_scoring.py`)
- [ ] **NOUVEAU** : API expose `structural_breakdown` et `window_used` pour audit
- [ ] **[New] Dual-Window** : Long-Term window disponible quand possible (≥80% couverture)
- [ ] **[New] Dual-Window** : Alerte exclusion si > 20% valeur exclue
- [ ] **[New] Dual-Window** : Tests dual-window passent (`pytest tests/unit/test_dual_window_metrics.py`)
- [ ] **[New] Score Structural** : Sharpe/Volatility non inversés (bon → +score)

---

## Migration Oct 2025 — V2 as Authoritative Engine

À partir du commit **[MIGRATE-TO-V2]**, le **Risk Score V2** (Dual-Window Blend + pénalités existantes) devient le moteur autoritaire pour l'API et l'UI.

### Changements

**API:**
- Défaut `risk_version="v2_active"` (était `"v2_shadow"`)
- Endpoint `/api/risk/dashboard` retourne désormais:
  - `risk_metrics.risk_score` → **V2** (Dual-Window Blend + pénalités)
  - `risk_metrics.risk_version_info.active_version` → `"v2"`
  - `risk_metrics.risk_version_info.risk_score_legacy` → Legacy (comparaison)

**Dashboard:**
- Affiche V2 comme score principal avec badge [OK] (vert)
- Legacy disponible pour comparaison (atténué, à droite)
- Badge "Comparaison des Versions" remplace "Shadow Mode V2"

**Formule:**
- Aucune modification (Dual-Window Blend + pénalités Oct 2025 inchangés)
- Voir sections "Dual Window System" et "Pénalités Adoucies (Oct 2025)"

### Raison

V2 est plus stable et représentatif grâce au système **Dual-Window** qui gère mieux les assets récents:
- **Long-Term Window** : Cohorte stable (≥180j historique, ≥80% valeur)
- **Full Intersection** : Vue complète (tous assets, fenêtre courte)
- **Blend dynamique** : Pondération selon couverture Long-Term + pénalités exclusions/memecoins

Avantages:
- [OK] Sharpe stable même avec assets récents (pas de biais fenêtre courte)
- [OK] Détection portfolios degen (pénalités memecoins jeunes + exclusions)
- [OK] Transparence (métadonnées dual-window exposées dans API)

### Migration pour Utilisateurs API

**Breaking Change Mineur:**
Si vos appels dépendaient du comportement Legacy par défaut, ajoutez explicitement `?risk_version=legacy` à vos requêtes:

```bash
# AVANT (implicite: Legacy)
GET /api/risk/dashboard?source=cointracking&user_id=demo

# APRÈS (explicite: Legacy pour compatibilité)
GET /api/risk/dashboard?source=cointracking&user_id=demo&risk_version=legacy
```

**Bénéfice:**
V2 offre des scores plus stables sur portfolios avec assets récents. Divergence Legacy/V2 indique problèmes structurels (memecoins jeunes, exclusions importantes).

### Validation

Tests existants passent sans modification (V2 déjà implémenté et testé):
```bash
pytest tests/unit/test_dual_window_metrics.py -v      # 7 tests
pytest tests/unit/test_risk_semantics_baseline.py -v  # Tests baseline
```

Sanity check API:
```bash
# Vérifier active_version = v2
curl -s "http://localhost:8080/api/risk/dashboard?source=cointracking&user_id=demo" \
  | jq '.risk_metrics.risk_version_info.active_version'
# Attendu: "v2"
```


# Gouvernance & Decision Engine

## Hiérarchie Décisionnelle

**SMART** (quoi) → **Decision Engine** (combien/tempo) avec garde-fous automatiques.

### SMART System
- **Allocation** : Détermine "quoi" acheter/vendre
- **Signaux ML** : Intelligence artificielle prédictive
- **Market Regime** : Classification bull/bear/neutral

### Decision Engine
- **Quantification** : "Combien" et "quand" exécuter
- **Caps dynamiques** : Limitation automatique selon volatilité
- **Tempo** : Gestion timing et fragmentation ordres

---

## Système de Caps (Priorité Stricte)

**Ordre de priorité** : `error 5%` > `stale 8%` > `alert_cap` > `engine_cap`

### 1. Error Cap (5%)
- **Déclenchement** : Erreur API/connexion critique
- **Action** : Freeze complet système
- **Reset** : Manuel uniquement

### 2. Stale Cap (8%)
- **Déclenchement** : Données obsolètes >30min
- **Action** : Mode dégradé conservateur

---

## Système de Contradiction Unifié

**Source unique**: `governance.contradiction_index` (0-1 normalisé)

### Classification Automatique

| Niveau | Seuil | Action |
|--------|-------|--------|
| **Low** | < 40% | Mode normal, stratégies actives |
| **Medium** | 40-70% | Vigilance, allocation équilibrée |
| **High** | ≥ 70% | Mode défensif, réduction risques |

### Poids Adaptatifs (Baseline Backtesting)

**Formule**: contradiction ↑ → risk ↑, cycle/onchain ↓

- **Cycle reduction**: jusqu'à -35%
- **OnChain reduction**: jusqu'à -15%
- **Risk increase**: jusqu'à +50%
- **Bornes**: [12%-65%] avec renormalisation stricte

### Caps de Risque Adaptatifs

**Segments ciblés**:
- **Memecoins**: 15% → 5% (réduction -67%)
- **Small Caps**: 25% → 12% (réduction -52%)

**Intégration**: Simulateur + validation allocations automatique

### Documentation Complète

Voir: `docs/contradiction-system.md` pour architecture détaillée
- **Reset** : Automatique dès données fraîches

### 3. Alert Cap (Variable)
- **Déclenchement** : Alertes ML critiques (S3)
- **Action** : Réduction -3pts allocation
- **Reset** : Expiration alerte (24h max)

### 4. Engine Cap (Variable)
- **Déclenchement** : Volatilité/corrélation excessive
- **Action** : Limitation progressive
- **Reset** : Retour conditions normales

---

## Hystérésis & Smoothing

### VaR Thresholds
- **VaR In** : 4.0% (déclenchement mode conservateur)
- **VaR Out** : 3.5% (retour mode normal)
- **Anti-flapping** : Minimum 15min entre transitions

### Stale Data
- **Stale In** : 60 minutes (données considérées obsolètes)
- **Stale Out** : 30 minutes (retour fraîcheur acceptable)

### Smoothing Alpha
- **Coefficient** : 0.7 (nouveaux) / 0.3 (historique)
- **Application** : Lissage caps pour éviter volatilité excessive
- **Formule** : `cap_final = 0.7 × cap_nouveau + 0.3 × cap_précédent`

---

## Système d'Alertes

### Types d'alertes (6)
1. **VOLATILITY_HIGH** : Volatilité excessive (>2σ)
2. **REGIME_CHANGE** : Changement régime marché
3. **CORRELATION_SYSTEMIC** : Corrélation systémique
4. **ML_CONTRADICTION** : Contradictions modèles ML
5. **ML_CONFIDENCE_LOW** : Confiance ML faible
6. **EXECUTION_COST_HIGH** : Coûts exécution élevés

### Niveaux de sévérité
- **S1 (Info)** : Information, aucune action
- **S2 (Warning)** : Mode Slow (-20% vélocité)
- **S3 (Critical)** : Freeze système

### Escalade automatique
- **Trigger** : 2+ alertes S2 simultanées
- **Action** : Génération automatique alerte S3
- **Anti-bruit** : Minimum 5min entre escalades

---

## Contradictions & Overrides

### Seuil contradiction
- **Critique** : >55% contradictions entre modèles
- **Action** : Downgrade allocations agressives
- **Bypass** : Seuil 70% pour neutralisation anti-circularité

### Overrides manuels
- **Gouvernance** : Surcharge temporaire caps automatiques
- **Durée** : Maximum 24h sans renouvellement
- **Audit** : Log complet avec justification

### Freeze S3 sémantique
- **Trigger** : Alertes critiques ou contradictions >70%
- **Action** : Arrêt complet nouvelles positions
- **Sortie** : Validation manuelle governance_admin

### Intégration ExecutionEngine ↔ Freeze

- **Vérification** : `ExecutionEngine` appelle `governance_engine.validate_operation("new_purchases")` avant tout achat
- **Comportement** : Si freeze actif → ordres d'achat bloqués (status `CANCELLED`)
- **Ventes autorisées** : Les ordres de vente passent même sous freeze S3
- **Dry-run** : Mode simulation bypass le check freeze
- **Tests** : `tests/unit/test_execution_engine.py::TestGovernanceFreezeIntegration`

---

## RBAC (Role-Based Access Control)

### Rôles disponibles
- **`governance_admin`** : Accès complet gouvernance + overrides
- **`ml_admin`** : Gestion modèles ML + paramètres IA
- **`viewer`** : Consultation uniquement

### Permissions par rôle

**governance_admin :**
- Modification caps manuels
- Approbation overrides
- Reset freeze S3
- Configuration alertes

**ml_admin :**
- Paramétrage modèles ML
- Debug ML endpoints
- Registry modèles
- Jobs training

### Endpoints protégés
- `/execution/governance/approve/{resource_id}` : governance_admin
- `/api/ml/debug/*` : Header `X-Admin-Key` requis
- `/api/alerts/acknowledge` : Tous rôles admin

---

## Badges Standards Gouvernance

**Format** : `Updated HH:MM:SS • Contrad XX% • Cap YY% • Overrides N`

### Éléments
- **Updated** : Timestamp dernière mise à jour (Europe/Zurich)
- **Contrad** : % contradictions entre modèles ML
- **Cap** : Cap effectif appliqué (plus restrictif)
- **Overrides** : Nombre overrides manuels actifs

### États visuels
- **[Positive] OK** : Système nominal (contrad <30%, cap >50%)
- **[Pending] STALE** : Données obsolètes ou caps réduits
- ** ERROR** : Freeze système ou erreur critique
## Cap d’exécution (alignement UI)

- Source de vérité: `GET /execution/governance/state.active_policy.cap_daily` (0–1). L’UI affiche en pourcentage.
- Sélecteurs centralisés (frontend): `static/selectors/governance.js`
  - `selectCapPercent(state)`: cap en % à utiliser partout côté UI
  - `selectPolicyCapPercent(state)`: cap policy en %
  - `selectEngineCapPercent(state)`: cap "SMART/engine" en % (affichage secondaire)
- Priorité d’affichage: Policy > SMART. Ne jamais afficher SMART seul si la policy existe.
- Badge serré: afficher “[Frozen] Freeze/Cap serré (±X%)” si mode Freeze ou cap ≤ 2%.
- Convergence: `iterations = ceil(maxDelta / (capPct / 100))`.

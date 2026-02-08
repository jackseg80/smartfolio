# Contributing – SmartFolio

SmartFolio est une plateforme multi-tenant de gestion de portefeuille (crypto + bourse) avec :

- Récupération des soldes via CoinTracking, Saxo Bank, sources manuelles
- Decision Index (DI) basé sur 4 piliers ML (Cycle, On-Chain, Risk, Sentiment)
- Allocation Engine V2, optimisation Markowitz/Black-Litterman
- Dashboard HTML multi-pages avec authentification JWT
- Backtest historique du Decision Index

Ce document définit les règles et bonnes pratiques pour contribuer efficacement.

======================================================================
1. Workflow Git
======================================================================

- Ne jamais travailler directement sur main.
- Créer une branche dédiée :
  - feature/...   → nouvelle fonctionnalité
  - fix/...       → correction de bug
  - refactor/...  → optimisation du code sans changement de logique
  - docs/...      → documentation
  - chore/...     → maintenance, dépendances, CI/CD
- Toute contribution doit passer par une Pull Request.

======================================================================
2. Conventional Commits
======================================================================

Format obligatoire :
<type>(scope): message court

Types autorisés :

- feat(scope): nouvelle fonctionnalité
- fix(scope): correction de bug
- refactor(scope): simplification ou optimisation
- docs(scope): documentation
- test(scope): ajout ou correction de tests
- chore(scope): maintenance, dépendances, CI/CD

Exemples :

- feat(rebalance): add proportional sub-allocation strategy
- fix(taxonomy): correct alias resolution for WSTETH
- refactor(pricing): simplify stablecoin handling
- docs: update README with CoinTracking API usage

NE JAMAIS ajouter "Co-Authored-By: Claude" dans les messages de commit.

======================================================================
2.5. Hooks pre-commit (recommandé)
======================================================================

Le projet utilise des hooks pour éviter les erreurs fréquentes :

Installation :

```bash
pip install pre-commit
pre-commit install
```

Hooks inclus (voir `.pre-commit-config.yaml`) :

- Formatage : ruff, black, mypy
- Qualité : trailing-whitespace, check-yaml, check-json, check-ast
- Sécurité : detect-secrets, gitleaks
- Tests unitaires (au push) : `pytest tests/unit -q`
- Validation OpenAPI : schéma auto-généré
- UTF-8 BOM : détection et suppression
- Inversions de Risk Score (100 - risk) → voir docs/RISK_SEMANTICS.md
- Messages de commit non conformes (doit suivre Conventional Commits)

======================================================================
3. Règles de développement
======================================================================

- Toujours commencer par un Plan (3–5 commits maximum).
- Chaque commit doit rester lisible (≤ 200 lignes de diff).
- Les modifications doivent inclure :
  - Mise à jour des tests si nécessaire
  - Respect strict des invariants métier (voir §4)
  - Mise à jour de README.md et TODO.md si applicable

======================================================================
3.5. Multi-Tenant OBLIGATOIRE
======================================================================

Tout endpoint doit être isolé par utilisateur.

Backend :

```python
from api.deps import get_required_user

@router.get("/endpoint")
async def endpoint(user: str = Depends(get_required_user), source: str = Query("cointracking")):
    res = await balance_service.resolve_current_balances(source=source, user_id=user)
```

Frontend :

```javascript
// Méthode recommandée : fetcher.js (X-User ajouté automatiquement)
import { apiCall } from './core/fetcher.js';
const response = await apiCall('/api/endpoint');

// Méthode legacy (encore supportée)
const activeUser = localStorage.getItem('activeUser') || 'demo';
const response = await fetch('/api/endpoint', { headers: { 'X-User': activeUser } });
```

Données isolées par utilisateur : `data/users/{user_id}/{source}/`

Ne jamais hardcoder `user_id='demo'`. Toujours utiliser `Depends(get_required_user)`.

======================================================================
3.6. Authentification JWT
======================================================================

Toutes les pages requièrent une authentification JWT.

Frontend :

```javascript
import { checkAuth, getAuthHeaders } from './core/auth-guard.js';
await checkAuth();
const response = await fetch('/api/endpoint', { headers: getAuthHeaders() });
```

Backend :

```python
from api.deps import get_current_user_jwt
@router.get("/endpoint")
async def endpoint(user: str = Depends(get_current_user_jwt)): pass
```

Voir docs/AUTHENTICATION.md pour les détails (tokens 7 jours, RBAC admin).

======================================================================
3.7. English-Only UI
======================================================================

Tout texte visible par l'utilisateur doit être en anglais :

- Labels, boutons, messages d'erreur, ARIA labels, tooltips, placeholders
- Pydantic Field `description=` (visible dans OpenAPI docs)
- HTTPException `detail=` messages

Exceptions (restent en français) :

- Commentaires de code et logs (`console.log`, `logger.info`)
- Noms de variables internes
- Documentation développeur (CLAUDE.md, CONTRIBUTING.md)

======================================================================
3.8. Response Formatting
======================================================================

Utiliser les formatters standard pour toute réponse API :

```python
from api.utils import success_response, error_response, paginated_response

return success_response(data, meta={"currency": "USD"})
return error_response("Not found", code=404)
return paginated_response(items, total=100, page=1, page_size=50)
```

Ne jamais retourner un dict brut ou un JSONResponse custom.

======================================================================
4. Invariants métier
======================================================================

A ne jamais casser :

- Somme des actions en USD = 0 (achats = ventes).
- Pas d'action avec |usd| < min_trade_usd.
- Valeur des stablecoins forcée à 1.0.
- Champs obligatoires à remplir :
  - price_used
  - est_quantity
  - meta.source_used

======================================================================
4.5. Normes & Conventions de Scoring
======================================================================

## Système Dual de Scoring

| Métrique                 | Usage                                    |
| ------------------------ | ---------------------------------------- |
| **Score de Régime**      | Communication du régime marché (0-100)   |
| **Decision Index (DI)**  | Score décisionnel stratégique (0-100)    |
| **Allocation Validity**  | Check technique V2 allocation (65 ou 45) |

## Decision Index — Formule complète

```text
raw_score = (cycle × 0.35 + onchain × 0.25 + risk × 0.25 + sentiment × 0.15)
adjusted  = raw_score × phase_factor + macro_penalty
DI        = clamp(adjusted, 0, 100)
```

- **4 composants** : Cycle (trend_regime), On-Chain (breadth_rotation), Risk, Sentiment
- **Phase factor** : bearish=0.85, moderate=1.0, bullish=1.05 (basé sur cycle score)
- **Macro penalty** : VIX > 30 ou DXY +5% → jusqu'à -15 points
- Poids configurables dans `config/score_registry.json`
- Voir docs/DECISION_INDEX_V2.md pour l'architecture détaillée

## RÈGLE CRITIQUE — Sémantique Risk

> Le **Risk Score** est un indicateur **positif** de robustesse, borné **[0..100]**.
>
> **Convention** : Plus haut = plus robuste (risque perçu plus faible).
>
> **Interdit** : Ne jamais inverser avec `100 - scoreRisk`.
>
> Source : [RISK_SEMANTICS.md](docs/RISK_SEMANTICS.md)

Toute Pull Request inversant Risk doit être REFUSÉE.

Modules concernés :

- static/core/unified-insights-v2.js (production)
- static/modules/simulation-engine.js (simulateur)
- static/components/decision-index-panel.js (visualisation)
- services/risk_scoring.py (backend — source de vérité)

Voir aussi :

- docs/architecture.md — Pilier Risk
- docs/UNIFIED_INSIGHTS_V2.md — Architecture détaillée

======================================================================
5. Tests locaux
======================================================================

Lancer l'API :

```bash
.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --port 8080
```

Tests :

```bash
pytest -q tests/unit && pytest -q tests/integration
```

Points de contrôle rapides :

- GET /healthz           → doit retourner {"ok": true}
- GET /balances/current  → soldes agrégés (header X-User requis)
- POST /rebalance/plan   → génération d'un plan JSON
- POST /rebalance/plan.csv → génération d'un plan CSV

La liste complète des endpoints est disponible sur `/docs` (OpenAPI auto-généré).

Interface utilisateur :

- Ouvrir login.html → se connecter
- Vérifier que dashboard.html, analytics-unified.html, rebalance.html fonctionnent

======================================================================
6. Pull Requests
======================================================================

- Une PR = une seule fonctionnalité ou un fix précis.
- Vérifier avant envoi :

  - [ ] Tests locaux passés (`pytest -q tests/unit`)
  - [ ] Invariants métier respectés
  - [ ] Multi-tenant respecté (pas de user_id hardcodé)
  - [ ] Risk Score non inversé
  - [ ] UI en anglais
  - [ ] Documentation mise à jour

- Utiliser le template PR dans .github/

======================================================================
Merci d'appliquer ces règles pour garantir un projet clair, stable et pro.

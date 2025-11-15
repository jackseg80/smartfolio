# API Endpoints Archive

Ce dossier contient les endpoints API archivés qui ne sont plus utilisés mais conservés pour référence historique.

## Contenu

### risk_dashboard_endpoints_legacy.py

**Archivé le :** 15 novembre 2025
**Taille :** 331 lignes
**Dernier commit actif :** d8e2e07 (Oct 2025) - "implement dependency injection with BalanceService"

#### Raison de l'Archivage

**Conflit de route** avec l'implémentation moderne dans `api/risk_endpoints.py`.

Les deux fichiers définissaient la même route `/api/risk/dashboard` :
- **risk_dashboard_endpoints.py** (version simple) - ❌ Désactivé
- **risk_endpoints.py** (version V2 avec Shadow Mode) - ✅ Active

#### Fonctionnalités Remplacées

Toutes les fonctionnalités de `risk_dashboard_endpoints_legacy.py` sont disponibles dans `api/risk_endpoints.py` avec des features supplémentaires :

| Feature | Legacy | risk_endpoints.py |
|---------|--------|-------------------|
| Risk Dashboard | ✅ | ✅ |
| Portfolio Risk Metrics | ✅ | ✅ |
| Correlation Matrix | ✅ | ✅ |
| Risk Score V2 Shadow Mode | ❌ | ✅ |
| Dual-Window System | ❌ | ✅ |
| Cache avec CSV hint | ❌ | ✅ |
| Paramètre `risk_version` | ❌ | ✅ (legacy/v2_shadow/v2_active) |

#### Route Moderne (Remplacement)

```python
# ANCIEN (désactivé)
GET /api/risk/dashboard
  → risk_dashboard_endpoints.py

# NOUVEAU (actif)
GET /api/risk/dashboard?risk_version=v2_active&use_dual_window=true
  → risk_endpoints.py
```

#### Réactivation (si nécessaire)

Si vous devez réactiver cette version legacy :

1. **Renommer le fichier** pour éviter conflit :
   ```bash
   git mv api/_archive/risk_dashboard_endpoints_legacy.py api/risk_dashboard_simple.py
   ```

2. **Changer le prefix de route** :
   ```python
   # Dans risk_dashboard_simple.py, ligne 18
   router = APIRouter(prefix="/api/risk-simple", tags=["risk-management"])
   ```

3. **Ajouter l'import dans main.py** :
   ```python
   from api.risk_dashboard_simple import router as risk_simple_router
   app.include_router(risk_simple_router)
   ```

4. **Accéder via nouvelle route** :
   ```
   GET /api/risk-simple/dashboard
   ```

#### Historique Git

Pour voir l'historique complet :
```bash
git log -- api/_archive/risk_dashboard_endpoints_legacy.py
git log -- api/risk_dashboard_endpoints.py
```

#### Tests Associés

Les tests suivants utilisent la route `/api/risk/dashboard` qui pointe maintenant vers `risk_endpoints.py` :

- `tests/integration/test_risk_dashboard_modules.py`
- `tests/integration/test_risk_dashboard_modules_fixed.py`
- `tests/integration/test_risk_dashboard_resilience.py`
- `tests/unit/test_risk_dashboard_metadata.py`
- `tests/performance/test_risk_dashboard_performance.py`
- `tests/e2e/test_phase3_integration.py`

#### Documentation Associée

- `docs/RISK_SCORE_V2_IMPLEMENTATION.md` - Implémentation Shadow Mode V2
- `docs/RISK_SEMANTICS.md` - Passage à v2_active par défaut
- `docs/STRUCTURE_MODULATION_V2.md` - Structure Score et modulation

---

## Politique d'Archivage

**Critères d'archivage :**
- Endpoint désactivé depuis > 1 mois
- Fonctionnalité entièrement remplacée par version moderne
- Import retiré du router principal
- Tests pointent vers nouvelle implémentation

**Conservation :**
- Fichiers conservés indéfiniment pour référence historique
- Possibilité de réactivation avec modifications (voir sections Réactivation)

---

*Archive créée automatiquement lors du nettoyage du codebase (Nov 2025)*

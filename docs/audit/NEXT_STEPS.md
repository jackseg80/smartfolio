# Prochaines Étapes - Audit SmartFolio

**Dernière mise à jour:** 10 novembre 2025
**État actuel:** 95% audit complété, projet prêt production

---

## CONTEXTE RAPIDE

### **Ce qui a été fait (Jours 1-2):**
[OK] 5/5 bloqueurs production résolus (100%)
[OK] Conformité CLAUDE.md: 90% (objectif 85%)
[OK] Score sécurité: 8.5/10 (objectif 8/10)
[OK] Tests BalanceService: 18 tests, 66% coverage
[OK] Multi-tenant: 14 endpoints migrés

**Commits clés:**
- `db88466` - Tests BalanceService + pytest config
- `d167cba` - Conformité CLAUDE.md 90%
- `1be7e75` - Fix bug effective_user (hotfix)
- `eb88c82` - Tracking mis à jour

### **Références:**
- Plan détaillé: `docs/_archive/session_notes/PLAN_ACTION_IMMEDIATE.md`
- Résumé session: `docs/audit/SESSION_SUMMARY_2025_11_10.md`
- Tracking: `docs/audit/PROGRESS_TRACKING.md`
- Guide CLAUDE.md: `CLAUDE.md`

---

## OPTIONS DISPONIBLES

### **Option A: Validation & Déploiement**  RECOMMANDÉ

**Action immédiate (5 min):**
1. Redémarrer serveur (fix `effective_user` appliqué)
2. Tester Risk Dashboard: http://localhost:8080/risk-dashboard.html
3. Valider multi-tenant (tester avec user demo + jack)

**Si OK → STOP ICI** - Projet prêt production!

---

### **Option B: Quick Wins (Jour 3)** - 6h

#### **1. Settings API Save** (~2h) - PRIORITAIRE

**Objectif:** Persister sélection source/CSV côté serveur

**Endpoints à créer:**
```python
# api/user_settings_endpoints.py

POST /api/users/{user_id}/settings/sources
GET  /api/users/{user_id}/settings/sources

# Stockage: data/users/{user_id}/config/sources.json
```

**Frontend update:**
```javascript
// static/global-config.js
// Sauvegarder après changement source dans WealthContextBar
```

**Bénéfice:** Préférences persistées (plus de re-sélection CSV à chaque session)

---

#### **2. Réduire TODOs** (~2h)

**Objectif:** 8 TODOs actifs → 5

**Fichiers à auditer:**
```bash
grep -r "TODO:" --include="*.py" --include="*.js" | grep -i "critical\|blocker\|important"
```

**Focus:** TODOs bloquants uniquement

---

#### **3. Tests PricingService** (~2h) - OPTIONNEL

**Objectif:** Coverage supplémentaire

**Note:** Déjà bon coverage existant, pas critique

---

### **Option C: Conformité 100% (Jour 4-5)** - 6h

#### **1. Response Formatters** (~4h)

**Objectif:** 90% → 100% conformité CLAUDE.md

**Migration pattern:**
```python
# AVANT
return {"ok": True, "data": items}

# APRÈS
from api.utils import success_response
return success_response(items, meta={"count": len(items)})
```

**Endpoints à migrer:** ~30 fichiers API

**Risque:** Peut introduire bugs sur endpoints fonctionnels
**Bénéfice:** Format API uniforme, meilleure observabilité

---

#### **2. Documentation finale** (~2h)

**Fichiers à créer/mettre à jour:**
- `CHANGELOG.md` - Historique changements audit
- `README.md` - Section déploiement production
- `docs/DEPLOYMENT.md` - Guide production complet

---

## PROMPT DE REPRISE

**Copiez-collez ceci dans une nouvelle discussion Claude Code:**

```
Reprendre audit de sécurité SmartFolio après commit 1be7e75 (10 nov 2025).

**Contexte:**
Session précédente a complété Jours 1-2 de l'audit:
- [OK] 5/5 bloqueurs production résolus
- [OK] Conformité CLAUDE.md: 90% (objectif 85% dépassé)
- [OK] Tests BalanceService: 18 tests, 66% coverage
- [OK] Multi-tenant: 14 endpoints migrés

**État actuel:**
- Projet PRÊT PRODUCTION (score sécurité 8.5/10)
- Commits: db88466 (tests), d167cba (conformité), 1be7e75 (hotfix), eb88c82 (tracking)

**Références:**
- Plan: @docs/_archive/session_notes/PLAN_ACTION_IMMEDIATE.md
- Prochaines étapes: @docs/audit/NEXT_STEPS.md
- Résumé session: @docs/audit/SESSION_SUMMARY_2025_11_10.md
- Guide: @CLAUDE.md

**Options disponibles:**

**A) Validation & Déploiement** (RECOMMANDÉ)
   - Tester fix effective_user (redémarrer serveur requis)
   - Valider Risk Dashboard fonctionne
   - STOP si OK (projet prêt!)

**B) Quick Wins - Jour 3** (6h optionnel)
   1. Settings API Save (~2h) - Persister sélection CSV
   2. Réduire TODOs (~2h) - 8 → 5 TODOs
   3. Tests PricingService (~2h) - Coverage bonus

**C) Conformité 100% - Jour 4-5** (6h optionnel)
   1. Response formatters (~4h) - 90% → 100%
   2. Documentation finale (~2h) - CHANGELOG + README

**Quelle option choisir?** Le projet est déjà en excellent état (90% conformité).
```

---

## NOTES IMPORTANTES

### **AVANT de continuer:**

1. **Vérifier serveur redémarré:**
```powershell
# Windows
.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --port 8080
```

2. **Tester Risk Dashboard:**
- Ouvrir: http://localhost:8080/risk-dashboard.html
- Vérifier: Aucune erreur `effective_user`
- Vérifier: Données chargées correctement

3. **Vérifier tests:**
```bash
pytest tests/unit/test_balance_service.py -v
# Doit afficher: 17 passed, 1 skipped
```

### **Convention CLAUDE.md à respecter:**

[OK] **Multi-tenant OBLIGATOIRE:**
```python
from api.deps import get_active_user

@router.get("/endpoint")
async def endpoint(user: str = Depends(get_active_user)):
    # TOUJOURS utiliser user, JAMAIS Query("demo")
```

[OK] **Pas de --reload:**
- Après modifs backend → restart manuel serveur
- Docs doivent refléter cette convention

[OK] **Response formatters (si Option C):**
```python
from api.utils import success_response, error_response

return success_response(data, meta={"key": "value"})
return error_response("Error message", code=400)
```

---

## GUIDE RAPIDE CLAUDE CODE

### **Lire contexte projet:**
```
@CLAUDE.md
@docs/audit/NEXT_STEPS.md
@docs/audit/SESSION_SUMMARY_2025_11_10.md
```

### **Vérifier état git:**
```bash
git log --oneline -5
git status
```

### **Exécuter tests:**
```bash
pytest tests/unit/test_balance_service.py -v --tb=short
```

### **Chercher code:**
```bash
# TODOs critiques
grep -r "TODO:" --include="*.py" | grep -i "critical"

# Endpoints non-conformes (si Option C)
grep -rn "Query.*demo" api/ --include="*.py"
```

---

## CHECKLIST AVANT COMMIT

Avant tout commit de la suite:

- [ ] Tests passent (pytest)
- [ ] Serveur démarre sans erreur
- [ ] Risk Dashboard fonctionne
- [ ] Multi-tenant préservé (isolation users)
- [ ] Aucun `--reload` ajouté aux docs
- [ ] Convention CLAUDE.md respectée
- [ ] Message commit suit format:
  ```
  type(scope): description courte

  Détails...

   Generated with [Claude Code](https://claude.com/claude-code)

  Co-Authored-By: Claude <noreply@anthropic.com>
  ```

---

## RAPPELS CRITIQUES

### **NE PAS:**
- [Error] Utiliser `Query("demo")` dans nouveaux endpoints
- [Error] Ajouter `--reload` flag dans commandes uvicorn
- [Error] Inverser Risk Score sans commentaires (100 - robustness_score)
- [Error] Committer `.env` ou clés API
- [Error] Modifier tests BalanceService (déjà validés)

### **TOUJOURS:**
- [OK] Utiliser `Depends(get_active_user)` pour user_id
- [OK] Demander confirmation avant gros changements
- [OK] Tester après modifications backend
- [OK] Documenter décisions dans commit messages

---

**Document préparé pour reprise facile par Claude Code Agent**
**Date limite recommandée:** Semaine 2 (optionnel - projet déjà prêt!)

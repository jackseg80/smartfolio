# AI Chat - Quick Test (5-10 minutes)

> **Tests rapides pour validation basique du système**
> **Temps estimé:** 5-10 minutes

---

## ⚡ Setup Rapide (2 min)

```bash
# 1. Démarrer serveur
.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --port 8080

# 2. Configurer clé Groq
# Aller sur http://localhost:8080/static/settings.html
# API Keys > Groq API Key > Ajouter clé (obtenir sur https://console.groq.com/keys)
```

---

## ✅ Test 1: Modal Fonctionne (1 min)

**Page:** <http://localhost:8080/static/dashboard.html>

**Actions:**
1. Vérifier bouton ✨ en bas à droite → ✅ Visible
2. Cliquer dessus → ✅ Modal s'ouvre
3. Appuyer Ctrl+K → ✅ Modal s'ouvre aussi
4. Cliquer × → ✅ Modal se ferme

**Résultat:** Modal fonctionne ✅

---

## ✅ Test 2: Questions Rapides (2 min)

**Page:** dashboard.html (modal ouvert)

**Actions:**
1. Cliquer sur "Résumé portefeuille" → ✅ Question envoyée
2. Attendre réponse (3-5 sec)
3. Vérifier réponse mentionne:
   - Total portfolio value → ✅
   - Top positions (BTC, ETH, etc.) → ✅
   - Allocation ou régime → ✅

**Prompt manuel:**
```
Quelles sont mes 3 plus grosses positions ?
```

**Résultat:** L'IA voit les données réelles du portfolio ✅

---

## ✅ Test 3: Knowledge Base Dynamique (3 min)

**Page:** n'importe laquelle

**Prompt:**
```
Explique-moi le Decision Index en une phrase.
```

**Vérifier réponse contient:**
- "65 (valid) ou 45 (invalid)" → ✅
- "binaire" ou "NOT weighted sum" → ✅
- "qualité allocation" → ✅

**Test modification docs:**

```bash
# 1. Vérifier cache stats
curl "http://localhost:8080/api/ai/knowledge-stats" -H "X-User: demo"

# Vérifier output JSON: entries, ttl_seconds=300, cached_pages
```

**Résultat:** Knowledge base lit CLAUDE.md ✅

---

## ✅ Test 4: Context Par Page (2 min)

**Test Risk Dashboard:**

**Page:** <http://localhost:8080/static/risk-dashboard.html>

**Prompt:**
```
Quel est mon risk score ?
```

**Vérifier:**
- Cite le score réel (ex: "68/100") → ✅
- Explique "higher = more robust" → ✅
- PAS "100 - score" → ✅

**Test Analytics:**

**Page:** <http://localhost:8080/static/analytics-unified.html>

**Prompt:**
```
Quelle est la différence entre Decision Index et Regime Score ?
```

**Vérifier:**
- "DEUX systèmes différents" → ✅
- DI = 65/45 binaire → ✅
- Regime = 0.5×CCS + 0.3×OnChain + 0.2×Risk → ✅

**Résultat:** Context builders fonctionnent ✅

---

## ✅ Test 5: Refresh Knowledge (1 min)

```bash
# Force refresh cache
curl -X POST "http://localhost:8080/api/ai/refresh-knowledge" -H "X-User: demo"

# Résultat attendu:
# {
#   "ok": true,
#   "message": "Knowledge base cache cleared successfully",
#   "entries_cleared": 1-6
# }
```

**Vérifier logs backend:**
```bash
Get-Content logs\app.log -Wait -Tail 10

# Chercher:
# INFO: Knowledge cache refreshed by user 'demo' - X entries cleared
```

**Résultat:** Refresh fonctionne ✅

---

## 📊 Checklist Rapide

- [ ] ✅ Bouton ✨ visible
- [ ] ✅ Modal s'ouvre/ferme
- [ ] ✅ Questions rapides fonctionnent
- [ ] ✅ IA voit données portfolio
- [ ] ✅ Knowledge base lit CLAUDE.md
- [ ] ✅ Context par page fonctionne
- [ ] ✅ Endpoints API répondent

---

## 🎯 Résultat

**Tous les tests ✅** → Système 100% opérationnel ! 🎉

**Un test ❌** → Voir [AI_CHAT_TEST_PROMPTS.md](AI_CHAT_TEST_PROMPTS.md) pour debug détaillé

---

**Temps total:** 5-10 minutes
**Status:** Production Ready ✅

# AI Chat - Session Handoff (Dec 27, 2025)

> **Pour reprendre le travail sur les context builders**
> **Context tokens:** 136k/200k (68% utilisé)

---

## État Actuel

### Ce qui Fonctionne (100%)

**Backend:**
- [OK] Dynamic knowledge base (lit CLAUDE.md avec cache 5 min)
- [OK] Endpoints `/api/ai/refresh-knowledge` et `/api/ai/knowledge-stats`
- [OK] Multi-provider (Groq + Claude API)
- [OK] Knowledge base explique correctement Decision Index, Risk Score, etc.

**Frontend:**
- [OK] Modal s'ouvre/ferme (bouton  + Ctrl+K)
- [OK] Questions rapides affichées
- [OK] Intégration dans 4 pages (dashboard, risk, analytics, wealth)

**Tests Réussis (Quick Test):**
- [OK] Test 1: Modal fonctionne
- [OK] Test 2: Questions rapides (partiellement - voir problèmes)
- [OK] Test 3: Knowledge base dynamique (parfait)
- [OK] Test 5: Refresh knowledge (parfait)

---

## Problèmes Identifiés (Tests User jack)

### Problème 1: Dashboard Context Incomplet

**Symptôme:**
L'IA ne voit QUE les cryptos. Manque:
- [Error] Bourse (positions Saxo)
- [Error] Patrimoine (wealth/banks)
- [Error] Scores de risque
- [Error] Régimes de marché
- [Error] Decision Index, ML Sentiment

**Exemple:**
```
User: "Fais-moi un résumé complet de mon portefeuille crypto et bourse."
IA: "Votre portefeuille est composé de 188 positions...
     Crypto : 93,5 % (299 911,16 $)
     Bourse : 0 % (aucune position enregistrée)"  ← FAUX, il y a des positions
```

**Cause Probable:**
```javascript
// static/components/ai-chat-context-builders.js:19
const unifiedState = window.getUnifiedState ? window.getUnifiedState() : {};
```
→ `window.getUnifiedState()` peut retourner `{}` vide si:
- Fonction pas définie
- Données pas encore chargées (timing)
- Erreur silencieuse

**Fichier:** [static/components/ai-chat-context-builders.js](../static/components/ai-chat-context-builders.js) lignes 9-66

---

### Problème 2: Risk Dashboard "Pas Accès aux Données"

**Symptôme:**
```
User: "Quel est mon risk score ?"
IA: "Je n'ai pas accès à vos données de portefeuille spécifiques."
```

**Alors que les logs montrent:**
```
INFO api.risk_endpoints: [OK] Returning cached risk dashboard (cache hit)
```

**Cause Probable:**
```javascript
// static/components/ai-chat-context-builders.js:80
const response = await fetch('/api/risk/dashboard', {
    headers: { 'X-User': activeUser }
});
```
→ Possible que:
1. Header `X-User` pas passé correctement
2. Réponse API vide ou erreur
3. Timing: API pas encore chargée quand context builder s'exécute

**Fichier:** [static/components/ai-chat-context-builders.js](../static/components/ai-chat-context-builders.js) lignes 71-130

---

## Solutions Proposées

### Solution 1: Enrichir Dashboard Context (Prioritaire)

**Objectif:** Récupérer données cross-asset via appels API directs

**Modifications à faire:**
```javascript
// static/components/ai-chat-context-builders.js

export async function buildDashboardContext() {
    const context = {
        page: 'Dashboard - Global Portfolio View'
    };
    const activeUser = localStorage.getItem('activeUser') || 'demo';

    try {
        // 1. Crypto (existant)
        const balanceResult = await window.loadBalanceData(true);
        // ... code existant ...

        // 2. NOUVEAU: Bourse (Saxo)
        const saxoResponse = await fetch('/api/bourse/dashboard', {
            headers: { 'X-User': activeUser }
        });
        if (saxoResponse.ok) {
            const saxoData = await saxoResponse.json();
            context.saxo = {
                total_value: saxoData.total_value,
                positions_count: saxoData.positions?.length || 0,
                top_positions: saxoData.positions?.slice(0, 5) || []
            };
        }

        // 3. NOUVEAU: Patrimoine (Wealth)
        const wealthResponse = await fetch('/api/wealth/patrimoine', {
            headers: { 'X-User': activeUser }
        });
        if (wealthResponse.ok) {
            const wealthData = await wealthResponse.json();
            context.wealth = {
                net_worth: wealthData.net_worth,
                liquidity: wealthData.liquidity
            };
        }

        // 4. NOUVEAU: Risk Score
        const riskResponse = await fetch('/api/risk/dashboard', {
            headers: { 'X-User': activeUser }
        });
        if (riskResponse.ok) {
            const riskData = await riskResponse.json();
            context.risk_score = riskData.risk_score;
        }

        // 5. NOUVEAU: Analytics (DI, ML Sentiment, Regime)
        const analyticsResponse = await fetch('/api/ml/unified-state', {
            headers: { 'X-User': activeUser }
        });
        if (analyticsResponse.ok) {
            const analyticsData = await analyticsResponse.json();
            context.decision_index = analyticsData.decision_index;
            context.ml_sentiment = analyticsData.ml_sentiment;
            context.regime = analyticsData.regime;
        }

    } catch (error) {
        console.error('Error building dashboard context:', error);
        context.error = 'Failed to load some portfolio data';
    }

    return context;
}
```

**Endpoints à vérifier:**
- [OK] `/api/risk/dashboard` (existe, testé)
- `/api/bourse/dashboard` (vérifier si existe)
- `/api/wealth/patrimoine` (vérifier endpoint exact)
- `/api/ml/unified-state` (vérifier endpoint exact)

---

### Solution 2: Renforcer Risk Dashboard Context

**Objectif:** Debug pourquoi données pas récupérées

**Modifications à faire:**
```javascript
// static/components/ai-chat-context-builders.js

export async function buildRiskDashboardContext() {
    const context = {
        page: 'Risk Dashboard - Risk Analysis'
    };

    try {
        const activeUser = localStorage.getItem('activeUser') || 'demo';

        // Debug: Log avant appel
        console.log('[AI Chat] Fetching risk dashboard for user:', activeUser);

        const response = await fetch('/api/risk/dashboard', {
            headers: { 'X-User': activeUser }
        });

        // Debug: Log réponse
        console.log('[AI Chat] Risk dashboard response status:', response.status);

        if (response.ok) {
            const data = await response.json();

            // Debug: Log données
            console.log('[AI Chat] Risk dashboard data:', data);

            // Vérifier structure données
            if (!data || !data.risk_score) {
                console.warn('[AI Chat] Risk score missing in response');
                context.error = 'Risk score not available';
                return context;
            }

            // Peupler contexte
            context.risk_score = data.risk_score;
            if (data.metrics) {
                context.var_95 = data.metrics.var_95;
                context.max_drawdown = data.metrics.max_drawdown;
                // ... etc
            }
        } else {
            console.error('[AI Chat] Risk dashboard API error:', response.status);
            context.error = `API error: ${response.status}`;
        }

    } catch (error) {
        console.error('[AI Chat] Error building risk context:', error);
        context.error = error.message;
    }

    return context;
}
```

---

## Fichiers à Modifier

| Fichier | Lignes | Action |
|---------|--------|--------|
| `static/components/ai-chat-context-builders.js` | 9-66 | Enrichir `buildDashboardContext()` |
| `static/components/ai-chat-context-builders.js` | 71-130 | Debug `buildRiskDashboardContext()` |
| `static/components/ai-chat-context-builders.js` | 135-195 | Vérifier `buildAnalyticsContext()` |
| `static/components/ai-chat-context-builders.js` | 200-260 | Vérifier `buildSaxoContext()` |
| `static/components/ai-chat-context-builders.js` | 265-320 | Vérifier `buildWealthContext()` |

---

## Debug Étapes (À Faire Avant Modification)

### 1. Vérifier window.getUnifiedState()

**Console F12 sur dashboard.html:**
```javascript
// Vérifier si fonction existe
console.log('getUnifiedState exists:', typeof window.getUnifiedState);

// Vérifier contenu
console.log('Unified state:', window.getUnifiedState ? window.getUnifiedState() : 'undefined');
```

**Résultat attendu:**
- Si `undefined` → Fonction pas chargée, utiliser API calls directs
- Si `{}` vide → Timing problem, attendre load event
- Si populated → Ok, juste compléter avec données manquantes

---

### 2. Vérifier Endpoints API Disponibles

**Tester dans terminal:**
```bash
# Risk Dashboard (fonctionne selon logs)
curl "http://localhost:8080/api/risk/dashboard" -H "X-User: jack"

# Bourse Dashboard (vérifier endpoint exact)
curl "http://localhost:8080/api/bourse/dashboard" -H "X-User: jack"
# OU
curl "http://localhost:8080/api/saxo/dashboard" -H "X-User: jack"

# Wealth/Patrimoine (vérifier endpoint exact)
curl "http://localhost:8080/api/wealth/patrimoine" -H "X-User: jack"

# Analytics/ML Unified State (vérifier endpoint exact)
curl "http://localhost:8080/api/ml/unified-state" -H "X-User: jack"
# OU
curl "http://localhost:8080/api/analytics/unified" -H "X-User: jack"
```

**Identifier les vrais endpoints** avant de coder.

---

### 3. Vérifier Timing (DOMContentLoaded)

**Problème possible:** Context builders exécutés avant que données globales soient chargées.

**Solution:** Attendre événement ou utiliser API calls (recommandé).

---

## Plan de Travail (Nouvelle Session)

### Étape 1: Debug (15 min)
1. Console F12 → Vérifier `window.getUnifiedState()`
2. Terminal → Tester endpoints API (risk, bourse, wealth, analytics)
3. Identifier quels endpoints existent et leur structure JSON

### Étape 2: Modifier Context Builders (30 min)
1. Enrichir `buildDashboardContext()` avec appels API directs
2. Ajouter logs debug dans tous les context builders
3. Gérer erreurs proprement (ne pas crasher si API fail)

### Étape 3: Tester (15 min)
1. Relancer Quick Test avec user `jack`
2. Vérifier Console F12 pour logs debug
3. Vérifier que l'IA voit maintenant:
   - [OK] Crypto
   - [OK] Bourse
   - [OK] Patrimoine
   - [OK] Risk Score
   - [OK] Decision Index, ML Sentiment, Regime

### Étape 4: Commit & Push (10 min)
1. Commit fixes
2. Push sur branche `feature/ai-chat-global-dynamic-kb`
3. Merger PR (déjà créée)

---

## Références Rapides

### Documentation
- **AI Chat Global:** [docs/AI_CHAT_GLOBAL.md](AI_CHAT_GLOBAL.md)
- **Quick Test:** [docs/AI_CHAT_QUICK_TEST.md](AI_CHAT_QUICK_TEST.md)
- **Context Builders Code:** [static/components/ai-chat-context-builders.js](../static/components/ai-chat-context-builders.js)

### Logs Tests
**Dashboard:**
```
User: "Résumé complet crypto et bourse"
IA: Crypto 93.5%, Bourse 0% ← FAUX (devrait voir positions Saxo)
```

**Risk:**
```
User: "Quel est mon risk score ?"
IA: "Je n'ai pas accès à vos données" ← FAUX (API fonctionne selon logs)
```

**Knowledge Base:**
```
User: "Explique Decision Index"
IA: "65 (valid) ou 45 (invalid), binaire" ← CORRECT [OK]
```

---

## Commandes Rapides

### Démarrer Serveur
```bash
.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --port 8080
```

### Tester Context
```bash
# Dashboard context
curl "http://localhost:8080/api/risk/dashboard" -H "X-User: jack"

# Knowledge stats
curl "http://localhost:8080/api/ai/knowledge-stats" -H "X-User: jack"

# Refresh knowledge
curl -X POST "http://localhost:8080/api/ai/refresh-knowledge" -H "X-User: jack"
```

### Logs Backend
```powershell
Get-Content logs\app.log -Wait -Tail 20
```

---

## Checklist Avant de Commencer

- [ ] Lire ce document complet
- [ ] Démarrer serveur backend
- [ ] Ouvrir dashboard.html + Console F12
- [ ] Tester `window.getUnifiedState()`
- [ ] Identifier endpoints API disponibles
- [ ] Modifier context builders
- [ ] Tester avec Quick Test
- [ ] Commit + Push

---

## État Git

**Branche actuelle:** `main` (local)
**Branche feature:** `feature/ai-chat-global-dynamic-kb` (créée, pas pushée)
**PR:** Créée sur GitHub (en attente merge)
**Commits ahead:** 5 commits (incluant le gros commit AI Chat Global)

**Note:** Le push a échoué à cause d'un "secret" détecté (faux positif `gsk_...` dans docs).
**Solution:** Autoriser le secret via GitHub → Re-pusher → Merger PR

---

**Date:** Dec 27, 2025
**Status:** Tests partiels OK, context builders à améliorer
**Priorité:** Enrichir dashboard context + debug risk context
**Temps estimé:** 1h (debug 15min + code 30min + test 15min)

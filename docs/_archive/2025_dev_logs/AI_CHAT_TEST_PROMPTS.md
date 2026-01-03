# AI Chat - Prompts de Tests Complets

> **Guide de tests pour valider le système AI Chat Global + Dynamic Knowledge Base**
> **Date:** Dec 27, 2025

---

## 🎯 Objectifs des Tests

1. ✅ Vérifier que le bouton ✨ apparaît sur toutes les pages
2. ✅ Vérifier que le modal s'ouvre correctement (clic + Ctrl+K)
3. ✅ Vérifier que les context builders fonctionnent (données page visibles par l'IA)
4. ✅ Vérifier que la knowledge base dynamique fonctionne
5. ✅ Vérifier que les providers (Groq/Claude) répondent correctement

---

## 📋 Checklist Préalable

### Configuration Initiale

- [ ] **Serveur démarré**
  ```bash
  .venv\Scripts\Activate.ps1
  python -m uvicorn api.main:app --port 8080
  ```

- [ ] **Clé API Groq configurée**
  - Aller sur <http://localhost:8080/static/settings.html>
  - Section "API Keys"
  - Ajouter "Groq API Key" (obtenir sur <https://console.groq.com/keys>)
  - Format: `gsk_...`
  - Sauvegarder

- [ ] **Portfolio chargé**
  - Avoir au moins un fichier CSV crypto ou API CoinTracking configurée
  - Vérifier sur dashboard.html que les balances s'affichent

---

## 🧪 Tests de Base

### Test 1: Bouton Flottant ✨

**Pages à tester:** dashboard.html, risk-dashboard.html, analytics-unified.html, wealth-dashboard.html

**Étapes:**
1. Ouvrir chaque page
2. Vérifier visuellement:
   - [ ] Bouton ✨ visible en bas à droite
   - [ ] Bouton a un gradient violet/bleu
   - [ ] Hover → Bouton grossit légèrement
   - [ ] Tooltip affiche "Assistant IA (Ctrl+K)"

**Résultat attendu:** Bouton visible et interactif sur les 4 pages

---

### Test 2: Ouverture Modal

**Page:** dashboard.html

**Méthode 1 - Clic:**
1. Cliquer sur le bouton ✨
2. Vérifier:
   - [ ] Modal s'affiche au centre de l'écran
   - [ ] Header "Assistant IA SmartFolio" visible
   - [ ] Sélecteur provider visible (Groq / Claude)
   - [ ] Questions rapides affichées
   - [ ] Zone de texte "Posez une question..." visible
   - [ ] Bouton "Envoyer" visible

**Méthode 2 - Raccourci clavier:**
1. Appuyer sur **Ctrl+K**
2. Vérifier que le modal s'ouvre de la même manière

**Méthode 3 - Fermeture:**
1. Cliquer sur le "×" en haut à droite
2. Vérifier que le modal se ferme
3. Réouvrir avec Ctrl+K
4. Cliquer en dehors du modal (fond gris)
5. Vérifier que le modal se ferme

**Résultat attendu:** Modal s'ouvre/ferme correctement avec les 3 méthodes

---

### Test 3: Provider Groq Configuré

**Page:** dashboard.html

**Étapes:**
1. Ouvrir modal AI Chat (Ctrl+K)
2. Vérifier sélecteur provider:
   - [ ] "Groq (Llama 3.3 70B) - Gratuit" sélectionné par défaut
   - [ ] Option "Claude (Sonnet 3.5) - Premium" disponible mais désactivée (si pas de clé Claude)

**Si Groq configuré correctement:**
- Provider Groq = enabled (sélectionnable)
- Pas de message d'erreur "API key not configured"

**Si Groq PAS configuré:**
- Message d'erreur affiché
- Sélecteur provider désactivé

**Résultat attendu:** Provider Groq actif si clé configurée

---

## 💬 Tests Questions Rapides

### Test 4: Questions Rapides - Dashboard

**Page:** dashboard.html

**Questions rapides attendues:**
1. "Résumé portefeuille"
2. "P&L Today"
3. "Allocation globale"
4. "Régime marché"

**Test:**
1. Ouvrir modal (Ctrl+K)
2. Cliquer sur "Résumé portefeuille"
3. Vérifier:
   - [ ] Question envoyée automatiquement
   - [ ] Loading indicator (optionnel)
   - [ ] Réponse de l'IA apparaît dans le chat
   - [ ] Réponse mentionne le contexte dashboard (total value, positions, etc.)

**Résultat attendu:** L'IA génère un résumé avec les données réelles du portfolio

---

### Test 5: Questions Rapides - Risk Dashboard

**Page:** risk-dashboard.html

**Questions rapides attendues:**
1. "Score de risque"
2. "VaR & Max Drawdown"
3. "Alertes actives"
4. "Cycles de marché"

**Test:**
1. Ouvrir modal (Ctrl+K)
2. Cliquer sur "Score de risque"
3. Vérifier:
   - [ ] Réponse mentionne le risk score actuel (ex: "Votre risk score est de 68/100")
   - [ ] Explication que 0-100 où higher = plus robust
   - [ ] Pas d'inversion (PAS "100 - score")

**Résultat attendu:** L'IA explique correctement le risk score (convention positive)

---

### Test 6: Questions Rapides - Analytics Unified

**Page:** analytics-unified.html

**Questions rapides attendues:**
1. "Decision Index"
2. "ML Sentiment"
3. "Phase Engine"
4. "Régimes"

**Test:**
1. Ouvrir modal (Ctrl+K)
2. Cliquer sur "Decision Index"
3. Vérifier:
   - [ ] Réponse explique que DI = 65 (valid) OU 45 (invalid)
   - [ ] Mentionne que c'est un score BINAIRE (pas une somme pondérée)
   - [ ] Distingue DI (allocation quality) vs Regime Score (market state)

**Résultat attendu:** L'IA explique correctement le système dual (DI vs Regime)

---

### Test 7: Questions Rapides - Wealth Dashboard

**Page:** wealth-dashboard.html

**Questions rapides attendues:**
1. "Patrimoine net"
2. "Diversification"
3. "Passifs"

**Test:**
1. Ouvrir modal (Ctrl+K)
2. Cliquer sur "Patrimoine net"
3. Vérifier:
   - [ ] Réponse mentionne net worth = actifs - passifs
   - [ ] Cite les chiffres réels si disponibles

**Résultat attendu:** L'IA analyse le patrimoine avec formule correcte

---

## 🧠 Tests Context Awareness

### Test 8: Context Dashboard (Crypto Portfolio)

**Page:** dashboard.html

**Prompt manuel:**
```
Quelles sont mes 3 plus grosses positions crypto et leur poids en % ?
```

**Vérifications:**
- [ ] L'IA liste les 3 top positions (ex: BTC, ETH, SOL)
- [ ] Donne les % exacts ou approximatifs
- [ ] Les valeurs correspondent aux données visibles sur la page

**Résultat attendu:** L'IA voit les données réelles du portfolio

---

### Test 9: Context Risk Dashboard (Métriques de Risque)

**Page:** risk-dashboard.html

**Prompt manuel:**
```
Quel est mon VaR 95% et qu'est-ce que ça signifie concrètement ?
```

**Vérifications:**
- [ ] L'IA cite la VaR 95% actuelle (ex: "$12,500")
- [ ] Explique: "perte maximale attendue à 95% de confiance"
- [ ] Utilise les données réelles de la page

**Résultat attendu:** L'IA cite les vraies métriques et explique bien

---

### Test 10: Context Analytics (Decision Index)

**Page:** analytics-unified.html

**Prompt manuel:**
```
Mon Decision Index actuel indique quoi sur la qualité de mon allocation ?
```

**Vérifications:**
- [ ] L'IA cite le DI actuel (65 ou 45)
- [ ] Explique: 65 = allocation valide, 45 = allocation invalide
- [ ] NE CONFOND PAS avec le Regime Score
- [ ] Mentionne que c'est binaire (pas une somme pondérée)

**Résultat attendu:** L'IA utilise les bonnes définitions de CLAUDE.md

---

## 📚 Tests Knowledge Base Dynamique

### Test 11: Vérifier Lecture CLAUDE.md

**Page:** n'importe laquelle

**Prompt manuel:**
```
Explique-moi le système dual de scoring: Decision Index vs Regime Score.
```

**Vérifications:**
- [ ] L'IA explique que ce sont DEUX systèmes PARALLÈLES
- [ ] Decision Index: 65/45 binaire → qualité allocation
- [ ] Regime Score: 0.5×CCS + 0.3×OnChain + 0.2×Risk → état marché
- [ ] Mentionne que Phase != Regime (normal d'avoir divergence)

**Test logs backend:**
```bash
# Regarder les logs serveur
Get-Content logs\app.log -Wait -Tail 20

# Chercher ces lignes:
# INFO: Building fresh knowledge base for page 'dashboard'
# INFO: Read 42853 chars from CLAUDE.md
# INFO: Knowledge base built: 2843 chars (cached for 300s)
```

**Résultat attendu:** L'IA explique le système dual EXACTEMENT comme dans CLAUDE.md

---

### Test 12: Modifier CLAUDE.md et Vérifier Sync

**Étapes:**

1. **Modifier CLAUDE.md:**
   - Ouvrir `CLAUDE.md`
   - Trouver section "### Decision Index (DI)"
   - Ajouter une phrase test: "**TEST: Score modifié pour validation**"
   - Sauvegarder

2. **Forcer refresh cache:**
   ```bash
   curl -X POST "http://localhost:8080/api/ai/refresh-knowledge" -H "X-User: demo"
   ```

3. **Poser question:**
   ```
   Qu'est-ce que le Decision Index ?
   ```

4. **Vérifier:**
   - [ ] L'IA mentionne "Score modifié pour validation" dans sa réponse
   - [ ] Confirme que la knowledge base lit bien CLAUDE.md

5. **Restaurer:**
   - Supprimer la phrase test de CLAUDE.md
   - Sauvegarder
   - Re-refresh cache

**Résultat attendu:** L'IA voit immédiatement les modifications de CLAUDE.md après refresh

---

### Test 13: Vérifier Cache TTL (5 minutes)

**Étapes:**

1. **Appeler stats:**
   ```bash
   curl "http://localhost:8080/api/ai/knowledge-stats" -H "X-User: demo"
   ```

2. **Vérifier output:**
   ```json
   {
     "ok": true,
     "stats": {
       "entries": 1,
       "ttl_seconds": 300,
       "cached_pages": [
         {
           "key": "knowledge_base_dashboard",
           "size_chars": 2843,
           "age_seconds": 42,
           "remaining_seconds": 258,
           "expired": false
         }
       ]
     }
   }
   ```

3. **Vérifications:**
   - [ ] `ttl_seconds` = 300 (5 minutes)
   - [ ] `age_seconds` augmente à chaque appel
   - [ ] `remaining_seconds` = 300 - age_seconds
   - [ ] `expired` = false si age < 300

**Résultat attendu:** Cache fonctionne avec TTL 5 min

---

## 🔍 Tests Prompts Avancés

### Test 14: Multi-Tenant Context

**Page:** dashboard.html

**Prompt manuel:**
```
Résume mon portfolio et dis-moi quel utilisateur je suis.
```

**Vérifications:**
- [ ] L'IA mentionne le user_id actuel (ex: "Vous êtes l'utilisateur demo")
- [ ] Les données correspondent au user sélectionné dans la navbar
- [ ] Si vous changez de user (navbar) et re-posez la question, les données changent

**Résultat attendu:** Context builder utilise le bon `user_id` (multi-tenant)

---

### Test 15: Pièges Fréquents (CLAUDE.md)

**Page:** n'importe laquelle

**Prompt manuel:**
```
Quelles sont les erreurs courantes à éviter dans SmartFolio ?
```

**Vérifications:**
- [ ] L'IA cite les pièges de CLAUDE.md:
  - ❌ Oublier user_id
  - ❌ Hardcoder user_id='demo'
  - ❌ fetch() direct au lieu de window.loadBalanceData()
  - ❌ Inverser Risk Score
  - ❌ Mélanger DI et Regime
- [ ] Les explications correspondent à la section "🚨 Pièges Fréquents"

**Résultat attendu:** L'IA connaît les pièges de CLAUDE.md

---

### Test 16: Patterns de Code (CLAUDE.md)

**Page:** n'importe laquelle

**Prompt manuel:**
```
Comment dois-je récupérer les balances d'un portfolio en frontend ?
```

**Vérifications:**
- [ ] L'IA recommande `window.loadBalanceData(true)`
- [ ] Explique qu'il NE FAUT PAS utiliser `fetch()` direct
- [ ] Donne un exemple de code correct
- [ ] Mentionne la dependency `get_active_user` côté backend

**Résultat attendu:** L'IA recommande les bons patterns de CLAUDE.md

---

### Test 17: Allocation Engine V2 (Knowledge Base)

**Page:** analytics-unified.html

**Prompt manuel:**
```
Explique-moi comment fonctionne l'Allocation Engine V2 topdown hierarchical.
```

**Vérifications:**
- [ ] L'IA explique les 3 niveaux:
  1. MACRO (BTC, ETH, Stablecoins, Alts)
  2. SECTEURS (SOL, L1/L0, L2, DeFi, Memecoins, etc.)
  3. COINS (assets individuels)
- [ ] Mentionne l'incumbency protection (aucun asset < 3%)
- [ ] Explique les floors contextuels (BASE vs BULLISH)

**Résultat attendu:** L'IA connaît les concepts avancés de CLAUDE.md

---

## 🐛 Tests Cas d'Erreur

### Test 18: Provider Non Configuré

**Étapes:**

1. **Supprimer clé Groq:**
   - Settings > API Keys
   - Vider "Groq API Key"
   - Sauvegarder

2. **Ouvrir modal AI Chat:**
   - Ctrl+K sur dashboard.html
   - Essayer de poser une question

**Vérifications:**
- [ ] Message d'erreur: "API key not configured"
- [ ] Sélecteur provider désactivé ou grayed out
- [ ] L'utilisateur est dirigé vers Settings

**Restaurer:** Re-configurer clé Groq

**Résultat attendu:** Erreur claire si pas de clé API

---

### Test 19: Network Error (Backend Down)

**Étapes:**

1. **Arrêter le serveur backend**
2. **Ouvrir modal AI Chat**
3. **Poser une question**

**Vérifications:**
- [ ] Message d'erreur réseau affiché
- [ ] Pas de crash JavaScript
- [ ] Bouton "Envoyer" redevient actif après erreur

**Restaurer:** Redémarrer serveur

**Résultat attendu:** Gestion gracieuse des erreurs réseau

---

### Test 20: CLAUDE.md Manquant (Fallback)

**Étapes:**

1. **Renommer CLAUDE.md:**
   ```bash
   mv CLAUDE.md CLAUDE.md.backup
   ```

2. **Forcer refresh:**
   ```bash
   curl -X POST "http://localhost:8080/api/ai/refresh-knowledge" -H "X-User: demo"
   ```

3. **Vérifier logs:**
   ```
   WARNING: Markdown file not found: CLAUDE.md
   WARNING: Could not read CLAUDE.md, using fallback
   ```

4. **Poser question:**
   ```
   Explique-moi le Decision Index.
   ```

**Vérifications:**
- [ ] L'IA répond avec connaissance basique (fallback)
- [ ] Pas de crash
- [ ] Réponse minimaliste mais correcte

**Restaurer:**
```bash
mv CLAUDE.md.backup CLAUDE.md
curl -X POST "http://localhost:8080/api/ai/refresh-knowledge" -H "X-User: demo"
```

**Résultat attendu:** Fallback knowledge base fonctionne

---

## 📊 Tests Performance

### Test 21: Temps de Réponse

**Prompt:**
```
Résumé rapide de mon portfolio.
```

**Mesure:**
- [ ] Temps total < 3s (Groq)
- [ ] Temps total < 5s (Claude API)

**Breakdown attendu:**
- Context builder: ~50-100ms
- Knowledge base (cache hit): ~1ms
- API call (Groq): ~500-2000ms
- Rendering: ~50ms

**Résultat attendu:** Réponse fluide en quelques secondes

---

### Test 22: Cache Performance

**Étapes:**

1. **Première requête (cold cache):**
   - Redémarrer serveur
   - Ouvrir AI Chat
   - Poser question → Noter temps backend

2. **Requête suivante (warm cache):**
   - Re-poser question
   - Noter temps backend

**Vérifications:**
- [ ] Cold cache: Logs montrent "Building fresh knowledge base"
- [ ] Warm cache: Logs montrent "Using cached knowledge"
- [ ] Warm cache ≈ 10x plus rapide (pas de lecture fichier)

**Résultat attendu:** Cache améliore significativement les performances

---

## ✅ Checklist Finale

### Fonctionnalités de Base

- [ ] Bouton ✨ visible sur les 4 pages
- [ ] Modal s'ouvre (clic + Ctrl+K)
- [ ] Modal se ferme (×, clic extérieur)
- [ ] Provider Groq configuré et actif
- [ ] Questions rapides affichées par page

### Context Awareness

- [ ] Dashboard: Voit positions crypto
- [ ] Risk: Voit risk score, VaR, etc.
- [ ] Analytics: Voit Decision Index, ML Sentiment
- [ ] Wealth: Voit net worth, actifs, passifs

### Knowledge Base Dynamique

- [ ] Lit CLAUDE.md au démarrage
- [ ] Cache TTL 5 min fonctionne
- [ ] Endpoint `/refresh-knowledge` force reload
- [ ] Endpoint `/knowledge-stats` retourne stats
- [ ] Modifications CLAUDE.md visibles après refresh
- [ ] Fallback fonctionne si CLAUDE.md manquant

### Qualité Réponses

- [ ] Explique Decision Index correctement (65/45 binaire)
- [ ] Explique Risk Score correctement (0-100, higher=robust)
- [ ] Distingue DI vs Regime Score
- [ ] Cite les pièges fréquents de CLAUDE.md
- [ ] Recommande les bons patterns de code

---

## 🎯 Résultat Global Attendu

Si tous les tests passent:

✅ **AI Chat Global:** Système 100% fonctionnel
✅ **Context Builders:** Données page correctement injectées
✅ **Knowledge Base:** Documentation CLAUDE.md lue dynamiquement
✅ **Cache:** TTL 5 min + refresh manuel fonctionnent
✅ **Multi-Provider:** Groq gratuit opérationnel
✅ **Qualité:** Réponses précises basées sur vraie documentation

---

## 📝 Rapporter les Bugs

Si un test échoue, noter:

1. **Test #** et nom
2. **Résultat obtenu** vs attendu
3. **Logs backend** (si applicable)
4. **Console browser** (F12 → Console)
5. **Screenshots** (si UI)

**Envoyer rapport à:** GitHub Issues ou documentation bug

---

**Bon tests !** 🚀

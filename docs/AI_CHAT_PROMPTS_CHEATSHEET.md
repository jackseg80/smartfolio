# AI Chat - Prompts Cheatsheet

> **Prompts prêts à copier-coller pour tester toutes les fonctionnalités**

---

## 🎯 Tests Basiques

### Questions Générales

```
Résume mon portfolio en 3 points clés.
```

```
Quelles sont mes 3 plus grosses positions et leur poids en % ?
```

```
Quel est mon patrimoine total (crypto + bourse + liquidités) ?
```

---

## 📊 Dashboard (Crypto Portfolio)

```
Résumé portefeuille
```

```
P&L Today
```

```
Allocation globale
```

```
Régime marché
```

```
Comment est réparti mon portfolio crypto entre BTC, ETH et altcoins ?
```

```
Quelle est ma position la plus risquée et pourquoi ?
```

---

## ⚠️ Risk Dashboard

```
Score de risque
```

```
VaR & Max Drawdown
```

```
Alertes actives
```

```
Cycles de marché
```

```
Quel est mon VaR 95% et qu'est-ce que ça signifie concrètement ?
```

```
Mon portfolio est-il trop concentré ? Explique avec le HHI.
```

```
Quelles sont les 3 métriques de risque les plus importantes à surveiller ?
```

---

## 📈 Analytics Unified

```
Decision Index
```

```
ML Sentiment
```

```
Phase Engine
```

```
Régimes
```

```
Explique-moi le système dual de scoring: Decision Index vs Regime Score.
```

```
Quelle est la différence entre Decision Index et Regime Score ?
```

```
Mon Decision Index actuel indique quoi sur la qualité de mon allocation ?
```

```
Pourquoi Phase et Régime peuvent diverger (ex: Phase bearish + Régime Expansion) ?
```

```
C'est quoi le ML Sentiment et comment il influence l'allocation ?
```

---

## 💰 Wealth Dashboard

```
Patrimoine net
```

```
Diversification
```

```
Passifs
```

```
Comment se répartit mon patrimoine entre actifs liquides et immobiliers ?
```

```
Quel est mon ratio actifs/passifs ?
```

---

## 🧠 Knowledge Base (Concepts SmartFolio)

### Decision Index

```
Qu'est-ce que le Decision Index ?
```

```
Explique-moi le Decision Index en une phrase.
```

```
Pourquoi le Decision Index est binaire (65/45) et pas une somme pondérée ?
```

### Risk Score

```
Comment fonctionne le Risk Score ?
```

```
Pourquoi Risk Score va de 0 à 100 avec higher = more robust ?
```

```
Quelle est la différence entre Risk Score V1 et V2 (shadow mode) ?
```

### Allocation Engine V2

```
Explique-moi l'Allocation Engine V2 topdown hierarchical.
```

```
Quels sont les 3 niveaux de l'Allocation Engine V2 ?
```

```
C'est quoi l'incumbency protection ?
```

```
Quels sont les floors contextuels (BASE vs BULLISH) ?
```

### Market Opportunities

```
Comment fonctionne le système Market Opportunities ?
```

```
Combien d'actions et ETFs couvre le système Market Opportunities ?
```

```
Explique le scoring 3-pillar (Momentum, Value, Diversification).
```

### Stop Loss

```
Quelles sont les 6 méthodes de stop loss disponibles ?
```

```
Quelle est la méthode de stop loss recommandée et pourquoi ?
```

```
Comment fonctionne le Trailing Stop (NEW Oct 2025) ?
```

---

## 🔧 Patterns de Code (Développeurs)

```
Comment dois-je récupérer les balances d'un portfolio en frontend ?
```

```
Quel est le pattern multi-tenant à utiliser côté backend ?
```

```
Comment charger un modèle ML de manière sécurisée ?
```

```
Quelles sont les erreurs courantes à éviter dans SmartFolio ?
```

```
Quels sont les pièges fréquents liés au multi-tenant ?
```

---

## 🚨 Tests Erreurs Courantes

### Inversion Risk Score (Doit détecter)

```
Mon risk score est de 68/100. Ça veut dire que mon portfolio est à 32% de robustesse ?
```

**Réponse attendue:** ❌ NON, 68/100 = 68% robust (pas 32%). Higher = more robust.

### Confusion DI vs Regime (Doit clarifier)

```
Mon Decision Index est 65 et mon Regime Score est 55. Pourquoi c'est différent ?
```

**Réponse attendue:** Deux systèmes différents. DI = qualité allocation (binaire 65/45). Regime = état marché (0-100 variable).

### Somme Pondérée DI (Doit corriger)

```
Le Decision Index est calculé comme 0.65×Cycle + 0.25×OnChain + 0.10×Risk ?
```

**Réponse attendue:** ❌ NON, DI = 65 (valid) ou 45 (invalid) basé sur total_check.isValid. PAS une somme pondérée.

---

## 🔄 Tests Dynamic Knowledge Base

### Vérifier Lecture CLAUDE.md

```
Cite-moi les 5 règles critiques de SmartFolio selon CLAUDE.md.
```

**Réponse attendue:**
1. Multi-Tenant OBLIGATOIRE
2. Risk Score = Positif (0-100)
3. Système Dual de Scoring
4. Design & Responsive
5. Autres Règles

### Vérifier Pièges Fréquents

```
Quelles sont les erreurs courantes à éviter dans SmartFolio ?
```

**Réponse attendue:**
- ❌ Oublier user_id
- ❌ Hardcoder user_id='demo'
- ❌ fetch() direct au lieu de window.loadBalanceData()
- ❌ Inverser Risk Score
- ❌ Mélanger DI et Regime

---

## 🧪 Tests Contexte Multi-Page

### Dashboard → Voit Portfolio

```
Combien de positions crypto j'ai au total ?
```

### Risk → Voit Metrics

```
Cite mes 3 principales métriques de risque avec leurs valeurs.
```

### Analytics → Voit DI + ML

```
Quels sont les 3 scores qui composent le Regime Score et leurs valeurs actuelles ?
```

### Wealth → Voit Net Worth

```
Quel est mon net worth actuel et comment il se décompose ?
```

---

## 📚 Tests Concepts Avancés

### Overrides

```
Quels sont les overrides qui modifient l'allocation automatiquement ?
```

**Réponse attendue:**
- ML Sentiment < 25 → Force défensif (+10 pts stables)
- Contradiction > 50% → Pénalise On-Chain/Risk (×0.9)
- Structure Score < 50 → +10 pts stables

### Freeze Semantics

```
Quels sont les 3 types de freeze et leurs différences ?
```

**Réponse attendue:**
- full_freeze: Tout bloqué
- s3_alert_freeze: Achats bloqués, sorties/hedge autorisés
- error_freeze: Achats bloqués, réductions risque autorisées

### Cache TTL

```
Quels sont les TTL des différents caches SmartFolio ?
```

**Réponse attendue:**
- On-Chain: 4h
- Cycle Score: 24h
- ML Sentiment: 15 min
- Prix crypto: 3 min
- Risk Metrics: 30 min

---

## 🔍 Tests Edge Cases

### Portfolio Vide

```
Analyse mon portfolio.
```

**Si aucune position:**
- L'IA doit détecter et dire "Portfolio vide" ou "Aucune position"

### Données Manquantes

```
Quelle est ma VaR 95% ?
```

**Si risk dashboard pas chargé:**
- L'IA doit dire "Données non disponibles" ou similaire

### Provider Non Configuré

**Si clé Groq manquante:**
- Modal affiche "API key not configured"
- Redirection vers Settings

---

## 📝 Tests Réponses Qualité

### Réponse Courte

```
Decision Index en 1 phrase.
```

**Attendu:** ~50 mots max

### Réponse Détaillée

```
Explique-moi en détail l'Allocation Engine V2.
```

**Attendu:** ~200-300 mots avec structure (niveaux, floors, incumbency, etc.)

### Réponse avec Exemples

```
Donne-moi un exemple concret de réallocation topdown hierarchical.
```

**Attendu:** Exemple chiffré (ex: BTC 40% → 15% macro, 12% secteur, 5% coin)

---

## 🎯 Tests Validation Finale

### Test Complet 1

```
Analyse complète de mon portfolio: positions, risque, allocation, et recommandations.
```

**Vérifier:**
- [ ] Cite positions réelles
- [ ] Mentionne risk score + VaR
- [ ] Analyse allocation (BTC/ETH/Alts ratio)
- [ ] Donne 2-3 recommandations concrètes

### Test Complet 2

```
Explique-moi le système SmartFolio en 5 points clés pour un nouveau utilisateur.
```

**Vérifier:**
- [ ] Multi-tenant
- [ ] Decision Index vs Regime
- [ ] Risk Score (0-100 positif)
- [ ] Allocation Engine V2
- [ ] Market Opportunities / Stop Loss

---

## 🚀 Commandes API (Curl)

### Refresh Knowledge

```bash
curl -X POST "http://localhost:8080/api/ai/refresh-knowledge" -H "X-User: demo"
```

### Knowledge Stats

```bash
curl "http://localhost:8080/api/ai/knowledge-stats" -H "X-User: demo"
```

### Providers List

```bash
curl "http://localhost:8080/api/ai/providers" -H "X-User: demo"
```

### Quick Questions Dashboard

```bash
curl "http://localhost:8080/api/ai/quick-questions/dashboard"
```

---

## ✅ Résultats Attendus

Si tous les prompts donnent des réponses correctes:

- ✅ **Context Builders:** IA voit données réelles de chaque page
- ✅ **Knowledge Base:** IA connaît CLAUDE.md (concepts, patterns, pièges)
- ✅ **Qualité:** Réponses précises, pas de confusion DI/Regime, pas d'inversion Risk
- ✅ **Dynamic Sync:** Modifications CLAUDE.md visibles après refresh
- ✅ **Multi-Provider:** Groq/Claude fonctionnent

---

**Guide complet:** [AI_CHAT_TEST_PROMPTS.md](AI_CHAT_TEST_PROMPTS.md) (22 tests détaillés)
**Quick Test:** [AI_CHAT_QUICK_TEST.md](AI_CHAT_QUICK_TEST.md) (5-10 min)

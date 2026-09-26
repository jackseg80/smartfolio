# Quick Start: Implémentation Logique Contextuelle ML Sentiment

** Document de travail complet:** [`WORK_SESSION_SENTIMENT_CONTEXTUAL_LOGIC.md`](./WORK_SESSION_SENTIMENT_CONTEXTUAL_LOGIC.md)

---

## TL;DR

**Problème:** Le système traite "Extreme Fear" toujours comme un danger, même en Bull Market (où c'est une opportunité).

**Solution:** Logique contextuelle intelligente basée sur régime marché:
- **Bull + Fear** → Opportunité (acheter le dip)
- **Bear + Fear** → Danger (protection)
- **Greed** → Toujours prise de profits [Warning]

**Fichier à modifier:** `static/core/unified-insights-v2.js` (lignes 194-216)

---

## État Actuel (22 Oct 2025)

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **ML Sentiment** | 52/100 | Neutral [OK] (Alternative.me: 25 + Social: 60+ → Agrégé: 52) |
| **Régime** | Bull 68% | Bullish  |
| **Cycle Score** | 59 | Bearish phase (<70) |
| **Contradiction** | 0.175 | Faible [OK] |

**Note:** ML Sentiment utilise maintenant **vraies données** (fix déjà appliqué dans `orchestrator.py`).

---

## Commande Rapide d'Implémentation

### Option 1: Implémentation Manuelle

```bash
# 1. Backup
cp static/core/unified-insights-v2.js static/core/unified-insights-v2.js.backup

# 2. Éditer le fichier
# Voir section "Code Proposé" dans WORK_SESSION_SENTIMENT_CONTEXTUAL_LOGIC.md
# Lignes à modifier: 194-216

# 3. Hard refresh
# CTRL+Shift+R sur analytics-unified.html, simulations.html, dashboard.html
```

### Option 2: Test dans Simulateur d'abord

```bash
# 1. Ouvrir simulateur
http://localhost:8080/static/simulations.html

# 2. Configurer manuellement:
#    - DI: 60
#    - Régime: Bull (68%)
#    - ML Sentiment: 20 (Extreme Fear)
#
# 3. Observer comportement actuel (défensif)
# 4. Appliquer le code proposé
# 5. Re-tester (devrait être opportuniste)
```

---

## Code Minimal à Ajouter

**Ajouter au contexte (ligne 565):**
```javascript
sentiment_value: sentimentData?.value || 50,  // ← AJOUTER
```

**Remplacer bloc (lignes 194-216):**
```javascript
// Détection
const isBull = !phaseEngineActive && (ctx?.regime === 'bull' || ctx?.cycle_score >= 70);
const isBear = !phaseEngineActive && (ctx?.regime === 'bear' || ctx?.cycle_score <= 30);
const mlSentiment = ctx?.sentiment_value || 50;
const extremeFear = mlSentiment < 25;
const extremeGreed = mlSentiment > 75;

// Logique contextuelle
if (extremeFear && isBull) {
  // Opportunité
  base.ETH *= 1.15;
  base.SOL *= 1.20;
  base.Memecoins *= 1.5;
}
else if (extremeFear && isBear) {
  // Danger
  base.Memecoins *= 0.3;
  base['Gaming/NFT'] *= 0.5;
}
else if (extremeGreed) {
  // Prise profits
  base.Memecoins *= 0.3;
}
// ... (voir document complet)
```

---

## Tests de Validation

### Scénarios Critiques:

```javascript
// Test 1: Bull + Neutral (pas de changement)
{regime: 'bull', sentiment: 55} → Boost ETH/SOL [OK]

// Test 2: Bull + Fear (NOUVEAU - opportuniste)
{regime: 'bull', sentiment: 20} → Boost x1.5 Memecoins [OK]

// Test 3: Bear + Fear (défensif maintenu)
{regime: 'bear', sentiment: 20} → Réduit -70% Memecoins [OK]

// Test 4: Greed (NOUVEAU - profits)
{regime: 'bull', sentiment: 85} → Réduit -70% Memecoins [OK]
```

---

## Impact Global

**Fichiers Affectés:**
- [OK] `analytics-unified.html` - Nouveaux targets
- [OK] `simulations.html` - Nouveaux targets
- [OK] `dashboard.html` - Nouveaux targets
- [OK] `risk-dashboard.html` - Nouveaux targets

**Fichiers NON Affectés:**
- [Error] Decision Index (DI) - Calcul inchangé
- [Error] Risk Budget - Stables base inchangé
- [Error] Governance Backend - Inchangé

---

## Documents de Référence

| Document | Description |
|----------|-------------|
| [`WORK_SESSION_SENTIMENT_CONTEXTUAL_LOGIC.md`](./WORK_SESSION_SENTIMENT_CONTEXTUAL_LOGIC.md) | **Document complet** avec contexte, analyse, code détaillé |
| [`DECISION_INDEX_V2.md`](./DECISION_INDEX_V2.md) | Documentation Decision Index + Overrides |
| [`CLAUDE.md`](../CLAUDE.md) | Guide agent (section Overrides ligne 56-64) |

---

## Checklist Avant Implémentation

- [ ] Lire document complet [`WORK_SESSION_SENTIMENT_CONTEXTUAL_LOGIC.md`](./WORK_SESSION_SENTIMENT_CONTEXTUAL_LOGIC.md)
- [ ] Comprendre flux de données (unified-insights → targets → UI)
- [ ] Backup `unified-insights-v2.js`
- [ ] Tester dans simulateur (optionnel)
- [ ] Implémenter modifications
- [ ] Tester 4 scénarios critiques
- [ ] Hard refresh toutes les pages
- [ ] Vérifier logs console (pas d'erreurs)
- [ ] Mettre à jour CLAUDE.md si nécessaire

---

## En Cas de Problème

**Rollback rapide:**
```bash
cp static/core/unified-insights-v2.js.backup static/core/unified-insights-v2.js
# Hard refresh navigateur (CTRL+Shift+R)
```

**Logs à vérifier:**
```bash
# Console navigateur (F12)
 Market conditions: {...}
 Opportunistic allocation: Bull + Fear detected

# Backend logs (si nécessaire)
tail -50 logs/app.log | grep -i "sentiment\|fear\|greed"
```

---

**Créé:** 22 Oct 2025 18:45 UTC
**Auteur:** Session Claude Code
**Statut:** [OK] Prêt pour implémentation

